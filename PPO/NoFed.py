import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch.distributions import Categorical
from policy import PPOActorCriticNetwork
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

device = torch.device("cpu")

# PPO Hyperparameters
PPO_EPOCHS    = 4
PPO_CLIP      = 0.2
VALUE_COEF    = 0.5
ENTROPY_COEF  = 0.01
MAX_GRAD_NORM = 0.5


class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device):
        self.env       = env
        self.gamma     = gamma
        self.device    = device
        self.policy    = PPOActorCriticNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def train_ppo(self):
        obs, _ = self.env.reset()
        states, actions, log_probs_old, rewards, values = [], [], [], [], []
        done = False

        with torch.no_grad():
            while not done:
                state        = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_probs, value = self.policy(state)

                dist     = Categorical(action_probs)
                action   = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                log_probs_old.append(log_prob)
                rewards.append(reward)
                values.append(value.squeeze())

                obs = next_obs

        # Discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns       = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns       = (returns - returns.mean()) / (returns.std() + 1e-5)

        values_tensor = torch.stack(values).to(self.device)
        advantages    = returns - values_tensor.detach()
        advantages    = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        states_tensor   = torch.cat(states, dim=0)
        actions_tensor  = torch.cat(actions, dim=0)
        log_probs_old_t = torch.cat([lp.unsqueeze(0) for lp in log_probs_old]).to(self.device)

        for _ in range(PPO_EPOCHS):
            action_probs_new, values_new = self.policy(states_tensor)
            dist_new      = Categorical(action_probs_new)
            log_probs_new = dist_new.log_prob(actions_tensor)
            entropy       = dist_new.entropy().mean()

            ratio  = torch.exp(log_probs_new - log_probs_old_t)
            surr1  = ratio * advantages
            surr2  = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = VALUE_COEF * torch.nn.functional.mse_loss(values_new.squeeze(), returns)
            loss        = actor_loss + critic_loss - ENTROPY_COEF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

        self.scheduler.step()
        return sum(rewards)


def main(seed, episodes, run):
    seeds        = [seed] * 8
    agents_count = 8
    envs         = [gym.make('LunarLander-v3') for _ in range(agents_count)]
    for i, env in enumerate(envs):
        np.random.seed(seeds[i])
        torch.manual_seed(seeds[i])
        env.reset(seed=seeds[i])

    state_size  = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n

    lr_list = [5e-4, 1e-3, 4e-4, 6e-4, 3e-4,
               4e-4, 5e-4, 1e-3, 2e-4, 1e-4]
    activation_list = ['relu', 'relu', 'tanh', 'relu', 'tanh',
                       'relu', 'tanh', 'relu', 'tanh', 'relu']
    gamma = 0.99

    hidden_sizes_list = [[128, 128, 256], [64, 64], [128, 128], [128, 256], [256, 256],
                         [512], [64, 128, 64], [32, 32], [512, 512], [1024]]

    agents = [
        Agent(
            envs[i], state_size, action_size,
            hidden_sizes_list[i % len(hidden_sizes_list)],
            lr_list[i % len(lr_list)],
            activation_list[i % len(activation_list)],
            gamma, device=device
        )
        for i in range(agents_count)
    ]

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards   = []

    pbar = tqdm(range(episodes), unit="ep")
    for episode in pbar:

        with ThreadPoolExecutor(max_workers=agents_count) as executor:
            total_rewards = list(executor.map(lambda a: a.train_ppo(), agents))

        for i, reward in enumerate(total_rewards):
            rewards_per_agent[i].append(reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        average_rewards.append(average_reward)

        rolling_avg = sum(
            sum(rewards_per_agent[i][-100:]) / len(rewards_per_agent[i][-100:])
            for i in range(agents_count)
        ) / agents_count

        pbar.set_postfix({
            "avg_reward":  f"{average_reward:.2f}",
            "rolling_100": f"{rolling_avg:.2f}"
        })

        # Per-agent logging (commented out)
        # for idx, reward in enumerate(total_rewards):
        #     window   = rewards_per_agent[idx][-100:]
        #     last_avg = sum(window) / len(window)
        #     print(f"  Agent {idx + 1} Reward: {reward:.2f} | Avg(100): {last_avg:.4f}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(f'results/rewards_per_agent_Lunar_PPO_NoFed_{run}_{timestamp}.csv', index=False)

    avg_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Average Reward': average_rewards})
    avg_df.to_csv(f'results/rewards_Lunar_PPO_NoFed_{run}_{timestamp}.csv', index=False)


if __name__ == "__main__":
    for run in range(3):
        print(f"\nRun {run + 1}:")
        seed     = 20 + run * 5
        episodes = 30
        print(f"Seed: {seed}, Episodes: {episodes}")
        main(seed, episodes, run)