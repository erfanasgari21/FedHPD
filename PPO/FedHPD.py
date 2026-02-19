import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import threading
import copy
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from torch.distributions import Categorical
from policy import PPOActorCriticNetwork
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

device = torch.device("cpu")

# PPO Hyperparameters
PPO_EPOCHS = 4          # Number of PPO update epochs per episode
PPO_CLIP = 0.2          # Clipping epsilon
VALUE_COEF = 0.5        # Coefficient for value loss
ENTROPY_COEF = 0.01     # Coefficient for entropy bonus
MAX_GRAD_NORM = 0.5     # Gradient clipping


def federated_average(probs):
    avg_probs = None
    for prob in probs:
        if avg_probs is None:
            avg_probs = prob.clone().detach()
        else:
            avg_probs += prob.clone().detach()
    return avg_probs / len(probs)


class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device):
        self.env = env
        self.gamma = gamma
        self.device = device
        # Use PPO Actor-Critic network instead of plain policy network
        self.policy = PPOActorCriticNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def train_ppo(self):
        """
        Collect a full episode of experience, then run PPO_EPOCHS of
        clipped surrogate + value + entropy updates on the collected data.
        """
        # ── 1. Rollout ────────────────────────────────────────────────────────
        obs, _ = self.env.reset()
        states, actions, log_probs_old, rewards, values = [], [], [], [], []
        done = False

        with torch.no_grad():
            while not done:
                state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action_probs, value = self.policy(state)

                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                log_probs_old.append(log_prob)
                rewards.append(reward)
                values.append(value.squeeze())

                obs = next_obs

        # ── 2. Compute discounted returns and advantages ───────────────────────
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)   # normalise

        values_tensor = torch.stack(values).to(self.device)             # (T,)
        advantages = returns - values_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Stack collected tensors
        states_tensor    = torch.cat(states, dim=0)                     # (T, state_dim)
        actions_tensor   = torch.cat(actions, dim=0)                    # (T,)
        log_probs_old_t  = torch.cat([lp.unsqueeze(0) for lp in log_probs_old]).to(self.device)  # (T,)

        # ── 3. PPO update epochs ──────────────────────────────────────────────
        for _ in range(PPO_EPOCHS):
            # Fresh forward pass (with gradients this time)
            action_probs_new, values_new = self.policy(states_tensor)
            dist_new = Categorical(action_probs_new)
            log_probs_new = dist_new.log_prob(actions_tensor)
            entropy = dist_new.entropy().mean()

            # Clipped surrogate objective
            ratio = torch.exp(log_probs_new - log_probs_old_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE between predicted values and discounted returns)
            critic_loss = VALUE_COEF * torch.nn.functional.mse_loss(
                values_new.squeeze(), returns
            )

            # Total loss
            loss = actor_loss + critic_loss - ENTROPY_COEF * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
            self.optimizer.step()

        self.scheduler.step()
        return sum(rewards)

    def calculate_kd_loss(self, sampled_states_part, agent_probs, avg_probs):
        """
        Knowledge-distillation step: pull this agent's action distribution
        toward the federated average using KL divergence.
        Only the actor head is involved here.
        """
        # PPOActorCriticNetwork returns (action_probs, value); we only need probs
        output_probs, _ = self.policy(sampled_states_part)

        kd_loss = torch.nn.functional.kl_div(
            output_probs.log(), avg_probs, reduction='batchmean'
        )

        self.optimizer.zero_grad()
        kd_loss.backward()
        self.optimizer.step()

        return kd_loss.item()


class Server:
    def __init__(self, agents, device):
        self.agents = agents
        self.device = device

    def collect_and_average_probs(self, sampled_states):
        # PPOActorCriticNetwork returns (action_probs, value) — detach only action_probs
        agent_probs = [agent.policy(sampled_states)[0].detach() for agent in self.agents]
        avg_probs = federated_average(agent_probs)
        return agent_probs, avg_probs


def main(seed, episodes, distill_interval, run):
    seeds = [seed] * 8
    agents_count = 8
    envs = [gym.make('LunarLander-v3') for _ in range(agents_count)]
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

    server = Server(agents, device)

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards   = []
    kd_losses         = {f'Agent {i + 1}': [] for i in range(agents_count)}

    pbar = tqdm(range(episodes), unit="ep")
    for episode in range(episodes):

        # ── Parallel PPO training via ThreadPoolExecutor ──────────────────────
        with ThreadPoolExecutor(max_workers=agents_count) as executor:
            total_rewards = list(executor.map(lambda a: a.train_ppo(), agents))

        for i, reward in enumerate(total_rewards):
            rewards_per_agent[i].append(reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        average_rewards.append(average_reward)

        # Compute rolling 100-episode average across all agents
        all_recent = [
            rewards_per_agent[i][-100:] for i in range(agents_count)
        ]
        rolling_avg = sum(sum(r) / len(r) for r in all_recent) / agents_count

        pbar.set_postfix({
            "avg_reward": f"{average_reward:.2f}",
            "rolling_100": f"{rolling_avg:.2f}"
        })

        # print(f"Episode {episode + 1}:")
        # for idx, reward in enumerate(total_rewards):
        #     history = rewards_per_agent[idx]
        #     window   = history[-100:] if len(history) >= 100 else history
        #     last_avg = sum(window) / len(window)
        #     print(f"  Agent {idx + 1} Reward: {reward:.2f} | Avg(100): {last_avg:.4f}")
        # print(f"  Average Reward: {average_reward:.4f}")



        # ── Federated Knowledge Distillation ──────────────────────────────────
        if (episode + 1) % distill_interval == 0:
            csv_file = 'KD_state_10-from-each.csv'
            df = pd.read_csv(csv_file)
            sampled_states = df['State'].apply(
                lambda x: torch.tensor([float(v) for v in x.split(',')])
            )
            sampled_states = torch.stack(sampled_states.values.tolist()).to(device)

            dataset_size = sampled_states.size(0)
            assert dataset_size == 10000, "sampled_states size should be 10000"

            quarter_size        = dataset_size // 4
            sampled_states_parts = [
                sampled_states[i * quarter_size: (i + 1) * quarter_size]
                for i in range(4)
            ]

            distill_loop = 5
            for distillation_round in range(distill_loop):
                # print(f"\nKnowledge Distillation Round {distillation_round + 1}:")

                for part_idx, sampled_states_part in enumerate(sampled_states_parts):
                    # print(f"  Processing part {part_idx + 1}")

                    agent_probs, avg_probs = server.collect_and_average_probs(sampled_states_part)

                    for agent_idx, agent in enumerate(agents):
                        kd_loss = agent.calculate_kd_loss(sampled_states_part, agent_probs, avg_probs)
                        kd_losses[f'Agent {agent_idx + 1}'].append(
                            (episode + 1, distillation_round + 1, kd_loss)
                        )
                        # print(f"  Agent {agent_idx + 1} | KD Loss: {kd_loss:.4f}")

                    # print(f"  Part {part_idx + 1} done.")
                tqdm.write(f"Episode {episode + 1}: KD distillation completed.")
                # print(f"KD Round {distillation_round + 1} Completed!")

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(f'results/rewards_per_agent_Lunar_PPO_{run}_{timestamp}.csv', index=False)

    avg_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Average Reward': average_rewards})
    avg_df.to_csv(f'results/rewards_Lunar_PPO_{run}_{timestamp}.csv', index=False)


if __name__ == "__main__":
    for run in range(3):
        print(f"\nRun {run + 1}:")
        seed = 20 + run * 5
        episodes = 3000
        distill_interval = 5
        print(f"Seed: {seed}, Episodes: {episodes}, Distill Interval: {distill_interval}")
        main(seed, episodes, distill_interval, run)