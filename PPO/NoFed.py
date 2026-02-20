import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from datetime import datetime
from collections import deque
from policy import PPOActorCriticNetwork
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

device = torch.device("cpu")

# PPO Hyperparameters
PPO_EPOCHS     = 4
PPO_CLIP       = 0.2
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.01
MAX_GRAD_NORM  = 0.5
ROLLOUT_LENGTH = 2048
BATCH_SIZE     = 64
GAE_LAMBDA     = 0.95


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma, gae_lambda):
        returns    = []
        advantages = []
        gae        = 0
        next_value = 0

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * next_value * (1 - float(self.dones[t]))
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * (1 - float(self.dones[t])) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])   # <-- FIX: compute inline
            next_value = self.values[t]

        return (
            torch.tensor(returns,    dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )

    def clear(self):
        self.states    = []
        self.actions   = []
        self.log_probs = []
        self.rewards   = []
        self.dones     = []
        self.values    = []

    def __len__(self):
        return len(self.rewards)

class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device):
        self.env       = env
        self.gamma     = gamma
        self.device    = device
        self.policy    = PPOActorCriticNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer    = RolloutBuffer()
        # Persistent state across calls
        self.state, _  = env.reset()
        self.ep_reward = 0.0
        self.completed_episode_rewards = []

    def collect_rollout(self, rollout_length):
        """Collect exactly rollout_length steps, spanning episode boundaries."""
        for _ in range(rollout_length):
            action, log_prob, value = self.select_action(self.state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(self.state, action, log_prob, reward, done, value)
            self.state     = next_state
            self.ep_reward += reward

            if done:
                self.completed_episode_rewards.append(self.ep_reward)
                self.ep_reward = 0.0
                self.state, _ = self.env.reset()

    def select_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_t)
        return action.item(), log_prob.item(), value.item()

    def update(self):
        states        = torch.tensor(np.array(self.buffer.states),  dtype=torch.float32).to(self.device)
        actions       = torch.tensor(np.array(self.buffer.actions), dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs,         dtype=torch.float32).to(self.device)

        returns, advantages = self.buffer.compute_returns_and_advantages(self.gamma, GAE_LAMBDA)
        returns    = returns.to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)

        dataset_size = states.size(0)
        indices      = np.arange(dataset_size)

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, BATCH_SIZE):
                batch_idx = indices[start: start + BATCH_SIZE]
                log_probs, entropy, values = self.policy.evaluate(states[batch_idx], actions[batch_idx])

                ratio  = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1  = ratio * advantages[batch_idx]
                surr2  = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages[batch_idx]

                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   = (returns[batch_idx] - values).pow(2).mean()
                entropy_loss = entropy.mean()
                loss         = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.buffer.clear()

def main(seed, total_timesteps, run):
    seeds        = [seed] * 2
    agents_count = 2
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

    
    num_updates = total_timesteps // ROLLOUT_LENGTH
    all_rewards = [[] for _ in range(agents_count)]  # per completed episode


    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards   = []

    
    pbar = tqdm(range(num_updates), unit="update")
    for update in pbar:

        # Collect rollouts in parallel
        with ThreadPoolExecutor(max_workers=agents_count) as executor:
            executor.map(lambda a: a.collect_rollout(ROLLOUT_LENGTH), agents)

        # Update all agents
        for agent in agents:
            agent.update()

        # Gather any completed episodes this rollout
        for i, agent in enumerate(agents):
            all_rewards[i].extend(agent.completed_episode_rewards)
            agent.completed_episode_rewards = []

        # Rolling average over last 100 completed episodes per agent
        rolling_avgs = []
        for i in range(agents_count):
            window = all_rewards[i][-100:] if all_rewards[i] else [0]
            rolling_avgs.append(sum(window) / len(window))

        avg_rolling = sum(rolling_avgs) / agents_count
        timestep    = (update + 1) * ROLLOUT_LENGTH

        pbar.set_postfix({
            "timestep":    timestep,
            "rolling_100": f"{avg_rolling:.2f}"
        })

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Pad to same length for DataFrame
    max_len    = max(len(r) for r in all_rewards)
    rewards_df = pd.DataFrame({
        f'Agent {i+1}': all_rewards[i] + [np.nan] * (max_len - len(all_rewards[i]))
        for i in range(agents_count)
    })
    rewards_df.to_csv(f'rewards_per_agent_Lunar_PPO_NoFed_{run}_{timestamp}.csv', index=False)


if __name__ == "__main__":
    for run in range(1):
        print(f"\nRun {run + 1}:")
        seed           = 20 + run * 5
        total_timesteps = 5000 * 2048  
        print(f"Seed: {seed}")
        main(seed, total_timesteps, run)

