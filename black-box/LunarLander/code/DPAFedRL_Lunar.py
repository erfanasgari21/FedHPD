import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
from datetime import datetime
from torch.distributions import Categorical
from policy import PolicyNetwork

device = torch.device("cpu")


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
        self.policy = PolicyNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

    def train_reinforce(self):
        obs, _ = self.env.reset()
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        log_probs = []
        rewards = []
        done = False

        while not done:
            probs = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # baseline

        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return sum(rewards)

    def update_with_server(self, avg_probs, state):
        output_probs = self.policy(state)
        # Calculate the loss as the L2 norm (squared difference) between the server average and local output
        kd_loss_avg = torch.sum((output_probs - avg_probs) ** 2)

        self.optimizer.zero_grad()
        kd_loss_avg.backward()
        self.optimizer.step()
        self.scheduler.step()


class ServerAgent:
    def __init__(self, env, state_size, action_size, hidden_sizes, activation_fn, device):
        self.env = env
        self.device = device
        self.policy = PolicyNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.obs, _ = self.env.reset()
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)

    def select_action_and_update_state(self, agents):
        print(self.state)
        agent_probs = [agent.policy(self.state).detach() for agent in agents]
        avg_probs = federated_average(agent_probs)

        action = Categorical(avg_probs).sample()
        next_state, _, done, _ = self.env.step(action.item())
        self.state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        return avg_probs, done


def main(seed, episodes, distill_interval):
    seeds = [seed] * 10
    agents_count = 10
    envs = [gym.make('LunarLander-v3') for _ in range(agents_count)]
    for i, env in enumerate(envs):
        seed = seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        # env.seed(seed)  # Deprecated in gym v0.26+, use reset(seed=...) instead

    state_size = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n
    lr_list = [5e-4, 1e-3, 4e-4, 6e-4, 3e-4,
               4e-4, 5e-4, 2e-3, 2e-4, 1e-4]
    activaton_list = ['relu', 'relu', 'tanh', 'relu', 'tanh',
                      'relu', 'tanh', 'relu', 'tanh', 'relu']
    gamma = 0.99

    hidden_sizes_list = [[128, 128, 256], [64, 64], [128, 128], [128, 256], [256, 256],
                         [512], [64, 128, 64], [32, 32], [512, 512], [1024]]
    agents = [Agent(envs[i], state_size, action_size, hidden_sizes_list[i % len(hidden_sizes_list)],
                    lr_list[i % len(lr_list)], activaton_list[i % len(hidden_sizes_list)], gamma, device=device) for i
              in range(agents_count)]

    # Server Agent setup
    server_env = gym.make('LunarLander-v3')
    server_agent = ServerAgent(server_env, state_size, action_size, hidden_sizes_list[0], activaton_list[0], device)

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards = []

    for episode in range(episodes):
        total_rewards = []

        # Sequential training of agents
        for agent in agents:
            reward = agent.train_reinforce()
            total_rewards.append(reward)

        for i, reward in enumerate(total_rewards):
            rewards_per_agent[i].append(reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        average_rewards.append(average_reward)
        print(f"Episode {episode + 1}:")
        for idx, reward in enumerate(total_rewards):
            if len(rewards_per_agent[idx]) >= 100:
                last_100_avg = sum(rewards_per_agent[idx][-100:]) / 100
            else:
                last_100_avg = sum(rewards_per_agent[idx]) / len(rewards_per_agent[idx])
            print(f"  Agent {idx + 1} Reward: {reward} | Average Reward: {last_100_avg:.4f}")

        print(f"  Average Reward: {average_reward}")

        if (episode + 1) % distill_interval == 0:
            avg_probs, done = server_agent.select_action_and_update_state(agents)
            for agent in agents:
                agent.update_with_server(avg_probs, server_agent.state)
            if done:
                server_agent.state = torch.FloatTensor(server_env.reset()).unsqueeze(0).to(device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_filename = f'rewards_per_agent_Black_Box_Lunar_10_{distill_interval}_{timestamp}.csv'
    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(rewards_filename, index=False)

    average_rewards_filename = f'rewards_Black_Box_Lunar_10_{distill_interval}_{timestamp}.csv'
    average_rewards_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Average Reward': average_rewards})
    average_rewards_df.to_csv(average_rewards_filename, index=False)


if __name__ == "__main__":
    for run in range(5):
        print(f"\nRun {run + 1}:")
        seed = 20 + run * 5
        episodes = 3000
        distill_interval = 1
        print(f"Seed: {seed}")
        print(f"Episodes: {episodes}")
        print(f"Distill Interval: {distill_interval}")
        main(seed, episodes, distill_interval)
