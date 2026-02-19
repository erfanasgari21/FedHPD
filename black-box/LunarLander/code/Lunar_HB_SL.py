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
from policy import PolicyNetwork
from concurrent.futures import ThreadPoolExecutor

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

    def calculate_kd_loss(self, sampled_states_part, agent_probs, avg_probs):
        output_probs = self.policy(sampled_states_part)

        kd_loss_avg = torch.nn.functional.kl_div(output_probs.log(), avg_probs, reduction='batchmean')
        kd_loss = kd_loss_avg

        self.optimizer.zero_grad()
        kd_loss.backward()
        self.optimizer.step()

        return kd_loss.item()

class Server:
    def __init__(self, agents, device):
        self.agents = agents
        self.device = device

    def collect_and_average_probs(self, sampled_states):
        agent_probs = [agent.policy(sampled_states).detach() for agent in self.agents]
        avg_probs = federated_average(agent_probs)
        return agent_probs, avg_probs

def main(seed, episodes, distill_interval):
    seeds = [seed] * 10  # 生成相同种子的列表
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
               4e-4, 5e-4, 1e-3, 2e-4, 1e-4]
    activaton_list = ['relu', 'relu', 'tanh', 'relu', 'tanh',
                      'relu', 'tanh', 'relu', 'tanh', 'relu']
    gamma = 0.99

    hidden_sizes_list = [[128, 128, 256], [64, 64], [128, 128], [128, 256], [256, 256],
                         [512], [64, 128, 64], [32, 32], [512, 512], [1024]]
    
    agents = [Agent(envs[i], state_size, action_size, hidden_sizes_list[i % len(hidden_sizes_list)],
                    lr_list[i % len(lr_list)], activaton_list[i % len(hidden_sizes_list)], gamma, device=device) for i
              in range(agents_count)]

    server = Server(agents, device)

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards = []
    kd_losses = {f'Agent {i + 1}': [] for i in range(agents_count)}

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
            csv_file = 'KD_state_10-from-each.csv'
            df = pd.read_csv(csv_file)
            sampled_states = df['State'].apply(lambda x: torch.tensor([float(i) for i in x.split(',')]))
            sampled_states = torch.stack(sampled_states.values.tolist()).to(device)

            dataset_size = sampled_states.size(0)
            assert dataset_size == 10000, "sampled_states 的大小应为 5000"

            # 分割数据集为四部分
            quarter_size = dataset_size // 4
            sampled_states_parts = [
                sampled_states[i * quarter_size: (i + 1) * quarter_size] for i in range(4)
            ]

            distill_loop = 5
            for distillation_round in range(distill_loop):
                print(f"\nKnowledge Distillation Round {distillation_round + 1}:")

                for part_idx, sampled_states_part in enumerate(sampled_states_parts):
                    print(f"Processing part {part_idx + 1}")

                    agent_probs, avg_probs = server.collect_and_average_probs(sampled_states_part)

                    for agent_idx, agent in enumerate(agents):
                        kd_loss = agent.calculate_kd_loss(sampled_states_part, agent_probs, avg_probs)
                        kd_losses[f'Agent {agent_idx + 1}'].append(
                            (episode + 1, distillation_round + 1, kd_loss))
                        print(f"Agent {agent_idx + 1} | Loss: {kd_loss:.4f}")

                    print(f"Part {part_idx + 1} of Knowledge Distillation Round {distillation_round + 1} Completed!")

                print(f"Knowledge Distillation Round {distillation_round + 1} Completed!")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_filename = f'rewards_per_agent_Lunar_Black_10_{distill_interval}_{timestamp}.csv'
    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(rewards_filename, index=False)

    average_rewards_filename = f'rewards_Lunar_Black_10_{distill_interval}_{timestamp}.csv'
    average_rewards_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Average Reward': average_rewards})
    average_rewards_df.to_csv(average_rewards_filename, index=False)

if __name__ == "__main__":
    for run in range(5):
        print(f"\nRun {run + 1}:")
        seed = 20 + run * 5
        episodes = 3000
        distill_interval = 5
        print(f"Seed: {seed}")
        print(f"Episodes: {episodes}")
        print(f"Distill Interval: {distill_interval}")
        main(seed, episodes, distill_interval)
