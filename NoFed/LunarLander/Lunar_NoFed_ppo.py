import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from torch.distributions import Categorical
from policy import PPOActorCriticNetwork
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


def federated_average_values(values):
    """Average value estimates across agents"""
    avg_values = None
    for value in values:
        if avg_values is None:
            avg_values = value.clone().detach()
        else:
            avg_values += value.clone().detach()
    return avg_values / len(values)


def compute_gae(rewards, values, gamma=0.99, lambda_gae=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    gae = 0
    next_value = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1].item()
        
        delta = rewards[t] + gamma * next_value - values[t].item()
        gae = delta + gamma * lambda_gae * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages).to(device)
    returns = advantages + torch.stack(values).squeeze()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    
    return advantages, returns


class Agent:
    def __init__(self, env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device,
                 ppo_epochs=3, ppo_batch_size=64, ppo_clip=0.2):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.ppo_clip = ppo_clip
        
        # PPO Actor-Critic network
        self.policy = PPOActorCriticNetwork(state_size, action_size, hidden_sizes, activation_fn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # Store network configuration for model upload/download
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.activation_fn = activation_fn

    def collect_trajectory(self):
        """Collect one trajectory/episode"""
        obs, _ = self.env.reset()
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        done = False

        while not done:
            with torch.no_grad():
                action_probs, value = self.policy.get_action_and_value(state)
            
            dist = Categorical(action_probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            states.append(state.clone())
            actions.append(action)
            rewards.append(reward)
            values.append(value.detach())
            log_probs.append(dist.log_prob(action))
            
            state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        return states, actions, rewards, values, log_probs, sum(rewards)

    def train_ppo(self):
        """Train using PPO algorithm"""
        states, actions, rewards, values, old_log_probs, total_reward = self.collect_trajectory()
        
        # Compute advantages and returns using GAE
        advantages, returns = compute_gae(rewards, values, self.gamma, lambda_gae=0.95)
        
        old_log_probs = torch.stack(old_log_probs).detach()
        states = torch.cat(states)
        actions = torch.stack(actions)
        returns = returns.detach()
        advantages = advantages.detach()
        
        # PPO training for multiple epochs
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(len(states))
            for i in range(0, len(states), self.ppo_batch_size):
                batch_indices = indices[i:i + self.ppo_batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                action_probs, values_pred = self.policy.get_action_and_value(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # PPO loss with clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (value function)
                critic_loss = F.smooth_l1_loss(values_pred.squeeze(), batch_returns)
                
                # Entropy bonus for exploration
                entropy = dist.entropy().mean()
                
                # Combined loss
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.scheduler.step()
        return total_reward

    def upload_model(self):
        """Upload model details for server distillation"""
        model_details = {
            'state_dict': self.policy.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_sizes': self.hidden_sizes,
            'lr': self.lr,
            'activation_fn': self.activation_fn,
        }
        return model_details

    def update_model(self, model_details):
        """Download updated model from server"""
        self.policy.load_state_dict(model_details['state_dict'])


class Server:
    def __init__(self, agents, device):
        self.agents = agents
        self.device = device

    def collect_and_average_probs_and_values(self, models, sampled_states):
        """Collect action probs and values from all agents, then average"""
        agent_probs = []
        agent_values = []
        
        for model_details in models:
            policy = PPOActorCriticNetwork(
                model_details['state_size'], 
                model_details['action_size'],
                model_details['hidden_sizes'], 
                model_details['activation_fn']
            ).to(self.device)
            policy.load_state_dict(model_details['state_dict'])
            
            with torch.no_grad():
                probs, values = policy.get_action_and_value(sampled_states)
            agent_probs.append(probs.detach())
            agent_values.append(values.detach())
        
        avg_probs = federated_average(agent_probs)
        avg_values = federated_average_values(agent_values)
        
        return agent_probs, avg_probs, agent_values, avg_values

    def distill_and_update_models(self, models, sampled_states, distill_loop=5):
        """Knowledge distillation for both actor and critic"""
        dataset_size = sampled_states.size(0)
        assert dataset_size == 10000, "sampled_states size should be 10000"

        # Split dataset into four parts
        quarter_size = dataset_size // 4
        sampled_states_parts = [
            sampled_states[i * quarter_size: (i + 1) * quarter_size] for i in range(4)
        ]

        for distillation_round in range(distill_loop):
            print(f"\nKnowledge Distillation Round {distillation_round + 1}:")

            for part_idx, sampled_states_part in enumerate(sampled_states_parts):
                print(f"Processing part {part_idx + 1}")

                agent_probs, avg_probs, agent_values, avg_values = self.collect_and_average_probs_and_values(
                    models, sampled_states_part
                )

                for agent_idx, model_details in enumerate(models):
                    policy = PPOActorCriticNetwork(
                        model_details['state_size'], 
                        model_details['action_size'],
                        model_details['hidden_sizes'], 
                        model_details['activation_fn']
                    ).to(self.device)
                    policy.load_state_dict(model_details['state_dict'])
                    optimizer = optim.Adam(policy.parameters(), lr=model_details['lr'])

                    output_probs, output_values = policy.get_action_and_value(sampled_states_part)

                    # Knowledge distillation losses for both actor and critic
                    kl_divergences = []
                    mse_losses = []
                    
                    for other_agent_probs, other_agent_values in zip(agent_probs, agent_values):
                        if torch.equal(other_agent_probs, output_probs):
                            continue
                        
                        # KL divergence for actor (policy)
                        kl_divergence = torch.nn.functional.kl_div(
                            output_probs.log(), other_agent_probs, reduction='batchmean'
                        )
                        kl_divergences.append(kl_divergence)
                        
                        # MSE loss for critic (value function)
                        mse_loss = F.mse_loss(output_values.squeeze(), other_agent_values.squeeze())
                        mse_losses.append(mse_loss)

                    # Average KL divergences (actor distillation with other agents)
                    kd_loss_inter_actor = sum(kl_divergences) / (len(agent_probs) - 1) if kl_divergences else 0
                    
                    # Average MSE losses (critic distillation with other agents)
                    kd_loss_inter_critic = sum(mse_losses) / (len(agent_values) - 1) if mse_losses else 0
                    
                    # KL divergence with server average (actor)
                    kd_loss_avg_actor = torch.nn.functional.kl_div(
                        output_probs.log(), avg_probs, reduction='batchmean'
                    )
                    
                    # MSE loss with server average (critic)
                    kd_loss_avg_critic = F.mse_loss(
                        output_values.squeeze(), avg_values.squeeze()
                    )
                    
                    # Combined KD loss: inter-agent + server average for both actor and critic
                    kd_loss = (kd_loss_inter_actor + kd_loss_avg_actor) + 0.5 * (kd_loss_inter_critic + kd_loss_avg_critic)

                    optimizer.zero_grad()
                    kd_loss.backward()
                    optimizer.step()

                    models[agent_idx]['state_dict'] = policy.state_dict()

                    print(f"Agent {agent_idx + 1} | Loss: {kd_loss.item():.4f}")

                print(f"Part {part_idx + 1} of Knowledge Distillation Round {distillation_round + 1} Completed!")

            print(f"Knowledge Distillation Round {distillation_round + 1} Completed!")

        return models


def main(seed, episodes, distill_interval):
    seeds = [seed] * 10
    agents_count = 10
    envs = [gym.make('LunarLander-v3') for _ in range(agents_count)]
    for i, env in enumerate(envs):
        seed_val = seeds[i]
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

    state_size = envs[0].observation_space.shape[0]
    action_size = envs[0].action_space.n
    lr_list = [5e-4, 1e-3, 4e-4, 6e-4, 3e-4,
               4e-4, 5e-4, 2e-3, 2e-4, 1e-4]
    activation_list = ['relu', 'relu', 'tanh', 'relu', 'tanh',
                       'relu', 'tanh', 'relu', 'tanh', 'relu']
    gamma = 0.99

    hidden_sizes_list = [[128, 128, 256], [64, 64], [128, 128], [128, 256], [256, 256],
                         [512], [64, 128, 64], [32, 32], [512, 512], [1024]]

    agents = [Agent(envs[i], state_size, action_size, hidden_sizes_list[i % len(hidden_sizes_list)],
                    lr_list[i % len(lr_list)], activation_list[i % len(activation_list)], 
                    gamma, device=device, ppo_epochs=3, ppo_batch_size=32, ppo_clip=0.2) 
              for i in range(agents_count)]

    server = Server(agents, device)

    rewards_per_agent = [[] for _ in range(agents_count)]
    average_rewards = []

    for episode in range(episodes):
        total_rewards = []

        def train_agent(agent):
            return agent.train_ppo()

        # Parallel training of agents
        with ThreadPoolExecutor(max_workers=agents_count) as executor:
            results = list(executor.map(train_agent, agents))

        for i, reward in enumerate(results):
            rewards_per_agent[i].append(reward)
            total_rewards.append(reward)

        average_reward = sum(total_rewards) / len(total_rewards)
        average_rewards.append(average_reward)
        print(f"Episode {episode + 1}:")
        for idx, reward in enumerate(total_rewards):
            if len(rewards_per_agent[idx]) >= 100:
                last_100_avg = sum(rewards_per_agent[idx][-100:]) / 100
            else:
                last_100_avg = sum(rewards_per_agent[idx]) / len(rewards_per_agent[idx])
            print(f"  Agent {idx + 1} Reward: {reward:.2f} | Average Reward: {last_100_avg:.4f}")

        print(f"  Average Reward: {average_reward:.2f}")

        if (episode + 1) % distill_interval == 0:
            csv_file = 'KD_state_10-from-each.csv'
            df = pd.read_csv(csv_file)
            sampled_states = df['State'].apply(lambda x: torch.tensor([float(i) for i in x.split(',')]))
            sampled_states = torch.stack(sampled_states.values.tolist()).to(device)

            # Upload all agent models
            models = [agent.upload_model() for agent in agents]
            
            # Server performs knowledge distillation
            updated_models = server.distill_and_update_models(models, sampled_states)

            # Download updated models
            for agent, updated_model in zip(agents, updated_models):
                agent.update_model(updated_model)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    rewards_filename = f'rewards_per_agent_PPO_Lunar_NoFed_{timestamp}.csv'
    rewards_df = pd.DataFrame({f'Agent {i + 1}': rewards_per_agent[i] for i in range(agents_count)})
    rewards_df.to_csv(rewards_filename, index=False)

    average_rewards_filename = f'rewards_PPO_Lunar_NoFed_{distill_interval}_{timestamp}.csv'
    average_rewards_df = pd.DataFrame({'Episode': range(1, episodes + 1), 'Average Reward': average_rewards})
    average_rewards_df.to_csv(average_rewards_filename, index=False)


if __name__ == "__main__":
    for run in range(1):
        print(f"\nRun {run + 1}:")
        seed = 20 + run * 5
        episodes = 10
        distill_interval = 5000
        print(f"Seed: {seed}")
        print(f"Episodes: {episodes}")
        print(f"Distill Interval: {distill_interval}")
        main(seed, episodes, distill_interval)