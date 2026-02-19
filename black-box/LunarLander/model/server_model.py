import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from policy.policy import PolicyNetwork

device = torch.device("cpu")

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
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        log_probs = []
        rewards = []
        done = False

        while not done:
            probs = self.policy(state)
            dist = Categorical(probs)
            action = dist.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            log_probs.append(dist.log_prob(action))
            rewards.append(reward)

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)
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


def main(seed, episodes):
    # Environment setup
    env = gym.make('LunarLander-v3')
    np.random.seed(seed)
    torch.manual_seed(seed)
    # env.seed(seed)  # Deprecated in gym v0.26+, use reset(seed=...) instead

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_sizes = [256, 256]
    lr = 4e-4
    activation_fn = 'tanh'
    gamma = 0.99

    agent = Agent(env, state_size, action_size, hidden_sizes, lr, activation_fn, gamma, device=device)

    rewards = []
    average_rewards = []

    for episode in range(episodes):
        reward = agent.train_reinforce()
        rewards.append(reward)
        average_reward = np.mean(rewards[-100:])  # Moving average of the last 100 rewards
        average_rewards.append(average_reward)

        print(f"Seed {seed} | Episode {episode + 1}: Reward: {reward} | Average Reward: {average_reward}")


        if (episode + 1) >= 2000 and (episode + 1) % 100 == 0:
            model_filename = f'agent_model_seed_{seed}_episode_{episode + 1}.pt'
            torch.save(agent.policy.state_dict(), model_filename)
            print(f"Model saved for seed {seed} at episode {episode + 1} to {model_filename}")

    print(f"Training completed for seed {seed}!")


if __name__ == "__main__":
    seed = 20
    episodes = 4000

    print(f"Starting training for seed {seed}")
    main(seed, episodes)
