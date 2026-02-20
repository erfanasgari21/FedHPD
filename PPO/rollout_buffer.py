import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma, gae_lambda):
        returns = []
        advantages = []

        gae = 0
        next_value = 0

        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + gamma * next_value * (1 - self.dones[step])
                - self.values[step]
            )
            gae = delta + gamma * gae_lambda * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
            next_value = self.values[step]

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )
