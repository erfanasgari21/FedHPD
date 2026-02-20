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
        self.actions.append(int(action))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma, gae_lambda, last_value):
      returns = []
      advantages = []

      gae = 0.0
      next_value = float(last_value)

      for step in reversed(range(len(self.rewards))):
          nonterminal = 1.0 - float(self.dones[step])

          delta = float(self.rewards[step]) + gamma * next_value * nonterminal - float(self.values[step])
          gae = delta + gamma * gae_lambda * nonterminal * gae

          advantages.insert(0, gae)
          returns.insert(0, gae + float(self.values[step]))

          next_value = float(self.values[step])

      return (
          torch.tensor(returns, dtype=torch.float32),
          torch.tensor(advantages, dtype=torch.float32),
      )
