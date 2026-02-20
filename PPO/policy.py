import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, activation_func='relu'):
        super(PolicyNetwork, self).__init__()

        self.activation_func = activation_func
        if self.activation_func == 'relu':
            activate_function = nn.ReLU()
        elif self.activation_func == 'tanh':
            activate_function = nn.Tanh()
        else:
            raise ValueError("Invalid activation function. Please use 'relu' or 'tanh'.")

        layers = [nn.Linear(state_size, hidden_sizes[0]), activate_function]
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), activate_function])
        layers.extend([nn.Linear(hidden_sizes[-1], action_size), nn.Softmax(dim=-1)])
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class GaussianPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, activation_fn):
        super(GaussianPolicyNetwork, self).__init__()

        self.layers = nn.ModuleList()
        previous_size = input_dim

        # Add hidden layers
        for size in hidden_sizes:
            self.layers.append(nn.Linear(previous_size, size))
            previous_size = size

        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(previous_size, output_dim)
        self.log_std_layer = nn.Linear(previous_size, output_dim)

        # Activation function
        if activation_fn == 'relu':
            self.activation_fn = F.relu
        elif activation_fn == 'tanh':
            self.activation_fn = torch.tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)

        return mean, std



class PPOActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, activation_fn='relu'):
        super().__init__()

        if activation_fn == 'tanh':
            activation = nn.Tanh
        else:
            activation = nn.ReLU

        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        self.actor_head  = nn.Linear(prev_size, action_size)
        self.critic_head = nn.Linear(prev_size, 1)


    def forward(self, state):
        x      = self.shared(state)
        logits = self.actor_head(x)
        value  = self.critic_head(x)
        return logits, value

    def act(self, state):
        logits, value = self.forward(state)
        dist     = Categorical(logits=logits)   
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist      = Categorical(logits=logits) 
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)