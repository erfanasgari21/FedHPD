import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


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
    """Shared base network with separate Actor (policy) and Critic (value) heads"""
    
    def __init__(self, state_size, action_size, hidden_sizes, activation_fn='relu'):
        super(PPOActorCriticNetwork, self).__init__()
        
        # Shared base network
        self.shared_layers = nn.ModuleList()
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Actor head (policy)
        self.actor_head = nn.Linear(prev_size, action_size)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(prev_size, 1)
        
        # Activation function
        if activation_fn == 'relu':
            self.activation = F.relu
        elif activation_fn == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
    
    def forward(self, state):
        """Forward pass through shared layers and both heads"""
        x = state
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        # Actor and Critic outputs
        action_probs = F.softmax(self.actor_head(x), dim=-1)
        value = self.critic_head(x)
        
        return action_probs, value
    
    def get_action_and_value(self, state):
        """Get action probabilities and value estimation"""
        return self.forward(state)
