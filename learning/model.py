import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPStatePredictor(nn.Module):
    """
    A simple MLP model for predicting the next state given the current state and action.
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers.
    """
    def __init__(self, state_dim=55, action_dim=21, hidden_dim=128):
        super(MLPStatePredictor, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, state_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x