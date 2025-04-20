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
        use_batch_norm (bool): Whether to use batch normalization.
        dropout_rate (float): Dropout probability (0 means no dropout).
    """
    def __init__(self, state_dim=55, action_dim=21, hidden_dim=128, use_batch_norm=False, dropout_rate=0.0, hidden_layers=2):
        super(MLPStatePredictor, self).__init__()
        self.use_batch_norm = use_batch_norm
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class FeatureAttentionStatePredictor(nn.Module):
    """
    A model that uses feature-wise attention to predict the next state given the current state and action.
    
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers.
    """
    def __init__(self, state_dim=55, action_dim=21, hidden_dim=128, attn_layers=2):
        super(FeatureAttentionStatePredictor, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.attentions = nn.ModuleList([attention for _ in range(attn_layers)])
        self.fc2 = nn.Linear(hidden_dim, state_dim)
        self.relu = nn.ReLU()

    def forward(self, x, return_attn=False):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(1)
        attentions = [] if return_attn else None
        for attn in self.attentions:
            x, attn_weights = attn(x, x, x)
            x = self.relu(x)
            attentions.append(attn_weights) if return_attn else None

        x = x.squeeze(1)
        x = self.fc2(x)

        if return_attn:
            return x, attentions
        else:
            return x
