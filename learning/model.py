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
    
    Each feature is treated as a token and can attend to all other features.
    This enables the model to capture complex interactions between different state and action components.
    
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dim (int): Dimension of the hidden layers.
        num_heads (int): Number of attention heads.
        attn_layers (int): Number of attention layers.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, state_dim=55, action_dim=21, hidden_dim=128, 
                 num_heads=4, attn_layers=2, dropout_rate=0.1):
        super(FeatureAttentionStatePredictor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        
        # Feature encoding - transform each scalar feature into a vector representation
        self.feature_encoding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Position embeddings for each feature
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.pos_embedding)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(attn_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'attention': nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True
                ),
                'dropout1': nn.Dropout(dropout_rate),
                'norm2': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'dropout2': nn.Dropout(dropout_rate)
            })
            self.layers.append(layer)
        
        # Output layer to predict the next state
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, return_attn=False):
        batch_size = x.shape[0]
        
        # Reshape input to [batch_size, input_dim, 1]
        x = x.view(batch_size, self.input_dim, 1)
        
        # Encode each feature
        x = self.feature_encoding(x)  # [batch_size, input_dim, hidden_dim]
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Store attention weights if needed
        attentions = [] if return_attn else None
        
        # Apply transformer layers
        for layer in self.layers:
            # Layer normalization and attention (Pre-LN architecture)
            residual = x
            x_norm = layer['norm1'](x)
            x_attn, attn_weights = layer['attention'](x_norm, x_norm, x_norm)
            if return_attn:
                attentions.append(attn_weights)
            
            # Residual connection and dropout
            x = residual + layer['dropout1'](x_attn)
            
            # Layer normalization and FFN
            residual = x
            x_norm = layer['norm2'](x)
            x_ffn = layer['ffn'](x_norm)
            
            # Residual connection and dropout
            x = residual + layer['dropout2'](x_ffn)
        
        # Project features back to scalars
        x = self.output_layer(x)  # [batch_size, input_dim, 1]
        x = x.squeeze(-1)  # [batch_size, input_dim]
        
        # Extract only the state part as output
        x = x[:, :self.state_dim]
        
        if return_attn:
            return x, attentions
        else:
            return x

if __name__ == "__main__":
    # Example usage
    model = MLPStatePredictor(state_dim=55, action_dim=21, hidden_dim=128, use_batch_norm=True, dropout_rate=0.2)
    print(model)
    
    # Example input: batch of 10 samples, each with state_dim + action_dim features
    example_input = torch.randn(10, 55 + 21)
    output = model(example_input)
    print("Output shape:", output.shape)  # Should be [10, 55]
    # Example usage of FeatureAttentionStatePredictor
    attn_model = FeatureAttentionStatePredictor(state_dim=55, action_dim=21, hidden_dim=128, num_heads=4, attn_layers=2)
    print(attn_model)
    example_input = torch.randn(10, 55 + 21)
    attn_output = attn_model(example_input)
    print("Attention Output shape:", attn_output.shape)  # Should be [10, 55]
