import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class StateActionDataset(Dataset):
    def __init__(self, state_csv_path, action_csv_path, normalize=False, add_noise=0.0, return_delta=False):
        """
        Dataset for loading state and action data from CSV files
        
        Args:
            state_csv_path: Path to CSV file containing state data
            action_csv_path: Path to CSV file containing action data
            normalize: Whether to normalize the data
        """
        # Load data from CSV files
        self.states = pd.read_csv(state_csv_path).values.astype(np.float32)
        self.actions = pd.read_csv(action_csv_path).values.astype(np.float32)
        self.add_noise = add_noise
        self.return_delta = return_delta

        print("=" * 20)
        print("Configuration: ",
              f"add_noise: {self.add_noise}, "
              f"return_delta: {self.return_delta}, "
              f"normalize: {normalize}")
        print("=" * 20)
        
        # Validate data
        if len(self.states) != len(self.actions):
            raise ValueError("State and action CSV files must have the same number of rows")
        if len(self.states) < 2:
            raise ValueError("Need at least 2 state entries to create input-output pairs")
        
        # Normalize data if requested
        if normalize:
            self.state_mean = self.states.mean(axis=0)
            self.state_std = self.states.std(axis=0) + 1e-6  # Avoid division by zero
            self.action_mean = self.actions.mean(axis=0)
            self.action_std = self.actions.std(axis=0) + 1e-6
            
            self.states = (self.states - self.state_mean) / self.state_std
            self.actions = (self.actions - self.action_mean) / self.action_std
    
    def __len__(self):
        # One less than total length because we need the next state
        return len(self.states) - 1
    
    def __getitem__(self, idx):
        # Current state and action
        current_state = self.states[idx]
        current_action = self.actions[idx]
        
        # Next state (target)
        next_state = self.states[idx + 1]
        
        # Concatenate state and action as input features
        input_features = np.concatenate((current_state, current_action))

        # for each of the feature, add a normal noise with std equal to feature * add_noise
        if self.add_noise > 0:
            noise = np.random.normal(0, self.add_noise * np.abs(input_features), input_features.shape)
            input_features += noise
        
        # Convert to tensors
        input_features = torch.tensor(input_features, dtype=torch.float32)
        if self.return_delta:
            target = torch.tensor(next_state - current_state, dtype=torch.float32)
        else:
            target = torch.tensor(next_state, dtype=torch.float32)
        
        return input_features, target