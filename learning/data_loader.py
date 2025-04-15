import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class StateActionDataset(Dataset):
    def __init__(self, state_csv_path, action_csv_path, normalize=False, add_noise=0.0, 
                 return_delta=False, split='train', train_ratio=0.8, random_split=False, seed=42):
        """
        Dataset for loading state and action data from CSV files
        
        Args:
            state_csv_path: Path to CSV file containing state data
            action_csv_path: Path to CSV file containing action data
            normalize: Whether to normalize the data
            add_noise: Amount of noise to add to inputs
            return_delta: Whether to return state difference instead of next state
            split: 'train' or 'eval' to specify which split to use
            train_ratio: Proportion of data to use for training
            random_split: Whether to randomly split data or use sequential split
            seed: Random seed for reproducibility
        """
        # Load data from CSV files
        self.states = pd.read_csv(state_csv_path).values.astype(np.float32)
        self.actions = pd.read_csv(action_csv_path).values.astype(np.float32)
        self.add_noise = add_noise
        self.return_delta = return_delta
        
        # Validate data
        if len(self.states) != len(self.actions):
            raise ValueError("State and action CSV files must have the same number of rows")
        if len(self.states) < 2:
            raise ValueError("Need at least 2 state entries to create input-output pairs")
        
        # Create train/eval split
        total_samples = len(self.states) - 1  # -1 because we need pairs
        train_size = int(total_samples * train_ratio)
        
        if random_split:
            np.random.seed(seed)
            indices = np.random.permutation(total_samples)
        else:
            indices = np.arange(total_samples)
            
        self.train_indices = indices[:train_size]
        self.eval_indices = indices[train_size:]
        self.indices = self.train_indices if split == 'train' else self.eval_indices
        
        # Normalize data if requested
        if normalize:
            # Only compute normalization stats on training data
            train_states = self.states[self.train_indices]
            train_actions = self.actions[self.train_indices]
            
            self.state_mean = train_states.mean(axis=0)
            self.state_std = train_states.std(axis=0) + 1e-6  # Avoid division by zero
            self.action_mean = train_actions.mean(axis=0)
            self.action_std = train_actions.std(axis=0) + 1e-6
            
            self.states = (self.states - self.state_mean) / self.state_std
            self.actions = (self.actions - self.action_mean) / self.action_std
        
        print("=" * 20)
        print(f"Configuration: split={split}, train_size={len(self.train_indices)}, "
              f"eval_size={len(self.eval_indices)}, add_noise={self.add_noise}, "
              f"return_delta={self.return_delta}, normalize={normalize}")
        print("=" * 20)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our split
        actual_idx = self.indices[idx]
        
        # Current state and action
        current_state = self.states[actual_idx]
        current_action = self.actions[actual_idx]
        
        # Next state (target)
        next_state = self.states[actual_idx + 1]
        
        # Concatenate state and action as input features
        input_features = np.concatenate((current_state, current_action))

        # Add noise if specified
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