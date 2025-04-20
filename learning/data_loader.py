import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class StateActionDataset(Dataset):
    def __init__(self, state_csv_path, action_csv_path, normalize=False, smooth_window_size=0, add_noise=0.0, 
                 return_type='raw', split='train', train_ratio=0.8, random_split=False, seed=42):
        """
        Dataset for loading state and action data from CSV files
        
        Args:
            state_csv_path: Path to CSV file containing state data
            action_csv_path: Path to CSV file containing action data
            normalize: Whether to normalize the data
            add_noise: Amount of noise to add to inputs
            return_type: 'raw', 'delta', 'pct' to specify the type of output
            split: 'train' or 'eval' to specify which split to use
            train_ratio: Proportion of data to use for training
            random_split: Whether to randomly split data or use sequential split
            seed: Random seed for reproducibility
        """
        # Load data from CSV files
        self.states = pd.read_csv(state_csv_path).values.astype(np.float32)[1:] # skip first row
        self.actions = pd.read_csv(action_csv_path).values.astype(np.float32)[1:] # skip first row
        self.add_noise = add_noise
        assert return_type in ['raw', 'delta', 'pct'], "return_type must be one of ['raw', 'delta', 'pct']"
        self.return_type = return_type
        
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
        
        if smooth_window_size:
            # Apply centered smoothing to each dimension separately
            window_size = smooth_window_size
            smoothed_states = np.copy(self.states)
            for col in range(self.states.shape[1]):
                smoothed_states[:, col] = pd.Series(self.states[:, col]).rolling(
                    window=window_size, min_periods=1, center=True).mean().values
            self.states = smoothed_states
        
        print("=" * 20)
        print(f"Configuration: split={split}, train_size={len(self.train_indices)}, "
              f"eval_size={len(self.eval_indices)}, add_noise={self.add_noise}, "
              f"return_type={self.return_type}, normalize={normalize}")
        print("=" * 20)
    
    def get_states_actions(self):
        return self.states, self.actions

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
        if self.return_type == 'delta':
            target = torch.tensor(next_state - current_state, dtype=torch.float32)
        elif self.return_type == 'raw':
            target = torch.tensor(next_state, dtype=torch.float32)
        elif self.return_type == 'pct':
            target = torch.tensor((next_state - current_state) / (current_state + 1e-6), dtype=torch.float32)
        
        return input_features, target
    
if __name__ == "__main__":
    state_csv = "data/2025-04-19_153833/states.csv"
    action_csv = "data/2025-04-19_153833/actions.csv"
    dataset = StateActionDataset(state_csv, action_csv, return_type='delta', smooth_window_size=5, random_split=True)
    states, actions = dataset.get_states_actions()
    states = states[:1000]  # Take a subset for plotting
    
    # plot states with one subplot for every 5 dimensions
    import matplotlib.pyplot as plt
    
    # Calculate number of subplots needed
    n_dims = states.shape[1]
    dim_per_subplot = 10
    n_subplots = (n_dims + dim_per_subplot-1) // dim_per_subplot  # Ceiling division to handle dimensions not divisible by 5
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 3*n_subplots))
    
    # Create a time axis
    time = np.arange(len(states))
    
    # Plot each group of 5 dimensions
    for i in range(n_subplots):
        start_dim = i * dim_per_subplot
        end_dim = min(start_dim + dim_per_subplot, n_dims)
        
        for j in range(start_dim, end_dim):
            axes[i].plot(time, states[:, j], label=f'Dim {j}')
        
        axes[i].set_title(f'State Dimensions {start_dim}-{end_dim-1}')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
