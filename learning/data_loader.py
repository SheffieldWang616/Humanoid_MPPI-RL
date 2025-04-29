import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class StateActionDataset(Dataset):
    def __init__(self, state_csv_path, action_csv_path, normalize=False, smooth_window_size=0, add_noise=0.0, 
                 return_type='raw', split='train', train_ratio=0.8, random_split=False, seed=42, state_idxes = []):
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
        
        self.state_idxes = np.array(state_idxes)
        print("=" * 20)
        print(f"Configuration: split={split}, train_size={len(self.train_indices)}, "
              f"eval_size={len(self.eval_indices)}, add_noise={self.add_noise}, "
              f"return_type={self.return_type}, normalize={normalize}")
        print("State idxes:", self.state_idxes)
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
        
        if self.state_idxes:
            current_state = current_state[self.state_idxes]
            next_state = next_state[self.state_idxes]

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
    
class MultiTrajectoryDataset(Dataset):
    def __init__(self, states_dir, actions_dir, normalize=False, smooth_window_size=0, add_noise=0.0,
                 return_type='raw', split='train', train_ratio=0.8, random_split=False, seed=42, state_idxes = []):
        """
        Dataset for loading state and action data from multiple CSV files in a directory
        
        Args:
            states_dir: Directory containing CSV files with state data
            actions_dir: Directory containing CSV files with action data
            normalize: Whether to normalize the data
            smooth_window_size: Window size for smoothing the data
            add_noise: Amount of noise to add to inputs
            return_type: 'raw', 'delta', 'pct' to specify the type of output
            split: 'train' or 'eval' to specify which split to use
            train_ratio: Proportion of data to use for training
            random_split: Whether to randomly split data or use sequential split
            seed: Random seed for reproducibility
        """
        # Validate input parameters
        assert return_type in ['raw', 'delta', 'pct'], "return_type must be one of ['raw', 'delta', 'pct']"
        self.return_type = return_type
        self.add_noise = add_noise
        
        # Find all CSV files in directories
        state_files = sorted([f for f in os.listdir(states_dir) if f.endswith('.csv')])
        action_files = sorted([f for f in os.listdir(actions_dir) if f.endswith('.csv')])
        
        if len(state_files) != len(action_files):
            raise ValueError("Number of state and action CSV files must be the same")
        
        if len(state_files) == 0:
            raise ValueError("No CSV files found in the specified directories")
        
        # Load all trajectories
        self.trajectories = []
        num_states = 0
        num_actions = 0
        
        for state_file, action_file in zip(state_files, action_files):
            state_path = os.path.join(states_dir, state_file)
            action_path = os.path.join(actions_dir, action_file)
            
            # Load data from CSV files
            states = pd.read_csv(state_path).values.astype(np.float32)[1:]  # skip first row
            actions = pd.read_csv(action_path).values.astype(np.float32)[1:]  # skip first row

            if num_states == 0 and num_actions == 0:
                num_states = states.shape[1]
                num_actions = actions.shape[1]
                print(f"Number of states: {num_states}, Number of actions: {num_actions}")

            if states.shape[1] != num_states or actions.shape[1] != num_actions:
                raise ValueError(f"State and action files {state_file} and {action_file} must have the same number of columns")
            
            # Validate data
            if len(states) != len(actions):
                raise ValueError(f"State and action files {state_file} and {action_file} must have same number of rows")
            if len(states) < 2:
                print(f"Warning: Skipping trajectory {state_file} because it has fewer than 2 states")
                continue
                
            # Store trajectory data
            self.trajectories.append({
                'states': states,
                'actions': actions,
                'length': len(states) - 1,  # -1 because we need pairs for input-output
                'file_name': state_file
            })
        
        if not self.trajectories:
            raise ValueError("No valid trajectories found")
            
        # Create train/eval split
        np.random.seed(seed)
        
        # Create a flat list of all valid sample indices across all trajectories
        all_indices = []
        for traj_idx, traj in enumerate(self.trajectories):
            for sample_idx in range(traj['length']):
                all_indices.append((traj_idx, sample_idx))
        
        if random_split:
            # Shuffle all indices and split into train and eval
            np.random.shuffle(all_indices)
            
            train_size = int(len(all_indices) * train_ratio)
            self.train_indices = all_indices[:train_size]
            self.eval_indices = all_indices[train_size:]
        else:
            # Split each trajectory by train_ratio (sequential split)
            self.train_indices = []
            self.eval_indices = []
            
            for traj_idx, traj in enumerate(self.trajectories):
                length = traj['length']
                train_size = int(length * train_ratio)
                
                for i in range(train_size):
                    self.train_indices.append((traj_idx, i))
                    
                for i in range(train_size, length):
                    self.eval_indices.append((traj_idx, i))
        
        self.indices = self.train_indices if split == 'train' else self.eval_indices
        
        # Normalize data if requested
        if normalize:
            # Collect all training data to compute normalization statistics
            train_states = []
            train_actions = []
            
            for traj_idx, sample_idx in self.train_indices:
                traj = self.trajectories[traj_idx]
                train_states.append(traj['states'][sample_idx])
                train_actions.append(traj['actions'][sample_idx])
                
            train_states = np.vstack(train_states)
            train_actions = np.vstack(train_actions)
            
            self.state_mean = train_states.mean(axis=0)
            self.state_std = train_states.std(axis=0) + 1e-6  # Avoid division by zero
            self.action_mean = train_actions.mean(axis=0)
            self.action_std = train_actions.std(axis=0) + 1e-6
            
            # Apply normalization to all trajectories
            for traj in self.trajectories:
                traj['states'] = (traj['states'] - self.state_mean) / self.state_std
                traj['actions'] = (traj['actions'] - self.action_mean) / self.action_std
        
        # Apply smoothing if requested
        if smooth_window_size > 0:
            window_size = smooth_window_size
            for traj in self.trajectories:
                smoothed_states = np.copy(traj['states'])
                for col in range(traj['states'].shape[1]):
                    smoothed_states[:, col] = pd.Series(traj['states'][:, col]).rolling(
                        window=window_size, min_periods=1, center=True).mean().values
                traj['states'] = smoothed_states
        print("State idxes:", state_idxes)
        self.state_idxes = np.array(state_idxes)
        print("=" * 20)
        print(f"Configuration: split={split}, num_trajectories={len(self.trajectories)}, "
              f"train_size={len(self.train_indices)}, eval_size={len(self.eval_indices)}, "
              f"add_noise={self.add_noise}, return_type={self.return_type}, normalize={normalize}")
        print("=" * 20)

    def get_states_actions(self):
        """
        Returns all states and actions in the dataset concatenated
        """
        all_states = []
        all_actions = []
        
        for traj_idx, sample_idx in self.indices:
            traj = self.trajectories[traj_idx]
            all_states.append(traj['states'][sample_idx])
            all_actions.append(traj['actions'][sample_idx])
            
        return np.vstack(all_states), np.vstack(all_actions)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the trajectory index and sample index
        traj_idx, sample_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        # Current state and action
        current_state = traj['states'][sample_idx]
        current_action = traj['actions'][sample_idx]
        
        # Next state (target) - guaranteed to be from the same trajectory
        next_state = traj['states'][sample_idx + 1]
        
        if self.state_idxes is not None:
            current_state = current_state[self.state_idxes]
            next_state = next_state[self.state_idxes]
        
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
    state_csv = "data/2025-04-20_221347/states.csv"
    action_csv = "data/2025-04-20_221347/actions.csv"
    dataset = StateActionDataset(state_csv, action_csv, return_type='delta', random_split=True)
    for input_features, target in dataset:
        # check none of them has 0 or NaN
        if np.any(np.isnan(input_features.numpy())):
            print("NaN found in input or target")
            print(f"Input: {input_features.numpy()}")
            raise ValueError("NaN found in input or target")
        if np.any(np.isclose(input_features.numpy(), 0)):
            print("0 found in input or target")
            print(f"Input: {input_features.numpy()}")
            raise ValueError("0 found in input or target")

    # states, actions = dataset.get_states_actions()
    # states = states[:1000]  # Take a subset for plotting
    
    # # plot states with one subplot for every 5 dimensions
    # import matplotlib.pyplot as plt
    
    # # Calculate number of subplots needed
    # n_dims = states.shape[1]
    # dim_per_subplot = 10
    # n_subplots = (n_dims + dim_per_subplot-1) // dim_per_subplot  # Ceiling division to handle dimensions not divisible by 5
    
    # # Create figure with subplots
    # fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 3*n_subplots))
    
    # # Create a time axis
    # time = np.arange(len(states))
    
    # # Plot each group of 5 dimensions
    # for i in range(n_subplots):
    #     start_dim = i * dim_per_subplot
    #     end_dim = min(start_dim + dim_per_subplot, n_dims)
        
    #     for j in range(start_dim, end_dim):
    #         axes[i].plot(time, states[:, j], label=f'Dim {j}')
        
    #     axes[i].set_title(f'State Dimensions {start_dim}-{end_dim-1}')
    #     axes[i].set_xlabel('Time Step')
    #     axes[i].set_ylabel('Value')
    #     axes[i].legend()
    
    # plt.tight_layout()
    # plt.show()
