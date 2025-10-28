"""
Data generation for VP-FNO training.
Generates training data by solving Vlasov-Poisson equations numerically.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import odeint
from scipy.fft import fft, ifft, fftfreq
import h5py
import os
from tqdm import tqdm

from config import PhysicsConfig, TrainingConfig


class VPDataGenerator:
    """
    Generate Vlasov-Poisson training data using numerical solver.
    """
    
    def __init__(self, physics_config: PhysicsConfig):
        self.config = physics_config
        self.x, self.v, self.t = physics_config.get_grids()
        self.X, self.V = np.meshgrid(self.x, self.v, indexing='ij')
    
    def initialize_two_stream(self, amplitude=None, v0=None, vth=None):
        """
        Create initial condition for two-stream instability.
        
        Args:
            amplitude: Perturbation amplitude (default from config)
            v0: Beam velocity (default from config)
            vth: Thermal velocity (default from config)
        
        Returns:
            f0: Initial distribution [Nx, Nv]
        """
        if amplitude is None:
            amplitude = self.config.amplitude
        if v0 is None:
            v0 = self.config.v0
        if vth is None:
            vth = self.config.vth
        
        # Two counter-streaming Maxwellian beams
        f1 = (1/(2*np.sqrt(2*np.pi)*vth)) * np.exp(-((self.V-v0)**2)/(2*vth**2))
        f2 = (1/(2*np.sqrt(2*np.pi)*vth)) * np.exp(-((self.V+v0)**2)/(2*vth**2))
        
        # Add sinusoidal perturbation
        k = self.config.k_mode
        perturbation = 1 + amplitude * np.cos(k * self.X)
        
        f0 = (f1 + f2) * perturbation
        
        return f0
    
    def solve_poisson(self, f):
        """
        Solve Poisson equation for electric field.
        
        Args:
            f: Distribution function [Nx, Nv]
        
        Returns:
            E: Electric field [Nx]
        """
        # Compute charge density: ρ = ∫f dv - 1
        rho = np.trapz(f, self.v, axis=1) - 1.0
        
        # Solve in Fourier space: E = -i*ρ_k / k
        rho_k = fft(rho)
        k = fftfreq(self.config.Nx, self.config.dx) * 2 * np.pi
        k[0] = 1.0  # Avoid division by zero
        
        E_k = -1j * rho_k / k
        E_k[0] = 0.0  # Zero mean field
        
        E = np.real(ifft(E_k))
        
        return E
    
    def vlasov_rhs(self, f_flat, t):
        """
        Right-hand side of Vlasov equation for time integration.
        
        df/dt = -v * df/dx - E * df/dv
        
        Args:
            f_flat: Flattened distribution [Nx*Nv]
            t: Time
        
        Returns:
            df_dt_flat: Time derivative [Nx*Nv]
        """
        # Reshape to 2D
        f = f_flat.reshape((self.config.Nx, self.config.Nv))
        
        # Compute electric field
        E = self.solve_poisson(f)
        
        # Compute derivatives using spectral method
        f_k = fft(f, axis=0)
        kx = fftfreq(self.config.Nx, self.config.dx) * 2 * np.pi
        
        # df/dx in Fourier space
        df_dx_k = 1j * kx[:, np.newaxis] * f_k
        df_dx = np.real(ifft(df_dx_k, axis=0))
        
        # df/dv using finite difference (central difference)
        df_dv = np.zeros_like(f)
        df_dv[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * self.config.dv)
        df_dv[:, 0] = (f[:, 1] - f[:, 0]) / self.config.dv
        df_dv[:, -1] = (f[:, -1] - f[:, -2]) / self.config.dv
        
        # Vlasov equation RHS
        v_expanded = self.v[np.newaxis, :]
        E_expanded = E[:, np.newaxis]
        
        df_dt = -v_expanded * df_dx - E_expanded * df_dv
        
        return df_dt.flatten()
    
    def solve_vlasov(self, f0, t_eval=None):
        """
        Solve Vlasov-Poisson equations.
        
        Args:
            f0: Initial distribution [Nx, Nv]
            t_eval: Time points for evaluation (default: all time steps)
        
        Returns:
            solution: Dictionary with 'f', 'E', 't'
        """
        if t_eval is None:
            t_eval = self.t
        
        # Flatten initial condition
        f0_flat = f0.flatten()
        
        # Integrate using scipy
        print(f"Solving VP system from t={t_eval[0]:.2f} to t={t_eval[-1]:.2f}...")
        sol = odeint(self.vlasov_rhs, f0_flat, t_eval, 
                    rtol=1e-6, atol=1e-8)
        
        # Reshape solution
        f_trajectory = sol.reshape((-1, self.config.Nx, self.config.Nv))
        
        # Compute electric field for each time step
        E_trajectory = np.array([self.solve_poisson(f) for f in f_trajectory])
        
        return {
            'f': f_trajectory,
            'E': E_trajectory,
            't': t_eval
        }
    
    def generate_dataset(self, n_samples, vary_parameters=True, save_path=None):
        """
        Generate multiple samples with varying initial conditions.
        
        Args:
            n_samples: Number of samples to generate
            vary_parameters: If True, vary amplitude, v0, vth
            save_path: Path to save dataset (HDF5 format)
        
        Returns:
            dataset: Dictionary containing all samples
        """
        dataset = {
            'f0': [],
            'f_trajectories': [],
            'E_trajectories': [],
            't': self.t,
            'x': self.x,
            'v': self.v,
            'parameters': []
        }
        
        print(f"Generating {n_samples} samples...")
        
        for i in tqdm(range(n_samples)):
            if vary_parameters:
                # Vary parameters randomly
                amplitude = np.random.uniform(0.05, 0.15)
                v0 = np.random.uniform(1.5, 2.5)
                vth = np.random.uniform(0.8, 1.2)
            else:
                amplitude = self.config.amplitude
                v0 = self.config.v0
                vth = self.config.vth
            
            # Generate initial condition
            f0 = self.initialize_two_stream(amplitude, v0, vth)
            
            # Solve VP system
            solution = self.solve_vlasov(f0)
            
            # Store results
            dataset['f0'].append(f0)
            dataset['f_trajectories'].append(solution['f'])
            dataset['E_trajectories'].append(solution['E'])
            dataset['parameters'].append({
                'amplitude': amplitude,
                'v0': v0,
                'vth': vth
            })
        
        # Convert lists to arrays
        dataset['f0'] = np.array(dataset['f0'])
        dataset['f_trajectories'] = np.array(dataset['f_trajectories'])
        dataset['E_trajectories'] = np.array(dataset['E_trajectories'])
        
        # Save to HDF5 if path provided
        if save_path:
            self.save_dataset(dataset, save_path)
        
        return dataset
    
    def save_dataset(self, dataset, filepath):
        """Save dataset to HDF5 file."""
        print(f"Saving dataset to {filepath}...")
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('f0', data=dataset['f0'])
            f.create_dataset('f_trajectories', data=dataset['f_trajectories'])
            f.create_dataset('E_trajectories', data=dataset['E_trajectories'])
            f.create_dataset('t', data=dataset['t'])
            f.create_dataset('x', data=dataset['x'])
            f.create_dataset('v', data=dataset['v'])
            
            # Save parameters
            params_group = f.create_group('parameters')
            for i, params in enumerate(dataset['parameters']):
                param_group = params_group.create_group(f'sample_{i}')
                for key, value in params.items():
                    param_group.attrs[key] = value
        
        print(f"Dataset saved successfully! Size: {os.path.getsize(filepath) / (1024**2):.2f} MB")
    
    @staticmethod
    def load_dataset(filepath):
        """Load dataset from HDF5 file."""
        print(f"Loading dataset from {filepath}...")
        
        dataset = {}
        
        with h5py.File(filepath, 'r') as f:
            dataset['f0'] = f['f0'][:]
            dataset['f_trajectories'] = f['f_trajectories'][:]
            dataset['E_trajectories'] = f['E_trajectories'][:]
            dataset['t'] = f['t'][:]
            dataset['x'] = f['x'][:]
            dataset['v'] = f['v'][:]
            
            # Load parameters
            dataset['parameters'] = []
            params_group = f['parameters']
            for i in range(len(dataset['f0'])):
                param_group = params_group[f'sample_{i}']
                params = {key: param_group.attrs[key] for key in param_group.attrs}
                dataset['parameters'].append(params)
        
        print(f"Dataset loaded! Samples: {len(dataset['f0'])}")
        return dataset


class VPDataset(Dataset):
    """
    PyTorch Dataset for VP-FNO training.
    """
    
    def __init__(self, data_dict, time_indices, normalize=True):
        """
        Args:
            data_dict: Dictionary from VPDataGenerator
            time_indices: List of time indices to use for training
            normalize: Whether to normalize data
        """
        self.f0 = torch.FloatTensor(data_dict['f0'])
        self.f_trajectories = torch.FloatTensor(data_dict['f_trajectories'])
        self.time_indices = time_indices
        
        # Create coordinate grids
        x = data_dict['x']
        v = data_dict['v']
        X, V = np.meshgrid(x, v, indexing='ij')
        
        # Normalize coordinates to [-1, 1]
        if normalize:
            X_norm = 2 * (X - X.min()) / (X.max() - X.min()) - 1
            V_norm = 2 * (V - V.min()) / (V.max() - V.min()) - 1
        else:
            X_norm = X
            V_norm = V
        
        self.X = torch.FloatTensor(X_norm)
        self.V = torch.FloatTensor(V_norm)
        
        # Normalization statistics
        if normalize:
            self.f_mean = self.f0.mean()
            self.f_std = self.f0.std()
        else:
            self.f_mean = 0.0
            self.f_std = 1.0
    
    def __len__(self):
        return len(self.f0) * len(self.time_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_data: [3, Nx, Nv] - (f0, x_grid, v_grid)
            target: [1, Nx, Nv] - f(t)
            time_idx: scalar - time index
        """
        sample_idx = idx // len(self.time_indices)
        time_idx = self.time_indices[idx % len(self.time_indices)]
        
        # Get initial distribution
        f0 = self.f0[sample_idx]
        
        # Get target distribution at time t
        f_t = self.f_trajectories[sample_idx, time_idx]
        
        # Normalize
        f0_norm = (f0 - self.f_mean) / (self.f_std + 1e-8)
        f_t_norm = (f_t - self.f_mean) / (self.f_std + 1e-8)
        
        # Prepare input: (f0, X, V)
        input_data = torch.stack([
            f0_norm,
            self.X,
            self.V
        ], dim=0)
        
        # Target
        target = f_t_norm.unsqueeze(0)
        
        return input_data, target, time_idx


def create_dataloaders(config: TrainingConfig, physics_config: PhysicsConfig,
                       dataset_path=None):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Training configuration
        physics_config: Physics configuration
        dataset_path: Path to existing dataset (if None, generate new)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load or generate dataset
    if dataset_path and os.path.exists(dataset_path):
        data_dict = VPDataGenerator.load_dataset(dataset_path)
    else:
        generator = VPDataGenerator(physics_config)
        total_samples = config.n_train + config.n_val + config.n_test
        data_dict = generator.generate_dataset(
            n_samples=total_samples,
            vary_parameters=True,
            save_path=dataset_path
        )
    
    # Split data
    n_train = config.n_train
    n_val = config.n_val
    
    train_dict = {key: value[:n_train] if isinstance(value, np.ndarray) and value.shape[0] == len(data_dict['f0'])
                  else value for key, value in data_dict.items()}
    
    val_dict = {key: value[n_train:n_train+n_val] if isinstance(value, np.ndarray) and value.shape[0] == len(data_dict['f0'])
                else value for key, value in data_dict.items()}
    
    test_dict = {key: value[n_train+n_val:] if isinstance(value, np.ndarray) and value.shape[0] == len(data_dict['f0'])
                 else value for key, value in data_dict.items()}
    
    # Create datasets
    train_dataset = VPDataset(train_dict, config.predict_steps, normalize=True)
    val_dataset = VPDataset(val_dict, config.predict_steps, normalize=True)
    test_dataset = VPDataset(test_dict, config.predict_steps, normalize=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config import get_default_config, get_fast_test_config
    
    print("Testing data generation...")
    
    # Use fast config for testing
    config = get_fast_test_config()
    
    # Create generator
    generator = VPDataGenerator(config.physics)
    
    # Test single solution
    print("\nTesting single VP solution:")
    f0 = generator.initialize_two_stream()
    print(f"Initial distribution shape: {f0.shape}")
    print(f"Mass: {np.trapz(np.trapz(f0, generator.v, axis=1), generator.x):.6f}")
    
    # Solve for a few time steps
    t_eval = np.linspace(0, 2, 5)
    solution = generator.solve_vlasov(f0, t_eval)
    print(f"Solution f shape: {solution['f'].shape}")
    print(f"Electric field shape: {solution['E'].shape}")
    
    # Test dataset generation
    print("\nTesting dataset generation:")
    dataset = generator.generate_dataset(n_samples=5, vary_parameters=True)
    print(f"f0 shape: {dataset['f0'].shape}")
    print(f"f_trajectories shape: {dataset['f_trajectories'].shape}")
    
    # Test PyTorch dataset
    print("\nTesting PyTorch Dataset:")
    vp_dataset = VPDataset(dataset, time_indices=[0, 2, 4])
    print(f"Dataset length: {len(vp_dataset)}")
    
    input_data, target, time_idx = vp_dataset[0]
    print(f"Input shape: {input_data.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Time index: {time_idx}")
    
    print("\nData generation test completed!")
