"""
Data utilities for PDE solving with neural networks.
Common functions for data generation, loading, and preprocessing.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
import matplotlib.pyplot as plt


def generate_1d_poisson_data(n_points=1000, domain=(-1, 1), source_func=None):
    """
    Generate training data for 1D Poisson equation: d²u/dx² = f(x)
    
    Args:
        n_points: Number of training points
        domain: Tuple of (x_min, x_max)
        source_func: Function f(x), if None uses default
    
    Returns:
        x: Spatial coordinates
        u_exact: Exact solution (if available)
        f: Source term
    """
    x = np.linspace(domain[0], domain[1], n_points)
    
    if source_func is None:
        # Default: f(x) = -π²sin(πx)
        f = -np.pi**2 * np.sin(np.pi * x)
        u_exact = np.sin(np.pi * x)
    else:
        f = source_func(x)
        u_exact = None
    
    return x.reshape(-1, 1), u_exact, f.reshape(-1, 1)


def generate_2d_poisson_data(n_points=50, domain=((-1, 1), (-1, 1))):
    """
    Generate training data for 2D Poisson equation: ∇²u = f(x,y)
    
    Args:
        n_points: Number of points per dimension
        domain: Tuple of ((x_min, x_max), (y_min, y_max))
    
    Returns:
        X: Meshgrid coordinates [n_points², 2]
        u_exact: Exact solution
        f: Source term
    """
    x = np.linspace(domain[0][0], domain[0][1], n_points)
    y = np.linspace(domain[1][0], domain[1][1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Example: f(x,y) = -2π²sin(πx)sin(πy)
    f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Reshape for network input
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    return coords, u_exact.ravel(), f.ravel()


def generate_burgers_data(n_x=256, n_t=100, domain_x=(-1, 1), domain_t=(0, 1), 
                         nu=0.01, initial_condition=None):
    """
    Generate data for Burgers' equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    
    Args:
        n_x: Number of spatial points
        n_t: Number of temporal points
        domain_x: Spatial domain (x_min, x_max)
        domain_t: Temporal domain (t_min, t_max)
        nu: Viscosity parameter
        initial_condition: Function for u(x, 0)
    
    Returns:
        x: Spatial coordinates
        t: Temporal coordinates
        u: Solution field
    """
    x = np.linspace(domain_x[0], domain_x[1], n_x)
    t = np.linspace(domain_t[0], domain_t[1], n_t)
    
    if initial_condition is None:
        # Default: Gaussian initial condition
        u0 = np.exp(-20 * x**2)
    else:
        u0 = initial_condition(x)
    
    # This is a simplified version - in practice, you'd solve numerically
    # or use analytical solutions for specific cases
    u = np.zeros((n_t, n_x))
    u[0] = u0
    
    # Simple finite difference approximation (for demonstration)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    for i in range(1, n_t):
        u_prev = u[i-1]
        # Simple upwind scheme (not stable for all cases)
        u_new = u_prev.copy()
        for j in range(1, n_x-1):
            u_new[j] = (u_prev[j] - dt * u_prev[j] * (u_prev[j] - u_prev[j-1]) / dx 
                       + nu * dt * (u_prev[j+1] - 2*u_prev[j] + u_prev[j-1]) / dx**2)
        u[i] = u_new
    
    return x, t, u


def create_boundary_conditions(coords, domain, bc_type='dirichlet', bc_values=None):
    """
    Create boundary condition data points.
    
    Args:
        coords: Interior coordinates
        domain: Domain specification
        bc_type: Type of boundary condition
        bc_values: Boundary values (function or constant)
    
    Returns:
        bc_coords: Boundary coordinates
        bc_vals: Boundary values
    """
    if len(coords.shape) == 2 and coords.shape[1] == 2:  # 2D case
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        
        # Create boundary points
        n_bc = 100
        x_bc1 = np.full(n_bc, x_min)  # Left boundary
        y_bc1 = np.linspace(y_min, y_max, n_bc)
        
        x_bc2 = np.full(n_bc, x_max)  # Right boundary
        y_bc2 = np.linspace(y_min, y_max, n_bc)
        
        x_bc3 = np.linspace(x_min, x_max, n_bc)  # Bottom boundary
        y_bc3 = np.full(n_bc, y_min)
        
        x_bc4 = np.linspace(x_min, x_max, n_bc)  # Top boundary
        y_bc4 = np.full(n_bc, y_max)
        
        bc_coords = np.vstack([
            np.column_stack([x_bc1, y_bc1]),
            np.column_stack([x_bc2, y_bc2]),
            np.column_stack([x_bc3, y_bc3]),
            np.column_stack([x_bc4, y_bc4])
        ])
        
        if bc_values is None:
            bc_vals = np.zeros(bc_coords.shape[0])
        elif callable(bc_values):
            bc_vals = bc_values(bc_coords[:, 0], bc_coords[:, 1])
        else:
            bc_vals = np.full(bc_coords.shape[0], bc_values)
            
    else:  # 1D case
        x_min, x_max = domain
        bc_coords = np.array([[x_min], [x_max]])
        
        if bc_values is None:
            bc_vals = np.zeros(2)
        elif callable(bc_values):
            bc_vals = bc_values(bc_coords.ravel())
        else:
            bc_vals = np.full(2, bc_values)
    
    return bc_coords, bc_vals


def numpy_to_torch(data, device='cpu'):
    """Convert numpy arrays to PyTorch tensors."""
    if isinstance(data, (list, tuple)):
        return [torch.FloatTensor(arr).to(device) for arr in data]
    else:
        return torch.FloatTensor(data).to(device)


def torch_to_numpy(data):
    """Convert PyTorch tensors to numpy arrays."""
    if isinstance(data, (list, tuple)):
        return [tensor.detach().cpu().numpy() for tensor in data]
    else:
        return data.detach().cpu().numpy()


def create_training_dataloader(X, y, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader for training."""
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)