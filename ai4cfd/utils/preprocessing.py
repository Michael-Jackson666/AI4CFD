"""
Data preprocessing utilities for AI4CFD.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataScaler:
    """Data scaling utilities for neural network training."""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize data scaler.
        
        Args:
            method: Scaling method ("standard", "minmax", "none")
        """
        self.method = method
        self.scaler = None
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "none":
            pass
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit scaler and transform data."""
        if self.method == "none":
            return data
        
        data_np = data.detach().cpu().numpy()
        scaled_data = self.scaler.fit_transform(data_np)
        return torch.from_numpy(scaled_data).to(data.device)
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted scaler."""
        if self.method == "none":
            return data
        
        data_np = data.detach().cpu().numpy()
        scaled_data = self.scaler.transform(data_np)
        return torch.from_numpy(scaled_data).to(data.device)
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform scaled data."""
        if self.method == "none":
            return data
        
        data_np = data.detach().cpu().numpy()
        original_data = self.scaler.inverse_transform(data_np)
        return torch.from_numpy(original_data).to(data.device)


def split_data(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Input features
        y: Target values
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        shuffle: Whether to shuffle data
        seed: Random seed
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    num_samples = len(X)
    indices = torch.arange(num_samples)
    
    if shuffle:
        if seed is not None:
            torch.manual_seed(seed)
        indices = indices[torch.randperm(num_samples)]
    
    train_end = int(train_ratio * num_samples)
    val_end = int((train_ratio + val_ratio) * num_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx]
    )


def create_batch_dataset(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create batched dataset.
    
    Args:
        X: Input features
        y: Target values
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        List of (X_batch, y_batch) tuples
    """
    num_samples = len(X)
    indices = torch.arange(num_samples)
    
    if shuffle:
        indices = indices[torch.randperm(num_samples)]
    
    batches = []
    for i in range(0, num_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches


def add_noise(
    data: torch.Tensor,
    noise_level: float = 0.01,
    noise_type: str = "gaussian",
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Add noise to data for robustness testing.
    
    Args:
        data: Input data
        noise_level: Noise amplitude
        noise_type: Type of noise ("gaussian", "uniform")
        seed: Random seed
        
    Returns:
        Noisy data
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if noise_type == "gaussian":
        noise = noise_level * torch.randn_like(data)
    elif noise_type == "uniform":
        noise = noise_level * (2 * torch.rand_like(data) - 1)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return data + noise


def generate_collocation_points(
    domain_bounds: List[Tuple[float, float]],
    num_interior: int = 1000,
    num_boundary: int = 400,
    num_initial: int = 100,
    method: str = "random",
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate collocation points for PINN training.
    
    Args:
        domain_bounds: List of (min, max) for each dimension
        num_interior: Number of interior points
        num_boundary: Number of boundary points
        num_initial: Number of initial condition points
        method: Sampling method ("random", "lhs", "sobol")
        seed: Random seed
        
    Returns:
        Tuple of (interior_points, boundary_points, initial_points)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    ndim = len(domain_bounds)
    
    # Interior points
    if method == "random":
        interior_points = []
        for i in range(ndim):
            low, high = domain_bounds[i]
            coords = torch.rand(num_interior) * (high - low) + low
            interior_points.append(coords)
        interior_points = torch.stack(interior_points, dim=1)
    else:
        # Fallback to random for other methods
        interior_points = []
        for i in range(ndim):
            low, high = domain_bounds[i]
            coords = torch.rand(num_interior) * (high - low) + low
            interior_points.append(coords)
        interior_points = torch.stack(interior_points, dim=1)
    
    # Boundary points (simplified for rectangular domains)
    boundary_points = []
    points_per_boundary = num_boundary // (2 * ndim)
    
    for dim in range(ndim):
        for boundary in [0, 1]:  # min and max boundaries
            point_coords = []
            for i in range(ndim):
                if i == dim:
                    if boundary == 0:
                        coords = torch.full((points_per_boundary,), domain_bounds[i][0])
                    else:
                        coords = torch.full((points_per_boundary,), domain_bounds[i][1])
                else:
                    low, high = domain_bounds[i]
                    coords = torch.rand(points_per_boundary) * (high - low) + low
                point_coords.append(coords)
            boundary_points.append(torch.stack(point_coords, dim=1))
    
    boundary_points = torch.cat(boundary_points, dim=0)
    
    # Initial points (for time-dependent problems)
    if ndim > 1:  # Assume last dimension is time
        initial_points = []
        for i in range(ndim - 1):
            low, high = domain_bounds[i]
            coords = torch.rand(num_initial) * (high - low) + low
            initial_points.append(coords)
        # Set time to initial time
        t_initial = torch.full((num_initial,), domain_bounds[-1][0])
        initial_points.append(t_initial)
        initial_points = torch.stack(initial_points, dim=1)
    else:
        initial_points = torch.empty(0, ndim)
    
    return interior_points, boundary_points, initial_points


def compute_derivatives_finite_diff(
    u: torch.Tensor,
    dx: float,
    order: int = 1,
    accuracy: int = 2
) -> torch.Tensor:
    """
    Compute derivatives using finite differences.
    
    Args:
        u: Function values (spatial_points,)
        dx: Grid spacing
        order: Derivative order
        accuracy: Accuracy order (2, 4, 6)
        
    Returns:
        Derivative values
    """
    if order == 1:
        if accuracy == 2:
            # Central difference
            du = torch.zeros_like(u)
            du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            # Forward/backward difference at boundaries
            du[0] = (u[1] - u[0]) / dx
            du[-1] = (u[-1] - u[-2]) / dx
        elif accuracy == 4:
            # 4th order central difference
            du = torch.zeros_like(u)
            du[2:-2] = (-u[4:] + 8*u[3:-1] - 8*u[1:-3] + u[:-4]) / (12 * dx)
            # Lower order at boundaries
            du[0] = (u[1] - u[0]) / dx
            du[1] = (u[2] - u[0]) / (2 * dx)
            du[-2] = (u[-1] - u[-3]) / (2 * dx)
            du[-1] = (u[-1] - u[-2]) / dx
        else:
            raise ValueError(f"Accuracy {accuracy} not implemented")
    
    elif order == 2:
        if accuracy == 2:
            # Central difference
            d2u = torch.zeros_like(u)
            d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
            # Zero at boundaries (Neumann-like)
            d2u[0] = d2u[1]
            d2u[-1] = d2u[-2]
        else:
            raise ValueError(f"Accuracy {accuracy} not implemented for order {order}")
    
    else:
        raise ValueError(f"Derivative order {order} not implemented")
    
    return du


def l2_relative_error(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute L2 relative error.
    
    Args:
        pred: Predicted values
        true: True values
        
    Returns:
        L2 relative error
    """
    numerator = torch.norm(pred - true)
    denominator = torch.norm(true)
    return (numerator / denominator).item()


def h1_error(
    pred: torch.Tensor,
    true: torch.Tensor,
    pred_grad: torch.Tensor,
    true_grad: torch.Tensor
) -> float:
    """
    Compute H1 error (L2 error + gradient L2 error).
    
    Args:
        pred: Predicted values
        true: True values
        pred_grad: Predicted gradients
        true_grad: True gradients
        
    Returns:
        H1 error
    """
    l2_error = torch.norm(pred - true)
    grad_error = torch.norm(pred_grad - true_grad)
    return torch.sqrt(l2_error**2 + grad_error**2).item()