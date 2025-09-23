"""
Mesh generation utilities for AI4CFD.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def generate_uniform_mesh_1d(
    x_min: float = 0.0,
    x_max: float = 1.0,
    num_points: int = 100
) -> torch.Tensor:
    """
    Generate uniform 1D mesh.
    
    Args:
        x_min: Minimum x coordinate
        x_max: Maximum x coordinate
        num_points: Number of mesh points
        
    Returns:
        Mesh coordinates (num_points, 1)
    """
    x = torch.linspace(x_min, x_max, num_points)
    return x.unsqueeze(1)


def generate_uniform_mesh_2d(
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    num_x: int = 50,
    num_y: int = 50
) -> torch.Tensor:
    """
    Generate uniform 2D mesh.
    
    Args:
        x_min: Minimum x coordinate
        x_max: Maximum x coordinate
        y_min: Minimum y coordinate
        y_max: Maximum y coordinate
        num_x: Number of points in x direction
        num_y: Number of points in y direction
        
    Returns:
        Mesh coordinates (num_x * num_y, 2)
    """
    x = torch.linspace(x_min, x_max, num_x)
    y = torch.linspace(y_min, y_max, num_y)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    mesh = torch.stack([X.flatten(), Y.flatten()], dim=1)
    return mesh


def generate_random_points_2d(
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    num_points: int = 1000,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate random points in 2D domain.
    
    Args:
        x_min: Minimum x coordinate
        x_max: Maximum x coordinate
        y_min: Minimum y coordinate
        y_max: Maximum y coordinate
        num_points: Number of random points
        seed: Random seed
        
    Returns:
        Random coordinates (num_points, 2)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    x = torch.rand(num_points) * (x_max - x_min) + x_min
    y = torch.rand(num_points) * (y_max - y_min) + y_min
    
    return torch.stack([x, y], dim=1)


def generate_boundary_points_2d(
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    num_points_per_edge: int = 100
) -> torch.Tensor:
    """
    Generate boundary points for 2D rectangular domain.
    
    Args:
        x_min: Minimum x coordinate
        x_max: Maximum x coordinate
        y_min: Minimum y coordinate
        y_max: Maximum y coordinate
        num_points_per_edge: Number of points per boundary edge
        
    Returns:
        Boundary coordinates (4 * num_points_per_edge, 2)
    """
    # Bottom edge
    x_bottom = torch.linspace(x_min, x_max, num_points_per_edge)
    y_bottom = torch.full_like(x_bottom, y_min)
    bottom = torch.stack([x_bottom, y_bottom], dim=1)
    
    # Top edge
    x_top = torch.linspace(x_min, x_max, num_points_per_edge)
    y_top = torch.full_like(x_top, y_max)
    top = torch.stack([x_top, y_top], dim=1)
    
    # Left edge
    y_left = torch.linspace(y_min, y_max, num_points_per_edge)
    x_left = torch.full_like(y_left, x_min)
    left = torch.stack([x_left, y_left], dim=1)
    
    # Right edge
    y_right = torch.linspace(y_min, y_max, num_points_per_edge)
    x_right = torch.full_like(y_right, x_max)
    right = torch.stack([x_right, y_right], dim=1)
    
    return torch.cat([bottom, top, left, right], dim=0)


def generate_circle_mesh(
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 1.0,
    num_radial: int = 20,
    num_angular: int = 50,
    include_center: bool = True
) -> torch.Tensor:
    """
    Generate circular mesh in polar coordinates.
    
    Args:
        center: Circle center (x, y)
        radius: Circle radius
        num_radial: Number of points in radial direction
        num_angular: Number of points in angular direction
        include_center: Whether to include center point
        
    Returns:
        Mesh coordinates (num_points, 2)
    """
    if include_center:
        r = torch.linspace(0, radius, num_radial)
    else:
        r = torch.linspace(radius / num_radial, radius, num_radial)
    
    theta = torch.linspace(0, 2 * np.pi, num_angular, endpoint=False)
    R, Theta = torch.meshgrid(r, theta, indexing='ij')
    
    # Convert to Cartesian coordinates
    x = R * torch.cos(Theta) + center[0]
    y = R * torch.sin(Theta) + center[1]
    
    mesh = torch.stack([x.flatten(), y.flatten()], dim=1)
    
    if include_center:
        # Remove duplicate center points (keep only one)
        center_mask = (mesh[:, 0] == center[0]) & (mesh[:, 1] == center[1])
        center_indices = torch.where(center_mask)[0]
        if len(center_indices) > 1:
            # Keep first center point, remove others
            remove_indices = center_indices[1:]
            keep_mask = torch.ones(len(mesh), dtype=torch.bool)
            keep_mask[remove_indices] = False
            mesh = mesh[keep_mask]
    
    return mesh


def generate_time_mesh(
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_time_points: int = 100
) -> torch.Tensor:
    """
    Generate time mesh for time-dependent problems.
    
    Args:
        t_min: Start time
        t_max: End time
        num_time_points: Number of time points
        
    Returns:
        Time coordinates (num_time_points, 1)
    """
    t = torch.linspace(t_min, t_max, num_time_points)
    return t.unsqueeze(1)


def generate_spacetime_mesh(
    x_min: float = 0.0,
    x_max: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_x: int = 50,
    num_t: int = 50
) -> torch.Tensor:
    """
    Generate space-time mesh for 1D time-dependent problems.
    
    Args:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Start time
        t_max: End time
        num_x: Number of spatial points
        num_t: Number of time points
        
    Returns:
        Space-time coordinates (num_x * num_t, 2)
    """
    x = torch.linspace(x_min, x_max, num_x)
    t = torch.linspace(t_min, t_max, num_t)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    mesh = torch.stack([X.flatten(), T.flatten()], dim=1)
    return mesh


def adaptive_mesh_refinement(
    points: torch.Tensor,
    solution: torch.Tensor,
    threshold: float = 0.1,
    max_refinement_levels: int = 3
) -> torch.Tensor:
    """
    Simple adaptive mesh refinement based on solution gradient.
    
    Args:
        points: Current mesh points (num_points, spatial_dim)
        solution: Solution values at mesh points (num_points, 1)
        threshold: Refinement threshold
        max_refinement_levels: Maximum refinement levels
        
    Returns:
        Refined mesh points
    """
    # This is a simplified implementation
    # In practice, more sophisticated AMR algorithms would be used
    
    refined_points = points.clone()
    
    for level in range(max_refinement_levels):
        # Compute approximate gradient magnitude
        if len(refined_points) < 3:
            break
            
        # Simple gradient estimation using finite differences
        grad_magnitude = torch.zeros(len(refined_points))
        
        for i in range(1, len(refined_points) - 1):
            grad_magnitude[i] = torch.abs(solution[i+1] - solution[i-1]).item()
        
        # Find points that need refinement
        refine_mask = grad_magnitude > threshold
        refine_indices = torch.where(refine_mask)[0]
        
        if len(refine_indices) == 0:
            break
        
        # Add midpoints for refinement
        new_points = []
        for idx in refine_indices:
            if idx < len(refined_points) - 1:
                midpoint = (refined_points[idx] + refined_points[idx + 1]) / 2
                new_points.append(midpoint)
        
        if new_points:
            new_points = torch.stack(new_points)
            refined_points = torch.cat([refined_points, new_points], dim=0)
    
    return refined_points