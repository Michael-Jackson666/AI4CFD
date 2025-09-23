"""
Utility functions for AI4CFD.
"""

from .mesh import (
    generate_uniform_mesh_1d,
    generate_uniform_mesh_2d,
    generate_random_points_2d,
    generate_boundary_points_2d,
    generate_circle_mesh,
    generate_time_mesh,
    generate_spacetime_mesh,
    adaptive_mesh_refinement,
)

from .preprocessing import (
    DataScaler,
    split_data,
    create_batch_dataset,
    add_noise,
    generate_collocation_points,
    compute_derivatives_finite_diff,
    l2_relative_error,
    h1_error,
)

from .training import (
    EarlyStopping,
    LossTracker,
    AdaptiveLossWeighting,
    train_pinn,
    train_deeponet,
)

__all__ = [
    # Mesh utilities
    "generate_uniform_mesh_1d",
    "generate_uniform_mesh_2d",
    "generate_random_points_2d",
    "generate_boundary_points_2d",
    "generate_circle_mesh",
    "generate_time_mesh",
    "generate_spacetime_mesh",
    "adaptive_mesh_refinement",
    
    # Preprocessing utilities
    "DataScaler",
    "split_data",
    "create_batch_dataset",
    "add_noise",
    "generate_collocation_points",
    "compute_derivatives_finite_diff",
    "l2_relative_error",
    "h1_error",
    
    # Training utilities
    "EarlyStopping",
    "LossTracker",
    "AdaptiveLossWeighting",
    "train_pinn",
    "train_deeponet",
]