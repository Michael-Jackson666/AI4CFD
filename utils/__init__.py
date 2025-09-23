"""
Utility functions for AI4CFD repository.
"""

from .data_utils import (
    generate_1d_poisson_data,
    generate_2d_poisson_data,
    generate_burgers_data,
    create_boundary_conditions,
    numpy_to_torch,
    torch_to_numpy,
    create_training_dataloader
)

from .plotting import (
    plot_1d_solution,
    plot_2d_solution,
    plot_2d_comparison,
    plot_training_history,
    plot_burgers_evolution,
    plot_residuals,
    save_animation_frames
)

from .metrics import (
    mse_loss,
    mae_loss,
    relative_l2_error,
    relative_linf_error,
    pointwise_error_statistics,
    evaluate_model_performance
)

__all__ = [
    # Data utilities
    'generate_1d_poisson_data',
    'generate_2d_poisson_data', 
    'generate_burgers_data',
    'create_boundary_conditions',
    'numpy_to_torch',
    'torch_to_numpy',
    'create_training_dataloader',
    
    # Plotting utilities
    'plot_1d_solution',
    'plot_2d_solution',
    'plot_2d_comparison',
    'plot_training_history',
    'plot_burgers_evolution',
    'plot_residuals',
    'save_animation_frames',
    
    # Metrics
    'mse_loss',
    'mae_loss',
    'relative_l2_error',
    'relative_linf_error',
    'pointwise_error_statistics',
    'evaluate_model_performance'
]