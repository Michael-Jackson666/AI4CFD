"""
AI4CFD Utility Package
======================

Comprehensive utilities for AI-based Computational Fluid Dynamics.
Includes data generation, neural network building blocks, training utilities, 
evaluation metrics, and visualization tools.

Supports:
- PINNs (Physics-Informed Neural Networks)
- DeepONet (Deep Operator Network)
- FNO (Fourier Neural Operator)
- TNN (Tensor Neural Network)
- Transformer-based methods

Usage:
------
>>> from utils import MLP, PINN, FNO2d, DeepONet
>>> from utils import create_pinn_trainer, train_model
>>> from utils import generate_burgers_data, plot_2d_solution
"""

# ==============================================================================
# Data Utilities
# ==============================================================================
from .data_utils import (
    # Basic utilities
    numpy_to_torch,
    torch_to_numpy,
    get_device,
    normalize_data,
    denormalize_data,
    create_mesh_grid,
    
    # PDE data generation
    generate_1d_poisson_data,
    generate_2d_poisson_data,
    generate_heat_equation_data,
    generate_burgers_data,
    generate_navier_stokes_data,
    generate_wave_equation_data,
    
    # Boundary/Initial conditions
    create_boundary_conditions,
    create_initial_conditions,
    
    # Operator learning data
    generate_operator_data,
    generate_parametric_pde_data,
    generate_fno_data,
    
    # DataLoaders
    PDEDataset,
    DeepONetDataset,
    create_training_dataloader,
    create_fno_dataloader,
    create_deeponet_dataloader,
)

# ==============================================================================
# Neural Network Building Blocks
# ==============================================================================
from .nn_blocks import (
    # Basic layers
    MLP,
    FourierFeatures,
    ModifiedMLP,
    ResidualBlock,
    ResMLP,
    
    # PINNs
    PINN,
    AdaptiveWeightPINN,
    
    # DeepONet
    DeepONet,
    StackedDeepONet,
    
    # FNO
    SpectralConv1d,
    SpectralConv2d,
    FNO1d,
    FNO2d,
    
    # TNN
    TensorLayer,
    TNN,
    TuckerTNN,
    
    # Transformer
    PositionalEncoding,
    PDETransformer,
    SpatioTemporalTransformer,
    
    # Utilities
    count_parameters,
    save_model,
    load_model,
)

# ==============================================================================
# Training Utilities
# ==============================================================================
from .training import (
    # Loss functions
    PINNLoss,
    WeightedMSELoss,
    RelativeMSELoss,
    SobolevLoss,
    SpectralLoss,
    
    # PDE residuals
    compute_pde_residual,
    compute_derivative,
    compute_laplacian,
    compute_gradient,
    compute_divergence,
    
    # Trainers
    BaseTrainer,
    PINNTrainer,
    DeepONetTrainer,
    FNOTrainer,
    TNNTrainer,
    
    # Optimizer utilities
    get_optimizer,
    get_scheduler,
    WarmupCosineScheduler,
    
    # Training utilities
    train_with_lbfgs,
    adaptive_sampling,
    gradient_clipping,
    EarlyStopping,
    GradientBalancer,
)

# ==============================================================================
# Plotting Utilities
# ==============================================================================
from .plotting import (
    setup_plotting_style,
    plot_1d_solution,
    plot_2d_solution,
    plot_2d_comparison,
    plot_training_history,
    plot_burgers_evolution,
    plot_residuals,
    save_animation_frames,
)

# ==============================================================================
# Evaluation Metrics
# ==============================================================================
from .metrics import (
    mse_loss,
    mae_loss,
    relative_l2_error,
    relative_linf_error,
    pointwise_error_statistics,
    evaluate_model_performance,
    physics_residual_l2,
    conservation_error,
    energy_error,
    compute_derivatives,
)

# ==============================================================================
# Quick Access Functions
# ==============================================================================

def create_pinn(input_dim, output_dim=1, hidden_dims=[64, 64, 64], 
                activation='tanh', use_fourier=False, use_adaptive_weights=False):
    """
    Quickly create a PINN model.
    
    Args:
        input_dim: Number of input dimensions (e.g., 2 for (x, t))
        output_dim: Number of output dimensions
        hidden_dims: List of hidden layer sizes
        activation: Activation function name
        use_fourier: Whether to use Fourier feature encoding
        use_adaptive_weights: Whether to use adaptive loss weighting
    
    Returns:
        PINN model
    
    Example:
        >>> model = create_pinn(2, 1, hidden_dims=[64, 64, 64])
    """
    if use_adaptive_weights:
        return AdaptiveWeightPINN(input_dim, output_dim, hidden_dims, activation)
    return PINN(input_dim, output_dim, hidden_dims, activation)


def create_deeponet(branch_input_dim, trunk_input_dim, hidden_dim=100, 
                    p=100, branch_layers=[100, 100], trunk_layers=[100, 100]):
    """
    Quickly create a DeepONet model.
    
    Args:
        branch_input_dim: Dimension of branch network input (sensor points)
        trunk_input_dim: Dimension of trunk network input (query points)
        hidden_dim: Width of hidden layers
        p: Dimension of output from branch and trunk networks
        branch_layers: List of hidden layer sizes for branch network
        trunk_layers: List of hidden layer sizes for trunk network
    
    Returns:
        DeepONet model
    
    Example:
        >>> model = create_deeponet(100, 1, hidden_dim=128, p=50)
    """
    return DeepONet(branch_input_dim, trunk_input_dim, 
                    branch_layers=branch_layers, trunk_layers=trunk_layers, p=p)


def create_fno(in_channels=1, out_channels=1, modes=12, width=32, 
               dim=2, depth=4, mlp_ratio=4):
    """
    Quickly create an FNO model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        modes: Number of Fourier modes
        width: Hidden dimension width
        dim: Spatial dimension (1 or 2)
        depth: Number of Fourier layers
        mlp_ratio: MLP expansion ratio
    
    Returns:
        FNO model
    
    Example:
        >>> model = create_fno(modes=16, width=64, dim=2)
    """
    if dim == 1:
        return FNO1d(in_channels, out_channels, modes, width, depth)
    elif dim == 2:
        return FNO2d(in_channels, out_channels, modes, modes, width, depth)
    else:
        raise ValueError(f"FNO dimension {dim} not supported (use 1 or 2)")


def create_tnn(input_dim, output_dim=1, rank=10, layers_per_dim=2, 
               hidden_dim=32, use_tucker=False):
    """
    Quickly create a TNN model.
    
    Args:
        input_dim: Number of input dimensions
        output_dim: Number of output dimensions
        rank: Tensor decomposition rank
        layers_per_dim: Number of layers per dimension
        hidden_dim: Hidden dimension size
        use_tucker: Whether to use Tucker decomposition
    
    Returns:
        TNN model
    
    Example:
        >>> model = create_tnn(input_dim=3, rank=20)
    """
    if use_tucker:
        return TuckerTNN(input_dim, output_dim, rank, layers_per_dim, hidden_dim)
    return TNN(input_dim, output_dim, rank, layers_per_dim, hidden_dim)


def create_pde_transformer(input_dim, output_dim=1, d_model=64, nhead=4, 
                            num_layers=4, dim_feedforward=256):
    """
    Quickly create a PDE Transformer model.
    
    Args:
        input_dim: Number of input dimensions
        output_dim: Number of output dimensions
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward network dimension
    
    Returns:
        PDETransformer model
    
    Example:
        >>> model = create_pde_transformer(2, 1, d_model=128)
    """
    return PDETransformer(input_dim, output_dim, d_model, nhead, 
                          num_layers, dim_feedforward)


def train_model(model, train_data, epochs=1000, lr=1e-3, method='pinn',
                pde_loss_fn=None, bc_data=None, ic_data=None, device='cpu',
                verbose=True, save_path=None):
    """
    Universal training function for all AI4CFD models.
    
    Args:
        model: Neural network model
        train_data: Training data (format depends on method)
        epochs: Number of training epochs
        lr: Learning rate
        method: Training method ('pinn', 'deeponet', 'fno', 'tnn')
        pde_loss_fn: PDE residual loss function (for PINNs)
        bc_data: Boundary condition data
        ic_data: Initial condition data
        device: Training device
        verbose: Print training progress
        save_path: Path to save trained model
    
    Returns:
        Trained model and training history
    
    Example:
        >>> model, history = train_model(pinn, train_data, method='pinn')
    """
    import torch.optim as optim
    
    model = model.to(device)
    
    if method == 'pinn':
        trainer = PINNTrainer(model, pde_loss_fn=pde_loss_fn, 
                              bc_data=bc_data, ic_data=ic_data, device=device)
        history = trainer.train(train_data, epochs=epochs, lr=lr, verbose=verbose)
    
    elif method == 'deeponet':
        trainer = DeepONetTrainer(model, device=device)
        history = trainer.train(train_data, epochs=epochs, lr=lr, verbose=verbose)
    
    elif method == 'fno':
        trainer = FNOTrainer(model, device=device)
        history = trainer.train(train_data, epochs=epochs, lr=lr, verbose=verbose)
    
    elif method == 'tnn':
        trainer = TNNTrainer(model, device=device)
        history = trainer.train(train_data, epochs=epochs, lr=lr, verbose=verbose)
    
    else:
        raise ValueError(f"Unknown training method: {method}")
    
    if save_path:
        save_model(model, save_path)
    
    return model, history


# ==============================================================================
# Exports
# ==============================================================================
__all__ = [
    # Data utilities
    'numpy_to_torch',
    'torch_to_numpy',
    'get_device',
    'normalize_data',
    'denormalize_data',
    'create_mesh_grid',
    'generate_1d_poisson_data',
    'generate_2d_poisson_data',
    'generate_heat_equation_data',
    'generate_burgers_data',
    'generate_navier_stokes_data',
    'generate_wave_equation_data',
    'create_boundary_conditions',
    'create_initial_conditions',
    'generate_operator_data',
    'generate_parametric_pde_data',
    'generate_fno_data',
    'PDEDataset',
    'DeepONetDataset',
    'create_training_dataloader',
    'create_fno_dataloader',
    'create_deeponet_dataloader',
    
    # Neural network building blocks
    'MLP',
    'FourierFeatures',
    'ModifiedMLP',
    'ResidualBlock',
    'ResMLP',
    'PINN',
    'AdaptiveWeightPINN',
    'DeepONet',
    'StackedDeepONet',
    'SpectralConv1d',
    'SpectralConv2d',
    'FNO1d',
    'FNO2d',
    'TensorLayer',
    'TNN',
    'TuckerTNN',
    'PositionalEncoding',
    'PDETransformer',
    'SpatioTemporalTransformer',
    'count_parameters',
    'save_model',
    'load_model',
    
    # Training utilities
    'PINNLoss',
    'WeightedMSELoss',
    'RelativeMSELoss',
    'SobolevLoss',
    'SpectralLoss',
    'compute_pde_residual',
    'compute_derivative',
    'compute_laplacian',
    'compute_gradient',
    'compute_divergence',
    'BaseTrainer',
    'PINNTrainer',
    'DeepONetTrainer',
    'FNOTrainer',
    'TNNTrainer',
    'get_optimizer',
    'get_scheduler',
    'WarmupCosineScheduler',
    'train_with_lbfgs',
    'adaptive_sampling',
    'gradient_clipping',
    'EarlyStopping',
    'GradientBalancer',
    
    # Plotting
    'setup_plotting_style',
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
    'evaluate_model_performance',
    'physics_residual_l2',
    'conservation_error',
    'energy_error',
    'compute_derivatives',
    
    # Quick access functions
    'create_pinn',
    'create_deeponet',
    'create_fno',
    'create_tnn',
    'create_pde_transformer',
    'train_model',
]

# Version info
__version__ = '2.0.0'
__author__ = 'AI4CFD Team'