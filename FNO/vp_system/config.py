"""
Configuration file for FNO Vlasov-Poisson System.
All hyperparameters and settings are defined here.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class PhysicsConfig:
    """Physical parameters for Vlasov-Poisson system."""
    
    # Spatial domain
    x_min: float = 0.0
    x_max: float = 4 * np.pi
    Nx: int = 64
    
    # Velocity domain
    v_min: float = -6.0
    v_max: float = 6.0
    Nv: int = 64
    
    # Time domain
    t_start: float = 0.0
    t_end: float = 10.0
    Nt: int = 100
    dt: float = 0.1
    
    # Two-stream instability parameters
    vth: float = 1.0      # Thermal velocity
    v0: float = 2.0       # Beam velocity
    amplitude: float = 0.1  # Perturbation amplitude
    k_mode: float = 0.5   # Wavenumber of perturbation
    
    @property
    def dx(self) -> float:
        """Spatial grid spacing."""
        return (self.x_max - self.x_min) / self.Nx
    
    @property
    def dv(self) -> float:
        """Velocity grid spacing."""
        return (self.v_max - self.v_min) / self.Nv
    
    def get_grids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spatial, velocity, and time grids."""
        x = np.linspace(self.x_min, self.x_max, self.Nx, endpoint=False)
        v = np.linspace(self.v_min, self.v_max, self.Nv)
        t = np.linspace(self.t_start, self.t_end, self.Nt)
        return x, v, t


@dataclass
class FNOConfig:
    """FNO model configuration."""
    
    # Model architecture
    modes_x: int = 16      # Fourier modes in x direction
    modes_v: int = 16      # Fourier modes in v direction
    width: int = 64        # Hidden channel width
    n_layers: int = 4      # Number of Fourier layers
    
    # Input/Output
    input_channels: int = 1   # Initial distribution f(x,v,0)
    output_channels: int = 1  # Predicted distribution f(x,v,t)
    
    # Model type
    model_type: str = "standard"  # "standard", "unet", "adaptive", "factorized"
    
    # Activation function
    activation: str = "gelu"  # "gelu", "relu", "tanh"
    
    # Factorization (for factorized FNO)
    rank_ratio: float = 0.5
    
    # Adaptive modes (for adaptive FNO)
    max_modes_x: int = 32
    max_modes_v: int = 32


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    
    # Architecture
    d_model: int = 256         # Model dimension
    nhead: int = 8             # Number of attention heads
    num_layers: int = 6        # Number of transformer layers
    dim_feedforward: int = 1024  # FFN dimension
    dropout: float = 0.1       # Dropout rate
    
    # Positional encoding
    max_len: int = 10000       # Maximum sequence length
    
    # Input/Output
    input_dim: int = 4096      # Flattened phase space (64x64)
    output_dim: int = 4096     # Same as input
    
    # Patch-based processing
    use_patches: bool = True
    patch_size: int = 8        # Patch size for spatial dimension


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Dataset
    n_train: int = 800         # Number of training samples
    n_val: int = 100           # Number of validation samples
    n_test: int = 100          # Number of test samples
    
    # Time steps for prediction
    predict_steps: List[int] = None  # Which time steps to predict
    
    def __post_init__(self):
        if self.predict_steps is None:
            # Predict at multiple time steps
            self.predict_steps = [20, 40, 60, 80, 100]
    
    # Training hyperparameters
    batch_size: int = 16
    epochs: int = 500
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # "cosine", "step", "exponential", "none"
    lr_decay_rate: float = 0.5
    lr_decay_steps: int = 100
    
    # Loss weights
    data_loss_weight: float = 1.0
    physics_loss_weight: float = 0.1  # Weight for physics-informed loss
    
    # Physics-informed components
    use_physics_loss: bool = True
    use_conservation_loss: bool = True
    mass_conservation_weight: float = 1.0
    energy_conservation_weight: float = 0.1
    
    # Optimizer
    optimizer_type: str = "adam"  # "adam", "adamw", "sgd"
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # Early stopping
    patience: int = 50
    min_delta: float = 1e-6
    
    # Checkpointing
    save_every: int = 50
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10
    use_tensorboard: bool = True
    tensorboard_dir: str = "./runs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class Config:
    """Complete configuration combining all sub-configs."""
    
    physics: PhysicsConfig = None
    fno: FNOConfig = None
    transformer: TransformerConfig = None
    training: TrainingConfig = None
    
    # Experiment settings
    experiment_name: str = "vp_fno"
    seed: int = 42
    
    # Model selection
    model_name: str = "fno"  # "fno", "transformer", "hybrid"
    
    def __post_init__(self):
        if self.physics is None:
            self.physics = PhysicsConfig()
        if self.fno is None:
            self.fno = FNOConfig()
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    def set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def print_config(self):
        """Print configuration summary."""
        print("=" * 80)
        print(f"Experiment: {self.experiment_name}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.training.device}")
        print("=" * 80)
        
        print("\nPhysics Configuration:")
        print(f"  Spatial: [{self.physics.x_min}, {self.physics.x_max}], Nx={self.physics.Nx}")
        print(f"  Velocity: [{self.physics.v_min}, {self.physics.v_max}], Nv={self.physics.Nv}")
        print(f"  Time: [{self.physics.t_start}, {self.physics.t_end}], Nt={self.physics.Nt}")
        print(f"  Two-stream: vth={self.physics.vth}, v0={self.physics.v0}, amp={self.physics.amplitude}")
        
        if self.model_name in ["fno", "hybrid"]:
            print("\nFNO Configuration:")
            print(f"  Modes: ({self.fno.modes_x}, {self.fno.modes_v})")
            print(f"  Width: {self.fno.width}, Layers: {self.fno.n_layers}")
            print(f"  Type: {self.fno.model_type}")
        
        if self.model_name in ["transformer", "hybrid"]:
            print("\nTransformer Configuration:")
            print(f"  d_model: {self.transformer.d_model}, heads: {self.transformer.nhead}")
            print(f"  Layers: {self.transformer.num_layers}")
            print(f"  Use patches: {self.transformer.use_patches}")
        
        print("\nTraining Configuration:")
        print(f"  Dataset: train={self.training.n_train}, val={self.training.n_val}, test={self.training.n_test}")
        print(f"  Batch size: {self.training.batch_size}, Epochs: {self.training.epochs}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Physics loss: {self.training.use_physics_loss}")
        print(f"  Conservation loss: {self.training.use_conservation_loss}")
        print("=" * 80)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_fast_test_config() -> Config:
    """Get configuration for fast testing (smaller grid, fewer samples)."""
    physics = PhysicsConfig(
        Nx=32,
        Nv=32,
        Nt=50
    )
    
    fno = FNOConfig(
        modes_x=8,
        modes_v=8,
        width=32,
        n_layers=3
    )
    
    training = TrainingConfig(
        n_train=100,
        n_val=20,
        n_test=20,
        batch_size=8,
        epochs=100
    )
    
    return Config(
        physics=physics,
        fno=fno,
        training=training,
        experiment_name="vp_fno_test"
    )


def get_high_res_config() -> Config:
    """Get configuration for high-resolution simulation."""
    physics = PhysicsConfig(
        Nx=128,
        Nv=128,
        Nt=200
    )
    
    fno = FNOConfig(
        modes_x=32,
        modes_v=32,
        width=128,
        n_layers=6
    )
    
    training = TrainingConfig(
        n_train=1500,
        n_val=200,
        n_test=300,
        batch_size=8,
        epochs=1000
    )
    
    return Config(
        physics=physics,
        fno=fno,
        training=training,
        experiment_name="vp_fno_highres"
    )


if __name__ == "__main__":
    # Test configurations
    print("Testing configuration system...\n")
    
    # Default config
    config = get_default_config()
    config.set_seed()
    config.print_config()
    
    print("\n" + "=" * 80)
    print("Fast Test Configuration:")
    print("=" * 80)
    test_config = get_fast_test_config()
    test_config.print_config()
    
    print("\nConfiguration system test completed!")
