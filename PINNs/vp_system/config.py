"""
Configuration file for Vlasov-Poisson PINN training.
Modify this file to change training parameters without touching the main script.
"""

# ==================== Domain Parameters ====================
DOMAIN = {
    't_max': 40.0,          # Maximum time (paper shows t=0,20,30,40)
    'x_max': 31.416,        # 2π/0.2 ≈ 31.416 (one wavelength based on ω=0.2)
    'v_max': 6.0,           # Maximum velocity (v_0=2.4, need some margin)
}

# ==================== Physics Parameters ====================
PHYSICS = {
    'beam_v': 1.0,          # Beam velocity for two-stream instability
    'thermal_v': 0.5,       # Thermal velocity spread
    'perturb_amp': 0.1,     # Initial perturbation amplitude
}

# ==================== Initial Condition Configuration ====================
# Available initial condition types:
# - 'two_stream': Two-stream instability (symmetric double Maxwellian)
# - 'two_stream_paper': Two-stream from paper eq (6.7) with small perturbation
# - 'landau': Landau damping (single Maxwellian with perturbation)
# - 'single_beam': Single beam (one Maxwellian)
# - 'custom': Custom initial condition (define your own in CUSTOM_IC)

INITIAL_CONDITION = {
    'type': 'two_stream',       # Initial condition type
    
    # Parameters for 'two_stream' mode (standard)
    'beam_v': 1.0,              # Beam velocity (±v_b)
    'thermal_v': 0.5,           # Thermal velocity spread
    'perturb_amp': 0.1,         # Perturbation amplitude (0-1)
    'perturb_mode': 1,          # Perturbation wave number k = 2π*mode/x_max
    
    # Parameters for 'two_stream_paper' mode (paper eq 6.7)
    'paper_v0': 2.4,            # Beam velocity v_0 from paper
    'paper_alpha': 0.003,       # Perturbation amplitude α from paper
    'paper_omega': 0.2,         # Wave number ω from paper
    
    # Parameters for 'landau' mode
    'landau_v_thermal': 1.0,    # Thermal velocity for Landau damping
    'landau_perturb_amp': 0.01, # Small perturbation for Landau damping
    'landau_mode': 1,           # Wave mode number
    
    # Parameters for 'single_beam' mode
    'single_v_center': 0.0,     # Center velocity of the beam
    'single_v_thermal': 0.5,    # Thermal spread
    'single_perturb_amp': 0.05, # Perturbation amplitude
    'single_mode': 1,           # Wave mode
}

# Custom initial condition (optional)
# If IC_TYPE='custom', define your custom function here
# Function signature: f(x, v, config) -> torch.Tensor
# Example:
# def CUSTOM_IC(x, v, config):
#     import torch
#     return torch.exp(-v**2 / 2) / torch.sqrt(torch.tensor(2 * torch.pi))
CUSTOM_IC = None

# ==================== Model Architecture ====================
# Options: 'mlp', 'transformer', 'hybrid_transformer', 'lightweight_transformer'
MODEL_TYPE = 'hybrid_transformer'

# --- MLP Configuration (used when MODEL_TYPE='mlp') ---
MLP_CONFIG = {
    'nn_layers': 8,         # Number of hidden layers
    'nn_neurons': 128,      # Neurons per layer
}

# --- Transformer Configuration (used when MODEL_TYPE contains 'transformer') ---
TRANSFORMER_CONFIG = {
    'd_model': 256,         # Transformer embedding dimension
    'nhead': 8,             # Number of attention heads
    'num_transformer_layers': 4,  # Number of transformer encoder layers
    'dim_feedforward': 512,       # Feedforward network dimension
    'dropout': 0.1,         # Dropout rate
}

# --- Hybrid Transformer Additional Config ---
HYBRID_CONFIG = {
    'num_mlp_layers': 4,    # Number of MLP layers in hybrid model
    'mlp_neurons': 512,     # Neurons per MLP layer in hybrid model
}

# ==================== Training Hyperparameters ====================
TRAINING = {
    'epochs': 10000,         # Total training epochs
    'learning_rate': 1e-4,   # Initial learning rate (decays with scheduler)
    'n_pde': 16000,          # Number of collocation points for PDE
    'n_ic': 1000,            # Number of initial condition points
    'n_bc': 1000,            # Number of boundary condition points
}

# ==================== Loss Function Weights ====================
LOSS_WEIGHTS = {
    'weight_pde': 7.0,       # Weight for governing equations
    'weight_ic': 3.0,        # Weight for initial condition
    'weight_bc': 2.0,       # Weight for boundary conditions
}

# ==================== Numerical & Logging Parameters ====================
NUMERICAL = {
    'v_quad_points': 128,    # Quadrature points for velocity integration
}

LOGGING = {
    'log_frequency': 50,    # Log every N epochs
    'plot_frequency': 1000,  # Plot every N epochs
    'plot_dir': '2025/11/02/2'  # Output directory
}


# ============================================================
# QUICK CONFIGURATION PRESETS
# ============================================================
# Uncomment one of the presets below to quickly switch configurations

def use_ic_preset(preset_name):
    """
    Apply a predefined initial condition preset.
    
    Available IC presets:
    - 'two_stream_strong': Strong two-stream instability (fast growth)
    - 'two_stream_weak': Weak two-stream instability (slow growth)
    - 'landau_damping': Standard Landau damping setup
    - 'single_beam': Single beam with perturbation
    """
    global INITIAL_CONDITION
    
    if preset_name == 'two_stream_strong':
        INITIAL_CONDITION['type'] = 'two_stream'
        INITIAL_CONDITION['beam_v'] = 1.0
        INITIAL_CONDITION['thermal_v'] = 0.3  # Narrow thermal spread
        INITIAL_CONDITION['perturb_amp'] = 0.2  # Large perturbation
        INITIAL_CONDITION['perturb_mode'] = 1
        
    elif preset_name == 'two_stream_weak':
        INITIAL_CONDITION['type'] = 'two_stream'
        INITIAL_CONDITION['beam_v'] = 0.8
        INITIAL_CONDITION['thermal_v'] = 0.5  # Wider thermal spread
        INITIAL_CONDITION['perturb_amp'] = 0.05  # Small perturbation
        INITIAL_CONDITION['perturb_mode'] = 1
        
    elif preset_name == 'landau_damping':
        INITIAL_CONDITION['type'] = 'landau'
        INITIAL_CONDITION['landau_v_thermal'] = 1.0
        INITIAL_CONDITION['landau_perturb_amp'] = 0.01
        INITIAL_CONDITION['landau_mode'] = 1
        
    elif preset_name == 'single_beam':
        INITIAL_CONDITION['type'] = 'single_beam'
        INITIAL_CONDITION['single_v_center'] = 1.5
        INITIAL_CONDITION['single_v_thermal'] = 0.5
        INITIAL_CONDITION['single_perturb_amp'] = 0.1
        INITIAL_CONDITION['single_mode'] = 1
        
    elif preset_name == 'two_stream_paper':
        INITIAL_CONDITION['type'] = 'two_stream_paper'
        INITIAL_CONDITION['paper_v0'] = 2.4
        INITIAL_CONDITION['paper_alpha'] = 0.003
        INITIAL_CONDITION['paper_omega'] = 0.2
        
    else:
        raise ValueError(f"Unknown IC preset: {preset_name}")
    
    print(f"✓ Applied IC preset: {preset_name}")


def use_preset(preset_name):
    """
    Apply a predefined configuration preset.
    
    Available presets:
    - 'standard_mlp': Default MLP configuration (fast training)
    - 'large_mlp': Larger MLP with more capacity
    - 'standard_transformer': Standard transformer configuration
    - 'lightweight_transformer': Smaller transformer (faster)
    - 'hybrid_transformer': Hybrid model (current default)
    """
    global MODEL_TYPE, MLP_CONFIG, TRANSFORMER_CONFIG, HYBRID_CONFIG
    
    if preset_name == 'standard_mlp':
        MODEL_TYPE = 'mlp'
        MLP_CONFIG['nn_layers'] = 8
        MLP_CONFIG['nn_neurons'] = 128
        
    elif preset_name == 'large_mlp':
        MODEL_TYPE = 'mlp'
        MLP_CONFIG['nn_layers'] = 12
        MLP_CONFIG['nn_neurons'] = 256
        
    elif preset_name == 'standard_transformer':
        MODEL_TYPE = 'transformer'
        TRANSFORMER_CONFIG['d_model'] = 256
        TRANSFORMER_CONFIG['nhead'] = 8
        TRANSFORMER_CONFIG['num_transformer_layers'] = 6
        
    elif preset_name == 'lightweight_transformer':
        MODEL_TYPE = 'lightweight_transformer'
        TRANSFORMER_CONFIG['d_model'] = 128
        TRANSFORMER_CONFIG['nhead'] = 4
        TRANSFORMER_CONFIG['num_transformer_layers'] = 3
        
    elif preset_name == 'hybrid_transformer':
        MODEL_TYPE = 'hybrid_transformer'
        TRANSFORMER_CONFIG['d_model'] = 256
        TRANSFORMER_CONFIG['nhead'] = 8
        TRANSFORMER_CONFIG['num_transformer_layers'] = 4
        HYBRID_CONFIG['num_mlp_layers'] = 4
        HYBRID_CONFIG['mlp_neurons'] = 512
        
    else:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    print(f"✓ Applied preset: {preset_name}")


# ============================================================
# Apply presets if needed (uncomment to use)
# ============================================================

# --- Initial Condition Presets ---
# use_ic_preset('two_stream_strong')
# use_ic_preset('two_stream_weak')
# use_ic_preset('two_stream_paper')    # Paper equation (6.7)
# use_ic_preset('landau_damping')
# use_ic_preset('single_beam')

# --- Model Architecture Presets ---
# use_preset('standard_mlp')
# use_preset('large_mlp')
# use_preset('standard_transformer')
# use_preset('lightweight_transformer')
# use_preset('hybrid_transformer')  # This is the current default


# ============================================================
# Build complete configuration dictionary
# ============================================================
def get_configuration():
    """
    Build and return the complete configuration dictionary.
    This function is called by the main script.
    """
    config = {}
    
    # Add all configuration sections
    config.update(DOMAIN)
    config.update(PHYSICS)
    config.update(INITIAL_CONDITION)
    config['custom_ic_function'] = CUSTOM_IC
    config['model_type'] = MODEL_TYPE
    config.update(MLP_CONFIG)
    config.update(TRANSFORMER_CONFIG)
    config.update(HYBRID_CONFIG)
    config.update(TRAINING)
    config.update(LOSS_WEIGHTS)
    config.update(NUMERICAL)
    config.update(LOGGING)
    
    return config


# ============================================================
# Configuration validation
# ============================================================
def validate_configuration():
    """Validate configuration parameters."""
    errors = []
    
    # Check domain parameters
    if DOMAIN['t_max'] <= 0:
        errors.append("t_max must be positive")
    if DOMAIN['x_max'] <= 0:
        errors.append("x_max must be positive")
    if DOMAIN['v_max'] <= 0:
        errors.append("v_max must be positive")
    
    # Check initial condition type
    valid_ic_types = ['two_stream', 'two_stream_paper', 'landau', 'single_beam', 'custom']
    if INITIAL_CONDITION['type'] not in valid_ic_types:
        errors.append(f"Initial condition type must be one of {valid_ic_types}")
    
    # Check custom IC function if needed
    if INITIAL_CONDITION['type'] == 'custom' and CUSTOM_IC is None:
        errors.append("CUSTOM_IC function must be defined when type='custom'")
    
    # Check model type
    valid_models = ['mlp', 'transformer', 'hybrid_transformer', 'lightweight_transformer']
    if MODEL_TYPE not in valid_models:
        errors.append(f"model_type must be one of {valid_models}")
    
    # Check training parameters
    if TRAINING['epochs'] <= 0:
        errors.append("epochs must be positive")
    if TRAINING['learning_rate'] <= 0:
        errors.append("learning_rate must be positive")
    if TRAINING['n_pde'] <= 0:
        errors.append("n_pde must be positive")
    
    # Check loss weights
    if any(w < 0 for w in LOSS_WEIGHTS.values()):
        errors.append("All loss weights must be non-negative")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


if __name__ == '__main__':
    # Test configuration
    print("=" * 70)
    print("CONFIGURATION TEST")
    print("=" * 70)
    
    try:
        validate_configuration()
        config = get_configuration()
        
        print("\n✓ Configuration is valid!")
        print(f"\nModel Type: {config['model_type']}")
        print(f"Initial Condition: {config['type']}")
        print(f"Domain: t∈[0,{config['t_max']}], x∈[0,{config['x_max']}], v∈[±{config['v_max']}]")
        print(f"Training: {config['epochs']} epochs, LR={config['learning_rate']}")
        print(f"Sampling: PDE={config['n_pde']}, IC={config['n_ic']}, BC={config['n_bc']}")
        
    except ValueError as e:
        print(f"\n✗ Configuration error:\n{e}")
