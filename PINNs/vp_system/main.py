"""
Main script for training the PINN on the Vlasov-Poisson system.
Configures hyperparameters and runs the training loop.
"""

import torch
import numpy as np
from vp_pinn import VlasovPoissonPINN


def main():
    """
    Main function to configure and run the PINN training.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ==================== Configuration ====================
    configuration = {
        # --- Domain Parameters ---
        't_max': 62.5,          # Maximum time (in units of ω_p^-1)
        'x_max': 10.0,          # Spatial domain length
        'v_max': 5.0,           # Maximum velocity

        # --- Physics Parameters ---
        'beam_v': 1.0,          # Beam velocity for two-stream instability
        'thermal_v': 0.5,       # Thermal velocity spread
        'perturb_amp': 0.1,     # Initial perturbation amplitude

        # ============================================================
        # MODEL ARCHITECTURE SELECTION
        # ============================================================
        # Options: 'mlp', 'transformer', 'hybrid_transformer', 'lightweight_transformer'
        'model_type': 'mlp',
        
        # --- MLP Configuration (used when model_type='mlp') ---
        'nn_layers': 8,         # Number of hidden layers
        'nn_neurons': 128,      # Neurons per layer
        
        # --- Transformer Configuration (used when model_type contains 'transformer') ---
        'd_model': 256,         # Transformer embedding dimension
        'nhead': 8,             # Number of attention heads
        'num_transformer_layers': 6,  # Number of transformer encoder layers
        'dim_feedforward': 1024,      # Feedforward network dimension
        'dropout': 0.1,         # Dropout rate
        
        # --- Hybrid Transformer Additional Config ---
        'num_mlp_layers': 4,    # Number of MLP layers in hybrid model
        'mlp_neurons': 512,     # Neurons per MLP layer in hybrid model

        # --- Training Hyperparameters ---
        'epochs': 2000,         # Total training epochs
        'learning_rate': 1e-4,  # Initial learning rate (decays with scheduler)
        'n_pde': 70000,         # Number of collocation points for PDE
        'n_ic': 1100,           # Number of initial condition points
        'n_bc': 1100,           # Number of boundary condition points

        # --- Loss Function Weights ---
        'weight_pde': 1.0,      # Weight for governing equations
        'weight_ic': 5.0,       # Weight for initial condition
        'weight_bc': 10.0,      # Weight for boundary conditions

        # --- Numerical & Logging Parameters ---
        'v_quad_points': 128,   # Quadrature points for velocity integration
        'log_frequency': 200,   # Log every N epochs
        'plot_frequency': 400,  # Plot every N epochs
        'plot_dir': '2025/10/13/2'  # Output directory
    }
    
    # ============================================================
    # QUICK CONFIGURATION PRESETS (uncomment to use)
    # ============================================================
    
    # # Preset 1: Standard MLP (default, fast training)
    # configuration['model_type'] = 'mlp'
    # configuration['nn_layers'] = 8
    # configuration['nn_neurons'] = 128
    
    # # Preset 2: Large MLP (more capacity)
    # configuration['model_type'] = 'mlp'
    # configuration['nn_layers'] = 12
    # configuration['nn_neurons'] = 256
    
    # # Preset 3: Standard Transformer (good for complex patterns)
    # configuration['model_type'] = 'transformer'
    # configuration['d_model'] = 256
    # configuration['nhead'] = 8
    # configuration['num_transformer_layers'] = 6
    
    # # Preset 4: Lightweight Transformer (faster, fewer parameters)
    # configuration['model_type'] = 'lightweight_transformer'
    # configuration['d_model'] = 128
    # configuration['nhead'] = 4
    # configuration['num_transformer_layers'] = 3
    
    # # Preset 5: Hybrid Model (combines both approaches)
    # configuration['model_type'] = 'hybrid_transformer'
    # configuration['d_model'] = 256
    # configuration['nhead'] = 8
    # configuration['num_transformer_layers'] = 4
    # configuration['num_mlp_layers'] = 4
    
    # ==================== Print Configuration ====================
    print("=" * 70)
    print("VLASOV-POISSON PINN TRAINING")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Domain: t ∈ [0, {configuration['t_max']}], "
          f"x ∈ [0, {configuration['x_max']}], "
          f"v ∈ [{-configuration['v_max']}, {configuration['v_max']}]")
    
    # Print model-specific info
    model_type = configuration.get('model_type', 'mlp')
    print(f"\n  Model Type: {model_type.upper()}")
    if model_type == 'mlp':
        print(f"  Network: {configuration['nn_layers']} layers × {configuration['nn_neurons']} neurons")
    elif 'transformer' in model_type:
        print(f"  d_model: {configuration.get('d_model', 256)}, "
              f"heads: {configuration.get('nhead', 8)}, "
              f"layers: {configuration.get('num_transformer_layers', 6)}")
        if model_type == 'hybrid_transformer':
            print(f"  MLP branch: {configuration.get('num_mlp_layers', 4)} layers × "
                  f"{configuration.get('mlp_neurons', 512)} neurons")
    
    print(f"\n  Training: {configuration['epochs']} epochs, LR={configuration['learning_rate']}")
    print(f"  Sampling: PDE={configuration['n_pde']}, IC={configuration['n_ic']}, BC={configuration['n_bc']}")
    print(f"  Loss weights: PDE={configuration['weight_pde']}, "
          f"IC={configuration['weight_ic']}, BC={configuration['weight_bc']}")
    print("=" * 70)
    
    # ==================== Initialize and Train ====================
    try:
        # Create PINN solver
        pinn_solver = VlasovPoissonPINN(configuration)
        
        # Run training
        pinn_solver.train()
        
        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print(f"Results saved to: {configuration['plot_dir']}/")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be available in the output directory.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()