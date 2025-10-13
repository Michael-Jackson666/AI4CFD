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
        'thermal_v': 0.02,      # Thermal velocity spread
        'perturb_amp': 0.05,    # Initial perturbation amplitude

        # --- Neural Network Architecture ---
        'nn_layers': 12,        # Number of hidden layers
        'nn_neurons': 256,      # Neurons per layer

        # --- Training Hyperparameters ---
        'epochs': 1000,         # Total training epochs
        'learning_rate': 1e-4,  # Initial learning rate (decays with scheduler)
        'n_pde': 70000,         # Number of collocation points for PDE
        'n_ic': 700,            # Number of initial condition points
        'n_bc': 1100,           # Number of boundary condition points

        # --- Loss Function Weights ---
        'weight_pde': 1.0,      # Weight for governing equations
        'weight_ic': 5.0,       # Weight for initial condition
        'weight_bc': 10.0,      # Weight for boundary conditions

        # --- Numerical & Logging Parameters ---
        'v_quad_points': 128,   # Quadrature points for velocity integration
        'log_frequency': 200,   # Log every N epochs
        'plot_frequency': 200,  # Plot every N epochs
        'plot_dir': 'results_normalized'  # Output directory
    }
    
    # ==================== Print Configuration ====================
    print("=" * 70)
    print("VLASOV-POISSON PINN TRAINING")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Domain: t ∈ [0, {configuration['t_max']}], "
          f"x ∈ [0, {configuration['x_max']}], "
          f"v ∈ [{-configuration['v_max']}, {configuration['v_max']}]")
    print(f"  Network: {configuration['nn_layers']} layers × {configuration['nn_neurons']} neurons")
    print(f"  Training: {configuration['epochs']} epochs, LR={configuration['learning_rate']}")
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