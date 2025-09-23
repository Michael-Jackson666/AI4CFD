"""
Example: Solving 1D Heat Equation using PINNs

This example demonstrates how to use PINNs to solve the 1D heat equation:
du/dt = α * d²u/dx²

with initial condition: u(x, 0) = sin(π*x)
and boundary conditions: u(0, t) = u(1, t) = 0
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ai4cfd.pinns import PINNs, heat_equation_residual
from ai4cfd.utils import generate_collocation_points, train_pinn

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Problem parameters
    alpha = 0.01  # Thermal diffusivity
    
    # Domain bounds: [x_min, x_max, t_min, t_max]
    domain_bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Generate collocation points
    interior_points, boundary_points, initial_points = generate_collocation_points(
        domain_bounds=domain_bounds,
        num_interior=1000,
        num_boundary=200,
        num_initial=100,
        seed=42
    )
    
    # Initial condition: u(x, 0) = sin(π*x)
    x_initial = initial_points[:, 0]
    u_initial = torch.sin(np.pi * x_initial).unsqueeze(1)
    
    # Boundary conditions: u(0, t) = u(1, t) = 0
    boundary_values = torch.zeros(len(boundary_points), 1)
    
    # Create PDE residual function
    def pde_residual(x, u):
        return heat_equation_residual(x, u, alpha=alpha)
    
    # Initialize PINN model
    model = PINNs(
        input_dim=2,  # x and t
        hidden_dim=50,
        output_dim=1,  # u
        num_layers=4,
        activation="tanh"
    )
    
    # Train the model
    print("Training PINN for 1D Heat Equation...")
    trained_model, loss_tracker = train_pinn(
        model=model,
        pde_func=pde_residual,
        interior_points=interior_points,
        boundary_points=boundary_points,
        boundary_values=boundary_values,
        initial_points=initial_points,
        initial_values=u_initial,
        num_epochs=2000,
        lr=1e-3,
        device=device,
        verbose=True
    )
    
    # Plot loss curves
    loss_tracker.plot()
    
    # Generate test points for visualization
    x_test = torch.linspace(0, 1, 100)
    t_test = torch.linspace(0, 1, 100)
    X_test, T_test = torch.meshgrid(x_test, t_test, indexing='ij')
    test_points = torch.stack([X_test.flatten(), T_test.flatten()], dim=1).to(device)
    
    # Predict solution
    trained_model.eval()
    with torch.no_grad():
        u_pred = trained_model(test_points).cpu()
    
    # Reshape for plotting
    U_pred = u_pred.reshape(100, 100)
    
    # Analytical solution for comparison
    def analytical_solution(x, t):
        return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)
    
    U_analytical = analytical_solution(X_test.numpy(), T_test.numpy())
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Predicted solution
    im1 = axes[0].contourf(X_test, T_test, U_pred, levels=50, cmap='viridis')
    axes[0].set_title('PINN Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    plt.colorbar(im1, ax=axes[0])
    
    # Analytical solution
    im2 = axes[1].contourf(X_test, T_test, U_analytical, levels=50, cmap='viridis')
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    error = np.abs(U_pred.numpy() - U_analytical)
    im3 = axes[2].contourf(X_test, T_test, error, levels=50, cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    # Compute L2 relative error
    l2_error = np.linalg.norm(U_pred.numpy() - U_analytical) / np.linalg.norm(U_analytical)
    print(f"L2 relative error: {l2_error:.6f}")

if __name__ == "__main__":
    main()