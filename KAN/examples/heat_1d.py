"""
1D Heat Equation Solver using KAN (Kolmogorov-Arnold Networks)

This example demonstrates how to use KAN to solve the 1D heat equation:
    ∂u/∂t = α * ∂²u/∂x²,  x ∈ [0, 1], t ∈ [0, T]

with:
    - Initial condition: u(x, 0) = sin(πx)
    - Boundary conditions: u(0, t) = u(1, t) = 0
    - Analytical solution: u(x, t) = sin(πx) * exp(-α * π² * t)

Author: AI4CFD Project
Date: 2026-01
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import KANPDE


def analytical_solution(x, t, alpha=0.01):
    """
    Analytical solution of the 1D heat equation.
    
    Args:
        x: Spatial coordinate (can be tensor or numpy array)
        t: Time coordinate (can be tensor or numpy array)
        alpha: Thermal diffusivity coefficient
    
    Returns:
        u(x, t) = sin(πx) * exp(-α * π² * t)
    """
    if isinstance(x, torch.Tensor):
        return torch.sin(np.pi * x) * torch.exp(-alpha * np.pi**2 * t)
    else:
        return np.sin(np.pi * x) * np.exp(-alpha * np.pi**2 * t)


def prepare_training_data(n_collocation=2000, n_initial=100, n_boundary=100, T=1.0):
    """
    Prepare training data for the heat equation.
    
    Args:
        n_collocation: Number of collocation points in space-time domain
        n_initial: Number of initial condition points at t=0
        n_boundary: Number of boundary condition points at x=0 and x=1
        T: Final time
    
    Returns:
        Dictionary containing training data points
    """
    # Collocation points (interior domain)
    x_col = torch.rand(n_collocation, 1, requires_grad=True)
    t_col = torch.rand(n_collocation, 1, requires_grad=True) * T
    
    # Initial condition points (t = 0)
    x_ic = torch.rand(n_initial, 1, requires_grad=True)
    t_ic = torch.zeros(n_initial, 1, requires_grad=True)
    u_ic = torch.sin(np.pi * x_ic)  # u(x, 0) = sin(πx)
    
    # Boundary condition points (x = 0 and x = 1)
    n_bc_per_side = n_boundary // 2
    
    # Left boundary (x = 0)
    x_bc_left = torch.zeros(n_bc_per_side, 1, requires_grad=True)
    t_bc_left = torch.rand(n_bc_per_side, 1, requires_grad=True) * T
    u_bc_left = torch.zeros(n_bc_per_side, 1)
    
    # Right boundary (x = 1)
    x_bc_right = torch.ones(n_bc_per_side, 1, requires_grad=True)
    t_bc_right = torch.rand(n_bc_per_side, 1, requires_grad=True) * T
    u_bc_right = torch.zeros(n_bc_per_side, 1)
    
    # Combine boundary points
    x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
    t_bc = torch.cat([t_bc_left, t_bc_right], dim=0)
    u_bc = torch.cat([u_bc_left, u_bc_right], dim=0)
    
    return {
        'x_col': x_col, 't_col': t_col,
        'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
        'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc
    }


def compute_pde_loss(model, data, alpha=0.01, lambda_pde=1.0, lambda_ic=10.0, 
                     lambda_bc=10.0, lambda_reg=1e-5):
    """
    Compute the total loss for the heat equation.
    
    Args:
        model: KANPDE model
        data: Dictionary containing training data
        alpha: Thermal diffusivity coefficient
        lambda_pde: Weight for PDE residual loss
        lambda_ic: Weight for initial condition loss
        lambda_bc: Weight for boundary condition loss
        lambda_reg: Weight for regularization loss
    
    Returns:
        Total loss and individual loss components
    """
    # Collocation points - PDE residual
    xt_col = torch.cat([data['x_col'], data['t_col']], dim=1)
    u_pred = model(xt_col)
    
    # Compute derivatives
    u_x = torch.autograd.grad(u_pred, data['x_col'], 
                              grad_outputs=torch.ones_like(u_pred),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, data['x_col'],
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, data['t_col'],
                              grad_outputs=torch.ones_like(u_pred),
                              create_graph=True)[0]
    
    # PDE residual: ∂u/∂t - α * ∂²u/∂x² = 0
    pde_residual = u_t - alpha * u_xx
    loss_pde = torch.mean(pde_residual ** 2)
    
    # Initial condition loss
    xt_ic = torch.cat([data['x_ic'], data['t_ic']], dim=1)
    u_pred_ic = model(xt_ic)
    loss_ic = torch.mean((u_pred_ic - data['u_ic']) ** 2)
    
    # Boundary condition loss
    xt_bc = torch.cat([data['x_bc'], data['t_bc']], dim=1)
    u_pred_bc = model(xt_bc)
    loss_bc = torch.mean((u_pred_bc - data['u_bc']) ** 2)
    
    # Regularization loss
    loss_reg = model.regularization_loss()
    
    # Total loss
    total_loss = (lambda_pde * loss_pde + 
                  lambda_ic * loss_ic + 
                  lambda_bc * loss_bc + 
                  lambda_reg * loss_reg)
    
    return total_loss, {
        'pde': loss_pde.item(),
        'ic': loss_ic.item(),
        'bc': loss_bc.item(),
        'reg': loss_reg.item(),
        'total': total_loss.item()
    }


def train_kan(model, data, epochs=10000, lr=1e-3, alpha=0.01, print_every=1000):
    """
    Train the KAN model to solve the heat equation.
    
    Args:
        model: KANPDE model
        data: Training data dictionary
        epochs: Number of training epochs
        lr: Learning rate
        alpha: Thermal diffusivity coefficient
        print_every: Print frequency
    
    Returns:
        Training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000, verbose=True
    )
    
    history = {'total': [], 'pde': [], 'ic': [], 'bc': [], 'reg': []}
    
    print("=" * 70)
    print("Training KAN for 1D Heat Equation".center(70))
    print("=" * 70)
    print(f"Model parameters: {model.count_parameters()}")
    print(f"Training epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Thermal diffusivity (α): {alpha}")
    print("=" * 70)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_dict = compute_pde_loss(model, data, alpha=alpha)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Record history
        for key in history.keys():
            history[key].append(loss_dict[key])
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d} | Total: {loss_dict['total']:.6f} | "
                  f"PDE: {loss_dict['pde']:.6f} | IC: {loss_dict['ic']:.6f} | "
                  f"BC: {loss_dict['bc']:.6f} | Reg: {loss_dict['reg']:.6f}")
    
    print("=" * 70)
    print("Training completed!".center(70))
    print("=" * 70)
    
    return history


def evaluate_and_plot(model, alpha=0.01, T=1.0, n_x=200, n_t=100):
    """
    Evaluate the trained model and create visualizations.
    
    Args:
        model: Trained KANPDE model
        alpha: Thermal diffusivity coefficient
        T: Final time
        n_x: Number of spatial points for evaluation
        n_t: Number of temporal points for evaluation
    """
    # Create evaluation grid
    x_eval = np.linspace(0, 1, n_x)
    t_eval = np.linspace(0, T, n_t)
    X, T_grid = np.meshgrid(x_eval, t_eval)
    
    # Flatten for model evaluation
    x_flat = torch.FloatTensor(X.flatten()).reshape(-1, 1)
    t_flat = torch.FloatTensor(T_grid.flatten()).reshape(-1, 1)
    xt_flat = torch.cat([x_flat, t_flat], dim=1)
    
    # Model prediction
    with torch.no_grad():
        u_pred = model(xt_flat).numpy().reshape(n_t, n_x)
    
    # Analytical solution
    u_exact = analytical_solution(X, T_grid, alpha)
    
    # Compute errors
    error = np.abs(u_pred - u_exact)
    l2_error = np.sqrt(np.mean(error ** 2))
    linf_error = np.max(error)
    relative_l2 = l2_error / np.sqrt(np.mean(u_exact ** 2))
    
    print("\n" + "=" * 70)
    print("Error Analysis".center(70))
    print("=" * 70)
    print(f"L2 Error:         {l2_error:.6e}")
    print(f"L∞ Error:         {linf_error:.6e}")
    print(f"Relative L2:      {relative_l2:.6e} ({relative_l2*100:.4f}%)")
    print("=" * 70)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Surface plot - Predicted solution
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T_grid, u_pred, cmap=cm.viridis, alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x, t)')
    ax1.set_title('KAN Prediction')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. Surface plot - Analytical solution
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, T_grid, u_exact, cmap=cm.viridis, alpha=0.9)
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(x, t)')
    ax2.set_title('Analytical Solution')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. Surface plot - Error
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, T_grid, error, cmap=cm.hot, alpha=0.9)
    ax3.set_xlabel('x')
    ax3.set_ylabel('t')
    ax3.set_zlabel('|Error|')
    ax3.set_title(f'Absolute Error (L2={l2_error:.2e})')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. Contour plot - Predicted solution
    ax4 = fig.add_subplot(2, 3, 4)
    contour1 = ax4.contourf(X, T_grid, u_pred, levels=20, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('t')
    ax4.set_title('KAN Prediction (Contour)')
    fig.colorbar(contour1, ax=ax4)
    
    # 5. Contour plot - Analytical solution
    ax5 = fig.add_subplot(2, 3, 5)
    contour2 = ax5.contourf(X, T_grid, u_exact, levels=20, cmap='viridis')
    ax5.set_xlabel('x')
    ax5.set_ylabel('t')
    ax5.set_title('Analytical Solution (Contour)')
    fig.colorbar(contour2, ax=ax5)
    
    # 6. Snapshots at different times
    ax6 = fig.add_subplot(2, 3, 6)
    time_snapshots = [0, 0.1, 0.3, 0.5, 1.0]
    for t_snap in time_snapshots:
        idx = int(t_snap / T * (n_t - 1))
        ax6.plot(x_eval, u_pred[idx, :], label=f't={t_snap:.1f} (KAN)', linestyle='--')
        ax6.plot(x_eval, u_exact[idx, :], label=f't={t_snap:.1f} (Exact)', alpha=0.7)
    ax6.set_xlabel('x')
    ax6.set_ylabel('u(x, t)')
    ax6.set_title('Solution Snapshots at Different Times')
    ax6.legend(fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heat_1d_kan_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: heat_1d_kan_results.png")
    plt.show()


def main():
    """Main function to run the complete example."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Problem parameters
    alpha = 0.01  # Thermal diffusivity
    T = 1.0       # Final time
    
    # Prepare training data
    print("\n" + "=" * 70)
    print("Preparing Training Data".center(70))
    print("=" * 70)
    data = prepare_training_data(n_collocation=2000, n_initial=100, n_boundary=100, T=T)
    print(f"Collocation points: {len(data['x_col'])}")
    print(f"Initial condition points: {len(data['x_ic'])}")
    print(f"Boundary condition points: {len(data['x_bc'])}")
    print("=" * 70)
    
    # Create KAN model
    # Input: (x, t) -> 2D, Output: u(x, t) -> 1D
    model = KANPDE(
        input_dim=2,
        hidden_dims=[20, 20, 20],  # 3 hidden layers with 20 neurons each
        output_dim=1,
        grid_size=5,
        spline_order=3
    )
    
    # Train the model
    history = train_kan(model, data, epochs=10000, lr=1e-3, alpha=alpha, print_every=1000)
    
    # Evaluate and visualize results
    evaluate_and_plot(model, alpha=alpha, T=T)
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Total loss
    axes[0].semilogy(history['total'], label='Total Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Training Loss History')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Individual losses
    axes[1].semilogy(history['pde'], label='PDE Loss', alpha=0.8)
    axes[1].semilogy(history['ic'], label='IC Loss', alpha=0.8)
    axes[1].semilogy(history['bc'], label='BC Loss', alpha=0.8)
    axes[1].semilogy(history['reg'], label='Reg Loss', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Individual Loss Components')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('heat_1d_training_history.png', dpi=300, bbox_inches='tight')
    print(f"Training history saved to: heat_1d_training_history.png")
    plt.show()


if __name__ == "__main__":
    main()
