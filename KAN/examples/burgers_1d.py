"""
1D Burgers Equation Solver using KAN (Kolmogorov-Arnold Networks)

This example demonstrates how to use KAN to solve the 1D viscous Burgers equation:
    ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²,  x ∈ [-1, 1], t ∈ [0, T]

with:
    - Initial condition: u(x, 0) = -sin(πx)
    - Boundary conditions: u(-1, t) = u(1, t) = 0
    - ν = 0.01/π (kinematic viscosity)

This is a nonlinear PDE that models shock wave formation and dissipation.

Author: AI4CFD Project
Date: 2026-01
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import KANPDE


def prepare_training_data(n_collocation=3000, n_initial=200, n_boundary=200, T=1.0):
    """
    Prepare training data for the Burgers equation.
    
    Args:
        n_collocation: Number of collocation points in space-time domain
        n_initial: Number of initial condition points at t=0
        n_boundary: Number of boundary condition points at x=-1 and x=1
        T: Final time
    
    Returns:
        Dictionary containing training data points
    """
    # Collocation points (interior domain)
    # Use Latin Hypercube Sampling for better coverage
    x_col = torch.rand(n_collocation, 1, requires_grad=True) * 2 - 1  # [-1, 1]
    t_col = torch.rand(n_collocation, 1, requires_grad=True) * T
    
    # Initial condition points (t = 0)
    x_ic = torch.linspace(-1, 1, n_initial).reshape(-1, 1).requires_grad_(True)
    t_ic = torch.zeros(n_initial, 1, requires_grad=True)
    u_ic = -torch.sin(np.pi * x_ic)  # u(x, 0) = -sin(πx)
    
    # Boundary condition points
    n_bc_per_side = n_boundary // 2
    
    # Left boundary (x = -1)
    x_bc_left = -torch.ones(n_bc_per_side, 1, requires_grad=True)
    t_bc_left = torch.linspace(0, T, n_bc_per_side).reshape(-1, 1).requires_grad_(True)
    u_bc_left = torch.zeros(n_bc_per_side, 1)
    
    # Right boundary (x = 1)
    x_bc_right = torch.ones(n_bc_per_side, 1, requires_grad=True)
    t_bc_right = torch.linspace(0, T, n_bc_per_side).reshape(-1, 1).requires_grad_(True)
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


def compute_pde_loss(model, data, nu=0.01/np.pi, lambda_pde=1.0, lambda_ic=20.0, 
                     lambda_bc=20.0, lambda_reg=1e-5):
    """
    Compute the total loss for the Burgers equation.
    
    Args:
        model: KANPDE model
        data: Dictionary containing training data
        nu: Kinematic viscosity coefficient
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
    
    # Burgers equation PDE residual: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0
    pde_residual = u_t + u_pred * u_x - nu * u_xx
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


def train_kan(model, data, epochs=15000, lr=1e-3, nu=0.01/np.pi, print_every=1000):
    """
    Train the KAN model to solve the Burgers equation.
    
    Args:
        model: KANPDE model
        data: Training data dictionary
        epochs: Number of training epochs
        lr: Learning rate
        nu: Kinematic viscosity coefficient
        print_every: Print frequency
    
    Returns:
        Training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Use learning rate scheduler with multiple milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5000, 10000, 12000], gamma=0.5
    )
    
    history = {'total': [], 'pde': [], 'ic': [], 'bc': [], 'reg': []}
    
    print("=" * 70)
    print("Training KAN for 1D Burgers Equation".center(70))
    print("=" * 70)
    print(f"Model parameters: {model.count_parameters()}")
    print(f"Training epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Kinematic viscosity (ν): {nu:.6f}")
    print("=" * 70)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, loss_dict = compute_pde_loss(model, data, nu=nu)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Record history
        for key in history.keys():
            history[key].append(loss_dict[key])
        
        # Print progress
        if (epoch + 1) % print_every == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:5d} | LR: {current_lr:.2e} | "
                  f"Total: {loss_dict['total']:.6f} | "
                  f"PDE: {loss_dict['pde']:.6f} | IC: {loss_dict['ic']:.6f} | "
                  f"BC: {loss_dict['bc']:.6f} | Reg: {loss_dict['reg']:.6f}")
    
    print("=" * 70)
    print("Training completed!".center(70))
    print("=" * 70)
    
    return history


def evaluate_and_plot(model, nu=0.01/np.pi, T=1.0, n_x=300, n_t=200):
    """
    Evaluate the trained model and create comprehensive visualizations.
    
    Args:
        model: Trained KANPDE model
        nu: Kinematic viscosity coefficient
        T: Final time
        n_x: Number of spatial points for evaluation
        n_t: Number of temporal points for evaluation
    """
    # Create evaluation grid
    x_eval = np.linspace(-1, 1, n_x)
    t_eval = np.linspace(0, T, n_t)
    X, T_grid = np.meshgrid(x_eval, t_eval)
    
    # Flatten for model evaluation
    x_flat = torch.FloatTensor(X.flatten()).reshape(-1, 1)
    t_flat = torch.FloatTensor(T_grid.flatten()).reshape(-1, 1)
    xt_flat = torch.cat([x_flat, t_flat], dim=1)
    
    # Model prediction
    with torch.no_grad():
        u_pred = model(xt_flat).numpy().reshape(n_t, n_x)
    
    # Compute some basic statistics
    print("\n" + "=" * 70)
    print("Solution Analysis".center(70))
    print("=" * 70)
    print(f"Solution range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")
    print(f"Initial min/max: [{u_pred[0].min():.6f}, {u_pred[0].max():.6f}]")
    print(f"Final min/max:   [{u_pred[-1].min():.6f}, {u_pred[-1].max():.6f}]")
    
    # Check shock formation (large gradients)
    u_x_grid = np.gradient(u_pred, x_eval, axis=1)
    max_gradient = np.max(np.abs(u_x_grid))
    print(f"Maximum spatial gradient: {max_gradient:.6f}")
    print("=" * 70)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Surface plot - Full solution
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, T_grid, u_pred, cmap=cm.coolwarm, alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x, t)')
    ax1.set_title('KAN Solution: Burgers Equation')
    ax1.view_init(elev=25, azim=45)
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # 2. Contour plot
    ax2 = fig.add_subplot(2, 3, 2)
    contour = ax2.contourf(X, T_grid, u_pred, levels=30, cmap='coolwarm')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Solution Contour (Shock Wave Formation)')
    fig.colorbar(contour, ax=ax2)
    
    # Add contour lines
    ax2.contour(X, T_grid, u_pred, levels=10, colors='k', alpha=0.3, linewidths=0.5)
    
    # 3. Time snapshots
    ax3 = fig.add_subplot(2, 3, 3)
    time_snapshots = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_snapshots)))
    
    for i, t_snap in enumerate(time_snapshots):
        idx = int(t_snap / T * (n_t - 1))
        ax3.plot(x_eval, u_pred[idx, :], label=f't={t_snap:.1f}', 
                color=colors[i], linewidth=2)
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('u(x, t)')
    ax3.set_title('Solution Evolution (Shock Development)')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 4. Initial vs Final comparison
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(x_eval, u_pred[0, :], 'b-', linewidth=2, label='Initial (t=0)')
    ax4.plot(x_eval, u_pred[-1, :], 'r-', linewidth=2, label=f'Final (t={T})')
    ax4.fill_between(x_eval, u_pred[0, :], u_pred[-1, :], alpha=0.2, color='gray')
    ax4.set_xlabel('x')
    ax4.set_ylabel('u(x, t)')
    ax4.set_title('Initial vs Final State')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 5. Spatial gradient magnitude (shock indicator)
    ax5 = fig.add_subplot(2, 3, 5)
    gradient_mag = np.abs(u_x_grid)
    contour_grad = ax5.contourf(X, T_grid, gradient_mag, levels=20, cmap='hot')
    ax5.set_xlabel('x')
    ax5.set_ylabel('t')
    ax5.set_title('Spatial Gradient Magnitude |∂u/∂x|')
    fig.colorbar(contour_grad, ax=ax5)
    
    # 6. Time evolution at specific locations
    ax6 = fig.add_subplot(2, 3, 6)
    x_locations = [-0.5, 0.0, 0.5]
    for x_loc in x_locations:
        idx_x = np.argmin(np.abs(x_eval - x_loc))
        ax6.plot(t_eval, u_pred[:, idx_x], label=f'x={x_loc:.1f}', linewidth=2)
    
    ax6.set_xlabel('t')
    ax6.set_ylabel('u(x, t)')
    ax6.set_title('Time Evolution at Fixed Locations')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('burgers_1d_kan_results.png', dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: burgers_1d_kan_results.png")
    plt.show()


def plot_training_history(history):
    """
    Plot detailed training history.
    
    Args:
        history: Dictionary containing training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].semilogy(history['total'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Individual losses
    axes[0, 1].semilogy(history['pde'], label='PDE Residual', alpha=0.8, linewidth=2)
    axes[0, 1].semilogy(history['ic'], label='Initial Condition', alpha=0.8, linewidth=2)
    axes[0, 1].semilogy(history['bc'], label='Boundary Condition', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss (log scale)')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Regularization loss
    axes[1, 0].semilogy(history['reg'], 'g-', linewidth=2, label='Regularization')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (log scale)')
    axes[1, 0].set_title('Regularization Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Loss ratio analysis
    total_arr = np.array(history['total'])
    pde_arr = np.array(history['pde'])
    ic_arr = np.array(history['ic'])
    bc_arr = np.array(history['bc'])
    
    axes[1, 1].plot(pde_arr / (total_arr + 1e-10), label='PDE/Total', alpha=0.8)
    axes[1, 1].plot(ic_arr / (total_arr + 1e-10), label='IC/Total', alpha=0.8)
    axes[1, 1].plot(bc_arr / (total_arr + 1e-10), label='BC/Total', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].set_title('Loss Contribution Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('burgers_1d_training_history.png', dpi=300, bbox_inches='tight')
    print(f"Training history saved to: burgers_1d_training_history.png")
    plt.show()


def main():
    """Main function to run the complete example."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Problem parameters
    nu = 0.01 / np.pi  # Kinematic viscosity
    T = 1.0            # Final time
    
    # Prepare training data
    print("\n" + "=" * 70)
    print("Preparing Training Data".center(70))
    print("=" * 70)
    data = prepare_training_data(n_collocation=3000, n_initial=200, n_boundary=200, T=T)
    print(f"Collocation points: {len(data['x_col'])}")
    print(f"Initial condition points: {len(data['x_ic'])}")
    print(f"Boundary condition points: {len(data['x_bc'])}")
    print("=" * 70)
    
    # Create KAN model with deeper architecture for nonlinear problem
    # Input: (x, t) -> 2D, Output: u(x, t) -> 1D
    model = KANPDE(
        input_dim=2,
        hidden_dims=[30, 30, 30, 30],  # 4 hidden layers with 30 neurons each
        output_dim=1,
        grid_size=7,      # Finer grid for better approximation
        spline_order=3
    )
    
    print(f"\nModel Architecture:")
    print(f"  - Input dimension: 2 (x, t)")
    print(f"  - Hidden layers: {[30, 30, 30, 30]}")
    print(f"  - Output dimension: 1 (u)")
    print(f"  - Total parameters: {model.count_parameters()}")
    print(f"  - Grid size: 7")
    print(f"  - Spline order: 3")
    
    # Train the model
    history = train_kan(model, data, epochs=15000, lr=1e-3, nu=nu, print_every=1000)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate and visualize results
    evaluate_and_plot(model, nu=nu, T=T)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'nu': nu,
        'T': T
    }, 'burgers_1d_kan_model.pth')
    print(f"\nModel saved to: burgers_1d_kan_model.pth")


if __name__ == "__main__":
    main()
