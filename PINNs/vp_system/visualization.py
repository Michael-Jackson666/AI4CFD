"""
Visualization module for PINN results.
Handles plotting of phase space distributions, densities, and training history.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


@torch.no_grad()
def plot_results(pinn_solver, epoch):
    """
    Generates and saves plots of the current simulation state.
    
    Args:
        pinn_solver (VlasovPoissonPINN): The PINN solver instance
        epoch (int): Current epoch number
    """
    pinn_solver.model.eval()
    print(f"Generating plots for epoch {epoch}...")
    
    # Create spatial and velocity grids
    x_grid = torch.linspace(
        pinn_solver.domain['x'][0], 
        pinn_solver.domain['x'][1], 
        100, 
        device=pinn_solver.device
    )
    v_grid = torch.linspace(
        pinn_solver.domain['v'][0], 
        pinn_solver.domain['v'][1], 
        100, 
        device=pinn_solver.device
    )
    X, V = torch.meshgrid(x_grid, v_grid, indexing='ij')
    
    # Select time snapshots to plot
    plot_times = [0, 25, 50]
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)

    # ==================== Top Row: Phase Space Evolution ====================
    for i, t_val in enumerate(plot_times):
        T = torch.full_like(X, t_val)
        
        # Normalize inputs before passing to model
        T_norm = pinn_solver._normalize_t(T.flatten()).unsqueeze(1)
        X_norm = pinn_solver._normalize_x(X.flatten()).unsqueeze(1)
        V_norm = pinn_solver._normalize_v(V.flatten()).unsqueeze(1)
        
        # Get prediction
        f_pred = pinn_solver.model(torch.cat([T_norm, X_norm, V_norm], dim=1)).reshape(X.shape)
        
        # Plot phase space
        ax = fig.add_subplot(gs[0, i])
        im = ax.pcolormesh(X.cpu(), V.cpu(), f_pred.cpu(), cmap='jet', shading='auto')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('x (position)')
        ax.set_ylabel('v (velocity)')
        ax.set_title(rf'PINN Solution f(t,x,v) at t={t_val:.2f} $\omega_p^{{-1}}$')
    
    # ==================== Bottom Row: Diagnostics ====================
    
    # Plot 1: True Initial Condition
    ax_ic = fig.add_subplot(gs[1, 0])
    f_ic_true = pinn_solver._initial_condition(X, V)
    im_ic = ax_ic.pcolormesh(X.cpu(), V.cpu(), f_ic_true.cpu(), cmap='jet', shading='auto')
    fig.colorbar(im_ic, ax=ax_ic)
    ax_ic.set_xlabel('x (position)')
    ax_ic.set_ylabel('v (velocity)')
    ax_ic.set_title('True Initial Condition f(0,x,v)')

    # Plot 2: Final Electron Density
    ax_ne = fig.add_subplot(gs[1, 1])
    t_final = torch.full((x_grid.shape[0], 1), pinn_solver.domain['t'][1], device=pinn_solver.device)
    n_e_final = pinn_solver._compute_ne(t_final, x_grid.unsqueeze(1))
    
    ax_ne.plot(x_grid.cpu(), n_e_final.cpu(), 'b-', linewidth=2, label='Electron Density')
    ax_ne.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Background Density')
    ax_ne.legend()
    ax_ne.grid(True, alpha=0.3)
    ax_ne.set_title(f'Electron Density n_e(t,x) at t={pinn_solver.domain["t"][1]:.2f}')
    ax_ne.set_xlabel('x (position)')
    ax_ne.set_ylabel('n_e')

    # Plot 3: Final Electric Field
    ax_e = fig.add_subplot(gs[1, 2])
    charge_dev_final = n_e_final - 1.0
    dx_final = x_grid[1] - x_grid[0]
    E_final = torch.cumsum(charge_dev_final, dim=0) * dx_final
    E_final -= torch.mean(E_final)
    
    ax_e.plot(x_grid.cpu(), E_final.cpu(), 'g-', linewidth=2)
    ax_e.grid(True, alpha=0.3)
    ax_e.set_title(f'Electric Field E(t,x) at t={pinn_solver.domain["t"][1]:.2f}')
    ax_e.set_xlabel('x (position)')
    ax_e.set_ylabel('E (Electric Field)')

    # Save figure
    plt.tight_layout()
    save_path = os.path.join(pinn_solver.config['plot_dir'], f'results_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_loss_history(pinn_solver):
    """
    Plots the training loss history from log file.
    
    Args:
        pinn_solver (VlasovPoissonPINN): The PINN solver instance
    """
    print("Plotting loss history...")
    try:
        # Load training log
        log_data = np.loadtxt(pinn_solver.log_file_path, delimiter=',', skiprows=1)
        
        if len(log_data) == 0:
            print("  Warning: No data to plot")
            return
        
        # Create figure
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        # Plot all loss components
        ax.plot(log_data[:, 0], log_data[:, 1], 'k-', linewidth=2, label='Total Loss')
        ax.plot(log_data[:, 0], log_data[:, 2], 'r--', alpha=0.7, linewidth=1.5, label='PDE Loss')
        ax.plot(log_data[:, 0], log_data[:, 3], 'g--', alpha=0.7, linewidth=1.5, label='IC Loss')
        ax.plot(log_data[:, 0], log_data[:, 4], 'b--', alpha=0.7, linewidth=1.5, label='BC Loss')
        
        # Format plot
        ax.set_yscale('log')
        ax.set_title('Training Loss History', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (log scale)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        save_path = os.path.join(pinn_solver.config['plot_dir'], 'loss_history.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Loss history saved: {save_path}")
        
    except Exception as e:
        print(f"  Could not plot loss history: {e}")


def plot_phase_space_animation(pinn_solver, num_frames=50):
    """
    Creates an animation of phase space evolution (placeholder for future implementation).
    
    Args:
        pinn_solver (VlasovPoissonPINN): The PINN solver instance
        num_frames (int): Number of frames in animation
    """
    print("Animation generation not yet implemented.")
    pass
