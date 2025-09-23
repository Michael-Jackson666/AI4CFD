"""
Plotting utilities for visualizing PDE solutions and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import torch


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
    })


def plot_1d_solution(x, u_pred, u_exact=None, title="1D Solution", xlabel="x", ylabel="u"):
    """
    Plot 1D solution comparison.
    
    Args:
        x: Coordinates
        u_pred: Predicted solution
        u_exact: Exact solution (optional)
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(u_pred, torch.Tensor):
        u_pred = u_pred.detach().cpu().numpy()
    
    plt.plot(x.ravel(), u_pred.ravel(), 'b-', linewidth=2, label='Predicted')
    
    if u_exact is not None:
        if isinstance(u_exact, torch.Tensor):
            u_exact = u_exact.detach().cpu().numpy()
        plt.plot(x.ravel(), u_exact.ravel(), 'r--', linewidth=2, label='Exact')
        
        # Compute and display error
        error = np.abs(u_pred.ravel() - u_exact.ravel())
        mse = np.mean(error**2)
        plt.plot(x.ravel(), error, 'g:', linewidth=1, alpha=0.7, label=f'Error (MSE: {mse:.2e})')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_2d_solution(X, Y, u, title="2D Solution", cmap='viridis'):
    """
    Plot 2D solution as contour plot.
    
    Args:
        X, Y: Meshgrid coordinates
        u: Solution values
        title: Plot title
        cmap: Colormap
    """
    plt.figure(figsize=(12, 5))
    
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    
    # Contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, u, levels=20, cmap=cmap)
    plt.colorbar(contour)
    plt.title(f"{title} - Contour")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # 3D surface plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, u, cmap=cmap, alpha=0.9)
    plt.colorbar(surf, shrink=0.5)
    ax.set_title(f"{title} - Surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    
    plt.tight_layout()


def plot_2d_comparison(X, Y, u_pred, u_exact, title="Solution Comparison"):
    """
    Plot comparison between predicted and exact 2D solutions.
    
    Args:
        X, Y: Meshgrid coordinates
        u_pred: Predicted solution
        u_exact: Exact solution
        title: Plot title
    """
    if isinstance(u_pred, torch.Tensor):
        u_pred = u_pred.detach().cpu().numpy()
    if isinstance(u_exact, torch.Tensor):
        u_exact = u_exact.detach().cpu().numpy()
    
    error = np.abs(u_pred - u_exact)
    vmin = min(u_pred.min(), u_exact.min())
    vmax = max(u_pred.max(), u_exact.max())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Predicted solution
    im1 = axes[0].contourf(X, Y, u_pred, levels=20, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[0].set_title("Predicted")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0])
    
    # Exact solution
    im2 = axes[1].contourf(X, Y, u_exact, levels=20, vmin=vmin, vmax=vmax, cmap='viridis')
    axes[1].set_title("Exact")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[1])
    
    # Error
    im3 = axes[2].contourf(X, Y, error, levels=20, cmap='Reds')
    axes[2].set_title(f"Error (MSE: {np.mean(error**2):.2e})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title)
    plt.tight_layout()


def plot_training_history(history, metrics=['loss'], title="Training History"):
    """
    Plot training history.
    
    Args:
        history: Dictionary with metric names as keys and lists of values
        metrics: List of metrics to plot
        title: Plot title
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].semilogy(history[metric], 'b-', linewidth=2)
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f"{metric.capitalize()} vs Epoch")
            axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()


def plot_burgers_evolution(x, t, u, title="Burgers' Equation Evolution"):
    """
    Plot time evolution of Burgers' equation solution.
    
    Args:
        x: Spatial coordinates
        t: Time coordinates
        u: Solution [n_t, n_x]
        title: Plot title
    """
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Contour plot of solution evolution
    T, X = np.meshgrid(t, x)
    contour = ax1.contourf(X, T, u.T, levels=20, cmap='viridis')
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_title("Solution Evolution")
    plt.colorbar(contour, ax=ax1)
    
    # Solution at different times
    time_indices = np.linspace(0, len(t)-1, 5, dtype=int)
    for i in time_indices:
        ax2.plot(x, u[i], label=f't = {t[i]:.2f}', linewidth=2)
    
    ax2.set_xlabel("x")
    ax2.set_ylabel("u")
    ax2.set_title("Solution at Different Times")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()


def plot_residuals(coords, residuals, title="Physics Residuals"):
    """
    Plot physics residuals for PINNs.
    
    Args:
        coords: Coordinate points
        residuals: Residual values
        title: Plot title
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()
    if isinstance(residuals, torch.Tensor):
        residuals = residuals.detach().cpu().numpy()
    
    if coords.shape[1] == 1:  # 1D case
        plt.figure(figsize=(10, 6))
        plt.plot(coords.ravel(), residuals.ravel(), 'ro', alpha=0.6, markersize=4)
        plt.xlabel("x")
        plt.ylabel("Residual")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
    elif coords.shape[1] == 2:  # 2D case
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=residuals.ravel(), 
                            cmap='RdBu', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        
    plt.tight_layout()


def save_animation_frames(X, Y, u_sequence, prefix="frame", save_dir="./"):
    """
    Save sequence of 2D solutions as animation frames.
    
    Args:
        X, Y: Meshgrid coordinates
        u_sequence: Sequence of solutions [n_frames, n_y, n_x]
        prefix: Filename prefix
        save_dir: Directory to save frames
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    vmin = u_sequence.min()
    vmax = u_sequence.max()
    
    for i, u in enumerate(u_sequence):
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, u, levels=20, vmin=vmin, vmax=vmax, cmap='viridis')
        plt.colorbar()
        plt.title(f"Frame {i}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"{save_dir}/{prefix}_{i:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()


# Initialize plotting style when module is imported
setup_plotting_style()