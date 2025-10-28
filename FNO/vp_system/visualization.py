"""
Visualization tools for VP-FNO results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import torch
import os


class VPVisualizer:
    """
    Visualization tools for Vlasov-Poisson results.
    """
    
    def __init__(self, x, v, figsize=(15, 10)):
        """
        Args:
            x: Spatial grid
            v: Velocity grid
            figsize: Figure size
        """
        self.x = x
        self.v = v
        self.X, self.V = np.meshgrid(x, v, indexing='ij')
        self.figsize = figsize
    
    def plot_phase_space(self, f, title="Distribution Function", 
                        save_path=None, vmin=None, vmax=None):
        """
        Plot phase space distribution.
        
        Args:
            f: Distribution function [Nx, Nv]
            title: Plot title
            save_path: Path to save figure
            vmin, vmax: Color scale limits
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.pcolormesh(self.X, self.V, f, shading='auto',
                          cmap='viridis', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel('Position $x$', fontsize=14)
        ax.set_ylabel('Velocity $v$', fontsize=14)
        ax.set_title(title, fontsize=16)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('$f(x,v)$', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved phase space plot to {save_path}")
        
        plt.show()
    
    def plot_comparison(self, f_true, f_pred, time_idx, 
                       save_path=None):
        """
        Plot comparison between true and predicted distributions.
        
        Args:
            f_true: True distribution [Nx, Nv]
            f_pred: Predicted distribution [Nx, Nv]
            time_idx: Time index
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 4, figure=fig, wspace=0.3)
        
        # True distribution
        ax1 = fig.add_subplot(gs[0])
        im1 = ax1.pcolormesh(self.X, self.V, f_true, shading='auto', cmap='viridis')
        ax1.set_title(f'True $f(x,v,t)$ at t={time_idx}', fontsize=14)
        ax1.set_xlabel('Position $x$', fontsize=12)
        ax1.set_ylabel('Velocity $v$', fontsize=12)
        plt.colorbar(im1, ax=ax1)
        
        # Predicted distribution
        ax2 = fig.add_subplot(gs[1])
        im2 = ax2.pcolormesh(self.X, self.V, f_pred, shading='auto', cmap='viridis',
                            vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
        ax2.set_title(f'Predicted $f(x,v,t)$', fontsize=14)
        ax2.set_xlabel('Position $x$', fontsize=12)
        ax2.set_ylabel('Velocity $v$', fontsize=12)
        plt.colorbar(im2, ax=ax2)
        
        # Error
        error = np.abs(f_true - f_pred)
        ax3 = fig.add_subplot(gs[2])
        im3 = ax3.pcolormesh(self.X, self.V, error, shading='auto', cmap='hot')
        ax3.set_title(f'Absolute Error', fontsize=14)
        ax3.set_xlabel('Position $x$', fontsize=12)
        ax3.set_ylabel('Velocity $v$', fontsize=12)
        plt.colorbar(im3, ax=ax3)
        
        # 1D profiles (integrated over v)
        ax4 = fig.add_subplot(gs[3])
        rho_true = np.trapz(f_true, self.v, axis=1)
        rho_pred = np.trapz(f_pred, self.v, axis=1)
        ax4.plot(self.x, rho_true, 'b-', label='True', linewidth=2)
        ax4.plot(self.x, rho_pred, 'r--', label='Predicted', linewidth=2)
        ax4.set_xlabel('Position $x$', fontsize=12)
        ax4.set_ylabel('Density $\\rho(x)$', fontsize=12)
        ax4.set_title('Spatial Density Profile', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Compute and display errors
        l2_error = np.linalg.norm(f_true - f_pred) / np.linalg.norm(f_true)
        max_error = np.max(error)
        fig.suptitle(f'L2 Error: {l2_error:.6f}, Max Error: {max_error:.6f}', 
                    fontsize=14, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.show()
    
    def plot_multi_time_comparison(self, f_true_list, f_pred_list, time_indices,
                                   save_path=None):
        """
        Plot comparisons at multiple time steps.
        
        Args:
            f_true_list: List of true distributions
            f_pred_list: List of predicted distributions
            time_indices: List of time indices
            save_path: Path to save figure
        """
        n_times = len(time_indices)
        fig, axes = plt.subplots(2, n_times, figsize=(5*n_times, 8))
        
        if n_times == 1:
            axes = axes.reshape(2, 1)
        
        for i, (f_true, f_pred, t_idx) in enumerate(zip(f_true_list, f_pred_list, time_indices)):
            # True distribution
            im1 = axes[0, i].pcolormesh(self.X, self.V, f_true, shading='auto', cmap='viridis')
            axes[0, i].set_title(f'True at t={t_idx}', fontsize=12)
            axes[0, i].set_xlabel('$x$')
            axes[0, i].set_ylabel('$v$')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Predicted distribution
            im2 = axes[1, i].pcolormesh(self.X, self.V, f_pred, shading='auto', cmap='viridis',
                                        vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
            axes[1, i].set_title(f'Predicted at t={t_idx}', fontsize=12)
            axes[1, i].set_xlabel('$x$')
            axes[1, i].set_ylabel('$v$')
            plt.colorbar(im2, ax=axes[1, i])
            
            # Compute error
            l2_error = np.linalg.norm(f_true - f_pred) / np.linalg.norm(f_true)
            axes[1, i].text(0.5, 0.95, f'L2: {l2_error:.4f}',
                           transform=axes[1, i].transAxes,
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved multi-time comparison to {save_path}")
        
        plt.show()
    
    def plot_electric_field(self, E, t_values, save_path=None):
        """
        Plot electric field evolution.
        
        Args:
            E: Electric field [Nt, Nx]
            t_values: Time values
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Spacetime plot
        T, X = np.meshgrid(t_values, self.x, indexing='ij')
        im1 = ax1.pcolormesh(X, T, E, shading='auto', cmap='RdBu_r')
        ax1.set_xlabel('Position $x$', fontsize=14)
        ax1.set_ylabel('Time $t$', fontsize=14)
        ax1.set_title('Electric Field $E(x,t)$', fontsize=16)
        plt.colorbar(im1, ax=ax1, label='$E$')
        
        # Energy evolution
        E_energy = 0.5 * np.sum(E**2, axis=1) * (self.x[1] - self.x[0])
        ax2.semilogy(t_values, E_energy, 'b-', linewidth=2)
        ax2.set_xlabel('Time $t$', fontsize=14)
        ax2.set_ylabel('Electric Field Energy', fontsize=14)
        ax2.set_title('Energy Evolution', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # Fit growth rate in exponential phase
        if len(t_values) > 10:
            idx_start = len(t_values) // 4
            idx_end = len(t_values) // 2
            log_E = np.log(E_energy[idx_start:idx_end])
            t_fit = t_values[idx_start:idx_end]
            gamma = np.polyfit(t_fit, log_E, 1)[0]
            
            # Plot fit
            ax2.semilogy(t_fit, np.exp(gamma * t_fit + log_E[0]), 'r--',
                        label=f'$\\gamma$ = {gamma:.4f}', linewidth=2)
            ax2.legend(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved electric field plot to {save_path}")
        
        plt.show()
    
    def create_animation(self, f_trajectory, save_path='vp_animation.mp4',
                        fps=10, title_prefix="VP Evolution"):
        """
        Create animation of phase space evolution.
        
        Args:
            f_trajectory: Distribution trajectory [Nt, Nx, Nv]
            save_path: Path to save animation
            fps: Frames per second
            title_prefix: Prefix for title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Determine color scale from all frames
        vmin = f_trajectory.min()
        vmax = f_trajectory.max()
        
        # Initial frame
        im = ax.pcolormesh(self.X, self.V, f_trajectory[0], 
                          shading='auto', cmap='viridis',
                          vmin=vmin, vmax=vmax)
        ax.set_xlabel('Position $x$', fontsize=14)
        ax.set_ylabel('Velocity $v$', fontsize=14)
        title = ax.set_title(f'{title_prefix} - Frame 0', fontsize=16)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('$f(x,v)$', fontsize=14)
        
        def update(frame):
            im.set_array(f_trajectory[frame].ravel())
            title.set_text(f'{title_prefix} - Frame {frame}')
            return im, title
        
        anim = animation.FuncAnimation(fig, update, frames=len(f_trajectory),
                                      interval=1000/fps, blit=False)
        
        # Save animation
        print(f"Creating animation with {len(f_trajectory)} frames...")
        anim.save(save_path, writer='pillow', fps=fps, dpi=100)
        print(f"Animation saved to {save_path}")
        
        plt.close()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].set_title('Training and Validation Loss', fontsize=16)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Learning rate (if available)
        if 'lr' in history:
            ax_lr = axes[0].twinx()
            ax_lr.plot(epochs, history['lr'], 'g--', label='Learning Rate', alpha=0.7)
            ax_lr.set_ylabel('Learning Rate', fontsize=14, color='g')
            ax_lr.tick_params(axis='y', labelcolor='g')
        
        # Relative L2 error (if available)
        if 'val_l2_error' in history:
            axes[1].plot(epochs, history['val_l2_error'], 'b-', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=14)
            axes[1].set_ylabel('Relative L2 Error', fontsize=14)
            axes[1].set_title('Validation Relative L2 Error', fontsize=16)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history to {save_path}")
        
        plt.show()
    
    def plot_conservation_properties(self, f_trajectory, t_values, dv, save_path=None):
        """
        Plot conservation properties (mass and energy).
        
        Args:
            f_trajectory: Distribution trajectory [Nt, Nx, Nv]
            t_values: Time values
            dv: Velocity grid spacing
            save_path: Path to save figure
        """
        # Compute mass and energy
        mass = np.array([np.trapz(np.trapz(f, self.v, axis=1), self.x) 
                        for f in f_trajectory])
        
        # Kinetic energy
        v2 = self.v ** 2 / 2.0
        energy = np.array([np.trapz(np.trapz(f * v2, self.v, axis=1), self.x)
                          for f in f_trajectory])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Mass conservation
        ax1.plot(t_values, mass, 'b-', linewidth=2)
        ax1.axhline(mass[0], color='r', linestyle='--', label='Initial Mass')
        ax1.set_xlabel('Time $t$', fontsize=14)
        ax1.set_ylabel('Total Mass', fontsize=14)
        ax1.set_title('Mass Conservation', fontsize=16)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Energy evolution
        ax2.plot(t_values, energy, 'b-', linewidth=2, label='Kinetic Energy')
        ax2.set_xlabel('Time $t$', fontsize=14)
        ax2.set_ylabel('Energy', fontsize=14)
        ax2.set_title('Energy Evolution', fontsize=16)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Compute relative changes
        mass_change = np.abs(mass - mass[0]) / mass[0]
        energy_change = np.abs(energy - energy[0]) / energy[0]
        
        fig.suptitle(f'Max Mass Change: {mass_change.max():.2e}, Max Energy Change: {energy_change.max():.2e}',
                    fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved conservation properties to {save_path}")
        
        plt.show()


def visualize_model_predictions(model, test_loader, device, x, v, 
                                n_samples=3, save_dir='./results'):
    """
    Visualize model predictions on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device (cpu/cuda)
        x, v: Spatial and velocity grids
        n_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    visualizer = VPVisualizer(x, v)
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets, time_idx) in enumerate(test_loader):
            if i >= n_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Predict
            outputs = model(inputs)
            
            # Convert to numpy
            f_true = targets[0, 0].cpu().numpy()
            f_pred = outputs[0, 0].cpu().numpy()
            t_idx = time_idx[0].item()
            
            # Plot comparison
            save_path = os.path.join(save_dir, f'comparison_sample_{i}_t_{t_idx}.png')
            visualizer.plot_comparison(f_true, f_pred, t_idx, save_path=save_path)
    
    print(f"Visualizations saved to {save_dir}")


if __name__ == "__main__":
    # Test visualization tools
    print("Testing visualization tools...")
    
    # Create dummy data
    Nx, Nv = 64, 64
    x = np.linspace(0, 4*np.pi, Nx)
    v = np.linspace(-6, 6, Nv)
    X, V = np.meshgrid(x, v, indexing='ij')
    
    # Create dummy distribution
    f = np.exp(-((V-2)**2 + (V+2)**2) / 2) * (1 + 0.1*np.cos(0.5*X))
    
    # Test visualizer
    visualizer = VPVisualizer(x, v)
    
    print("\nPlotting phase space...")
    visualizer.plot_phase_space(f, title="Test Distribution")
    
    # Test comparison plot
    f_pred = f + np.random.randn(*f.shape) * 0.01
    print("\nPlotting comparison...")
    visualizer.plot_comparison(f, f_pred, time_idx=10)
    
    print("\nVisualization tests completed!")
