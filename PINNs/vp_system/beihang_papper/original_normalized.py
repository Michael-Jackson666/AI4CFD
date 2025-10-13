"""
================================================================================
Physics-Informed Neural Network (PINN) for solving the 1D Vlasov-Poisson system.
This version uses a classic three-component loss function:
1. PDE Loss (Governing Equations)
2. Initial Condition (IC) Loss
3. Boundary Condition (BC) Loss
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os
from torch.utils.tensorboard import SummaryWriter

# Set a global seed for reproducibility
torch.manual_seed(42)

class MLP(nn.Module):
    """Defines the MLP, ensuring f >= 0 with Softplus."""
    def __init__(self, input_dim=3, output_dim=1, layers=12, neurons=512):
        super(MLP, self).__init__()
        modules = [nn.Linear(input_dim, neurons), nn.Tanh()]
        for _ in range(layers - 2):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(neurons, output_dim))
        modules.append(nn.Softplus())
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class VlasovPoissonPINN:
    """A comprehensive PINN solver for the 1D Vlasov-Poisson system."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.domain = {
            't': (0.0, config['t_max']),
            'x': (0.0, config['x_max']),
            'v': (-config['v_max'], config['v_max'])
        }
        
        # Normalization parameters: map domain to [-1, 1]
        self.t_mean = config['t_max'] / 2.0
        self.t_scale = config['t_max'] / 2.0
        self.x_mean = config['x_max'] / 2.0
        self.x_scale = config['x_max'] / 2.0
        self.v_mean = 0.0  # v is already symmetric around 0
        self.v_scale = config['v_max']
        
        print(f"Normalization enabled:")
        print(f"  t: [{self.domain['t'][0]}, {self.domain['t'][1]}] -> [-1, 1]")
        print(f"  x: [{self.domain['x'][0]}, {self.domain['x'][1]}] -> [-1, 1]")
        print(f"  v: [{self.domain['v'][0]}, {self.domain['v'][1]}] -> [-1, 1]")

        self.model = MLP(
            layers=config['nn_layers'], neurons=config['nn_neurons']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999) # Standard betas
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        v_quad = torch.linspace(
            -config['v_max'], config['v_max'],
            config['v_quad_points'], device=self.device
        )
        self.v_quad = v_quad.view(1, 1, -1)

        os.makedirs(self.config['plot_dir'], exist_ok=True)
        self.log_file_path = os.path.join(self.config['plot_dir'], 'training_log.txt')
        self.writer = SummaryWriter(log_dir=self.config['plot_dir'])
    
    def _normalize_t(self, t):
        """Normalize time to [-1, 1]"""
        return (t - self.t_mean) / self.t_scale
    
    def _normalize_x(self, x):
        """Normalize space to [-1, 1]"""
        return (x - self.x_mean) / self.x_scale
    
    def _normalize_v(self, v):
        """Normalize velocity to [-1, 1]"""
        return (v - self.v_mean) / self.v_scale
    
    def _denormalize_t(self, t_norm):
        """Denormalize time from [-1, 1] to original range"""
        return t_norm * self.t_scale + self.t_mean
    
    def _denormalize_x(self, x_norm):
        """Denormalize space from [-1, 1] to original range"""
        return x_norm * self.x_scale + self.x_mean
    
    def _denormalize_v(self, v_norm):
        """Denormalize velocity from [-1, 1] to original range"""
        return v_norm * self.v_scale + self.v_mean

    def _initial_condition(self, x, v):
        """Initial condition: perturbed double Maxwellian."""
        k = 2 * torch.pi / self.config['x_max']
        norm_factor = 1.0 / (self.config['thermal_v'] * torch.sqrt(torch.tensor(2 * torch.pi)))
        term1 = norm_factor * torch.exp(-((v - self.config['beam_v'])**2) / (2 * self.config['thermal_v']**2))
        term2 = norm_factor * torch.exp(-((v + self.config['beam_v'])**2) / (2 * self.config['thermal_v']**2))
        return 0.5 * (term1 + term2) * (1 + self.config['perturb_amp'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """Computes electron density n_e(t,x) by integrating f over v using trapezoidal rule."""
        t_exp = t.unsqueeze(2).expand(-1, -1, self.config['v_quad_points'])
        x_exp = x.unsqueeze(2).expand(-1, -1, self.config['v_quad_points'])
        t_flat, x_flat = t_exp.reshape(-1, 1), x_exp.reshape(-1, 1)
        v_flat = self.v_quad.expand(t.shape[0], -1, -1).reshape(-1, 1)
        
        # Normalize inputs before passing to model
        t_norm = self._normalize_t(t_flat)
        x_norm = self._normalize_x(x_flat)
        v_norm = self._normalize_v(v_flat)
        
        txv_norm = torch.cat([t_norm, x_norm, v_norm], dim=1)
        f_vals = self.model(txv_norm).view(t.shape[0], self.config['v_quad_points'])
        integral = torch.trapezoid(f_vals, self.v_quad.squeeze(), dim=1)
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        """
        Calculates the residuals for the Vlasov and Poisson equations.
        Inputs are in physical domain, will be normalized internally.
        """
        # Normalize inputs to [-1, 1]
        t_norm = self._normalize_t(t)
        x_norm = self._normalize_x(x)
        v_norm = self._normalize_v(v)
        
        txv_norm = torch.cat([t_norm, x_norm, v_norm], dim=1)
        f = self.model(txv_norm)
        
        # Compute gradients w.r.t. normalized coordinates
        df_d_txv_norm = torch.autograd.grad(f, txv_norm, torch.ones_like(f), create_graph=True)[0]
        df_dt_norm, df_dx_norm, df_dv_norm = df_d_txv_norm.split(1, dim=1)
        
        # Transform gradients back to physical domain using chain rule
        # df/dt_physical = df/dt_normalized * dt_normalized/dt_physical = df/dt_norm / t_scale
        df_dt = df_dt_norm / self.t_scale
        df_dx = df_dx_norm / self.x_scale
        df_dv = df_dv_norm / self.v_scale
        
        x_grid_E = torch.linspace(0, self.config['x_max'], 101, device=self.device).unsqueeze(1).requires_grad_()
        t_mean_E = torch.full_like(x_grid_E, t.mean().item())
        n_e_on_grid = self._compute_ne(t_mean_E, x_grid_E)
        charge_dev_on_grid = n_e_on_grid - 1.0
        
        dx_E = x_grid_E[1] - x_grid_E[0]
        E_on_grid = torch.cumsum(charge_dev_on_grid, dim=0) * dx_E
        E_on_grid = E_on_grid - torch.mean(E_on_grid)
        
        # Use torch's interpolation instead of numpy
        x_flat = x.flatten()
        x_grid_flat = x_grid_E.flatten()
        E_grid_flat = E_on_grid.flatten()
        
        # Find indices for interpolation using searchsorted
        indices = torch.searchsorted(x_grid_flat.detach(), x_flat.detach())
        indices = torch.clamp(indices, 1, len(x_grid_flat) - 1)
        
        x0 = x_grid_flat[indices - 1]
        x1 = x_grid_flat[indices]
        y0 = E_grid_flat[indices - 1]
        y1 = E_grid_flat[indices]
        
        # Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        E = y0 + (x_flat - x0) * (y1 - y0) / (x1 - x0 + 1e-10)
        E = E.unsqueeze(1)
        
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        dE_dx_on_grid = torch.autograd.grad(E_on_grid, x_grid_E, torch.ones_like(E_on_grid), create_graph=True)[0]
        poisson_residual_on_grid = dE_dx_on_grid - charge_dev_on_grid

        return vlasov_residual, poisson_residual_on_grid
    
    # --- NEW: Classic 3-component loss function ---
    def compute_loss(self, t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc):
        """
        Calculates the classic PINN loss, comprising PDE, IC, and BC residuals.
        """
        # --- 1. PDE Loss (Governing Equations) ---
        t_pde.requires_grad_(True); x_pde.requires_grad_(True); v_pde.requires_grad_(True)
        vlasov_res, poisson_res_grid = self._get_residuals(t_pde, x_pde, v_pde)
        loss_pde = torch.mean(vlasov_res**2) + torch.mean(poisson_res_grid**2)

        # --- 2. Initial Condition (IC) Loss ---
        t_ic_norm = self._normalize_t(t_ic)
        x_ic_norm = self._normalize_x(x_ic)
        v_ic_norm = self._normalize_v(v_ic)
        ic_txv_norm = torch.cat([t_ic_norm, x_ic_norm, v_ic_norm], dim=1)
        f_pred_ic = self.model(ic_txv_norm)
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        # --- 3. Boundary Condition (BC) Loss ---
        # Spatial periodic boundary: f(t, x_min, v) = f(t, x_max, v)
        # x_min = torch.full_like(x_bc, self.domain['x'][0])
        # x_max = torch.full_like(x_bc, self.domain['x'][1])
        # txv_min = torch.cat([t_bc, x_min, v_bc], dim=1)
        # txv_max = torch.cat([t_bc, x_max, v_bc], dim=1)
        # f_bc_min = self.model(txv_min)
        # f_bc_max = self.model(txv_max)
        # loss_bc_periodic = torch.mean((f_bc_min - f_bc_max)**2)
        
        # Velocity boundary: f(t, x, v_min/v_max) = 0
        v_min = torch.full_like(v_bc, self.domain['v'][0])
        v_max = torch.full_like(v_bc, self.domain['v'][1])
        
        # Normalize boundary condition points
        t_bc_norm = self._normalize_t(t_bc)
        x_bc_norm = self._normalize_x(x_bc)
        v_min_norm = self._normalize_v(v_min)
        v_max_norm = self._normalize_v(v_max)
        
        txv_vmin_norm = torch.cat([t_bc_norm, x_bc_norm, v_min_norm], dim=1)
        txv_vmax_norm = torch.cat([t_bc_norm, x_bc_norm, v_max_norm], dim=1)
        f_bc_vmin = self.model(txv_vmin_norm)
        f_bc_vmax = self.model(txv_vmax_norm)
        loss_bc_zero = torch.mean(f_bc_vmin**2) + torch.mean(f_bc_vmax**2)

        # loss_bc = loss_bc_periodic + loss_bc_zero
        loss_bc = loss_bc_zero  # Using only velocity BC for simplicity

        # --- Total Loss ---
        total_loss = (
            self.config['weight_pde'] * loss_pde +
            self.config['weight_ic'] * loss_ic +
            self.config['weight_bc'] * loss_bc
        )
        
        return total_loss, loss_pde, loss_ic, loss_bc

    def train(self):
        """The main training loop using the classic 3-component loss."""
        print("Starting training with classic PDE, IC, BC loss...")
        start_time = time.time()
        
        with open(self.log_file_path, 'w') as f:
            f.write('Epoch,Total_Loss,PDE_Loss,IC_Loss,BC_Loss,Time_s\n')

        for epoch in range(self.config['epochs']):
            self.model.train()
            
            # Sample points for PDE, IC, and BC
            t_pde = torch.rand(self.config['n_pde'], 1, device=self.device) * self.domain['t'][1]
            x_pde = torch.rand(self.config['n_pde'], 1, device=self.device) * self.domain['x'][1]
            v_pde = (torch.rand(self.config['n_pde'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            t_ic = torch.zeros(self.config['n_ic'], 1, device=self.device)
            x_ic = torch.rand(self.config['n_ic'], 1, device=self.device) * self.domain['x'][1]
            v_ic = (torch.rand(self.config['n_ic'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            t_bc = torch.rand(self.config['n_bc'], 1, device=self.device) * self.domain['t'][1]
            x_bc = torch.rand(self.config['n_bc'], 1, device=self.device) * self.domain['x'][1]
            v_bc = (torch.rand(self.config['n_bc'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]

            self.optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_bc = \
                self.compute_loss(t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}. Skipping.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if (epoch + 1) % 1000 == 0: self.scheduler.step()

            if (epoch + 1) % self.config['log_frequency'] == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch [{epoch+1}/{self.config['epochs']}] | "
                    f"Loss: {loss.item():.4e} | L_pde: {loss_pde.item():.4e} | "
                    f"L_ic: {loss_ic.item():.4e} | L_bc: {loss_bc.item():.4e} | "
                    f"Time: {elapsed_time:.2f}s"
                )
                
                log_data = (f"{epoch+1},{loss.item()},{loss_pde.item()},"
                            f"{loss_ic.item()},{loss_bc.item()},{elapsed_time}\n")
                with open(self.log_file_path, 'a') as f:
                    f.write(log_data)
                
                self.writer.add_scalar('Loss/Total', loss.item(), epoch + 1)
                self.writer.add_scalar('Loss/PDE', loss_pde.item(), epoch + 1)
                self.writer.add_scalar('Loss/Initial_Condition', loss_ic.item(), epoch + 1)
                self.writer.add_scalar('Loss/Boundary_Condition', loss_bc.item(), epoch + 1)
            
            if (epoch + 1) % self.config['plot_frequency'] == 0:
                self.plot_results(epoch + 1)
        
        self.writer.close()
        self.plot_loss_history()
        print("Training finished.")

    # (plot_results function can remain the same)
    @torch.no_grad()
    def plot_results(self, epoch):
        """Generates and saves plots of the current simulation state."""
        self.model.eval()
        print(f"Generating plots for epoch {epoch}...")
        x_grid = torch.linspace(self.domain['x'][0], self.domain['x'][1], 100, device=self.device)
        v_grid = torch.linspace(self.domain['v'][0], self.domain['v'][1], 100, device=self.device)
        X, V = torch.meshgrid(x_grid, v_grid, indexing='ij')
        plot_times = [0.0, self.domain['t'][1] / 2, self.domain['t'][1]]
        
        fig = plt.figure(figsize=(18, 12)); gs = GridSpec(2, 3, figure=fig)

        # Plot f(t,x,v) at different times
        for i, t_val in enumerate(plot_times):
            T = torch.full_like(X, t_val)
            f_pred = self.model(torch.stack([T.flatten(), X.flatten(), V.flatten()], dim=1)).reshape(X.shape)
            ax = fig.add_subplot(gs[0, i])
            im = ax.pcolormesh(X.cpu(), V.cpu(), f_pred.cpu(), cmap='jet', shading='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('x (position)'); ax.set_ylabel('v (velocity)')
            ax.set_title(f'PINN Solution f(t,x,v) at t={t_val:.2f}')
        
        # Plot True Initial Condition
        ax_ic = fig.add_subplot(gs[1, 0])
        f_ic_true = self._initial_condition(X, V)
        im_ic = ax_ic.pcolormesh(X.cpu(), V.cpu(), f_ic_true.cpu(), cmap='jet', shading='auto')
        fig.colorbar(im_ic, ax=ax_ic)
        ax_ic.set_xlabel('x (position)'); ax_ic.set_ylabel('v (velocity)')
        ax_ic.set_title('True Initial Condition f(0,x,v)')

        # Plot final electron density
        t_final = torch.full((x_grid.shape[0], 1), self.domain['t'][1], device=self.device)
        n_e_final = self._compute_ne(t_final, x_grid.unsqueeze(1))
        ax_ne = fig.add_subplot(gs[1, 1])
        ax_ne.plot(x_grid.cpu(), n_e_final.cpu(), 'b-', label='Electron Density')
        ax_ne.axhline(y=1.0, color='r', linestyle='--', label='Background Density')
        ax_ne.legend(); ax_ne.grid(True)
        ax_ne.set_title(f'Electron Density n_e(t,x) at t={self.domain["t"][1]:.2f}')
        ax_ne.set_xlabel('x (position)'); ax_ne.set_ylabel('n_e')

        # Plot final electric field
        ax_e = fig.add_subplot(gs[1, 2])
        charge_dev_final = n_e_final - 1.0
        dx_final = x_grid[1] - x_grid[0]
        E_final = torch.cumsum(charge_dev_final, dim=0) * dx_final
        E_final -= torch.mean(E_final)
        ax_e.plot(x_grid.cpu(), E_final.cpu(), 'g-')
        ax_e.grid(True)
        ax_e.set_title(f'Electric Field E(t,x) at t={self.domain["t"][1]:.2f}')
        ax_e.set_xlabel('x (position)'); ax_e.set_ylabel('E (Electric Field)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['plot_dir'], f'results_epoch_{epoch}.png'))
        plt.close(fig)

    # --- UPDATED: plot_loss_history function ---
    def plot_loss_history(self):
        """Plots the history for the classic 3-component loss."""
        print("Plotting loss history...")
        try:
            log_data = np.loadtxt(self.log_file_path, delimiter=',', skiprows=1)
            plt.figure(figsize=(12, 8))
            plt.plot(log_data[:, 0], log_data[:, 1], 'k', label='Total Loss')
            plt.plot(log_data[:, 0], log_data[:, 2], 'r--', alpha=0.7, label='PDE Loss')
            plt.plot(log_data[:, 0], log_data[:, 3], 'g--', alpha=0.7, label='IC Loss')
            plt.plot(log_data[:, 0], log_data[:, 4], 'b--', alpha=0.7, label='BC Loss')
            plt.yscale('log'); plt.title('Loss History'); plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(self.config['plot_dir'], 'loss_history.png'))
            plt.close()
            print("Loss history plot saved.")
        except Exception as e:
            print(f"Could not plot loss history: {e}")


# --- UPDATED: Main configuration block ---
if __name__ == '__main__':
    configuration = {
        # --- Domain Parameters ---
        't_max': 62.5,
        'x_max': 10.0,
        'v_max': 5.0,

        # --- Physics Parameters ---
        'beam_v': 1.0,
        'thermal_v': 0.02,
        'perturb_amp': 0.05,

        # --- Neural Network Architecture ---
        'nn_layers': 12,
        'nn_neurons': 256,

        # --- Training Hyperparameters ---
        'epochs': 1000,
        'learning_rate': 1e-4,
        'n_pde': 70000,                # Number of points for PDE residuals
        'n_ic': 700,                  # Number of points for Initial Condition
        'n_bc': 1100,                  # Number of points for Boundary Condition

        # --- Loss Function Weights (Classic Setup) ---
        'weight_pde': 1.0,             # Weight for the governing equations
        'weight_ic': 5.0,            # High weight for the initial condition
        'weight_bc': 10.0,            # High weight for the boundary conditions

        # --- Numerical & Logging Parameters ---
        'v_quad_points': 128,
        'log_frequency': 200,
        'plot_frequency': 200,
        'plot_dir': 'local_1000_new'
    }
    
    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()