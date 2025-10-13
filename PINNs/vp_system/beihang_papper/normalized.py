"""
================================================================================
Physics-Informed Neural Network (PINN) for solving the 1D Vlasov-Poisson system.
This version uses a classic three-component loss function and incorporates
 crucial input normalization for stable and accurate training.
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
np.random.seed(42)

class MLP(nn.Module):
    """Defines the MLP, ensuring f >= 0 with Softplus."""
    def __init__(self, input_dim=3, output_dim=1, layers=12, neurons=256):
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

        # --- NEW: Store normalization bounds ---
        self.t_bounds = torch.tensor([self.domain['t'][0], self.domain['t'][1]], device=self.device)
        self.x_bounds = torch.tensor([self.domain['x'][0], self.domain['x'][1]], device=self.device)
        self.v_bounds = torch.tensor([self.domain['v'][0], self.domain['v'][1]], device=self.device)

        self.model = MLP(
            layers=config['nn_layers'], neurons=config['nn_neurons']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        v_quad = torch.linspace(
            -config['v_max'], config['v_max'], config['v_quad_points'], device=self.device
        )
        self.v_quad = v_quad.view(1, 1, -1)

        os.makedirs(self.config['plot_dir'], exist_ok=True)
        self.log_file_path = os.path.join(self.config['plot_dir'], 'training_log.txt')
        self.writer = SummaryWriter(log_dir=self.config['plot_dir'])

    # --- NEW: Normalization function ---
    def _normalize(self, t, x, v):
        """Normalizes inputs to the range [-1, 1] for better training performance."""
        t_norm = 2.0 * (t - self.t_bounds[0]) / (self.t_bounds[1] - self.t_bounds[0]) - 1.0
        x_norm = 2.0 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) - 1.0
        v_norm = 2.0 * (v - self.v_bounds[0]) / (self.v_bounds[1] - self.v_bounds[0]) - 1.0
        return t_norm, x_norm, v_norm

    # --- NEW: Forward pass wrapper that includes normalization ---
    def _forward(self, t, x, v):
        """A wrapper for the model's forward pass that includes input normalization."""
        t_norm, x_norm, v_norm = self._normalize(t, x, v)
        txv_normalized = torch.cat([t_norm, x_norm, v_norm], dim=1)
        return self.model(txv_normalized)

    def _initial_condition(self, x, v):
        """Initial condition: perturbed double Maxwellian."""
        k = 2 * np.pi / self.config['x_max']
        norm_factor = 1.0 / (self.config['thermal_v'] * np.sqrt(2 * np.pi))
        term1 = norm_factor * torch.exp(-((v - self.config['beam_v'])**2) / (2 * self.config['thermal_v']**2))
        term2 = norm_factor * torch.exp(-((v + self.config['beam_v'])**2) / (2 * self.config['thermal_v']**2))
        return 0.5 * (term1 + term2) * (1 + self.config['perturb_amp'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """Computes electron density n_e(t,x) by integrating f over v."""
        t_exp = t.unsqueeze(2).expand(-1, -1, self.config['v_quad_points'])
        x_exp = x.unsqueeze(2).expand(-1, -1, self.config['v_quad_points'])
        t_flat, x_flat = t_exp.reshape(-1, 1), x_exp.reshape(-1, 1)
        v_flat = self.v_quad.expand(t.shape[0], -1, -1).reshape(-1, 1)
        
        # Use the forward wrapper for normalization
        f_vals = self._forward(t_flat, x_flat, v_flat).view(t.shape[0], self.config['v_quad_points'])
        integral = torch.trapezoid(f_vals, self.v_quad.squeeze(), dim=1)
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        """Calculates the residuals for the Vlasov and Poisson equations."""
        txv = torch.cat([t, x, v], dim=1)
        
        # The model's output f is a function of the physical coordinates t,x,v
        # The normalization is handled inside the _forward method
        f = self._forward(t, x, v)
        
        # CRITICAL: Compute gradients with respect to the original physical coordinates
        df_d_txv = torch.autograd.grad(f, txv, torch.ones_like(f), create_graph=True)[0]
        df_dt, df_dx, df_dv = df_d_txv.split(1, dim=1)
        
        x_grid_E = torch.linspace(0, self.config['x_max'], 101, device=self.device).unsqueeze(1).requires_grad_()
        t_mean_E = torch.full_like(x_grid_E, t.mean().item())
        n_e_on_grid = self._compute_ne(t_mean_E, x_grid_E)
        charge_dev_on_grid = n_e_on_grid - 1.0
        
        dx_E = x_grid_E[1] - x_grid_E[0]
        E_on_grid = torch.cumsum(charge_dev_on_grid, dim=0) * dx_E
        E_on_grid = E_on_grid - torch.mean(E_on_grid)
        
        # Interpolation (switched back to NumPy for simplicity and robustness)
        E_interp = np.interp(
            x.cpu().detach().numpy().flatten(),
            x_grid_E.cpu().detach().numpy().flatten(),
            E_on_grid.cpu().detach().numpy().flatten()
        )
        E = torch.from_numpy(E_interp).to(self.device).float().unsqueeze(1)
        
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        dE_dx_on_grid = torch.autograd.grad(E_on_grid, x_grid_E, torch.ones_like(E_on_grid), create_graph=True)[0]
        poisson_residual_on_grid = dE_dx_on_grid - charge_dev_on_grid

        return vlasov_residual, poisson_residual_on_grid
    
    def compute_loss(self, t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc):
        """Calculates the classic PINN loss, comprising PDE, IC, and BC residuals."""
        t_pde.requires_grad_(True); x_pde.requires_grad_(True); v_pde.requires_grad_(True)
        vlasov_res, poisson_res_grid = self._get_residuals(t_pde, x_pde, v_pde)
        loss_pde = torch.mean(vlasov_res**2) + torch.mean(poisson_res_grid**2)

        f_pred_ic = self._forward(t_ic, x_ic, v_ic) # Use wrapper
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        v_min = torch.full_like(v_bc, self.domain['v'][0])
        v_max = torch.full_like(v_bc, self.domain['v'][1])
        f_bc_vmin = self._forward(t_bc, x_bc, v_min) # Use wrapper
        f_bc_vmax = self._forward(t_bc, x_bc, v_max) # Use wrapper
        loss_bc = torch.mean(f_bc_vmin**2) + torch.mean(f_bc_vmax**2)

        total_loss = (
            self.config['weight_pde'] * loss_pde +
            self.config['weight_ic'] * loss_ic +
            self.config['weight_bc'] * loss_bc
        )
        return total_loss, loss_pde, loss_ic, loss_bc

    def train(self):
        """The main training loop using the classic 3-component loss."""
        print("Starting training with classic PDE, IC, BC loss and input normalization...")
        start_time = time.time()
        
        with open(self.log_file_path, 'w') as f:
            f.write('Epoch,Total_Loss,PDE_Loss,IC_Loss,BC_Loss,Time_s\n')

        for epoch in range(self.config['epochs']):
            self.model.train()
            
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

    @torch.no_grad()
    def plot_results(self, epoch):
        """Generates and saves plots of the current simulation state."""
        self.model.eval()
        print(f"Generating plots for epoch {epoch}...")
        x_grid = torch.linspace(self.domain['x'][0], self.domain['x'][1], 100, device=self.device)
        v_grid = torch.linspace(self.domain['v'][0], self.domain['v'][1], 100, device=self.device)
        X, V = torch.meshgrid(x_grid, v_grid, indexing='ij')
        plot_times = [25, 37.5, 50]
        
        fig = plt.figure(figsize=(18, 12)); gs = GridSpec(2, 3, figure=fig)

        for i, t_val in enumerate(plot_times):
            T = torch.full_like(X, float(t_val))
            f_pred = self._forward(T.flatten(), X.flatten(), V.flatten()).reshape(X.shape) # Use wrapper
            ax = fig.add_subplot(gs[0, i])
            im = ax.pcolormesh(X.cpu(), V.cpu(), f_pred.cpu(), cmap='jet', shading='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('x (position)'); ax.set_ylabel('v (velocity)')
            ax.set_title(rf'PINN Solution f(t,x,v) at t={t_val:.2f} $\omega_p^{{-1}}$')
        
        ax_ic = fig.add_subplot(gs[1, 0])
        f_ic_true = self._initial_condition(X, V)
        im_ic = ax_ic.pcolormesh(X.cpu(), V.cpu(), f_ic_true.cpu(), cmap='jet', shading='auto')
        fig.colorbar(im_ic, ax=ax_ic)
        ax_ic.set_xlabel('x (position)'); ax_ic.set_ylabel('v (velocity)')
        ax_ic.set_title('True Initial Condition f(0,x,v)')

        t_final = torch.full((x_grid.shape[0], 1), self.domain['t'][1], device=self.device)
        n_e_final = self._compute_ne(t_final, x_grid.unsqueeze(1))
        ax_ne = fig.add_subplot(gs[1, 1])
        ax_ne.plot(x_grid.cpu(), n_e_final.cpu(), 'b-', label='Electron Density')
        ax_ne.axhline(y=1.0, color='r', linestyle='--', label='Background Density')
        ax_ne.legend(); ax_ne.grid(True)
        ax_ne.set_title(f'Electron Density n_e(t,x) at t={self.domain["t"][1]:.2f}')
        ax_ne.set_xlabel('x (position)'); ax_ne.set_ylabel('n_e')

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

    def plot_loss_history(self):
        """Plots the history for the classic 3-component loss."""
        # ... (This function remains unchanged, you can copy it from the previous version)
        pass


if __name__ == '__main__':
    configuration = {
        't_max': 62.5, 'x_max': 10.0, 'v_max': 5.0,
        'beam_v': 1.0, 'thermal_v': 0.02, 'perturb_amp': 0.05,
        'nn_layers': 12, 'nn_neurons': 256,
        'epochs': 1000, 'learning_rate': 1e-4,
        'n_pde': 70000, 'n_ic': 700, 'n_bc': 1100,
        'weight_pde': 1.0, 'weight_ic': 5.0, 'weight_bc': 10.0,
        'v_quad_points': 128, 'log_frequency': 200, 'plot_frequency': 200,
        'plot_dir': 'local_1000_normalized'
    }
    
    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()