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
np.random.seed(42)


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
            't': (0.0, config['T_MAX']),
            'x': (0.0, config['X_MAX']),
            'v': (-config['V_MAX'], config['V_MAX'])
        }

        self.model = MLP(
            layers=config['NN_LAYERS'], neurons=config['NN_NEURONS']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.999) # Standard betas
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        v_quad = torch.linspace(
            -config['V_MAX'], config['V_MAX'],
            config['V_QUAD_POINTS'], device=self.device
        )
        self.v_quad = v_quad.view(1, 1, -1)

        os.makedirs(self.config['PLOT_DIR'], exist_ok=True)
        self.log_file_path = os.path.join(self.config['PLOT_DIR'], 'training_log.txt')
        self.writer = SummaryWriter(log_dir=self.config['PLOT_DIR'])

    def _initial_condition(self, x, v):
        k = 2 * np.pi / self.config['X_MAX']
        norm_factor = 1.0 / (self.config['THERMAL_V'] * np.sqrt(2 * np.pi))
        term1 = norm_factor * torch.exp(-((v - self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        term2 = norm_factor * torch.exp(-((v + self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        return 0.5 * (term1 + term2) * (1 + self.config['PERTURB_AMP'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        t_exp = t.unsqueeze(2).expand(-1, -1, self.config['V_QUAD_POINTS'])
        x_exp = x.unsqueeze(2).expand(-1, -1, self.config['V_QUAD_POINTS'])
        t_flat, x_flat = t_exp.reshape(-1, 1), x_exp.reshape(-1, 1)
        v_flat = self.v_quad.expand(t.shape[0], -1, -1).reshape(-1, 1)
        
        txv = torch.cat([t_flat, x_flat, v_flat], dim=1)
        f_vals = self.model(txv).view(t.shape[0], self.config['V_QUAD_POINTS'])
        integral = torch.trapezoid(f_vals, self.v_quad.squeeze(), dim=1)
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        txv = torch.cat([t, x, v], dim=1)
        f = self.model(txv)
        df_d_txv = torch.autograd.grad(f, txv, torch.ones_like(f), create_graph=True)[0]
        df_dt, df_dx, df_dv = df_d_txv.split(1, dim=1)
        
        x_grid_E = torch.linspace(0, self.config['X_MAX'], 101, device=self.device).unsqueeze(1).requires_grad_()
        t_mean_E = torch.full_like(x_grid_E, t.mean().item())
        n_e_on_grid = self._compute_ne(t_mean_E, x_grid_E)
        charge_dev_on_grid = n_e_on_grid - 1.0
        
        dx_E = x_grid_E[1] - x_grid_E[0]
        E_on_grid = torch.cumsum(charge_dev_on_grid, dim=0) * dx_E
        E_on_grid = E_on_grid - torch.mean(E_on_grid)
        
        E_interp = np.interp(
            x.cpu().detach().numpy().flatten(), x_grid_E.cpu().detach().numpy().flatten(),
            E_on_grid.cpu().detach().numpy().flatten()
        )
        E = torch.from_numpy(E_interp).to(self.device).float().unsqueeze(1)
        
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        dE_dx_on_grid = torch.autograd.grad(E_on_grid, x_grid_E, torch.ones_like(E_on_grid), create_graph=True)[0]
        poisson_residual_on_grid = dE_dx_on_grid - charge_dev_on_grid

        return vlasov_residual, poisson_residual_on_grid
    
    # --- NEW: Classic 3-component loss function ---
    def compute_classic_loss(self, t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc):
        """
        Calculates the classic PINN loss, comprising PDE, IC, and BC residuals.
        """
        # --- 1. PDE Loss (Governing Equations) ---
        t_pde.requires_grad_(True); x_pde.requires_grad_(True); v_pde.requires_grad_(True)
        vlasov_res, poisson_res_grid = self._get_residuals(t_pde, x_pde, v_pde)
        loss_pde = torch.mean(vlasov_res**2) + torch.mean(poisson_res_grid**2)

        # --- 2. Initial Condition (IC) Loss ---
        ic_txv = torch.cat([t_ic, x_ic, v_ic], dim=1)
        f_pred_ic = self.model(ic_txv)
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        # --- 3. Boundary Condition (BC) Loss ---
        # Spatial periodic boundary: f(t, x_min, v) = f(t, x_max, v)
        x_min = torch.full_like(x_bc, self.domain['x'][0])
        x_max = torch.full_like(x_bc, self.domain['x'][1])
        txv_min = torch.cat([t_bc, x_min, v_bc], dim=1)
        txv_max = torch.cat([t_bc, x_max, v_bc], dim=1)
        f_bc_min = self.model(txv_min)
        f_bc_max = self.model(txv_max)
        loss_bc_periodic = torch.mean((f_bc_min - f_bc_max)**2)
        
        # Velocity boundary: f(t, x, v_min/v_max) = 0
        v_min = torch.full_like(v_bc, self.domain['v'][0])
        v_max = torch.full_like(v_bc, self.domain['v'][1])
        txv_vmin = torch.cat([t_bc, x_bc, v_min], dim=1)
        txv_vmax = torch.cat([t_bc, x_bc, v_max], dim=1)
        f_bc_vmin = self.model(txv_vmin)
        f_bc_vmax = self.model(txv_vmax)
        loss_bc_zero = torch.mean(f_bc_vmin**2) + torch.mean(f_bc_vmax**2)

        loss_bc = loss_bc_periodic + loss_bc_zero

        # --- Total Loss ---
        total_loss = (
            self.config['LAMBDA_PDE'] * loss_pde +
            self.config['LAMBDA_IC'] * loss_ic +
            self.config['LAMBDA_BC'] * loss_bc
        )
        
        return total_loss, loss_pde, loss_ic, loss_bc

    # --- UPDATED: train function ---
    def train(self):
        """The main training loop using the classic 3-component loss."""
        print("Starting training with classic PDE, IC, BC loss...")
        start_time = time.time()
        
        with open(self.log_file_path, 'w') as f:
            f.write('Epoch,Total_Loss,PDE_Loss,IC_Loss,BC_Loss,Time_s\n')

        for epoch in range(self.config['EPOCHS']):
            self.model.train()
            
            # Sample points for PDE, IC, and BC
            t_pde = torch.rand(self.config['N_PDE'], 1, device=self.device) * self.domain['t'][1]
            x_pde = torch.rand(self.config['N_PDE'], 1, device=self.device) * self.domain['x'][1]
            v_pde = (torch.rand(self.config['N_PDE'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            t_ic = torch.zeros(self.config['N_IC'], 1, device=self.device)
            x_ic = torch.rand(self.config['N_IC'], 1, device=self.device) * self.domain['x'][1]
            v_ic = (torch.rand(self.config['N_IC'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            t_bc = torch.rand(self.config['N_BC'], 1, device=self.device) * self.domain['t'][1]
            x_bc = torch.rand(self.config['N_BC'], 1, device=self.device) * self.domain['x'][1]
            v_bc = (torch.rand(self.config['N_BC'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]

            self.optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_bc = \
                self.compute_classic_loss(t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}. Skipping.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if (epoch + 1) % 1000 == 0: self.scheduler.step()

            if (epoch + 1) % self.config['LOG_FREQUENCY'] == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch [{epoch+1}/{self.config['EPOCHS']}] | "
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
            
            if (epoch + 1) % self.config['PLOT_FREQUENCY'] == 0:
                self.plot_results(epoch + 1)
        
        self.writer.close()
        self.plot_loss_history()
        print("Training finished.")

    # (plot_results function can remain the same)
    @torch.no_grad()
    def plot_results(self, epoch):
        # ... (no changes needed here, you can use your preferred 2x3 or 2x4 layout)
        pass

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
            plt.savefig(os.path.join(self.config['PLOT_DIR'], 'loss_history.png'))
            plt.close()
            print("Loss history plot saved.")
        except Exception as e:
            print(f"Could not plot loss history: {e}")


# --- UPDATED: Main configuration block ---
if __name__ == '__main__':
    configuration = {
        # --- Domain Parameters ---
        'T_MAX': 15.0,
        'X_MAX': 2 * np.pi / 0.5,
        'V_MAX': 6.0,

        # --- Physics Parameters ---
        'BEAM_V': 2.4,
        'THERMAL_V': 0.2,
        'PERTURB_AMP': 0.05,

        # --- Neural Network Architecture ---
        'NN_LAYERS': 8,
        'NN_NEURONS': 128,

        # --- Training Hyperparameters ---
        'EPOCHS': 1000,
        'LEARNING_RATE': 1e-4,
        'N_PDE': 15000,                # Number of points for PDE residuals
        'N_IC': 5000,                  # Number of points for Initial Condition
        'N_BC': 5000,                  # Number of points for Boundary Condition

        # --- Loss Function Weights (Classic Setup) ---
        'LAMBDA_PDE': 1.0,             # Weight for the governing equations
        'LAMBDA_IC': 100.0,            # High weight for the initial condition
        'LAMBDA_BC': 100.0,            # High weight for the boundary conditions

        # --- Numerical & Logging Parameters ---
        'V_QUAD_POINTS': 128,
        'LOG_FREQUENCY': 200,
        'PLOT_FREQUENCY': 250,
        'PLOT_DIR': 'classic_1000'
    }
    
    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()