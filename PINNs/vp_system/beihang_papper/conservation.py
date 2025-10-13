"""
================================================================================
Physics-Informed Neural Network (PINN) for solving the 1D Vlasov-Poisson system.
================================================================================
This script implements a PINN to simulate the two-stream instability, a classic
phenomenon in plasma physics.

Key Features:
- Self-consistent E-field calculation via numerical integration.
- Enforces f(t,x,v) >= 0 using a Softplus activation layer.
- Includes symmetry loss (f(t,x,v) = f(t,x,-v)) for physical accuracy.
- Enforces total particle number conservation as a loss term.
- Robust logging with TensorBoard, TXT files, and loss history plots.
- Highly modular and configurable for easy maintenance and innovation.
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
    """
    Defines the Multilayer Perceptron (MLP) for the PINN.
    The final Softplus activation ensures the output f(t,x,v) is non-negative.
    """
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
            self.model.parameters(), lr=config['LEARNING_RATE'], betas=(0.99, 0.999)
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
        """Defines the properly normalized initial distribution f(0,x,v)."""
        k = 2 * np.pi / self.config['X_MAX']
        norm_factor = 1.0 / (self.config['THERMAL_V'] * np.sqrt(2 * np.pi))
        term1 = norm_factor * torch.exp(-((v - self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        term2 = norm_factor * torch.exp(-((v + self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        return 0.5 * (term1 + term2) * (1 + self.config['PERTURB_AMP'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """Computes electron density n_e by integrating f(t,x,v) over v."""
        t_exp = t.unsqueeze(2).expand(-1, -1, self.config['V_QUAD_POINTS'])
        x_exp = x.unsqueeze(2).expand(-1, -1, self.config['V_QUAD_POINTS'])
        t_flat, x_flat = t_exp.reshape(-1, 1), x_exp.reshape(-1, 1)
        v_flat = self.v_quad.expand(t.shape[0], -1, -1).reshape(-1, 1)
        
        txv = torch.cat([t_flat, x_flat, v_flat], dim=1)
        f_vals = self.model(txv).view(t.shape[0], self.config['V_QUAD_POINTS'])
        integral = torch.trapezoid(f_vals, self.v_quad.squeeze(), dim=1)
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        """Computes the physically correct residuals of the Vlasov-Poisson equations."""
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
            x.cpu().detach().numpy().flatten(),
            x_grid_E.cpu().detach().numpy().flatten(),
            E_on_grid.cpu().detach().numpy().flatten()
        )
        E = torch.from_numpy(E_interp).to(self.device).float().unsqueeze(1)
        
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        dE_dx_on_grid = torch.autograd.grad(E_on_grid, x_grid_E, torch.ones_like(E_on_grid), create_graph=True)[0]
        poisson_residual_on_grid = dE_dx_on_grid - charge_dev_on_grid

        return vlasov_residual, poisson_residual_on_grid
    
    def compute_loss_with_conservation(self, t_phy, x_phy, v_phy, t_ic, x_ic, v_ic):
        """Calculates total loss, including all constraints plus particle conservation."""
        t_phy.requires_grad_(True); x_phy.requires_grad_(True); v_phy.requires_grad_(True)

        vlasov_res, poisson_res_grid = self._get_residuals(t_phy, x_phy, v_phy)
        loss_vlasov = torch.mean(vlasov_res**2)
        loss_poisson = torch.mean(poisson_res_grid**2)

        ic_txv = torch.cat([t_ic, x_ic, v_ic], dim=1)
        f_pred_ic = self.model(ic_txv)
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        txv_pos_v = torch.cat([t_phy, x_phy, v_phy], dim=1)
        txv_neg_v = torch.cat([t_phy, x_phy, -v_phy], dim=1)
        f_pos_v = self.model(txv_pos_v)
        f_neg_v = self.model(txv_neg_v)
        loss_symm = torch.mean((f_pos_v - f_neg_v)**2)

        # --- Total Particle Number Conservation Loss ---
        x_grid_cons = torch.linspace(0, self.config['X_MAX'], 201, device=self.device)
        with torch.no_grad():
            t_zero = torch.zeros((x_grid_cons.shape[0], 1), device=self.device)
            n_e_ic_on_grid = self._compute_ne(t_zero, x_grid_cons.unsqueeze(1))
            total_particles_true = torch.trapezoid(n_e_ic_on_grid.squeeze(), x_grid_cons).detach()
        
        t_phy_mean = t_phy.mean().expand(x_grid_cons.shape[0], 1)
        n_e_phy_on_grid = self._compute_ne(t_phy_mean, x_grid_cons.unsqueeze(1))
        total_particles_pred = torch.trapezoid(n_e_phy_on_grid.squeeze(), x_grid_cons)
        loss_conservation = (total_particles_pred - total_particles_true)**2
        
        total_loss = (
            self.config['LAMBDA_VLASOV'] * loss_vlasov +
            self.config['LAMBDA_POISSON'] * loss_poisson +
            self.config['LAMBDA_IC'] * loss_ic +
            self.config['LAMBDA_SYMM'] * loss_symm +
            self.config['LAMBDA_CONSERVATION'] * loss_conservation
        )
        return total_loss, loss_vlasov, loss_poisson, loss_ic, loss_symm, loss_conservation

    def train(self):
        """The main training loop, with full logging capabilities."""
        print("Starting training with conservation loss and TensorBoard...")
        start_time = time.time()
        
        with open(self.log_file_path, 'w') as f:
            f.write('Epoch,Total_Loss,Vlasov_Loss,Poisson_Loss,IC_Loss,Symmetry_Loss,Conservation_Loss,Time_s\n')

        for epoch in range(self.config['EPOCHS']):
            self.model.train()
            
            t_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['t'][1]
            x_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['x'][1]
            v_phy = (torch.rand(self.config['N_PHY'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            t_ic = torch.zeros(self.config['N_IC'], 1, device=self.device)
            x_ic = torch.rand(self.config['N_IC'], 1, device=self.device) * self.domain['x'][1]
            v_ic = (torch.rand(self.config['N_IC'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]

            self.optimizer.zero_grad()
            loss, loss_v, loss_p, loss_ic, loss_s, loss_c = \
                self.compute_loss_with_conservation(t_phy, x_phy, v_phy, t_ic, x_ic, v_ic)
            
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
                    f"Epoch [{epoch+1}/{self.config['EPOCHS']}] | Loss: {loss.item():.4e} | "
                    f"L_v: {loss_v.item():.4e} | L_p: {loss_p.item():.4e} | "
                    f"L_ic: {loss_ic.item():.4e} | L_s: {loss_s.item():.4e} | "
                    f"L_c: {loss_c.item():.4e} | Time: {elapsed_time:.2f}s"
                )
                
                log_data = (f"{epoch+1},{loss.item()},{loss_v.item()},{loss_p.item()},"
                            f"{loss_ic.item()},{loss_s.item()},{loss_c.item()},{elapsed_time}\n")
                with open(self.log_file_path, 'a') as f:
                    f.write(log_data)
                
                self.writer.add_scalar('Loss/Total', loss.item(), epoch + 1)
                self.writer.add_scalar('Loss/Vlasov', loss_v.item(), epoch + 1)
                self.writer.add_scalar('Loss/Poisson', loss_p.item(), epoch + 1)
                self.writer.add_scalar('Loss/Initial_Condition', loss_ic.item(), epoch + 1)
                self.writer.add_scalar('Loss/Symmetry', loss_s.item(), epoch + 1)
                self.writer.add_scalar('Loss/Conservation', loss_c.item(), epoch + 1)
            
            if (epoch + 1) % self.config['PLOT_FREQUENCY'] == 0:
                self.plot_results(epoch + 1)
        
        self.writer.close()
        self.plot_loss_history()
        print("Training finished.")

    @torch.no_grad()
    def plot_results(self, epoch):
        """Generates plots in a 2x4 grid comparing PINN solution to the initial condition."""
        self.model.eval()
        print(f"Generating plots for epoch {epoch}...")
        x_grid = torch.linspace(self.domain['x'][0], self.domain['x'][1], 100, device=self.device)
        v_grid = torch.linspace(self.domain['v'][0], self.domain['v'][1], 100, device=self.device)
        X, V = torch.meshgrid(x_grid, v_grid, indexing='ij')
        plot_times = [0.0, self.domain['t'][1] / 2, self.domain['t'][1]]
        
        fig = plt.figure(figsize=(24, 12)); gs = GridSpec(2, 4, figure=fig)

        for i, t_val in enumerate(plot_times):
            T = torch.full_like(X, t_val)
            f_pred = self.model(torch.stack([T.flatten(), X.flatten(), V.flatten()], dim=1)).reshape(X.shape)
            ax = fig.add_subplot(gs[0, i])
            im = ax.pcolormesh(X.cpu(), V.cpu(), f_pred.cpu(), cmap='jet', shading='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('x (position)'); ax.set_ylabel('v (velocity)')
            ax.set_title(f'PINN Solution f(t,x,v) at t={t_val:.2f}')
        
        f_ic_true = self._initial_condition(X, V)
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            im = ax.pcolormesh(X.cpu(), V.cpu(), f_ic_true.cpu(), cmap='jet', shading='auto')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('x (position)'); ax.set_ylabel('v (velocity)')
            ax.set_title(f'True Initial Condition (t=0)')

        t_final = torch.full((x_grid.shape[0], 1), self.domain['t'][1], device=self.device)
        n_e_final = self._compute_ne(t_final, x_grid.unsqueeze(1))
        ax_ne = fig.add_subplot(gs[0, 3])
        ax_ne.plot(x_grid.cpu(), n_e_final.cpu(), 'b-', label='Electron Density')
        ax_ne.axhline(y=1.0, color='r', linestyle='--', label='Background Density')
        ax_ne.legend(); ax_ne.grid(True)
        ax_ne.set_title(f'Electron Density n_e at t={self.domain["t"][1]:.2f}')
        ax_ne.set_xlabel('x (position)'); ax_ne.set_ylabel('n_e')

        ax_e = fig.add_subplot(gs[1, 3])
        charge_dev_final = n_e_final - 1.0
        dx_final = x_grid[1] - x_grid[0]
        E_final = torch.cumsum(charge_dev_final, dim=0) * dx_final
        E_final -= torch.mean(E_final)
        ax_e.plot(x_grid.cpu(), E_final.cpu(), 'g-')
        ax_e.grid(True)
        ax_e.set_title(f'Electric Field E at t={self.domain["t"][1]:.2f}')
        ax_e.set_xlabel('x (position)'); ax_e.set_ylabel('E (Electric Field)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'results_epoch_{epoch}.png'))
        plt.close(fig)

    def plot_loss_history(self):
        """Plots the full loss history, including the new conservation loss."""
        print("Plotting loss history...")
        try:
            log_data = np.loadtxt(self.log_file_path, delimiter=',', skiprows=1)
            plt.figure(figsize=(12, 8))
            plt.plot(log_data[:, 0], log_data[:, 1], 'k', label='Total Loss')
            plt.plot(log_data[:, 0], log_data[:, 2], 'r--', alpha=0.7, label='Vlasov Loss')
            plt.plot(log_data[:, 0], log_data[:, 3], 'b--', alpha=0.7, label='Poisson Loss')
            plt.plot(log_data[:, 0], log_data[:, 4], 'g--', alpha=0.7, label='IC Loss')
            plt.plot(log_data[:, 0], log_data[:, 5], 'm--', alpha=0.7, label='Symmetry Loss')
            plt.plot(log_data[:, 0], log_data[:, 6], 'c--', alpha=0.7, label='Conservation Loss') # New plot line
            plt.yscale('log'); plt.title('Loss History'); plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(self.config['PLOT_DIR'], 'loss_history.png'))
            plt.close()
            print("Loss history plot saved.")
        except Exception as e:
            print(f"Could not plot loss history: {e}")

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
        'NN_LAYERS': 6,
        'NN_NEURONS': 512,

        # --- Training Hyperparameters ---
        'EPOCHS': 1000,
        'LEARNING_RATE': 1e-4,
        'N_PHY': 10000,
        'N_IC': 2000,

        # --- Loss Function Weights ---
        'LAMBDA_VLASOV': 1.0,
        'LAMBDA_POISSON': 1.0,
        'LAMBDA_IC': 3.0,
        'LAMBDA_SYMM': 3.0,
        'LAMBDA_CONSERVATION': 5.0,  # <-- NEW: Weight for conservation loss

        # --- Numerical & Logging Parameters ---
        'V_QUAD_POINTS': 128,
        'LOG_FREQUENCY': 100,
        'PLOT_FREQUENCY': 100,
        'PLOT_DIR': 'local_ultra_conserv'
    }
    
    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()