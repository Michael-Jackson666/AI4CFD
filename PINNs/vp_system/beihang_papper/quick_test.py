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
- Robust logging with TensorBoard, TXT files, and loss history plots.
- Highly modular and configurable for easy maintenance and innovation.

Dependencies:
- PyTorch
- NumPy
- Matplotlib
- TensorBoard

How to Run:
1. Ensure all dependencies are installed (`pip install torch numpy matplotlib tensorboard`).
2. Run the script: `python your_script_name.py`
3. To view live training metrics, open a new terminal and run:
   `tensorboard --logdir=./path_to_your_plot_dir`
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
    
    The final Softplus activation function is crucial for enforcing the physical
    constraint that the distribution function f(t,x,v) must be non-negative.
    """
    def __init__(self, input_dim=3, output_dim=1, layers=6, neurons=64):
        super(MLP, self).__init__()
        
        modules = [nn.Linear(input_dim, neurons), nn.Tanh()]
        for _ in range(layers - 2):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(neurons, output_dim))
        
        # Enforce non-negativity (f >= 0), a critical physical constraint.
        modules.append(nn.Softplus())
        
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class VlasovPoissonPINN:
    """
    A comprehensive PINN solver for the 1D Vlasov-Poisson system.

    This class encapsulates all the necessary components for the simulation:
    - The neural network model.
    - The definition of physical equations and constraints.
    - The training loop and optimization logic.
    - Visualization and logging functionalities.

    Future Work & Innovation Points:
    - Implement additional physical constraints, like energy conservation,
      as new loss terms in `compute_loss_with_symmetry`.
    - Experiment with different numerical integration schemes in `_compute_ne`.
    - Explore adaptive sampling for collocation points inside the `train` loop.
    - Modify the network architecture in the `MLP` class.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Define the simulation domain from the configuration
        self.domain = {
            't': (0.0, config['T_MAX']),
            'x': (0.0, config['X_MAX']),
            'v': (-config['V_MAX'], config['V_MAX'])
        }

        # Initialize the model, optimizer, and scheduler
        self.model = MLP(
            layers=config['NN_LAYERS'], neurons=config['NN_NEURONS']
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['LEARNING_RATE']
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        # Create a grid for numerical integration over the velocity space
        v_quad = torch.linspace(
            -config['V_MAX'], config['V_MAX'],
            config['V_QUAD_POINTS'], device=self.device
        )
        self.v_quad = v_quad.view(1, 1, -1)

        # Setup logging infrastructure
        os.makedirs(self.config['PLOT_DIR'], exist_ok=True)
        self.log_file_path = os.path.join(self.config['PLOT_DIR'], 'training_log.txt')
        with open(self.log_file_path, 'w') as f:
            f.write('Epoch,Total_Loss,Vlasov_Loss,Poisson_Loss,IC_Loss,Symmetry_Loss,Time_s\n')
        self.writer = SummaryWriter(log_dir=self.config['PLOT_DIR'])

    def _initial_condition(self, x, v):
        """Defines the initial distribution f(0,x,v) for the two-stream instability."""
        k = 2 * np.pi / self.config['X_MAX']
        term1 = torch.exp(-((v - self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        term2 = torch.exp(-((v + self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        # The 0.5 factor properly normalizes the two-beam distribution.
        return 0.5 * (term1 + term2) * (1 + self.config['PERTURB_AMP'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """Computes electron density n_e by integrating f(t,x,v) over v via the trapezoidal rule."""
        # Use broadcasting to efficiently create the (t,x,v) grid for integration
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
        # Step 1: Compute derivatives using Automatic Differentiation
        txv = torch.cat([t, x, v], dim=1)
        f = self.model(txv)
        df_d_txv = torch.autograd.grad(f, txv, torch.ones_like(f), create_graph=True)[0]
        df_dt, df_dx, df_dv = df_d_txv.split(1, dim=1)
        
        # Step 2: Self-consistent E-field calculation on a grid
        x_grid_E = torch.linspace(0, self.config['X_MAX'], 101, device=self.device).unsqueeze(1).requires_grad_()
        t_mean_E = torch.full_like(x_grid_E, t.mean().item()) # Use mean time for efficiency
        
        n_e_on_grid = self._compute_ne(t_mean_E, x_grid_E)
        charge_dev_on_grid = n_e_on_grid - 1.0
        
        dx_E = x_grid_E[1] - x_grid_E[0]
        E_on_grid = torch.cumsum(charge_dev_on_grid, dim=0) * dx_E
        E_on_grid = E_on_grid - torch.mean(E_on_grid) # Enforce periodic BC
        
        # Interpolate E back to the random collocation points
        E_interp = np.interp(
            x.cpu().detach().numpy().flatten(),
            x_grid_E.cpu().detach().numpy().flatten(),
            E_on_grid.cpu().detach().numpy().flatten()
        )
        E = torch.from_numpy(E_interp).to(self.device).float().unsqueeze(1)
        
        # Step 3: Compute Vlasov residual using the self-consistent E-field
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        # Step 4: Compute a robust Poisson residual on the grid
        dE_dx_on_grid = torch.autograd.grad(E_on_grid, x_grid_E, torch.ones_like(E_on_grid), create_graph=True)[0]
        poisson_residual_on_grid = dE_dx_on_grid - charge_dev_on_grid

        return vlasov_residual, poisson_residual_on_grid
    
    def compute_loss_with_symmetry(self, t_phy, x_phy, v_phy, t_ic, x_ic, v_ic):
        """Calculates the total loss function, combining all physical constraints."""
        t_phy.requires_grad_(True); x_phy.requires_grad_(True); v_phy.requires_grad_(True)

        # PDE residuals loss
        vlasov_res, poisson_res_grid = self._get_residuals(t_phy, x_phy, v_phy)
        loss_vlasov = torch.mean(vlasov_res**2)
        loss_poisson = torch.mean(poisson_res_grid**2)

        # Initial condition loss
        ic_txv = torch.cat([t_ic, x_ic, v_ic], dim=1)
        f_pred_ic = self.model(ic_txv)
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        # Symmetry condition loss (f(t,x,v) = f(t,x,-v))
        txv_pos_v = torch.cat([t_phy, x_phy, v_phy], dim=1)
        txv_neg_v = torch.cat([t_phy, x_phy, -v_phy], dim=1)
        f_pos_v = self.model(txv_pos_v)
        f_neg_v = self.model(txv_neg_v)
        loss_symm = torch.mean((f_pos_v - f_neg_v)**2)

        # Weighted sum of all losses
        total_loss = (
            self.config['LAMBDA_VLASOV'] * loss_vlasov +
            self.config['LAMBDA_POISSON'] * loss_poisson +
            self.config['LAMBDA_IC'] * loss_ic +
            self.config['LAMBDA_SYMM'] * loss_symm
        )
        return total_loss, loss_vlasov, loss_poisson, loss_ic, loss_symm

    def train(self):
        """The main training loop."""
        print("Starting final fast prototype training with TensorBoard logging...")
        start_time = time.time()

        for epoch in range(self.config['EPOCHS']):
            self.model.train()
            
            # Sample new collocation points for each epoch
            t_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['t'][1]
            x_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['x'][1]
            v_phy = (torch.rand(self.config['N_PHY'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            t_ic = torch.zeros(self.config['N_IC'], 1, device=self.device)
            x_ic = torch.rand(self.config['N_IC'], 1, device=self.device) * self.domain['x'][1]
            v_ic = (torch.rand(self.config['N_IC'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]

            # Optimization step
            self.optimizer.zero_grad()
            loss, loss_v, loss_p, loss_ic, loss_s = \
                self.compute_loss_with_symmetry(t_phy, x_phy, v_phy, t_ic, x_ic, v_ic)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}. Skipping.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Decay learning rate periodically
            if (epoch + 1) % 1000 == 0:
                self.scheduler.step()

            # Logging and visualization
            if (epoch + 1) % self.config['LOG_FREQUENCY'] == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch [{epoch+1}/{self.config['EPOCHS']}] | "
                    f"Loss: {loss.item():.4e} | L_v: {loss_v.item():.4e} | "
                    f"L_p: {loss_p.item():.4e} | L_ic: {loss_ic.item():.4e} | "
                    f"L_s: {loss_s.item():.4e} | Time: {elapsed_time:.2f}s"
                )
                
                log_data = (f"{epoch+1},{loss.item()},{loss_v.item()},{loss_p.item()},"
                            f"{loss_ic.item()},{loss_s.item()},{elapsed_time}\n")
                with open(self.log_file_path, 'a') as f:
                    f.write(log_data)
                
                self.writer.add_scalar('Loss/Total', loss.item(), epoch + 1)
                self.writer.add_scalar('Loss/Vlasov', loss_v.item(), epoch + 1)
                self.writer.add_scalar('Loss/Poisson', loss_p.item(), epoch + 1)
                self.writer.add_scalar('Loss/Initial_Condition', loss_ic.item(), epoch + 1)
                self.writer.add_scalar('Loss/Symmetry', loss_s.item(), epoch + 1)
            
            if (epoch + 1) % self.config['PLOT_FREQUENCY'] == 0:
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
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'results_epoch_{epoch}.png'))
        plt.close(fig)

    def plot_loss_history(self):
        """Plots the full loss history from the generated log file."""
        print("Plotting loss history...")
        try:
            log_data = np.loadtxt(self.log_file_path, delimiter=',', skiprows=1)
            plt.figure(figsize=(12, 8))
            plt.plot(log_data[:, 0], log_data[:, 1], 'k', label='Total Loss')
            plt.plot(log_data[:, 0], log_data[:, 2], 'r--', alpha=0.7, label='Vlasov Loss')
            plt.plot(log_data[:, 0], log_data[:, 3], 'b--', alpha=0.7, label='Poisson Loss')
            plt.plot(log_data[:, 0], log_data[:, 4], 'g--', alpha=0.7, label='IC Loss')
            plt.plot(log_data[:, 0], log_data[:, 5], 'm--', alpha=0.7, label='Symmetry Loss')
            plt.yscale('log'); plt.title('Loss History'); plt.xlabel('Epoch')
            plt.ylabel('Loss (log scale)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(self.config['PLOT_DIR'], 'loss_history.png'))
            plt.close()
            print("Loss history plot saved.")
        except Exception as e:
            print(f"Could not plot loss history: {e}")

if __name__ == '__main__':
    # ========================================================================
    #               MAIN CONFIGURATION BLOCK
    # ========================================================================
    # This dictionary centralizes all parameters for the simulation.
    # Modify these values to easily explore different physical scenarios
    # and training strategies.
    
    configuration = {
        # --- Domain Parameters ---
        'T_MAX': 15.0,                  # Simulation end time
        'X_MAX': 2 * np.pi / 0.5,       # Length of the spatial domain (one wavelength of the most unstable mode)
        'V_MAX': 6.0,                   # Max velocity in phase space (must be large enough to contain evolved particles)

        # --- Physics Parameters (Two-Stream Instability) ---
        'BEAM_V': 2.4,                  # Initial velocity of the two electron beams
        'THERMAL_V': 0.2,               # Thermal spread of the beams (lower value means "colder" beams)
        'PERTURB_AMP': 0.05,            # Amplitude of the initial density perturbation that seeds the instability

        # --- Neural Network Architecture ---
        'NN_LAYERS': 6,                 # Number of hidden layers in the MLP
        'NN_NEURONS': 64,               # Number of neurons per hidden layer

        # --- Training Hyperparameters ---
        'EPOCHS': 1000,                # Total number of training epochs
        'LEARNING_RATE': 2e-4,          # Initial learning rate for the Adam optimizer
        'N_PHY': 10000,                 # Number of collocation points for PDE residuals
        'N_IC': 2000,                   # Number of points for the initial condition

        # --- Loss Function Weights (Crucial for convergence) ---
        'LAMBDA_VLASOV': 1.0,           # Weight for the Vlasov equation residual
        'LAMBDA_POISSON': 1.0,          # Weight for the Poisson equation residual
        'LAMBDA_IC': 100.0,             # Weight for the initial condition (should be high)
        'LAMBDA_SYMM': 50.0,            # Weight for the velocity-symmetry constraint

        # --- Numerical & Logging Parameters ---
        'V_QUAD_POINTS': 128,           # Number of points for numerical integration over velocity
        'LOG_FREQUENCY': 100,           # How often to print logs and save to TensorBoard (in epochs)
        'PLOT_FREQUENCY': 1000,         # How often to generate and save plots (in epochs)
        'PLOT_DIR': 'vlasov_poisson_final_prototype' # Directory to save all results
    }
    
    # Instantiate the solver and start the training
    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()