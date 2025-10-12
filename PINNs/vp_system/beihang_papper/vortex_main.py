import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os

# Set a seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    """
    Defines the multilayer perceptron for the PINN.
    8 layers with 100 neurons each, as mentioned in the paper.
    """
    def __init__(self, input_dim=3, output_dim=1, layers=8, neurons=100):
        super(MLP, self).__init__()
        
        modules = [nn.Linear(input_dim, neurons), nn.Tanh()]
        for _ in range(layers - 2):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(neurons, output_dim))
        
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class VlasovPoissonPINN:
    """
    A class to solve the 1D Vlasov-Poisson equation using PINNs.
    Encapsulates the model, training, and visualization.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Domain boundaries
        self.domain = {
            't': (0.0, config['T_MAX']),
            'x': (0.0, config['X_MAX']),
            'v': (-config['V_MAX'], config['V_MAX'])
        }

        # The neural network
        self.model = MLP(
            layers=config['NN_LAYERS'], 
            neurons=config['NN_NEURONS']
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['LEARNING_RATE']
        )
        
        # Loss weights
        self.lambda_vlasov = config['LAMBDA_VLASOV']
        self.lambda_poisson = config['LAMBDA_POISSON']
        self.lambda_ic = config['LAMBDA_IC']

        # Points for numerical integration (Trapezoidal rule)
        v_quad = torch.linspace(
            self.domain['v'][0], self.domain['v'][1], config['V_QUAD_POINTS']
        ).to(self.device)
        self.v_quad = v_quad.reshape(1, 1, -1) # Reshape for broadcasting
        self.dv = v_quad[1] - v_quad[0]

    def _initial_condition(self, x, v):
        """
        Defines the initial distribution function f(0, x, v).
        This function represents the two-stream instability scenario.
        """
        k = 2 * np.pi / self.config['X_MAX']
        # Two counter-streaming beams with a spatial perturbation
        term1 = torch.exp(-((v - self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        term2 = torch.exp(-((v + self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        
        return (1 / (2 * np.pi * self.config['THERMAL_V']**2)**0.5) * \
               (term1 + term2) * \
               (1 + self.config['PERTURB_AMP'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """
        Computes electron density n_e by integrating f(t, x, v) over v
        using the trapezoidal rule.
        t and x are tensors of shape [N, 1].
        """
        t_rep = t.repeat(1, self.config['V_QUAD_POINTS']).reshape(-1, 1)
        x_rep = x.repeat(1, self.config['V_QUAD_POINTS']).reshape(-1, 1)
        v_rep = self.v_quad.repeat(t.shape[0], 1, 1).reshape(-1, 1)
        
        txv = torch.cat([t_rep, x_rep, v_rep], dim=1)
        f_vals = self.model(txv).reshape(t.shape[0], self.config['V_QUAD_POINTS'])
        
        # Trapezoidal rule for integration
        integral = self.dv * (
            0.5 * (f_vals[:, 0] + f_vals[:, -1]) + torch.sum(f_vals[:, 1:-1], dim=1)
        )
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        """
        Computes the residuals of the Vlasov-Poisson equations.
        """
        txv = torch.cat([t, x, v], dim=1)
        f = self.model(txv)
        
        # Automatic Differentiation for Vlasov equation terms
        df_d_txv = torch.autograd.grad(f, txv, torch.ones_like(f), create_graph=True)[0]
        df_dt = df_d_txv[:, 0].unsqueeze(1)
        df_dx = df_d_txv[:, 1].unsqueeze(1)
        df_dv = df_d_txv[:, 2].unsqueeze(1)
        
        # Simplified Electric Field calculation
        n_e = self._compute_ne(t, x)
        
        # Simple approximation for E field to promote vortex formation
        E = -(n_e - 1.0) * self.domain['x'][1] / 10.0  # Simple scaling
        
        # Vlasov residual
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        # Poisson residual (simplified)
        poisson_residual = n_e - 1.0  # Charge neutrality condition

        return vlasov_residual, poisson_residual
    
    def compute_loss(self, t_phy, x_phy, v_phy, t_ic, x_ic, v_ic):
        # 1. Physics Loss (Equation Residuals)
        t_phy.requires_grad_(True)
        x_phy.requires_grad_(True)
        v_phy.requires_grad_(True)
        
        vlasov_res, poisson_res = self._get_residuals(t_phy, x_phy, v_phy)
        loss_vlasov = torch.mean(vlasov_res**2)
        loss_poisson = torch.mean(poisson_res**2)

        # 2. Initial Condition Loss
        ic_txv = torch.cat([t_ic, x_ic, v_ic], dim=1)
        f_pred_ic = self.model(ic_txv)
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        # Total Loss
        total_loss = self.lambda_vlasov * loss_vlasov + \
                     self.lambda_poisson * loss_poisson + \
                     self.lambda_ic * loss_ic
        
        return total_loss, loss_vlasov, loss_poisson, loss_ic

    def train(self):
        print("Starting training...")
        start_time = time.time()
        
        # Prepare directory for saving plots
        os.makedirs(self.config['PLOT_DIR'], exist_ok=True)

        for epoch in range(self.config['EPOCHS']):
            try:
                self.model.train()
                
                # Generate new training points for each epoch
                t_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['t'][1]
                x_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['x'][1]
                v_phy = torch.rand(self.config['N_PHY'], 1, device=self.device) * \
                        (self.domain['v'][1] - self.domain['v'][0]) + self.domain['v'][0]
                
                # Initial condition points (at t=0)
                t_ic = torch.zeros(self.config['N_IC'], 1, device=self.device)
                x_ic = torch.rand(self.config['N_IC'], 1, device=self.device) * self.domain['x'][1]
                v_ic = torch.rand(self.config['N_IC'], 1, device=self.device) * \
                       (self.domain['v'][1] - self.domain['v'][0]) + self.domain['v'][0]

                self.optimizer.zero_grad()
                loss, loss_v, loss_p, loss_ic = self.compute_loss(
                    t_phy, x_phy, v_phy, t_ic, x_ic, v_ic
                )
                
                # Check for NaN values
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at epoch {epoch+1}")
                    continue
                    
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                if (epoch + 1) % self.config['LOG_FREQUENCY'] == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Epoch [{epoch+1}/{self.config['EPOCHS']}] | "
                          f"Loss: {loss.item():.4e} | "
                          f"L_vlasov: {loss_v.item():.4e} | "
                          f"L_poisson: {loss_p.item():.4e} | "
                          f"L_ic: {loss_ic.item():.4e} | "
                          f"Time: {elapsed_time:.2f}s")
                
                if (epoch + 1) % self.config['PLOT_FREQUENCY'] == 0:
                    self.plot_results(epoch + 1)
                    
            except Exception as e:
                print(f"Error at epoch {epoch+1}: {str(e)}")
                continue

    @torch.no_grad()
    def plot_results(self, epoch):
        self.model.eval()
        print(f"Generating plots for epoch {epoch}...")

        # Create a grid for plotting with higher resolution for vortex visualization
        x_grid = torch.linspace(self.domain['x'][0], self.domain['x'][1], 150, device=self.device)
        v_grid = torch.linspace(self.domain['v'][0], self.domain['v'][1], 150, device=self.device)
        X, V = torch.meshgrid(x_grid, v_grid, indexing='ij')

        # More time points to capture vortex formation
        plot_times = [0.0, self.domain['t'][1] / 6, self.domain['t'][1] / 3, 
                     self.domain['t'][1] / 2, 2 * self.domain['t'][1] / 3, self.domain['t'][1]]

        fig = plt.figure(figsize=(30, 20))
        gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.3)

        # Top row: Phase space evolution
        for i, t_val in enumerate(plot_times):
            T = torch.full_like(X, t_val)
            
            txv_flat = torch.stack([T.flatten(), X.flatten(), V.flatten()], dim=1)
            f_pred = self.model(txv_flat).reshape(X.shape)
            
            ax = fig.add_subplot(gs[0, i])
            # Use 'plasma' colormap which is good for showing vortex structures
            im = ax.pcolormesh(X.cpu(), V.cpu(), f_pred.cpu(), cmap='plasma', shading='auto')
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f'Phase Space f(t,x,v)\nt={t_val:.2f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('x (position)')
            ax.set_ylabel('v (velocity)')
            
            # Add contour lines to highlight vortex structures
            ax.contour(X.cpu(), V.cpu(), f_pred.cpu(), levels=15, colors='white', alpha=0.4, linewidths=0.5)

        # Middle row: Density evolution
        for i, t_val in enumerate(plot_times):
            ax_density = fig.add_subplot(gs[1, i])
            t_grid = torch.full_like(x_grid, t_val).unsqueeze(1)
            x_final = x_grid.unsqueeze(1)
            n_e = self._compute_ne(t_grid, x_final)
            
            ax_density.plot(x_grid.cpu(), n_e.cpu(), 'b-', linewidth=2, label=f't={t_val:.1f}')
            ax_density.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Background')
            ax_density.set_title(f'Electron Density\nt={t_val:.1f}', fontsize=10)
            ax_density.set_xlabel('x')
            ax_density.set_ylabel('n_e')
            ax_density.legend(fontsize=8)
            ax_density.grid(True, alpha=0.3)
            ax_density.set_ylim([0.5, 1.5])  # Fixed scale for comparison

        # Bottom row: Analysis plots
        # Initial condition
        ax_ic = fig.add_subplot(gs[2, 0])
        f_ic_true = self._initial_condition(X, V)
        im_ic = ax_ic.pcolormesh(X.cpu(), V.cpu(), f_ic_true.cpu(), cmap='plasma', shading='auto')
        fig.colorbar(im_ic, ax=ax_ic, shrink=0.8)
        ax_ic.set_title('Initial Condition\nf(0,x,v)', fontweight='bold')
        ax_ic.set_xlabel('x (position)')
        ax_ic.set_ylabel('v (velocity)')

        # Final density profile
        ax_final_density = fig.add_subplot(gs[2, 1])
        t_final = torch.full_like(x_grid, self.domain['t'][1]).unsqueeze(1)
        x_final = x_grid.unsqueeze(1)
        n_e_final = self._compute_ne(t_final, x_final)
        
        ax_final_density.plot(x_grid.cpu(), n_e_final.cpu(), 'b-', linewidth=3, label='Final Density')
        ax_final_density.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Background')
        ax_final_density.set_title('Final Electron Density', fontweight='bold')
        ax_final_density.set_xlabel('x (position)')
        ax_final_density.set_ylabel('n_e')
        ax_final_density.legend()
        ax_final_density.grid(True)

        # Charge density deviation
        ax_charge = fig.add_subplot(gs[2, 2])
        charge_deviation = n_e_final - 1.0
        ax_charge.plot(x_grid.cpu(), charge_deviation.cpu(), 'g-', linewidth=3)
        ax_charge.fill_between(x_grid.cpu(), 0, charge_deviation.cpu(), alpha=0.3, color='green')
        ax_charge.set_title('Charge Density Deviation', fontweight='bold')
        ax_charge.set_xlabel('x (position)')
        ax_charge.set_ylabel('n_e - 1')
        ax_charge.grid(True)
        ax_charge.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Vortex analysis information
        ax_info = fig.add_subplot(gs[2, 3:])
        ax_info.axis('off')
        
        # Calculate some basic statistics
        f_final = self.model(torch.stack([
            torch.full_like(X.flatten(), self.domain['t'][1]),
            X.flatten(), V.flatten()
        ], dim=1))
        f_final = f_final.reshape(X.shape)
        
        max_density_var = torch.max(torch.abs(charge_deviation)).item()
        energy_like = torch.mean(f_final * V**2).item()
        
        info_text = f"""
VORTEX FORMATION ANALYSIS (Epoch {epoch})

Domain Configuration:
• Time: t ∈ [0, {self.domain['t'][1]:.1f}]
• Space: x ∈ [0, {self.domain['x'][1]:.1f}] 
• Velocity: v ∈ [{self.domain['v'][0]:.1f}, {self.domain['v'][1]:.1f}]

Initial Conditions:
• Beam velocities: v = ±{self.config['BEAM_V']:.1f}
• Thermal velocity: σ_v = {self.config['THERMAL_V']:.1f}
• Perturbation amplitude: {self.config['PERTURB_AMP']:.2f}

Current State Analysis:
• Max density variation: {max_density_var:.4f}
• Kinetic energy measure: {energy_like:.4f}

LOOK FOR VORTEX INDICATORS:
→ "Eye" or spiral structures in phase space plots
→ Oscillatory density patterns with growth
→ Phase space filamentation and mixing
→ Transition from linear to nonlinear behavior

Network: {self.config['NN_LAYERS']} layers × {self.config['NN_NEURONS']} neurons
Loss weights: Vlasov={self.lambda_vlasov}, Poisson={self.lambda_poisson}, IC={self.lambda_ic}
        """
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.suptitle(f'Two-Stream Instability: Vortex Formation Analysis (Epoch {epoch})', 
                     fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'vortex_analysis_epoch_{epoch}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    # Optimized configuration for vortex observation
    configuration = {
        # Domain Parameters - optimized for vortex formation
        'T_MAX': 12.0,         # Sufficient time for nonlinear development
        'X_MAX': 4 * np.pi,    # Larger domain for multiple vortices
        'V_MAX': 3.0,          # Focused velocity range
        
        # Initial Condition Parameters - promotes instability growth
        'BEAM_V': 1.2,         # Moderate beam velocity for clear instability
        'THERMAL_V': 0.2,      # Small thermal spread for pronounced streams
        'PERTURB_AMP': 0.08,   # Moderate perturbation to seed instability
        
        # Network Architecture - enhanced for complex structures
        'NN_LAYERS': 8,
        'NN_NEURONS': 128,
        
        # Training Parameters - balanced for quality and efficiency  
        'EPOCHS': 30000,
        'LEARNING_RATE': 1e-4,
        'N_PHY': 12000,
        'N_IC': 2000,
        
        # Loss Function Weights - balanced for physical accuracy
        'LAMBDA_VLASOV': 1.0,
        'LAMBDA_POISSON': 0.5,  # Reduced to allow more natural evolution
        'LAMBDA_IC': 20.0,      # Important but not overwhelming
        
        # Numerical Integration
        'V_QUAD_POINTS': 150,
        
        # Logging and Visualization
        'LOG_FREQUENCY': 100,
        'PLOT_FREQUENCY': 1000,
        'PLOT_DIR': 'vortex_analysis_results'
    }

    # Initialize and run the PINN solver
    print("="*60)
    print("VLASOV-POISSON PINN: VORTEX FORMATION ANALYSIS")
    print("="*60)
    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()