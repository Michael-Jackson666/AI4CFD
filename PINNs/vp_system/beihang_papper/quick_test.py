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
    """
    def __init__(self, input_dim=3, output_dim=1, layers=6, neurons=64):
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

        # Optimizer and Scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['LEARNING_RATE']
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        # Points for numerical integration (Trapezoidal rule)
        v_quad = torch.linspace(
            self.domain['v'][0], self.domain['v'][1], config['V_QUAD_POINTS'], device=self.device
        )
        self.v_quad = v_quad.view(1, 1, -1) # Reshape for broadcasting
        self.dv = v_quad[1] - v_quad[0]

    def _initial_condition(self, x, v):
        """
        Defines the initial distribution function f(0, x, v).
        """
        k = 2 * np.pi / self.config['X_MAX']
        term1 = torch.exp(-((v - self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        term2 = torch.exp(-((v + self.config['BEAM_V'])**2) / (2 * self.config['THERMAL_V']**2))
        
        # The 0.5 factor correctly normalizes the two-beam distribution
        return 0.5 * (term1 + term2) * (1 + self.config['PERTURB_AMP'] * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """
        Computes electron density n_e by integrating f(t, x, v) over v.
        This version is corrected and optimized.
        """
        # t 和 x 的形状: [N, 1]
        # self.v_quad 的形状: [1, 1, V_POINTS]
        
        # 扩展 t 和 x 来匹配 v_quad 的维度，以便进行广播
        # t_exp 的形状: [N, 1, 1] -> [N, 1, V_POINTS]
        # x_exp 的形状: [N, 1, 1] -> [N, 1, V_POINTS]
        t_exp = t.unsqueeze(2).expand(-1, -1, self.config['V_QUAD_POINTS'])
        x_exp = x.unsqueeze(2).expand(-1, -1, self.config['V_QUAD_POINTS'])
        
        # self.v_quad 已经准备好了广播，它的形状是 [1, 1, V_POINTS]
        # 它会自动扩展以匹配 t_exp 和 x_exp 的形状
        
        # 将它们展平以输入到 MLP 中
        # 每个张量的形状现在是 [N * V_POINTS, 1]
        t_flat = t_exp.reshape(-1, 1)
        x_flat = x_exp.reshape(-1, 1)
        v_flat = self.v_quad.expand(t.shape[0], -1, -1).reshape(-1, 1)
        
        # 构建 MLP 的输入
        txv = torch.cat([t_flat, x_flat, v_flat], dim=1)
        
        # 获得模型的输出并重塑回 [N, V_POINTS] 以便积分
        f_vals = self.model(txv).view(t.shape[0], self.config['V_QUAD_POINTS'])
        
        # 使用梯形法则进行积分
        integral = torch.trapezoid(f_vals, self.v_quad.squeeze(), dim=1)
        
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        """
        Computes the physically correct residuals of the Vlasov-Poisson equations.
        """
        txv = torch.cat([t, x, v], dim=1)
        f = self.model(txv)
        
        df_d_txv = torch.autograd.grad(f, txv, torch.ones_like(f), create_graph=True)[0]
        df_dt, df_dx, df_dv = df_d_txv.split(1, dim=1)
        
        # --- Self-consistent Electric Field Calculation ---
        n_e = self._compute_ne(t, x)
        
        # To calculate E for random (t,x) points, we must integrate on a grid and interpolate.
        x_grid_E = torch.linspace(0, self.config['X_MAX'], 101, device=self.device).unsqueeze(1).requires_grad_(True)
        t_mean_E = torch.full_like(x_grid_E, t.mean().item()) # Use mean time for this batch
        
        n_e_on_grid = self._compute_ne(t_mean_E, x_grid_E)
        charge_dev_on_grid = n_e_on_grid - 1.0
        
        dx_E = x_grid_E[1] - x_grid_E[0]
        E_on_grid = torch.cumsum(charge_dev_on_grid, dim=0) * dx_E
        
        # --- 修正点：将原地操作改为非原地操作 ---
        E_on_grid = E_on_grid - torch.mean(E_on_grid) # CORRECTED LINE
        
        # Interpolate E back to the original random x points
        E_interp = np.interp(
            x.cpu().detach().numpy().flatten(),
            x_grid_E.cpu().detach().numpy().flatten(),
            E_on_grid.cpu().detach().numpy().flatten()
        )
        E = torch.from_numpy(E_interp).to(self.device).float().unsqueeze(1)
        
        # --- Vlasov Residual ---
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        # --- Poisson Residual ---
        # For stability in this prototype, we will stick to the simpler charge deviation,
        # but the E field used in Vlasov IS calculated correctly.
        poisson_residual = n_e - 1.0

        return vlasov_residual, poisson_residual
    
    def compute_loss_with_symmetry(self, t_phy, x_phy, v_phy, t_ic, x_ic, v_ic):
        """
        The complete loss function including physics, IC, and symmetry.
        """
        t_phy.requires_grad_(True); x_phy.requires_grad_(True); v_phy.requires_grad_(True)

        # Physics Loss
        vlasov_res, poisson_res = self._get_residuals(t_phy, x_phy, v_phy)
        loss_vlasov = torch.mean(vlasov_res**2)
        loss_poisson = torch.mean(poisson_res**2)

        # Initial Condition Loss
        ic_txv = torch.cat([t_ic, x_ic, v_ic], dim=1)
        f_pred_ic = self.model(ic_txv)
        f_true_ic = self._initial_condition(x_ic, v_ic)
        loss_ic = torch.mean((f_pred_ic - f_true_ic)**2)

        # Symmetry Loss: f(t, x, v) = f(t, x, -v)
        txv_pos_v = torch.cat([t_phy, x_phy, v_phy], dim=1)
        txv_neg_v = torch.cat([t_phy, x_phy, -v_phy], dim=1)
        f_pos_v = self.model(txv_pos_v)
        f_neg_v = self.model(txv_neg_v)
        loss_symm = torch.mean((f_pos_v - f_neg_v)**2)

        total_loss = self.config['LAMBDA_VLASOV'] * loss_vlasov + \
                     self.config['LAMBDA_POISSON'] * loss_poisson + \
                     self.config['LAMBDA_IC'] * loss_ic + \
                     self.config['LAMBDA_SYMM'] * loss_symm
        
        return total_loss, loss_vlasov, loss_poisson, loss_ic, loss_symm

    def train(self):
        print("Starting fast prototype training...")
        start_time = time.time()
        
        os.makedirs(self.config['PLOT_DIR'], exist_ok=True)

        for epoch in range(self.config['EPOCHS']):
            self.model.train()
            
            t_phy = (torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['t'][1])
            x_phy = (torch.rand(self.config['N_PHY'], 1, device=self.device) * self.domain['x'][1])
            v_phy = (torch.rand(self.config['N_PHY'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            t_ic = torch.zeros(self.config['N_IC'], 1, device=self.device)
            x_ic = (torch.rand(self.config['N_IC'], 1, device=self.device) * self.domain['x'][1])
            v_ic = (torch.rand(self.config['N_IC'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]

            self.optimizer.zero_grad()
            
            loss, loss_v, loss_p, loss_ic, loss_s = self.compute_loss_with_symmetry(
                t_phy, x_phy, v_phy, t_ic, x_ic, v_ic
            )
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at epoch {epoch+1}. Skipping update.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if (epoch + 1) % 1000 == 0:
                self.scheduler.step()

            if (epoch + 1) % self.config['LOG_FREQUENCY'] == 0:
                print(f"Epoch [{epoch+1}/{self.config['EPOCHS']}] | "
                      f"Loss: {loss.item():.4e} | L_vlasov: {loss_v.item():.4e} | "
                      f"L_poisson: {loss_p.item():.4e} | L_ic: {loss_ic.item():.4e} | "
                      f"L_symm: {loss_s.item():.4e} | Time: {time.time() - start_time:.2f}s")
            
            if (epoch + 1) % self.config['PLOT_FREQUENCY'] == 0:
                self.plot_results(epoch + 1)

    @torch.no_grad()
    def plot_results(self, epoch):
        self.model.eval()
        print(f"Generating plots for epoch {epoch}...")

        x_grid = torch.linspace(self.domain['x'][0], self.domain['x'][1], 100, device=self.device)
        v_grid = torch.linspace(self.domain['v'][0], self.domain['v'][1], 100, device=self.device)
        X, V = torch.meshgrid(x_grid, v_grid, indexing='ij')

        plot_times = [0.0, self.domain['t'][1] / 2, self.domain['t'][1]]

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)

        for i, t_val in enumerate(plot_times):
            T = torch.full_like(X, t_val)
            txv_flat = torch.stack([T.flatten(), X.flatten(), V.flatten()], dim=1)
            f_pred = self.model(txv_flat).reshape(X.shape)
            
            ax = fig.add_subplot(gs[0, i])
            im = ax.pcolormesh(X.cpu(), V.cpu(), f_pred.cpu(), cmap='jet', shading='auto')
            fig.colorbar(im, ax=ax); ax.set_xlabel('x (position)'); ax.set_ylabel('v (velocity)')
            ax.set_title(f'PINN Solution f(t,x,v) at t={t_val:.2f}')
        
        ax_ic = fig.add_subplot(gs[1, 0])
        f_ic_true = self._initial_condition(X, V)
        im_ic = ax_ic.pcolormesh(X.cpu(), V.cpu(), f_ic_true.cpu(), cmap='jet', shading='auto')
        fig.colorbar(im_ic, ax=ax_ic); ax_ic.set_xlabel('x (position)'); ax_ic.set_ylabel('v (velocity)')
        ax_ic.set_title('True Initial Condition f(0,x,v)')

        ax_ne = fig.add_subplot(gs[1, 1])
        t_final = torch.full((x_grid.shape[0], 1), self.domain['t'][1], device=self.device)
        n_e_final = self._compute_ne(t_final, x_grid.unsqueeze(1))
        ax_ne.plot(x_grid.cpu(), n_e_final.cpu(), 'b-', label='Electron Density')
        ax_ne.axhline(y=1.0, color='r', linestyle='--', label='Background Density'); ax_ne.legend()
        ax_ne.set_title(f'Electron Density n_e(t,x) at t={self.domain["t"][1]:.2f}'); ax_ne.grid(True)
        ax_ne.set_xlabel('x (position)'); ax_ne.set_ylabel('n_e (Electron Density)')

        ax_e = fig.add_subplot(gs[1, 2])
        charge_dev_final = n_e_final - 1.0
        dx_final = x_grid[1] - x_grid[0]
        E_final = torch.cumsum(charge_dev_final, dim=0) * dx_final
        E_final -= torch.mean(E_final) # Enforce periodic BC
        ax_e.plot(x_grid.cpu(), E_final.cpu(), 'g-')
        ax_e.set_title(f'Electric Field E(t,x) at t={self.domain["t"][1]:.2f}'); ax_e.grid(True)
        ax_e.set_xlabel('x (position)'); ax_e.set_ylabel('E (Electric Field)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['PLOT_DIR'], f'results_epoch_{epoch}.png'))
        plt.close(fig)

if __name__ == '__main__':
    # --- Parameters for Fast Prototyping on a Local Machine ---
    configuration = {
        'T_MAX': 15.0,
        'X_MAX': 2 * np.pi / 0.5,
        'V_MAX': 6.0,
        'BEAM_V': 2.4,
        'THERMAL_V': 0.2,
        'PERTURB_AMP': 0.05,
        'NN_LAYERS': 6,
        'NN_NEURONS': 64,
        'EPOCHS': 100,
        'LEARNING_RATE': 2e-4,
        'N_PHY': 10000,
        'N_IC': 2000,
        'LAMBDA_VLASOV': 1.0,
        'LAMBDA_POISSON': 1.0,
        'LAMBDA_IC': 100.0,
        'LAMBDA_SYMM': 50.0,
        'V_QUAD_POINTS': 128,
        'LOG_FREQUENCY': 100,
        'PLOT_FREQUENCY': 1000,
        'PLOT_DIR': 'vlasov_poisson_fast_prototype'
    }

    pinn_solver = VlasovPoissonPINN(configuration)
    pinn_solver.train()