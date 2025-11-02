"""
VlasovPoissonPINN: Physics-Informed Neural Network for solving 
the 1D Vlasov-Poisson system with input normalization.
"""

import torch
import os
import time
from mlp import MLP
from transformer import TransformerPINN, HybridTransformerPINN, LightweightTransformerPINN


class VlasovPoissonPINN:
    """
    A comprehensive PINN solver for the 1D Vlasov-Poisson system.
    Includes input normalization to improve training stability.
    """
    
    def __init__(self, config):
        """
        Initialize the PINN solver.
        
        Args:
            config (dict): Configuration dictionary containing all hyperparameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Define physical domain boundaries
        self.domain = {
            't': (0.0, config['t_max']),
            'x': (0.0, config['x_max']),
            'v': (-config['v_max'], config['v_max'])
        }
        
        # Setup normalization parameters: map domain to [-1, 1]
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

        # Initialize neural network model based on architecture choice
        model_type = config.get('model_type', 'mlp').lower()
        print(f"\nInitializing model: {model_type}")
        
        if model_type == 'mlp':
            self.model = MLP(
                input_dim=3,
                output_dim=1,
                layers=config['nn_layers'], 
                neurons=config['nn_neurons']
            ).to(self.device)
        elif model_type == 'transformer':
            self.model = TransformerPINN(
                input_dim=3,
                output_dim=1,
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_transformer_layers', 6),
                dim_feedforward=config.get('dim_feedforward', 1024),
                dropout=config.get('dropout', 0.1)
            ).to(self.device)
        elif model_type == 'hybrid_transformer':
            self.model = HybridTransformerPINN(
                input_dim=3,
                output_dim=1,
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_transformer_layers=config.get('num_transformer_layers', 4),
                num_mlp_layers=config.get('num_mlp_layers', 4),
                mlp_neurons=config.get('mlp_neurons', 512),
                dropout=config.get('dropout', 0.1)
            ).to(self.device)
        elif model_type == 'lightweight_transformer':
            self.model = LightweightTransformerPINN(
                input_dim=3,
                output_dim=1,
                d_model=config.get('d_model', 128),
                nhead=config.get('nhead', 4),
                num_layers=config.get('num_transformer_layers', 3),
                dim_feedforward=config.get('dim_feedforward', 512),
                dropout=config.get('dropout', 0.1)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose from 'mlp', 'transformer', 'hybrid_transformer', 'lightweight_transformer'")
        
        # Print model statistics
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model architecture: {model_type}")
        print(f"Total trainable parameters: {total_params:,}")
        
        # Setup optimizer and learning rate scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'], 
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.99
        )
        
        # Setup quadrature points for velocity integration
        v_quad = torch.linspace(
            -config['v_max'], config['v_max'],
            config['v_quad_points'], device=self.device
        )
        self.v_quad = v_quad.view(1, 1, -1)

        # Setup logging and visualization
        os.makedirs(self.config['plot_dir'], exist_ok=True)
        self.log_file_path = os.path.join(self.config['plot_dir'], 'training_log.txt')
    
    # ==================== Normalization Methods ====================
    
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

    # ==================== Physics Methods ====================

    def _initial_condition(self, x, v):
        """
        Initial condition dispatcher based on configuration.
        Supports multiple initial condition types for easy experimentation.
        
        Args:
            x (torch.Tensor): Spatial coordinates
            v (torch.Tensor): Velocity coordinates
            
        Returns:
            torch.Tensor: Initial distribution function f(0, x, v)
        """
        ic_type = self.config.get('type', 'two_stream')
        
        if ic_type == 'two_stream':
            return self._ic_two_stream(x, v)
        elif ic_type == 'landau':
            return self._ic_landau(x, v)
        elif ic_type == 'single_beam':
            return self._ic_single_beam(x, v)
        elif ic_type == 'custom':
            custom_func = self.config.get('custom_ic_function')
            if custom_func is None:
                raise ValueError("Custom IC function not provided in config!")
            return custom_func(x, v, self.config)
        else:
            raise ValueError(f"Unknown initial condition type: {ic_type}")
    
    def _ic_two_stream(self, x, v):
        """
        Two-stream instability initial condition.
        Two counter-streaming electron beams with spatial perturbation.
        
        Args:
            x (torch.Tensor): Spatial coordinates
            v (torch.Tensor): Velocity coordinates
            
        Returns:
            torch.Tensor: f(0, x, v) = 0.5 * [M(v-v_b) + M(v+v_b)] * [1 + α*cos(kx)]
        """
        k = 2 * torch.pi * self.config.get('perturb_mode', 1) / self.config['x_max']
        v_th = self.config.get('thermal_v', 0.5)
        v_b = self.config.get('beam_v', 1.0)
        alpha = self.config.get('perturb_amp', 0.1)
        
        norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
        
        # Two Gaussian beams moving in opposite directions
        term1 = norm_factor * torch.exp(-((v - v_b)**2) / (2 * v_th**2))
        term2 = norm_factor * torch.exp(-((v + v_b)**2) / (2 * v_th**2))
        
        # Add spatial perturbation
        return 0.5 * (term1 + term2) * (1 + alpha * torch.cos(k * x))
    
    def _ic_landau(self, x, v):
        """
        Landau damping initial condition.
        Single Maxwellian with small spatial perturbation.
        
        Args:
            x (torch.Tensor): Spatial coordinates
            v (torch.Tensor): Velocity coordinates
            
        Returns:
            torch.Tensor: f(0, x, v) = M(v) * [1 + α*cos(kx)]
        """
        k = 2 * torch.pi * self.config.get('landau_mode', 1) / self.config['x_max']
        v_th = self.config.get('landau_v_thermal', 1.0)
        alpha = self.config.get('landau_perturb_amp', 0.01)
        
        # Single Maxwellian centered at v=0
        norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
        maxwell = norm_factor * torch.exp(-(v**2) / (2 * v_th**2))
        
        # Small spatial perturbation for Landau damping
        return maxwell * (1 + alpha * torch.cos(k * x))
    
    def _ic_single_beam(self, x, v):
        """
        Single beam initial condition.
        One Maxwellian beam with spatial perturbation.
        
        Args:
            x (torch.Tensor): Spatial coordinates
            v (torch.Tensor): Velocity coordinates
            
        Returns:
            torch.Tensor: f(0, x, v) = M(v-v_c) * [1 + α*cos(kx)]
        """
        k = 2 * torch.pi * self.config.get('single_mode', 1) / self.config['x_max']
        v_c = self.config.get('single_v_center', 0.0)
        v_th = self.config.get('single_v_thermal', 0.5)
        alpha = self.config.get('single_perturb_amp', 0.05)
        
        # Single Maxwellian beam
        norm_factor = 1.0 / (v_th * torch.sqrt(torch.tensor(2 * torch.pi)))
        maxwell = norm_factor * torch.exp(-((v - v_c)**2) / (2 * v_th**2))
        
        # Add spatial perturbation
        return maxwell * (1 + alpha * torch.cos(k * x))

    def _compute_ne(self, t, x):
        """
        Computes electron density n_e(t,x) by integrating f over v.
        Uses trapezoidal rule for numerical integration.
        
        Args:
            t (torch.Tensor): Time coordinates [N, 1]
            x (torch.Tensor): Spatial coordinates [N, 1]
            
        Returns:
            torch.Tensor: Electron density [N, 1]
        """
        # Expand to create grid for all v values
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
        
        # Integrate over velocity using trapezoidal rule
        integral = torch.trapezoid(f_vals, self.v_quad.squeeze(), dim=1)
        return integral.unsqueeze(1)

    def _get_residuals(self, t, x, v):
        """
        Calculates the residuals for the Vlasov and Poisson equations.
        Inputs are in physical domain and will be normalized internally.
        
        Args:
            t, x, v (torch.Tensor): Physical coordinates requiring gradients
            
        Returns:
            tuple: (vlasov_residual, poisson_residual_on_grid)
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
        # df/dt_physical = df/dt_normalized * (dt_normalized/dt_physical) = df/dt_norm / t_scale
        df_dt = df_dt_norm / self.t_scale
        df_dx = df_dx_norm / self.x_scale
        df_dv = df_dv_norm / self.v_scale
        
        # Compute electric field on a spatial grid
        x_grid_E = torch.linspace(0, self.config['x_max'], 101, device=self.device).unsqueeze(1).requires_grad_()
        t_mean_E = torch.full_like(x_grid_E, t.mean().item())
        n_e_on_grid = self._compute_ne(t_mean_E, x_grid_E)
        charge_dev_on_grid = n_e_on_grid - 1.0
        
        # Integrate charge deviation to get electric field: dE/dx = n_e - 1
        dx_E = x_grid_E[1] - x_grid_E[0]
        E_on_grid = torch.cumsum(charge_dev_on_grid, dim=0) * dx_E
        E_on_grid = E_on_grid - torch.mean(E_on_grid)  # Remove mean for periodic BC
        
        # Interpolate E to sample points
        x_flat = x.flatten()
        x_grid_flat = x_grid_E.flatten()
        E_grid_flat = E_on_grid.flatten()
        
        # Linear interpolation
        indices = torch.searchsorted(x_grid_flat.detach(), x_flat.detach())
        indices = torch.clamp(indices, 1, len(x_grid_flat) - 1)
        
        x0 = x_grid_flat[indices - 1]
        x1 = x_grid_flat[indices]
        y0 = E_grid_flat[indices - 1]
        y1 = E_grid_flat[indices]
        
        E = y0 + (x_flat - x0) * (y1 - y0) / (x1 - x0 + 1e-10)
        E = E.unsqueeze(1)
        
        # Vlasov residual: df/dt + v*df/dx - E*df/dv = 0
        vlasov_residual = df_dt + v * df_dx - E * df_dv
        
        # Poisson residual: dE/dx - (n_e - 1) = 0
        dE_dx_on_grid = torch.autograd.grad(E_on_grid, x_grid_E, torch.ones_like(E_on_grid), create_graph=True)[0]
        poisson_residual_on_grid = dE_dx_on_grid - charge_dev_on_grid

        return vlasov_residual, poisson_residual_on_grid
    
    # ==================== Loss Computation ====================
    
    def compute_loss(self, t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc):
        """
        Calculates the classic PINN loss with three components:
        1. PDE Loss (governing equations)
        2. Initial Condition Loss
        3. Boundary Condition Loss
        
        Args:
            t_pde, x_pde, v_pde: Collocation points for PDE residuals
            t_ic, x_ic, v_ic: Points for initial condition
            t_bc, x_bc, v_bc: Points for boundary conditions
            
        Returns:
            tuple: (total_loss, loss_pde, loss_ic, loss_bc)
        """
        # --- 1. PDE Loss (Governing Equations) ---
        t_pde.requires_grad_(True)
        x_pde.requires_grad_(True)
        v_pde.requires_grad_(True)
        
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
        # Velocity boundary: f(t, x, v_min/v_max) ≈ 0
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
        loss_bc = torch.mean(f_bc_vmin**2) + torch.mean(f_bc_vmax**2)

        # --- Total Loss ---
        # Weighted average of losses using user-specified weights
        w_pde = self.config.get('weight_pde', 1.0)
        w_ic = self.config.get('weight_ic', 1.0)
        w_bc = self.config.get('weight_bc', 1.0)
        sum_w = w_pde + w_ic + w_bc
        if sum_w == 0:
            raise ValueError("Sum of loss weights is zero; please provide positive weights.")
        total_loss = (w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc) / sum_w
        
        return total_loss, loss_pde, loss_ic, loss_bc

    # ==================== Training Loop ====================
    
    def train(self):
        """
        Main training loop using the classic 3-component loss.
        Includes logging, checkpointing, and periodic visualization.
        """
        print("Starting training with classic PDE, IC, BC loss...")
        print(f"Total epochs: {self.config['epochs']}")
        print(f"Sample sizes: PDE={self.config['n_pde']}, IC={self.config['n_ic']}, BC={self.config['n_bc']}")
        start_time = time.time()
        
        # Initialize log file
        with open(self.log_file_path, 'w') as f:
            f.write('Epoch,Total_Loss,PDE_Loss,IC_Loss,BC_Loss,Time_s\n')

        for epoch in range(self.config['epochs']):
            self.model.train()
            
            # Sample random collocation points in physical domain
            t_pde = torch.rand(self.config['n_pde'], 1, device=self.device) * self.domain['t'][1]
            x_pde = torch.rand(self.config['n_pde'], 1, device=self.device) * self.domain['x'][1]
            v_pde = (torch.rand(self.config['n_pde'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            # Initial condition points (at t=0)
            t_ic = torch.zeros(self.config['n_ic'], 1, device=self.device)
            x_ic = torch.rand(self.config['n_ic'], 1, device=self.device) * self.domain['x'][1]
            v_ic = (torch.rand(self.config['n_ic'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]
            
            # Boundary condition points
            t_bc = torch.rand(self.config['n_bc'], 1, device=self.device) * self.domain['t'][1]
            x_bc = torch.rand(self.config['n_bc'], 1, device=self.device) * self.domain['x'][1]
            v_bc = (torch.rand(self.config['n_bc'], 1, device=self.device) - 0.5) * 2 * self.domain['v'][1]

            # Forward pass and loss computation
            self.optimizer.zero_grad()
            loss, loss_pde, loss_ic, loss_bc = \
                self.compute_loss(t_pde, x_pde, v_pde, t_ic, x_ic, v_ic, t_bc, x_bc, v_bc)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}. Skipping.")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Learning rate decay
            if (epoch + 1) % 1000 == 0:
                self.scheduler.step()

            # Logging
            if (epoch + 1) % self.config['log_frequency'] == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch [{epoch+1}/{self.config['epochs']}] | "
                    f"Loss: {loss.item():.4e} | L_pde: {loss_pde.item():.4e} | "
                    f"L_ic: {loss_ic.item():.4e} | L_bc: {loss_bc.item():.4e} | "
                    f"Time: {elapsed_time:.2f}s"
                )
                
                # Write to log file
                log_data = (f"{epoch+1},{loss.item()},{loss_pde.item()},"
                            f"{loss_ic.item()},{loss_bc.item()},{elapsed_time}\n")
                with open(self.log_file_path, 'a') as f:
                    f.write(log_data)
            
            # Visualization
            if (epoch + 1) % self.config['plot_frequency'] == 0:
                # Import here to avoid circular dependency
                from visualization import plot_results
                plot_results(self, epoch + 1)
        
        # Final tasks
        from visualization import plot_loss_history
        plot_loss_history(self)
        print("Training finished.")
        print(f"Total time: {time.time() - start_time:.2f}s")
