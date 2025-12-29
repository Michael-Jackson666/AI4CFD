"""
Training utilities for AI4CFD methods.
Includes trainers for PINNs, DeepONet, FNO, TNN with common training patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import time
from tqdm import tqdm


# ==============================================================================
# Base Trainer
# ==============================================================================

class BaseTrainer:
    """
    Base trainer class with common training utilities.
    """
    def __init__(self, model, device='cpu', lr=1e-3, optimizer='adam', 
                 scheduler=None, grad_clip=None):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        
        # Setup optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer == 'lbfgs':
            self.optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        elif scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)
        elif scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100)
        else:
            self.scheduler = scheduler
        
        self.history = {'loss': [], 'lr': []}
    
    def train_step(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError
    
    def train(self, epochs, *args, verbose=True, **kwargs):
        """Generic training loop."""
        self.model.train()
        
        pbar = tqdm(range(epochs), disable=not verbose)
        for epoch in pbar:
            loss = self.train_step(*args, **kwargs)
            
            self.history['loss'].append(loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
            
            if verbose:
                pbar.set_postfix({'loss': f'{loss:.6e}', 'lr': f'{self.history["lr"][-1]:.2e}'})
        
        return self.history


# ==============================================================================
# PINNs Trainer
# ==============================================================================

class PINNTrainer(BaseTrainer):
    """
    Trainer for Physics-Informed Neural Networks.
    
    Example:
        >>> from utils.nn_blocks import PINN
        >>> model = PINN(input_dim=2, output_dim=1, hidden_layers=[64, 64, 64])
        >>> trainer = PINNTrainer(model, device='cuda')
        >>> 
        >>> # Define PDE residual
        >>> def pde_residual(model, x):
        ...     x.requires_grad_(True)
        ...     u = model(x)
        ...     u_x, u_t = model.gradient(u, x)
        ...     u_xx = model.gradient(u_x, x)[0]
        ...     return u_t - 0.01 * u_xx  # Heat equation
        >>> 
        >>> history = trainer.train(
        ...     epochs=10000,
        ...     x_interior=x_interior,
        ...     x_boundary=x_boundary,
        ...     u_boundary=u_boundary,
        ...     pde_residual=pde_residual
        ... )
    """
    def __init__(self, model, device='cpu', lr=1e-3, 
                 lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0, lambda_data=1.0,
                 adaptive_weights=False, **kwargs):
        super().__init__(model, device, lr, **kwargs)
        
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.lambda_data = lambda_data
        self.adaptive_weights = adaptive_weights
        
        self.history.update({
            'pde_loss': [], 'bc_loss': [], 'ic_loss': [], 'data_loss': []
        })
    
    def train_step(self, x_interior, x_boundary, u_boundary, pde_residual,
                   x_ic=None, u_ic=None, x_data=None, u_data=None):
        """Single training step for PINNs."""
        self.optimizer.zero_grad()
        
        # PDE residual loss
        x_interior = x_interior.to(self.device).requires_grad_(True)
        residual = pde_residual(self.model, x_interior)
        pde_loss = torch.mean(residual ** 2)
        
        # Boundary condition loss
        x_boundary = x_boundary.to(self.device)
        u_boundary = u_boundary.to(self.device)
        u_pred_bc = self.model(x_boundary)
        bc_loss = torch.mean((u_pred_bc - u_boundary) ** 2)
        
        # Initial condition loss (if provided)
        ic_loss = torch.tensor(0.0, device=self.device)
        if x_ic is not None and u_ic is not None:
            x_ic = x_ic.to(self.device)
            u_ic = u_ic.to(self.device)
            u_pred_ic = self.model(x_ic)
            ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)
        
        # Data loss (if provided)
        data_loss = torch.tensor(0.0, device=self.device)
        if x_data is not None and u_data is not None:
            x_data = x_data.to(self.device)
            u_data = u_data.to(self.device)
            u_pred_data = self.model(x_data)
            data_loss = torch.mean((u_pred_data - u_data) ** 2)
        
        # Total loss
        if self.adaptive_weights and hasattr(self.model, 'get_weights'):
            weights = self.model.get_weights()
            loss = weights[0] * pde_loss + weights[1] * bc_loss + weights[2] * ic_loss
        else:
            loss = (self.lambda_pde * pde_loss + 
                   self.lambda_bc * bc_loss + 
                   self.lambda_ic * ic_loss +
                   self.lambda_data * data_loss)
        
        loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        # Record losses
        self.history['pde_loss'].append(pde_loss.item())
        self.history['bc_loss'].append(bc_loss.item())
        self.history['ic_loss'].append(ic_loss.item())
        self.history['data_loss'].append(data_loss.item())
        
        return loss.item()
    
    def train_lbfgs(self, x_interior, x_boundary, u_boundary, pde_residual,
                    x_ic=None, u_ic=None, max_iter=500, verbose=True):
        """Train using L-BFGS optimizer (often better for PINNs)."""
        optimizer = optim.LBFGS(self.model.parameters(), lr=1.0, 
                               max_iter=max_iter, history_size=50,
                               line_search_fn='strong_wolfe')
        
        x_interior = x_interior.to(self.device)
        x_boundary = x_boundary.to(self.device)
        u_boundary = u_boundary.to(self.device)
        
        if x_ic is not None:
            x_ic = x_ic.to(self.device)
            u_ic = u_ic.to(self.device)
        
        iteration = [0]
        
        def closure():
            optimizer.zero_grad()
            
            x_int = x_interior.requires_grad_(True)
            residual = pde_residual(self.model, x_int)
            pde_loss = torch.mean(residual ** 2)
            
            u_pred_bc = self.model(x_boundary)
            bc_loss = torch.mean((u_pred_bc - u_boundary) ** 2)
            
            ic_loss = torch.tensor(0.0, device=self.device)
            if x_ic is not None and u_ic is not None:
                u_pred_ic = self.model(x_ic)
                ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)
            
            loss = self.lambda_pde * pde_loss + self.lambda_bc * bc_loss + self.lambda_ic * ic_loss
            loss.backward()
            
            iteration[0] += 1
            if verbose and iteration[0] % 50 == 0:
                print(f"Iter {iteration[0]}: Loss = {loss.item():.6e}")
            
            self.history['loss'].append(loss.item())
            return loss
        
        optimizer.step(closure)
        return self.history


# ==============================================================================
# DeepONet Trainer
# ==============================================================================

class DeepONetTrainer(BaseTrainer):
    """
    Trainer for Deep Operator Networks.
    
    Example:
        >>> from utils.nn_blocks import DeepONet
        >>> model = DeepONet(branch_input=100, trunk_input=1, hidden_dim=64)
        >>> trainer = DeepONetTrainer(model, device='cuda')
        >>> history = trainer.train(
        ...     epochs=5000,
        ...     u_sensors=u_sensors,  # [N, n_sensors]
        ...     y_query=y_query,      # [n_query, trunk_dim]
        ...     outputs=outputs       # [N, n_query, output_dim]
        ... )
    """
    def __init__(self, model, device='cpu', lr=1e-3, batch_size=32, **kwargs):
        super().__init__(model, device, lr, **kwargs)
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
    
    def train_step(self, u_sensors, y_query, outputs):
        """Single training step for DeepONet."""
        u_sensors = u_sensors.to(self.device)
        y_query = y_query.to(self.device)
        outputs = outputs.to(self.device)
        
        # Mini-batch training
        n_samples = u_sensors.shape[0]
        indices = torch.randperm(n_samples)[:self.batch_size]
        
        u_batch = u_sensors[indices]
        out_batch = outputs[indices]
        
        self.optimizer.zero_grad()
        
        pred = self.model(u_batch, y_query)
        loss = self.criterion(pred, out_batch)
        
        loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_physics_informed(self, u_sensors, y_query, outputs, 
                               pde_residual, epochs, lambda_data=1.0, lambda_pde=0.1,
                               verbose=True):
        """
        Physics-informed training for DeepONet.
        Combines data loss with physics constraints.
        """
        self.model.train()
        pbar = tqdm(range(epochs), disable=not verbose)
        
        u_sensors = u_sensors.to(self.device)
        y_query = y_query.to(self.device).requires_grad_(True)
        outputs = outputs.to(self.device)
        
        for epoch in pbar:
            self.optimizer.zero_grad()
            
            # Data loss
            pred = self.model(u_sensors, y_query)
            data_loss = self.criterion(pred, outputs)
            
            # Physics loss
            physics_loss = pde_residual(self.model, u_sensors, y_query)
            
            loss = lambda_data * data_loss + lambda_pde * physics_loss
            loss.backward()
            self.optimizer.step()
            
            self.history['loss'].append(loss.item())
            
            if verbose:
                pbar.set_postfix({'loss': f'{loss.item():.6e}'})
        
        return self.history


# ==============================================================================
# FNO Trainer
# ==============================================================================

class FNOTrainer(BaseTrainer):
    """
    Trainer for Fourier Neural Operators.
    
    Example:
        >>> from utils.nn_blocks import FNO2d
        >>> model = FNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12)
        >>> trainer = FNOTrainer(model, device='cuda')
        >>> history = trainer.train(
        ...     epochs=500,
        ...     train_loader=train_loader,  # DataLoader with (input, output) pairs
        ...     val_loader=val_loader
        ... )
    """
    def __init__(self, model, device='cpu', lr=1e-3, **kwargs):
        super().__init__(model, device, lr, **kwargs)
        self.criterion = nn.MSELoss()
        self.history.update({'val_loss': []})
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, epochs, train_loader, val_loader=None, verbose=True):
        """Train FNO model."""
        pbar = tqdm(range(epochs), disable=not verbose)
        
        for epoch in pbar:
            train_loss = self.train_epoch(train_loader)
            self.history['loss'].append(train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loss else train_loss)
                else:
                    self.scheduler.step()
            
            if verbose:
                msg = {'train': f'{train_loss:.6e}'}
                if val_loss:
                    msg['val'] = f'{val_loss:.6e}'
                pbar.set_postfix(msg)
        
        return self.history


# ==============================================================================
# TNN Trainer
# ==============================================================================

class TNNTrainer(BaseTrainer):
    """
    Trainer for Tensor Neural Networks.
    Designed for high-dimensional PDE problems.
    
    Example:
        >>> from utils.nn_blocks import TNN
        >>> model = TNN(dim=5, hidden_size=32, rank=20)
        >>> trainer = TNNTrainer(model, device='cuda')
        >>> 
        >>> def pde_residual(model, x):
        ...     # 5D Poisson equation
        ...     return compute_laplacian(model, x) + f(x)
        >>> 
        >>> history = trainer.train(
        ...     epochs=5000,
        ...     x_interior=x_interior,
        ...     x_boundary=x_boundary,
        ...     u_boundary=u_boundary,
        ...     pde_residual=pde_residual
        ... )
    """
    def __init__(self, model, device='cpu', lr=1e-3, **kwargs):
        super().__init__(model, device, lr, **kwargs)
    
    def train_step(self, x_interior, x_boundary, u_boundary, pde_residual):
        """Single training step for TNN."""
        self.optimizer.zero_grad()
        
        x_interior = x_interior.to(self.device).requires_grad_(True)
        x_boundary = x_boundary.to(self.device)
        u_boundary = u_boundary.to(self.device)
        
        # PDE residual
        residual = pde_residual(self.model, x_interior)
        pde_loss = torch.mean(residual ** 2)
        
        # Boundary loss
        u_pred_bc = self.model(x_boundary)
        bc_loss = torch.mean((u_pred_bc - u_boundary) ** 2)
        
        loss = pde_loss + 10 * bc_loss
        loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        
        return loss.item()


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_gradient(model, x, output_idx=None):
    """
    Compute gradient of model output with respect to input.
    
    Args:
        model: Neural network
        x: Input tensor with requires_grad=True
        output_idx: Index of output component (for multi-output models)
    
    Returns:
        Gradient tensor
    """
    u = model(x)
    if output_idx is not None:
        u = u[:, output_idx:output_idx+1]
    
    grad = torch.autograd.grad(
        u.sum(), x, create_graph=True, retain_graph=True
    )[0]
    
    return grad


def compute_laplacian(model, x):
    """
    Compute Laplacian of model output.
    
    Args:
        model: Neural network
        x: Input tensor with requires_grad=True
    
    Returns:
        Laplacian tensor
    """
    u = model(x)
    laplacian = torch.zeros_like(u)
    
    for i in range(x.shape[1]):
        u_i = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, i:i+1]
        u_ii = torch.autograd.grad(u_i.sum(), x, create_graph=True)[0][:, i:i+1]
        laplacian += u_ii
    
    return laplacian


def compute_hessian(model, x):
    """
    Compute Hessian matrix of model output.
    
    Args:
        model: Neural network
        x: Input tensor with requires_grad=True
    
    Returns:
        Hessian tensor [batch, dim, dim]
    """
    batch_size, dim = x.shape
    u = model(x)
    
    hessian = torch.zeros(batch_size, dim, dim, device=x.device)
    
    # First derivatives
    grads = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    # Second derivatives
    for i in range(dim):
        grad_i = grads[:, i]
        grad2_i = torch.autograd.grad(grad_i.sum(), x, create_graph=True)[0]
        hessian[:, i, :] = grad2_i
    
    return hessian


class EarlyStopping:
    """
    Early stopping to avoid overfitting.
    
    Example:
        >>> early_stop = EarlyStopping(patience=100, min_delta=1e-6)
        >>> for epoch in range(epochs):
        ...     loss = train_step()
        ...     if early_stop(loss):
        ...         print("Early stopping triggered")
        ...         break
    """
    def __init__(self, patience=100, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class AdaptiveSampler:
    """
    Adaptive sampling for PINNs based on residual magnitude.
    Focuses sampling on regions with high PDE residual.
    
    Example:
        >>> sampler = AdaptiveSampler(domain=[(0, 1), (0, 1)])
        >>> for epoch in range(epochs):
        ...     if epoch % 100 == 0:
        ...         x_interior = sampler.sample(model, pde_residual, n_points=1000)
        ...     train_step(x_interior, ...)
    """
    def __init__(self, domain, n_candidates=10000):
        self.domain = domain
        self.n_candidates = n_candidates
        self.dim = len(domain)
    
    def sample(self, model, pde_residual, n_points, device='cpu'):
        """Sample points adaptively based on residual magnitude."""
        # Generate candidate points
        candidates = torch.zeros(self.n_candidates, self.dim, device=device)
        for i, (low, high) in enumerate(self.domain):
            candidates[:, i] = torch.rand(self.n_candidates, device=device) * (high - low) + low
        
        # Compute residuals
        candidates.requires_grad_(True)
        with torch.no_grad():
            residuals = torch.abs(pde_residual(model, candidates))
        
        # Sample proportionally to residual magnitude
        probs = residuals.squeeze() / residuals.sum()
        indices = torch.multinomial(probs, n_points, replacement=True)
        
        return candidates[indices].detach().requires_grad_(True)


class LossBalancer:
    """
    Automatic loss balancing using gradient magnitudes.
    Reference: "When and Why PINNs Fail to Train" (Wang et al.)
    
    Example:
        >>> balancer = LossBalancer(n_losses=3)
        >>> for epoch in range(epochs):
        ...     losses = [pde_loss, bc_loss, ic_loss]
        ...     weights = balancer.get_weights(model, losses)
        ...     total_loss = sum(w * l for w, l in zip(weights, losses))
    """
    def __init__(self, n_losses, alpha=0.9):
        self.n_losses = n_losses
        self.alpha = alpha
        self.running_grads = [None] * n_losses
    
    def get_weights(self, model, losses):
        """Compute balanced weights based on gradient magnitudes."""
        grad_norms = []
        
        for loss in losses:
            model.zero_grad()
            loss.backward(retain_graph=True)
            
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm() ** 2
            grad_norms.append(total_norm.sqrt().item())
        
        # Update running average
        for i, norm in enumerate(grad_norms):
            if self.running_grads[i] is None:
                self.running_grads[i] = norm
            else:
                self.running_grads[i] = self.alpha * self.running_grads[i] + (1 - self.alpha) * norm
        
        # Compute weights (inverse of gradient magnitude)
        mean_grad = np.mean(self.running_grads)
        weights = [mean_grad / (g + 1e-8) for g in self.running_grads]
        
        return weights


def create_pde_dataset(x_domain, t_domain, n_interior, n_boundary, n_initial, device='cpu'):
    """
    Create dataset for time-dependent PDE training.
    
    Args:
        x_domain: Tuple (x_min, x_max) or list of tuples for multi-dim
        t_domain: Tuple (t_min, t_max)
        n_interior: Number of interior points
        n_boundary: Number of boundary points per face
        n_initial: Number of initial condition points
    
    Returns:
        Dictionary with interior, boundary, and initial condition points
    """
    # Handle 1D and multi-D cases
    if isinstance(x_domain[0], (int, float)):
        x_domain = [x_domain]
    
    dim = len(x_domain) + 1  # spatial dims + time
    
    # Interior points
    x_interior = torch.zeros(n_interior, dim, device=device)
    for i, (low, high) in enumerate(x_domain):
        x_interior[:, i] = torch.rand(n_interior, device=device) * (high - low) + low
    x_interior[:, -1] = torch.rand(n_interior, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
    
    # Boundary points
    boundary_points = []
    for i, (low, high) in enumerate(x_domain):
        # Low boundary
        x_low = torch.zeros(n_boundary, dim, device=device)
        for j, (l, h) in enumerate(x_domain):
            if j == i:
                x_low[:, j] = low
            else:
                x_low[:, j] = torch.rand(n_boundary, device=device) * (h - l) + l
        x_low[:, -1] = torch.rand(n_boundary, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
        boundary_points.append(x_low)
        
        # High boundary
        x_high = torch.zeros(n_boundary, dim, device=device)
        for j, (l, h) in enumerate(x_domain):
            if j == i:
                x_high[:, j] = high
            else:
                x_high[:, j] = torch.rand(n_boundary, device=device) * (h - l) + l
        x_high[:, -1] = torch.rand(n_boundary, device=device) * (t_domain[1] - t_domain[0]) + t_domain[0]
        boundary_points.append(x_high)
    
    x_boundary = torch.cat(boundary_points, dim=0)
    
    # Initial condition points
    x_initial = torch.zeros(n_initial, dim, device=device)
    for i, (low, high) in enumerate(x_domain):
        x_initial[:, i] = torch.rand(n_initial, device=device) * (high - low) + low
    x_initial[:, -1] = t_domain[0]
    
    return {
        'interior': x_interior,
        'boundary': x_boundary,
        'initial': x_initial
    }
