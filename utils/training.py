"""
Training utilities for AI4CFD methods.
Includes trainers, loss functions, optimizers, and schedulers for PINNs, FNO, DeepONet, TNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import time
from tqdm import tqdm


# ==============================================================================
# Loss Functions
# ==============================================================================

class PINNLoss(nn.Module):
    """
    Composite loss for Physics-Informed Neural Networks.
    
    Example:
        >>> loss_fn = PINNLoss(lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0)
        >>> total_loss, losses = loss_fn(pde_loss, bc_loss, ic_loss)
    """
    def __init__(self, lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0, 
                 lambda_data=1.0, adaptive=False):
        super(PINNLoss, self).__init__()
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.lambda_data = lambda_data
        self.adaptive = adaptive
        
        if adaptive:
            self.log_weights = nn.Parameter(torch.zeros(4))
    
    def forward(self, pde_loss, bc_loss=None, ic_loss=None, data_loss=None):
        losses = {'pde': pde_loss}
        
        if self.adaptive:
            weights = F.softmax(self.log_weights, dim=0) * 4
            total = weights[0] * pde_loss
            if bc_loss is not None:
                total += weights[1] * bc_loss
                losses['bc'] = bc_loss
            if ic_loss is not None:
                total += weights[2] * ic_loss
                losses['ic'] = ic_loss
            if data_loss is not None:
                total += weights[3] * data_loss
                losses['data'] = data_loss
        else:
            total = self.lambda_pde * pde_loss
            if bc_loss is not None:
                total += self.lambda_bc * bc_loss
                losses['bc'] = bc_loss
            if ic_loss is not None:
                total += self.lambda_ic * ic_loss
                losses['ic'] = ic_loss
            if data_loss is not None:
                total += self.lambda_data * data_loss
                losses['data'] = data_loss
        
        losses['total'] = total
        return total, losses


class WeightedMSELoss(nn.Module):
    """MSE loss with spatial weighting."""
    def __init__(self, weights=None):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights
    
    def forward(self, pred, target, weights=None):
        w = weights if weights is not None else self.weights
        diff = (pred - target) ** 2
        if w is not None:
            diff = diff * w
        return diff.mean()


class RelativeMSELoss(nn.Module):
    """Relative MSE loss: MSE / ||target||^2."""
    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        norm = (target ** 2).mean()
        return mse / (norm + 1e-8)


class SobolevLoss(nn.Module):
    """
    Sobolev loss incorporating gradients.
    Useful for ensuring smooth solutions.
    """
    def __init__(self, alpha=0.1):
        super(SobolevLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, pred_grad=None, target_grad=None):
        l2_loss = F.mse_loss(pred, target)
        
        if pred_grad is not None and target_grad is not None:
            grad_loss = F.mse_loss(pred_grad, target_grad)
            return l2_loss + self.alpha * grad_loss
        
        return l2_loss


class SpectralLoss(nn.Module):
    """
    Loss in Fourier space for FNO training.
    Penalizes errors in frequency domain.
    """
    def __init__(self, alpha=0.5):
        super(SpectralLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        # Spatial loss
        spatial_loss = F.mse_loss(pred, target)
        
        # Spectral loss
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        spectral_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
        
        return spatial_loss + self.alpha * spectral_loss


# ==============================================================================
# PDE Residual Functions
# ==============================================================================

def compute_pde_residual(model, x, pde_type='poisson', **kwargs):
    """
    Compute PDE residual for various equation types.
    
    Args:
        model: Neural network
        x: Input coordinates with requires_grad=True
        pde_type: Type of PDE ('poisson', 'heat', 'wave', 'burgers', 'navier_stokes')
        **kwargs: Additional parameters (nu, f, etc.)
    
    Returns:
        PDE residual
    """
    x.requires_grad_(True)
    u = model(x)
    
    if pde_type == 'poisson':
        # -∇²u = f
        lap = compute_laplacian(u, x)
        f = kwargs.get('f', torch.zeros_like(u))
        return -lap - f
    
    elif pde_type == 'heat':
        # ∂u/∂t = α∇²u
        alpha = kwargs.get('alpha', 1.0)
        u_t = compute_derivative(u, x, dim=-1)  # time is last dimension
        lap = compute_laplacian(u, x, exclude_dims=[-1])
        return u_t - alpha * lap
    
    elif pde_type == 'wave':
        # ∂²u/∂t² = c²∇²u
        c = kwargs.get('c', 1.0)
        u_tt = compute_derivative(u, x, dim=-1, order=2)
        lap = compute_laplacian(u, x, exclude_dims=[-1])
        return u_tt - c**2 * lap
    
    elif pde_type == 'burgers':
        # ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
        nu = kwargs.get('nu', 0.01)
        u_t = compute_derivative(u, x, dim=1)  # t is second column
        u_x = compute_derivative(u, x, dim=0)  # x is first column
        u_xx = compute_derivative(u, x, dim=0, order=2)
        return u_t + u * u_x - nu * u_xx
    
    elif pde_type == 'advection':
        # ∂u/∂t + a·∇u = 0
        a = kwargs.get('a', 1.0)
        u_t = compute_derivative(u, x, dim=-1)
        u_x = compute_derivative(u, x, dim=0)
        return u_t + a * u_x
    
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


def compute_derivative(u, x, dim=0, order=1):
    """Compute derivative of u with respect to x[:, dim]."""
    grad = u
    for _ in range(order):
        grad = torch.autograd.grad(
            grad.sum(), x, create_graph=True, retain_graph=True
        )[0][:, dim:dim+1]
    return grad


def compute_laplacian(u, x, exclude_dims=None):
    """Compute Laplacian (sum of second derivatives)."""
    lap = 0
    for i in range(x.shape[1]):
        if exclude_dims is not None and i in exclude_dims:
            continue
        if exclude_dims is not None and -1 in exclude_dims and i == x.shape[1] - 1:
            continue
        
        u_i = torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0][:, i:i+1]
        u_ii = torch.autograd.grad(u_i.sum(), x, create_graph=True, retain_graph=True)[0][:, i:i+1]
        lap += u_ii
    return lap


def compute_gradient(u, x):
    """Compute gradient of u with respect to all x dimensions."""
    return torch.autograd.grad(u.sum(), x, create_graph=True, retain_graph=True)[0]


def compute_divergence(vec_field, x):
    """Compute divergence of a vector field."""
    div = 0
    for i in range(vec_field.shape[1]):
        div_i = torch.autograd.grad(
            vec_field[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0][:, i]
        div += div_i
    return div.unsqueeze(1)


# ==============================================================================
# Trainers
# ==============================================================================

class BaseTrainer:
    """Base trainer class with common functionality."""
    
    def __init__(self, model, optimizer=None, scheduler=None, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        self.history = {'loss': [], 'time': []}
    
    def train_epoch(self, dataloader):
        """Override in subclasses."""
        raise NotImplementedError
    
    def train(self, epochs, dataloader=None, verbose=True):
        """Main training loop."""
        start_time = time.time()
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            loss = self.train_epoch(dataloader)
            self.history['loss'].append(loss)
            self.history['time'].append(time.time() - start_time)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({'loss': f'{loss:.6f}'})
        
        return self.history


class PINNTrainer(BaseTrainer):
    """
    Trainer for Physics-Informed Neural Networks.
    
    Example:
        >>> trainer = PINNTrainer(model, pde_residual_fn, device='cuda')
        >>> trainer.set_data(x_interior, x_boundary, u_boundary)
        >>> history = trainer.train(epochs=5000)
    """
    def __init__(self, model, pde_residual_fn, optimizer=None, scheduler=None,
                 device='cpu', lambda_pde=1.0, lambda_bc=10.0, lambda_ic=10.0):
        super().__init__(model, optimizer, scheduler, device)
        self.pde_residual_fn = pde_residual_fn
        self.loss_fn = PINNLoss(lambda_pde, lambda_bc, lambda_ic)
        
        self.x_interior = None
        self.x_boundary = None
        self.u_boundary = None
        self.x_initial = None
        self.u_initial = None
        
        self.history.update({'pde_loss': [], 'bc_loss': [], 'ic_loss': []})
    
    def set_data(self, x_interior, x_boundary=None, u_boundary=None,
                 x_initial=None, u_initial=None):
        """Set training data."""
        self.x_interior = x_interior.to(self.device).requires_grad_(True)
        if x_boundary is not None:
            self.x_boundary = x_boundary.to(self.device)
            self.u_boundary = u_boundary.to(self.device)
        if x_initial is not None:
            self.x_initial = x_initial.to(self.device)
            self.u_initial = u_initial.to(self.device)
    
    def train_epoch(self, dataloader=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        # PDE loss
        u = self.model(self.x_interior)
        pde_residual = self.pde_residual_fn(self.model, self.x_interior)
        pde_loss = (pde_residual ** 2).mean()
        
        # Boundary loss
        bc_loss = None
        if self.x_boundary is not None:
            u_bc_pred = self.model(self.x_boundary)
            bc_loss = F.mse_loss(u_bc_pred, self.u_boundary)
        
        # Initial condition loss
        ic_loss = None
        if self.x_initial is not None:
            u_ic_pred = self.model(self.x_initial)
            ic_loss = F.mse_loss(u_ic_pred, self.u_initial)
        
        total_loss, losses = self.loss_fn(pde_loss, bc_loss, ic_loss)
        total_loss.backward()
        self.optimizer.step()
        
        # Record losses
        self.history['pde_loss'].append(pde_loss.item())
        if bc_loss is not None:
            self.history['bc_loss'].append(bc_loss.item())
        if ic_loss is not None:
            self.history['ic_loss'].append(ic_loss.item())
        
        return total_loss.item()


class DeepONetTrainer(BaseTrainer):
    """
    Trainer for DeepONet.
    
    Example:
        >>> trainer = DeepONetTrainer(model, device='cuda')
        >>> history = trainer.train(epochs=2000, dataloader=train_loader)
    """
    def __init__(self, model, optimizer=None, scheduler=None, device='cpu'):
        super().__init__(model, optimizer, scheduler, device)
        self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            u_sensors, y_query, G_u_target = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            G_u_pred = self.model(u_sensors, y_query)
            loss = self.criterion(G_u_pred, G_u_target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches


class FNOTrainer(BaseTrainer):
    """
    Trainer for Fourier Neural Operators.
    
    Example:
        >>> trainer = FNOTrainer(model, device='cuda')
        >>> history = trainer.train(epochs=500, dataloader=train_loader)
    """
    def __init__(self, model, optimizer=None, scheduler=None, device='cpu', 
                 use_spectral_loss=True, spectral_alpha=0.5):
        super().__init__(model, optimizer, scheduler, device)
        
        if use_spectral_loss:
            self.criterion = SpectralLoss(alpha=spectral_alpha)
        else:
            self.criterion = nn.MSELoss()
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches


class TNNTrainer(BaseTrainer):
    """
    Trainer for Tensor Neural Networks.
    
    Example:
        >>> trainer = TNNTrainer(model, pde_residual_fn, device='cuda')
        >>> history = trainer.train(epochs=3000)
    """
    def __init__(self, model, pde_residual_fn=None, optimizer=None, 
                 scheduler=None, device='cpu', mode='supervised'):
        super().__init__(model, optimizer, scheduler, device)
        self.pde_residual_fn = pde_residual_fn
        self.mode = mode
        self.criterion = nn.MSELoss()
        
        self.x_train = None
        self.y_train = None
    
    def set_data(self, x_train, y_train=None, x_boundary=None, u_boundary=None):
        """Set training data."""
        self.x_train = x_train.to(self.device)
        if y_train is not None:
            self.y_train = y_train.to(self.device)
        if x_boundary is not None:
            self.x_boundary = x_boundary.to(self.device)
            self.u_boundary = u_boundary.to(self.device)
    
    def train_epoch(self, dataloader=None):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.mode == 'supervised':
            pred = self.model(self.x_train)
            loss = self.criterion(pred, self.y_train)
        
        elif self.mode == 'pinn':
            self.x_train.requires_grad_(True)
            residual = self.pde_residual_fn(self.model, self.x_train)
            pde_loss = (residual ** 2).mean()
            
            bc_loss = 0
            if hasattr(self, 'x_boundary') and self.x_boundary is not None:
                u_bc_pred = self.model(self.x_boundary)
                bc_loss = self.criterion(u_bc_pred, self.u_boundary)
            
            loss = pde_loss + 10 * bc_loss
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


# ==============================================================================
# Optimizer Utilities
# ==============================================================================

def get_optimizer(model, optimizer_type='adam', lr=1e-3, weight_decay=0, **kwargs):
    """
    Get optimizer by name.
    
    Args:
        model: Neural network
        optimizer_type: 'adam', 'adamw', 'sgd', 'lbfgs', 'rmsprop'
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
    
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, 
                              momentum=kwargs.get('momentum', 0.9))
    elif optimizer_type.lower() == 'lbfgs':
        return torch.optim.LBFGS(params, lr=lr, max_iter=kwargs.get('max_iter', 20))
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: 'step', 'cosine', 'exponential', 'plateau', 'warmup_cosine'
    
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 1000),
            gamma=kwargs.get('gamma', 0.5)
        )
    elif scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 5000),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.999)
        )
    elif scheduler_type.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 100)
        )
    elif scheduler_type.lower() == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 100),
            total_epochs=kwargs.get('total_epochs', 5000)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs
            ]


# ==============================================================================
# Training Utilities
# ==============================================================================

def train_with_lbfgs(model, loss_fn, x_data, max_iter=500, verbose=True):
    """
    Train using L-BFGS optimizer (second-order method).
    Good for PINNs after initial Adam training.
    
    Example:
        >>> history = train_with_lbfgs(pinn, pinn_loss_fn, x_collocation)
    """
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                   max_eval=25, tolerance_grad=1e-7,
                                   tolerance_change=1e-9, history_size=50)
    
    history = {'loss': []}
    
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model, x_data)
        loss.backward()
        return loss
    
    iterator = tqdm(range(max_iter), desc="L-BFGS") if verbose else range(max_iter)
    
    for i in iterator:
        loss = optimizer.step(closure)
        history['loss'].append(loss.item())
        
        if verbose and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return history


def adaptive_sampling(model, pde_residual_fn, domain, n_new=100, n_candidates=1000):
    """
    Adaptive sampling: add more points where PDE residual is large.
    
    Args:
        model: PINN model
        pde_residual_fn: Function computing PDE residual
        domain: Domain specification (list of tuples)
        n_new: Number of new points to sample
        n_candidates: Number of candidate points to evaluate
    
    Returns:
        New collocation points
    """
    dim = len(domain)
    device = next(model.parameters()).device
    
    # Generate candidate points
    candidates = torch.zeros(n_candidates, dim, device=device)
    for i, (low, high) in enumerate(domain):
        candidates[:, i] = torch.rand(n_candidates, device=device) * (high - low) + low
    
    candidates.requires_grad_(True)
    
    # Compute residuals
    with torch.no_grad():
        model.eval()
        residuals = pde_residual_fn(model, candidates)
        residual_magnitude = residuals.abs().squeeze()
    
    # Sample proportionally to residual magnitude
    probs = residual_magnitude / residual_magnitude.sum()
    indices = torch.multinomial(probs, n_new, replacement=False)
    
    return candidates[indices].detach().requires_grad_(True)


def gradient_clipping(model, max_norm=1.0):
    """Apply gradient clipping to model parameters."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=100, min_delta=1e-6, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self._is_improvement(loss):
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, loss):
        if self.mode == 'min':
            return loss < self.best_loss - self.min_delta
        else:
            return loss > self.best_loss + self.min_delta


class GradientBalancer:
    """
    Balance gradients from multiple loss terms.
    Reference: "Multi-Task Learning as Multi-Objective Optimization" (Sener & Koltun)
    """
    
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.weights = torch.ones(num_tasks) / num_tasks
    
    def compute_weights(self, grads):
        """
        Compute optimal weights for gradient balancing.
        
        Args:
            grads: List of gradients from each task
        """
        # Simple gradient magnitude balancing
        magnitudes = [g.norm() for g in grads]
        avg_magnitude = sum(magnitudes) / len(magnitudes)
        
        weights = torch.tensor([avg_magnitude / (m + 1e-8) for m in magnitudes])
        self.weights = weights / weights.sum() * len(weights)
        
        return self.weights


def compute_ntk_eigenvalues(model, x, num_eigenvalues=10):
    """
    Compute eigenvalues of Neural Tangent Kernel.
    Useful for understanding PINN training dynamics.
    """
    n_samples = x.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    
    # Compute Jacobian
    jacobian = torch.zeros(n_samples, n_params)
    
    for i in range(n_samples):
        model.zero_grad()
        output = model(x[i:i+1])
        output.backward()
        
        j_row = torch.cat([p.grad.flatten() for p in model.parameters()])
        jacobian[i] = j_row
    
    # NTK = J @ J^T
    ntk = jacobian @ jacobian.T
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(ntk)
    
    return eigenvalues[-num_eigenvalues:]  # Largest eigenvalues
