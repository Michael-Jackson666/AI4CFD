"""
Training utilities for AI4CFD methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Callable, Union, Tuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which training stops
            min_delta: Minimum change in monitored quantity to qualify as improvement
            restore_best_weights: Whether to restore model weights from best epoch
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


class LossTracker:
    """Track and visualize training losses."""
    
    def __init__(self):
        self.losses = {
            'train': [],
            'val': [],
            'pde': [],
            'boundary': [],
            'initial': [],
            'data': []
        }
        self.epochs = []
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        pde_loss: Optional[float] = None,
        boundary_loss: Optional[float] = None,
        initial_loss: Optional[float] = None,
        data_loss: Optional[float] = None
    ):
        """Update loss tracking."""
        self.epochs.append(epoch)
        self.losses['train'].append(train_loss)
        
        if val_loss is not None:
            self.losses['val'].append(val_loss)
        if pde_loss is not None:
            self.losses['pde'].append(pde_loss)
        if boundary_loss is not None:
            self.losses['boundary'].append(boundary_loss)
        if initial_loss is not None:
            self.losses['initial'].append(initial_loss)
        if data_loss is not None:
            self.losses['data'].append(data_loss)
    
    def plot(self, save_path: Optional[str] = None):
        """Plot loss curves."""
        plt.figure(figsize=(12, 8))
        
        for loss_type, values in self.losses.items():
            if values:  # Only plot non-empty losses
                plt.plot(self.epochs[:len(values)], values, label=f'{loss_type} loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.title('Training Loss Curves')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class AdaptiveLossWeighting:
    """Adaptive loss weighting for multi-objective optimization."""
    
    def __init__(
        self,
        initial_weights: Dict[str, float],
        adaptation_method: str = "gradnorm",
        alpha: float = 0.12,
        update_frequency: int = 10
    ):
        """
        Initialize adaptive loss weighting.
        
        Args:
            initial_weights: Initial loss weights
            adaptation_method: Method for adaptation ("gradnorm", "uncertainty")
            alpha: GradNorm hyperparameter
            update_frequency: How often to update weights (in epochs)
        """
        self.weights = initial_weights.copy()
        self.adaptation_method = adaptation_method
        self.alpha = alpha
        self.update_frequency = update_frequency
        
        self.initial_losses = {}
        self.iteration = 0
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module
    ):
        """Update loss weights based on gradients."""
        self.iteration += 1
        
        if self.iteration == 1:
            # Store initial losses
            self.initial_losses = {k: v.item() for k, v in losses.items()}
        
        if self.iteration % self.update_frequency == 0 and self.adaptation_method == "gradnorm":
            self._update_gradnorm(losses, model)
    
    def _update_gradnorm(self, losses: Dict[str, torch.Tensor], model: nn.Module):
        """Update weights using GradNorm algorithm."""
        # This is a simplified implementation
        # In practice, would compute gradients w.r.t. shared parameters
        
        loss_ratios = {}
        for task, loss in losses.items():
            if task in self.initial_losses:
                loss_ratios[task] = loss.item() / self.initial_losses[task]
        
        # Simple heuristic: increase weight for tasks with higher relative loss
        total_ratio = sum(loss_ratios.values())
        for task in self.weights:
            if task in loss_ratios:
                relative_ratio = loss_ratios[task] / total_ratio
                self.weights[task] *= (1 + self.alpha * relative_ratio)


def train_pinn(
    model: nn.Module,
    pde_func: Callable,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    boundary_values: torch.Tensor,
    initial_points: Optional[torch.Tensor] = None,
    initial_values: Optional[torch.Tensor] = None,
    data_points: Optional[torch.Tensor] = None,
    data_values: Optional[torch.Tensor] = None,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    loss_weights: Optional[Dict[str, float]] = None,
    optimizer_type: str = "adam",
    scheduler_type: Optional[str] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[nn.Module, LossTracker]:
    """
    Train a PINN model.
    
    Args:
        model: PINN model
        pde_func: PDE residual function
        interior_points: Interior domain points
        boundary_points: Boundary points
        boundary_values: Boundary condition values
        initial_points: Initial condition points
        initial_values: Initial condition values
        data_points: Data points (if available)
        data_values: Data values (if available)
        num_epochs: Number of training epochs
        lr: Learning rate
        loss_weights: Weights for different loss components
        optimizer_type: Optimizer type ("adam", "lbfgs", "sgd")
        scheduler_type: Learning rate scheduler type
        early_stopping: Early stopping instance
        device: Training device
        verbose: Whether to print progress
        
    Returns:
        Trained model and loss tracker
    """
    # Default loss weights
    if loss_weights is None:
        loss_weights = {
            'pde': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'data': 1.0
        }
    
    # Move to device
    model = model.to(device)
    interior_points = interior_points.to(device)
    boundary_points = boundary_points.to(device)
    boundary_values = boundary_values.to(device)
    
    if initial_points is not None:
        initial_points = initial_points.to(device)
        initial_values = initial_values.to(device)
    
    if data_points is not None:
        data_points = data_points.to(device)
        data_values = data_values.to(device)
    
    # Setup optimizer
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Setup scheduler
    scheduler = None
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss tracker
    loss_tracker = LossTracker()
    
    # Training loop
    for epoch in tqdm(range(num_epochs), disable=not verbose):
        def closure():
            optimizer.zero_grad()
            
            # PDE loss
            pde_loss = model.pde_loss(interior_points, pde_func)
            
            # Boundary loss
            boundary_loss = model.boundary_loss(boundary_points, boundary_values)
            
            # Initial loss
            initial_loss = torch.tensor(0.0, device=device)
            if initial_points is not None:
                initial_loss = model.initial_loss(initial_points, initial_values)
            
            # Data loss
            data_loss = torch.tensor(0.0, device=device)
            if data_points is not None:
                data_loss = model.data_loss(data_points, data_values)
            
            # Total loss
            total_loss = (
                loss_weights['pde'] * pde_loss +
                loss_weights['boundary'] * boundary_loss +
                loss_weights['initial'] * initial_loss +
                loss_weights['data'] * data_loss
            )
            
            total_loss.backward()
            return total_loss
        
        if optimizer_type == "lbfgs":
            optimizer.step(closure)
            loss = closure()
        else:
            loss = closure()
            optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Track losses
        if epoch % 10 == 0:
            with torch.no_grad():
                pde_loss = model.pde_loss(interior_points, pde_func)
                boundary_loss = model.boundary_loss(boundary_points, boundary_values)
                
                loss_tracker.update(
                    epoch=epoch,
                    train_loss=loss.item(),
                    pde_loss=pde_loss.item(),
                    boundary_loss=boundary_loss.item()
                )
        
        # Early stopping
        if early_stopping and early_stopping(loss.item(), model):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    return model, loss_tracker


def train_deeponet(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 1000,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    scheduler_type: Optional[str] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: str = "cpu",
    verbose: bool = True
) -> Tuple[nn.Module, LossTracker]:
    """
    Train a DeepONet model.
    
    Args:
        model: DeepONet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        optimizer_type: Optimizer type
        scheduler_type: Learning rate scheduler type
        early_stopping: Early stopping instance
        device: Training device
        verbose: Whether to print progress
        
    Returns:
        Trained model and loss tracker
    """
    model = model.to(device)
    
    # Setup optimizer
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Setup scheduler
    scheduler = None
    if scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    elif scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    loss_tracker = LossTracker()
    
    for epoch in tqdm(range(num_epochs), disable=not verbose):
        # Training
        model.train()
        train_loss = 0.0
        
        for branch_input, trunk_input, target in train_loader:
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            loss = model.operator_loss(branch_input, trunk_input, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for branch_input, trunk_input, target in val_loader:
                    branch_input = branch_input.to(device)
                    trunk_input = trunk_input.to(device)
                    target = target.to(device)
                    
                    loss = model.operator_loss(branch_input, trunk_input, target)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
        
        if scheduler:
            scheduler.step()
        
        # Track losses
        if epoch % 10 == 0:
            loss_tracker.update(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        
        # Early stopping
        if early_stopping and val_loss and early_stopping(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    return model, loss_tracker