"""
Evaluation metrics for PDE solving methods.
"""

import numpy as np
import torch
import torch.nn.functional as F


def mse_loss(pred, target):
    """Mean Squared Error loss."""
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        return F.mse_loss(pred, target)
    else:
        return np.mean((pred - target) ** 2)


def mae_loss(pred, target):
    """Mean Absolute Error loss."""
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        return F.l1_loss(pred, target)
    else:
        return np.mean(np.abs(pred - target))


def relative_l2_error(pred, target):
    """
    Relative L2 error: ||pred - target||_2 / ||target||_2
    
    Args:
        pred: Predicted values
        target: Target values
    
    Returns:
        Relative L2 error
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        diff = pred - target
        rel_error = torch.norm(diff) / torch.norm(target)
        return rel_error.item()
    else:
        diff = pred - target
        rel_error = np.linalg.norm(diff) / np.linalg.norm(target)
        return rel_error


def relative_linf_error(pred, target):
    """
    Relative L∞ error: ||pred - target||_∞ / ||target||_∞
    
    Args:
        pred: Predicted values
        target: Target values
    
    Returns:
        Relative L∞ error
    """
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        diff = torch.abs(pred - target)
        rel_error = torch.max(diff) / torch.max(torch.abs(target))
        return rel_error.item()
    else:
        diff = np.abs(pred - target)
        rel_error = np.max(diff) / np.max(np.abs(target))
        return rel_error


def physics_residual_l2(coords, net, pde_func):
    """
    Compute L2 norm of physics residuals for PINNs.
    
    Args:
        coords: Input coordinates with requires_grad=True
        net: Neural network
        pde_func: Function that computes PDE residual
    
    Returns:
        L2 norm of residuals
    """
    u = net(coords)
    residual = pde_func(coords, u)
    return torch.norm(residual)


def conservation_error(pred, conservation_law):
    """
    Check conservation law violation.
    
    Args:
        pred: Predicted solution
        conservation_law: Function that computes conserved quantity
    
    Returns:
        Conservation error
    """
    conserved_quantity = conservation_law(pred)
    if isinstance(conserved_quantity, torch.Tensor):
        # For time-dependent problems, check if quantity is conserved over time
        if len(conserved_quantity.shape) > 0:
            return torch.std(conserved_quantity)
        else:
            return torch.abs(conserved_quantity)
    else:
        if len(conserved_quantity.shape) > 0:
            return np.std(conserved_quantity)
        else:
            return np.abs(conserved_quantity)


def energy_error(pred, exact, energy_func):
    """
    Compute energy error between predicted and exact solutions.
    
    Args:
        pred: Predicted solution
        exact: Exact solution
        energy_func: Function that computes energy
    
    Returns:
        Relative energy error
    """
    energy_pred = energy_func(pred)
    energy_exact = energy_func(exact)
    
    if isinstance(energy_pred, torch.Tensor):
        rel_error = torch.abs(energy_pred - energy_exact) / torch.abs(energy_exact)
        return rel_error.item()
    else:
        rel_error = np.abs(energy_pred - energy_exact) / np.abs(energy_exact)
        return rel_error


def pointwise_error_statistics(pred, target):
    """
    Compute pointwise error statistics.
    
    Args:
        pred: Predicted values
        target: Target values
    
    Returns:
        Dictionary with error statistics
    """
    if isinstance(pred, torch.Tensor):
        error = (pred - target).detach().cpu().numpy()
    else:
        error = pred - target
    
    abs_error = np.abs(error)
    
    stats = {
        'mean_error': np.mean(error),
        'std_error': np.std(error),
        'mean_abs_error': np.mean(abs_error),
        'max_abs_error': np.max(abs_error),
        'min_abs_error': np.min(abs_error),
        'rmse': np.sqrt(np.mean(error**2)),
        'relative_l2': relative_l2_error(pred, target),
        'relative_linf': relative_linf_error(pred, target)
    }
    
    return stats


def compute_derivatives(coords, u, order=1):
    """
    Compute derivatives using automatic differentiation.
    
    Args:
        coords: Input coordinates with requires_grad=True
        u: Output from neural network
        order: Order of derivative
    
    Returns:
        Derivatives with respect to each coordinate
    """
    grad_outputs = torch.ones_like(u)
    derivatives = []
    
    for i in range(coords.shape[1]):
        if order == 1:
            grad = torch.autograd.grad(u, coords, grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True)[0][:, i:i+1]
        elif order == 2:
            # First derivative
            grad_1 = torch.autograd.grad(u, coords, grad_outputs=grad_outputs,
                                       create_graph=True, retain_graph=True)[0][:, i:i+1]
            # Second derivative
            grad = torch.autograd.grad(grad_1, coords, grad_outputs=torch.ones_like(grad_1),
                                     create_graph=True, retain_graph=True)[0][:, i:i+1]
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")
        
        derivatives.append(grad)
    
    return derivatives


def gradient_based_metrics(coords, net):
    """
    Compute gradient-based metrics for solution quality.
    
    Args:
        coords: Input coordinates
        net: Neural network
    
    Returns:
        Dictionary with gradient metrics
    """
    coords.requires_grad_(True)
    u = net(coords)
    
    # Compute gradients
    grad_u = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
    
    # Gradient magnitude
    grad_magnitude = torch.norm(grad_u, dim=1)
    
    # Gradient smoothness (Laplacian of gradient magnitude)
    grad_mag_grad = torch.autograd.grad(grad_magnitude.sum(), coords, create_graph=True)[0]
    smoothness = torch.norm(grad_mag_grad, dim=1)
    
    metrics = {
        'mean_grad_magnitude': grad_magnitude.mean().item(),
        'max_grad_magnitude': grad_magnitude.max().item(),
        'grad_smoothness': smoothness.mean().item()
    }
    
    return metrics


def boundary_condition_error(coords_bc, pred_bc, target_bc, bc_type='dirichlet'):
    """
    Compute boundary condition violation error.
    
    Args:
        coords_bc: Boundary coordinates
        pred_bc: Predicted values at boundary
        target_bc: Target boundary values
        bc_type: Type of boundary condition
    
    Returns:
        Boundary condition error
    """
    if bc_type == 'dirichlet':
        return mse_loss(pred_bc, target_bc)
    elif bc_type == 'neumann':
        # For Neumann BC, we need to compute normal derivatives
        # This is a simplified version
        return mse_loss(pred_bc, target_bc)
    else:
        raise NotImplementedError(f"BC type {bc_type} not implemented")


def temporal_consistency_error(u_sequence, dt):
    """
    Check temporal consistency for time-dependent problems.
    
    Args:
        u_sequence: Sequence of solutions at different times [n_times, ...]
        dt: Time step
    
    Returns:
        Temporal consistency error
    """
    if isinstance(u_sequence, torch.Tensor):
        # Compute temporal derivatives using finite differences
        du_dt = (u_sequence[1:] - u_sequence[:-1]) / dt
        # Measure smoothness of temporal derivative
        consistency_error = torch.std(du_dt)
        return consistency_error.item()
    else:
        du_dt = (u_sequence[1:] - u_sequence[:-1]) / dt
        consistency_error = np.std(du_dt)
        return consistency_error


def evaluate_model_performance(pred, target, coords=None, net=None):
    """
    Comprehensive model performance evaluation.
    
    Args:
        pred: Predicted values
        target: Target values
        coords: Input coordinates (optional, for gradient metrics)
        net: Neural network (optional, for gradient metrics)
    
    Returns:
        Dictionary with comprehensive performance metrics
    """
    metrics = {}
    
    # Basic error metrics
    metrics.update(pointwise_error_statistics(pred, target))
    
    # Additional metrics if coordinates and network are provided
    if coords is not None and net is not None:
        metrics.update(gradient_based_metrics(coords, net))
    
    return metrics