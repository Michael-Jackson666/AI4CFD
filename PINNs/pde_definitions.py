"""
PDE definitions and residual functions for PINNs.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_derivatives(u, x, order=1):
    """
    Compute derivatives using automatic differentiation.
    
    Args:
        u: Output from neural network
        x: Input coordinates (requires_grad=True)
        order: Order of derivative (1 or 2)
    
    Returns:
        List of derivatives with respect to each coordinate
    """
    grad_outputs = torch.ones_like(u)
    derivatives = []
    
    for i in range(x.shape[1]):
        if order == 1:
            grad = torch.autograd.grad(u, x, grad_outputs=grad_outputs,
                                     create_graph=True, retain_graph=True)[0][:, i:i+1]
        elif order == 2:
            # First derivative
            grad_1 = torch.autograd.grad(u, x, grad_outputs=grad_outputs,
                                       create_graph=True, retain_graph=True)[0][:, i:i+1]
            # Second derivative
            grad = torch.autograd.grad(grad_1, x, grad_outputs=torch.ones_like(grad_1),
                                     create_graph=True, retain_graph=True)[0][:, i:i+1]
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")
        
        derivatives.append(grad)
    
    return derivatives


class PoissonPDE:
    """
    Poisson equation: ∇²u = f(x)
    """
    
    def __init__(self, source_func=None, dim=1):
        self.source_func = source_func
        self.dim = dim
    
    def residual(self, x, u):
        """Compute PDE residual."""
        # Compute second derivatives
        second_derivatives = compute_derivatives(u, x, order=2)
        
        # Laplacian
        laplacian = sum(second_derivatives)
        
        # Source term
        if self.source_func is not None:
            if self.dim == 1:
                f = self.source_func(x[:, 0:1])
            elif self.dim == 2:
                f = self.source_func(x[:, 0:1], x[:, 1:2])
            else:
                f = self.source_func(x)
        else:
            # Default source: f = -π²sin(πx) for 1D, -2π²sin(πx)sin(πy) for 2D
            if self.dim == 1:
                f = -torch.pi**2 * torch.sin(torch.pi * x[:, 0:1])
            elif self.dim == 2:
                f = -2 * torch.pi**2 * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
            else:
                raise NotImplementedError("Default source not implemented for dim > 2")
        
        # PDE residual: ∇²u - f = 0
        residual = laplacian - f
        return residual


class BurgersPDE:
    """
    Burgers' equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    """
    
    def __init__(self, nu=0.01):
        self.nu = nu  # Viscosity parameter
    
    def residual(self, xt, u):
        """Compute PDE residual."""
        # First derivatives
        first_derivatives = compute_derivatives(u, xt, order=1)
        u_t = first_derivatives[1]  # ∂u/∂t
        u_x = first_derivatives[0]  # ∂u/∂x
        
        # Second derivative in x
        u_xx = compute_derivatives(u, xt, order=2)[0]  # ∂²u/∂x²
        
        # Burgers' equation residual
        residual = u_t + u * u_x - self.nu * u_xx
        return residual


class HeatPDE:
    """
    Heat equation: ∂u/∂t = α∇²u
    """
    
    def __init__(self, alpha=1.0, dim=1):
        self.alpha = alpha  # Thermal diffusivity
        self.dim = dim
    
    def residual(self, xt, u):
        """Compute PDE residual."""
        # Time derivative
        u_t = compute_derivatives(u, xt, order=1)[-1]  # Last coordinate is time
        
        # Spatial second derivatives (Laplacian)
        second_derivatives = compute_derivatives(u, xt, order=2)
        laplacian = sum(second_derivatives[:-1])  # Exclude time derivative
        
        # Heat equation residual
        residual = u_t - self.alpha * laplacian
        return residual


class WavePDE:
    """
    Wave equation: ∂²u/∂t² = c²∇²u
    """
    
    def __init__(self, c=1.0, dim=1):
        self.c = c  # Wave speed
        self.dim = dim
    
    def residual(self, xt, u):
        """Compute PDE residual."""
        # Second derivatives
        second_derivatives = compute_derivatives(u, xt, order=2)
        
        # Time second derivative
        u_tt = second_derivatives[-1]  # Last coordinate is time
        
        # Spatial Laplacian
        laplacian = sum(second_derivatives[:-1])  # Exclude time derivative
        
        # Wave equation residual
        residual = u_tt - self.c**2 * laplacian
        return residual


class NavierStokesPDE:
    """
    Incompressible Navier-Stokes equations in 2D:
    ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
    ∂u/∂x + ∂v/∂y = 0
    """
    
    def __init__(self, nu=0.01):
        self.nu = nu  # Kinematic viscosity
    
    def residual(self, xyt, uvp):
        """
        Compute PDE residual for Navier-Stokes equations.
        
        Args:
            xyt: Coordinates [x, y, t]
            uvp: Solution [u, v, p] (velocity components and pressure)
        
        Returns:
            Residuals for momentum and continuity equations
        """
        u = uvp[:, 0:1]  # x-velocity
        v = uvp[:, 1:2]  # y-velocity
        p = uvp[:, 2:3]  # pressure
        
        # First derivatives
        u_derivs = compute_derivatives(u, xyt, order=1)
        v_derivs = compute_derivatives(v, xyt, order=1)
        p_derivs = compute_derivatives(p, xyt, order=1)
        
        u_x, u_y, u_t = u_derivs[0], u_derivs[1], u_derivs[2]
        v_x, v_y, v_t = v_derivs[0], v_derivs[1], v_derivs[2]
        p_x, p_y = p_derivs[0], p_derivs[1]
        
        # Second derivatives
        u_second = compute_derivatives(u, xyt, order=2)
        v_second = compute_derivatives(v, xyt, order=2)
        
        u_xx, u_yy = u_second[0], u_second[1]
        v_xx, v_yy = v_second[0], v_second[1]
        
        # Momentum equations
        residual_u = u_t + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        residual_v = v_t + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        
        # Continuity equation
        residual_continuity = u_x + v_y
        
        return torch.cat([residual_u, residual_v, residual_continuity], dim=1)


class ReactionDiffusionPDE:
    """
    Reaction-diffusion equation: ∂u/∂t = D∇²u + f(u)
    """
    
    def __init__(self, D=1.0, reaction_func=None):
        self.D = D  # Diffusion coefficient
        self.reaction_func = reaction_func
    
    def residual(self, xt, u):
        """Compute PDE residual."""
        # Time derivative
        u_t = compute_derivatives(u, xt, order=1)[-1]
        
        # Spatial Laplacian
        second_derivatives = compute_derivatives(u, xt, order=2)
        laplacian = sum(second_derivatives[:-1])  # Exclude time
        
        # Reaction term
        if self.reaction_func is not None:
            reaction = self.reaction_func(u)
        else:
            # Default: logistic growth f(u) = u(1-u)
            reaction = u * (1 - u)
        
        # Reaction-diffusion residual
        residual = u_t - self.D * laplacian - reaction
        return residual


class AdvectionDiffusionPDE:
    """
    Advection-diffusion equation: ∂u/∂t + v·∇u = D∇²u + f
    """
    
    def __init__(self, velocity, D=1.0, source_func=None):
        self.velocity = velocity  # Velocity field (function or constant)
        self.D = D  # Diffusion coefficient
        self.source_func = source_func
    
    def residual(self, xt, u):
        """Compute PDE residual."""
        # Derivatives
        first_derivatives = compute_derivatives(u, xt, order=1)
        second_derivatives = compute_derivatives(u, xt, order=2)
        
        u_t = first_derivatives[-1]  # Time derivative
        spatial_grads = first_derivatives[:-1]  # Spatial gradients
        laplacian = sum(second_derivatives[:-1])  # Spatial Laplacian
        
        # Velocity field
        if callable(self.velocity):
            v = self.velocity(xt[:, :-1])  # Exclude time coordinate
        else:
            v = self.velocity
        
        # Advection term: v·∇u
        if isinstance(v, torch.Tensor):
            advection = sum(v[:, i:i+1] * spatial_grads[i] for i in range(len(spatial_grads)))
        else:
            advection = sum(v[i] * spatial_grads[i] for i in range(len(spatial_grads)))
        
        # Source term
        if self.source_func is not None:
            source = self.source_func(xt)
        else:
            source = 0
        
        # Advection-diffusion residual
        residual = u_t + advection - self.D * laplacian - source
        return residual


class SchrodingerPDE:
    """
    Time-dependent Schrödinger equation: iℏ∂ψ/∂t = Ĥψ
    For simplicity, we consider the 1D case with harmonic oscillator potential.
    """
    
    def __init__(self, hbar=1.0, m=1.0, omega=1.0):
        self.hbar = hbar
        self.m = m
        self.omega = omega
    
    def residual(self, xt, psi):
        """
        Compute PDE residual for Schrödinger equation.
        Note: This is a complex-valued PDE, so psi should have real and imaginary parts.
        """
        # Split into real and imaginary parts
        psi_real = psi[:, 0:1]
        psi_imag = psi[:, 1:2]
        
        # Time derivatives
        psi_real_t = compute_derivatives(psi_real, xt, order=1)[1]
        psi_imag_t = compute_derivatives(psi_imag, xt, order=1)[1]
        
        # Spatial second derivatives
        psi_real_xx = compute_derivatives(psi_real, xt, order=2)[0]
        psi_imag_xx = compute_derivatives(psi_imag, xt, order=2)[0]
        
        # Potential energy (harmonic oscillator)
        x = xt[:, 0:1]
        V = 0.5 * self.m * self.omega**2 * x**2
        
        # Hamiltonian: H = -ℏ²/(2m)∇² + V
        H_psi_real = -self.hbar**2 / (2 * self.m) * psi_real_xx + V * psi_real
        H_psi_imag = -self.hbar**2 / (2 * self.m) * psi_imag_xx + V * psi_imag
        
        # Schrödinger equation: iℏ∂ψ/∂t = Ĥψ
        # Real part: ℏ∂ψ_imag/∂t = Ĥψ_real
        # Imaginary part: -ℏ∂ψ_real/∂t = Ĥψ_imag
        residual_real = self.hbar * psi_imag_t - H_psi_real
        residual_imag = -self.hbar * psi_real_t - H_psi_imag
        
        return torch.cat([residual_real, residual_imag], dim=1)


def create_pde(pde_type, **kwargs):
    """
    Factory function to create PDE objects.
    
    Args:
        pde_type: Type of PDE
        **kwargs: Additional parameters for PDE
    
    Returns:
        PDE object with residual method
    """
    pde_classes = {
        'poisson': PoissonPDE,
        'burgers': BurgersPDE,
        'heat': HeatPDE,
        'wave': WavePDE,
        'navier_stokes': NavierStokesPDE,
        'reaction_diffusion': ReactionDiffusionPDE,
        'advection_diffusion': AdvectionDiffusionPDE,
        'schrodinger': SchrodingerPDE
    }
    
    if pde_type not in pde_classes:
        raise ValueError(f"Unknown PDE type: {pde_type}. Available: {list(pde_classes.keys())}")
    
    return pde_classes[pde_type](**kwargs)


if __name__ == "__main__":
    # Test PDE residual computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test Poisson equation
    print("Testing Poisson PDE...")
    x = torch.randn(100, 1, requires_grad=True).to(device)
    u = torch.sin(torch.pi * x)
    
    poisson = create_pde('poisson', dim=1)
    residual = poisson.residual(x, u)
    print(f"Poisson residual shape: {residual.shape}")
    print(f"Residual mean: {residual.mean().item():.6f}")
    
    # Test Burgers equation
    print("\nTesting Burgers PDE...")
    xt = torch.randn(100, 2, requires_grad=True).to(device)
    u = torch.tanh(xt[:, 0:1] - xt[:, 1:2])
    
    burgers = create_pde('burgers', nu=0.01)
    residual = burgers.residual(xt, u)
    print(f"Burgers residual shape: {residual.shape}")
    print(f"Residual mean: {residual.mean().item():.6f}")
    
    print("\nAll PDE tests completed!")