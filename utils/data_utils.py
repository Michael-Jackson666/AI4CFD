"""
Data utilities for PDE solving with neural networks.
Comprehensive functions for data generation, loading, preprocessing, and augmentation.
Supports PINNs, DeepONet, FNO, TNN, and Transformer methods.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy import interpolate
from scipy.integrate import solve_ivp, odeint
from scipy.fft import fft, ifft, fft2, ifft2
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union


# ==============================================================================
# Basic Data Utilities
# ==============================================================================

def numpy_to_torch(data, device='cpu', dtype=torch.float32):
    """Convert numpy arrays to PyTorch tensors."""
    if isinstance(data, (list, tuple)):
        return [torch.tensor(arr, dtype=dtype, device=device) for arr in data]
    elif isinstance(data, dict):
        return {k: torch.tensor(v, dtype=dtype, device=device) for k, v in data.items()}
    else:
        return torch.tensor(data, dtype=dtype, device=device)


def torch_to_numpy(data):
    """Convert PyTorch tensors to numpy arrays."""
    if isinstance(data, (list, tuple)):
        return [tensor.detach().cpu().numpy() for tensor in data]
    elif isinstance(data, dict):
        return {k: v.detach().cpu().numpy() for k, v in data.items()}
    else:
        return data.detach().cpu().numpy()


def get_device(prefer_gpu=True):
    """Get the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    elif prefer_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_mesh_grid(domain: List[Tuple[float, float]], n_points: List[int], 
                     flatten=True) -> Tuple[np.ndarray, ...]:
    """
    Create mesh grid for multi-dimensional domain.
    
    Args:
        domain: List of (min, max) tuples for each dimension
        n_points: List of number of points per dimension
        flatten: Whether to flatten the output
    
    Returns:
        Coordinate arrays
    
    Example:
        >>> X, Y = create_mesh_grid([(0, 1), (0, 1)], [50, 50])
    """
    axes = [np.linspace(d[0], d[1], n) for d, n in zip(domain, n_points)]
    grids = np.meshgrid(*axes, indexing='ij')
    
    if flatten:
        coords = np.stack([g.ravel() for g in grids], axis=1)
        return coords
    return grids


def normalize_data(data, method='minmax', params=None):
    """
    Normalize data using various methods.
    
    Args:
        data: Input data
        method: 'minmax', 'standard', 'robust'
        params: Pre-computed normalization parameters
    
    Returns:
        Normalized data and parameters
    """
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()
    else:
        data_np = data
    
    if params is None:
        if method == 'minmax':
            params = {'min': data_np.min(axis=0), 'max': data_np.max(axis=0)}
            normalized = (data_np - params['min']) / (params['max'] - params['min'] + 1e-8)
        elif method == 'standard':
            params = {'mean': data_np.mean(axis=0), 'std': data_np.std(axis=0)}
            normalized = (data_np - params['mean']) / (params['std'] + 1e-8)
        elif method == 'robust':
            params = {'median': np.median(data_np, axis=0), 
                     'iqr': np.percentile(data_np, 75, axis=0) - np.percentile(data_np, 25, axis=0)}
            normalized = (data_np - params['median']) / (params['iqr'] + 1e-8)
    else:
        if method == 'minmax':
            normalized = (data_np - params['min']) / (params['max'] - params['min'] + 1e-8)
        elif method == 'standard':
            normalized = (data_np - params['mean']) / (params['std'] + 1e-8)
        elif method == 'robust':
            normalized = (data_np - params['median']) / (params['iqr'] + 1e-8)
    
    if isinstance(data, torch.Tensor):
        return torch.tensor(normalized, dtype=data.dtype), params
    return normalized, params


def denormalize_data(data, method, params):
    """Inverse of normalize_data."""
    if isinstance(data, torch.Tensor):
        data_np = data.numpy()
    else:
        data_np = data
    
    if method == 'minmax':
        denormalized = data_np * (params['max'] - params['min']) + params['min']
    elif method == 'standard':
        denormalized = data_np * params['std'] + params['mean']
    elif method == 'robust':
        denormalized = data_np * params['iqr'] + params['median']
    
    if isinstance(data, torch.Tensor):
        return torch.tensor(denormalized, dtype=data.dtype)
    return denormalized


# ==============================================================================
# PDE Data Generation
# ==============================================================================

def generate_1d_poisson_data(n_points=1000, domain=(-1, 1), source_func=None):
    """
    Generate training data for 1D Poisson equation: -d²u/dx² = f(x)
    
    Args:
        n_points: Number of training points
        domain: Tuple of (x_min, x_max)
        source_func: Function f(x), if None uses default
    
    Returns:
        x: Spatial coordinates
        u_exact: Exact solution (if available)
        f: Source term
    """
    x = np.linspace(domain[0], domain[1], n_points)
    
    if source_func is None:
        # Default: f(x) = -π²sin(πx), u(x) = sin(πx)
        f = -np.pi**2 * np.sin(np.pi * x)
        u_exact = np.sin(np.pi * x)
    else:
        f = source_func(x)
        u_exact = None
    
    return x.reshape(-1, 1), u_exact, f.reshape(-1, 1)


def generate_2d_poisson_data(n_points=50, domain=((-1, 1), (-1, 1)), source_func=None):
    """
    Generate training data for 2D Poisson equation: ∇²u = f(x,y)
    
    Args:
        n_points: Number of points per dimension
        domain: Tuple of ((x_min, x_max), (y_min, y_max))
    
    Returns:
        coords: Coordinates [n_points², 2]
        u_exact: Exact solution
        f: Source term
    """
    x = np.linspace(domain[0][0], domain[0][1], n_points)
    y = np.linspace(domain[1][0], domain[1][1], n_points)
    X, Y = np.meshgrid(x, y)
    
    if source_func is None:
        # Example: f(x,y) = -2π²sin(πx)sin(πy)
        f = -2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    else:
        f = source_func(X, Y)
        u_exact = None
    
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    
    return coords, u_exact.ravel() if u_exact is not None else None, f.ravel()


def generate_heat_equation_data(n_x=100, n_t=50, domain_x=(0, 1), domain_t=(0, 1),
                                 alpha=0.1, initial_condition=None):
    """
    Generate data for heat equation: ∂u/∂t = α∇²u
    
    Args:
        n_x: Number of spatial points
        n_t: Number of temporal points
        domain_x: Spatial domain
        domain_t: Temporal domain
        alpha: Thermal diffusivity
        initial_condition: Function for u(x, 0)
    
    Returns:
        x, t: Coordinate arrays
        u: Solution field [n_t, n_x]
    """
    x = np.linspace(domain_x[0], domain_x[1], n_x)
    t = np.linspace(domain_t[0], domain_t[1], n_t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    if initial_condition is None:
        # Default: sin(πx)
        u0 = np.sin(np.pi * x)
    else:
        u0 = initial_condition(x)
    
    # Analytical solution for sin(πx) initial condition
    u = np.zeros((n_t, n_x))
    for i, ti in enumerate(t):
        u[i] = np.exp(-alpha * np.pi**2 * ti) * np.sin(np.pi * x)
    
    return x, t, u


def generate_burgers_data(n_x=256, n_t=100, domain_x=(-1, 1), domain_t=(0, 1),
                          nu=0.01, initial_condition=None):
    """
    Generate data for Burgers' equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    Using spectral method for high accuracy.
    
    Args:
        n_x: Number of spatial points
        n_t: Number of temporal points
        domain_x: Spatial domain
        domain_t: Temporal domain
        nu: Viscosity
        initial_condition: Function for u(x, 0)
    
    Returns:
        x, t: Coordinate arrays
        u: Solution field [n_t, n_x]
    """
    L = domain_x[1] - domain_x[0]
    x = np.linspace(domain_x[0], domain_x[1], n_x, endpoint=False)
    t = np.linspace(domain_t[0], domain_t[1], n_t)
    
    if initial_condition is None:
        u0 = -np.sin(np.pi * x)
    else:
        u0 = initial_condition(x)
    
    # Wavenumbers
    k = np.fft.fftfreq(n_x) * n_x * 2 * np.pi / L
    
    # Time integration using RK4
    def rhs(u, t):
        u_hat = fft(u)
        u_x = np.real(ifft(1j * k * u_hat))
        u_xx = np.real(ifft(-k**2 * u_hat))
        return -u * u_x + nu * u_xx
    
    u = np.zeros((n_t, n_x))
    u[0] = u0
    
    dt = t[1] - t[0]
    u_current = u0.copy()
    
    for i in range(1, n_t):
        # RK4 step
        k1 = rhs(u_current, t[i-1])
        k2 = rhs(u_current + 0.5*dt*k1, t[i-1] + 0.5*dt)
        k3 = rhs(u_current + 0.5*dt*k2, t[i-1] + 0.5*dt)
        k4 = rhs(u_current + dt*k3, t[i])
        u_current = u_current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        u[i] = u_current
    
    return x, t, u


def generate_navier_stokes_data(n_x=64, n_y=64, n_t=20, Re=100, domain=((0, 2*np.pi), (0, 2*np.pi)),
                                 initial_vorticity=None):
    """
    Generate 2D Navier-Stokes data (vorticity-stream function formulation).
    
    Args:
        n_x, n_y: Number of spatial points
        n_t: Number of time steps
        Re: Reynolds number
        domain: Spatial domain
        initial_vorticity: Initial vorticity field function
    
    Returns:
        x, y, t: Coordinate arrays
        omega: Vorticity field [n_t, n_x, n_y]
    """
    Lx = domain[0][1] - domain[0][0]
    Ly = domain[1][1] - domain[1][0]
    
    x = np.linspace(0, Lx, n_x, endpoint=False)
    y = np.linspace(0, Ly, n_y, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Wavenumbers
    kx = np.fft.fftfreq(n_x) * n_x * 2 * np.pi / Lx
    ky = np.fft.fftfreq(n_y) * n_y * 2 * np.pi / Ly
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1  # Avoid division by zero
    
    # Initial vorticity
    if initial_vorticity is None:
        omega0 = np.sin(4*X) * np.cos(4*Y) + 0.5 * np.cos(X) * np.sin(2*Y)
    else:
        omega0 = initial_vorticity(X, Y)
    
    nu = 1 / Re
    dt = 0.01
    t = np.linspace(0, dt * n_t, n_t)
    
    omega = np.zeros((n_t, n_x, n_y))
    omega[0] = omega0
    
    omega_current = omega0.copy()
    
    for i in range(1, n_t):
        omega_hat = fft2(omega_current)
        
        # Stream function from vorticity: ∇²ψ = -ω
        psi_hat = -omega_hat / K2
        psi_hat[0, 0] = 0
        
        # Velocity from stream function
        u_hat = 1j * KY * psi_hat
        v_hat = -1j * KX * psi_hat
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))
        
        # Advection (spectral derivatives)
        omega_x = np.real(ifft2(1j * KX * omega_hat))
        omega_y = np.real(ifft2(1j * KY * omega_hat))
        
        # Diffusion (Laplacian in spectral space)
        laplacian_omega = np.real(ifft2(-K2 * omega_hat))
        
        # Time step (forward Euler for simplicity)
        omega_current = omega_current + dt * (-u * omega_x - v * omega_y + nu * laplacian_omega)
        omega[i] = omega_current
    
    return x, y, t, omega


def generate_wave_equation_data(n_x=100, n_t=100, domain_x=(0, 1), domain_t=(0, 2),
                                 c=1.0, initial_displacement=None, initial_velocity=None):
    """
    Generate data for wave equation: ∂²u/∂t² = c²∂²u/∂x²
    
    Args:
        n_x: Number of spatial points
        n_t: Number of temporal points
        c: Wave speed
        initial_displacement: Function for u(x, 0)
        initial_velocity: Function for ∂u/∂t(x, 0)
    
    Returns:
        x, t: Coordinate arrays
        u: Solution field [n_t, n_x]
    """
    x = np.linspace(domain_x[0], domain_x[1], n_x)
    t = np.linspace(domain_t[0], domain_t[1], n_t)
    
    if initial_displacement is None:
        # Default: Gaussian pulse
        u0 = np.exp(-100 * (x - 0.5)**2)
    else:
        u0 = initial_displacement(x)
    
    if initial_velocity is None:
        v0 = np.zeros_like(x)
    else:
        v0 = initial_velocity(x)
    
    # d'Alembert solution for simple cases
    u = np.zeros((n_t, n_x))
    
    # Use finite difference for general case
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = (c * dt / dx)**2
    
    u[0] = u0
    # First time step using initial velocity
    u[1] = u0 + dt * v0
    
    for n in range(1, n_t - 1):
        for i in range(1, n_x - 1):
            u[n+1, i] = 2*(1 - r)*u[n, i] + r*(u[n, i+1] + u[n, i-1]) - u[n-1, i]
        # Dirichlet BC
        u[n+1, 0] = 0
        u[n+1, -1] = 0
    
    return x, t, u


# ==============================================================================
# Boundary Condition Utilities
# ==============================================================================

def create_boundary_conditions(domain, n_points=100, bc_type='dirichlet', bc_value=0):
    """
    Create boundary condition data points for various domain shapes.
    
    Args:
        domain: Domain specification
        n_points: Number of boundary points per face
        bc_type: 'dirichlet', 'neumann', 'periodic'
        bc_value: Boundary value (scalar or function)
    
    Returns:
        bc_coords: Boundary coordinates
        bc_vals: Boundary values
    """
    if len(domain) == 1:  # 1D
        x_min, x_max = domain[0]
        bc_coords = np.array([[x_min], [x_max]])
        if callable(bc_value):
            bc_vals = bc_value(bc_coords.ravel())
        else:
            bc_vals = np.full(2, bc_value)
    
    elif len(domain) == 2:  # 2D
        (x_min, x_max), (y_min, y_max) = domain
        
        # Four edges
        x_left = np.full(n_points, x_min)
        y_left = np.linspace(y_min, y_max, n_points)
        
        x_right = np.full(n_points, x_max)
        y_right = np.linspace(y_min, y_max, n_points)
        
        x_bottom = np.linspace(x_min, x_max, n_points)
        y_bottom = np.full(n_points, y_min)
        
        x_top = np.linspace(x_min, x_max, n_points)
        y_top = np.full(n_points, y_max)
        
        bc_coords = np.vstack([
            np.column_stack([x_left, y_left]),
            np.column_stack([x_right, y_right]),
            np.column_stack([x_bottom, y_bottom]),
            np.column_stack([x_top, y_top])
        ])
        
        if callable(bc_value):
            bc_vals = bc_value(bc_coords[:, 0], bc_coords[:, 1])
        else:
            bc_vals = np.full(bc_coords.shape[0], bc_value)
    
    elif len(domain) == 3:  # 3D
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain
        n = int(np.sqrt(n_points))
        
        faces = []
        # x faces
        y_face, z_face = np.meshgrid(np.linspace(y_min, y_max, n), 
                                      np.linspace(z_min, z_max, n))
        faces.append(np.column_stack([np.full(n*n, x_min), y_face.ravel(), z_face.ravel()]))
        faces.append(np.column_stack([np.full(n*n, x_max), y_face.ravel(), z_face.ravel()]))
        
        # y faces
        x_face, z_face = np.meshgrid(np.linspace(x_min, x_max, n), 
                                      np.linspace(z_min, z_max, n))
        faces.append(np.column_stack([x_face.ravel(), np.full(n*n, y_min), z_face.ravel()]))
        faces.append(np.column_stack([x_face.ravel(), np.full(n*n, y_max), z_face.ravel()]))
        
        # z faces
        x_face, y_face = np.meshgrid(np.linspace(x_min, x_max, n), 
                                      np.linspace(y_min, y_max, n))
        faces.append(np.column_stack([x_face.ravel(), y_face.ravel(), np.full(n*n, z_min)]))
        faces.append(np.column_stack([x_face.ravel(), y_face.ravel(), np.full(n*n, z_max)]))
        
        bc_coords = np.vstack(faces)
        
        if callable(bc_value):
            bc_vals = bc_value(bc_coords[:, 0], bc_coords[:, 1], bc_coords[:, 2])
        else:
            bc_vals = np.full(bc_coords.shape[0], bc_value)
    
    return bc_coords, bc_vals


def create_initial_conditions(domain, n_points, ic_function, t0=0):
    """
    Create initial condition points for time-dependent PDEs.
    
    Args:
        domain: Spatial domain
        n_points: Number of points
        ic_function: Initial condition function
        t0: Initial time
    
    Returns:
        ic_coords: Coordinates with t=t0
        ic_vals: Initial values
    """
    dim = len(domain)
    
    if dim == 1:
        x = np.linspace(domain[0][0], domain[0][1], n_points)
        ic_coords = np.column_stack([x, np.full(n_points, t0)])
        ic_vals = ic_function(x)
    
    elif dim == 2:
        n = int(np.sqrt(n_points))
        x = np.linspace(domain[0][0], domain[0][1], n)
        y = np.linspace(domain[1][0], domain[1][1], n)
        X, Y = np.meshgrid(x, y)
        ic_coords = np.column_stack([X.ravel(), Y.ravel(), np.full(n*n, t0)])
        ic_vals = ic_function(X, Y).ravel()
    
    return ic_coords, ic_vals.reshape(-1, 1)


# ==============================================================================
# DeepONet Data Generation
# ==============================================================================

def generate_operator_data(operator_type='antiderivative', n_samples=1000, 
                           n_sensors=100, n_query=100, noise=0.0):
    """
    Generate training data for DeepONet operator learning.
    
    Args:
        operator_type: Type of operator to learn
        n_samples: Number of training samples
        n_sensors: Number of sensor points for input function
        n_query: Number of query points for output
        noise: Noise level to add to outputs
    
    Returns:
        u_sensors: Input function values at sensors [n_samples, n_sensors]
        y_query: Query locations [n_query, 1]
        G_u: Operator output at query locations [n_samples, n_query, 1]
    """
    from scipy.integrate import quad, cumtrapz
    
    x_sensors = np.linspace(0, 1, n_sensors)
    y_query = np.linspace(0, 1, n_query).reshape(-1, 1)
    
    u_data = []
    G_u_data = []
    
    for _ in range(n_samples):
        # Generate random input function (sum of sine waves)
        n_modes = np.random.randint(3, 8)
        coeffs = np.random.randn(n_modes) * 0.5
        freqs = np.random.uniform(1, 5, n_modes)
        phases = np.random.uniform(0, 2*np.pi, n_modes)
        
        def input_func(x):
            return sum(c * np.sin(f * np.pi * x + p) for c, f, p in zip(coeffs, freqs, phases))
        
        # Evaluate input at sensors
        u_sensors = np.array([input_func(x) for x in x_sensors])
        u_data.append(u_sensors)
        
        # Compute operator output
        if operator_type == 'antiderivative':
            # G(u)(y) = ∫₀ʸ u(s) ds
            G_u_query = []
            for y in y_query.ravel():
                integral, _ = quad(input_func, 0, y)
                G_u_query.append(integral)
            G_u_query = np.array(G_u_query)
        
        elif operator_type == 'derivative':
            # G(u)(y) = du/dx(y)
            h = 1e-5
            G_u_query = np.array([(input_func(y + h) - input_func(y - h)) / (2*h) 
                                   for y in y_query.ravel()])
        
        elif operator_type == 'laplacian_inverse':
            # Solve -u'' = f with u(0) = u(1) = 0
            # Using Green's function
            def green_function(x, s):
                if x <= s:
                    return x * (1 - s)
                else:
                    return s * (1 - x)
            
            G_u_query = []
            for y in y_query.ravel():
                integral = sum(green_function(y, s) * input_func(s) * (x_sensors[1] - x_sensors[0]) 
                              for s in x_sensors)
                G_u_query.append(integral)
            G_u_query = np.array(G_u_query)
        
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")
        
        # Add noise
        if noise > 0:
            G_u_query += noise * np.random.randn(*G_u_query.shape)
        
        G_u_data.append(G_u_query)
    
    u_sensors = np.array(u_data)
    G_u = np.array(G_u_data).reshape(n_samples, n_query, 1)
    
    return u_sensors, y_query, G_u


def generate_parametric_pde_data(pde_type='heat', n_samples=100, n_x=50, n_t=20, 
                                  param_range=(0.01, 0.1)):
    """
    Generate data for parametric PDE solving with DeepONet.
    
    Args:
        pde_type: Type of PDE
        n_samples: Number of parameter samples
        n_x: Number of spatial points
        n_t: Number of temporal points
        param_range: Range of PDE parameter
    
    Returns:
        params: Parameter values [n_samples, 1]
        solutions: PDE solutions [n_samples, n_t, n_x]
    """
    params = np.random.uniform(param_range[0], param_range[1], n_samples)
    solutions = []
    
    for param in params:
        if pde_type == 'heat':
            x, t, u = generate_heat_equation_data(n_x=n_x, n_t=n_t, alpha=param)
        elif pde_type == 'burgers':
            x, t, u = generate_burgers_data(n_x=n_x, n_t=n_t, nu=param)
        else:
            raise ValueError(f"Unknown PDE type: {pde_type}")
        
        solutions.append(u)
    
    return params.reshape(-1, 1), np.array(solutions)


# ==============================================================================
# FNO Data Generation
# ==============================================================================

def generate_fno_data(pde_type='darcy', n_samples=1000, resolution=64, 
                       train_ratio=0.8):
    """
    Generate training data for Fourier Neural Operator.
    
    Args:
        pde_type: Type of PDE ('darcy', 'navier_stokes', 'heat')
        n_samples: Number of samples
        resolution: Spatial resolution
        train_ratio: Ratio for train/test split
    
    Returns:
        train_data, test_data: Dictionaries with inputs and outputs
    """
    if pde_type == 'darcy':
        # Darcy flow: -∇·(a(x)∇u) = f
        inputs = []
        outputs = []
        
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        for _ in range(n_samples):
            # Random permeability field (log-normal)
            k = np.fft.fftfreq(resolution) * resolution * 2 * np.pi
            KX, KY = np.meshgrid(k, k, indexing='ij')
            K2 = KX**2 + KY**2 + 1
            
            # Random Gaussian field in Fourier space
            noise = np.random.randn(resolution, resolution) + 1j * np.random.randn(resolution, resolution)
            a_hat = noise / K2**(1.5)  # Smooth field
            a = np.real(ifft2(a_hat))
            a = np.exp(a)  # Log-normal
            
            # Simple solution (placeholder - real solver needed)
            f = np.sin(np.pi * X) * np.sin(np.pi * Y)
            u = f / (a + 1)  # Approximate solution
            
            inputs.append(a)
            outputs.append(u)
        
        inputs = np.array(inputs)
        outputs = np.array(outputs)
    
    elif pde_type == 'navier_stokes':
        # Generate vorticity data
        inputs = []
        outputs = []
        
        for _ in range(n_samples):
            x, y, t, omega = generate_navier_stokes_data(
                n_x=resolution, n_y=resolution, n_t=11, Re=1000
            )
            inputs.append(omega[0])  # Initial vorticity
            outputs.append(omega[10])  # Final vorticity
        
        inputs = np.array(inputs)
        outputs = np.array(outputs)
    
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    # Add channel dimension
    inputs = inputs[..., np.newaxis]
    outputs = outputs[..., np.newaxis]
    
    # Train/test split
    n_train = int(n_samples * train_ratio)
    
    train_data = {
        'input': torch.FloatTensor(inputs[:n_train]),
        'output': torch.FloatTensor(outputs[:n_train])
    }
    test_data = {
        'input': torch.FloatTensor(inputs[n_train:]),
        'output': torch.FloatTensor(outputs[n_train:])
    }
    
    return train_data, test_data


# ==============================================================================
# DataLoader Utilities
# ==============================================================================

class PDEDataset(Dataset):
    """Generic dataset for PDE data."""
    
    def __init__(self, inputs, outputs, transform=None):
        self.inputs = inputs
        self.outputs = outputs
        self.transform = transform
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class DeepONetDataset(Dataset):
    """Dataset for DeepONet training."""
    
    def __init__(self, u_sensors, y_query, G_u):
        self.u_sensors = u_sensors
        self.y_query = y_query
        self.G_u = G_u
    
    def __len__(self):
        return len(self.u_sensors)
    
    def __getitem__(self, idx):
        return self.u_sensors[idx], self.y_query, self.G_u[idx]


def create_training_dataloader(X, y, batch_size=32, shuffle=True, num_workers=0):
    """Create PyTorch DataLoader for training."""
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    if not isinstance(y, torch.Tensor):
        y = torch.FloatTensor(y)
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_fno_dataloader(data, batch_size=20, shuffle=True):
    """Create DataLoader for FNO training."""
    dataset = TensorDataset(data['input'], data['output'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_deeponet_dataloader(u_sensors, y_query, G_u, batch_size=32, shuffle=True):
    """Create DataLoader for DeepONet training."""
    dataset = DeepONetDataset(
        torch.FloatTensor(u_sensors),
        torch.FloatTensor(y_query),
        torch.FloatTensor(G_u)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)