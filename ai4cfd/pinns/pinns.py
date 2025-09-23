"""
Physics-Informed Neural Networks (PINNs) for solving PDEs

PINNs incorporate physics laws in forms of PDEs into the training process
of neural networks through the loss function.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple, List


class PINNs(nn.Module):
    """
    Physics-Informed Neural Networks implementation.
    
    This class implements a basic PINN architecture for solving partial
    differential equations by incorporating physics constraints into the
    neural network training process.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 50,
        output_dim: int = 1,
        num_layers: int = 4,
        activation: str = "tanh"
    ):
        """
        Initialize PINN model.
        
        Args:
            input_dim: Dimension of input (e.g., space + time coordinates)
            hidden_dim: Number of neurons in hidden layers
            output_dim: Dimension of output (e.g., velocity, pressure)
            num_layers: Number of hidden layers
            activation: Activation function ("tanh", "relu", "sigmoid")
        """
        super(PINNs, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Define activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def compute_derivatives(self, x: torch.Tensor, order: Tuple[int, ...]) -> torch.Tensor:
        """
        Compute derivatives of the network output with respect to input.
        
        Args:
            x: Input tensor with requires_grad=True
            order: Tuple specifying derivative order for each input dimension
            
        Returns:
            Derivative tensor
        """
        u = self.forward(x)
        
        # Compute gradients iteratively
        for i, ord_i in enumerate(order):
            for _ in range(ord_i):
                if u.shape[0] == 1:
                    grad = torch.autograd.grad(
                        u, x, 
                        grad_outputs=torch.ones_like(u),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                else:
                    grad = torch.autograd.grad(
                        u.sum(), x,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                u = grad[:, i:i+1]
        
        return u
    
    def pde_loss(
        self, 
        x: torch.Tensor, 
        pde_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute PDE residual loss.
        
        Args:
            x: Input coordinates
            pde_func: Function that computes PDE residual given (x, u)
            
        Returns:
            PDE loss tensor
        """
        x.requires_grad_(True)
        u = self.forward(x)
        residual = pde_func(x, u)
        return torch.mean(residual ** 2)
    
    def boundary_loss(
        self,
        x_boundary: torch.Tensor,
        u_boundary: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            x_boundary: Boundary coordinates
            u_boundary: Known boundary values
            
        Returns:
            Boundary loss tensor
        """
        u_pred = self.forward(x_boundary)
        return torch.mean((u_pred - u_boundary) ** 2)
    
    def initial_loss(
        self,
        x_initial: torch.Tensor,
        u_initial: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            x_initial: Initial coordinates
            u_initial: Known initial values
            
        Returns:
            Initial condition loss tensor
        """
        u_pred = self.forward(x_initial)
        return torch.mean((u_pred - u_initial) ** 2)
    
    def data_loss(
        self,
        x_data: torch.Tensor,
        u_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute data fitting loss.
        
        Args:
            x_data: Data coordinates
            u_data: Known data values
            
        Returns:
            Data loss tensor
        """
        u_pred = self.forward(x_data)
        return torch.mean((u_pred - u_data) ** 2)


def heat_equation_residual(x: torch.Tensor, u: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Residual function for 1D heat equation: du/dt - alpha * d²u/dx² = 0
    
    Args:
        x: Input tensor [t, x]
        u: Network output
        alpha: Thermal diffusivity
        
    Returns:
        PDE residual
    """
    # Compute derivatives
    u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]  # du/dt
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]  # du/dx
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 1:2]  # d²u/dx²
    
    return u_t - alpha * u_xx


def burgers_equation_residual(x: torch.Tensor, u: torch.Tensor, nu: float = 0.01) -> torch.Tensor:
    """
    Residual function for Burgers' equation: du/dt + u * du/dx - nu * d²u/dx² = 0
    
    Args:
        x: Input tensor [t, x]
        u: Network output
        nu: Viscosity parameter
        
    Returns:
        PDE residual
    """
    # Compute derivatives
    u_t = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]  # du/dt
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]  # du/dx
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 1:2]  # d²u/dx²
    
    return u_t + u * u_x - nu * u_xx