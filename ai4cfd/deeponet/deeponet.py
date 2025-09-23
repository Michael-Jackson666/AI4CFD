"""
Deep Operator Networks (DeepONet) for learning operators

DeepONet learns operators that map functions to functions by using
a branch network to encode the input function and a trunk network
to encode the evaluation coordinates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


class DeepONet(nn.Module):
    """
    Deep Operator Network implementation.
    
    DeepONet learns nonlinear operators by approximating them using
    two deep neural networks: a branch net and a trunk net.
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        branch_hidden_dims: List[int] = [100, 100],
        trunk_hidden_dims: List[int] = [100, 100],
        output_dim: int = 100,
        activation: str = "relu"
    ):
        """
        Initialize DeepONet model.
        
        Args:
            branch_input_dim: Input dimension for branch network (function sensors)
            trunk_input_dim: Input dimension for trunk network (evaluation points)
            branch_hidden_dims: Hidden layer dimensions for branch network
            trunk_hidden_dims: Hidden layer dimensions for trunk network
            output_dim: Output dimension (basis functions)
            activation: Activation function
        """
        super(DeepONet, self).__init__()
        
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.output_dim = output_dim
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build branch network
        branch_layers = []
        prev_dim = branch_input_dim
        
        for hidden_dim in branch_hidden_dims:
            branch_layers.append(nn.Linear(prev_dim, hidden_dim))
            branch_layers.append(self.activation)
            prev_dim = hidden_dim
        
        branch_layers.append(nn.Linear(prev_dim, output_dim))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Build trunk network
        trunk_layers = []
        prev_dim = trunk_input_dim
        
        for hidden_dim in trunk_hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            trunk_layers.append(self.activation)
            prev_dim = hidden_dim
        
        trunk_layers.append(nn.Linear(prev_dim, output_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        branch_input: torch.Tensor, 
        trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DeepONet.
        
        Args:
            branch_input: Input functions (batch_size, branch_input_dim)
            trunk_input: Evaluation coordinates (batch_size, trunk_input_dim)
            
        Returns:
            Operator output (batch_size, 1)
        """
        # Get branch and trunk outputs
        branch_out = self.branch_net(branch_input)  # (batch_size, output_dim)
        trunk_out = self.trunk_net(trunk_input)     # (batch_size, output_dim)
        
        # Compute dot product and add bias
        output = torch.sum(branch_out * trunk_out, dim=1, keepdim=True) + self.bias
        
        return output
    
    def operator_loss(
        self,
        branch_input: torch.Tensor,
        trunk_input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute operator learning loss.
        
        Args:
            branch_input: Input functions
            trunk_input: Evaluation coordinates  
            target: Target operator output
            
        Returns:
            MSE loss
        """
        pred = self.forward(branch_input, trunk_input)
        return torch.mean((pred - target) ** 2)


class PODDeepONet(nn.Module):
    """
    POD-DeepONet with Proper Orthogonal Decomposition.
    
    Uses POD to reduce the dimension of the trunk network output
    for more efficient operator learning.
    """
    
    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int,
        pod_modes: int = 50,
        branch_hidden_dims: List[int] = [100, 100],
        trunk_hidden_dims: List[int] = [100, 100],
        activation: str = "relu"
    ):
        """
        Initialize POD-DeepONet model.
        
        Args:
            branch_input_dim: Input dimension for branch network
            trunk_input_dim: Input dimension for trunk network
            pod_modes: Number of POD modes
            branch_hidden_dims: Hidden layer dimensions for branch network
            trunk_hidden_dims: Hidden layer dimensions for trunk network
            activation: Activation function
        """
        super(PODDeepONet, self).__init__()
        
        self.pod_modes = pod_modes
        
        # Define activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build branch network
        branch_layers = []
        prev_dim = branch_input_dim
        
        for hidden_dim in branch_hidden_dims:
            branch_layers.append(nn.Linear(prev_dim, hidden_dim))
            branch_layers.append(self.activation)
            prev_dim = hidden_dim
        
        branch_layers.append(nn.Linear(prev_dim, pod_modes))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Build trunk network
        trunk_layers = []
        prev_dim = trunk_input_dim
        
        for hidden_dim in trunk_hidden_dims:
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            trunk_layers.append(self.activation)
            prev_dim = hidden_dim
        
        trunk_layers.append(nn.Linear(prev_dim, pod_modes))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # POD basis (to be set after training data analysis)
        self.register_buffer('pod_basis', torch.eye(pod_modes))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def set_pod_basis(self, basis: torch.Tensor):
        """
        Set the POD basis from training data.
        
        Args:
            basis: POD basis matrix (output_dim, pod_modes)
        """
        self.pod_basis = basis
    
    def forward(
        self, 
        branch_input: torch.Tensor, 
        trunk_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through POD-DeepONet.
        
        Args:
            branch_input: Input functions (batch_size, branch_input_dim)
            trunk_input: Evaluation coordinates (batch_size, trunk_input_dim)
            
        Returns:
            Operator output (batch_size, output_dim)
        """
        # Get branch and trunk outputs
        branch_out = self.branch_net(branch_input)  # (batch_size, pod_modes)
        trunk_out = self.trunk_net(trunk_input)     # (batch_size, pod_modes)
        
        # Compute coefficients
        coeffs = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)  # (batch_size, 1)
        
        # Reconstruct using POD basis
        output = coeffs @ self.pod_basis.T  # (batch_size, output_dim)
        
        return output


def compute_pod_basis(data: torch.Tensor, n_modes: int) -> torch.Tensor:
    """
    Compute POD basis from training data.
    
    Args:
        data: Training data (n_samples, spatial_dim)
        n_modes: Number of POD modes to retain
        
    Returns:
        POD basis matrix (spatial_dim, n_modes)
    """
    # Center the data
    data_mean = torch.mean(data, dim=0, keepdim=True)
    data_centered = data - data_mean
    
    # Compute SVD
    U, S, V = torch.svd(data_centered.T)
    
    # Return first n_modes
    return U[:, :n_modes]