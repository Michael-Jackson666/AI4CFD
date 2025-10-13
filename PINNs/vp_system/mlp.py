"""
Multi-Layer Perceptron (MLP) module for PINN.
Defines the neural network architecture with Softplus activation to ensure f >= 0.
"""

import torch.nn as nn


class MLP(nn.Module):
    """
    Defines the MLP for the PINN, ensuring f >= 0 with Softplus activation.
    
    Args:
        input_dim (int): Number of input features (t, x, v) = 3
        output_dim (int): Number of output features (f) = 1
        layers (int): Number of hidden layers
        neurons (int): Number of neurons per hidden layer
    """
    def __init__(self, input_dim=3, output_dim=1, layers=12, neurons=512):
        super(MLP, self).__init__()
        
        # Build the network layer by layer
        modules = [nn.Linear(input_dim, neurons), nn.Tanh()]
        
        # Add hidden layers
        for _ in range(layers - 2):
            modules.append(nn.Linear(neurons, neurons))
            modules.append(nn.Tanh())
        
        # Output layer with Softplus to ensure non-negative values
        modules.append(nn.Linear(neurons, output_dim))
        modules.append(nn.Softplus())
        
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        return self.net(x)