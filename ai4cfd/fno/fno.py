"""
Fourier Neural Operator (FNO) for solving PDEs

FNO learns operators in Fourier space and is particularly effective
for solving PDEs with periodic boundary conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Initialize 1D spectral convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            modes: Number of Fourier modes to use
        """
        super(SpectralConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Args:
            x: Input tensor (batch_size, in_channels, spatial_dim)
            
        Returns:
            Output tensor (batch_size, out_channels, spatial_dim)
        """
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Initialize 2D spectral convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super(SpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2D spectral convolution.
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor (batch_size, out_channels, height, width)
        """
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes1, :self.modes2], 
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, -self.modes1:, :self.modes2], 
            self.weights2
        )
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator
    """
    
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        input_dim: int = 1,
        output_dim: int = 1
    ):
        """
        Initialize 1D FNO.
        
        Args:
            modes: Number of Fourier modes
            width: Hidden channel dimension
            num_layers: Number of FNO layers
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        
        # Input projection
        self.fc0 = nn.Linear(input_dim, self.width)
        
        # Fourier layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes)
            for _ in range(self.num_layers)
        ])
        
        # Skip connections
        self.skip_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 1D FNO.
        
        Args:
            x: Input tensor (batch_size, spatial_dim, input_dim)
            
        Returns:
            Output tensor (batch_size, spatial_dim, output_dim)
        """
        # Input projection
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # (batch, width, spatial)
        
        # Fourier layers
        for i in range(self.num_layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.skip_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        
        # Output projection
        x = x.permute(0, 2, 1)  # (batch, spatial, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator
    """
    
    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        num_layers: int = 4,
        input_dim: int = 3,
        output_dim: int = 1
    ):
        """
        Initialize 2D FNO.
        
        Args:
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
            width: Hidden channel dimension
            num_layers: Number of FNO layers
            input_dim: Input dimension (e.g., 3 for x, y, and initial condition)
            output_dim: Output dimension
        """
        super(FNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        
        # Input projection
        self.fc0 = nn.Linear(input_dim, self.width)
        
        # Fourier layers
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.num_layers)
        ])
        
        # Skip connections
        self.skip_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2D FNO.
        
        Args:
            x: Input tensor (batch_size, height, width, input_dim)
            
        Returns:
            Output tensor (batch_size, height, width, output_dim)
        """
        # Input projection
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (batch, width, height, spatial_width)
        
        # Fourier layers
        for i in range(self.num_layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.skip_layers[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        
        # Output projection
        x = x.permute(0, 2, 3, 1)  # (batch, height, spatial_width, width)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for time-dependent problems
    """
    
    def __init__(
        self,
        modes1: int = 8,
        modes2: int = 8, 
        modes3: int = 8,
        width: int = 32,
        num_layers: int = 4,
        input_dim: int = 4,
        output_dim: int = 1
    ):
        """
        Initialize 3D FNO.
        
        Args:
            modes1: Number of Fourier modes in first spatial dimension
            modes2: Number of Fourier modes in second spatial dimension
            modes3: Number of Fourier modes in time dimension
            width: Hidden channel dimension
            num_layers: Number of FNO layers
            input_dim: Input dimension (e.g., 4 for x, y, t, and initial condition)
            output_dim: Output dimension
        """
        super(FNO3d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_layers = num_layers
        
        # Input projection
        self.fc0 = nn.Linear(input_dim, self.width)
        
        # Note: For 3D, we would need SpectralConv3d which is more complex
        # For simplicity, using 2D operations here
        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.num_layers)
        ])
        
        self.skip_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D FNO.
        Note: This is a simplified version that processes spatial dimensions.
        """
        # Process each time step separately for this simplified version
        batch_size, height, width, time_steps, input_dim = x.shape
        
        outputs = []
        for t in range(time_steps):
            x_t = x[:, :, :, t, :]  # (batch, height, width, input_dim)
            
            # Input projection
            x_t = self.fc0(x_t)
            x_t = x_t.permute(0, 3, 1, 2)  # (batch, width, height, spatial_width)
            
            # Fourier layers
            for i in range(self.num_layers):
                x1 = self.spectral_layers[i](x_t)
                x2 = self.skip_layers[i](x_t)
                x_t = x1 + x2
                if i < self.num_layers - 1:
                    x_t = F.gelu(x_t)
            
            # Output projection
            x_t = x_t.permute(0, 2, 3, 1)  # (batch, height, spatial_width, width)
            x_t = self.fc1(x_t)
            x_t = F.gelu(x_t)
            x_t = self.fc2(x_t)
            
            outputs.append(x_t)
        
        return torch.stack(outputs, dim=3)  # (batch, height, width, time_steps, output_dim)