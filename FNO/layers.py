"""
Core layers for Fourier Neural Operators (FNO).
Spectral convolution layers and supporting operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer for FNO."""
    
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to use
        
        # Initialize weights for spectral convolution
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        """
        Forward pass of spectral convolution.
        
        Args:
            x: Input tensor [batch_size, in_channels, n_points]
        
        Returns:
            Output tensor [batch_size, out_channels, n_points]
        """
        batch_size = x.shape[0]
        
        # Fourier transform
        x_ft = torch.fft.rfft(x)
        
        # Initialize output in Fourier domain
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1)//2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        # Spectral convolution (multiply in Fourier domain)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        # Inverse Fourier transform
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    """2D Spectral convolution layer for FNO."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in x-direction
        self.modes2 = modes2  # Number of Fourier modes in y-direction
        
        # Initialize weights
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 
                                  dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 
                                  dtype=torch.cfloat)
        )
    
    def forward(self, x):
        """
        Forward pass of 2D spectral convolution.
        
        Args:
            x: Input tensor [batch_size, in_channels, height, width]
        
        Returns:
            Output tensor [batch_size, out_channels, height, width]
        """
        batch_size = x.shape[0]
        
        # Fourier transform
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output in Fourier domain
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        
        # Spectral convolution for upper left modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes1, :self.modes2], 
            self.weights1
        )
        
        # Spectral convolution for upper right modes (conjugate symmetry)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, -self.modes1:, :self.modes2], 
            self.weights2
        )
        
        # Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    """3D Spectral convolution layer for FNO."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        # Initialize weights for different regions of Fourier domain
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                  dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                  dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                  dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                  dtype=torch.cfloat)
        )
    
    def forward(self, x):
        """Forward pass of 3D spectral convolution."""
        batch_size = x.shape[0]
        
        # Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels, 
                           x.size(-3), x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        
        # Apply spectral convolution to different regions
        # Upper left front
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights1
        )
        
        # Upper left back
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
            self.weights2
        )
        
        # Upper right front
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3],
            self.weights3
        )
        
        # Upper right back
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3],
            self.weights4
        )
        
        # Inverse Fourier transform
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FourierLayer1d(nn.Module):
    """Complete Fourier layer with spectral convolution and pointwise operation."""
    
    def __init__(self, in_channels, out_channels, modes, activation='gelu'):
        super(FourierLayer1d, self).__init__()
        
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        # Spectral path
        x1 = self.spectral_conv(x)
        
        # Pointwise path
        x2 = self.pointwise_conv(x)
        
        # Combine and apply activation
        return self.activation(x1 + x2)


class FourierLayer2d(nn.Module):
    """Complete 2D Fourier layer."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2, activation='gelu'):
        super(FourierLayer2d, self).__init__()
        
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        # Spectral path
        x1 = self.spectral_conv(x)
        
        # Pointwise path
        x2 = self.pointwise_conv(x)
        
        # Combine and apply activation
        return self.activation(x1 + x2)


class FourierLayer3d(nn.Module):
    """Complete 3D Fourier layer."""
    
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, activation='gelu'):
        super(FourierLayer3d, self).__init__()
        
        self.spectral_conv = SpectralConv3d(in_channels, out_channels, modes1, modes2, modes3)
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, 1)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.pointwise_conv(x)
        return self.activation(x1 + x2)


class AdaptiveSpectralConv2d(nn.Module):
    """
    Adaptive spectral convolution that can adjust the number of modes during training.
    """
    
    def __init__(self, in_channels, out_channels, max_modes1, max_modes2):
        super(AdaptiveSpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_modes1 = max_modes1
        self.max_modes2 = max_modes2
        
        # Current number of active modes (can be adjusted during training)
        self.register_buffer('current_modes1', torch.tensor(max_modes1 // 4))
        self.register_buffer('current_modes2', torch.tensor(max_modes2 // 4))
        
        # Initialize weights for maximum modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, max_modes1, max_modes2,
                                  dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, max_modes1, max_modes2,
                                  dtype=torch.cfloat)
        )
    
    def set_modes(self, modes1, modes2):
        """Set the number of active modes."""
        self.current_modes1 = torch.tensor(min(modes1, self.max_modes1))
        self.current_modes2 = torch.tensor(min(modes2, self.max_modes2))
    
    def forward(self, x):
        batch_size = x.shape[0]
        modes1 = int(self.current_modes1.item())
        modes2 = int(self.current_modes2.item())
        
        # Fourier transform
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        
        # Use only current number of modes
        out_ft[:, :, :modes1, :modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes1, :modes2],
            self.weights1[:, :, :modes1, :modes2]
        )
        
        out_ft[:, :, -modes1:, :modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -modes1:, :modes2],
            self.weights2[:, :, :modes1, :modes2]
        )
        
        # Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralNorm(nn.Module):
    """
    Spectral normalization layer for FNO to improve training stability.
    """
    
    def __init__(self, num_channels, eps=1e-5):
        super(SpectralNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        
        # Learnable parameters for normalization
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # Compute Fourier transform
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Compute spectral statistics
        power_spectrum = torch.abs(x_ft) ** 2
        mean_power = torch.mean(power_spectrum, dim=(-2, -1), keepdim=True)
        std_power = torch.std(power_spectrum, dim=(-2, -1), keepdim=True)
        
        # Normalize in spectral domain
        x_ft_norm = (power_spectrum - mean_power) / (std_power + self.eps)
        
        # Apply learnable transformation
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        
        # Transform back to spatial domain
        x_norm = torch.fft.irfft2(x_ft_norm * weight + bias, s=(x.size(-2), x.size(-1)))
        
        return x_norm


class FactorizedSpectralConv2d(nn.Module):
    """
    Factorized spectral convolution for reduced parameter count.
    Separates the 2D convolution into two 1D convolutions.
    """
    
    def __init__(self, in_channels, out_channels, modes1, modes2, rank=None):
        super(FactorizedSpectralConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Factorization rank (if None, use min of modes)
        if rank is None:
            rank = min(modes1, modes2) // 2
        self.rank = rank
        
        # Factorized weights
        self.scale = 1 / (in_channels * out_channels)
        
        # First direction (modes1)
        self.weights1_1 = nn.Parameter(
            self.scale * torch.rand(in_channels, rank, modes1, dtype=torch.cfloat)
        )
        self.weights1_2 = nn.Parameter(
            self.scale * torch.rand(rank, out_channels, modes2, dtype=torch.cfloat)
        )
        
        # Second direction (conjugate part)
        self.weights2_1 = nn.Parameter(
            self.scale * torch.rand(in_channels, rank, modes1, dtype=torch.cfloat)
        )
        self.weights2_2 = nn.Parameter(
            self.scale * torch.rand(rank, out_channels, modes2, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Fourier transform
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
                           dtype=torch.cfloat, device=x.device)
        
        # Factorized convolution for upper modes
        # First apply convolution in first dimension
        temp = torch.einsum("bixy,irx->brxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1_1)
        # Then apply convolution in second dimension
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("brxy,roy->boxy", temp, self.weights1_2)
        
        # Factorized convolution for conjugate modes
        temp = torch.einsum("bixy,irx->brxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2_1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("brxy,roy->boxy", temp, self.weights2_2)
        
        # Inverse Fourier transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


if __name__ == "__main__":
    # Test spectral convolution layers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing FNO layers...")
    
    # Test 1D spectral convolution
    print("\n1D Spectral Convolution:")
    batch_size, channels, length = 4, 8, 64
    x1d = torch.randn(batch_size, channels, length).to(device)
    
    conv1d = SpectralConv1d(channels, channels, modes=16).to(device)
    out1d = conv1d(x1d)
    print(f"Input: {x1d.shape} -> Output: {out1d.shape}")
    
    # Test 2D spectral convolution
    print("\n2D Spectral Convolution:")
    batch_size, channels, height, width = 4, 8, 32, 32
    x2d = torch.randn(batch_size, channels, height, width).to(device)
    
    conv2d = SpectralConv2d(channels, channels, modes1=12, modes2=12).to(device)
    out2d = conv2d(x2d)
    print(f"Input: {x2d.shape} -> Output: {out2d.shape}")
    
    # Test Fourier layer
    print("\n2D Fourier Layer:")
    fourier_layer = FourierLayer2d(channels, channels, modes1=12, modes2=12).to(device)
    out_fourier = fourier_layer(x2d)
    print(f"Input: {x2d.shape} -> Output: {out_fourier.shape}")
    
    # Test adaptive spectral convolution
    print("\nAdaptive Spectral Convolution:")
    adaptive_conv = AdaptiveSpectralConv2d(channels, channels, max_modes1=20, max_modes2=20).to(device)
    adaptive_conv.set_modes(8, 8)
    out_adaptive = adaptive_conv(x2d)
    print(f"Input: {x2d.shape} -> Output: {out_adaptive.shape}")
    print(f"Current modes: {adaptive_conv.current_modes1.item()}, {adaptive_conv.current_modes2.item()}")
    
    # Test factorized spectral convolution
    print("\nFactorized Spectral Convolution:")
    factorized_conv = FactorizedSpectralConv2d(channels, channels, modes1=12, modes2=12, rank=6).to(device)
    out_factorized = factorized_conv(x2d)
    print(f"Input: {x2d.shape} -> Output: {out_factorized.shape}")
    
    # Parameter count comparison
    standard_params = sum(p.numel() for p in conv2d.parameters())
    factorized_params = sum(p.numel() for p in factorized_conv.parameters())
    print(f"\nParameter count:")
    print(f"Standard: {standard_params:,}")
    print(f"Factorized: {factorized_params:,}")
    print(f"Reduction: {standard_params/factorized_params:.2f}x")
    
    print("\nAll layer tests completed successfully!")