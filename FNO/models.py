"""
Complete Fourier Neural Operator (FNO) models for different dimensionalities and applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import (
    FourierLayer1d, FourierLayer2d, FourierLayer3d,
    SpectralConv1d, SpectralConv2d, SpectralConv3d,
    AdaptiveSpectralConv2d, FactorizedSpectralConv2d
)


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator for problems like Burgers' equation, wave propagation.
    
    Args:
        modes: Number of Fourier modes to use
        width: Width of the spectral convolution layers
        input_dim: Input dimension (e.g., 2 for (u, x))
        output_dim: Output dimension (e.g., 1 for scalar field)
        n_layers: Number of Fourier layers
    """
    
    def __init__(self, modes=16, width=64, input_dim=2, output_dim=1, n_layers=4):
        super(FNO1d, self).__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer1d(width, width, modes) for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n_points, input_dim]
        
        Returns:
            Output tensor [batch_size, n_points, output_dim]
        """
        # Project to hidden dimension
        x = self.input_proj(x)  # [batch, n_points, width]
        x = x.permute(0, 2, 1)  # [batch, width, n_points]
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Project to output
        x = x.permute(0, 2, 1)  # [batch, n_points, width]
        x = self.output_proj(x)  # [batch, n_points, output_dim]
        
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for problems like Navier-Stokes, Darcy flow.
    
    Args:
        modes1, modes2: Number of Fourier modes in each direction
        width: Width of the spectral convolution layers
        input_channels: Number of input channels
        output_channels: Number of output channels
        n_layers: Number of Fourier layers
    """
    
    def __init__(self, modes1=12, modes2=12, width=32, input_channels=3, 
                 output_channels=1, n_layers=4):
        super(FNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, width, 1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer2d(width, width, modes1, modes2) 
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, output_channels, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_channels, height, width]
        
        Returns:
            Output tensor [batch_size, output_channels, height, width]
        """
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Project to output
        x = self.output_proj(x)
        
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for volumetric problems.
    
    Args:
        modes1, modes2, modes3: Number of Fourier modes in each direction
        width: Width of the spectral convolution layers
        input_channels: Number of input channels
        output_channels: Number of output channels
        n_layers: Number of Fourier layers
    """
    
    def __init__(self, modes1=8, modes2=8, modes3=8, width=32, 
                 input_channels=4, output_channels=1, n_layers=4):
        super(FNO3d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Conv3d(input_channels, width, 1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer3d(width, width, modes1, modes2, modes3)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv3d(width, 128, 1),
            nn.GELU(),
            nn.Conv3d(128, output_channels, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_channels, depth, height, width]
        
        Returns:
            Output tensor [batch_size, output_channels, depth, height, width]
        """
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
        
        # Project to output
        x = self.output_proj(x)
        
        return x


class UFNO2d(nn.Module):
    """
    U-Net style FNO with skip connections for better multi-scale representation.
    """
    
    def __init__(self, modes1=12, modes2=12, width=32, input_channels=3,
                 output_channels=1, n_layers=4):
        super(UFNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, width, 1)
        
        # Encoder (downsampling)
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(n_layers // 2):
            self.encoder_layers.append(
                FourierLayer2d(width, width, modes1, modes2)
            )
            self.downsample_layers.append(
                nn.Conv2d(width, width, 3, stride=2, padding=1)
            )
        
        # Bottleneck
        self.bottleneck = FourierLayer2d(width, width, modes1, modes2)
        
        # Decoder (upsampling)
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(n_layers // 2):
            self.upsample_layers.append(
                nn.ConvTranspose2d(width, width, 3, stride=2, padding=1, output_padding=1)
            )
            self.decoder_layers.append(
                FourierLayer2d(width * 2, width, modes1, modes2)  # *2 for skip connection
            )
        
        # Output projection
        self.output_proj = nn.Conv2d(width, output_channels, 1)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Encoder with skip connections
        skip_connections = []
        for encoder, downsample in zip(self.encoder_layers, self.downsample_layers):
            x = encoder(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for upsample, decoder in zip(self.upsample_layers, self.decoder_layers):
            x = upsample(x)
            skip = skip_connections.pop()
            
            # Resize if necessary
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Output projection
        x = self.output_proj(x)
        return x


class AdaptiveFNO2d(nn.Module):
    """
    Adaptive FNO that can adjust the number of modes during training.
    """
    
    def __init__(self, max_modes1=20, max_modes2=20, width=32, 
                 input_channels=3, output_channels=1, n_layers=4):
        super(AdaptiveFNO2d, self).__init__()
        
        self.max_modes1 = max_modes1
        self.max_modes2 = max_modes2
        self.width = width
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, width, 1)
        
        # Adaptive Fourier layers
        self.adaptive_layers = nn.ModuleList([
            AdaptiveSpectralConv2d(width, width, max_modes1, max_modes2)
            for _ in range(n_layers)
        ])
        
        # Pointwise convolutions
        self.pointwise_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Activation
        self.activation = nn.GELU()
        
        # Output projection
        self.output_proj = nn.Conv2d(width, output_channels, 1)
    
    def set_modes(self, modes1, modes2):
        """Set the number of active modes for all layers."""
        for layer in self.adaptive_layers:
            layer.set_modes(modes1, modes2)
    
    def get_current_modes(self):
        """Get current number of active modes."""
        return (self.adaptive_layers[0].current_modes1.item(),
                self.adaptive_layers[0].current_modes2.item())
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Apply adaptive Fourier layers
        for adaptive_layer, pointwise_layer in zip(self.adaptive_layers, self.pointwise_layers):
            x1 = adaptive_layer(x)
            x2 = pointwise_layer(x)
            x = self.activation(x1 + x2)
        
        # Output projection
        x = self.output_proj(x)
        return x


class FactorizedFNO2d(nn.Module):
    """
    Factorized FNO with reduced parameter count.
    """
    
    def __init__(self, modes1=12, modes2=12, width=32, input_channels=3,
                 output_channels=1, n_layers=4, rank_ratio=0.5):
        super(FactorizedFNO2d, self).__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        
        # Rank for factorization
        rank = int(min(modes1, modes2) * rank_ratio)
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, width, 1)
        
        # Factorized Fourier layers
        self.spectral_layers = nn.ModuleList([
            FactorizedSpectralConv2d(width, width, modes1, modes2, rank)
            for _ in range(n_layers)
        ])
        
        # Pointwise layers
        self.pointwise_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(n_layers)
        ])
        
        # Activation
        self.activation = nn.GELU()
        
        # Output projection
        self.output_proj = nn.Conv2d(width, output_channels, 1)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Apply factorized Fourier layers
        for spectral_layer, pointwise_layer in zip(self.spectral_layers, self.pointwise_layers):
            x1 = spectral_layer(x)
            x2 = pointwise_layer(x)
            x = self.activation(x1 + x2)
        
        # Output projection
        x = self.output_proj(x)
        return x


class MultiScaleFNO2d(nn.Module):
    """
    Multi-scale FNO that operates on multiple resolutions simultaneously.
    """
    
    def __init__(self, modes1=12, modes2=12, width=32, input_channels=3,
                 output_channels=1, n_layers=4, scales=[1, 2]):
        super(MultiScaleFNO2d, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Create FNO for each scale
        self.scale_models = nn.ModuleList([
            FNO2d(modes1//scale, modes2//scale, width, input_channels, width, n_layers)
            for scale in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Conv2d(width * self.num_scales, output_channels, 1)
    
    def forward(self, x):
        scale_outputs = []
        
        for i, (scale, model) in enumerate(zip(self.scales, self.scale_models)):
            if scale > 1:
                # Downsample input
                x_scaled = F.avg_pool2d(x, scale)
                out_scaled = model(x_scaled)
                # Upsample output
                out = F.interpolate(out_scaled, size=x.shape[-2:], 
                                  mode='bilinear', align_corners=False)
            else:
                out = model(x)
            
            scale_outputs.append(out)
        
        # Concatenate and fuse
        combined = torch.cat(scale_outputs, dim=1)
        output = self.fusion(combined)
        
        return output


class PINO2d(nn.Module):
    """
    Physics-Informed Neural Operator (PINO) combining FNO with physics constraints.
    """
    
    def __init__(self, modes1=12, modes2=12, width=32, input_channels=3,
                 output_channels=1, n_layers=4):
        super(PINO2d, self).__init__()
        
        self.fno = FNO2d(modes1, modes2, width, input_channels, output_channels, n_layers)
    
    def forward(self, x):
        return self.fno(x)
    
    def compute_derivatives(self, x, order=1):
        """
        Compute spatial derivatives for physics constraints.
        """
        x.requires_grad_(True)
        u = self.forward(x)
        
        # Compute gradients
        batch_size = u.shape[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, :, :, 0:1]
        u_y = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, :, :, 1:2]
        
        if order == 1:
            return u_x, u_y
        elif order == 2:
            u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, :, :, 0:1]
            u_yy = torch.autograd.grad(u_y.sum(), x, create_graph=True)[0][:, :, :, 1:2]
            u_xy = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, :, :, 1:2]
            return u_x, u_y, u_xx, u_yy, u_xy
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")


def create_fno_model(model_type='fno2d', **kwargs):
    """
    Factory function to create different types of FNO models.
    
    Args:
        model_type: Type of FNO model
        **kwargs: Additional arguments for model initialization
    
    Returns:
        FNO model instance
    """
    models = {
        'fno1d': FNO1d,
        'fno2d': FNO2d,
        'fno3d': FNO3d,
        'ufno2d': UFNO2d,
        'adaptive_fno2d': AdaptiveFNO2d,
        'factorized_fno2d': FactorizedFNO2d,
        'multiscale_fno2d': MultiScaleFNO2d,
        'pino2d': PINO2d
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


if __name__ == "__main__":
    # Test different FNO models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing FNO models...")
    
    # Test data
    batch_size = 4
    
    # 1D test
    print("\n1D FNO:")
    x1d = torch.randn(batch_size, 64, 2).to(device)  # [batch, points, (u, x)]
    model1d = create_fno_model('fno1d', modes=16, width=32, input_dim=2, output_dim=1).to(device)
    out1d = model1d(x1d)
    print(f"Input: {x1d.shape} -> Output: {out1d.shape}")
    print(f"Parameters: {sum(p.numel() for p in model1d.parameters()):,}")
    
    # 2D test
    print("\n2D FNO:")
    x2d = torch.randn(batch_size, 3, 32, 32).to(device)  # [batch, channels, height, width]
    model2d = create_fno_model('fno2d', modes1=12, modes2=12, width=32).to(device)
    out2d = model2d(x2d)
    print(f"Input: {x2d.shape} -> Output: {out2d.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2d.parameters()):,}")
    
    # U-FNO test
    print("\nU-FNO 2D:")
    ufno_model = create_fno_model('ufno2d', modes1=12, modes2=12, width=32).to(device)
    out_ufno = ufno_model(x2d)
    print(f"Input: {x2d.shape} -> Output: {out_ufno.shape}")
    print(f"Parameters: {sum(p.numel() for p in ufno_model.parameters()):,}")
    
    # Adaptive FNO test
    print("\nAdaptive FNO 2D:")
    adaptive_model = create_fno_model('adaptive_fno2d', max_modes1=20, max_modes2=20, width=32).to(device)
    adaptive_model.set_modes(8, 8)
    out_adaptive = adaptive_model(x2d)
    print(f"Input: {x2d.shape} -> Output: {out_adaptive.shape}")
    print(f"Current modes: {adaptive_model.get_current_modes()}")
    
    # Factorized FNO test
    print("\nFactorized FNO 2D:")
    factorized_model = create_fno_model('factorized_fno2d', modes1=12, modes2=12, width=32).to(device)
    out_factorized = factorized_model(x2d)
    print(f"Input: {x2d.shape} -> Output: {out_factorized.shape}")
    
    # Parameter comparison
    standard_params = sum(p.numel() for p in model2d.parameters())
    factorized_params = sum(p.numel() for p in factorized_model.parameters())
    print(f"Parameter reduction: {standard_params/factorized_params:.2f}x")
    
    # 3D test (smaller due to memory constraints)
    print("\n3D FNO:")
    x3d = torch.randn(2, 4, 16, 16, 16).to(device)  # Smaller for memory
    model3d = create_fno_model('fno3d', modes1=6, modes2=6, modes3=6, width=16).to(device)
    out3d = model3d(x3d)
    print(f"Input: {x3d.shape} -> Output: {out3d.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3d.parameters()):,}")
    
    print("\nAll FNO model tests completed successfully!")