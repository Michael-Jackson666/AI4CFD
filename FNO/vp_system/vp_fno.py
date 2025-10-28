"""
Fourier Neural Operator (FNO) for Vlasov-Poisson System.
Specialized FNO architecture for phase space evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers import SpectralConv2d, FourierLayer2d
from config import FNOConfig


class VPFNO2d(nn.Module):
    """
    FNO for Vlasov-Poisson phase space evolution.
    
    Maps initial distribution f(x,v,0) to distribution at later time f(x,v,t).
    
    Args:
        config: FNO configuration object
    """
    
    def __init__(self, config: FNOConfig):
        super(VPFNO2d, self).__init__()
        
        self.config = config
        self.modes_x = config.modes_x
        self.modes_v = config.modes_v
        self.width = config.width
        self.n_layers = config.n_layers
        
        # Input projection: (f0, x, v) -> hidden
        # Input has 3 channels: initial distribution + coordinates
        self.input_proj = nn.Conv2d(3, self.width, 1)
        
        # Fourier layers for phase space evolution
        self.fourier_layers = nn.ModuleList([
            FourierLayer2d(self.width, self.width, self.modes_x, self.modes_v,
                          activation=config.activation)
            for _ in range(self.n_layers)
        ])
        
        # Output projection: hidden -> f(t)
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.width, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, config.output_channels, 1)
        )
        
        # Normalization layers
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm2d(self.width) for _ in range(self.n_layers)
        ])
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, 3, Nx, Nv]
               Channel 0: Initial distribution f(x,v,0)
               Channel 1: x coordinates (normalized)
               Channel 2: v coordinates (normalized)
        
        Returns:
            Output tensor [batch, 1, Nx, Nv]: Predicted distribution f(x,v,t)
        """
        # Input projection
        x = self.input_proj(x)  # [batch, width, Nx, Nv]
        
        # Apply Fourier layers with residual connections
        for i, (fourier_layer, norm) in enumerate(zip(self.fourier_layers, self.norm_layers)):
            residual = x
            x = fourier_layer(x)
            x = norm(x)
            
            # Residual connection
            if i > 0:  # Skip first layer for residual
                x = x + residual
        
        # Output projection
        x = self.output_proj(x)  # [batch, 1, Nx, Nv]
        
        return x
    
    def predict_sequence(self, f0, x_grid, v_grid, time_steps):
        """
        Predict distribution at multiple time steps using rollout.
        
        Args:
            f0: Initial distribution [batch, 1, Nx, Nv]
            x_grid: Spatial coordinates [batch, 1, Nx, Nv]
            v_grid: Velocity coordinates [batch, 1, Nx, Nv]
            time_steps: List of time step indices
        
        Returns:
            Dictionary of predictions at each time step
        """
        predictions = {}
        current_f = f0
        
        for t in time_steps:
            # Prepare input
            input_data = torch.cat([current_f, x_grid, v_grid], dim=1)
            
            # Predict next state
            current_f = self.forward(input_data)
            predictions[t] = current_f.clone()
        
        return predictions


class UFNO_VP(nn.Module):
    """
    U-Net style FNO for Vlasov-Poisson with skip connections.
    Better for capturing multi-scale features in phase space.
    """
    
    def __init__(self, config: FNOConfig):
        super(UFNO_VP, self).__init__()
        
        self.config = config
        self.modes_x = config.modes_x
        self.modes_v = config.modes_v
        self.width = config.width
        self.n_layers = config.n_layers
        
        # Input projection
        self.input_proj = nn.Conv2d(3, self.width, 1)
        
        # Encoder (downsampling path)
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(self.n_layers // 2):
            self.encoder_layers.append(
                FourierLayer2d(self.width, self.width, 
                             self.modes_x >> i, self.modes_v >> i)
            )
            self.downsample_layers.append(
                nn.Conv2d(self.width, self.width, 3, stride=2, padding=1)
            )
        
        # Bottleneck
        self.bottleneck = FourierLayer2d(
            self.width, self.width,
            self.modes_x >> (self.n_layers // 2),
            self.modes_v >> (self.n_layers // 2)
        )
        
        # Decoder (upsampling path)
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(self.n_layers // 2):
            self.upsample_layers.append(
                nn.ConvTranspose2d(self.width, self.width, 3, 
                                  stride=2, padding=1, output_padding=1)
            )
            # *2 for concatenated skip connection
            self.decoder_layers.append(
                FourierLayer2d(self.width * 2, self.width,
                             self.modes_x >> (self.n_layers // 2 - i - 1),
                             self.modes_v >> (self.n_layers // 2 - i - 1))
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.width, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, config.output_channels, 1)
        )
    
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
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], 
                                mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Output projection
        x = self.output_proj(x)
        return x


class PhysicsInformedFNO_VP(nn.Module):
    """
    Physics-Informed FNO for Vlasov-Poisson.
    Includes methods to compute physics residuals.
    """
    
    def __init__(self, config: FNOConfig):
        super(PhysicsInformedFNO_VP, self).__init__()
        
        self.fno = VPFNO2d(config)
        self.config = config
    
    def forward(self, x):
        return self.fno(x)
    
    def compute_electric_field(self, f, dx, dv):
        """
        Compute electric field from distribution function via Poisson equation.
        
        Args:
            f: Distribution function [batch, 1, Nx, Nv]
            dx: Spatial grid spacing
            dv: Velocity grid spacing
        
        Returns:
            E: Electric field [batch, 1, Nx]
        """
        # Compute charge density: ρ = ∫f dv - 1
        rho = torch.trapz(f.squeeze(1), dx=dv, dim=-1) - 1.0  # [batch, Nx]
        
        # Solve Poisson equation in Fourier space: E = -i*ρ_k / k
        rho_k = torch.fft.fft(rho, dim=-1)
        
        Nx = rho.shape[-1]
        k = torch.fft.fftfreq(Nx, d=dx).to(rho.device)
        k[0] = 1.0  # Avoid division by zero
        
        E_k = -1j * rho_k / k
        E_k[..., 0] = 0.0  # Zero mean field
        
        E = torch.real(torch.fft.ifft(E_k, dim=-1))
        
        return E.unsqueeze(1)  # [batch, 1, Nx]
    
    def compute_vlasov_residual(self, f, E, x_grid, v_grid, dx, dv):
        """
        Compute Vlasov equation residual: ∂f/∂t + v*∂f/∂x + E*∂f/∂v
        
        Args:
            f: Distribution function [batch, 1, Nx, Nv]
            E: Electric field [batch, 1, Nx]
            x_grid, v_grid: Coordinate grids
            dx, dv: Grid spacings
        
        Returns:
            Residual [batch, 1, Nx, Nv]
        """
        # Compute spatial derivative ∂f/∂x using spectral method
        f_k = torch.fft.fft2(f.squeeze(1))
        Nx, Nv = f.shape[2], f.shape[3]
        
        kx = torch.fft.fftfreq(Nx, d=dx).to(f.device)
        kv = torch.fft.fftfreq(Nv, d=dv).to(f.device)
        
        KX, KV = torch.meshgrid(kx, kv, indexing='ij')
        
        # ∂f/∂x in Fourier space
        df_dx_k = 1j * 2 * torch.pi * KX * f_k
        df_dx = torch.real(torch.fft.ifft2(df_dx_k))
        
        # ∂f/∂v in Fourier space
        df_dv_k = 1j * 2 * torch.pi * KV * f_k
        df_dv = torch.real(torch.fft.ifft2(df_dv_k))
        
        # Expand E to phase space [batch, Nx, Nv]
        E_expanded = E.squeeze(1).unsqueeze(-1).expand_as(f.squeeze(1))
        
        # Expand v grid [batch, Nx, Nv]
        v_expanded = v_grid.squeeze(1)
        
        # Vlasov equation advection terms: v*∂f/∂x + E*∂f/∂v
        advection = v_expanded * df_dx + E_expanded * df_dv
        
        return advection.unsqueeze(1)
    
    def compute_conservation_loss(self, f, f0, dv):
        """
        Compute conservation losses (mass and energy).
        
        Args:
            f: Current distribution [batch, 1, Nx, Nv]
            f0: Initial distribution [batch, 1, Nx, Nv]
            dv: Velocity grid spacing
        
        Returns:
            mass_loss, energy_loss
        """
        # Mass conservation: ∫f dv should be constant
        mass = torch.trapz(f.squeeze(1), dx=dv, dim=-1)  # [batch, Nx]
        mass0 = torch.trapz(f0.squeeze(1), dx=dv, dim=-1)
        mass_loss = F.mse_loss(mass, mass0)
        
        # Energy conservation (kinetic): ∫(v^2/2)*f dv
        v = torch.linspace(-6, 6, f.shape[-1]).to(f.device)
        v2 = (v ** 2) / 2.0
        
        energy = torch.trapz(f.squeeze(1) * v2, dx=dv, dim=-1)
        energy0 = torch.trapz(f0.squeeze(1) * v2, dx=dv, dim=-1)
        energy_loss = F.mse_loss(energy, energy0)
        
        return mass_loss, energy_loss


def create_vp_fno_model(config: FNOConfig, model_type="standard"):
    """
    Factory function to create VP-FNO models.
    
    Args:
        config: FNO configuration
        model_type: "standard", "unet", or "physics_informed"
    
    Returns:
        VP-FNO model instance
    """
    models = {
        "standard": VPFNO2d,
        "unet": UFNO_VP,
        "physics_informed": PhysicsInformedFNO_VP
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](config)


if __name__ == "__main__":
    from config import get_default_config
    
    print("Testing VP-FNO models...")
    
    # Get configuration
    config = get_default_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 4
    Nx, Nv = 64, 64
    
    # Create input (f0, x_grid, v_grid)
    f0 = torch.randn(batch_size, 1, Nx, Nv).to(device)
    x_grid = torch.linspace(0, 1, Nx).view(1, 1, Nx, 1).expand(batch_size, 1, Nx, Nv).to(device)
    v_grid = torch.linspace(-1, 1, Nv).view(1, 1, 1, Nv).expand(batch_size, 1, Nx, Nv).to(device)
    
    input_data = torch.cat([f0, x_grid, v_grid], dim=1)
    
    # Test standard FNO
    print("\nStandard VP-FNO:")
    model_std = create_vp_fno_model(config.fno, "standard").to(device)
    out_std = model_std(input_data)
    print(f"Input: {input_data.shape} -> Output: {out_std.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_std.parameters()):,}")
    
    # Test U-Net FNO
    print("\nU-Net VP-FNO:")
    model_unet = create_vp_fno_model(config.fno, "unet").to(device)
    out_unet = model_unet(input_data)
    print(f"Input: {input_data.shape} -> Output: {out_unet.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_unet.parameters()):,}")
    
    # Test Physics-Informed FNO
    print("\nPhysics-Informed VP-FNO:")
    model_pi = create_vp_fno_model(config.fno, "physics_informed").to(device)
    out_pi = model_pi(input_data)
    print(f"Input: {input_data.shape} -> Output: {out_pi.shape}")
    
    # Test physics computations
    E = model_pi.compute_electric_field(out_pi, dx=0.1, dv=0.1)
    print(f"Electric field shape: {E.shape}")
    
    mass_loss, energy_loss = model_pi.compute_conservation_loss(out_pi, f0, dv=0.1)
    print(f"Mass loss: {mass_loss.item():.6f}, Energy loss: {energy_loss.item():.6f}")
    
    print("\nAll VP-FNO tests passed!")
