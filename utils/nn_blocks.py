"""
Neural network building blocks for AI4CFD methods.
Reusable layers and modules for PINNs, DeepONet, FNO, TNN, and Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ==============================================================================
# Basic Building Blocks
# ==============================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable activation and normalization.
    
    Example:
        >>> net = MLP([2, 64, 64, 1], activation='tanh')
        >>> x = torch.randn(100, 2)
        >>> y = net(x)  # shape: (100, 1)
    """
    def __init__(self, layers, activation='tanh', output_activation=None, 
                 use_batch_norm=False, dropout=0.0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        activation_map = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'sin': SinActivation(),
            'swish': Swish(),
        }
        
        self.activation = activation_map.get(activation.lower(), nn.Tanh())
        self.output_activation = activation_map.get(output_activation, None) if output_activation else None
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if use_batch_norm and i < len(layers) - 2:
                self.norms.append(nn.BatchNorm1d(layers[i+1]))
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        x = self.layers[-1](x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class SinActivation(nn.Module):
    """Sinusoidal activation function - excellent for periodic problems."""
    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for improved high-frequency learning.
    Reference: "Fourier Features Let Networks Learn High Frequency Functions" (Tancik et al.)
    
    Example:
        >>> ff = FourierFeatures(2, 128, scale=10.0)
        >>> x = torch.randn(100, 2)
        >>> features = ff(x)  # shape: (100, 256)
    """
    def __init__(self, in_features, out_features, scale=1.0, learnable=False):
        super(FourierFeatures, self).__init__()
        B = torch.randn(in_features, out_features // 2) * scale
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ModifiedMLP(nn.Module):
    """
    Modified MLP with Fourier features for better high-frequency learning.
    Combines Fourier feature encoding with standard MLP.
    """
    def __init__(self, in_dim, hidden_dims, out_dim, fourier_features=64, 
                 scale=10.0, activation='tanh'):
        super(ModifiedMLP, self).__init__()
        self.fourier = FourierFeatures(in_dim, fourier_features * 2, scale)
        layers = [fourier_features * 2] + hidden_dims + [out_dim]
        self.mlp = MLP(layers, activation=activation)
    
    def forward(self, x):
        x = self.fourier(x)
        return self.mlp(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, hidden_dim, activation='gelu'):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU() if activation == 'gelu' else nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.activation = nn.GELU() if activation == 'gelu' else nn.Tanh()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class ResMLP(nn.Module):
    """MLP with residual connections."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks=4, activation='gelu'):
        super(ResMLP, self).__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation) for _ in range(num_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


# ==============================================================================
# PINNs Components
# ==============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network base class.
    
    Example:
        >>> pinn = PINN(input_dim=2, output_dim=1, hidden_layers=[64, 64, 64])
        >>> x = torch.randn(100, 2, requires_grad=True)
        >>> u = pinn(x)
        >>> u_x, u_t = pinn.gradient(u, x)
    """
    def __init__(self, input_dim, output_dim, hidden_layers=[64, 64, 64], 
                 activation='tanh', use_fourier=False, fourier_scale=1.0):
        super(PINN, self).__init__()
        
        if use_fourier:
            self.net = ModifiedMLP(input_dim, hidden_layers, output_dim, 
                                   scale=fourier_scale, activation=activation)
        else:
            layers = [input_dim] + hidden_layers + [output_dim]
            self.net = MLP(layers, activation=activation)
    
    def forward(self, x):
        return self.net(x)
    
    def gradient(self, u, x, order=1):
        """Compute gradients using automatic differentiation."""
        grads = []
        for i in range(x.shape[1]):
            grad_i = u
            for _ in range(order):
                grad_i = torch.autograd.grad(
                    grad_i.sum(), x, create_graph=True, retain_graph=True
                )[0][:, i:i+1]
            grads.append(grad_i)
        return grads if len(grads) > 1 else grads[0]
    
    def laplacian(self, u, x):
        """Compute Laplacian (sum of second derivatives)."""
        lap = 0
        for i in range(x.shape[1]):
            u_i = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, i:i+1]
            u_ii = torch.autograd.grad(u_i.sum(), x, create_graph=True)[0][:, i:i+1]
            lap += u_ii
        return lap


class AdaptiveWeightPINN(PINN):
    """
    PINN with adaptive loss weighting.
    Automatically balances PDE, BC, and IC losses during training.
    """
    def __init__(self, *args, num_losses=3, **kwargs):
        super(AdaptiveWeightPINN, self).__init__(*args, **kwargs)
        self.log_weights = nn.Parameter(torch.zeros(num_losses))
    
    def get_weights(self):
        """Get normalized weights for each loss term."""
        return F.softmax(self.log_weights, dim=0) * len(self.log_weights)


# ==============================================================================
# DeepONet Components
# ==============================================================================

class DeepONet(nn.Module):
    """
    Deep Operator Network for learning operators between function spaces.
    
    Example:
        >>> don = DeepONet(branch_input=100, trunk_input=1, hidden_dim=64, output_dim=1)
        >>> u_sensors = torch.randn(32, 100)  # 32 samples, 100 sensor points
        >>> y_query = torch.randn(50, 1)       # 50 query locations
        >>> output = don(u_sensors, y_query)   # shape: (32, 50, 1)
    """
    def __init__(self, branch_input, trunk_input, hidden_dim=64, 
                 output_dim=1, p=40, branch_layers=3, trunk_layers=3):
        super(DeepONet, self).__init__()
        
        # Branch network: encodes input function
        branch_dims = [branch_input] + [hidden_dim] * branch_layers + [p * output_dim]
        self.branch = MLP(branch_dims, activation='tanh')
        
        # Trunk network: encodes query location
        trunk_dims = [trunk_input] + [hidden_dim] * trunk_layers + [p * output_dim]
        self.trunk = MLP(trunk_dims, activation='tanh')
        
        self.p = p
        self.output_dim = output_dim
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, u_sensors, y_query):
        """
        Args:
            u_sensors: Input function values at sensor points [batch, n_sensors]
            y_query: Query locations [n_query, trunk_input] or [batch, n_query, trunk_input]
        Returns:
            Output at query locations [batch, n_query, output_dim]
        """
        batch_size = u_sensors.shape[0]
        
        # Branch output: [batch, p * output_dim]
        b = self.branch(u_sensors)
        
        # Handle different query formats
        if y_query.dim() == 2:
            # Same query for all samples: [n_query, trunk_input]
            t = self.trunk(y_query)  # [n_query, p * output_dim]
            # Reshape for proper broadcasting
            b = b.view(batch_size, self.output_dim, self.p)  # [batch, output_dim, p]
            t = t.view(-1, self.output_dim, self.p)  # [n_query, output_dim, p]
            # Inner product
            output = torch.einsum('bop,qop->bqo', b, t) + self.bias
        else:
            # Different queries per sample: [batch, n_query, trunk_input]
            n_query = y_query.shape[1]
            t = self.trunk(y_query.view(-1, y_query.shape[-1]))  # [batch*n_query, p*output_dim]
            t = t.view(batch_size, n_query, self.output_dim, self.p)
            b = b.view(batch_size, 1, self.output_dim, self.p)
            output = (b * t).sum(-1) + self.bias  # [batch, n_query, output_dim]
        
        return output


class StackedDeepONet(nn.Module):
    """
    Stacked DeepONet for improved accuracy on complex operators.
    Uses multiple DeepONet layers with residual connections.
    """
    def __init__(self, branch_input, trunk_input, hidden_dim=64, 
                 output_dim=1, p=40, num_stacks=3):
        super(StackedDeepONet, self).__init__()
        
        self.layers = nn.ModuleList([
            DeepONet(branch_input, trunk_input, hidden_dim, output_dim, p)
            for _ in range(num_stacks)
        ])
        self.combine = nn.Linear(num_stacks * output_dim, output_dim)
    
    def forward(self, u_sensors, y_query):
        outputs = [layer(u_sensors, y_query) for layer in self.layers]
        combined = torch.cat(outputs, dim=-1)
        return self.combine(combined)


# ==============================================================================
# FNO Components
# ==============================================================================

class SpectralConv1d(nn.Module):
    """1D Fourier layer for FNO."""
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, 2))
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)
        
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :self.modes], 
            torch.view_as_complex(self.weights)
        )
        
        return torch.fft.irfft(out_ft, n=x.size(-1))


class SpectralConv2d(nn.Module):
    """2D Fourier layer for FNO."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            torch.view_as_complex(self.weights1)
        )
        
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            torch.view_as_complex(self.weights2)
        )
        
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator.
    
    Example:
        >>> fno = FNO1d(in_channels=1, out_channels=1, modes=16, width=64)
        >>> x = torch.randn(32, 128, 1)  # [batch, spatial, channels]
        >>> y = fno(x)  # shape: (32, 128, 1)
    """
    def __init__(self, in_channels=1, out_channels=1, modes=16, width=64, num_layers=4):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        
        self.fc0 = nn.Linear(in_channels + 1, width)  # +1 for grid
        
        self.convs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(num_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x):
        # x: [batch, spatial, in_channels]
        batch_size, spatial_size, _ = x.shape
        
        # Add spatial grid
        grid = torch.linspace(0, 1, spatial_size, device=x.device).view(1, -1, 1).expand(batch_size, -1, -1)
        x = torch.cat([x, grid], dim=-1)
        
        x = self.fc0(x)  # [batch, spatial, width]
        x = x.permute(0, 2, 1)  # [batch, width, spatial]
        
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)
        
        x = x.permute(0, 2, 1)  # [batch, spatial, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator.
    
    Example:
        >>> fno = FNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12, width=32)
        >>> x = torch.randn(8, 64, 64, 1)  # [batch, x, y, channels]
        >>> y = fno(x)  # shape: (8, 64, 64, 1)
    """
    def __init__(self, in_channels=1, out_channels=1, modes1=12, modes2=12, 
                 width=32, num_layers=4):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for grid
        
        self.convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x):
        # x: [batch, x, y, in_channels]
        batch_size = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]
        
        # Add spatial grid
        grid_x = torch.linspace(0, 1, size_x, device=x.device)
        grid_y = torch.linspace(0, 1, size_y, device=x.device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = torch.cat([x, grid], dim=-1)
        
        x = self.fc0(x)  # [batch, x, y, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, x, y]
        
        for conv, w in zip(self.convs, self.ws):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)
        
        x = x.permute(0, 2, 3, 1)  # [batch, x, y, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x


# ==============================================================================
# TNN Components (Tensor Neural Networks)
# ==============================================================================

class TensorLayer(nn.Module):
    """
    Tensor decomposition layer for high-dimensional PDEs.
    Uses CP decomposition to avoid curse of dimensionality.
    """
    def __init__(self, dim, hidden_size=20, rank=10):
        super(TensorLayer, self).__init__()
        self.dim = dim
        self.rank = rank
        
        # Create factor networks for each dimension
        self.factors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, rank)
            ) for _ in range(dim)
        ])
        
        self.weights = nn.Parameter(torch.ones(rank) / rank)
    
    def forward(self, x):
        # x: [batch, dim]
        batch_size = x.shape[0]
        
        # Compute tensor product
        result = torch.ones(batch_size, self.rank, device=x.device)
        for i in range(self.dim):
            factor_i = self.factors[i](x[:, i:i+1])  # [batch, rank]
            result = result * factor_i
        
        # Weighted sum
        output = (result * self.weights).sum(dim=1, keepdim=True)
        return output


class TNN(nn.Module):
    """
    Tensor Neural Network for high-dimensional PDEs.
    
    Example:
        >>> tnn = TNN(dim=5, hidden_size=32, rank=20, num_layers=3)
        >>> x = torch.randn(100, 5)  # 100 points in 5D
        >>> u = tnn(x)  # shape: (100, 1)
    """
    def __init__(self, dim, hidden_size=32, rank=20, num_layers=3, output_dim=1):
        super(TNN, self).__init__()
        self.dim = dim
        
        self.layers = nn.ModuleList([
            TensorLayer(dim, hidden_size, rank) for _ in range(num_layers)
        ])
        
        self.combine = nn.Linear(num_layers, output_dim)
    
    def forward(self, x):
        outputs = [layer(x) for layer in self.layers]
        combined = torch.cat(outputs, dim=1)
        return self.combine(combined)


class TuckerTNN(nn.Module):
    """
    TNN using Tucker decomposition for better expressiveness.
    """
    def __init__(self, dim, hidden_size=32, ranks=[10]*5, core_size=10):
        super(TuckerTNN, self).__init__()
        self.dim = dim
        
        # Factor matrices for each dimension
        self.factors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, ranks[i] if i < len(ranks) else ranks[-1])
            ) for i in range(dim)
        ])
        
        # Core tensor (learnable)
        core_shape = tuple(ranks[:dim]) if len(ranks) >= dim else tuple(ranks + [ranks[-1]] * (dim - len(ranks)))
        self.core = nn.Parameter(torch.randn(*core_shape) * 0.01)
        
        self.output = nn.Linear(1, 1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute factor matrices
        factors = [self.factors[i](x[:, i:i+1]) for i in range(self.dim)]
        
        # Contract with core tensor
        result = self.core
        for i in range(self.dim):
            result = torch.tensordot(factors[i], result, dims=[[1], [0]])
            result = result.permute(list(range(1, result.dim())) + [0])
        
        return result.view(batch_size, 1)


# ==============================================================================
# Transformer Components for PDEs
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]


class PDETransformer(nn.Module):
    """
    Transformer for PDE time-series prediction.
    
    Example:
        >>> transformer = PDETransformer(spatial_dim=64, d_model=128, nhead=8)
        >>> u_t = torch.randn(32, 64, 1)  # [batch, spatial, features]
        >>> u_next = transformer(u_t)  # predict next timestep
    """
    def __init__(self, spatial_dim=64, d_model=128, nhead=8, 
                 num_layers=4, dim_feedforward=512, in_channels=1, out_channels=1):
        super(PDETransformer, self).__init__()
        
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=spatial_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, out_channels)
    
    def forward(self, x):
        # x: [batch, spatial, in_channels]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return x


class SpatioTemporalTransformer(nn.Module):
    """
    Transformer for spatio-temporal PDE prediction.
    Handles both spatial and temporal dimensions.
    """
    def __init__(self, spatial_size, temporal_size, d_model=128, 
                 nhead=8, num_layers=4, in_channels=1, out_channels=1):
        super(SpatioTemporalTransformer, self).__init__()
        
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Separate positional encodings for space and time
        self.spatial_pe = nn.Parameter(torch.randn(1, spatial_size, d_model) * 0.02)
        self.temporal_pe = nn.Parameter(torch.randn(1, temporal_size, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, out_channels)
    
    def forward(self, x):
        # x: [batch, time, space, channels]
        batch_size = x.shape[0]
        
        x = self.input_proj(x)  # [batch, time, space, d_model]
        
        # Add positional encodings
        x = x + self.spatial_pe.unsqueeze(1) + self.temporal_pe.unsqueeze(2)
        
        # Flatten space-time
        x = x.view(batch_size, -1, x.shape[-1])  # [batch, time*space, d_model]
        
        x = self.transformer(x)
        
        # Reshape back
        x = x.view(batch_size, self.temporal_size, self.spatial_size, -1)
        x = self.output_proj(x)
        
        return x


# ==============================================================================
# Utility Functions
# ==============================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_gpu=True):
    """Get the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    elif prefer_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def save_model(model, path, optimizer=None, epoch=None, loss=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model, path, optimizer=None, device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', None)
    loss = checkpoint.get('loss', None)
    
    print(f"Model loaded from {path}")
    return epoch, loss
