"""
Neural network architectures for Physics-Informed Neural Networks (PINNs).
"""

import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.
    
    Args:
        input_dim: Input dimension (spatial + temporal coordinates)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (solution components)
        num_layers: Number of hidden layers
        activation: Activation function
    """
    
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, 
                 num_layers=4, activation='tanh'):
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Define activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
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
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Xavier initialization for better training stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)


class ModifiedMLP(nn.Module):
    """
    Modified MLP with Fourier features for better representation of high-frequency components.
    """
    
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1, 
                 num_layers=8, fourier_features=False, sigma=1.0):
        super(ModifiedMLP, self).__init__()
        
        self.fourier_features = fourier_features
        self.sigma = sigma
        
        if fourier_features:
            # Random Fourier features
            self.fourier_dim = hidden_dim // 2
            self.B = nn.Parameter(torch.randn(input_dim, self.fourier_dim) * sigma)
            self.B.requires_grad = False
            input_dim = 2 * self.fourier_dim
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def fourier_embedding(self, x):
        """Apply random Fourier features."""
        x_proj = torch.matmul(x, self.B) * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def forward(self, x):
        if self.fourier_features:
            x = self.fourier_embedding(x)
        return self.network(x)


class ResNetPINN(nn.Module):
    """
    Residual network architecture for PINNs to handle very deep networks.
    """
    
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1, 
                 num_blocks=4, layers_per_block=2):
        super(ResNetPINN, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ResidualBlock(hidden_dim, layers_per_block)
            self.blocks.append(block)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)


class ResidualBlock(nn.Module):
    """Residual block for ResNetPINN."""
    
    def __init__(self, hidden_dim, num_layers=2):
        super(ResidualBlock, self).__init__()
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.block(x)


class MultiScalePINN(nn.Module):
    """
    Multi-scale PINN that combines multiple networks for different frequency components.
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1, 
                 num_scales=3, base_frequency=1.0):
        super(MultiScalePINN, self).__init__()
        
        self.num_scales = num_scales
        self.networks = nn.ModuleList()
        
        # Create networks for different scales
        for i in range(num_scales):
            freq = base_frequency * (2 ** i)
            net = ModifiedMLP(input_dim, hidden_dim, output_dim, 
                            fourier_features=True, sigma=freq)
            self.networks.append(net)
        
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
    
    def forward(self, x):
        outputs = []
        for net in self.networks:
            outputs.append(net(x))
        
        # Weighted combination
        combined = torch.zeros_like(outputs[0])
        weights = torch.softmax(self.scale_weights, dim=0)
        
        for i, output in enumerate(outputs):
            combined += weights[i] * output
        
        return combined


class AdaptivePINN(nn.Module):
    """
    Adaptive PINN with learnable activation functions and architecture.
    """
    
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=1, 
                 num_layers=6, adaptive_activation=True):
        super(AdaptivePINN, self).__init__()
        
        self.adaptive_activation = adaptive_activation
        
        # Learnable activation parameters
        if adaptive_activation:
            self.activation_params = nn.ParameterList([
                nn.Parameter(torch.ones(1)) for _ in range(num_layers)
            ])
        
        # Network layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def adaptive_tanh(self, x, param):
        """Adaptive tanh activation with learnable parameter."""
        return torch.tanh(param * x)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.adaptive_activation and i < len(self.activation_params):
                x = self.adaptive_tanh(x, self.activation_params[i])
            else:
                x = torch.tanh(x)
        
        # Output layer without activation
        x = self.layers[-1](x)
        return x


class EnsemblePINN(nn.Module):
    """
    Ensemble of PINNs for uncertainty quantification.
    """
    
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1, 
                 num_networks=5, diversity_reg=0.01):
        super(EnsemblePINN, self).__init__()
        
        self.num_networks = num_networks
        self.diversity_reg = diversity_reg
        
        # Create ensemble of networks
        self.networks = nn.ModuleList()
        for _ in range(num_networks):
            net = PINN(input_dim, hidden_dim, output_dim)
            self.networks.append(net)
    
    def forward(self, x, return_all=False):
        outputs = []
        for net in self.networks:
            outputs.append(net(x))
        
        if return_all:
            return torch.stack(outputs, dim=0)
        else:
            # Return mean prediction
            return torch.mean(torch.stack(outputs, dim=0), dim=0)
    
    def compute_uncertainty(self, x):
        """Compute prediction uncertainty as ensemble variance."""
        outputs = self.forward(x, return_all=True)
        mean = torch.mean(outputs, dim=0)
        variance = torch.var(outputs, dim=0)
        return mean, variance
    
    def diversity_loss(self, x):
        """Compute diversity regularization term."""
        outputs = self.forward(x, return_all=True)
        mean = torch.mean(outputs, dim=0, keepdim=True)
        diversity = torch.mean(torch.var(outputs - mean, dim=0))
        return -self.diversity_reg * diversity  # Negative to encourage diversity


def create_pinn_model(model_type='standard', **kwargs):
    """
    Factory function to create different types of PINN models.
    
    Args:
        model_type: Type of model ('standard', 'fourier', 'resnet', 'multiscale', 'adaptive', 'ensemble')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        PINN model instance
    """
    if model_type == 'standard':
        return PINN(**kwargs)
    elif model_type == 'fourier':
        return ModifiedMLP(fourier_features=True, **kwargs)
    elif model_type == 'resnet':
        return ResNetPINN(**kwargs)
    elif model_type == 'multiscale':
        return MultiScalePINN(**kwargs)
    elif model_type == 'adaptive':
        return AdaptivePINN(**kwargs)
    elif model_type == 'ensemble':
        return EnsemblePINN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test different model architectures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test input
    x = torch.randn(100, 2).to(device)
    
    models = {
        'Standard PINN': create_pinn_model('standard', input_dim=2, hidden_dim=50),
        'Fourier PINN': create_pinn_model('fourier', input_dim=2, hidden_dim=256),
        'ResNet PINN': create_pinn_model('resnet', input_dim=2, hidden_dim=128),
        'MultiScale PINN': create_pinn_model('multiscale', input_dim=2, hidden_dim=128),
        'Adaptive PINN': create_pinn_model('adaptive', input_dim=2, hidden_dim=128),
        'Ensemble PINN': create_pinn_model('ensemble', input_dim=2, hidden_dim=64)
    }
    
    for name, model in models.items():
        model = model.to(device)
        with torch.no_grad():
            output = model(x)
            print(f"{name}: Input {x.shape} -> Output {output.shape}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print()