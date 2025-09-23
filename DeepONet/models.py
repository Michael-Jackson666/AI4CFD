"""
Deep Operator Network (DeepONet) architectures for learning operators between function spaces.
"""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='tanh', 
                 dropout=0.0, batch_norm=False):
        super(MLP, self).__init__()
        
        # Choose activation function
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
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Add batch normalization if specified (not for output layer)
            if batch_norm and i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Add activation (not for output layer)
            if i < len(dims) - 2:
                layers.append(self.activation)
                
                # Add dropout if specified
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class DeepONet(nn.Module):
    """
    Standard Deep Operator Network.
    
    Args:
        branch_input_dim: Dimension of branch network input (number of sensors)
        trunk_input_dim: Dimension of trunk network input (coordinate dimension)
        branch_hidden_dims: List of hidden layer dimensions for branch network
        trunk_hidden_dims: List of hidden layer dimensions for trunk network
        output_dim: Output dimension (usually 1 for scalar fields)
        activation: Activation function
        use_bias: Whether to use bias in the final combination
    """
    
    def __init__(self, branch_input_dim, trunk_input_dim, 
                 branch_hidden_dims=[100, 100, 100], 
                 trunk_hidden_dims=[100, 100, 100],
                 output_dim=1, activation='tanh', use_bias=True):
        super(DeepONet, self).__init__()
        
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.latent_dim = branch_hidden_dims[-1]  # Should match trunk output dim
        
        # Branch network: processes input function values at sensor locations
        self.branch_net = MLP(
            input_dim=branch_input_dim,
            hidden_dims=branch_hidden_dims[:-1],
            output_dim=self.latent_dim * output_dim,
            activation=activation
        )
        
        # Trunk network: processes query coordinates
        self.trunk_net = MLP(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden_dims[:-1],
            output_dim=self.latent_dim * output_dim,
            activation=activation
        )
        
        # Bias term (optional)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, branch_input, trunk_input):
        """
        Forward pass of DeepONet.
        
        Args:
            branch_input: Input function values at sensors [batch_size, n_sensors]
            trunk_input: Query coordinates [batch_size, n_queries, coord_dim]
        
        Returns:
            Output function values at query points [batch_size, n_queries, output_dim]
        """
        batch_size = branch_input.shape[0]
        n_queries = trunk_input.shape[1]
        
        # Branch network output: [batch_size, latent_dim * output_dim]
        branch_output = self.branch_net(branch_input)
        branch_output = branch_output.view(batch_size, self.output_dim, self.latent_dim)
        
        # Trunk network output: [batch_size, n_queries, latent_dim * output_dim]
        trunk_input_flat = trunk_input.view(-1, trunk_input.shape[-1])
        trunk_output = self.trunk_net(trunk_input_flat)
        trunk_output = trunk_output.view(batch_size, n_queries, self.output_dim, self.latent_dim)
        
        # Compute inner product: [batch_size, n_queries, output_dim]
        output = torch.einsum('bki,bqki->bqk', branch_output, trunk_output)
        
        # Add bias if specified
        if self.use_bias:
            output = output + self.bias.unsqueeze(0).unsqueeze(0)
        
        return output


class ModifiedDeepONet(nn.Module):
    """
    Modified DeepONet with additional features.
    
    - Separate output dimensions for branch and trunk
    - Residual connections
    - Attention mechanisms
    """
    
    def __init__(self, branch_input_dim, trunk_input_dim,
                 branch_hidden_dims=[100, 100], 
                 trunk_hidden_dims=[100, 100],
                 latent_dim=100, output_dim=1, 
                 activation='tanh', use_residual=False,
                 use_attention=False):
        super(ModifiedDeepONet, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Branch network
        self.branch_net = MLP(
            input_dim=branch_input_dim,
            hidden_dims=branch_hidden_dims,
            output_dim=latent_dim * output_dim,
            activation=activation
        )
        
        # Trunk network
        self.trunk_net = MLP(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden_dims,
            output_dim=latent_dim * output_dim,
            activation=activation
        )
        
        # Residual connection (optional)
        if use_residual:
            self.residual_net = MLP(
                input_dim=trunk_input_dim,
                hidden_dims=[50, 50],
                output_dim=output_dim,
                activation=activation
            )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=4,
                batch_first=True
            )
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, branch_input, trunk_input):
        batch_size = branch_input.shape[0]
        n_queries = trunk_input.shape[1]
        
        # Branch network
        branch_output = self.branch_net(branch_input)
        branch_output = branch_output.view(batch_size, self.output_dim, self.latent_dim)
        
        # Trunk network
        trunk_input_flat = trunk_input.view(-1, trunk_input.shape[-1])
        trunk_output = self.trunk_net(trunk_input_flat)
        trunk_output = trunk_output.view(batch_size, n_queries, self.output_dim, self.latent_dim)
        
        # Apply attention if specified
        if self.use_attention:
            # Reshape for attention: [batch_size, n_queries, latent_dim]
            trunk_attn = trunk_output.view(batch_size, n_queries, -1)
            branch_attn = branch_output.view(batch_size, 1, -1).expand(-1, n_queries, -1)
            
            attended_output, _ = self.attention(trunk_attn, branch_attn, branch_attn)
            attended_output = attended_output.view(batch_size, n_queries, self.output_dim, self.latent_dim)
            
            # Use attended output for combination
            output = torch.einsum('bki,bqki->bqk', branch_output, attended_output)
        else:
            # Standard inner product
            output = torch.einsum('bki,bqki->bqk', branch_output, trunk_output)
        
        # Add residual connection if specified
        if self.use_residual:
            residual = self.residual_net(trunk_input_flat)
            residual = residual.view(batch_size, n_queries, self.output_dim)
            output = output + residual
        
        # Add bias
        output = output + self.bias.unsqueeze(0).unsqueeze(0)
        
        return output


class FourierDeepONet(nn.Module):
    """
    DeepONet with Fourier feature encoding for periodic problems.
    """
    
    def __init__(self, branch_input_dim, trunk_input_dim,
                 branch_hidden_dims=[100, 100, 100],
                 trunk_hidden_dims=[100, 100, 100],
                 output_dim=1, fourier_modes=20, activation='tanh'):
        super(FourierDeepONet, self).__init__()
        
        self.fourier_modes = fourier_modes
        self.output_dim = output_dim
        self.latent_dim = branch_hidden_dims[-1]
        
        # Fourier feature encoding for trunk network
        self.register_buffer('fourier_basis', 
                            torch.randn(trunk_input_dim, fourier_modes) * 2 * np.pi)
        
        # Branch network (unchanged)
        self.branch_net = MLP(
            input_dim=branch_input_dim,
            hidden_dims=branch_hidden_dims[:-1],
            output_dim=self.latent_dim * output_dim,
            activation=activation
        )
        
        # Trunk network with Fourier features
        trunk_input_expanded = 2 * fourier_modes * trunk_input_dim
        self.trunk_net = MLP(
            input_dim=trunk_input_expanded,
            hidden_dims=trunk_hidden_dims[:-1],
            output_dim=self.latent_dim * output_dim,
            activation=activation
        )
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def fourier_encode(self, coords):
        """Apply Fourier feature encoding to coordinates."""
        # coords: [batch_size * n_queries, coord_dim]
        proj = torch.matmul(coords, self.fourier_basis)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
    
    def forward(self, branch_input, trunk_input):
        batch_size = branch_input.shape[0]
        n_queries = trunk_input.shape[1]
        
        # Branch network
        branch_output = self.branch_net(branch_input)
        branch_output = branch_output.view(batch_size, self.output_dim, self.latent_dim)
        
        # Trunk network with Fourier encoding
        trunk_input_flat = trunk_input.view(-1, trunk_input.shape[-1])
        trunk_encoded = self.fourier_encode(trunk_input_flat)
        trunk_output = self.trunk_net(trunk_encoded)
        trunk_output = trunk_output.view(batch_size, n_queries, self.output_dim, self.latent_dim)
        
        # Inner product
        output = torch.einsum('bki,bqki->bqk', branch_output, trunk_output)
        output = output + self.bias.unsqueeze(0).unsqueeze(0)
        
        return output


class MultiScaleDeepONet(nn.Module):
    """
    Multi-scale DeepONet for problems with multiple length scales.
    """
    
    def __init__(self, branch_input_dim, trunk_input_dim,
                 scales=[1, 2, 4], hidden_dim=100, output_dim=1):
        super(MultiScaleDeepONet, self).__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        self.output_dim = output_dim
        self.latent_dim = hidden_dim
        
        # Branch networks for different scales
        self.branch_nets = nn.ModuleList([
            MLP(branch_input_dim, [hidden_dim, hidden_dim], 
                self.latent_dim * output_dim)
            for _ in scales
        ])
        
        # Trunk networks for different scales
        self.trunk_nets = nn.ModuleList([
            MLP(trunk_input_dim, [hidden_dim, hidden_dim], 
                self.latent_dim * output_dim)
            for _ in scales
        ])
        
        # Scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, branch_input, trunk_input):
        batch_size = branch_input.shape[0]
        n_queries = trunk_input.shape[1]
        
        outputs = []
        
        for i, scale in enumerate(self.scales):
            # Scale the inputs
            scaled_branch = branch_input * scale
            scaled_trunk = trunk_input * scale
            
            # Branch network
            branch_output = self.branch_nets[i](scaled_branch)
            branch_output = branch_output.view(batch_size, self.output_dim, self.latent_dim)
            
            # Trunk network
            trunk_input_flat = scaled_trunk.view(-1, scaled_trunk.shape[-1])
            trunk_output = self.trunk_nets[i](trunk_input_flat)
            trunk_output = trunk_output.view(batch_size, n_queries, self.output_dim, self.latent_dim)
            
            # Inner product
            scale_output = torch.einsum('bki,bqki->bqk', branch_output, trunk_output)
            outputs.append(scale_output)
        
        # Combine scales with learnable weights
        combined_output = torch.zeros_like(outputs[0])
        weights = torch.softmax(self.scale_weights, dim=0)
        
        for i, output in enumerate(outputs):
            combined_output += weights[i] * output
        
        combined_output = combined_output + self.bias.unsqueeze(0).unsqueeze(0)
        
        return combined_output


class PINNDeepONet(nn.Module):
    """
    Physics-Informed DeepONet that incorporates PDE constraints.
    """
    
    def __init__(self, branch_input_dim, trunk_input_dim,
                 branch_hidden_dims=[100, 100, 100],
                 trunk_hidden_dims=[100, 100, 100],
                 output_dim=1, activation='tanh'):
        super(PINNDeepONet, self).__init__()
        
        self.deeponet = DeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            branch_hidden_dims=branch_hidden_dims,
            trunk_hidden_dims=trunk_hidden_dims,
            output_dim=output_dim,
            activation=activation
        )
    
    def forward(self, branch_input, trunk_input):
        return self.deeponet(branch_input, trunk_input)
    
    def compute_derivatives(self, branch_input, trunk_input, order=1):
        """
        Compute derivatives of the DeepONet output for physics constraints.
        """
        trunk_input.requires_grad_(True)
        output = self.forward(branch_input, trunk_input)
        
        # Compute gradients
        gradients = []
        for i in range(trunk_input.shape[-1]):
            grad = torch.autograd.grad(
                output.sum(), trunk_input,
                create_graph=True, retain_graph=True
            )[0][..., i:i+1]
            gradients.append(grad)
        
        if order == 1:
            return gradients
        elif order == 2:
            # Compute second derivatives
            second_gradients = []
            for grad in gradients:
                second_grad = []
                for j in range(trunk_input.shape[-1]):
                    grad2 = torch.autograd.grad(
                        grad.sum(), trunk_input,
                        create_graph=True, retain_graph=True
                    )[0][..., j:j+1]
                    second_grad.append(grad2)
                second_gradients.append(second_grad)
            return gradients, second_gradients
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")


def create_deeponet(model_type='standard', **kwargs):
    """
    Factory function to create different types of DeepONet models.
    
    Args:
        model_type: Type of DeepONet ('standard', 'modified', 'fourier', 'multiscale', 'pinn')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        DeepONet model instance
    """
    if model_type == 'standard':
        return DeepONet(**kwargs)
    elif model_type == 'modified':
        return ModifiedDeepONet(**kwargs)
    elif model_type == 'fourier':
        return FourierDeepONet(**kwargs)
    elif model_type == 'multiscale':
        return MultiScaleDeepONet(**kwargs)
    elif model_type == 'pinn':
        return PINNDeepONet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test different DeepONet architectures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 32
    n_sensors = 50
    n_queries = 100
    coord_dim = 2
    
    branch_input = torch.randn(batch_size, n_sensors).to(device)
    trunk_input = torch.randn(batch_size, n_queries, coord_dim).to(device)
    
    models = {
        'Standard DeepONet': create_deeponet('standard', 
                                           branch_input_dim=n_sensors,
                                           trunk_input_dim=coord_dim),
        'Modified DeepONet': create_deeponet('modified',
                                           branch_input_dim=n_sensors,
                                           trunk_input_dim=coord_dim,
                                           use_residual=True),
        'Fourier DeepONet': create_deeponet('fourier',
                                          branch_input_dim=n_sensors,
                                          trunk_input_dim=coord_dim),
        'MultiScale DeepONet': create_deeponet('multiscale',
                                             branch_input_dim=n_sensors,
                                             trunk_input_dim=coord_dim),
        'PINN DeepONet': create_deeponet('pinn',
                                       branch_input_dim=n_sensors,
                                       trunk_input_dim=coord_dim)
    }
    
    for name, model in models.items():
        model = model.to(device)
        with torch.no_grad():
            output = model(branch_input, trunk_input)
            print(f"{name}:")
            print(f"  Input: Branch {branch_input.shape}, Trunk {trunk_input.shape}")
            print(f"  Output: {output.shape}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print()