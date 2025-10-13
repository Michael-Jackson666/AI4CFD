"""
Transformer-based architecture for PINN.
Uses multi-head self-attention to capture long-range dependencies in the Vlasov-Poisson system.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for input coordinates (t, x, v).
    Helps the transformer understand the ordering and relationships of inputs.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerPINN(nn.Module):
    """
    Transformer-based PINN architecture for solving the Vlasov-Poisson system.
    
    Architecture:
    1. Input embedding: (t, x, v) -> d_model dimensional space
    2. Positional encoding: Add position information
    3. Multi-head self-attention layers: Capture correlations
    4. Feed-forward networks: Non-linear transformations
    5. Output projection: Map back to f(t, x, v)
    
    Args:
        input_dim (int): Number of input features (t, x, v) = 3
        output_dim (int): Number of output features (f) = 1
        d_model (int): Dimension of the model (embedding size)
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        dim_feedforward (int): Dimension of feedforward network
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim=3, output_dim=1, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.1):
        super(TransformerPINN, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding: map (t, x, v) to d_model dimensions
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection: map from d_model to output_dim
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, output_dim),
            nn.Softplus()  # Ensure f >= 0
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass through the transformer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
                              where input_dim = 3 (t, x, v)
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
                         representing f(t, x, v)
        """
        # Input embedding: [batch_size, 3] -> [batch_size, 1, d_model]
        x = self.input_embedding(x).unsqueeze(1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension and project to output
        x = x.squeeze(1)  # [batch_size, d_model]
        output = self.output_projection(x)  # [batch_size, output_dim]
        
        return output


class HybridTransformerPINN(nn.Module):
    """
    Hybrid architecture combining Transformer and MLP.
    Uses Transformer for global feature extraction and MLP for local refinement.
    
    This hybrid approach can capture both:
    - Long-range dependencies (via Transformer)
    - Local non-linear patterns (via MLP)
    
    Args:
        input_dim (int): Number of input features (t, x, v) = 3
        output_dim (int): Number of output features (f) = 1
        d_model (int): Transformer embedding dimension
        nhead (int): Number of attention heads
        num_transformer_layers (int): Number of transformer layers
        num_mlp_layers (int): Number of MLP layers
        mlp_neurons (int): Number of neurons per MLP layer
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim=3, output_dim=1, d_model=256, nhead=8,
                 num_transformer_layers=4, num_mlp_layers=4, mlp_neurons=512, dropout=0.1):
        super(HybridTransformerPINN, self).__init__()
        
        # Transformer branch for global features
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        
        # MLP branch for local features
        mlp_modules = [nn.Linear(input_dim, mlp_neurons), nn.Tanh()]
        for _ in range(num_mlp_layers - 2):
            mlp_modules.append(nn.Linear(mlp_neurons, mlp_neurons))
            mlp_modules.append(nn.Tanh())
        mlp_modules.append(nn.Linear(mlp_neurons, d_model))
        self.mlp_branch = nn.Sequential(*mlp_modules)
        
        # Fusion layer: combine transformer and MLP features
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
            nn.Softplus()  # Ensure f >= 0
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1]
        """
        # Transformer branch: global features
        x_trans = self.input_embedding(x).unsqueeze(1)
        x_trans = self.pos_encoder(x_trans)
        x_trans = self.transformer_encoder(x_trans).squeeze(1)
        
        # MLP branch: local features
        x_mlp = self.mlp_branch(x)
        
        # Concatenate and fuse features
        x_combined = torch.cat([x_trans, x_mlp], dim=1)
        output = self.fusion(x_combined)
        
        return output


class LightweightTransformerPINN(nn.Module):
    """
    Lightweight Transformer for faster training.
    Reduced parameters while maintaining good performance.
    
    Good for quick experimentation and prototyping.
    """
    def __init__(self, input_dim=3, output_dim=1, d_model=128, nhead=4, 
                 num_layers=3, dim_feedforward=512, dropout=0.1):
        super(LightweightTransformerPINN, self).__init__()
        
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU()
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim),
            nn.Softplus()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        x = self.input_embedding(x).unsqueeze(1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x).squeeze(1)
        return self.output_projection(x)
