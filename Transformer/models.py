"""
Transformer models adapted for solving partial differential equations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding for spatial and temporal coordinates in PDEs.
    """
    
    def __init__(self, d_model, max_len=5000, coord_dim=1):
        super(PositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.coord_dim = coord_dim
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Use different frequencies for different dimensions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class PhysicsPositionalEncoding(nn.Module):
    """
    Positional encoding based on physical coordinates (x, y, t).
    """
    
    def __init__(self, d_model, coord_dim=2):
        super(PhysicsPositionalEncoding, self).__init__()
        
        self.d_model = d_model
        self.coord_dim = coord_dim
        
        # Linear projection for coordinates
        self.coord_proj = nn.Linear(coord_dim, d_model)
        
        # Frequency encoding
        self.freq_bands = nn.Parameter(torch.randn(d_model // 2, coord_dim))
    
    def forward(self, coords):
        """
        Encode physical coordinates.
        
        Args:
            coords: Coordinate tensor [batch_size, seq_len, coord_dim]
        
        Returns:
            Positional encoding [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = coords.shape
        
        # Linear encoding
        linear_encoding = self.coord_proj(coords)
        
        # Frequency encoding
        coords_expanded = coords.unsqueeze(-2)  # [batch, seq_len, 1, coord_dim]
        freq_expanded = self.freq_bands.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model//2, coord_dim]
        
        # Compute frequencies
        freqs = torch.sum(coords_expanded * freq_expanded, dim=-1)  # [batch, seq_len, d_model//2]
        
        # Apply sine and cosine
        sin_encoding = torch.sin(freqs)
        cos_encoding = torch.cos(freqs)
        
        freq_encoding = torch.cat([sin_encoding, cos_encoding], dim=-1)  # [batch, seq_len, d_model]
        
        return linear_encoding + freq_encoding


class PDETransformer(nn.Module):
    """
    Basic transformer for PDE solving using sequence-to-sequence approach.
    """
    
    def __init__(self, input_dim=1, output_dim=1, d_model=256, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=1024, dropout=0.1, coord_dim=2):
        super(PDETransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(output_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PhysicsPositionalEncoding(d_model, coord_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        initrange = 0.1
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_coords, tgt_coords, 
                src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass.
        
        Args:
            src: Source sequence [batch_size, src_len, input_dim]
            tgt: Target sequence [batch_size, tgt_len, output_dim]
            src_coords: Source coordinates [batch_size, src_len, coord_dim]
            tgt_coords: Target coordinates [batch_size, tgt_len, coord_dim]
        """
        # Embed inputs
        src_emb = self.input_embedding(src) + self.pos_encoding(src_coords)
        tgt_emb = self.output_embedding(tgt) + self.pos_encoding(tgt_coords)
        
        # Encode
        memory = self.transformer_encoder(src_emb, src_mask)
        
        # Decode
        output = self.transformer_decoder(
            tgt_emb, memory, tgt_mask, memory_mask
        )
        
        # Project to output
        return self.output_proj(output)


class VisionTransformerPDE(nn.Module):
    """
    Vision Transformer adapted for PDE solving by treating solution fields as images.
    """
    
    def __init__(self, img_size=64, patch_size=8, input_channels=1, 
                 output_channels=1, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformerPDE, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embed_dim = embed_dim
        
        # Number of patches
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head - reconstruct image
        self.head = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size * output_channels),
            nn.Unflatten(-1, (output_channels, patch_size, patch_size))
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_channels, height, width]
        
        Returns:
            Output tensor [batch_size, output_channels, height, width]
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [batch, embed_dim, h//p, w//p]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Remove class token and reshape for reconstruction
        x = x[:, 1:]  # Remove cls token
        
        # Reconstruct patches
        x = self.head(x)  # [batch, num_patches, output_channels, patch_size, patch_size]
        
        # Reshape to image
        h_patches = w_patches = int(math.sqrt(self.num_patches))
        x = x.view(batch_size, h_patches, w_patches, 
                  self.output_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(batch_size, self.output_channels, 
                  h_patches * self.patch_size, w_patches * self.patch_size)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer that processes spatial discretization points.
    """
    
    def __init__(self, input_dim=1, output_dim=1, d_model=256, nhead=8,
                 num_layers=6, coord_dim=2, max_seq_len=1024):
        super(SpatialTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Coordinate encoding
        self.coord_encoding = PhysicsPositionalEncoding(d_model, coord_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
    
    def forward(self, x, coords):
        """
        Forward pass.
        
        Args:
            x: Input field values [batch_size, seq_len, input_dim]
            coords: Spatial coordinates [batch_size, seq_len, coord_dim]
        
        Returns:
            Output field values [batch_size, seq_len, output_dim]
        """
        # Project input and add coordinate encoding
        x = self.input_proj(x) + self.coord_encoding(coords)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project to output
        return self.output_proj(x)


class TemporalTransformer(nn.Module):
    """
    Transformer for modeling temporal evolution of PDE solutions.
    """
    
    def __init__(self, field_dim=64*64, d_model=512, nhead=8, num_layers=6,
                 seq_len=10, output_steps=1):
        super(TemporalTransformer, self).__init__()
        
        self.field_dim = field_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.output_steps = output_steps
        
        # Field embedding
        self.field_embed = nn.Linear(field_dim, d_model)
        
        # Temporal positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + output_steps)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, field_dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input sequence [batch_size, seq_len, field_dim]
        
        Returns:
            Output predictions [batch_size, output_steps, field_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed fields
        x = self.field_embed(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply transformer
        x = self.transformer(x)
        
        # For autoregressive prediction, we can use the last hidden state
        # to generate future steps
        outputs = []
        hidden = x[:, -1:, :]  # Last time step
        
        for _ in range(self.output_steps):
            # Predict next step
            next_field = self.output_proj(hidden)
            outputs.append(next_field)
            
            # Use prediction as input for next step (autoregressive)
            hidden = self.field_embed(next_field)
            hidden = hidden + self.pos_encoding.pe[seq_len:seq_len+1].transpose(0, 1)
            hidden = self.transformer(hidden)
        
        return torch.cat(outputs, dim=1)


class PhysicsInformedTransformer(nn.Module):
    """
    Transformer with physics-informed constraints.
    """
    
    def __init__(self, input_dim=1, output_dim=1, d_model=256, nhead=8,
                 num_layers=6, coord_dim=2):
        super(PhysicsInformedTransformer, self).__init__()
        
        # Base transformer
        self.transformer = SpatialTransformer(
            input_dim, output_dim, d_model, nhead, num_layers, coord_dim
        )
    
    def forward(self, x, coords):
        """Forward pass with physics residual computation."""
        u_pred = self.transformer(x, coords)
        return u_pred
    
    def compute_physics_loss(self, u_pred, coords, pde_func):
        """
        Compute physics-informed loss.
        
        Args:
            u_pred: Predicted solution
            coords: Coordinates
            pde_func: Function that computes PDE residual
        
        Returns:
            Physics loss
        """
        # Enable gradient computation for coordinates
        coords.requires_grad_(True)
        
        # Compute PDE residual
        residual = pde_func(coords, u_pred)
        
        # Return L2 norm of residual
        return torch.mean(residual**2)


def create_transformer_model(model_type='pde_transformer', **kwargs):
    """
    Factory function to create different transformer models.
    
    Args:
        model_type: Type of transformer model
        **kwargs: Model-specific arguments
    
    Returns:
        Transformer model instance
    """
    models = {
        'pde_transformer': PDETransformer,
        'vision_transformer': VisionTransformerPDE,
        'spatial_transformer': SpatialTransformer,
        'temporal_transformer': TemporalTransformer,
        'physics_transformer': PhysicsInformedTransformer
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


if __name__ == "__main__":
    # Test transformer models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing Transformer models for PDEs...")
    
    # Test spatial transformer
    print("\nSpatial Transformer:")
    batch_size, seq_len, input_dim, coord_dim = 4, 256, 1, 2
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    coords = torch.randn(batch_size, seq_len, coord_dim).to(device)
    
    spatial_model = create_transformer_model(
        'spatial_transformer', 
        input_dim=input_dim, 
        output_dim=1, 
        d_model=128,
        coord_dim=coord_dim
    ).to(device)
    
    out_spatial = spatial_model(x, coords)
    print(f"Input: {x.shape}, Coords: {coords.shape} -> Output: {out_spatial.shape}")
    print(f"Parameters: {sum(p.numel() for p in spatial_model.parameters()):,}")
    
    # Test vision transformer
    print("\nVision Transformer:")
    x_img = torch.randn(4, 1, 64, 64).to(device)
    
    vit_model = create_transformer_model(
        'vision_transformer',
        img_size=64,
        patch_size=8,
        input_channels=1,
        output_channels=1,
        embed_dim=384,
        depth=6,
        num_heads=6
    ).to(device)
    
    out_vit = vit_model(x_img)
    print(f"Input: {x_img.shape} -> Output: {out_vit.shape}")
    print(f"Parameters: {sum(p.numel() for p in vit_model.parameters()):,}")
    
    # Test temporal transformer
    print("\nTemporal Transformer:")
    field_dim = 32 * 32  # 32x32 spatial field
    seq_len = 10
    x_temporal = torch.randn(4, seq_len, field_dim).to(device)
    
    temporal_model = create_transformer_model(
        'temporal_transformer',
        field_dim=field_dim,
        d_model=256,
        seq_len=seq_len,
        output_steps=3
    ).to(device)
    
    out_temporal = temporal_model(x_temporal)
    print(f"Input: {x_temporal.shape} -> Output: {out_temporal.shape}")
    print(f"Parameters: {sum(p.numel() for p in temporal_model.parameters()):,}")
    
    print("\nAll transformer model tests completed successfully!")