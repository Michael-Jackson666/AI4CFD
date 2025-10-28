"""
Transformer architecture for Vlasov-Poisson System.
Uses attention mechanisms to capture long-range dependencies in phase space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import TransformerConfig


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Adds positional information to the input embeddings.
    """
    
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Patch2DEncoding(nn.Module):
    """
    Convert 2D phase space to patches for transformer processing.
    Similar to Vision Transformer (ViT) approach.
    """
    
    def __init__(self, input_channels=1, patch_size=8, d_model=256):
        super(Patch2DEncoding, self).__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding: flatten patches and project to d_model
        patch_dim = input_channels * patch_size * patch_size
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embedding (will be initialized based on input size)
        self.pos_embedding = None
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, channels, height, width]
        
        Returns:
            Patches [batch_size, num_patches+1, d_model]
        """
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Ensure dimensions are divisible by patch_size
        assert H % P == 0 and W % P == 0, f"Image size ({H}, {W}) must be divisible by patch size {P}"
        
        # Number of patches
        n_h = H // P
        n_w = W // P
        n_patches = n_h * n_w
        
        # Extract patches: [B, C, H, W] -> [B, n_patches, C*P*P]
        x = x.unfold(2, P, P).unfold(3, P, P)  # [B, C, n_h, n_w, P, P]
        x = x.contiguous().view(B, C, n_patches, P*P)
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, n_patches, C, P*P]
        x = x.view(B, n_patches, -1)  # [B, n_patches, C*P*P]
        
        # Embed patches
        x = self.patch_embedding(x)  # [B, n_patches, d_model]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, n_patches+1, d_model]
        
        # Add positional embedding
        if self.pos_embedding is None or self.pos_embedding.size(1) != x.size(1):
            self.pos_embedding = nn.Parameter(
                torch.randn(1, x.size(1), self.d_model)
            ).to(x.device)
        
        x = x + self.pos_embedding
        
        return x


class VPTransformer(nn.Module):
    """
    Transformer model for Vlasov-Poisson phase space evolution.
    
    Uses patch-based encoding and self-attention to model
    the evolution of the distribution function.
    """
    
    def __init__(self, config: TransformerConfig):
        super(VPTransformer, self).__init__()
        
        self.config = config
        
        if config.use_patches:
            # Patch-based encoding
            self.patch_encoder = Patch2DEncoding(
                input_channels=3,  # (f0, x, v)
                patch_size=config.patch_size,
                d_model=config.d_model
            )
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(
                d_model=config.d_model,
                dropout=config.dropout
            )
        else:
            # Direct flattening approach
            self.input_proj = nn.Linear(config.input_dim * 3, config.d_model)
            self.pos_encoder = PositionalEncoding(
                d_model=config.d_model,
                dropout=config.dropout
            )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output projection
        if config.use_patches:
            # Reconstruct from patches
            Nx = Nv = int(math.sqrt(config.output_dim))
            n_patches = (Nx // config.patch_size) * (Nv // config.patch_size)
            patch_dim = config.patch_size * config.patch_size
            
            self.output_proj = nn.Sequential(
                nn.Linear(config.d_model, patch_dim),
                nn.GELU(),
                nn.Linear(patch_dim, patch_dim)
            )
            self.patch_size = config.patch_size
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(config.d_model, config.dim_feedforward),
                nn.GELU(),
                nn.Linear(config.dim_feedforward, config.output_dim)
            )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
               If use_patches: [batch, 3, Nx, Nv]
               Else: [batch, 3*input_dim]
        
        Returns:
            Output tensor [batch, 1, Nx, Nv]
        """
        B = x.shape[0]
        
        if self.config.use_patches:
            # Patch-based processing
            x = self.patch_encoder(x)  # [B, n_patches+1, d_model]
            x = self.pos_encoder(x)
            
            # Apply transformer
            x = self.transformer_encoder(x)
            
            # Remove class token and reconstruct
            x = x[:, 1:, :]  # Remove CLS token [B, n_patches, d_model]
            
            # Project to patch space
            x = self.output_proj(x)  # [B, n_patches, patch_size^2]
            
            # Reconstruct 2D image from patches
            Nx = Nv = int(math.sqrt(self.config.output_dim))
            n_h = n_w = Nx // self.patch_size
            P = self.patch_size
            
            # Reshape: [B, n_patches, P^2] -> [B, n_h, n_w, P, P]
            x = x.view(B, n_h, n_w, P, P)
            
            # Rearrange to image: [B, n_h, n_w, P, P] -> [B, Nx, Nv]
            x = x.permute(0, 1, 3, 2, 4).contiguous()
            x = x.view(B, Nx, Nv)
            
            # Add channel dimension
            x = x.unsqueeze(1)  # [B, 1, Nx, Nv]
        
        else:
            # Flatten input: [B, 3, Nx, Nv] -> [B, 1, 3*Nx*Nv]
            x = x.view(B, 1, -1)
            
            # Project to d_model
            x = self.input_proj(x)  # [B, 1, d_model]
            x = self.pos_encoder(x)
            
            # Apply transformer
            x = self.transformer_encoder(x)
            
            # Project to output
            x = self.output_proj(x)  # [B, 1, output_dim]
            
            # Reshape to 2D
            Nx = Nv = int(math.sqrt(self.config.output_dim))
            x = x.view(B, 1, Nx, Nv)
        
        return x


class HybridFNOTransformer(nn.Module):
    """
    Hybrid model combining FNO and Transformer.
    
    Uses FNO for local spectral features and Transformer for
    long-range dependencies.
    """
    
    def __init__(self, fno_config, transformer_config):
        super(HybridFNOTransformer, self).__init__()
        
        # Import here to avoid circular dependency
        from vp_fno import VPFNO2d
        
        # FNO for spectral processing
        self.fno = VPFNO2d(fno_config)
        
        # Transformer for attention mechanism
        # Reduce d_model for efficiency
        reduced_config = transformer_config
        reduced_config.use_patches = True
        reduced_config.d_model = 128
        
        self.transformer = VPTransformer(reduced_config)
        
        # Fusion layer
        self.fusion = nn.Conv2d(2, 1, 1)
    
    def forward(self, x):
        """
        Forward pass combining FNO and Transformer.
        
        Args:
            x: Input [batch, 3, Nx, Nv]
        
        Returns:
            Output [batch, 1, Nx, Nv]
        """
        # FNO path - spectral processing
        fno_out = self.fno(x)  # [B, 1, Nx, Nv]
        
        # Transformer path - attention mechanism
        trans_out = self.transformer(x)  # [B, 1, Nx, Nv]
        
        # Combine outputs
        combined = torch.cat([fno_out, trans_out], dim=1)  # [B, 2, Nx, Nv]
        output = self.fusion(combined)  # [B, 1, Nx, Nv]
        
        return output


class SpatialTemporalAttention(nn.Module):
    """
    Spatial-Temporal attention mechanism for phase space.
    
    Separately models spatial (x) and velocity (v) attention.
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SpatialTemporalAttention, self).__init__()
        
        # Spatial attention (x-direction)
        self.spatial_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Velocity attention (v-direction)
        self.velocity_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward networks
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        """
        # Spatial attention with residual
        attn_out, _ = self.spatial_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Velocity attention with residual
        attn_out, _ = self.velocity_attn(x, x, x)
        x = self.norm2(x + attn_out)
        
        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x


def create_vp_transformer_model(config: TransformerConfig, model_type="standard"):
    """
    Factory function to create VP-Transformer models.
    
    Args:
        config: Transformer configuration
        model_type: "standard", "hybrid"
    
    Returns:
        VP-Transformer model instance
    """
    if model_type == "standard":
        return VPTransformer(config)
    elif model_type == "hybrid":
        # Need FNO config for hybrid model
        from config import get_default_config
        full_config = get_default_config()
        return HybridFNOTransformer(full_config.fno, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    from config import get_default_config
    
    print("Testing VP-Transformer models...")
    
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
    
    input_data = torch.cat([f0, x_grid, v_grid], dim=1)  # [B, 3, Nx, Nv]
    
    # Test standard Transformer
    print("\nStandard VP-Transformer (with patches):")
    model_trans = create_vp_transformer_model(config.transformer, "standard").to(device)
    out_trans = model_trans(input_data)
    print(f"Input: {input_data.shape} -> Output: {out_trans.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_trans.parameters()):,}")
    
    # Test Hybrid model
    print("\nHybrid FNO-Transformer:")
    model_hybrid = create_vp_transformer_model(config.transformer, "hybrid").to(device)
    out_hybrid = model_hybrid(input_data)
    print(f"Input: {input_data.shape} -> Output: {out_hybrid.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_hybrid.parameters()):,}")
    
    print("\nAll VP-Transformer tests passed!")
