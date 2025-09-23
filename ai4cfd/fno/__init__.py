"""
Fourier Neural Operator (FNO) module.
"""

from .fno import FNO1d, FNO2d, FNO3d, SpectralConv1d, SpectralConv2d

__all__ = [
    "FNO1d",
    "FNO2d", 
    "FNO3d",
    "SpectralConv1d",
    "SpectralConv2d",
]