"""
AI4CFD: Various methods for solving PDEs using artificial intelligence

This package provides implementations of state-of-the-art AI methods for 
computational fluid dynamics and partial differential equation solving.
"""

__version__ = "0.1.0"
__author__ = "Wenjie Huang"

from .pinns import PINNs
from .deeponet import DeepONet
from .fno import FNO1d, FNO2d, FNO3d
from .neural_ode import NeuralODE
from . import utils

__all__ = [
    "PINNs",
    "DeepONet", 
    "FNO1d",
    "FNO2d",
    "FNO3d",
    "NeuralODE",
    "utils",
]