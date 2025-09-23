"""
Physics-Informed Neural Networks (PINNs) module.
"""

from .pinns import PINNs, heat_equation_residual, burgers_equation_residual

__all__ = [
    "PINNs",
    "heat_equation_residual", 
    "burgers_equation_residual",
]