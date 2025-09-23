"""
Deep Operator Networks (DeepONet) module.
"""

from .deeponet import DeepONet, PODDeepONet, compute_pod_basis

__all__ = [
    "DeepONet",
    "PODDeepONet",
    "compute_pod_basis",
]