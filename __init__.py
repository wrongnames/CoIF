"""
CoIF: Covariance-Inverse Fusion for Zero-Cost Proxies
"""

from .coif import (
    CoIF,
    compute_weights,
    optimize_weights,
    process_proxy_values,
    fuse_proxies,
)

__version__ = "1.0.0"
__all__ = [
    "CoIF",
    "compute_weights",
    "optimize_weights", 
    "process_proxy_values",
    "fuse_proxies",
]
