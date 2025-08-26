"""
echostate: a batteries-included Echo State Network (ESN) package.

Public API:
- ESN, Reservoir, Trainer
- mean_absolute_error, mean_squared_error
- save_esn, load_esn
"""
from .core.esn import ESN
from .core.reservoir import Reservoir
from .core.trainer import Trainer
from .utils import mean_absolute_error, mean_squared_error


__all__ = [
    "ESN", "Reservoir", "Trainer",
    "mean_absolute_error", "mean_squared_error",
]

__version__ = "0.1.0"
