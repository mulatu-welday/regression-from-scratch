"""gdlab: linear regression and gradient-based optimizers, built from scratch with NumPy."""
from .linear import LinearRegression
from .optimizers import SGD, Adam
from .preprocessing import StandardScaler
from .metrics import mse, r2_score

__all__ = ["LinearRegression", "SGD", "Adam", "StandardScaler", "mse", "r2_score"]
__version__ = "0.1.0"
