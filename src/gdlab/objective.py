"""Loss, analytic gradient, and a numerical gradient for verifying it.

Objective (ridge-regularised least squares; l2=0 gives plain least squares):

    L(w) = 1/(2n) * ||X w - y||^2  +  (l2/2) * ||mask * w||^2
    dL/dw = 1/n * X^T (X w - y)    +  l2 * mask * w

`mask` is 0 for the bias weight and 1 elsewhere, so the intercept is never penalised.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def loss(w, X, y, l2: float = 0.0, mask=None) -> float:
    r = X @ w - y
    value = 0.5 * np.mean(r ** 2)
    if l2:
        m = np.ones_like(w) if mask is None else mask
        value += 0.5 * l2 * np.sum((m * w) ** 2)
    return float(value)


def gradient(w, X, y, l2: float = 0.0, mask=None):
    g = X.T @ (X @ w - y) / len(y)
    if l2:
        m = np.ones_like(w) if mask is None else mask
        g = g + l2 * m * w
    return g


def numerical_gradient(f: Callable[[np.ndarray], float], w, eps: float = 1e-6):
    """Central finite differences: (f(w+eps*e_i) - f(w-eps*e_i)) / (2*eps)."""
    g = np.zeros_like(w, dtype=float)
    for i in range(len(w)):
        step = np.zeros_like(w, dtype=float)
        step[i] = eps
        g[i] = (f(w + step) - f(w - step)) / (2 * eps)
    return g
