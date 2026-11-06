"""First-order optimizers. Each exposes `step(w, grad) -> new_w`."""
from __future__ import annotations

import numpy as np


class SGD:
    """Gradient descent with optional (heavy-ball) momentum.

    v <- momentum * v - lr * g
    w <- w + v
    With momentum=0 this is exactly  w <- w - lr * g.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr, self.momentum = lr, momentum
        self._v = None

    def step(self, w, grad):
        if self._v is None:
            self._v = np.zeros_like(w)
        self._v = self.momentum * self._v - self.lr * grad
        return w + self._v


class Adam:
    """Adam (Kingma & Ba, 2014): per-parameter step sizes from running moment estimates."""

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self._m = self._v = None
        self._t = 0

    def step(self, w, grad):
        if self._m is None:
            self._m, self._v = np.zeros_like(w), np.zeros_like(w)
        self._t += 1
        self._m = self.b1 * self._m + (1 - self.b1) * grad
        self._v = self.b2 * self._v + (1 - self.b2) * grad ** 2
        m_hat = self._m / (1 - self.b1 ** self._t)   # bias correction
        v_hat = self._v / (1 - self.b2 ** self._t)
        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
