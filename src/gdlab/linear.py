"""Linear regression with a closed-form solver and gradient-based solvers."""
from __future__ import annotations

import numpy as np

from .objective import gradient, loss
from .optimizers import SGD, Adam


class LinearRegression:
    """Linear (optionally ridge) regression, scikit-learn style API.

    Parameters
    ----------
    solver : "normal" | "gd"
        "normal" solves (X^T X / n + l2*I') w = X^T y / n directly.
        "gd" minimises the same objective with a first-order optimizer.
    optimizer : "sgd" | "momentum" | "adam"      (solver="gd" only)
    lr, momentum : step size, and momentum coefficient for optimizer="momentum".
    n_epochs : passes over the data (gd only).
    batch_size : None = full-batch gradient descent; int = mini-batch SGD.
    l2 : ridge penalty strength (the intercept is never penalised).
    tol : stop when the epoch-to-epoch loss change falls below this.
    """

    def __init__(self, solver="normal", optimizer="sgd", lr=0.01, momentum=0.9,
                 n_epochs=1000, batch_size=None, l2=0.0, tol=1e-12, seed=0, fit_intercept=True):
        self.solver, self.optimizer, self.lr, self.momentum = solver, optimizer, lr, momentum
        self.n_epochs, self.batch_size, self.l2, self.tol = n_epochs, batch_size, l2, tol
        self.seed, self.fit_intercept = seed, fit_intercept

    # -- helpers ---------------------------------------------------------
    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.column_stack([np.ones(len(X)), X]) if self.fit_intercept else X

    def _make_optimizer(self):
        if self.optimizer == "sgd":
            return SGD(self.lr, momentum=0.0)
        if self.optimizer == "momentum":
            return SGD(self.lr, momentum=self.momentum)
        if self.optimizer == "adam":
            return Adam(self.lr)
        raise ValueError(f"unknown optimizer {self.optimizer!r}")

    # -- API -------------------------------------------------------------
    def fit(self, X, y):
        Xb, y = self._design(X), np.asarray(y, dtype=float)
        n, p = Xb.shape
        mask = np.ones(p)
        if self.fit_intercept:
            mask[0] = 0.0  # don't regularise the bias
        self.diverged_ = False

        if self.solver == "normal":
            A = Xb.T @ Xb / n + self.l2 * np.diag(mask)
            b = Xb.T @ y / n
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:       # singular (e.g. perfectly collinear features)
                w = np.linalg.lstsq(A, b, rcond=None)[0]
            self.history_ = [loss(w, Xb, y, self.l2, mask)]

        elif self.solver == "gd":
            rng = np.random.default_rng(self.seed)
            opt = self._make_optimizer()
            bs = n if self.batch_size is None else min(self.batch_size, n)
            w = np.zeros(p)
            self.history_ = [loss(w, Xb, y, self.l2, mask)]
            with np.errstate(over="ignore", invalid="ignore"):
                for _ in range(self.n_epochs):
                    idx = rng.permutation(n) if bs < n else np.arange(n)
                    for s in range(0, n, bs):
                        b = idx[s:s + bs]
                        w = opt.step(w, gradient(w, Xb[b], y[b], self.l2, mask))
                    self.history_.append(loss(w, Xb, y, self.l2, mask))
                    # diverged: non-finite, or loss blown up by >10 orders of magnitude
                    if (not np.isfinite(self.history_[-1])
                            or self.history_[-1] > 1e10 * max(self.history_[0], 1e-12)):
                        self.diverged_ = True
                        break
                    if abs(self.history_[-2] - self.history_[-1]) < self.tol:
                        break
        else:
            raise ValueError(f"unknown solver {self.solver!r}")

        if self.fit_intercept:
            self.intercept_, self.coef_ = float(w[0]), w[1:]
        else:
            self.intercept_, self.coef_ = 0.0, w
        self._w = w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        from .metrics import r2_score
        return r2_score(y, self.predict(X))
