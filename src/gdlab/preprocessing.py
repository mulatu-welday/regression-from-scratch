import numpy as np


class StandardScaler:
    """Rescale each feature to zero mean and unit variance."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.scale_ = np.where(std == 0, 1.0, std)  # avoid dividing by zero for constant columns
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.scale_ + self.mean_
