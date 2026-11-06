import numpy as np


def make_regression(n_samples=200, n_features=1, noise=1.0, coef=None, intercept=2.0,
                    feature_scale=5.0, seed=0):
    """Synthetic data: y = intercept + X @ coef + N(0, noise^2).

    X[:, j] ~ Uniform(-1, 1) * feature_scale[j]. Passing different scales per feature
    produces an ill-conditioned problem (useful for demonstrating why scaling matters).
    Returns X, y, true_coef, intercept.
    """
    rng = np.random.default_rng(seed)
    scale = np.broadcast_to(np.asarray(feature_scale, dtype=float), (n_features,))
    X = rng.uniform(-1, 1, size=(n_samples, n_features)) * scale
    coef = rng.normal(0, 3, size=n_features) if coef is None else np.asarray(coef, dtype=float)
    y = intercept + X @ coef + rng.normal(0, noise, size=n_samples)
    return X, y, coef, intercept
