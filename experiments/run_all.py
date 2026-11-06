"""Run every experiment, print a summary, and save figures to docs/figures/.

    python experiments/run_all.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gdlab import LinearRegression, StandardScaler
from gdlab.data import make_regression
from gdlab.objective import gradient, loss, numerical_gradient

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def design(X):
    return np.column_stack([np.ones(len(X)), X])


def hessian_eigs(X):
    """Eigenvalues of the Hessian H = Xb^T Xb / n. Max stable lr is 2/lambda_max."""
    Xb = design(X)
    return np.linalg.eigvalsh(Xb.T @ Xb / len(Xb))


def epochs_to_gap(history, optimum, gap=1e-4):
    """First epoch at which loss - optimum < gap (None if never reached)."""
    h = np.asarray(history) - optimum
    hit = np.where(h < gap)[0]
    return int(hit[0]) if len(hit) else None


# ------------------------------------------------------------------ 1
def exp_correctness():
    print("\n[1] Correctness: gradient check and GD vs normal equation")
    X, y, _, _ = make_regression(200, 1, noise=1.0, coef=[3.0], seed=42)
    Xb = design(X)
    w = np.random.default_rng(0).normal(size=2)
    err = np.max(np.abs(gradient(w, Xb, y) - numerical_gradient(lambda v: loss(v, Xb, y), w)))
    print(f"  max |analytic - numerical gradient| = {err:.2e}")

    ne = LinearRegression("normal").fit(X, y)
    gd = LinearRegression("gd", lr=0.05, n_epochs=5000).fit(X, y)
    wn, wg = np.r_[ne.intercept_, ne.coef_], np.r_[gd.intercept_, gd.coef_]
    print(f"  normal eq: {wn}   GD: {wg}   ||diff|| = {np.linalg.norm(wn - wg):.2e}"
          f"   ({len(gd.history_) - 1} epochs)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].scatter(X[:, 0], y, s=10, alpha=0.5, label="data")
    xs = np.linspace(-5, 5, 100)
    ax[0].plot(xs, ne.predict(xs), "k-", lw=3, label="normal equation")
    ax[0].plot(xs, gd.predict(xs), "r--", lw=2, label="gradient descent")
    ax[0].set(title="Both solvers find the same line", xlabel="x", ylabel="y"); ax[0].legend()

    ax[1].plot(np.array(gd.history_) - ne.history_[0])
    ax[1].set(yscale="log", xlabel="epoch", ylabel="loss - optimal loss", title="Convergence (log scale)")

    w0s, w1s = np.linspace(-1, 5, 100), np.linspace(0, 6, 100)
    Z = np.array([[loss(np.array([a, b]), Xb, y) for a in w0s] for b in w1s])
    ax[2].contour(w0s, w1s, Z, levels=30, cmap="viridis")
    path, wv = [np.zeros(2)], np.zeros(2)
    for _ in range(60):
        wv = wv - 0.05 * gradient(wv, Xb, y); path.append(wv)
    path = np.array(path)
    ax[2].plot(path[:, 0], path[:, 1], "r.-", label="GD path")
    ax[2].plot(*wn, "k*", ms=15, label="normal equation")
    ax[2].set(xlabel="w0 (bias)", ylabel="w1 (slope)", title="Descent on the loss surface"); ax[2].legend()
    fig.tight_layout(); fig.savefig(FIG / "overview.png"); plt.close(fig)
    return dict(grad_err=err, diff=float(np.linalg.norm(wn - wg)))


# ------------------------------------------------------------------ 2
def exp_learning_rate():
    print("\n[2] Learning-rate sweep (x ~ U(-5,5))")
    X, y, _, _ = make_regression(200, 1, noise=1.0, coef=[3.0], seed=42)
    lam = hessian_eigs(X)
    lim = 2 / lam.max()
    print(f"  Hessian eigenvalues {lam.round(3)}  ->  theoretical stability limit 2/lambda_max = {lim:.4f}")
    opt = LinearRegression("normal").fit(X, y).history_[0]
    lrs = [0.001, 0.01, 0.05, 0.1, 0.2, lim * 0.99, lim * 1.02]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lr in lrs:
        m = LinearRegression("gd", lr=lr, n_epochs=200, tol=0).fit(X, y)
        h = np.maximum(np.array(m.history_) - opt, 1e-16)
        ax.plot(h, label=f"lr={lr:.3g}" + ("  (above 2/λmax: unstable)" if lr > lim else ""))
        print(f"  lr={lr:<8.4g} loss gap after 200 epochs = {m.history_[-1] - opt:.3e}   "
              f"above theoretical limit: {lr > lim}")
    ax.set(yscale="log", xlabel="epoch", ylabel="loss - optimal loss", title="Effect of the learning rate")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "learning_rate.png"); plt.close(fig)
    return dict(limit=float(lim))


# ------------------------------------------------------------------ 3
def exp_scaling():
    print("\n[3] Feature scaling (two features on scales 1 and 30)")
    X, y, _, _ = make_regression(500, 2, noise=1.0, feature_scale=[1.0, 30.0], seed=1)
    Xs = StandardScaler().fit_transform(X)
    out = {}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, data in (("raw features", X), ("standardised", Xs)):
        lam = hessian_eigs(data)
        lr = 1.0 / lam.max()            # a good, safe step size for *this* problem
        ne_loss = LinearRegression("normal").fit(data, y).history_[0]
        m = LinearRegression("gd", lr=lr, n_epochs=20000, tol=0).fit(data, y)
        e = epochs_to_gap(m.history_, ne_loss)
        cond = lam.max() / lam.min()
        out[name] = (cond, e)
        print(f"  {name:<13} condition number = {cond:9.1f}  lr = {lr:.4f}  epochs to loss-gap<1e-4: {e}")
        ax.plot(np.maximum(np.array(m.history_) - ne_loss, 1e-16), label=f"{name} (cond={cond:.0f})")
    ax.axhline(1e-4, color="gray", ls=":")
    ax.set(yscale="log", xscale="symlog", xlabel="epoch", ylabel="loss - optimal loss",
           title="Scaling features makes the problem easy")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "scaling.png"); plt.close(fig)
    return out


# ------------------------------------------------------------------ 4
def exp_optimizers():
    print("\n[4] Optimizer comparison on the ill-conditioned (unscaled) problem")
    X, y, _, _ = make_regression(500, 2, noise=1.0, feature_scale=[1.0, 30.0], seed=1)
    lam = hessian_eigs(X)
    opt = LinearRegression("normal").fit(X, y).history_[0]
    L = lam.max()
    configs = [
        ("GD (lr=1/L)",              dict(optimizer="sgd", lr=1.0 / L)),
        ("Momentum 0.9 (lr=1/L)",    dict(optimizer="momentum", lr=1.0 / L, momentum=0.9)),
        ("Adam (lr=0.5)",            dict(optimizer="adam", lr=0.5)),
        ("Mini-batch SGD (bs=32)",   dict(optimizer="sgd", lr=0.5 / L, batch_size=32)),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, kw in configs:
        m = LinearRegression("gd", n_epochs=1500, tol=0, **kw).fit(X, y)
        h = np.maximum(np.array(m.history_) - opt, 1e-16)
        e = epochs_to_gap(m.history_, opt)
        print(f"  {name:<26} epochs to gap<1e-4: {e}   final gap: {h[-1]:.2e}")
        ax.plot(h, label=name)
    ax.set(yscale="log", xlabel="epoch", ylabel="loss - optimal loss", title="Optimizers on an ill-conditioned problem")
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(FIG / "optimizers.png"); plt.close(fig)


# ------------------------------------------------------------------ 5
def exp_ridge():
    print("\n[5] Ridge regression with collinear features (train / validation / test = 50/50/50)")
    rng = np.random.default_rng(7)
    n = 150
    x1 = rng.normal(size=n)
    X = np.column_stack([x1, x1 + 0.01 * rng.normal(size=n), rng.normal(size=n)])
    y = 1.0 + 2.0 * x1 + 0.5 * X[:, 2] + rng.normal(0, 0.5, size=n)
    Xtr, ytr = X[:50], y[:50]
    Xva, yva = X[50:100], y[50:100]
    Xte, yte = X[100:], y[100:]
    mse_ = lambda m, A, b: float(np.mean((m.predict(A) - b) ** 2))
    l2s = np.logspace(-6, 2, 40)
    models = [LinearRegression("normal", l2=l2).fit(Xtr, ytr) for l2 in l2s]
    paths = np.array([m.coef_ for m in models])
    val_mse = np.array([mse_(m, Xva, yva) for m in models])
    best = int(np.argmin(val_mse))          # l2 chosen on validation data only
    ols, ridge = models[0], models[best]
    print("  true coefficients: [2.0, 0.0, 0.5]  (features 1 and 2 are near-copies)")
    print(f"  OLS   (l2~0)     coefs {ols.coef_.round(2)}   test MSE {mse_(ols, Xte, yte):.3f}")
    print(f"  Ridge (l2={l2s[best]:.3g}) coefs {ridge.coef_.round(2)}   test MSE {mse_(ridge, Xte, yte):.3f}")
    Xs = StandardScaler().fit_transform(Xtr)
    a = LinearRegression("normal", l2=0.5).fit(Xs, ytr)
    b = LinearRegression("gd", l2=0.5, lr=0.1, n_epochs=5000).fit(Xs, ytr)
    print(f"  GD vs closed-form ridge, max coef diff: {np.max(np.abs(a.coef_ - b.coef_)):.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j in range(3):
        ax[0].plot(l2s, paths[:, j], label=f"feature {j + 1}" + (" (≈ copy of 1)" if j == 1 else ""))
    ax[0].set(xscale="log", xlabel="l2 penalty", ylabel="coefficient", title="Ridge path: collinear coefficients stabilise")
    ax[0].legend()
    ax[1].plot(l2s, val_mse); ax[1].axvline(l2s[best], color="r", ls="--", label=f"chosen l2={l2s[best]:.2g}")
    ax[1].set(xscale="log", xlabel="l2 penalty", ylabel="validation MSE", title="Validation error vs penalty"); ax[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "ridge.png"); plt.close(fig)


# ------------------------------------------------------------------ 6
def exp_timing():
    print("\n[6] Cost vs number of features (n=5000 samples)")
    ds = [10, 50, 100, 250, 500, 1000]
    t_ne, t_epoch = [], []
    for d in ds:
        X, y, _, _ = make_regression(5000, d, noise=1.0, feature_scale=1.0, seed=0)
        t0 = time.perf_counter(); LinearRegression("normal").fit(X, y); t_ne.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); LinearRegression("gd", lr=0.01, n_epochs=50, tol=0).fit(X, y)
        t_epoch.append((time.perf_counter() - t0) / 50)
        print(f"  d={d:<5} normal eq: {t_ne[-1] * 1e3:8.2f} ms     one GD epoch: {t_epoch[-1] * 1e3:7.3f} ms")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ds, np.array(t_ne) * 1e3, "o-", label="normal equation (whole fit)")
    ax.loglog(ds, np.array(t_epoch) * 1e3, "s-", label="one GD epoch")
    ax.set(xlabel="number of features", ylabel="time (ms)", title="Cost vs number of features"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "timing.png"); plt.close(fig)


if __name__ == "__main__":
    exp_correctness(); exp_learning_rate(); exp_scaling(); exp_optimizers(); exp_ridge(); exp_timing()
    print(f"\nFigures saved to {FIG}")
