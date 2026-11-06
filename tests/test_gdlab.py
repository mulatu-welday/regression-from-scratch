import numpy as np

from gdlab import LinearRegression, StandardScaler, r2_score
from gdlab.data import make_regression
from gdlab.objective import gradient, loss, numerical_gradient


def _problem(n_features=3, seed=0, **kw):
    X, y, coef, b = make_regression(n_samples=300, n_features=n_features, noise=0.5, seed=seed, **kw)
    return X, y, coef, b


def test_analytic_gradient_matches_numerical():
    X, y, *_ = _problem()
    Xb = np.column_stack([np.ones(len(X)), X])
    mask = np.array([0.0, 1, 1, 1])
    w = np.random.default_rng(1).normal(size=4)
    for l2 in (0.0, 0.7):
        g_a = gradient(w, Xb, y, l2, mask)
        g_n = numerical_gradient(lambda v: loss(v, Xb, y, l2, mask), w)
        assert np.max(np.abs(g_a - g_n)) < 1e-6


def test_normal_equation_recovers_true_parameters():
    X, y, coef, b = _problem()
    m = LinearRegression("normal").fit(X, y)
    assert np.allclose(m.coef_, coef, atol=0.1)
    assert abs(m.intercept_ - b) < 0.2


def test_gradient_descent_matches_normal_equation():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    ne = LinearRegression("normal").fit(Xs, y)
    gd = LinearRegression("gd", lr=0.1, n_epochs=5000).fit(Xs, y)
    assert np.allclose(gd.coef_, ne.coef_, atol=1e-6)
    assert abs(gd.intercept_ - ne.intercept_) < 1e-6


def test_loss_decreases_monotonically_for_stable_lr():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    gd = LinearRegression("gd", lr=0.1, n_epochs=200).fit(Xs, y)
    h = np.array(gd.history_)
    assert np.all(np.diff(h) <= 1e-12)


def test_too_large_learning_rate_diverges():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    gd = LinearRegression("gd", lr=5.0, n_epochs=200).fit(Xs, y)
    assert gd.diverged_


def test_ridge_gd_matches_closed_form_and_shrinks_coefficients():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    ols = LinearRegression("normal").fit(Xs, y)
    ridge_ne = LinearRegression("normal", l2=1.0).fit(Xs, y)
    ridge_gd = LinearRegression("gd", l2=1.0, lr=0.1, n_epochs=5000).fit(Xs, y)
    assert np.allclose(ridge_gd.coef_, ridge_ne.coef_, atol=1e-6)
    assert np.linalg.norm(ridge_ne.coef_) < np.linalg.norm(ols.coef_)


def test_intercept_is_not_penalised():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    heavy = LinearRegression("normal", l2=1e6).fit(Xs, y)
    assert abs(heavy.intercept_ - y.mean()) < 1e-6   # coefs ~ 0, bias still fits the mean


def test_adam_and_momentum_converge():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    ne = LinearRegression("normal").fit(Xs, y)
    for opt, lr in (("momentum", 0.05), ("adam", 0.05)):
        m = LinearRegression("gd", optimizer=opt, lr=lr, n_epochs=3000).fit(Xs, y)
        assert np.allclose(m.coef_, ne.coef_, atol=1e-3), opt


def test_minibatch_sgd_gets_close_and_is_reproducible():
    X, y, *_ = _problem()
    Xs = StandardScaler().fit_transform(X)
    ne = LinearRegression("normal").fit(Xs, y)
    a = LinearRegression("gd", lr=0.01, batch_size=32, n_epochs=300, seed=3).fit(Xs, y)
    b = LinearRegression("gd", lr=0.01, batch_size=32, n_epochs=300, seed=3).fit(Xs, y)
    assert np.allclose(a.coef_, ne.coef_, atol=0.1)
    assert np.array_equal(a.coef_, b.coef_)


def test_scaler_zero_mean_unit_variance_and_inverse():
    X, *_ = _problem(feature_scale=[1.0, 10.0, 100.0])
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-12) and np.allclose(Z.std(0), 1)
    assert np.allclose(sc.inverse_transform(Z), X)


def test_r2_perfect_and_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0
    assert abs(r2_score(y, np.full(4, y.mean()))) < 1e-12


def test_predict_shape_and_score():
    X, y, *_ = _problem()
    m = LinearRegression("normal").fit(X, y)
    assert m.predict(X).shape == y.shape
    assert m.score(X, y) > 0.99
