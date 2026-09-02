"""Machine learning basics on lists and data files: regression, logistic regression, k-means, PCA, nearest neighbours."""
from __future__ import annotations

import numpy as np
import sympy as sp

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "ml"
DESCRIPTION = "Linear and logistic regression, k-means, PCA, k-nearest neighbours and scores on tables."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _matrix(X, what="X"):
    """A 2D float array from a matrix, a list of rows or a single list (one column)."""
    if isinstance(X, sp.MatrixBase):
        rows = X.tolist()
    elif isinstance(X, (list, tuple)):
        rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in X]
    else:
        raise EvalError(f"{what} must be a table: a matrix, read_csv(file) or a list of rows.")
    try:
        return np.array([[float(U.strip_units(sp.sympify(v))[0]) for v in r] for r in rows], dtype=float)
    except (TypeError, ValueError):
        raise EvalError(f"{what} must contain numbers only.") from None


def _vector(y, what="y"):
    M = _matrix(y, what)
    if M.shape[1] != 1 and M.shape[0] == 1:
        M = M.T
    if M.shape[1] != 1:
        raise EvalError(f"{what} must be a single list of numbers.")
    return M[:, 0]


def _clean(a):
    """Round away numerical dust so a zero intercept reads 0, not 7e-16."""
    a = np.asarray(a, dtype=float)
    big = np.abs(a).max() if a.size else 0.0
    return np.where(np.abs(a) < 1e-12 * max(big, 1.0), 0.0, a)


def _out(M, digits=6):
    return sp.ImmutableMatrix([[sp.Float(v, digits) if v != 0 else sp.S.Zero for v in row] for row in np.atleast_2d(_clean(M))])


def _list(v, digits=6):
    return [sp.Float(x, digits) if x != 0 else sp.S.Zero for x in _clean(v).ravel()]


def standardize(X):
    """Columns centred and scaled to unit variance."""
    M = _matrix(X)
    s = M.std(axis=0, ddof=0)
    s[s == 0] = 1
    return _out((M - M.mean(axis=0)) / s)


def train_test_split(X, fraction=sp.Rational(8, 10), seed=1):
    """[train rows, test rows] after a seeded shuffle."""
    M = _matrix(X)
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(M))
    k = int(round(float(fraction) * len(M)))
    return [_out(M[idx[:k]]), _out(M[idx[k:]])]


# ---- regression
def multi_regression(X, y):
    """Least-squares coefficients [intercept, w1, w2, ...] for y ≈ intercept + w · x."""
    M, yv = _matrix(X), _vector(y)
    if len(M) != len(yv):
        raise EvalError(f"X has {len(M)} rows but y has {len(yv)} values.")
    A = np.hstack([np.ones((len(M), 1)), M])
    w, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ w
    ss_res, ss_tot = float(np.sum((yv - pred) ** 2)), float(np.sum((yv - yv.mean()) ** 2))
    _note(f"R² = {1 - ss_res / ss_tot if ss_tot else 1:.4f} on {len(M)} rows")
    return _list(w)


def predict(coefficients, x):
    """intercept + w · x for coefficients from multi_regression."""
    w = _vector(coefficients, "coefficients")
    xv = np.atleast_1d(_matrix(x if isinstance(x, (list, tuple, sp.MatrixBase)) else [x], "x").ravel())
    if len(xv) != len(w) - 1:
        raise EvalError(f"Expected {len(w) - 1} feature values.")
    return sp.Float(w[0] + w[1:] @ xv, 6)


def logistic_regression(X, y, steps=500, rate=sp.Float(0.1)):
    """Coefficients [intercept, w...] of P(y=1) = sigmoid(intercept + w · x), by gradient descent on standardized features."""
    M, yv = _matrix(X), _vector(y)
    if len(M) != len(yv):
        raise EvalError(f"X has {len(M)} rows but y has {len(yv)} values.")
    if not set(np.unique(yv)) <= {0.0, 1.0}:
        raise EvalError("y must contain 0 and 1 only.")
    mean, std = M.mean(axis=0), M.std(axis=0)
    std[std == 0] = 1
    Z = np.hstack([np.ones((len(M), 1)), (M - mean) / std])
    w = np.zeros(Z.shape[1])
    lr = float(rate)
    for _ in range(int(steps)):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * Z.T @ (p - yv) / len(yv)
    # back to the original feature scale
    w_orig = np.concatenate([[w[0] - np.sum(w[1:] * mean / std)], w[1:] / std])
    p = 1 / (1 + np.exp(-(np.hstack([np.ones((len(M), 1)), M]) @ w_orig)))
    acc = float(np.mean((p >= 0.5) == (yv == 1)))
    _note(f"gradient descent, {int(steps)} steps; training accuracy {acc:.3f}")
    return _list(w_orig)


def logistic_predict(coefficients, x):
    """Probability of class 1."""
    w = _vector(coefficients, "coefficients")
    xv = _matrix(x if isinstance(x, (list, tuple, sp.MatrixBase)) else [x], "x").ravel()
    if len(xv) != len(w) - 1:
        raise EvalError(f"Expected {len(w) - 1} feature values.")
    return sp.Float(1 / (1 + np.exp(-(w[0] + w[1:] @ xv))), 6)


# ---- clustering and dimensionality reduction
def kmeans(X, k, seed=1, iterations=100):
    """Cluster centres (rows) by Lloyd's algorithm with k-means++ seeding."""
    M = _matrix(X)
    k = int(k)
    if not 1 <= k <= len(M):
        raise EvalError("k must be between 1 and the number of rows.")
    rng = np.random.default_rng(int(seed))
    centres = [M[rng.integers(len(M))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((M - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(M[rng.choice(len(M), p=d2 / d2.sum())] if d2.sum() > 0 else M[rng.integers(len(M))])
    C = np.array(centres)
    for _ in range(int(iterations)):
        labels = np.argmin(((M[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
        new = np.array([M[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    labels = np.argmin(((M[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
    inertia = float(np.sum((M - C[labels]) ** 2))
    _note(f"k-means: {k} clusters, sizes {[int(np.sum(labels == j)) for j in range(k)]}, inertia {inertia:.4g}")
    return _out(C)


def kmeans_labels(X, centres):
    """Cluster index (1-based) of every row for the given centres."""
    M, C = _matrix(X), _matrix(centres, "centres")
    labels = np.argmin(((M[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
    return [sp.Integer(int(v) + 1) for v in labels]


def pca(X, components=2):
    """Principal directions (rows) of the centred data; the explained variance goes in a note."""
    M = _matrix(X)
    Z = M - M.mean(axis=0)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    var = s ** 2 / max(len(M) - 1, 1)
    n = int(components)
    if not 1 <= n <= len(s):
        raise EvalError(f"components must be between 1 and {len(s)}.")
    _note("explained variance: " + ", ".join(f"{v / var.sum():.1%}" for v in var[:n]))
    return _out(vt[:n])


def pca_transform(X, components=2):
    """The data projected on its first principal components (rows)."""
    M = _matrix(X)
    Z = M - M.mean(axis=0)
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    n = int(components)
    return _out(Z @ vt[:n].T)


def explained_variance(X):
    """Fraction of variance carried by each principal component."""
    M = _matrix(X)
    Z = M - M.mean(axis=0)
    s = np.linalg.svd(Z, compute_uv=False) ** 2
    return _list(s / s.sum())


# ---- neighbours and scores
def knn_predict(X, y, x, k=3):
    """Majority class (or mean value for non-integer y) of the k nearest rows."""
    M, yv = _matrix(X), _vector(y)
    xv = _matrix(x if isinstance(x, (list, tuple, sp.MatrixBase)) else [x], "x").ravel()
    d = np.sqrt(np.sum((M - xv) ** 2, axis=1))
    idx = np.argsort(d)[: int(k)]
    vals = yv[idx]
    if np.all(vals == np.round(vals)):
        classes, counts = np.unique(vals, return_counts=True)
        return sp.Integer(int(classes[np.argmax(counts)]))
    return sp.Float(vals.mean(), 6)


def accuracy(y_true, y_pred):
    a, b = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    if len(a) != len(b):
        raise EvalError("The two lists differ in length.")
    return sp.Rational(int(np.sum(a == b)), len(a))


def confusion_matrix(y_true, y_pred):
    """Rows: true class, columns: predicted class (classes sorted)."""
    a, b = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    classes = sorted(set(a) | set(b))
    return sp.ImmutableMatrix([[int(np.sum((a == ci) & (b == cj))) for cj in classes] for ci in classes])


def r2_score(y_true, y_pred):
    a, b = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return sp.Float(1 - float(np.sum((a - b) ** 2)) / ss_tot, 6) if ss_tot else sp.S.One


def rmse(y_true, y_pred):
    a, b = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    return sp.Float(np.sqrt(np.mean((a - b) ** 2)), 6)


def register(api):
    R = "Machine learning"
    api.function("standardize", standardize, signature="standardize(X)", doc="columns centred and scaled", category=R)
    api.function("train_test_split", train_test_split, signature="train_test_split(X, fraction, seed)", doc="[train, test] rows", category=R)
    api.function("multi_regression", multi_regression, signature="multi_regression(X, y)", doc="[intercept, w...] by least squares (several features)", category=R,
                 example="multi_regression([[1, 2], [2, 1], [3, 4], [4, 3]], [5, 4, 11, 10])")
    api.function("predict", predict, signature="predict(coefficients, x)", doc="linear prediction", category=R)
    api.function("logistic_regression", logistic_regression, signature="logistic_regression(X, y, steps, rate)", doc="[intercept, w...] for P(y = 1)", category=R,
                 example="logistic_regression([[1], [2], [3], [6], [7], [8]], [0, 0, 0, 1, 1, 1])")
    api.function("logistic_predict", logistic_predict, signature="logistic_predict(coefficients, x)", doc="probability of class 1", category=R)
    api.function("kmeans", kmeans, signature="kmeans(X, k, seed)", doc="cluster centres", category=R,
                 example="kmeans([[1, 1], [1.2, 0.8], [0.9, 1.1], [8, 8], [8.2, 7.9], [7.8, 8.1]], 2)")
    api.function("kmeans_labels", kmeans_labels, signature="kmeans_labels(X, centres)", doc="cluster of every row", category=R)
    api.function("pca", pca, signature="pca(X, components)", doc="principal directions (rows)", category=R)
    api.function("pca_transform", pca_transform, signature="pca_transform(X, components)", doc="data projected on the components", category=R)
    api.function("explained_variance", explained_variance, signature="explained_variance(X)", doc="variance fraction per component", category=R)
    api.function("knn_predict", knn_predict, signature="knn_predict(X, y, x, k)", doc="k-nearest-neighbour class or value", category=R)
    api.function("accuracy", accuracy, signature="accuracy(y_true, y_pred)", doc="fraction of matches", category=R)
    api.function("confusion_matrix", confusion_matrix, signature="confusion_matrix(y_true, y_pred)", doc="true rows × predicted columns", category=R)
    api.function("r2_score", r2_score, signature="r2_score(y_true, y_pred)", doc="coefficient of determination", category=R)
    api.function("rmse", rmse, signature="rmse(y_true, y_pred)", doc="root-mean-square error", category=R)
