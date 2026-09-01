"""Descriptive statistics and least-squares fitting.

This file is a Quire module: ``register(api)`` adds functions to the worksheet
namespace. Arguments arrive as sympy objects (numbers, symbols, lists, matrices).
"""
import sympy as sp

NAME = "stats"
DESCRIPTION = "Descriptive statistics and linear regression."


def _values(xs):
    if isinstance(xs, sp.MatrixBase):
        return list(xs)
    if isinstance(xs, (list, tuple)):
        return list(xs)
    raise TypeError("Expected a list, e.g. [1, 2, 3].")


def mean(xs):
    v = _values(xs)
    return sp.Add(*v) / len(v)


def median(xs):
    v = sorted(_values(xs), key=lambda t: float(t))
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def variance(xs):
    v = _values(xs)
    m = mean(v)
    return sp.Add(*[(t - m) ** 2 for t in v]) / (len(v) - 1)


def stdev(xs):
    return sp.sqrt(variance(xs))


def linfit(xs, ys):
    """Least-squares line through the points; returns [slope, intercept]."""
    x, y = _values(xs), _values(ys)
    if len(x) != len(y):
        raise ValueError("x and y lists must have the same length.")
    n = len(x)
    sx, sy = sp.Add(*x), sp.Add(*y)
    sxx = sp.Add(*[t * t for t in x])
    sxy = sp.Add(*[a * b for a, b in zip(x, y)])
    slope = sp.simplify((n * sxy - sx * sy) / (n * sxx - sx ** 2))
    intercept = sp.simplify((sy - slope * sx) / n)
    return [slope, intercept]


def correlation(xs, ys):
    x, y = _values(xs), _values(ys)
    mx, my = mean(x), mean(y)
    num = sp.Add(*[(a - mx) * (b - my) for a, b in zip(x, y)])
    den = sp.sqrt(sp.Add(*[(a - mx) ** 2 for a in x]) * sp.Add(*[(b - my) ** 2 for b in y]))
    return num / den


def register(api):
    S = "Statistics"
    api.function("mean", mean, signature="mean([x1, x2, ...])", doc="arithmetic mean", category=S,
                 example="mean([1, 2, 3, 4])")
    api.function("median", median, signature="median([x1, x2, ...])", doc="middle value", category=S)
    api.function("variance", variance, signature="variance([...])", doc="sample variance", category=S)
    api.function("stdev", stdev, signature="stdev([...])", doc="sample standard deviation", category=S,
                 example="stdev([2, 4, 4, 4, 5, 5, 7, 9])")
    api.function("linfit", linfit, signature="linfit([x...], [y...])",
                 doc="least-squares line; gives [slope, intercept]", category=S,
                 example="linfit([0, 1, 2, 3], [1, 3, 5, 7.2])")
    api.function("correlation", correlation, signature="correlation([x...], [y...])",
                 doc="Pearson correlation coefficient", category=S)
