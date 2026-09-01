"""Integral transforms and recurrences."""
import sympy as sp

from ...engine.errors import EvalError
from ._util import as_residual, sym


def _laplace(f, t, s):
    return sp.laplace_transform(f, sym(t), sym(s), noconds=True)


def _ilaplace(F, s, t):
    return sp.inverse_laplace_transform(F, sym(s), sym(t))


def _fourier(f, x, k):
    return sp.fourier_transform(f, sym(x), sym(k))


def _ifourier(F, k, x):
    return sp.inverse_fourier_transform(F, sym(k), sym(x))


def _ztransform(f, n, z):
    # sympy has no direct Z transform; sum the definition.
    return sp.summation(f * z ** (-sym(n)), (n, 0, sp.oo))


def _rsolve(eq, a, n, *initial):
    """Solve a recurrence written with a[n], a[n+1], ...: rsolve(a[n+1] == 2 a[n], a, n, 1)."""
    a, n = sym(a), sym(n)
    f = sp.Function(a.name)
    expr = as_residual(eq).replace(lambda e: isinstance(e, sp.Indexed) and str(e.base.label) == a.name,
                                   lambda e: f(e.indices[0]))
    if not expr.has(f):
        raise EvalError(f"Write the sequence as {a}[n], {a}[n+1], ...")
    ics = {f(i): v for i, v in enumerate(initial)} or None
    res = sp.rsolve(expr, f(n), ics)
    if res is None:
        raise EvalError("No closed form found for this recurrence.")
    return res


def register(api):
    T = "Transforms"
    api.function("laplace", _laplace, signature="laplace(f, t, s)", doc="Laplace transform", category=T,
                 example="laplace(exp(-a t), t, s)")
    api.function("ilaplace", _ilaplace, signature="ilaplace(F, s, t)", doc="inverse Laplace transform",
                 category=T, example="ilaplace(1/(s + a), s, t)")
    api.function("fourier", _fourier, signature="fourier(f, x, k)", doc="Fourier transform (e^{-2πikx} convention)",
                 category=T, example="fourier(exp(-x^2), x, k)")
    api.function("ifourier", _ifourier, signature="ifourier(F, k, x)", doc="inverse Fourier transform", category=T)
    api.function("ztransform", _ztransform, signature="ztransform(f, n, z)", doc="Z transform (sum of f z^-n)",
                 category=T)
    R = "Sequences"
    api.function("rsolve", _rsolve, signature="rsolve(eq, a, n, a0, a1, ...)",
                 doc="closed form of a recurrence written with a[n], a[n+1]; initial values optional",
                 category=R, example="rsolve(a[n+1] == 2 a[n], a, n, 1)")
