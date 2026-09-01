"""Ordinary differential equations, symbolic and numeric.

Symbolic:   dsolve(d(y, x, 2) + y == 0, y, x)
            dsolve(d(y, x) == -k y, y, x, 0, y0)        with y(0) = y0
Numeric:    sol = odesolve(-k y, y, x, 0, 1, 10)        dy/dx = -k y, y(0) = 1, up to x = 10
            sol(2.5)                                     then use it like a function, or plot it
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.utilities.lambdify import implemented_function

NAME = "ode"
DESCRIPTION = "Differential equations: dsolve (exact) and odesolve (numeric)."

_counter = [0]


def d(f, x, n=1):
    """Derivative marker for use inside dsolve equations."""
    if isinstance(f, sp.Symbol):
        f = sp.Function(f.name)(x)
    return sp.Derivative(f, (x, int(n)))


def dsolve(eq, y, x, x0=None, *initial):
    if not isinstance(y, sp.Symbol) or not isinstance(x, sp.Symbol):
        raise TypeError("Use dsolve(equation, y, x) with plain names for y and x.")
    yf = sp.Function(y.name)(x)
    eq = eq.subs(y, yf)
    ics = None
    if x0 is not None and initial:
        ics = {}
        for k, val in enumerate(initial):
            key = yf.subs(x, x0) if k == 0 else sp.Derivative(yf, (x, k)).subs(x, x0)
            ics[key] = val
    sol = sp.dsolve(eq, yf, ics=ics)
    return sol.rhs if isinstance(sol, sp.Eq) else sol


def odesolve(rhs, y, x, x0, y0, x1, name=None):
    """Numeric solution of dy/dx = rhs(x, y) from x0 to x1 with y(x0) = y0.

    ``y`` and ``rhs`` may be lists for a system. Returns a function of x
    (or a list of them) that interpolates the solution.
    """
    system = isinstance(y, (list, tuple))
    ys = list(y) if system else [y]
    rs = list(rhs) if system else [rhs]
    y0s = [float(v) for v in (y0 if system else [y0])]
    f = sp.lambdify([x, ys], rs, modules="numpy")
    res = solve_ivp(lambda t, v: np.asarray(f(t, v), dtype=float), (float(x0), float(x1)), y0s,
                    dense_output=True, rtol=1e-8, atol=1e-10)
    if not res.success:
        raise RuntimeError(res.message)
    _counter[0] += 1
    funcs = []
    for k, yk in enumerate(ys):
        label = f"{yk}_sol"

        def make(idx):
            def _interp(t):
                t = np.asarray(t, dtype=float)
                return res.sol(t)[idx]
            return _interp

        funcs.append(implemented_function(label, make(k)))
    return funcs if system else funcs[0]


def register(api):
    C = "Differential equations"
    api.function("d", d, signature="d(y, x, n)", doc="n-th derivative marker for dsolve equations", category=C,
                 example="d(y, x, 2) + y == 0")
    api.function("dsolve", dsolve, signature="dsolve(eq, y, x, x0, y0, dy0)",
                 doc="exact solution; initial values are optional", category=C,
                 example="dsolve(d(y, x) == -y, y, x, 0, 1)")
    api.function("odesolve", odesolve, signature="odesolve(rhs, y, x, x0, y0, x1)",
                 doc="numeric solution of dy/dx = rhs; result is a function of x", category=C,
                 example="odesolve(-0.5 y + sin(x), y, x, 0, 1, 20)")
