"""Ordinary differential equations, symbolic and numeric.

Symbolic:   dsolve(D(y, x, 2) + y == 0, y, x)
            dsolve(D(y, x) == -k y, y, x, 0, y0)        with y(0) = y0
Numeric:    sol = odesolve(-k y, y, x, 0, 1, 10)        dy/dx = -k y, y(0) = 1, up to x = 10
            sol(2.5)                                     then use it like a function, or plot it
"""
import math

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.utilities.lambdify import implemented_function

NAME = "ode"
DESCRIPTION = "Differential equations: dsolve (exact) and odesolve (numeric)."



class NumericSolution:
    """A numeric ODE solution that behaves like a function of the independent variable."""

    def __init__(self, label, interp, x0, x1, var):
        self.label, self.interp, self.x0, self.x1, self.var = label, interp, x0, x1, var
        self.func = implemented_function(label, interp)

    def __call__(self, arg):
        arg = sp.sympify(arg)
        if not arg.free_symbols:
            return sp.Float(float(self.interp(float(arg))))
        return self.func(arg)

    def __repr__(self):
        return f"{self.label}: numeric solution for {self.var} in [{self.x0}, {self.x1}]"


def D(f, x, n=1):
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
    funcs = []
    for k, yk in enumerate(ys):
        def make(idx):
            def _interp(t):
                t = np.asarray(t, dtype=float)
                return res.sol(t)[idx]
            return _interp

        funcs.append(NumericSolution(f"{yk}_sol", make(k), x0, x1, x))
    return funcs if system else funcs[0]



def _phase_plot(cell, env, ev):
    """Plot kind: direction field, trajectories and equilibria of dx/dt = f(x, y), dy/dt = g(x, y)."""
    from quire.engine import plotting as P
    from quire.engine.errors import QuireError
    from quire.engine.parser import parse, split_top
    from quire.modules.builtin._util import with_budget

    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    parts = [p for p in split_top(src, ",") if p.strip()]
    if len(parts) != 2:
        raise QuireError("A phase portrait needs two expressions: dx/dt, dy/dt.")
    xv, yv = P.two_vars(cell)
    ns = ev.namespace(env, [xv, yv])
    xs, ys = ns[xv], ns[yv]
    ex, ey = P.resolve(parts[0], ns, ev, xs, ys), P.resolve(parts[1], ns, ev, xs, ys)
    fx, _ = P.numeric2(ex, xs, ys)
    fy, _ = P.numeric2(ey, xs, ys)
    a, b, c, d = P.plain_bounds(cell, ns, ev, ("-3", "3", "-3", "3"))
    n = P.samples(cell, 16, 40, 5)
    X, Y = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
    UU, VV = P.call2(fx, X, Y), P.call2(fy, X, Y)
    hx, hy = (b - a) / n * 0.4, (d - c) / n * 0.4
    segs, speeds = [], []
    for i in range(n):
        for j in range(n):
            u, v = UU[i, j], VV[i, j]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            speeds.append(math.hypot(u, v))
            ux, vy = u / (b - a), v / (d - c)
            norm = math.hypot(ux, vy)
            if norm == 0:
                continue
            segs.append([X[i, j] - ux / norm * hx, Y[i, j] - vy / norm * hy, X[i, j] + ux / norm * hx, Y[i, j] + vy / norm * hy])
    # trajectories from the given start points, or from a 3 x 3 grid, forward and backward in time
    seeds_src = (cell.get("expr2") or "").strip()
    if seeds_src:
        raw = parse(seeds_src, ns, ev.unit_names)
        if isinstance(raw, sp.MatrixBase):
            raw = raw.tolist()
        try:
            seeds = [(float(p[0]), float(p[1])) for p in raw]
        except (TypeError, IndexError):
            raise QuireError("Start points are a list of [x, y] pairs, e.g. [[1, 0], [0, 2]].") from None
    else:
        seeds = [(a + (b - a) * fx_, c + (d - c) * fy_) for fx_ in (0.25, 0.5, 0.75) for fy_ in (0.25, 0.5, 0.75)]
    typical = float(np.median(speeds)) if speeds else 0.0
    T = min(60.0, 4 * max(b - a, d - c) / typical) if typical > 0 else 10.0
    mx, my, hw, hh = (a + b) / 2, (c + d) / 2, (b - a) / 2, (d - c) / 2

    def rhs(t, sv):
        with np.errstate(all="ignore"):
            return [float(fx(sv[0], sv[1])), float(fy(sv[0], sv[1]))]

    def leave(t, sv):
        return max(abs(sv[0] - mx) / hw, abs(sv[1] - my) / hh) - 1.15
    leave.terminal = True

    tx, ty = [], []
    for x0, y0 in seeds:
        for direction in (1, -1):
            try:
                sol = solve_ivp(rhs, (0, direction * T), [x0, y0], t_eval=np.linspace(0, direction * T, 300),
                                events=leave, rtol=1e-6, atol=1e-9)
            except Exception:  # noqa: BLE001
                continue
            px, py = list(sol.y[0]), list(sol.y[1])
            if direction == -1:
                px, py = px[::-1], py[::-1]
            if len(px) < 2:
                continue
            tx.extend(px + [None])
            ty.extend(py + [None])
    series = [{"type": "segments", "arrows": True, "label": r"\text{direction field}", "label_plain": "direction field",
               "segments": segs}]
    if tx:
        series.append({"type": "line", "label": r"\text{trajectories}", "label_plain": "trajectories",
                       "x": [None if v is None else float(v) for v in tx], "y": [None if v is None else float(v) for v in ty]})
    # equilibria: solutions of f = g = 0 inside the window
    try:
        sols, timed_out = with_budget(2.0, sp.solve, [ex, ey], [xs, ys], dict=True)
    except Exception:  # noqa: BLE001
        sols, timed_out = [], False
    eq = []
    if not timed_out:
        for sol in sols:
            try:
                px, py = complex(sol.get(xs, 0)), complex(sol.get(ys, 0))
            except TypeError:
                continue
            if abs(px.imag) < 1e-9 and abs(py.imag) < 1e-9 and a <= px.real <= b and c <= py.real <= d:
                eq.append((px.real, py.real))
    if eq:
        series.append({"type": "points", "marker": "o", "label": r"\text{equilibria}", "label_plain": "equilibria",
                       "x": [p[0] for p in eq], "y": [p[1] for p in eq]})
    return {"series": series, "xlabel": xv, "ylabel": yv, "xrange": [a, b], "yrange": [c, d]}


def register(api):
    api.plot_kind("phase", _phase_plot, label="phase portrait", f1="dx/dt, dy/dt =", f2="start points",
                  var="variables", range="x", yrange=True, ph1="y, -sin(x) - 0.3 y", ph2="", samples="16",
                  doc="direction field, trajectories and equilibria of a planar system dx/dt = f, dy/dt = g")
    C = "Differential equations"
    api.function("D", D, signature="D(y, x, n)", doc="n-th derivative marker for dsolve equations", category=C,
                 example="D(y, x, 2) + y == 0")
    api.function("dsolve", dsolve, signature="dsolve(eq, y, x, x0, y0, dy0)",
                 doc="exact solution; initial values are optional", category=C,
                 example="dsolve(D(y, x) == -y, y, x, 0, 1)")
    api.function("odesolve", odesolve, signature="odesolve(rhs, y, x, x0, y0, x1)",
                 doc="numeric solution of dy/dx = rhs; result is a function of x", category=C,
                 example="odesolve(-0.5 y + sin(x), y, x, 0, 1, 20)")
