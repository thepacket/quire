"""Numerical methods, organised by how the approximation is constructed.

  Series / expansion      taylor, chebyshev_approx, pade, series_solve
  Iterative               newton_raphson, fixed_point, bisection, secant, jacobi_iter, gauss_seidel_iter (+ *_steps tables)
  Discretized             finite_difference, fdm_solve, bvp_solve, fem_solve, heat_fdm
  Time-stepping           euler, heun, rk4 (+ *_steps tables), alongside the adaptive odesolve

Results are worksheet values: numbers, polynomials, tables (matrices) or numeric
functions that can be called and plotted. Each result carries a note describing
the construction (steps, tolerance, grid) so the approximation is never mistaken
for an exact answer.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import implemented_function

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "numerics"
DESCRIPTION = "Numerical methods by construction: series, iterative, discretized, time-stepping."


# ---------------------------------------------------------------- helpers
def _note(text: str):
    hooks.context.setdefault("notes", []).append(text)


def _expr(f, *vars_):
    """Accept an expression in the given variables or a worksheet function of them."""
    if callable(f) and not isinstance(f, sp.Basic):
        f = f(*vars_)
    f = sp.sympify(f)
    if U.has_units(f):
        raise EvalError("Numerical methods work on plain numbers; convert units first (e.g. 'x -> m' then use the value).")
    return f


def _num(v, what="a number") -> float:
    v = sp.sympify(v)
    if U.has_units(v):
        raise EvalError(f"{what} must be a plain number, without units.")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise EvalError(f"{what} must be a number, got '{v}'.") from None


def _sym(x):
    if not isinstance(x, sp.Symbol):
        raise EvalError(f"Expected a variable name, got '{x}'.")
    return x


def _lam(expr, *vars_):
    return sp.lambdify(vars_, expr, modules=["numpy", "scipy"])


def _table(rows, columns: str):
    _note(f"columns: {columns}")
    return sp.ImmutableMatrix([[sp.Float(float(v), 12) for v in r] for r in rows])


class NumericFunction:
    """A function defined by data: callable with numbers, plottable with a symbol."""

    def __init__(self, label, fn, x0, x1, var, nargs=1):
        self.label, self.fn, self.x0, self.x1, self.var, self.nargs = label, fn, x0, x1, var, nargs
        self.func = implemented_function(label, fn)

    def __call__(self, *args):
        args = tuple(sp.sympify(a) for a in args)
        if len(args) != self.nargs:
            raise EvalError(f"{self.label} takes {self.nargs} argument(s).")
        if all(not a.free_symbols for a in args):
            val = float(np.asarray(self.fn(*[float(a) for a in args]), dtype=float).ravel()[0])
            if not np.isfinite(val):
                raise EvalError(f"{self.label} is only defined on [{self.x0:g}, {self.x1:g}].")
            return sp.Float(val)
        return self.func(*args)

    def __repr__(self):
        return f"{self.label}({self.var}) on [{self.x0:g}, {self.x1:g}]"


def _interp_function(label, xs, ys, var):
    """Cubic-spline interpolation through the computed points (nan outside the interval)."""
    from scipy.interpolate import CubicSpline

    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) >= 4:
        spline = CubicSpline(xs, ys, extrapolate=False)
        fn = lambda t: spline(np.asarray(t, float))  # noqa: E731
    else:
        fn = lambda t: np.interp(np.asarray(t, float), xs, ys, left=np.nan, right=np.nan)  # noqa: E731
    return NumericFunction(label, fn, xs[0], xs[-1], var)


# ================================================================ series / expansion
def taylor(f, x, x0=0, n=6):
    x = _sym(x)
    f = _expr(f, x)
    res = sp.series(f, x, x0, int(n)).removeO()
    _note(f"Taylor polynomial about {x} = {x0}, terms up to order {int(n) - 1}; the remainder is dropped")
    return res


def chebyshev_approx(f, x, a, b, n=8):
    """Chebyshev interpolation of degree n on [a, b], returned as a polynomial in x."""
    from numpy.polynomial import chebyshev as C

    x = _sym(x)
    fn = _lam(_expr(f, x), x)
    a, b, n = _num(a), _num(b), int(n)
    nodes = np.cos(np.pi * (np.arange(n + 1) + 0.5) / (n + 1))  # Chebyshev points on [-1, 1]
    xs = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    coeffs = C.chebfit(nodes, [float(fn(v)) for v in xs], n)
    t = (2 * x - (a + b)) / (b - a)
    poly = sp.expand(sum(sp.Float(c, 12) * sp.chebyshevt(k, t) for k, c in enumerate(coeffs)))
    _note(f"Chebyshev interpolant of degree {n} on [{a:g}, {b:g}] ({n + 1} nodes); near-minimax error on that interval")
    return poly


def pade(f, x, x0=0, m=2, n=2):
    """Padé approximant [m/n] of f about x0."""
    import mpmath as mp

    x = _sym(x)
    f = _expr(f, x)
    m, n = int(m), int(n)
    ser = sp.series(f, x, x0, m + n + 1).removeO()
    t = sp.Symbol("t")
    poly = sp.Poly(sp.expand(ser.subs(x, x0 + t)), t)
    coeffs = [poly.coeff_monomial(t ** k) for k in range(m + n + 1)]
    p, q = mp.pade([mp.mpf(str(sp.N(c, 30))) for c in coeffs], m, n)
    num = sum(sp.nsimplify(sp.Float(str(c), 15), rational=True, tolerance=1e-12) * (x - x0) ** k for k, c in enumerate(p))
    den = sum(sp.nsimplify(sp.Float(str(c), 15), rational=True, tolerance=1e-12) * (x - x0) ** k for k, c in enumerate(q))
    _note(f"Padé [{m}/{n}] approximant: rational function matching the Taylor series through order {m + n}")
    return num / den


def series_solve(eq, y, x, x0, *initial, n=6):
    """Power-series solution of an ODE about x0 with y(x0) = y0 (and y'(x0) = y1 for second order)."""
    y, x = _sym(y), _sym(x)
    n = int(n)
    yf = sp.Function(y.name)(x)
    expr = (eq.lhs - eq.rhs) if isinstance(eq, sp.Eq) else sp.sympify(eq)
    expr = expr.subs(y, yf)
    ics = {}
    for k, v in enumerate(initial):
        ics[yf.subs(x, x0) if k == 0 else sp.Derivative(yf, (x, k)).subs(x, x0)] = v
    order = max([d.derivative_count for d in expr.atoms(sp.Derivative)] or [1])
    hint = "1st_power_series" if order == 1 else "2nd_power_series_ordinary"
    try:
        sol = sp.dsolve(expr, yf, hint=hint, ics=ics or None, n=n, x0=x0)
    except Exception as exc:
        raise EvalError(f"No power-series solution: {str(exc).splitlines()[0]}") from None
    res = sol.rhs if isinstance(sol, sp.Eq) else sol
    res = res.removeO() if res.has(sp.Order) else res
    _note(f"power series about {x} = {x0}, truncated at order {n}")
    return res


# ================================================================ iterative
def _iterate(step, x0, tol, maxiter, name):
    xs = [float(x0)]
    for k in range(int(maxiter)):
        try:
            xn = step(xs[-1])
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            raise EvalError(f"{name} broke down at iteration {k + 1} (division by zero or overflow).") from None
        xs.append(float(xn))
        if not np.isfinite(xs[-1]):
            raise EvalError(f"{name} diverged at iteration {k + 1}.")
        if abs(xs[-1] - xs[-2]) <= tol * max(1.0, abs(xs[-1])):
            return xs
    raise EvalError(f"{name} did not converge in {int(maxiter)} iterations (last value {xs[-1]:.6g}).")


def _newton_steps(f, x, x0, tol, maxiter):
    x = _sym(x)
    f = _expr(f, x)
    fn, dfn = _lam(f, x), _lam(sp.diff(f, x), x)
    xs = _iterate(lambda v: v - fn(v) / dfn(v), _num(x0), _num(tol), maxiter, "Newton-Raphson")
    return xs, fn


def newton_raphson(f, x, x0, tol=1e-10, maxiter=50):
    xs, _ = _newton_steps(f, x, x0, tol, maxiter)
    _note(f"Newton-Raphson: {len(xs) - 1} iterations from {float(x0):g} to tolerance {float(tol):g}")
    return sp.Float(xs[-1])


def newton_raphson_steps(f, x, x0, tol=1e-10, maxiter=50):
    xs, fn = _newton_steps(f, x, x0, tol, maxiter)
    return _table([[k, v, fn(v)] for k, v in enumerate(xs)], "k, x_k, f(x_k)")


def _fixed_point_steps(g, x, x0, tol, maxiter):
    x = _sym(x)
    gn = _lam(_expr(g, x), x)
    return _iterate(lambda v: gn(v), _num(x0), _num(tol), maxiter, "Fixed-point iteration")


def fixed_point(g, x, x0, tol=1e-10, maxiter=200):
    xs = _fixed_point_steps(g, x, x0, tol, maxiter)
    _note(f"fixed-point iteration x = g(x): {len(xs) - 1} iterations to tolerance {float(tol):g}")
    return sp.Float(xs[-1])


def fixed_point_steps(g, x, x0, tol=1e-10, maxiter=200):
    xs = _fixed_point_steps(g, x, x0, tol, maxiter)
    return _table([[k, v] for k, v in enumerate(xs)], "k, x_k")


def bisection(f, x, a, b, tol=1e-10):
    x = _sym(x)
    fn = _lam(_expr(f, x), x)
    a, b = _num(a), _num(b)
    fa, fb = fn(a), fn(b)
    if fa * fb > 0:
        raise EvalError("Bisection needs a sign change: f(a) and f(b) have the same sign.")
    k = 0
    while (b - a) / 2 > float(tol) and k < 200:
        m = (a + b) / 2
        fm = fn(m)
        if fm == 0:
            a = b = m
            break
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        k += 1
    _note(f"bisection: {k} halvings, bracket width {b - a:.3g}")
    return sp.Float((a + b) / 2)


def secant(f, x, x0, x1, tol=1e-10, maxiter=50):
    x = _sym(x)
    fn = _lam(_expr(f, x), x)
    xs = [_num(x0), _num(x1)]
    for k in range(int(maxiter)):
        f0, f1 = fn(xs[-2]), fn(xs[-1])
        if f1 == f0:
            raise EvalError("Secant method broke down (equal function values).")
        xs.append(xs[-1] - f1 * (xs[-1] - xs[-2]) / (f1 - f0))
        if abs(xs[-1] - xs[-2]) <= float(tol) * max(1.0, abs(xs[-1])):
            _note(f"secant method: {k + 1} iterations to tolerance {float(tol):g}")
            return sp.Float(xs[-1])
    raise EvalError(f"Secant method did not converge in {int(maxiter)} iterations.")


def _linear_iteration(A, b, x0, tol, maxiter, gauss_seidel_):
    A = np.array(sp.Matrix(A).tolist(), dtype=float)
    b = np.array([float(v) for v in (list(b) if not isinstance(b, sp.MatrixBase) else list(b))], dtype=float)
    n = len(b)
    if A.shape != (n, n):
        raise EvalError("A must be square and match the length of b.")
    if np.any(np.diag(A) == 0):
        raise EvalError("Zero on the diagonal: reorder the equations.")
    x = np.zeros(n) if x0 is None else np.array([float(v) for v in x0], dtype=float)
    steps = [x.copy()]
    for _ in range(int(maxiter)):
        xn = x.copy()
        for i in range(n):
            ref = xn if gauss_seidel_ else x
            s_ = A[i, :i] @ ref[:i] + A[i, i + 1:] @ x[i + 1:]
            xn[i] = (b[i] - s_) / A[i, i]
        steps.append(xn.copy())
        if not np.all(np.isfinite(xn)) or np.max(np.abs(xn)) > 1e100:
            raise EvalError("Iteration diverged; the matrix is probably not diagonally dominant.")
        if np.max(np.abs(xn - x)) <= float(tol) * max(1.0, np.max(np.abs(xn))):
            return steps
        x = xn
    raise EvalError(f"Did not converge in {int(maxiter)} iterations; the matrix may not be diagonally dominant.")


def jacobi_iter(A, b, x0=None, tol=1e-10, maxiter=500):
    steps = _linear_iteration(A, b, x0, tol, maxiter, False)
    _note(f"Jacobi iteration: {len(steps) - 1} sweeps to tolerance {float(tol):g}")
    return sp.ImmutableMatrix([sp.Float(v) for v in steps[-1]])


def gauss_seidel_iter(A, b, x0=None, tol=1e-10, maxiter=500):
    steps = _linear_iteration(A, b, x0, tol, maxiter, True)
    _note(f"Gauss-Seidel iteration: {len(steps) - 1} sweeps to tolerance {float(tol):g}")
    return sp.ImmutableMatrix([sp.Float(v) for v in steps[-1]])


def jacobi_iter_steps(A, b, x0=None, tol=1e-10, maxiter=500):
    steps = _linear_iteration(A, b, x0, tol, maxiter, False)
    return _table([[k, *s_] for k, s_ in enumerate(steps)], "k, x_1 ... x_n")


def gauss_seidel_iter_steps(A, b, x0=None, tol=1e-10, maxiter=500):
    steps = _linear_iteration(A, b, x0, tol, maxiter, True)
    return _table([[k, *s_] for k, s_ in enumerate(steps)], "k, x_1 ... x_n")


# ================================================================ discretized
def finite_difference(f, x, x0, h=sp.Rational(1, 1000), order=1):
    """Central difference approximation of the derivative; symbolic x0 shows the stencil itself."""
    x = _sym(x)
    f = _expr(f, x)
    h = sp.sympify(h)
    order = int(order)
    if order == 1:
        expr = (f.subs(x, x0 + h) - f.subs(x, x0 - h)) / (2 * h)
    elif order == 2:
        expr = (f.subs(x, x0 + h) - 2 * f.subs(x, x0) + f.subs(x, x0 - h)) / h ** 2
    else:
        raise EvalError("finite_difference supports order 1 or 2.")
    _note(f"central difference stencil, order {order}, step h = {h}; truncation error O(h^2)")
    return sp.Float(float(expr)) if not expr.free_symbols else expr


def _split_ode(eq, y, x):
    """Return (rhs for y'', p, q, r) if the ODE is y'' = p(x) y' + q(x) y + r(x), else the general rhs."""
    yf = sp.Function(y.name)(x)
    expr = (eq.lhs - eq.rhs) if isinstance(eq, sp.Eq) else sp.sympify(eq)
    expr = expr.subs(y, yf)
    d2, d1 = sp.Derivative(yf, (x, 2)), sp.Derivative(yf, x)
    Y2, Y1, Y0 = sp.symbols("Y2 Y1 Y0")
    e = expr.subs(d2, Y2).subs(d1, Y1).subs(yf, Y0)
    if e.has(sp.Derivative):
        raise EvalError("Write the equation with D(y, x, 2), D(y, x) and y only.")
    sols = sp.solve(e, Y2)
    if len(sols) != 1:
        raise EvalError("Could not solve the equation for the second derivative.")
    rhs = sols[0]
    try:
        linear = sp.Poly(rhs, Y1, Y0).total_degree() <= 1 if rhs.has(Y1, Y0) else True
    except sp.PolynomialError:
        linear = False
    return rhs, Y1, Y0, linear


def fdm_solve(eq, y, x, a, ya, b, yb, n=50):
    """Linear two-point boundary value problem by central finite differences on n interior points."""
    y, x = _sym(y), _sym(x)
    rhs, Y1, Y0, linear = _split_ode(eq, y, x)
    if not linear:
        raise EvalError("fdm_solve handles linear equations; use bvp_solve for nonlinear ones.")
    p, q, r = rhs.coeff(Y1), rhs.coeff(Y0), rhs.subs({Y1: 0, Y0: 0})
    pf, qf, rf = (_lam(sp.sympify(e), x) for e in (p, q, r))
    a, b, ya, yb, n = _num(a), _num(b), _num(ya), _num(yb), int(n)
    h = (b - a) / (n + 1)
    xs = a + h * np.arange(1, n + 1)
    M = np.zeros((n, n))
    rhsv = np.zeros(n)
    for i, xi in enumerate(xs):
        pi_, qi, ri = float(pf(xi)), float(qf(xi)), float(rf(xi))
        # (y[i+1] - 2 y[i] + y[i-1])/h^2 = p (y[i+1] - y[i-1])/(2h) + q y[i] + r
        M[i, i] = -2 / h ** 2 - qi
        rhsv[i] = ri
        if i > 0:
            M[i, i - 1] = 1 / h ** 2 + pi_ / (2 * h)
        else:
            rhsv[i] -= (1 / h ** 2 + pi_ / (2 * h)) * ya
        if i < n - 1:
            M[i, i + 1] = 1 / h ** 2 - pi_ / (2 * h)
        else:
            rhsv[i] -= (1 / h ** 2 - pi_ / (2 * h)) * yb
    sol = np.linalg.solve(M, rhsv)
    grid = np.concatenate([[a], xs, [b]])
    vals = np.concatenate([[ya], sol, [yb]])
    _note(f"finite difference method: {n} interior points, h = {h:.4g}, second-order central stencils, one linear solve")
    return _interp_function(f"{y.name}_fdm", grid, vals, x.name)


def bvp_solve(eq, y, x, a, ya, b, yb, n=100):
    """General second-order boundary value problem (scipy solve_bvp, collocation on a mesh)."""
    from scipy.integrate import solve_bvp

    y, x = _sym(y), _sym(x)
    rhs, Y1, Y0, _ = _split_ode(eq, y, x)
    fn = sp.lambdify((x, Y0, Y1), rhs, modules="numpy")
    a, b, ya, yb, n = _num(a), _num(b), _num(ya), _num(yb), int(n)
    mesh = np.linspace(a, b, n)
    guess = np.vstack([ya + (yb - ya) * (mesh - a) / (b - a), np.full(n, (yb - ya) / (b - a))])
    res = solve_bvp(lambda t, Y: np.vstack([Y[1], fn(t, Y[0], Y[1])]),
                    lambda Ya, Yb: np.array([Ya[0] - ya, Yb[0] - yb]), mesh, guess, tol=1e-8, max_nodes=20000)
    if not res.success:
        raise EvalError(f"bvp_solve failed: {res.message}")
    _note(f"collocation boundary value solver on a mesh of {res.x.size} points (scipy solve_bvp)")
    return NumericFunction(f"{y.name}_bvp", lambda t: res.sol(np.asarray(t, float))[0], a, b, x.name)


def fem_solve(f, x, a, b, ua, ub, n=20):
    """-u'' = f(x) on [a, b] with u(a) = ua, u(b) = ub, by the finite element method (linear elements)."""
    x = _sym(x)
    fn = _lam(_expr(f, x), x)
    a, b, ua, ub, n = _num(a), _num(b), _num(ua), _num(ub), int(n)
    nodes = np.linspace(a, b, n + 1)
    h = nodes[1] - nodes[0]
    K = np.zeros((n + 1, n + 1))
    F = np.zeros(n + 1)
    for e in range(n):  # assemble element by element
        K[e:e + 2, e:e + 2] += np.array([[1, -1], [-1, 1]]) / h
        # load: 2-point Gauss quadrature of f * hat functions on the element
        for g, w in ((-1 / np.sqrt(3), 1.0), (1 / np.sqrt(3), 1.0)):
            xg = nodes[e] + (g + 1) / 2 * h
            fv = float(fn(xg))
            F[e] += w * fv * (1 - g) / 2 * h / 2
            F[e + 1] += w * fv * (1 + g) / 2 * h / 2
    # Dirichlet conditions
    F -= K[:, 0] * ua + K[:, -1] * ub
    Ki, Fi = K[1:-1, 1:-1], F[1:-1]
    u = np.concatenate([[ua], np.linalg.solve(Ki, Fi), [ub]])
    _note(f"finite element method: {n} linear elements, h = {h:.4g}, stiffness matrix assembled and solved")
    return _interp_function("u_fem", nodes, u, x.name)


def heat_fdm(u0, x, alpha, L, T, nx=40, nt=400):
    """Heat equation u_t = alpha u_xx on [0, L] with u = 0 at both ends, explicit FTCS scheme.

    Returns u(x, t): call it with numbers, or plot u(x, 0.5).
    """
    from scipy.interpolate import RegularGridInterpolator

    x = _sym(x)
    u0f = _lam(_expr(u0, x), x)
    alpha, L, T, nx, nt = _num(alpha), _num(L), _num(T), int(nx), int(nt)
    dx, dt = L / nx, T / nt
    r = alpha * dt / dx ** 2
    if r > 0.5:
        raise EvalError(f"FTCS is unstable for alpha dt/dx^2 = {r:.3g} > 0.5; use more time steps (nt >= {int(2 * alpha * T * nx ** 2 / L ** 2) + 1}).")
    xs = np.linspace(0, L, nx + 1)
    u = np.array([float(u0f(v)) for v in xs])
    u[0] = u[-1] = 0.0
    hist = [u.copy()]
    for _ in range(nt):
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        hist.append(u.copy())
    ts = np.linspace(0, T, nt + 1)
    interp = RegularGridInterpolator((xs, ts), np.array(hist).T, bounds_error=False, fill_value=None)

    def fn(xv, tv):
        xv, tv = np.asarray(xv, float), np.asarray(tv, float)
        pts = np.stack(np.broadcast_arrays(xv, tv), axis=-1)
        return interp(pts)

    _note(f"explicit finite differences (FTCS): {nx} space cells, {nt} time steps, alpha dt/dx^2 = {r:.3g}")
    nf = NumericFunction("u_heat", fn, 0.0, L, f"{x.name}, t", nargs=2)
    return nf


# ================================================================ time-stepping
def _steps_solver(kind, rhs, y, x, x0, y0, x1, h):
    x = _sym(x)
    system = isinstance(y, (list, tuple))
    ys = [_sym(v) for v in (y if system else [y])]
    rs = [_expr(r, x, *ys) for r in (rhs if system else [rhs])]
    fn = sp.lambdify([x, ys], rs, modules="numpy")
    F = lambda t, v: np.asarray(fn(t, v), dtype=float)  # noqa: E731
    x0, x1, h = _num(x0), _num(x1), _num(h)
    if h <= 0 or x1 <= x0:
        raise EvalError("Need x1 > x0 and a positive step h.")
    v = np.array([float(a) for a in (y0 if system else [y0])], dtype=float)
    t = x0
    ts, vs = [t], [v.copy()]
    nsteps = int(round((x1 - x0) / h))
    for _ in range(nsteps):
        if kind == "euler":
            v = v + h * F(t, v)
        elif kind == "heun":
            k1 = F(t, v)
            k2 = F(t + h, v + h * k1)
            v = v + h / 2 * (k1 + k2)
        else:
            k1 = F(t, v)
            k2 = F(t + h / 2, v + h / 2 * k1)
            k3 = F(t + h / 2, v + h / 2 * k2)
            k4 = F(t + h, v + h * k3)
            v = v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        ts.append(t)
        vs.append(v.copy())
        if not np.all(np.isfinite(v)):
            raise EvalError(f"{kind} blew up at x = {t:g}; try a smaller step.")
    return np.array(ts), np.array(vs), ys, system, x


_LABEL = {"euler": "Euler's method (first order)", "heun": "Heun's method (second order)", "rk4": "classical Runge-Kutta RK4 (fourth order)"}


def _stepper(kind):
    def solve(rhs, y, x, x0, y0, x1, h):
        ts, vs, ys, system, xsym = _steps_solver(kind, rhs, y, x, x0, y0, x1, h)
        _note(f"{_LABEL[kind]}: {len(ts) - 1} steps of h = {float(h):g} from {float(x0):g} to {float(x1):g}")
        funcs = [_interp_function(f"{yk.name}_{kind}", ts, vs[:, i], xsym.name) for i, yk in enumerate(ys)]
        return funcs if system else funcs[0]

    def steps(rhs, y, x, x0, y0, x1, h):
        ts, vs, ys, system, _ = _steps_solver(kind, rhs, y, x, x0, y0, x1, h)
        rows = [[t, *row] for t, row in zip(ts, vs)]
        if len(rows) > 500:
            raise EvalError("More than 500 steps; use the function form and plot it instead.")
        return _table(rows, f"{x}, " + ", ".join(v.name for v in ys))

    return solve, steps


euler, euler_steps = _stepper("euler")
heun, heun_steps = _stepper("heun")
rk4, rk4_steps = _stepper("rk4")


# ================================================================ registration
def register(api):
    S = "Series & expansions"
    api.function("taylor", taylor, signature="taylor(f, x, x0, n)", doc="Taylor polynomial (terms below order n)",
                 category=S, example="taylor(exp(x), x, 0, 5)")
    api.function("chebyshev_approx", chebyshev_approx, signature="chebyshev_approx(f, x, a, b, n)",
                 doc="Chebyshev polynomial approximation of degree n on [a, b]", category=S,
                 example="chebyshev_approx(exp(x), x, -1, 1, 4)")
    api.function("pade", pade, signature="pade(f, x, x0, m, n)", doc="Padé [m/n] rational approximant", category=S,
                 example="pade(exp(x), x, 0, 2, 2)")
    api.function("series_solve", series_solve, signature="series_solve(eq, y, x, x0, y0, y1, n)",
                 doc="power-series solution of an ODE about x0", category=S,
                 example="series_solve(D(y, x) == x + y, y, x, 0, 1)")

    I = "Iterative methods"
    api.function("newton_raphson", newton_raphson, signature="newton_raphson(f, x, x0, tol, maxiter)", doc="Newton-Raphson root from a guess",
                 category=I, example="newton_raphson(x^3 - 2 x - 5, x, 2)")
    api.function("newton_raphson_steps", newton_raphson_steps, signature="newton_raphson_steps(f, x, x0, tol)",
                 doc="table of Newton iterates: k, x_k, f(x_k)", category=I, example="newton_raphson_steps(x^2 - 2, x, 1)")
    api.function("fixed_point", fixed_point, signature="fixed_point(g, x, x0, tol)", doc="iterate x = g(x)",
                 category=I, example="fixed_point(cos(x), x, 1)")
    api.function("fixed_point_steps", fixed_point_steps, signature="fixed_point_steps(g, x, x0, tol)",
                 doc="table of fixed-point iterates", category=I)
    api.function("bisection", bisection, signature="bisection(f, x, a, b, tol)", doc="bisection on a sign change",
                 category=I, example="bisection(x^2 - 2, x, 1, 2)")
    api.function("secant", secant, signature="secant(f, x, x0, x1, tol)", doc="secant method", category=I)
    api.function("jacobi_iter", jacobi_iter, signature="jacobi_iter(A, b, x0, tol)", doc="Jacobi iteration for A x = b", category=I,
                 example="jacobi_iter(matrix([[4, 1], [2, 5]]), [1, 2])")
    api.function("gauss_seidel_iter", gauss_seidel_iter, signature="gauss_seidel_iter(A, b, x0, tol)",
                 doc="Gauss-Seidel iteration for A x = b", category=I)
    api.function("jacobi_iter_steps", jacobi_iter_steps, signature="jacobi_iter_steps(A, b, x0, tol)",
                 doc="table of Jacobi iterates", category=I)
    api.function("gauss_seidel_iter_steps", gauss_seidel_iter_steps, signature="gauss_seidel_iter_steps(A, b, x0, tol)",
                 doc="table of Gauss-Seidel iterates", category=I)

    D = "Discretized methods"
    api.function("finite_difference", finite_difference, signature="finite_difference(f, x, x0, h, order)",
                 doc="central difference derivative; symbolic x0 shows the stencil", category=D,
                 example="finite_difference(sin(x), x, 1, 0.01)")
    api.function("fdm_solve", fdm_solve, signature="fdm_solve(eq, y, x, a, ya, b, yb, n)",
                 doc="linear boundary value problem by finite differences (n interior points)", category=D,
                 example="fdm_solve(D(y, x, 2) == -y, y, x, 0, 0, pi/2, 1, 40)")
    api.function("bvp_solve", bvp_solve, signature="bvp_solve(eq, y, x, a, ya, b, yb, n)",
                 doc="general boundary value problem by collocation on a mesh", category=D,
                 example="bvp_solve(D(y, x, 2) == -exp(y), y, x, 0, 0, 1, 0)")
    api.function("fem_solve", fem_solve, signature="fem_solve(f, x, a, b, ua, ub, n)",
                 doc="-u'' = f on [a, b] by finite elements (n linear elements)", category=D,
                 example="fem_solve(1, x, 0, 1, 0, 0, 10)")
    api.function("heat_fdm", heat_fdm, signature="heat_fdm(u0, x, alpha, L, T, nx, nt)",
                 doc="heat equation on [0, L], explicit finite differences; gives u(x, t)", category=D,
                 example="heat_fdm(sin(pi x), x, 1, 1, 0.1, 40, 400)")

    T = "Time-stepping methods"
    api.function("euler", euler, signature="euler(rhs, y, x, x0, y0, x1, h)",
                 doc="Euler's method for dy/dx = rhs with fixed step h; gives a function", category=T,
                 example="euler(-2 y, y, x, 0, 1, 2, 0.1)")
    api.function("euler_steps", euler_steps, signature="euler_steps(rhs, y, x, x0, y0, x1, h)",
                 doc="table of Euler steps", category=T)
    api.function("heun", heun, signature="heun(rhs, y, x, x0, y0, x1, h)", doc="Heun's method (RK2)", category=T)
    api.function("heun_steps", heun_steps, signature="heun_steps(rhs, y, x, x0, y0, x1, h)",
                 doc="table of Heun steps", category=T)
    api.function("rk4", rk4, signature="rk4(rhs, y, x, x0, y0, x1, h)", doc="classical Runge-Kutta RK4", category=T,
                 example="rk4(-2 y, y, x, 0, 1, 2, 0.1)")
    api.function("rk4_steps", rk4_steps, signature="rk4_steps(rhs, y, x, x0, y0, x1, h)",
                 doc="table of RK4 steps", category=T)
