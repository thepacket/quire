"""Optimization: symbolic critical points and Lagrange multipliers, numeric minimization, linear programming, curve fitting."""
import numpy as np
import sympy as sp
from scipy import optimize as opt

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "optimization"
DESCRIPTION = "Critical points, Lagrange multipliers, numeric minimization, linear programming, least-squares fits."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _syms(vars_):
    vs = list(vars_) if isinstance(vars_, (list, tuple)) else [vars_]
    for v in vs:
        if not isinstance(v, sp.Symbol):
            raise EvalError(f"'{v}' is not a variable name.")
    return vs


def _f(x):
    return sp.Float(float(x))


def critical_points(f, vars_):
    """Rows: [x*, ..., f(x*), kind] where kind is min, max, saddle or unknown (Hessian test)."""
    vs = _syms(vars_)
    f = sp.sympify(f)
    sols = sp.solve([sp.diff(f, v) for v in vs], vs, dict=True)
    H = sp.hessian(f, vs)
    rows = []
    for s_ in sols:
        Hs = H.subs(s_)
        try:
            eig = [sp.re(sp.N(e)) for e in Hs.eigenvals()]
            if all(e > 0 for e in eig):
                kind = "min"
            elif all(e < 0 for e in eig):
                kind = "max"
            elif all(e != 0 for e in eig):
                kind = "saddle"
            else:
                kind = "unknown"
        except Exception:  # noqa: BLE001
            kind = "unknown"
        rows.append([sp.simplify(s_.get(v, v)) for v in vs] + [sp.simplify(f.subs(s_)), sp.Symbol(kind)])
    _note("columns: " + ", ".join(str(v) for v in vs) + ", f, kind (second-derivative test)")
    return sp.ImmutableMatrix(rows) if rows else []


def lagrange(f, constraints, vars_):
    """Stationary points of f subject to g_i == 0: rows [x*, ..., lambda_i..., f]."""
    vs = _syms(vars_)
    f = sp.sympify(f)
    cons = [c.lhs - c.rhs if isinstance(c, sp.Eq) else sp.sympify(c) for c in (constraints if isinstance(constraints, (list, tuple)) else [constraints])]
    lams = sp.symbols(f"lambda_1:{len(cons) + 1}")
    L = f - sum(l * c for l, c in zip(lams, cons))
    eqs = [sp.diff(L, v) for v in vs] + cons
    sols = sp.solve(eqs, list(vs) + list(lams), dict=True)
    rows = [[sp.simplify(s_.get(v, v)) for v in vs] + [sp.simplify(s_.get(l, l)) for l in lams] + [sp.simplify(f.subs(s_))] for s_ in sols]
    _note("Lagrange conditions: grad f = sum lambda_i grad g_i, g_i = 0; columns: " + ", ".join(str(v) for v in vs)
          + ", " + ", ".join(str(l) for l in lams) + ", f")
    return sp.ImmutableMatrix(rows) if rows else []


def minimize(f, vars_, x0=None, constraints=None):
    """Numeric minimum: [x*, ..., f*]. Constraints are equations (== 0) or inequalities (<= 0, >= 0)."""
    vs = _syms(vars_)
    f = sp.sympify(f)
    fn = sp.lambdify(vs, f, modules="numpy")
    grad = sp.lambdify(vs, [sp.diff(f, v) for v in vs], modules="numpy")
    x0 = np.array([float(v) for v in (x0 if x0 is not None else [1.0] * len(vs))], dtype=float)
    cons = []
    for c in (constraints or []):
        if isinstance(c, sp.Eq):
            expr, kind = c.lhs - c.rhs, "eq"
        elif isinstance(c, (sp.LessThan, sp.StrictLessThan)):
            expr, kind = c.rhs - c.lhs, "ineq"
        elif isinstance(c, (sp.GreaterThan, sp.StrictGreaterThan)):
            expr, kind = c.lhs - c.rhs, "ineq"
        else:
            raise EvalError("Constraints must be written as g == 0, g <= 0 or g >= 0.")
        cf = sp.lambdify(vs, expr, modules="numpy")
        cons.append({"type": kind, "fun": (lambda cf: lambda x: float(cf(*x)))(cf)})
    kwargs = {"constraints": cons, "method": "SLSQP"} if cons else {"jac": lambda x: np.array(grad(*x), dtype=float), "method": "BFGS"}
    res = opt.minimize(lambda x: float(fn(*x)), x0, **kwargs)
    if not res.success:
        raise EvalError(f"minimize did not converge: {res.message}")
    _note(f"{'SLSQP with constraints' if cons else 'BFGS with the symbolic gradient'}, {res.nit} iterations from {x0.tolist()}")
    return [_f(v) for v in res.x] + [_f(res.fun)]


def maximize(f, vars_, x0=None, constraints=None):
    out = minimize(-sp.sympify(f), vars_, x0, constraints)
    return out[:-1] + [-out[-1]]


def golden_section(f, x, a, b, tol=1e-8):
    """Minimum of a one-variable function on [a, b]."""
    if not isinstance(x, sp.Symbol):
        raise EvalError("Give the variable, e.g. golden_section(f, x, 0, 2).")
    fn = sp.lambdify(x, f, modules="numpy")
    res = opt.minimize_scalar(lambda v: float(fn(v)), bounds=(float(a), float(b)), method="bounded", options={"xatol": float(tol)})
    _note("bounded scalar minimization (Brent) on the interval")
    return [_f(res.x), _f(res.fun)]


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """Minimize c·x subject to A_ub x <= b_ub, A_eq x = b_eq, x >= 0 (or given bounds): [x, ..., value]."""
    def arr(m):
        return None if m is None else np.array([[float(v) for v in row] for row in (m.tolist() if isinstance(m, sp.MatrixBase) else m)], dtype=float)

    def vec(v):
        return None if v is None else np.array([float(x) for x in v], dtype=float)

    res = opt.linprog(vec(c), A_ub=arr(A_ub), b_ub=vec(b_ub), A_eq=arr(A_eq), b_eq=vec(b_eq),
                      bounds=None if bounds is None else [(None if lo is None else float(lo), None if hi is None else float(hi)) for lo, hi in bounds],
                      method="highs")
    if not res.success:
        raise EvalError(f"linprog: {res.message}")
    _note("linear programme solved with HiGHS; variables are non-negative unless bounds are given")
    return [_f(v) for v in res.x] + [_f(res.fun)]


def curve_fit(model, params, x, xs, ys, p0=None):
    """Nonlinear least squares: fitted parameter values [p1, ...] for a model expression in x."""
    ps = _syms(params)
    if not isinstance(x, sp.Symbol):
        raise EvalError("Give the independent variable, e.g. curve_fit(a exp(-b x), [a, b], x, xs, ys).")
    fn = sp.lambdify([x] + ps, model, modules="numpy")
    jac = sp.lambdify([x] + ps, [sp.diff(model, p) for p in ps], modules="numpy")
    xv = np.array([float(v) for v in xs], dtype=float)
    yv = np.array([float(v) for v in ys], dtype=float)
    guess = [float(v) for v in (p0 if p0 is not None else [1.0] * len(ps))]

    def jacobian(xd, *p):
        cols = jac(xd, *p)
        return np.column_stack([np.broadcast_to(np.asarray(c, dtype=float), xd.shape) for c in cols])

    popt, pcov = opt.curve_fit(lambda xd, *p: np.asarray(fn(xd, *p), dtype=float), xv, yv, p0=guess, jac=jacobian, maxfev=20000)
    _note("Levenberg-Marquardt least squares with the symbolic Jacobian; standard errors: "
          + ", ".join(f"{p} ± {e:.3g}" for p, e in zip(ps, np.sqrt(np.diag(pcov)))))
    return [_f(v) for v in popt]


def gradient_descent(f, vars_, x0, rate=0.1, steps=20):
    """Table of gradient-descent iterates: k, x..., f."""
    vs = _syms(vars_)
    f = sp.sympify(f)
    fn = sp.lambdify(vs, f, modules="numpy")
    grad = sp.lambdify(vs, [sp.diff(f, v) for v in vs], modules="numpy")
    x = np.array([float(v) for v in x0], dtype=float)
    rows = [[0, *x, float(fn(*x))]]
    for k in range(1, int(steps) + 1):
        x = x - float(rate) * np.array(grad(*x), dtype=float)
        rows.append([k, *x, float(fn(*x))])
    _note("columns: k, " + ", ".join(str(v) for v in vs) + f", f; learning rate {float(rate):g}")
    return sp.ImmutableMatrix([[sp.Integer(r[0])] + [_f(v) for v in r[1:]] for r in rows])


def register(api):
    O = "Optimization"
    api.function("critical_points", critical_points, signature="critical_points(f, [x, y])", doc="stationary points classified by the Hessian",
                 category=O, example="critical_points(x^3 - 3 x + y^2, [x, y])")
    api.function("lagrange", lagrange, signature="lagrange(f, [g == 0, ...], [x, y])", doc="constrained stationary points", category=O,
                 example="lagrange(x y, [x + y == 10], [x, y])")
    api.function("minimize", minimize, signature="minimize(f, [x, y], [x0, y0], [constraints])", doc="numeric minimum [x*, f*]", category=O,
                 example="minimize((x - 1)^2 + (y - 2)^2, [x, y])")
    api.function("maximize", maximize, signature="maximize(f, [x, y], [x0, y0], [constraints])", doc="numeric maximum", category=O)
    api.function("golden_section", golden_section, signature="golden_section(f, x, a, b)", doc="one-variable bounded minimum", category=O)
    api.function("linprog", linprog, signature="linprog(c, A_ub, b_ub, A_eq, b_eq)", doc="linear programming (minimize c·x)", category=O,
                 example="linprog([-1, -2], [[1, 1], [1, 3]], [4, 6])")
    api.function("curve_fit", curve_fit, signature="curve_fit(model, [params], x, xs, ys, p0)", doc="nonlinear least squares", category=O,
                 example="curve_fit(a exp(-b x), [a, b], x, [0, 1, 2, 3], [2, 1.2, 0.7, 0.45])")
    api.function("gradient_descent", gradient_descent, signature="gradient_descent(f, [x, y], [x0, y0], rate, steps)", doc="iterate table",
                 category=O)
