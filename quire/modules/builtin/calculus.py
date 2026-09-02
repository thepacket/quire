"""Derivatives, integrals, limits, series, sums, vector calculus, unevaluated forms."""
import sympy as sp

from ...engine import units as U
from ...engine.errors import EvalError
from .. import hooks
from ._util import as_list, matrix, prune_piecewise, sym, with_budget

SYMBOLIC_BUDGET = 8.0  # seconds for sympy's own attempt before backends and numerics get their turn


def _diff(f, *args):
    if not args:
        raise EvalError("diff needs a variable: diff(f, x).")
    sym(args[0])
    return sp.diff(f, *args)


def _ranges(args):
    """Group integration arguments: x | x, a, b | (x, a, b), repeated."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if isinstance(a, (tuple, list)):
            out.append(tuple(a))
            i += 1
        elif isinstance(a, sp.Symbol):
            rest = len(args) - i
            if rest >= 3 and rest % 3 == 0 and not isinstance(args[i + 1], sp.Symbol):
                out.append((a, args[i + 1], args[i + 2]))
                i += 3
            else:
                out.append(a)
                i += 1
        else:
            raise EvalError("Use integrate(f, x) or integrate(f, x, a, b).")
    return out


def _quad(f, ranges):
    """Numeric value of a definite integral via scipy (nested for multiple ranges)."""
    import numpy as np
    from scipy.integrate import quad

    if not ranges:
        return float(f)
    (x, a, b), rest = ranges[0], ranges[1:]
    if rest:
        inner = lambda v: _quad(f.subs(x, v), rest)  # noqa: E731
        val, _ = quad(inner, float(a), float(b))
        return val
    fn = sp.lambdify(x, f, modules=["numpy", "scipy"])
    lo = -np.inf if a == -sp.oo else float(a)
    hi = np.inf if b == sp.oo else float(b)

    def g(v):
        with np.errstate(all="ignore"):
            y = float(np.real(fn(v)))
        return y if np.isfinite(y) else 0.0  # overflow far in the tails of a convergent integrand

    val, _ = quad(g, lo, hi, limit=200)
    return val


def _only_equalities(cond) -> bool:
    """True when a condition is built only from equalities (a degenerate parameter case)."""
    if isinstance(cond, sp.Eq):
        return True
    if isinstance(cond, (sp.And, sp.Or)):
        return all(_only_equalities(c) for c in cond.args)
    return False


def generic_branch(res):
    """Drop degenerate-parameter branches like (..., Eq(a, 0)); keep the generic result."""
    if isinstance(res, sp.Piecewise) and res.args[-1].cond == sp.true \
            and all(_only_equalities(c) for _, c in res.args[:-1]):
        return res.args[-1].expr
    return res


def _backend_integral_ok(alt, f, ranges) -> bool:
    """Accept a backend antiderivative only if its derivative matches f numerically,
    and a definite result with numeric bounds only if it matches quadrature."""
    from .algebra import numerically_equal

    try:
        if all(isinstance(r, tuple) for r in ranges):
            if alt.free_symbols or any(e.free_symbols for r in ranges for e in r[1:]):
                return True  # symbolic bounds or parameters: nothing cheap to check against
            val = float(sp.N(alt, 15))
            ref = _quad(f, ranges)
            return abs(val - ref) <= 1e-6 * max(1.0, abs(ref))
        expr = alt
        for r in ranges:
            expr = sp.diff(expr, r if not isinstance(r, tuple) else r[0])
        return numerically_equal(expr, f)
    except Exception:  # noqa: BLE001
        return False


_UGLY = (sp.Integral, sp.meijerg, sp.exp_polar, sp.hyper, sp.expint, sp.floor)


def _ugliness(e) -> int:
    """Operation count plus a penalty per unevaluated or obscure construct."""
    return int(sp.count_ops(e)) + 20 * sum(len(e.atoms(k)) for k in _UGLY)


def _series_integrate(f, x, a, b):
    """Integrals over (0, oo) of g(x)/(exp(c x) -+ 1): expand as a geometric series, integrate termwise, sum.

    This is the Mellin-transform route to Gamma(s) zeta(s) and the Fermi-Dirac analogues.
    """
    if a != 0 or b != sp.oo:
        return None
    for factor in sp.Mul.make_args(f):
        if isinstance(factor, sp.Pow) and factor.exp == -1 and isinstance(factor.base, sp.Add) \
                and len(factor.base.args) == 2:
            terms = factor.base.args
            consts = [t for t in terms if t in (1, -1)]
            exps = [t for t in terms if isinstance(t, sp.exp)]
            if len(consts) != 1 or len(exps) != 1:
                continue
            sign = consts[0]  # 1/(e^cx + 1) or 1/(e^cx - 1)
            c = sp.simplify(exps[0].args[0] / x)
            if c.has(x) or c.is_positive is False:
                continue
            k = sp.Dummy("k", integer=True, positive=True)
            g = f / factor
            term = g * sp.exp(-k * c * x) * ((-1) ** (k + 1) if sign == 1 else 1)
            piece = sp.integrate(term, (x, 0, sp.oo))
            if piece.has(sp.Integral):
                return None
            piece = sp.powsimp(sp.expand_power_base(piece, force=True), force=True)
            total = prune_piecewise(sp.summation(piece, (k, 1, sp.oo)), hooks.context.get("bounds", {}))
            if total.has(sp.Sum) or isinstance(total, sp.Piecewise):
                total = _dirichlet_sum(piece, k, alternating=(sign == 1))
                if total is None:
                    return None
            return sp.simplify(total)
    return None


def _exceeds(p, threshold) -> bool:
    """Is p > threshold, using numeric values or the worksheet bounds (assume s > 1)?"""
    from ._util import _bound_samples

    if p.is_number:
        return bool(p > threshold)
    bounds = hooks.context.get("bounds", {})
    if isinstance(p, sp.Symbol):
        pts = _bound_samples(p.name, bounds)
        return bool(pts) and all(v > threshold for v in pts)
    return False


def _dirichlet_sum(piece, k, alternating: bool):
    """sum c k^(-p) = c zeta(p) for p > 1; sum (-1)^(k+1) c k^(-p) = c (1 - 2^(1-p)) zeta(p) for p > 0."""
    c, rest = piece.as_independent(k)
    if alternating:
        rest = sp.powsimp(rest / (-1) ** (k + 1), force=True)
        if rest.has(-1):
            rest = sp.simplify(rest * (-1) ** (k + 1) / (-1) ** (k + 1))
    if isinstance(rest, sp.Pow) and rest.base == k:
        p = -rest.exp
    elif rest == 1 / k:
        p = sp.S.One
    else:
        return None
    if alternating:
        return c * (1 - 2 ** (1 - p)) * sp.zeta(p) if _exceeds(p, 0) else None
    return c * sp.zeta(p) if _exceeds(p, 1) else None


def _integrate(f, *args):
    ranges = _ranges(args)
    res, timed_out = with_budget(SYMBOLIC_BUDGET, sp.integrate, f, *ranges)
    if timed_out:
        res = sp.Integral(f, *ranges)
    res = generic_branch(prune_piecewise(res, hooks.context.get("bounds", {})))
    if res.has(sp.Integral) and len(ranges) == 1 and isinstance(ranges[0], tuple):
        alt = _series_integrate(f, *ranges[0])
        if alt is not None:
            return alt
    indefinite = any(not isinstance(r, tuple) for r in ranges)
    if res.has(sp.Integral) and res.has(sp.oo, sp.zoo, sp.nan):
        res = sp.Integral(f, *ranges)  # a divergent-looking mix of oo and an unevaluated part: treat as no answer
    ugly = res.has(sp.Integral, sp.meijerg, sp.exp_polar, sp.hyper) or (indefinite and res.has(sp.floor, sp.expint))
    if ugly and hooks.available("integrate"):
        # Every verified backend answer competes; the least ugly wins (sympy's own only if it has one).
        candidates = hooks.run_all("integrate", f, ranges, accept=lambda a: _backend_integral_ok(a, f, ranges))
        if candidates:
            best = min(candidates, key=_ugliness)
            if res.has(sp.Integral) or _ugliness(best) < _ugliness(res):
                return best
    if res.has(sp.Integral, sp.meijerg, sp.hyper) and not res.free_symbols \
            and all(isinstance(r, tuple) for r in ranges):
        # No usable closed form; a definite integral with numeric bounds still has a value.
        exact = _recognize_definite(f, ranges)
        if exact is not None:
            return exact
        try:
            val = res.evalf()
            if val.is_number and not val.has(sp.Integral, sp.meijerg, sp.hyper):
                return val
        except Exception:  # noqa: BLE001
            pass
        try:
            return sp.Float(_quad(f, ranges), 15)
        except Exception:  # noqa: BLE001
            pass
    return res


def _recognize_definite(f, ranges):
    """No closed form: evaluate to 50 digits with mpmath and try to name the number (PSLQ)."""
    import mpmath as mp

    from .recognize import DPS, identify

    if len(ranges) != 1 or f.free_symbols - {ranges[0][0]}:
        return None
    x, a, b = ranges[0]
    try:
        with mp.workdps(DPS):
            fn = sp.lambdify(x, f, modules="mpmath")
            lo = -mp.inf if a == -sp.oo else mp.mpf(str(sp.N(a, DPS)))
            hi = mp.inf if b == sp.oo else mp.mpf(str(sp.N(b, DPS)))
            val = mp.quad(fn, [lo, hi])
            if isinstance(val, mp.mpc):
                if abs(val.imag) > mp.mpf(10) ** (-(DPS - 10)):
                    return None
                val = val.real
        res = identify(val)
    except Exception:  # noqa: BLE001
        return None
    if res is not None:
        hooks.context.setdefault("notes", []).append(
            f"no closed form was found symbolically; this value was recognized numerically from {DPS} digits")
    return res


def _integral(f, *args):
    return sp.Integral(f, *_ranges(args))


def _nintegrate(f, x, a, b):
    num, _ = U.strip_units(f)
    a_num, _ = U.strip_units(a)
    b_num, _ = U.strip_units(b)
    try:
        val = sp.Integral(num, (sym(x), a_num, b_num)).evalf()
        if not val.is_number or val.has(sp.Integral):
            raise ValueError
    except Exception:  # noqa: BLE001 - mpmath quadrature failed; use scipy
        val = sp.Float(_quad(num, [(sym(x), a_num, b_num)]), 15)
    unit = sp.S.One
    if U.has_units(f) or U.has_units(b):
        _, fu = U.split_units(U.to_base(f)) if U.has_units(f) else (None, sp.S.One)
        _, xu = U.split_units(U.to_base(b)) if U.has_units(b) else (None, sp.S.One)
        unit = fu * xu
    return val * unit


def _limit(f, x, x0, direction=None):
    d = "+-"
    if direction is not None:
        d = "+" if sp.sympify(direction) > 0 else "-"
    elif x0 in (sp.oo, -sp.oo):
        d = "+" if x0 == -sp.oo else "-"
    res = sp.limit(f, sym(x), x0, d)
    if res.has(sp.Limit):
        alt = hooks.run("limit", f, x, x0, d)
        if alt is not None:
            return alt
    return res


def _series(f, x, x0=0, n=6):
    return sp.series(f, sym(x), x0, int(n)).removeO()


def _with_integer_bounds(op, f, k, a, b):
    """Symbolic bounds are taken as positive integers unless assumed otherwise; the index is an integer."""
    k = sym(k)
    rep = {}
    for bound in (a, b):
        for s_ in getattr(bound, "free_symbols", set()):
            if s_ is not k and s_.is_integer is None:
                rep[s_] = sp.Dummy(s_.name, integer=True, positive=True)
    if k.is_integer is None:
        rep[k] = sp.Dummy(k.name, integer=True)
    f2, a2, b2 = (sp.sympify(e).subs(rep) for e in (f, a, b))
    res = op(f2, (rep.get(k, k), a2, b2))
    if res.has(sp.Sum, sp.Product):
        alt = hooks.run("sum" if op is sp.summation else "product", f, k, a, b)
        if alt is not None:
            return alt
    finite = b2 not in (sp.oo, -sp.oo) and a2 not in (sp.oo, -sp.oo)
    if finite and isinstance(res, sp.Piecewise):
        # A finite sum has a closed form everywhere; drop sympy's "otherwise keep the Sum" branch.
        kept = [(e, c) for e, c in res.args if not e.has(sp.Sum, sp.Product)]
        if len(kept) == 1:
            res = kept[0][0]
        elif kept:
            res = sp.Piecewise(*kept)
    return res.subs({v: s_ for s_, v in rep.items()})


def _sum(f, k, a, b):
    return _with_integer_bounds(sp.summation, f, k, a, b)


def _product(f, k, a, b):
    return _with_integer_bounds(sp.product, f, k, a, b)


def _doit(expr):
    return sp.sympify(expr).doit()


def _gradient(f, vars_):
    return sp.ImmutableMatrix([sp.diff(f, v) for v in as_list(vars_)])


def _divergence(F, vars_):
    return sp.Add(*[sp.diff(fi, v) for fi, v in zip(as_list(F), as_list(vars_))])


def _curl(F, vars_):
    Fx, Fy, Fz = as_list(F)
    x, y, z = as_list(vars_)
    return sp.ImmutableMatrix([sp.diff(Fz, y) - sp.diff(Fy, z), sp.diff(Fx, z) - sp.diff(Fz, x),
                               sp.diff(Fy, x) - sp.diff(Fx, y)])


def _laplacian(f, vars_):
    return sp.Add(*[sp.diff(f, v, 2) for v in as_list(vars_)])


def _jacobian(F, vars_):
    return sp.ImmutableMatrix(sp.Matrix(as_list(F)).jacobian(as_list(vars_)))


def _hessian(f, vars_):
    return sp.ImmutableMatrix(sp.hessian(f, as_list(vars_)))


def _implicit_diff(eq, y, x, n=1):
    from ._util import as_residual
    return sp.idiff(as_residual(eq), sym(y), sym(x), int(n))


def _fourier_series(f, x, n=3):
    fs = sp.fourier_series(f, (sym(x), -sp.pi, sp.pi))
    return fs.truncate(int(n))


def dprime_(f, n, arg):
    """f'(arg): derivative of a worksheet function evaluated at arg."""
    if not callable(f):
        raise EvalError("The prime notation f'(x) needs a defined function f.")
    t = sp.Dummy("t")
    return sp.diff(f(t), t, int(n)).subs(t, arg)


def seq_(base, idx):
    """a[n]: an indexed sequence term."""
    return sp.IndexedBase(base)[idx]


def register(api):
    C = "Calculus"
    api.function("diff", _diff, signature="diff(f, x, n)", doc="derivative; diff(f, x, y) for mixed partials",
                 category=C, example="diff(sin(x) x, x)")
    api.function("implicit_diff", _implicit_diff, signature="implicit_diff(eq, y, x)",
                 doc="dy/dx from an implicit equation", category=C, example="implicit_diff(x^2 + y^2 == 1, y, x)")
    api.function("integrate", _integrate, signature="integrate(f, x, a, b)",
                 doc="integral; omit a, b for an antiderivative; repeat x, a, b for multiple integrals",
                 category=C, example="integrate(x^2, x, 0, 1)")
    api.function("nintegrate", _nintegrate, signature="nintegrate(f, x, a, b)", doc="numeric integral",
                 category=C, example="nintegrate(exp(-x^2), x, 0, 2)")
    api.function("limit", _limit, signature="limit(f, x, x0, dir)",
                 doc="limit; dir = 1 from the right, -1 from the left", category=C,
                 example="limit(sin(x)/x, x, 0)")
    api.function("series", _series, signature="series(f, x, x0, n)", doc="Taylor polynomial of order n",
                 category=C, example="series(cos(x), x, 0, 6)")
    api.function("fourier_series", _fourier_series, signature="fourier_series(f, x, n)",
                 doc="Fourier series on [-π, π], n terms", category=C, example="fourier_series(x, x, 3)")
    api.function("sum", _sum, signature="sum(f, k, a, b)", doc="sum of f for k from a to b", category=C,
                 example="sum(k^2, k, 1, n)")
    api.function("product", _product, signature="product(f, k, a, b)", doc="product of f for k from a to b",
                 category=C)
    api.function("integral", _integral, signature="integral(f, x, a, b)",
                 doc="an integral left unevaluated; use doit to evaluate", category=C,
                 example="integral(x^2, x, 0, 1)")
    api.function("derivative", lambda f, *a: sp.Derivative(f, *a), signature="derivative(f, x)",
                 doc="a derivative left unevaluated", category=C, example="derivative(sin(x), x)")
    api.function("doit", _doit, signature="doit(expr)", doc="evaluate an unevaluated form", category=C,
                 example="doit(integral(x^2, x, 0, 1))")
    api.function("dprime_", dprime_, hidden=True)
    api.function("seq_", seq_, hidden=True)

    V = "Vector calculus"
    api.function("gradient", _gradient, signature="gradient(f, [x, y, z])", doc="gradient vector", category=V,
                 example="gradient(x^2 + y^2, [x, y])")
    api.function("divergence", _divergence, signature="divergence([Fx, Fy, Fz], [x, y, z])", doc="divergence",
                 category=V, example="divergence([x^2, y^2], [x, y])")
    api.function("curl", _curl, signature="curl([Fx, Fy, Fz], [x, y, z])", doc="curl", category=V,
                 example="curl([-y, x, 0], [x, y, z])")
    api.function("laplacian", _laplacian, signature="laplacian(f, [x, y])", doc="Laplacian", category=V)
    api.function("jacobian", _jacobian, signature="jacobian([f, g], [x, y])", doc="Jacobian matrix", category=V)
    api.function("hessian", _hessian, signature="hessian(f, [x, y])", doc="Hessian matrix", category=V)
