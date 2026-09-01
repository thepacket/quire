"""Derivatives, integrals, limits, series, sums, vector calculus, unevaluated forms."""
import sympy as sp

from ...engine import units as U
from ...engine.errors import EvalError
from ._util import as_list, matrix, sym


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


def _integrate(f, *args):
    return sp.integrate(f, *_ranges(args))


def _integral(f, *args):
    return sp.Integral(f, *_ranges(args))


def _nintegrate(f, x, a, b):
    num, _ = U.strip_units(f)
    a_num, _ = U.strip_units(a)
    b_num, _ = U.strip_units(b)
    val = sp.Integral(num, (sym(x), a_num, b_num)).evalf()
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
    return sp.limit(f, sym(x), x0, d)


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
