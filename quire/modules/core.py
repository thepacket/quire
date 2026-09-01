"""The built-in module: units, constants and the standard math functions.

Everything a worksheet can call by name is registered here (or in an external
module). The registry turns these entries into the parser namespace and the
reference panel in the UI.
"""
from __future__ import annotations

import sympy as sp

from ..engine import units as U
from ..engine.errors import EvalError

NAME = "core"
DESCRIPTION = "Units, constants, algebra, calculus, equations, matrices."


def _angle_aware(fn, name):
    def f(x):
        return fn(U.strip_angles(x))

    f.__name__ = name
    return f


def _sym(x):
    if not isinstance(x, sp.Symbol):
        raise EvalError(f"Expected a variable name, got '{x}'.")
    return x


def _integrate(f, *args):
    if len(args) == 1:
        return sp.integrate(f, _sym(args[0]))
    if len(args) == 3 and not isinstance(args[0], (tuple, list)):
        return sp.integrate(f, (_sym(args[0]), args[1], args[2]))
    if len(args) == 1 or all(isinstance(a, (tuple, list)) for a in args):
        return sp.integrate(f, *[tuple(a) for a in args])
    raise EvalError("Use integrate(f, x) or integrate(f, x, a, b).")


def _nintegrate(f, x, a, b):
    num, dim = U.strip_units(f)
    xs = _sym(x)
    a_num, _ = U.strip_units(a)
    b_num, _ = U.strip_units(b)
    val = sp.Integral(num, (xs, a_num, b_num)).evalf()
    unit = sp.S.One
    if U.has_units(f) or U.has_units(a):
        _, fu = U.split_units(U.to_base(f)) if U.has_units(f) else (None, sp.S.One)
        _, xu = U.split_units(U.to_base(b)) if U.has_units(b) else (None, sp.S.One)
        unit = fu * xu
    return val * unit


def _diff(f, x, n=1):
    return sp.diff(f, _sym(x), n)


def _limit(f, x, x0, direction="+"):
    return sp.limit(f, _sym(x), x0, str(direction))


def _series(f, x, x0=0, n=6):
    return sp.series(f, _sym(x), x0, int(n)).removeO()


def _sum(f, k, a, b):
    return sp.summation(f, (_sym(k), a, b))


def _product(f, k, a, b):
    return sp.product(f, (_sym(k), a, b))


_CANDIDATE_UNITS = None


def _candidate_units():
    global _CANDIDATE_UNITS
    if _CANDIDATE_UNITS is None:
        u = U.u
        _CANDIDATE_UNITS = [sp.S.One, u.m, u.s, u.kg, u.A, u.K, u.mol, u.cd, u.m / u.s, u.m / u.s**2, u.N, u.J,
                            u.W, u.Pa, u.V, u.ohm, u.C, u.F, u.henry, u.Hz, u.m**2, u.m**3, u.kg / u.m**3,
                            u.N * u.m, u.J / u.K, u.W / u.m**2, 1 / u.s, u.kg / u.s, u.m**3 / u.s, u.T, u.Wb]
    return _CANDIDATE_UNITS


def _as_residual(e):
    return e.lhs - e.rhs if isinstance(e, sp.Eq) else e


def _needs_stripping(exprs):
    """True when a transcendental function has units inside its argument (sympy's solve hangs on those)."""
    for e in exprs:
        for f in e.atoms(sp.Function):
            if any(U.has_units(a) for a in f.args):
                return True
    return False


def _infer_unit(residuals, sym, value):
    """Find a unit for sym such that every residual is dimensionally consistent."""
    for cand in _candidate_units():
        try:
            for r in residuals:
                bound = r.subs(sym, value * cand)
                U.check_dimensions(U.to_base(bound))
                U.to_base(bound)
        except Exception:  # noqa: BLE001
            continue
        return cand
    return None


def _solve(eqs, *syms):
    if len(syms) == 1 and isinstance(syms[0], (list, tuple)):
        syms = tuple(syms[0])
    eq_list = list(eqs) if isinstance(eqs, (list, tuple)) else [eqs]
    residuals = [_as_residual(sp.sympify(e)) for e in eq_list]
    floats = any(r.atoms(sp.Float) for r in residuals)
    strip = any(U.has_units(r) for r in residuals) and _needs_stripping(residuals)
    work = [U.strip_units(r)[0] for r in residuals] if strip else residuals
    try:
        res = sp.solve(work if len(work) > 1 else work[0], *syms, dict=False, rational=False if floats else None)
    except NotImplementedError:
        names = ", ".join(str(s) for s in syms)
        raise EvalError(f"No exact solution found for {names}. Try nsolve(equation, {names}, guess) "
                        f"for a numeric answer.") from None
    if floats and isinstance(res, list) and res:
        real = [r for r in res if getattr(r, "is_real", None) is not False and not (isinstance(r, sp.Expr) and r.has(sp.I))]
        if real:
            res = real
    if strip and len(syms) == 1 and isinstance(res, list) and res:
        unit = _infer_unit(residuals, syms[0], res[0])
        if unit is not None and unit != 1:
            res = [r * unit for r in res]
    return res


def _nsolve(eqs, syms, guess):
    if isinstance(eqs, (list, tuple)):
        eqs = [U.strip_units(e)[0] if U.has_units(e) else e for e in eqs]
    try:
        return sp.nsolve(eqs, syms, guess)
    except Exception as exc:
        raise EvalError(f"nsolve failed: {str(exc).splitlines()[0]}") from None


def _subs(expr, *pairs):
    if len(pairs) == 2:
        return expr.subs(pairs[0], pairs[1])
    if len(pairs) == 1 and isinstance(pairs[0], (list, tuple)):
        return expr.subs(list(pairs[0]))
    raise EvalError("Use subs(expr, x, value).")


def _N(expr, digits=15):
    return sp.N(expr, int(digits))


def _round(x, n=0):
    num, unit = U.split_units(x)
    return sp.Float(round(float(num), int(n))) * unit if n else sp.Integer(round(float(num))) * unit


def _matrix(rows):
    return sp.ImmutableMatrix(rows)


def _piecewise(*pieces):
    return sp.Piecewise(*[tuple(p) for p in pieces])


def _log(x, base=None):
    return sp.log(x) if base is None else sp.log(x, base)


def _atan2(y, x):
    return sp.atan2(U.strip_angles(y), U.strip_angles(x))


def _max(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    return sp.Max(*args)


def _min(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    return sp.Min(*args)


def register(api):
    # --- constants
    api.constant("pi", sp.pi, doc="π", category="Constants", example="2 pi")
    api.constant("e", sp.E, doc="Euler's number", category="Constants", example="e^x")
    api.constant("i", sp.I, doc="imaginary unit", category="Constants", example="3 + 4 i")
    api.constant("oo", sp.oo, doc="infinity", category="Constants", example="limit(1/x, x, oo)")
    api.constant("inf", sp.oo, doc="infinity", category="Constants")
    for name, (val, doc) in U.CONSTANT_TABLE.items():
        api.constant(name, val, doc=doc, category="Physical constants")

    # --- units
    for name, (q, doc) in U.UNIT_TABLE.items():
        api.unit(name, q, doc=doc, category="Units")

    # --- arithmetic
    A = "Arithmetic"
    api.function("abs", sp.Abs, signature="abs(x)", doc="absolute value", category=A)
    api.function("sqrt", sp.sqrt, signature="sqrt(x)", doc="square root", category=A, example="sqrt(2)")
    api.function("cbrt", sp.cbrt, signature="cbrt(x)", doc="cube root", category=A)
    api.function("root", sp.root, signature="root(x, n)", doc="n-th root", category=A)
    api.function("exp", sp.exp, signature="exp(x)", doc="e^x", category=A)
    api.function("ln", sp.log, signature="ln(x)", doc="natural logarithm", category=A)
    api.function("log", _log, signature="log(x, base)", doc="logarithm; base defaults to e", category=A,
                 example="log(100, 10)")
    api.function("log10", lambda x: sp.log(x, 10), signature="log10(x)", doc="base-10 logarithm", category=A)
    api.function("log2", lambda x: sp.log(x, 2), signature="log2(x)", doc="base-2 logarithm", category=A)
    api.function("floor", sp.floor, signature="floor(x)", doc="round down", category=A)
    api.function("ceil", sp.ceiling, signature="ceil(x)", doc="round up", category=A)
    api.function("round", _round, signature="round(x, n)", doc="round to n decimals", category=A)
    api.function("sign", sp.sign, signature="sign(x)", doc="-1, 0 or 1", category=A)
    api.function("max", _max, signature="max(a, b, ...)", doc="largest value", category=A)
    api.function("min", _min, signature="min(a, b, ...)", doc="smallest value", category=A)
    api.function("mod", sp.Mod, signature="mod(a, b)", doc="remainder of a / b", category=A)
    api.function("gcd", sp.gcd, signature="gcd(a, b)", doc="greatest common divisor", category=A)
    api.function("lcm", sp.lcm, signature="lcm(a, b)", doc="least common multiple", category=A)
    api.function("factorial", sp.factorial, signature="factorial(n)", doc="n! (also n!)", category=A)
    api.function("binomial", sp.binomial, signature="binomial(n, k)", doc="binomial coefficient", category=A)
    api.function("re", sp.re, signature="re(z)", doc="real part", category=A)
    api.function("im", sp.im, signature="im(z)", doc="imaginary part", category=A)
    api.function("conj", sp.conjugate, signature="conj(z)", doc="complex conjugate", category=A)
    api.function("arg", sp.arg, signature="arg(z)", doc="complex argument", category=A)

    # --- trigonometry (angle aware: sin(90 deg) = 1)
    T = "Trigonometry"
    for name in ["sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh"]:
        api.function(name, _angle_aware(getattr(sp, name), name), signature=f"{name}(x)",
                     doc=f"{name}; accepts deg or rad", category=T, example=f"{name}(30 deg)")
    for name in ["asin", "acos", "atan", "asinh", "acosh", "atanh"]:
        api.function(name, getattr(sp, name), signature=f"{name}(x)", doc=f"inverse {name[1:]} (radians)",
                     category=T, example=f"{name}(1/2) -> deg")
    api.function("atan2", _atan2, signature="atan2(y, x)", doc="two-argument arctangent", category=T)

    # --- algebra
    G = "Algebra"
    api.function("simplify", sp.simplify, signature="simplify(expr)", doc="simplify an expression", category=G)
    api.function("expand", sp.expand, signature="expand(expr)", doc="multiply out", category=G,
                 example="expand((x+1)^3)")
    api.function("factor", sp.factor, signature="factor(expr)", doc="factor a polynomial", category=G,
                 example="factor(x^2 - 1)")
    api.function("collect", sp.collect, signature="collect(expr, x)", doc="group by powers of x", category=G)
    api.function("cancel", sp.cancel, signature="cancel(expr)", doc="cancel common factors", category=G)
    api.function("apart", sp.apart, signature="apart(expr, x)", doc="partial fractions", category=G)
    api.function("together", sp.together, signature="together(expr)", doc="combine over a common denominator",
                 category=G)
    api.function("trigsimp", sp.trigsimp, signature="trigsimp(expr)", doc="simplify trig identities", category=G)
    api.function("subs", _subs, signature="subs(expr, x, value)", doc="substitute a value", category=G,
                 example="subs(x^2, x, 3)")
    api.function("nsimplify", sp.nsimplify, signature="nsimplify(x)", doc="find an exact form for a decimal",
                 category=G, example="nsimplify(0.5)")
    api.function("N", _N, signature="N(expr, digits)", doc="numeric value", category=G, example="N(pi, 20)")
    api.function("numer", sp.numer, signature="numer(expr)", doc="numerator", category=G)
    api.function("denom", sp.denom, signature="denom(expr)", doc="denominator", category=G)
    api.function("Eq", sp.Eq, signature="a == b", doc="an equation (write a == b)", category=G)
    api.function("piecewise", _piecewise, signature="piecewise((expr, cond), ...)",
                 doc="piecewise definition", category=G, example="piecewise((x, x < 1), (1, x >= 1))")

    # --- calculus
    C = "Calculus"
    api.function("diff", _diff, signature="diff(f, x, n)", doc="n-th derivative (n defaults to 1)", category=C,
                 example="diff(sin(x) x, x)")
    api.function("integrate", _integrate, signature="integrate(f, x, a, b)",
                 doc="integral; omit a, b for an antiderivative", category=C, example="integrate(x^2, x, 0, 1)")
    api.function("nintegrate", _nintegrate, signature="nintegrate(f, x, a, b)", doc="numeric integral",
                 category=C, example="nintegrate(exp(-x^2), x, 0, 2)")
    api.function("limit", _limit, signature="limit(f, x, x0)", doc="limit as x approaches x0", category=C,
                 example="limit(sin(x)/x, x, 0)")
    api.function("series", _series, signature="series(f, x, x0, n)", doc="Taylor polynomial of order n",
                 category=C, example="series(cos(x), x, 0, 6)")
    api.function("sum", _sum, signature="sum(f, k, a, b)", doc="sum of f for k from a to b", category=C,
                 example="sum(k^2, k, 1, n)")
    api.function("product", _product, signature="product(f, k, a, b)", doc="product of f for k from a to b",
                 category=C)

    # --- equations
    E = "Equations"
    api.function("solve", _solve, signature="solve(eq, x)", doc="solve exactly; eq is 'a == b' or a list",
                 category=E, example="solve(x^2 == 4, x)")
    api.function("nsolve", _nsolve, signature="nsolve(eq, x, guess)", doc="solve numerically from a guess",
                 category=E, example="nsolve(cos(x) == x, x, 1)")

    # --- matrices
    M = "Matrices"
    api.function("matrix", _matrix, signature="matrix([[a, b], [c, d]])", doc="build a matrix", category=M,
                 example="matrix([[1, 2], [3, 4]])")
    api.function("identity", sp.eye, signature="identity(n)", doc="n×n identity matrix", category=M)
    api.function("zeros", sp.zeros, signature="zeros(n, m)", doc="n×m zero matrix", category=M)
    api.function("det", lambda A: sp.Matrix(A).det(), signature="det(A)", doc="determinant", category=M)
    api.function("inv", lambda A: sp.ImmutableMatrix(A).inv(), signature="inv(A)", doc="inverse", category=M)
    api.function("transpose", lambda A: sp.ImmutableMatrix(A).T, signature="transpose(A)", doc="transpose",
                 category=M)
    api.function("eigenvals", lambda A: list(sp.Matrix(A).eigenvals().keys()), signature="eigenvals(A)",
                 doc="eigenvalues", category=M)
    api.function("dot", lambda a, b: sp.Matrix(a).dot(sp.Matrix(b)), signature="dot(a, b)", doc="dot product",
                 category=M)
    api.function("cross", lambda a, b: sp.ImmutableMatrix(sp.Matrix(a).cross(sp.Matrix(b))),
                 signature="cross(a, b)", doc="cross product", category=M)
    api.function("norm", lambda a: sp.Matrix(a).norm(), signature="norm(a)", doc="Euclidean length", category=M)
