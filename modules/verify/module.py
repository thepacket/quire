"""Verification tools: dimensional analysis, identity checking, uncertainty propagation, solution checks."""
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "verify"
DESCRIPTION = "Dimension checks, numeric identity tests with counterexamples, uncertainty propagation, solution checks."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def dimension(expr):
    """The dimension of an expression (length, mass/time**2, ...), or 1 if dimensionless."""
    expr = sp.sympify(expr)
    if not U.has_units(expr):
        return sp.S.One
    d = U.dimension_of(U.to_base(expr))
    _note(f"SI unit: {U.unit_label(expr) or '1'}")
    return d


def check_units(eq):
    """True when both sides of an equation have the same dimension; explains otherwise."""
    if not isinstance(eq, sp.Eq):
        raise EvalError("Write the relation with ==, e.g. check_units(F == m a).")
    dl, dr = dimension(eq.lhs), dimension(eq.rhs)
    dimsys = U.SI.get_dimension_system()
    same = dimsys.equivalent_dims(u.Dimension(dl), u.Dimension(dr))
    _note(f"left side: {dl}; right side: {dr}")
    return bool(same)


def check_identity(eq, tries=6):
    """Test an identity numerically at random points (complex unless symbols are assumed real)."""
    from quire.modules.builtin.algebra import numerically_equal

    if not isinstance(eq, sp.Eq):
        raise EvalError("Write the identity with ==, e.g. check_identity(sin(x)^2 + cos(x)^2 == 1).")
    ok = numerically_equal(sp.sympify(eq.lhs), sp.sympify(eq.rhs), tries=int(tries))
    _note(f"checked at {int(tries)} random points; agreement to 1e-9 is strong evidence, not a proof" if ok else "the two sides differ at a random point")
    return ok


def counterexample(eq):
    """A point where the two sides differ, or None if none was found in 40 random tries."""
    import random

    if not isinstance(eq, sp.Eq):
        raise EvalError("Write the claim with ==.")
    lhs, rhs = sp.sympify(eq.lhs), sp.sympify(eq.rhs)
    syms = sorted(lhs.free_symbols | rhs.free_symbols, key=lambda s: s.name)
    rng = random.Random(11)
    for _ in range(40):
        point = {s: sp.Rational(rng.randint(-20, 20), rng.randint(1, 7)) for s in syms}
        try:
            a, b = complex(sp.N(lhs.subs(point))), complex(sp.N(rhs.subs(point)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if abs(a - b) > 1e-9 * max(1.0, abs(a), abs(b)):
            _note(f"left = {a.real if not a.imag else a}, right = {b.real if not b.imag else b}")
            return [sp.Eq(s, v, evaluate=False) for s, v in point.items()]
    return sp.Symbol("none_found")


def propagate(expr, measurements):
    """[value, uncertainty] by linear error propagation for measurements [[x, x_value, x_error], ...].

    Symbolic when the values are symbols: propagate(a b, [[a, a, s_a], [b, b, s_b]]).
    """
    expr = sp.sympify(expr)
    rows = [list(m) for m in measurements]
    subs = {}
    var = sp.S.Zero
    for row in rows:
        if len(row) != 3:
            raise EvalError("Each measurement is [variable, value, uncertainty].")
        x, val, err = sp.sympify(row[0]), sp.sympify(row[1]), sp.sympify(row[2])
        if not isinstance(x, sp.Symbol):
            raise EvalError(f"'{x}' is not a variable.")
        subs[x] = val
        var += (sp.diff(expr, x) * err) ** 2
    value = sp.simplify(expr.subs(subs))
    sigma = sp.simplify(sp.sqrt(var.subs(subs)))
    _note("first-order propagation: sigma_f^2 = sum (df/dx_i sigma_i)^2, independent errors")
    return [value, sigma]


def relative_error(expr, measurements):
    v, s = propagate(expr, measurements)
    return sp.simplify(s / v)


def significant(x, n=3):
    """Round to n significant figures (units preserved)."""
    x = sp.sympify(x)
    num, unit = U.split_units(x)
    if not num.is_number:
        raise EvalError("significant needs a number.")
    val = float(num)
    if val == 0:
        return sp.S.Zero * unit
    import math
    digits = int(n) - int(math.floor(math.log10(abs(val)))) - 1
    return sp.Float(round(val, digits), int(n)) * unit


def percent_error(measured, true_value):
    return sp.simplify(100 * sp.Abs(sp.sympify(measured) - sp.sympify(true_value)) / sp.Abs(sp.sympify(true_value)))


def check_solution(eq, x, value):
    """Substitute a candidate into an equation and report the residual."""
    if not isinstance(eq, sp.Eq):
        raise EvalError("Write the equation with ==.")
    residual = sp.simplify((eq.lhs - eq.rhs).subs(x, value))
    _note(f"residual after substitution: {residual}")
    return residual == 0


def check_ode(eq, y, x, solution):
    """Does y = solution satisfy the ODE written with D(y, x, n)?"""
    if not isinstance(eq, sp.Eq):
        raise EvalError("Write the equation with ==.")
    yf = sp.Function(y.name)(x)
    expr = (eq.lhs - eq.rhs).subs(y, yf)
    residual = sp.simplify(expr.subs(yf, solution).doit())
    _note(f"residual: {residual}")
    return residual == 0


def register(api):
    V = "Verify"
    api.function("dimension", dimension, signature="dimension(expr)", doc="dimension of an expression", category=V,
                 example="dimension(1/2 m v^2)")
    api.function("check_units", check_units, signature="check_units(lhs == rhs)", doc="same dimension on both sides?", category=V,
                 example="check_units(F == m a)")
    api.function("check_identity", check_identity, signature="check_identity(lhs == rhs)", doc="numeric identity test", category=V,
                 example="check_identity(sin(x)^2 + cos(x)^2 == 1)")
    api.function("counterexample", counterexample, signature="counterexample(lhs == rhs)", doc="a point where a claim fails", category=V,
                 example="counterexample((x + y)^2 == x^2 + y^2)")
    api.function("propagate", propagate, signature="propagate(expr, [[x, value, error], ...])", doc="[value, uncertainty]", category=V,
                 example="propagate(V/I, [[V, 12.0, 0.1], [I, 2.0, 0.05]])")
    api.function("relative_error", relative_error, signature="relative_error(expr, measurements)", doc="sigma_f / f", category=V)
    api.function("significant", significant, signature="significant(x, n)", doc="round to n significant figures", category=V,
                 example="significant(pi 1000, 3)")
    api.function("percent_error", percent_error, signature="percent_error(measured, true)", doc="percent error", category=V)
    api.function("check_solution", check_solution, signature="check_solution(eq, x, value)", doc="does the value satisfy the equation?", category=V)
    api.function("check_ode", check_ode, signature="check_ode(eq, y, x, solution)", doc="does the function satisfy the ODE?", category=V,
                 example="check_ode(D(y, x) == y, y, x, exp(x))")
