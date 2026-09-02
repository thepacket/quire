"""Constants, units, arithmetic, trigonometry, complex numbers."""
import sympy as sp

from ...engine import units as U
from ._util import as_list


def _angle_aware(fn, name):
    def f(x):
        return fn(U.strip_angles(x))

    f.__name__ = name
    return f


def _log(x, base=None):
    return sp.log(x) if base is None else sp.log(x, base)


def _round(x, n=0):
    num, unit = U.split_units(x)
    return (sp.Float(round(float(num), int(n))) if n else sp.Integer(round(float(num)))) * unit


def _max(*args):
    return sp.Max(*(as_list(args[0]) if len(args) == 1 else args))


def _min(*args):
    return sp.Min(*(as_list(args[0]) if len(args) == 1 else args))


def _polar(z):
    return [sp.Abs(z), sp.arg(z)]


def _slider(value, lo, hi, step=None):
    """The current value of a slider; the UI draws the control under the cell."""
    from .. import hooks

    v, lo, hi = sp.sympify(value), sp.sympify(lo), sp.sympify(hi)
    if any(not x.is_number for x in (v, lo, hi)):
        raise ValueError("slider(value, min, max) needs numbers.")
    hooks.context["slider"] = {"value": float(v), "min": float(lo), "max": float(hi),
                               "step": float(step) if step is not None else None}
    return v


def register(api):
    api.function("slider", _slider, signature="slider(value, min, max, step)",
                 doc="a value you can drag; everything that uses it updates", category="Interactive",
                 example="a = slider(1, 0, 5)")
    C = "Constants"
    api.constant("pi", sp.pi, doc="π", category=C, example="2 pi")
    api.constant("e", sp.E, doc="Euler's number", category=C, example="e^x")
    api.constant("i", sp.I, doc="imaginary unit", category=C, example="3 + 4 i")
    api.constant("oo", sp.oo, doc="infinity", category=C, example="limit(1/x, x, oo)")
    api.constant("inf", sp.oo, doc="infinity", category=C)
    api.constant("EulerGamma", sp.EulerGamma, doc="Euler–Mascheroni constant", category=C)
    api.constant("GoldenRatio", sp.GoldenRatio, doc="golden ratio", category=C)
    for name, (val, doc) in U.CONSTANT_TABLE.items():
        api.constant(name, val, doc=doc, category="Physical constants")
    for name, (q, doc) in U.UNIT_TABLE.items():
        api.unit(name, q, doc=doc, category="Units")

    A = "Arithmetic"
    api.function("abs", sp.Abs, signature="abs(x)", doc="absolute value / modulus", category=A)
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

    T = "Trigonometry"
    for name in ["sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh", "sech", "csch", "coth"]:
        api.function(name, _angle_aware(getattr(sp, name), name), signature=f"{name}(x)",
                     doc=f"{name}; accepts deg or rad", category=T, example=f"{name}(30 deg)")
    for name in ["asin", "acos", "atan", "acot", "asec", "acsc", "asinh", "acosh", "atanh", "acoth"]:
        api.function(name, getattr(sp, name), signature=f"{name}(x)", doc=f"inverse {name[1:]} (radians)",
                     category=T, example=f"{name}(1/2) -> deg")
    api.function("atan2", lambda y, x: sp.atan2(U.strip_angles(y), U.strip_angles(x)), signature="atan2(y, x)",
                 doc="two-argument arctangent", category=T)
    api.function("sinc", sp.sinc, signature="sinc(x)", doc="sin(x)/x", category=T)

    Z = "Complex numbers"
    api.function("re", sp.re, signature="re(z)", doc="real part", category=Z)
    api.function("im", sp.im, signature="im(z)", doc="imaginary part", category=Z)
    api.function("conj", sp.conjugate, signature="conj(z)", doc="complex conjugate", category=Z)
    api.function("arg", sp.arg, signature="arg(z)", doc="argument (angle) of z", category=Z)
    api.function("polar", _polar, signature="polar(z)", doc="[modulus, angle]", category=Z, example="polar(1 + i)")
    api.function("expand_complex", sp.expand_complex, signature="expand_complex(z)",
                 doc="split into real and imaginary parts", category=Z, example="expand_complex(exp(i x))")
    api.function("residue", lambda f, z, z0: sp.residue(f, z, z0), signature="residue(f, z, z0)",
                 doc="residue of f at the pole z0", category=Z, example="residue(1/(z^2 + 1), z, i)")
