"""Maxima backend.

Registers fallbacks for integrate, limit, sum and simplify: they run only when
SymPy gives up (unevaluated Integral/Limit/Sum, or an expression it cannot
shrink). Also exposes maxima_integrate / maxima_limit / maxima_sum /
maxima_simplify for explicit use. Each call is a short-lived Maxima process
with stdin closed, so an interactive question fails instead of hanging.
"""
from __future__ import annotations

import re
import shutil
import subprocess

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.printing.str import StrPrinter

from quire.modules import hooks

_SYMPY_NS = {k: getattr(sp, k) for k in dir(sp) if not k.startswith("_")}
_SYMPY_NS["__builtins__"] = {}

NAME = "maxima"
MAXIMA = shutil.which("maxima") or ("/opt/homebrew/bin/maxima" if shutil.which("/opt/homebrew/bin/maxima") else None)
DESCRIPTION = ("Maxima backend for integrals, limits, sums and simplification SymPy cannot finish."
               if MAXIMA else "Maxima backend (maxima not found on PATH; install it to enable).")
TIMEOUT = 15
MARK = "QRES" + "MARK"  # concatenated at runtime so the echo of the code never contains it


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------- sympy -> maxima
_FUNC_OUT = {
    "Abs": "abs", "sign": "signum", "ceiling": "ceiling", "floor": "floor", "exp": "exp", "log": "log",
    "sqrt": "sqrt", "gamma": "gamma", "loggamma": "log_gamma", "erf": "erf", "erfc": "erfc", "erfi": "erfi",
    "besselj": "bessel_j", "bessely": "bessel_y", "besseli": "bessel_i", "besselk": "bessel_k",
    "LambertW": "lambert_w", "Heaviside": "unit_step", "DiracDelta": "delta", "zeta": "zeta",
    "factorial": "factorial", "binomial": "binomial", "beta": "beta", "atan2": "atan2",
    "elliptic_k": "elliptic_kc", "elliptic_e": "elliptic_ec", "Ei": "expintegral_ei", "Si": "expintegral_si",
    "Ci": "expintegral_ci", "li": "expintegral_li", "expint": "expintegral_e", "uppergamma": "gamma_incomplete",
    "harmonic": "harmonic_number", "digamma": "psi",
}
for _t in ["sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "acot", "asec", "acsc",
           "sinh", "cosh", "tanh", "sech", "csch", "coth", "asinh", "acosh", "atanh", "acoth"]:
    _FUNC_OUT[_t] = _t


class MaximaPrinter(StrPrinter):
    def _print_Pi(self, e):
        return "%pi"

    def _print_Exp1(self, e):
        return "%e"

    def _print_ImaginaryUnit(self, e):
        return "%i"

    def _print_EulerGamma(self, e):
        return "%gamma"

    def _print_Catalan(self, e):
        return "%catalan"

    def _print_Infinity(self, e):
        return "inf"

    def _print_NegativeInfinity(self, e):
        return "minf"

    def _print_Function(self, e):
        name = e.func.__name__
        if name == "polylog":
            return f"li[{self._print(e.args[0])}]({self._print(e.args[1])})"
        if name == "polygamma":
            return f"psi[{self._print(e.args[0])}]({self._print(e.args[1])})"
        if name == "digamma":
            return f"psi[0]({self._print(e.args[0])})"
        if name not in _FUNC_OUT:
            raise Unsupported(name)
        return f"{_FUNC_OUT[name]}({', '.join(self._print(a) for a in e.args)})"

    def _print_Pow(self, e, rational=False):
        return f"({self._print(e.base)})^({self._print(e.exp)})"

    def _print_Piecewise(self, e):
        raise Unsupported("Piecewise")

    def _print_Integral(self, e):
        raise Unsupported("Integral")

    def _print_Sum(self, e):
        raise Unsupported("Sum")

    def _print_Derivative(self, e):
        raise Unsupported("Derivative")

    def _print_Relational(self, e):
        raise Unsupported("Relational")


def _to_maxima(expr, symmap: dict) -> str:
    expr = sp.sympify(expr).subs(symmap)
    return MaximaPrinter().doprint(expr)


def _assumptions(symbols, symmap) -> str:
    """Maxima assume/declare statements from sympy symbol assumptions and worksheet bounds."""
    out = []
    bounds = hooks.context.get("bounds", {})
    for s in symbols:
        q = symmap[s]
        a = s.assumptions0
        if s.name in bounds:
            for op, val in bounds[s.name]:
                out.append(f"assume({q}{op}{_to_maxima(val, symmap)})$")
        elif a.get("positive"):
            out.append(f"assume({q}>0)$")
        elif a.get("negative"):
            out.append(f"assume({q}<0)$")
        elif a.get("nonnegative"):
            out.append(f"assume({q}>=0)$")
        elif a.get("nonpositive"):
            out.append(f"assume({q}<=0)$")
        if a.get("nonzero") and not a.get("positive") and not a.get("negative"):
            out.append(f"assume(notequal({q},0))$")
        if a.get("integer"):
            out.append(f"declare({q}, integer)$")
        elif a.get("real"):
            out.append(f"declare({q}, real)$")
        if a.get("even"):
            out.append(f"declare({q}, even)$")
        if a.get("odd"):
            out.append(f"declare({q}, odd)$")
    return "".join(out)


# ---------------------------------------------------------------- maxima -> sympy
_BACK = [
    (r"%pi", "pi"), (r"%e\b", "E"), (r"%i\b", "I"), (r"%gamma", "EulerGamma"), (r"%catalan", "Catalan"),
    (r"\bminf\b", "(-oo)"), (r"\binf\b", "oo"), (r"\^", "**"),
    (r"\bbessel_j\(", "besselj("), (r"\bbessel_y\(", "bessely("), (r"\bbessel_i\(", "besseli("),
    (r"\bbessel_k\(", "besselk("), (r"\blambert_w\(", "LambertW("), (r"\bunit_step\(", "Heaviside("),
    (r"\bdelta\(", "DiracDelta("), (r"\bsignum\(", "sign("), (r"\bgamma_incomplete\(", "uppergamma("),
    (r"\bexpintegral_e\(", "expint("), (r"\bexpintegral_ei\(", "Ei("), (r"\bexpintegral_si\(", "Si("),
    (r"\bexpintegral_ci\(", "Ci("), (r"\bexpintegral_li\(", "li("), (r"\belliptic_kc\(", "elliptic_k("),
    (r"\belliptic_ec\(", "elliptic_e("), (r"\blog_gamma\(", "loggamma("), (r"\bharmonic_number\(", "harmonic("),
    (r"\babs\(", "Abs("), (r"\bli\[([^\]]+)\]\(", r"polylog(\1, "), (r"\bpsi\[([^\]]+)\]\(", r"polygamma(\1, "),
    (r"%c\b", "C1"), (r"%k1\b", "C1"), (r"%k2\b", "C2"), (r"\bfalse\b", "False"), (r"\btrue\b", "True"),
]
_FAILED = re.compile(r"'|\?|%if|\bintegrate\(|\blimit\(|\bsum\(|\bproduct\(|%r\d|%z\d|\bund\b|\bind\b|\binfinity\b")


def _from_maxima(text: str, symmap: dict):
    text = text.strip()
    if not text or _FAILED.search(text):
        return None
    for pat, rep in _BACK:
        text = re.sub(pat, rep, text)
    if "%" in text:
        return None
    local = {v.name: k for k, v in symmap.items()}
    local.update({"C1": sp.Symbol("C1"), "C2": sp.Symbol("C2")})
    try:
        return parse_expr(text, local_dict=local, global_dict=dict(_SYMPY_NS),
                          transformations=standard_transformations)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- running
QUESTION = "RETRIEVE: End of file encountered"  # Maxima asked something; stdin is closed


def _run(code: str, generic: str = "") -> list[str]:
    """Run Maxima code; return the results printed with the marker, in order.

    If Maxima asks a question (typically "Is q an integer?"), retry once with the
    generic declarations in ``generic`` (non-integer parameters). Sign questions are
    never guessed: a second question means no result.
    """
    if not MAXIMA:
        return []
    prog = 'display2d:false$ ratprint:false$ ' + code
    try:
        r = subprocess.run([MAXIMA, "--very-quiet", "--batch-string", prog], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return []
    out = [line[len(MARK):] for line in r.stdout.splitlines() if line.startswith(MARK)]
    if not out and generic and QUESTION in (r.stdout + r.stderr):
        return _run(generic + code)
    return out


def _emit(expr_code: str) -> str:
    return f'printf(true, "~%~a~a~a~%", "{MARK[:4]}", "{MARK[4:]}", string({expr_code}))$'


def _prepare(exprs):
    """Map symbols to safe Maxima names; returns (symmap, assume code, generic declarations)."""
    symbols = set()
    for e in exprs:
        symbols |= sp.sympify(e).free_symbols
    symmap = {s: sp.Symbol(f"q{i}") for i, s in enumerate(sorted(symbols, key=lambda s: s.name))}
    generic = "".join(f"declare({symmap[s]}, noninteger)$" for s in symbols if s.is_integer is None)
    return symmap, _assumptions(symbols, symmap), generic


def _call(build, *exprs):
    """build(m) -> maxima code, where m(expr) prints an expression. Returns the parsed first result."""
    try:
        symmap, pre, generic = _prepare(exprs)
        code = pre + build(lambda e: _to_maxima(e, symmap))
    except Unsupported:
        return None
    for text in _run(code, generic):
        res = _from_maxima(text, symmap)
        if res is not None:
            return res
    return None


# ---------------------------------------------------------------- operations
def integrate(f, ranges):
    def build(m):
        code = m(f)
        for r in ranges:
            if isinstance(r, (tuple, list)):
                x, a, b = r
                code = f"integrate({code}, {m(x)}, {m(a)}, {m(b)})"
            else:
                code = f"integrate({code}, {m(r)})"
        return _emit(code)

    flat = [f] + [e for r in ranges for e in (r if isinstance(r, (tuple, list)) else (r,))]
    return _call(build, *flat)


def limit(f, x, x0, direction="+-"):
    d = {"+": ", plus", "-": ", minus"}.get(direction, "")
    return _call(lambda m: _emit(f"limit({m(f)}, {m(x)}, {m(x0)}{d})"), f, x, x0)


def summation(f, k, a, b):
    return _call(lambda m: "load(simplify_sum)$" + _emit(f"simplify_sum(sum({m(f)}, {m(k)}, {m(a)}, {m(b)}))"),
                 f, k, a, b)


def simplify(expr):
    expr = sp.sympify(expr)
    trig = expr.has(sp.sin, sp.cos, sp.tan, sp.sinh, sp.cosh, sp.tanh)

    def build(m):
        e = m(expr)
        forms = [f"ratsimp({e})", f"radcan({e})", f"factor({e})", f"fullratsimp({e})"]
        if trig:
            forms += [f"trigsimp({e})", f"trigrat({e})", f"ratsimp(trigexpand({e}))", f"trigreduce({e})"]
        return "".join(_emit(f) for f in forms)

    try:
        symmap, pre, generic = _prepare([expr])
        code = pre + build(lambda e: _to_maxima(e, symmap))
    except Unsupported:
        return None
    best = None
    for text in _run(code, generic):
        cand = _from_maxima(text, symmap)
        if cand is None:
            continue
        cand = sp.simplify(cand) if sp.count_ops(cand) < 40 else cand
        if best is None or sp.count_ops(cand) < sp.count_ops(best):
            best = cand
    return best


def register(api):
    if not MAXIMA:
        return
    api.fallback("integrate", integrate, priority=10)
    api.fallback("limit", limit, priority=10)
    api.fallback("sum", summation, priority=10)
    api.fallback("simplify", simplify, priority=10)
    M = "Maxima backend"
    api.function("maxima_integrate", lambda f, x, a=None, b=None: integrate(f, [(x, a, b)] if a is not None else [x]),
                 signature="maxima_integrate(f, x, a, b)", doc="integrate with Maxima", category=M,
                 example="maxima_integrate(x/(exp(x) - 1), x, 0, oo)")
    api.function("maxima_limit", lambda f, x, x0: limit(f, x, x0), signature="maxima_limit(f, x, x0)",
                 doc="limit with Maxima", category=M)
    api.function("maxima_sum", summation, signature="maxima_sum(f, k, a, b)", doc="sum with Maxima", category=M)
    api.function("maxima_simplify", simplify, signature="maxima_simplify(expr)",
                 doc="simplify with Maxima (ratsimp, radcan, trigsimp, ...)", category=M,
                 example="maxima_simplify(sin(x)^6 + cos(x)^6 + 3 sin(x)^2 cos(x)^2)")
