"""FriCAS backend.

FriCAS carries the most complete Risch implementation among open-source systems,
so it is registered for integrate (and limit) after Maxima. Each call runs a
short-lived ``fricas -nosman`` process fed on stdin, and reads results printed
with a marker through ``unparse(expr :: InputForm)``.
"""
from __future__ import annotations

import re
import shutil
import subprocess

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations
from sympy.printing.str import StrPrinter

NAME = "fricas"
FRICAS = shutil.which("fricas") or ("/opt/homebrew/bin/fricas" if shutil.which("/opt/homebrew/bin/fricas") else None)
DESCRIPTION = ("FriCAS backend (Risch integration) for integrals SymPy and Maxima cannot finish."
               if FRICAS else "FriCAS backend (fricas not found on PATH; install it to enable).")
TIMEOUT = 25
MARK = "QRES" + "MARK"

_SYMPY_NS = {k: getattr(sp, k) for k in dir(sp) if not k.startswith("_")}
_SYMPY_NS["__builtins__"] = {}


class Unsupported(Exception):
    pass


_FUNC_OUT = {
    "Abs": "abs", "exp": "exp", "log": "log", "sqrt": "sqrt", "gamma": "Gamma", "erf": "erf",
    "besselj": "besselJ", "bessely": "besselY", "besseli": "besselI", "besselk": "besselK",
    "airyai": "airyAi", "airybi": "airyBi", "Ei": "Ei", "Si": "Si", "Ci": "Ci", "li": "li",
    "digamma": "digamma", "polygamma": "polygamma", "beta": "Beta", "LambertW": "lambertW",
    "factorial": "factorial", "binomial": "binomial", "floor": "floor", "ceiling": "ceiling", "sign": "sign",
}
for _t in ["sin", "cos", "tan", "sec", "csc", "cot", "asin", "acos", "atan", "acot", "asec", "acsc",
           "sinh", "cosh", "tanh", "sech", "csch", "coth", "asinh", "acosh", "atanh", "acoth"]:
    _FUNC_OUT[_t] = _t


class FricasPrinter(StrPrinter):
    def _print_Pi(self, e):
        return "%pi"

    def _print_Exp1(self, e):
        return "%e"

    def _print_ImaginaryUnit(self, e):
        return "%i"

    def _print_Infinity(self, e):
        return "%plusInfinity"

    def _print_NegativeInfinity(self, e):
        return "%minusInfinity"

    def _print_EulerGamma(self, e):
        raise Unsupported("EulerGamma")

    def _print_Function(self, e):
        name = e.func.__name__
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


_BACK = [
    (r"%pi", "pi"), (r"%e\b", "E"), (r"%i\b", "I"), (r"%plusInfinity", "oo"), (r"%minusInfinity", "(-oo)"),
    (r"\^", "**"), (r"\bGamma\(", "gamma("), (r"\bBeta\(", "beta("), (r"\bbesselJ\(", "besselj("),
    (r"\bbesselY\(", "bessely("), (r"\bbesselI\(", "besseli("), (r"\bbesselK\(", "besselk("),
    (r"\bairyAi\(", "airyai("), (r"\bairyBi\(", "airybi("), (r"\blambertW\(", "LambertW("), (r"\babs\(", "Abs("),
]
_FAILED = re.compile(r"potentialPole|failed|\bintegrate\(|\blimit\(|%%|\?|\bplusInfinity\b|\bminusInfinity\b")


def _to_fricas(expr, symmap) -> str:
    return FricasPrinter().doprint(sp.sympify(expr).subs(symmap))


def _from_fricas(text: str, symmap: dict):
    text = text.strip().strip('"')
    if not text or _FAILED.search(text):
        return None
    for pat, rep in _BACK:
        text = re.sub(pat, rep, text)
    if "%" in text:
        return None
    local = {v.name: k for k, v in symmap.items()}
    try:
        return parse_expr(text, local_dict=local, global_dict=dict(_SYMPY_NS),
                          transformations=standard_transformations)
    except Exception:  # noqa: BLE001
        return None


def _run(lines: list[str]) -> list[str]:
    if not FRICAS:
        return []
    script = "\n".join([")set message type off", ")set message time off", ")set output algebra off",
                        ")set output length 20000", ")set quit unprotected"] + lines + [")quit", ""])
    try:
        r = subprocess.run([FRICAS, "-nosman"], input=script, capture_output=True, text=True, timeout=TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return []
    out, current = [], None
    for line in r.stdout.splitlines():
        if MARK in line:
            if current is not None:
                out.append(current)
            current = line.split(MARK, 1)[1].strip()
        elif current is not None:
            # unparse wraps long results; continuation lines are indented and carry no prompt
            if line.startswith(" ") and not re.match(r"\s*\(\d+\) ->", line):
                current += line.strip()
            else:
                out.append(current)
                current = None
    if current is not None:
        out.append(current)
    return [o.strip().strip('"') for o in out]


def _emit(code: str) -> str:
    return f'output(concat(concat("{MARK[:4]}", "{MARK[4:]}"), unparse(({code}) :: InputForm)))'


def _prepare(exprs):
    symbols = set()
    for e in exprs:
        symbols |= sp.sympify(e).free_symbols
    return {s: sp.Symbol(f"q{i}") for i, s in enumerate(sorted(symbols, key=lambda s: s.name))}


def integrate(f, ranges):
    flat = [f] + [e for r in ranges for e in (r if isinstance(r, (tuple, list)) else (r,))]
    try:
        symmap = _prepare(flat)
        code = _to_fricas(f, symmap)
        for r in ranges:
            if isinstance(r, (tuple, list)):
                x, a, b = r
                code = f'integrate({code}, {_to_fricas(x, symmap)} = {_to_fricas(a, symmap)}..{_to_fricas(b, symmap)}, "noPole")'
            else:
                code = f"integrate({code}, {_to_fricas(r, symmap)})"
    except Unsupported:
        return None
    for text in _run([_emit(code)]):
        res = _from_fricas(text, symmap)
        if res is not None and not res.has(sp.Integral):
            return res
    return None


def limit(f, x, x0, direction="+-"):
    try:
        symmap = _prepare([f, x, x0])
        d = {"+": ', "right"', "-": ', "left"'}.get(direction, "")
        code = f"limit({_to_fricas(f, symmap)}, {_to_fricas(x, symmap)} = {_to_fricas(x0, symmap)}{d})"
    except Unsupported:
        return None
    for text in _run([_emit(code)]):
        res = _from_fricas(text, symmap)
        if res is not None:
            return res
    return None


def register(api):
    if not FRICAS:
        return
    api.fallback("integrate", integrate, priority=30)
    api.fallback("limit", limit, priority=30)
    F = "FriCAS backend"
    api.function("fricas_integrate", lambda f, x, a=None, b=None: integrate(f, [(x, a, b)] if a is not None else [x]),
                 signature="fricas_integrate(f, x, a, b)", doc="integrate with FriCAS (Risch algorithm)", category=F,
                 example="fricas_integrate(exp(x) (1 + x)/x^2, x)")
    api.function("fricas_limit", lambda f, x, x0: limit(f, x, x0), signature="fricas_limit(f, x, x0)",
                 doc="limit with FriCAS", category=F)
