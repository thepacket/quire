"""Evaluate a worksheet: an ordered list of cells, top to bottom.

Definitions bind names for every cell below them (worksheet semantics). A
symbol that has no definition stays symbolic. The whole document is
re-evaluated on every change; it is cheap at worksheet scale and makes the
reactive behaviour trivially correct.
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.logic.boolalg import BooleanAtom

from . import units as U
from .errors import QuireError
from .parser import UNIT_ALIAS, alias_units, classify, identifiers, parse

INTERNAL_NAMES = {"Symbol", "Integer", "Float", "Rational", "Function", "Lambda"}
ASSUME_KEY = "$assume"  # env slot for symbol assumptions; '$' cannot appear in a name

_ASSUMPTION_LATEX = {
    "positive": "{n} > 0", "negative": "{n} < 0", "nonnegative": r"{n} \geq 0", "nonpositive": r"{n} \leq 0",
    "nonzero": r"{n} \neq 0", "real": r"{n} \in \mathbb{{R}}", "integer": r"{n} \in \mathbb{{Z}}",
    "rational": r"{n} \in \mathbb{{Q}}", "complex": r"{n} \in \mathbb{{C}}",
    "irrational": r"{n} \notin \mathbb{{Q}}", "even": r"{n} \text{{ even}}", "odd": r"{n} \text{{ odd}}",
    "prime": r"{n} \text{{ prime}}", "finite": r"{n} \text{{ finite}}",
}


class DefinedFunction:
    """A worksheet function ``f(x) = body [-> unit]``.

    Calling it substitutes the arguments. The optional conversion is applied
    to the result once it is fully numeric; a still-symbolic result (a plot
    variable, or a nested call) is returned as is, since conversion only
    changes representation, never value.
    """

    def __init__(self, name: str, params: list[str], body, target=None, target_text: str | None = None,
                 symbols=None):
        self.name = name
        self.params = params
        self.lam = sp.Lambda(tuple(symbols or [sp.Symbol(p) for p in params]), body)
        self.expr = body
        self.target = target
        self.target_text = target_text

    def __call__(self, *args):
        result = self.lam(*args)
        if self.target is None or not isinstance(result, sp.Expr):
            return result
        if result.free_symbols:
            try:
                return U.convert(result, self.target)
            except Exception:  # noqa: BLE001 - defer until the value is bound
                return result
        return U.convert(result, self.target)

    @property
    def free_symbols(self):
        return self.lam.free_symbols

    def __repr__(self):
        return f"{self.name}({', '.join(self.params)}) = {self.expr}"


def fmt_number(x, digits: int) -> tuple[str, str]:
    """Return (latex, plain) for a numeric sympy value."""

    def one(v: float) -> tuple[str, str]:
        if v != v:  # nan
            return r"\text{undefined}", "undefined"
        if v in (float("inf"), float("-inf")):
            return (r"\infty", "inf") if v > 0 else (r"-\infty", "-inf")
        s = f"{v:.{digits}g}"
        if "e" in s:
            mant, exp = s.split("e")
            return rf"{mant} \times 10^{{{int(exp)}}}", f"{mant}e{int(exp)}"
        return s, s

    if x.is_real:
        return one(float(x))
    c = complex(x)
    rl, rp = one(c.real)
    il, ip = one(abs(c.imag))
    sign = "-" if c.imag < 0 else "+"
    if abs(c.real) < 10 ** (-digits) * max(1.0, abs(c.imag)):
        return (f"{'-' if c.imag < 0 else ''}{il} i", f"{'-' if c.imag < 0 else ''}{ip}i")
    return f"{rl} {sign} {il} i", f"{rp} {sign} {ip}i"


def _is_plain_number(x) -> bool:
    return isinstance(x, (sp.Integer, sp.Float))


def approx_of(value, digits: int):
    """Numeric approximation as (latex, plain), or None if not useful."""
    if isinstance(value, (list, tuple)):
        parts = [approx_of(v, digits) for v in value]
        if not parts or any(p is None for p in parts):
            return None
        return ("\\left[ " + ",\\  ".join(p[0] for p in parts) + " \\right]", "[" + ", ".join(p[1] for p in parts) + "]")
    if isinstance(value, sp.MatrixBase):
        if value.free_symbols or all(_is_plain_number(v) for v in value):
            return None
        return sp.latex(value.evalf(digits)), str(value.evalf(digits))
    if not isinstance(value, sp.Expr) or value.free_symbols:
        return None
    if isinstance(value, BooleanAtom):
        return None
    exact_num, _ = U.split_units(value)
    if _is_plain_number(exact_num):
        return None
    try:
        val = sp.N(value, digits + 3)
    except Exception:
        return None
    num, unit = U.split_units(val)
    if not num.is_number:
        return None
    lat, plain = fmt_number(num, digits)
    if unit != 1:
        lat += " \\, " + pretty_units(sp.latex(unit))
        plain += " " + str(unit)
    return lat, plain


_UNIT_AFTER_NUMBER = re.compile(r"(\d|\}) (\\text\{)")
_UNIT_AFTER_UNIT = re.compile(r"(\\text\{[^}]*\}(?:\^\{[^}]*\})?) (\\text\{)")


def pretty_units(latex: str) -> str:
    """Thin spaces between a number and its unit, and between unit symbols."""
    latex = _UNIT_AFTER_NUMBER.sub(r"\1\\,\2", latex)
    return _UNIT_AFTER_UNIT.sub(r"\1\\,\2", latex)


def to_latex(value) -> str:
    try:
        if isinstance(value, (sp.Lambda, DefinedFunction)):
            return to_latex(value.expr)
        if isinstance(value, (list, tuple)):
            return "\\left[ " + ",\\  ".join(to_latex(v) for v in value) + " \\right]"
        if isinstance(value, sp.Expr) and U.has_units(value) and not isinstance(value, sp.Add):
            num, unit = U.split_units(value)
            if num == 1 and unit != 1:
                return "1\\," + pretty_units(sp.latex(unit))  # "1 s", not a bare "s"
        if not isinstance(value, (sp.Basic, list, tuple, dict)):
            return r"\text{" + str(value).replace("_", r"\_") + "}"
        return pretty_units(sp.latex(value))
    except Exception:
        return r"\text{" + str(value).replace("_", r"\_") + "}"


def to_plain(value) -> str:
    if isinstance(value, (sp.Lambda, DefinedFunction)):
        return str(value.expr)
    return str(value)


class Evaluator:
    def __init__(self, registry, digits: int = 6):
        self.registry = registry
        self.base_namespace = registry.namespace()
        self.units_namespace = {e.name: e.value for m in registry.modules for e in m.entries if e.kind == "unit"}
        self.unit_names = frozenset(self.units_namespace)
        # Single-letter unit symbols (m, s, g, N, V, ...) collide with the names people give variables.
        # They count as units only in unit position: after a number ("3 m"), or as a "->" target.
        # Longer names (kg, Hz, meter, second) are always units.
        for name in self.units_namespace:
            if len(name) == 1 and self.base_namespace.get(name) is self.units_namespace[name]:
                del self.base_namespace[name]
        # aliases used by the parser for "number followed by unit" phrases
        self.base_namespace.update({UNIT_ALIAS + k: v for k, v in self.units_namespace.items()})
        self.digits = digits

    # -- namespaces -----------------------------------------------------
    def namespace(self, env: dict, bound: list[str] = ()) -> dict:
        ns = dict(self.base_namespace)
        assumed = env.get(ASSUME_KEY, {})
        for name, a in assumed.items():
            ns[name] = sp.Symbol(name, **a)
        ns.update({k: v for k, v in env.items() if k != ASSUME_KEY})
        for b in bound:
            ns[b] = sp.Symbol(b, **assumed.get(b, {}))
        return ns

    def parse_unit(self, text: str):
        return parse(text, dict(self.units_namespace))

    # -- document -------------------------------------------------------
    def evaluate(self, cells: list[dict]) -> list[dict]:
        from .plotting import sample_plot

        env: dict = {}
        results = []
        for cell in cells:
            kind = cell.get("type", "math")
            cid = cell.get("id")
            if kind == "text":
                results.append({"id": cid, "ok": True})
            elif kind == "plot":
                results.append({"id": cid, **sample_plot(cell, env, self)})
            else:
                results.append({"id": cid, **self.evaluate_math(cell.get("source", ""), env)})
        return results

    # -- one math cell --------------------------------------------------
    def evaluate_math(self, source: str, env: dict) -> dict:
        outputs, defines, uses = [], [], set()
        error = warning = None
        self.last_values = []  # raw values of each output line (used by the benchmark)
        for line in source.split("\n"):
            if not line.strip():
                continue
            try:
                st = classify(line)
                if st.kind == "assume":
                    outputs.append(self.assume(st, env))
                    continue
                ns = self.namespace(env, st.params)
                # names used, ignoring unit phrases such as "3 m" even when m is also defined
                uses |= (identifiers(alias_units(st.body, self.unit_names)) - set(st.params)) & set(env) - {ASSUME_KEY}
                value = parse(st.body, ns, self.unit_names)
                if st.kind == "function":
                    # The body is checked and converted when called: parameters may carry units.
                    if not isinstance(value, sp.Expr):
                        raise QuireError("A function body must be a single expression.")
                    target = self.parse_unit(st.convert_to) if st.convert_to else None
                    value = DefinedFunction(st.name, st.params, value, target, st.convert_to,
                                            symbols=[ns[p] for p in st.params])
                else:
                    value = self.finalize(value, st.convert_to)
                if st.kind in ("definition", "function") and st.name in self.unit_names and len(st.name) > 1:
                    warning = (f"'{st.name}' now means your definition. Directly after a number it still "
                               f"means the unit ({U.UNIT_TABLE[st.name][1]}), as in '3 {st.name}'; "
                               f"write '3*{st.name}' for three times yours.")
                if st.kind in ("definition", "function"):
                    env[st.name] = value
                    defines.append(st.name)
                self.last_values.append(value)
                outputs.append(self.render(st, value))
            except QuireError as exc:
                error = str(exc)
                break
            except RecursionError:
                error = "The expression is too deeply nested."
                break
            except Exception as exc:  # noqa: BLE001 - surface anything sympy throws
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                break
        return {"ok": error is None, "outputs": outputs, "defines": defines, "uses": sorted(uses), "error": error,
                "warning": warning}

    def assume(self, st, env: dict) -> dict:
        assumed = env.setdefault(ASSUME_KEY, {})
        for n in st.names:
            merged = {**assumed.get(n, {}), **st.assumptions[n]}
            try:
                sp.Symbol(n, **merged)
            except Exception as exc:  # noqa: BLE001 - inconsistent assumptions
                raise QuireError(f"Assumptions on '{n}' contradict each other: {exc}") from None
            assumed[n] = merged
        parts = []
        for n in st.names:
            head = sp.latex(sp.Symbol(n))
            for key in st.assumptions[n]:
                parts.append(_ASSUMPTION_LATEX.get(key, "{n}").format(n=head))
        return {"kind": "assume", "latex": ",\\ ".join(parts),
                "plain": "; ".join(f"{n} {' '.join(st.assumptions[n])}" for n in st.names)}

    def finalize(self, value, convert_to: str | None, check: bool = True):
        if isinstance(value, (list, tuple)):
            return type(value)(self.finalize(v, convert_to, check) for v in value)
        if isinstance(value, (int, float, complex, bool)):
            value = sp.sympify(value)
        if isinstance(value, sp.Expr):
            if check:
                U.check_dimensions(value)
            if convert_to:
                target = self.parse_unit(convert_to)
                value = U.convert(value, target)
            else:
                value = U.tidy_units(value)
        elif convert_to:
            raise QuireError("'->' conversion needs a single quantity on the left.")
        return value

    def display_value(self, value):
        """What to show: inexact inputs (decimals) give a rounded numeric result."""
        if isinstance(value, (list, tuple)):
            return type(value)(self.display_value(v) for v in value)
        if isinstance(value, sp.Expr) and not isinstance(value, (sp.Lambda, DefinedFunction, BooleanAtom)) \
                and not value.free_symbols and value.atoms(sp.Float):
            num, unit = U.split_units(value)
            if isinstance(num, sp.Float) or (isinstance(num, sp.Expr) and not num.is_number):
                return value  # already a plain decimal (keep the user's precision), or not numeric
            try:
                num = sp.N(num, self.digits + 3)
            except Exception:
                return value
            if num.is_number and num.is_real:
                return sp.Float(float(num), self.digits) * unit if float(num) != 0 else sp.S.Zero * unit
        return value

    def render(self, st, value) -> dict:
        shown = self.display_value(value)
        out = {"kind": st.kind, "latex": to_latex(shown), "plain": to_plain(shown)}
        if st.kind in ("definition", "function"):
            out["name"] = st.name
            head = sp.latex(sp.Symbol(st.name))
            if st.kind == "function":
                head += r"\left(" + ", ".join(sp.latex(sp.Symbol(p)) for p in st.params) + r"\right)"
            out["head"] = head
        approx = approx_of(shown.expr if isinstance(shown, (sp.Lambda, DefinedFunction)) else shown, self.digits)
        if approx:
            out["approx"], out["approx_plain"] = approx
        return out
