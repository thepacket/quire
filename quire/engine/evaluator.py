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
from .parser import GREEK_NAMES, SYM_ALIAS, UNIT_ALIAS, alias_units, called_names, classify, identifiers, parse
from .steps import Steps
from .notation import normalize, uses_notation

INTERNAL_NAMES = {"Symbol", "Integer", "Float", "Rational", "Function", "Lambda"}
IMPORT_RE = re.compile(r"^\s*import\s+(.+?)\s*$")
MEASURED_KEY = "$measured"  # env slot: name -> (symbol, nominal value, standard uncertainty) for "x = 12.3 ± 0.2 m"
DIMS_KEY = "$dims"          # env slot: name -> unit for "assume L length"
NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
MEASURED_RE = re.compile(rf"^\s*(?P<a>{NUMBER})\s*(?P<ua>[^±]*?)\s*(?:±|\+-|\+/-)\s*(?P<b>{NUMBER[5:]})\s*(?P<ub>.*?)\s*$")


def _same(a, b) -> bool:
    """Structural equality for cached environment values (identity first, then sympy equality)."""
    if a is b:
        return True
    if isinstance(a, DefinedFunction) and isinstance(b, DefinedFunction):
        return a.params == b.params and a.target_text == b.target_text and _same(a.expr, b.expr)
    if isinstance(a, sp.Basic) and isinstance(b, sp.Basic):
        try:
            return bool(a == b)
        except Exception:  # noqa: BLE001
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float, complex, bool, str)) and isinstance(b, (int, float, complex, bool, str)):
        return a == b
    return False


def placeholders(text: str) -> list[str]:
    """The {{expr}} placeholders of a text, skipping $...$ math spans."""
    out = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "`":  # skip a code span
            j = text.find("`", i + 1)
            if j < 0:
                break
            i = j + 1
            continue
        if text[i] == "$":  # skip a math span ($...$ or $$...$$)
            j = text.find("$$" if text.startswith("$$", i) else "$", i + (2 if text.startswith("$$", i) else 1))
            if j < 0:
                break
            i = j + (2 if text.startswith("$$", i) else 1)
            continue
        if text.startswith("{{", i):
            j = text.find("}}", i + 2)
            if j < 0:
                break
            out.append(text[i + 2:j].strip())
            i = j + 2
            continue
        i += 1
    return out
ASSUME_KEY = "$assume"  # env slot for symbol assumptions; '$' cannot appear in a name
BOUNDS_KEY = "$bounds"  # env slot for numeric bounds from 'assume s > 1' (used by backends)
DIGITS_KEY = "$digits"  # env slot for the display precision set by 'digits n' 
SPECIAL_COPY = (ASSUME_KEY, BOUNDS_KEY, DIGITS_KEY, MEASURED_KEY, DIMS_KEY, "$sliders")

_ASSUMPTION_LATEX = {
    "positive": "{n} > 0", "negative": "{n} < 0", "nonnegative": r"{n} \geq 0", "nonpositive": r"{n} \leq 0",
    "nonzero": r"{n} \neq 0", "real": r"{n} \in \mathbb{{R}}", "integer": r"{n} \in \mathbb{{Z}}",
    "rational": r"{n} \in \mathbb{{Q}}", "complex": r"{n} \in \mathbb{{C}}",
    "irrational": r"{n} \notin \mathbb{{Q}}", "even": r"{n} \text{{ even}}", "odd": r"{n} \text{{ odd}}",
    "prime": r"{n} \text{{ prime}}", "finite": r"{n} \text{{ finite}}",
}


def _hold_ranges(args):
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if isinstance(a, (tuple, list)):
            out.append(tuple(a))
            i += 1
        elif isinstance(a, sp.Symbol) and len(args) - i >= 3 and not isinstance(args[i + 1], sp.Symbol):
            out.append((a, args[i + 1], args[i + 2]))
            i += 3
        else:
            out.append(a)
            i += 1
    return out


def grade(answer, reference) -> tuple[bool, str]:
    """Is the student's value the same as the reference? (correct, feedback)."""
    def as_list(v):
        if isinstance(v, (list, tuple, set, frozenset)):
            return list(v)
        if isinstance(v, sp.FiniteSet):
            return list(v.args)
        return None

    la, lr = as_list(answer), as_list(reference)
    if la is not None or lr is not None:
        if lr is None:
            return False, "Expected a single value, not a list."
        if la is None:
            return False, f"Expected a list of {len(lr)} value{'s' if len(lr) != 1 else ''}."
        if len(la) != len(lr):
            return False, f"Expected {len(lr)} value{'s' if len(lr) != 1 else ''}, got {len(la)}."
        remaining = list(lr)
        for a in la:
            hit = next((i for i, r in enumerate(remaining) if _values_equal(a, r)[0]), None)
            if hit is None:
                return False, "One of the values is not right."
            remaining.pop(hit)
        return True, "Correct."
    return _values_equal(answer, reference)


def _values_equal(a, b) -> tuple[bool, str]:
    from ..modules.builtin.algebra import numerically_equal

    if isinstance(a, sp.Eq) or isinstance(b, sp.Eq):
        if not isinstance(b, sp.Eq):
            return False, "Expected an expression, not an equation."
        if not isinstance(a, sp.Eq):
            return False, "Expected an equation (write it with ==)."
        da, db = sp.sympify(a.lhs - a.rhs), sp.sympify(b.lhs - b.rhs)
        try:
            ratio = sp.simplify(da / db)
            if ratio.is_number and ratio != 0 and ratio.is_finite:
                return True, "Correct."
        except Exception:  # noqa: BLE001
            pass
        return False, "The equation is not equivalent to the expected one."
    if isinstance(a, (bool, BooleanAtom)) or isinstance(b, (bool, BooleanAtom)):
        return bool(a) == bool(b), "Correct." if bool(a) == bool(b) else "Not right."
    try:
        a, b = sp.sympify(a), sp.sympify(b)
    except (sp.SympifyError, TypeError, ValueError):
        return False, "Expected a mathematical value."
    if isinstance(a, sp.MatrixBase) or isinstance(b, sp.MatrixBase):
        if not (isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase)):
            return False, "Expected a matrix." if isinstance(b, sp.MatrixBase) else "Expected a value, not a matrix."
        if a.shape != b.shape:
            return False, f"Expected a {b.rows}×{b.cols} matrix."
        try:
            same = all(sp.simplify(x - y) == 0 for x, y in zip(a, b))
        except Exception:  # noqa: BLE001
            same = False
        return (True, "Correct.") if same else (False, "Some entries differ.")
    if U.has_units(a) or U.has_units(b):
        try:
            same = U.SI.get_dimension_system().equivalent_dims(U.u.Dimension(U.dimension_of(a)), U.u.Dimension(U.dimension_of(b)))
        except Exception:  # noqa: BLE001
            same = U.dimension_of(a) == U.dimension_of(b)
        if not same:
            return False, f"Check the units: expected a quantity with dimension {U.dimension_of(b)}."
        a, b = U.strip_units(U.strip_angles(U.to_base(a)))[0], U.strip_units(U.strip_angles(U.to_base(b)))[0]
    extra = getattr(a, "free_symbols", set()) - getattr(b, "free_symbols", set())
    if extra:
        return False, f"Your answer still contains {', '.join(sorted(str(x) for x in extra))}."
    if not getattr(a, "free_symbols", set()) and not getattr(b, "free_symbols", set()):
        try:
            fa, fb = complex(sp.N(a)), complex(sp.N(b))
        except (TypeError, ValueError):
            fa = fb = None
        if fa is not None:
            tol = 5e-3 if (a.atoms(sp.Float) or b.atoms(sp.Float)) else 1e-9
            if abs(fa - fb) <= tol * max(1.0, abs(fb)):
                return True, "Correct."
            if abs(fa - fb) <= 0.05 * max(1.0, abs(fb)):
                return False, "Close, but not within rounding: check the arithmetic or the number of digits."
            return False, "That value is not right."
    try:
        if sp.simplify(a - b) == 0:
            return True, "Correct."
    except Exception:  # noqa: BLE001
        pass
    try:
        if numerically_equal(a, b):
            return True, "Correct (a different form, but the same expression)."
    except Exception:  # noqa: BLE001
        pass
    return False, "Not the expected expression."


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
    if isinstance(x, (sp.Integer, sp.Float)):
        return True
    coeff, rest = x.as_coeff_Mul() if isinstance(x, sp.Expr) else (None, None)
    return isinstance(coeff, (sp.Integer, sp.Float)) and rest == sp.I  # plain decimal times i


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
    try:
        lat, plain = fmt_number(num, digits)
    except (TypeError, ValueError):
        return None
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
    def __init__(self, registry, digits: int = 6, loader=None):
        self.registry = registry
        self.loader = loader  # name -> (doc, stamp) for 'import name', or None
        self._import_cache: dict = {}
        self._import_stack: list = []
        self.base_namespace = registry.namespace()
        self.plot_kinds = registry.plot_kinds() if hasattr(registry, "plot_kinds") else {}
        self.units_namespace = {e.name: e.value for m in registry.modules for e in m.entries
                                if e.kind == "unit" or (e.kind == "constant" and isinstance(e.value, sp.Basic)
                                                        and U.has_units(e.value))}
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
        ns.update({k: v for k, v in env.items() if not k.startswith("$")})
        for b in bound:
            ns[b] = sp.Symbol(b, **assumed.get(b, {}))
        for g in GREEK_NAMES:  # 'beta' as a value is the variable beta (a user definition still wins)
            ns[SYM_ALIAS + g] = env[g] if g in env else sp.Symbol(g, **assumed.get(g, {}))
        measured = env.get(MEASURED_KEY) or {}
        if measured:
            ns["nominal"] = lambda e: sp.sympify(e).subs(self.nominals(env))
            ns["uncertainty"] = lambda e: (self.propagate(sp.sympify(e), env) or (None, sp.S.Zero))[1]
        return ns

    # -- measured values: x = 12.3 ± 0.2 m ---------------------------------
    def nominals(self, env: dict) -> dict:
        return {sym: sp.Float(v) for sym, v, _ in (env.get(MEASURED_KEY) or {}).values()}

    def dim_symbols(self, env: dict) -> set:
        """Symbols that stand for a declared dimension or a measured value (their unit travels beside them)."""
        out = {sym for sym, _, _ in (env.get(MEASURED_KEY) or {}).values()}
        out |= {sym for sym, _ in (env.get(DIMS_KEY) or {}).values()}
        return out

    def propagate(self, expr, env: dict):
        """(nominal value, standard uncertainty) by linear propagation, or None if expr has other symbols."""
        measured = env.get(MEASURED_KEY) or {}
        if not measured or not isinstance(expr, sp.Expr):
            return None
        syms = {sym: (v, sg) for sym, v, sg in measured.values() if sym in expr.free_symbols}
        if not syms or (expr.free_symbols - set(syms)):
            return None
        nominal = {sym: sp.Float(v) for sym, (v, _) in syms.items()}
        value = expr.subs(nominal)
        var = sp.S.Zero
        for sym, (_, sg) in syms.items():
            var += (sp.diff(expr, sym).subs(nominal) * sg) ** 2
        num, unit = U.split_units(sp.expand(var))
        try:
            sigma = sp.sqrt(sp.Abs(num)) * sp.sqrt(unit)
        except Exception:  # noqa: BLE001
            return None
        return value, sigma

    def measured_definition(self, st, ns: dict, env: dict):
        """Bind st.name to a symbol whose unit travels beside it; returns the value or None if not measured."""
        m = MEASURED_RE.match(st.body)
        if not m or st.kind != "definition":
            return None
        a, ua, b, ub = m.group("a"), m.group("ua").strip(), m.group("b"), m.group("ub").strip()
        val = parse(f"{a} {ua}" if ua else a, ns, self.unit_names)
        sig = parse(f"{b} {ub}" if ub else b, ns, self.unit_names)
        if not isinstance(val, sp.Expr) or not isinstance(sig, sp.Expr) or val.free_symbols or sig.free_symbols:
            raise QuireError("A measured value is 'number ± number', with units after either, e.g. 12.3 ± 0.2 m.")
        if U.has_units(sig) and not U.has_units(val):
            val = val * U.split_units(sig)[1]
        num, unit = U.split_units(val)
        if unit != 1:
            sig = U.convert(sig, unit) if U.has_units(sig) else sig * unit
            sig_num = float(U.strip_units(U.split_units(sig)[0])[0])
        else:
            if U.has_units(sig):
                raise QuireError("The uncertainty has units but the value has none.")
            sig_num = float(sig)
        sym = sp.Symbol(st.name, real=True)
        env.setdefault(MEASURED_KEY, {})[st.name] = (sym, float(num), abs(sig_num))
        return sym * unit

    def render_measured(self, st, value, env: dict, digits=None):
        """Output for a value that depends on measured quantities: nominal ± uncertainty."""
        pv = self.propagate(value, env)
        if pv is None:
            return None
        nominal, sigma = pv
        digits = digits or self.digits
        vn, vu = U.split_units(nominal)
        sn, su = U.split_units(sigma)
        if not (vn.is_number and sn.is_number):
            return None
        if vu != su:
            try:
                sn = U.strip_units(U.convert(sigma, vu))[0] if vu != 1 else sn
            except Exception:  # noqa: BLE001
                return None
        lv, pv_ = fmt_number(vn, digits)
        ls, ps = fmt_number(sn, 2)
        unit_l = (" \\, " + pretty_units(sp.latex(vu))) if vu != 1 else ""
        unit_p = (" " + to_plain(vu)) if vu != 1 else ""
        out = {"kind": st.kind, "latex": f"{lv} \\pm {ls}{unit_l}", "plain": f"{pv_} ± {ps}{unit_p}",
               "measured": {"value": float(vn), "sigma": float(sn), "unit": to_plain(vu) if vu != 1 else ""}}
        if st.kind in ("definition", "function"):
            out["name"] = st.name
            out["head"] = sp.latex(sp.Symbol(st.name))
        names = sorted(str(s) for s in value.free_symbols)
        if not (st.kind == "definition" and names == [st.name]):
            out["notes"] = [f"uncertainty by linear propagation from {', '.join(names)}"]
        return out

    def parse_unit(self, text: str):
        return parse(text, dict(self.units_namespace))

    # -- document -------------------------------------------------------
    def evaluate(self, cells: list[dict], cache: dict | None = None) -> list[dict]:
        return list(self.iter_evaluate(cells, cache))

    def iter_evaluate(self, cells: list[dict], cache: dict | None = None):
        """Yield one result per cell, top to bottom.

        With a ``cache`` (kept between evaluations by the worker), a cell whose text and
        inputs are unchanged is not re-run: its result is returned again and the changes it
        made to the environment are replayed. The key of a cell is its text plus the value
        of every name it mentions (or the fact that the name is undefined) plus the
        settings in force (digits, assumptions, measured values), so a change anywhere
        above that matters to the cell invalidates it. Each result carries the time it took
        ("ms") and whether it came from the cache ("cached").
        """
        import time

        from .plotting import sample_plot

        env: dict = {}
        ids = set()
        for cell in cells:
            kind = cell.get("type", "math")
            cid = cell.get("id")
            ids.add(cid)
            key = self._cell_key(cell, env) if cache is not None else None
            entry = cache.get(cid) if key is not None else None
            if entry is not None and self._key_equal(entry["key"], key):
                self._replay(env, entry["delta"])
                yield dict(entry["result"], cached=True, ms=0)
                continue
            before = dict(env)
            trace_len = len(env.get("$trace", []))
            t0 = time.perf_counter()
            if kind == "text":
                res = {"id": cid, **self.evaluate_text(cell.get("source", ""), env)}
            elif kind == "check":
                res = {"id": cid, **self.evaluate_check(cell, env)}
            elif kind == "plot":
                res = {"id": cid, **sample_plot(cell, env, self)}
            else:
                res = {"id": cid, **self.evaluate_math(cell.get("source", ""), env)}
            res["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if key is not None:
                cache[cid] = {"key": key, "result": dict(res), "delta": self._delta(before, env, trace_len)}
            elif cache is not None:
                cache.pop(cid, None)
            yield res
        if cache is not None:
            for stale in [k for k in cache if k not in ids]:
                del cache[stale]

    def _cell_key(self, cell: dict, env: dict):
        """(static part, [(name, value or None)]) or None when the cell must always run."""
        kind = cell.get("type", "math")
        if kind == "text":
            source = cell.get("source", "")
            if "{{" not in source:
                return (("text", source), [])
            texts = placeholders(source)
            static = ("text", source)
        elif kind == "check":
            static = ("check", cell.get("prompt", ""), cell.get("reference", ""), cell.get("answer", ""), cell.get("hint", ""))
            texts = [cell.get("reference", ""), cell.get("answer", "")]
        elif kind == "plot":
            fields = ("kind", "exprs", "expr2", "expr3", "var", "xmin", "xmax", "ymin", "ymax", "samples", "annot")
            static = ("plot",) + tuple(str(cell.get(f) or "") for f in fields)
            texts = [str(cell.get(f) or "") for f in ("exprs", "expr2", "expr3", "xmin", "xmax", "ymin", "ymax", "annot")]
        else:
            source = cell.get("source", "")
            if any(IMPORT_RE.match(ln) for ln in source.split("\n")):
                return None
            static = ("math", source)
            texts = [source]
        names = set()
        for t in texts:
            try:
                names |= identifiers(t) | set(called_names(t))
            except Exception:  # noqa: BLE001 - unparsable text: still keyed on its source
                pass
        values = [(n, env.get(n)) for n in sorted(names)]
        context = (env.get(DIGITS_KEY), repr(env.get(ASSUME_KEY)), repr(env.get(BOUNDS_KEY)),
                   repr({k: v[1:] for k, v in env.get(MEASURED_KEY, {}).items()}), repr(env.get(DIMS_KEY)),
                   self._data_stamp())
        return (static + context, values)

    @staticmethod
    def _key_equal(a, b) -> bool:
        if a[0] != b[0] or len(a[1]) != len(b[1]):
            return False
        return all(n1 == n2 and _same(v1, v2) for (n1, v1), (n2, v2) in zip(a[1], b[1]))

    @staticmethod
    def _data_stamp():
        """Change marker for uploaded data files (read_csv and friends read them from disk)."""
        import os

        d = os.environ.get("QUIRE_WORKSHEETS")
        try:
            return os.stat(os.path.join(d, "data")).st_mtime_ns if d else None
        except OSError:
            return None

    @staticmethod
    def _delta(before: dict, env: dict, trace_len: int) -> dict:
        import copy

        delta = {"set": {k: v for k, v in env.items() if not k.startswith("$") and (k not in before or before[k] is not v)},
                 "special": {}, "trace": list(env.get("$trace", [])[trace_len:])}
        for k in SPECIAL_COPY:
            if k in env and (k not in before or before[k] != env[k] or isinstance(env[k], dict)):
                delta["special"][k] = copy.deepcopy(env[k]) if k != MEASURED_KEY else dict(env[k])
        return delta

    @staticmethod
    def _replay(env: dict, delta: dict) -> None:
        import copy

        env.update(delta["set"])
        for k, v in delta["special"].items():
            env[k] = copy.deepcopy(v) if k != MEASURED_KEY else dict(v)
        if delta["trace"]:
            env.setdefault("$trace", []).extend(delta["trace"])

    # -- check cells: grade an answer against the author's reference ------
    def _scratch(self, env: dict) -> dict:
        scratch = dict(env)
        scratch["$trace"] = list(env.get("$trace", []))
        return scratch

    def _value_of(self, text: str, env: dict):
        r = self.evaluate_math(text, self._scratch(env))
        if not r["ok"]:
            raise QuireError(r["error"])
        if not self.last_values:
            raise QuireError("Nothing to evaluate.")
        v = self.last_values[-1]
        return v.result if isinstance(v, Steps) else v

    def evaluate_check(self, cell: dict, env: dict) -> dict:
        reference = (cell.get("reference") or "").strip()
        answer = (cell.get("answer") or "").strip()
        out = {"ok": True, "correct": None, "feedback": ""}
        if not reference:
            out["feedback"] = "No reference answer yet: open the author section and fill it in."
            return out
        if not answer:
            return out
        try:
            ref = self._value_of(reference, env)
        except QuireError as exc:
            return {"ok": False, "correct": None, "feedback": "", "error": f"The reference answer does not evaluate: {exc}"}
        try:
            ans = self._value_of(answer, env)
        except QuireError as exc:
            out.update(correct=False, feedback=f"I could not read the answer: {exc}")
            return out
        correct, feedback = grade(ans, ref)
        out.update(correct=correct, feedback=feedback)
        try:
            out["answer_latex"] = to_latex(self.display_value(ans, env.get(DIGITS_KEY)))
        except Exception:  # noqa: BLE001
            pass
        if not correct and (cell.get("hint") or "").strip():
            out["hint"] = cell["hint"].strip()
        return out

    # -- text cells: {{expr}} placeholders --------------------------------
    def evaluate_text(self, source: str, env: dict) -> dict:
        """Values for the {{expr}} placeholders of a text cell (outside $...$ math), in order."""
        if "{{" not in source:
            return {"ok": True}
        values = []
        for expr in placeholders(source):
            r = self.evaluate_math(expr, self._scratch(env))  # a copy: a placeholder never defines anything
            if r["ok"] and r["outputs"]:
                o = r["outputs"][-1]
                values.append({"latex": o.get("latex", ""), "plain": o.get("plain", ""), "approx": o.get("approx"),
                               "approx_plain": o.get("approx_plain")})
            else:
                values.append({"error": r["error"] or "No value."})
        return {"ok": True, "values": values}

    # -- import name: definitions of another worksheet ------------------
    def import_worksheet(self, name: str, env: dict) -> list[str]:
        if self.loader is None:
            raise QuireError("Importing worksheets is not available here.")
        if name in self._import_stack:
            raise QuireError(f"'{name}' imports itself (through {' -> '.join(self._import_stack + [name])}).")
        if len(self._import_stack) >= 8:
            raise QuireError("Imports are nested too deeply.")
        loaded = self.loader(name)
        if loaded is None:
            raise QuireError(f"No worksheet named '{name}'. Use the name it was saved under.")
        doc, stamp = loaded
        cached = self._import_cache.get(name)
        if cached is None or cached[0] != stamp:
            self._import_stack.append(name)
            try:
                sub: dict = {}
                names: list[str] = []
                for cell in doc.get("cells", []):
                    if cell.get("type", "math") != "math":
                        continue
                    r = self.evaluate_math(cell.get("source", ""), sub)
                    if not r["ok"]:
                        raise QuireError(f"In '{name}': {r['error']}")
                    names.extend(n for n in r["defines"] if n not in names)
            finally:
                self._import_stack.pop()
            cached = (stamp, {n: sub[n] for n in names if n in sub}, dict(sub.get(ASSUME_KEY, {})))
            self._import_cache[name] = cached
        _, defs, assumed = cached
        env.update(defs)
        if assumed:
            env.setdefault(ASSUME_KEY, {}).update({k: v for k, v in assumed.items() if k in defs})
        return list(defs)

    # -- one math cell --------------------------------------------------
    def evaluate_math(self, source: str, env: dict) -> dict:
        from ..modules import hooks

        hooks.context["bounds"] = env.setdefault(BOUNDS_KEY, {})  # same dict the assume lines fill in
        outputs, defines, uses = [], [], set()
        error = warning = None
        self.last_values = []  # raw values of each output line (used by the benchmark)
        for line_index, line in enumerate(source.split("\n")):
            if not line.strip():
                continue
            try:
                hooks.context["notes"] = []
                hooks.context.pop("slider", None)
                m = IMPORT_RE.match(line)
                if m:
                    name = m.group(1).strip().strip("\"'")
                    names = self.import_worksheet(name, env)
                    defines.extend(n for n in names if n not in defines)
                    shown = ", ".join(names) if names else "nothing"
                    outputs.append({"kind": "import", "latex": r"\text{imported " + shown.replace("_", r"\_") + " from " + name.replace("_", r"\_") + "}",
                                    "plain": f"imported {shown} from {name}", "imported": names, "worksheet": name})
                    continue
                st = classify(line)
                if st.kind == "assume":
                    out = self.assume(st, env)
                    defines.extend(n for n in out.pop("$defines", []) if n not in defines)
                    outputs.append(out)
                    continue
                if st.kind == "digits":
                    env[DIGITS_KEY] = int(st.body)
                    outputs.append({"kind": "setting", "latex": r"\text{digits: }" + st.body, "plain": f"digits {st.body}"})
                    continue
                ns = self.namespace(env, st.params)
                # names used, ignoring unit phrases such as "3 m" even when m is also defined
                uses |= (identifiers(alias_units(st.body, self.unit_names)) - set(st.params)) & set(env) - {ASSUME_KEY, BOUNDS_KEY}
                measured = self.measured_definition(st, ns, env)
                value = measured if measured is not None else parse(st.body, ns, self.unit_names)
                if measured is not None:
                    value = self.finalize(value, st.convert_to, env=env)
                elif st.kind == "function":
                    # The body is checked and converted when called: parameters may carry units.
                    if not isinstance(value, sp.Expr):
                        raise QuireError("A function body must be a single expression.")
                    target = self.parse_unit(st.convert_to) if st.convert_to else None
                    value = DefinedFunction(st.name, st.params, value, target, st.convert_to,
                                            symbols=[ns[p] for p in st.params])
                elif isinstance(value, Steps):
                    value.result = self.finalize(value.result, st.convert_to, env=env)
                else:
                    value = self.finalize(value, st.convert_to, env=env)
                if st.kind in ("definition", "function") and st.name in self.unit_names and len(st.name) > 1:
                    warning = (f"'{st.name}' now means your definition. Directly after a number it still "
                               f"means the unit ({U.UNIT_TABLE[st.name][1]}), as in '3 {st.name}'; "
                               f"write '3*{st.name}' for three times yours.")
                if st.kind in ("definition", "function"):
                    env[st.name] = value.result if isinstance(value, Steps) else value
                    defines.append(st.name)
                self.last_values.append(value)
                out = (self.render_measured(st, value, env, env.get(DIGITS_KEY)) if env.get(MEASURED_KEY) and isinstance(value, sp.Expr) else None) \
                    or self.render(st, value, env.get(DIGITS_KEY))
                if uses_notation(st.source):
                    reading = self.reading(st, ns)
                    if reading:
                        out["reading"] = reading
                if hooks.context.get("notes"):
                    out["notes"] = out.get("notes", []) + list(hooks.context["notes"])
                if hooks.context.get("slider") and st.kind == "definition":
                    out["slider"] = dict(hooks.context["slider"], line=line_index, name=st.name)
                    env.setdefault("$sliders", {})[st.name] = hooks.context["slider"]["value"]
                outputs.append(out)
            except QuireError as exc:
                error = str(exc)
                break
            except RecursionError:
                error = "The expression is too deeply nested."
                break
            except Exception as exc:  # noqa: BLE001 - surface anything sympy throws
                error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
                break
        if defines:  # remembered so plots can re-run a dependency chain with sliders held symbolic
            env.setdefault("$trace", []).append({"source": source, "defines": list(defines), "uses": sorted(uses)})
        return {"ok": error is None, "outputs": outputs, "defines": defines, "uses": sorted(uses), "error": error,
                "warning": warning}

    def reading(self, st, ns: dict):
        """LaTeX of how a line written in paper notation was read, with integrals and sums left unevaluated."""
        try:
            text = normalize(st.body)
            held = dict(ns)
            held["integrate"] = lambda f, *a: sp.Integral(f, *_hold_ranges(a))
            held["sum"] = lambda f, k, a, b: sp.Sum(f, (k, a, b))
            held["product"] = lambda f, k, a, b: sp.Product(f, (k, a, b))
            held["diff"] = lambda f, *a: sp.Derivative(f, *a)
            held["sqrt"] = lambda x: sp.Pow(x, sp.S.Half, evaluate=False)
            expr = parse(text, held, self.unit_names)
            latex = to_latex(expr)
            if st.kind in ("definition", "function"):
                head = sp.latex(sp.Symbol(st.name))
                if st.kind == "function":
                    head += r"\left(" + ", ".join(sp.latex(sp.Symbol(p)) for p in st.params) + r"\right)"
                latex = head + " = " + latex
            if st.convert_to:
                latex += r"\ \rightarrow\ " + sp.latex(self.parse_unit(st.convert_to))
            return latex
        except Exception:  # noqa: BLE001
            return None

    def assume(self, st, env: dict) -> dict:
        assumed = env.setdefault(ASSUME_KEY, {})
        bounds = env.setdefault(BOUNDS_KEY, {})
        dims = {}
        for n in st.names:
            spec = dict(st.assumptions[n])
            bound = spec.pop("$bound", None)
            dim = spec.pop("$dim", None)
            if dim is not None:
                unit = U.DIMENSION_UNITS.get(dim)
                if unit is None:
                    unit = self.parse_unit(dim)
                dims[n] = (dim, unit)
            if bound is not None:
                bounds.setdefault(n, []).append(bound)
            merged = {**assumed.get(n, {}), **spec}
            try:
                sp.Symbol(n, **merged)
            except Exception as exc:  # noqa: BLE001 - inconsistent assumptions
                raise QuireError(f"Assumptions on '{n}' contradict each other: {exc}") from None
            assumed[n] = merged
        parts, plain = [], []
        for n, (dim, unit) in dims.items():
            sym = sp.Symbol(n, **assumed.get(n, {}))
            env[n] = sym * unit
            env.setdefault(DIMS_KEY, {})[n] = (sym, unit)
        for n in st.names:
            head = sp.latex(sp.Symbol(n))
            if n in dims:
                dim, unit = dims[n]
                label = U.unit_label(unit) if unit != 1 else "1"
                parts.append(f"{head} : \\text{{{dim}}}" + (f"\\ [{pretty_units(sp.latex(unit))}]" if unit != 1 and dim != label else ""))
                plain.append(f"{n} {dim}")
                continue
            bound = st.assumptions[n].get("$bound")
            if bound is not None:
                op = {">=": r"\geq", "<=": r"\leq", "!=": r"\neq"}.get(bound[0], bound[0])
                parts.append(f"{head} {op} {sp.latex(bound[1])}")
                plain.append(f"{n} {bound[0]} {bound[1]}")
                continue
            for key in st.assumptions[n]:
                parts.append(_ASSUMPTION_LATEX.get(key, "{n}").format(n=head))
            plain.append(f"{n} {' '.join(st.assumptions[n])}")
        out = {"kind": "assume", "latex": ",\\ ".join(parts), "plain": "; ".join(plain)}
        if dims:
            out["$defines"] = list(dims)
        return out

    def finalize(self, value, convert_to: str | None, check: bool = True, env: dict | None = None):
        if isinstance(value, (list, tuple)):
            return type(value)(self.finalize(v, convert_to, check, env) for v in value)
        if isinstance(value, (int, float, complex, bool)):
            value = sp.sympify(value)
        if isinstance(value, sp.Expr):
            if check:
                U.check_dimensions(value, self.dim_symbols(env) if env else ())
            if convert_to:
                target = self.parse_unit(convert_to)
                value = U.convert(value, target)
            else:
                value = U.tidy_units(value)
        elif convert_to:
            raise QuireError("'->' conversion needs a single quantity on the left.")
        return value

    def display_value(self, value, digits=None):
        """What to show: inexact inputs (decimals) give a rounded numeric result.

        With an explicit 'digits n' in the worksheet, plain decimals are rounded to n
        significant digits too; otherwise they keep the precision they were computed with.
        """
        explicit = digits is not None
        digits = digits or self.digits
        if isinstance(value, (list, tuple)):
            return type(value)(self.display_value(v, digits if explicit else None) for v in value)
        if isinstance(value, sp.MatrixBase) and explicit and value.atoms(sp.Float):
            return value.applyfunc(lambda e: self.display_value(e, digits))
        if isinstance(value, sp.Expr) and not isinstance(value, (sp.Lambda, DefinedFunction, BooleanAtom)) \
                and not value.free_symbols and value.atoms(sp.Float):
            num, unit = U.split_units(value)
            if isinstance(num, sp.Float) and not explicit:
                return value  # keep the precision it was computed with
            if isinstance(num, sp.Expr) and not num.is_number:
                return value
            try:
                num = sp.N(num, digits + 3)
                if num.is_number and num.is_real:
                    return sp.Float(float(num), digits) * unit if float(num) != 0 else sp.S.Zero * unit
            except (TypeError, ValueError):
                return value
        return value

    def render(self, st, value, digits=None) -> dict:
        if isinstance(value, Steps):
            out = self.render(st, value.result, digits)
            out["steps"] = value.describe()
            if value.title:
                out["steps_title"] = value.title
            return out
        shown = self.display_value(value, digits)
        out = {"kind": st.kind, "latex": to_latex(shown), "plain": to_plain(shown)}
        if st.kind in ("definition", "function"):
            out["name"] = st.name
            head = sp.latex(sp.Symbol(st.name))
            if st.kind == "function":
                head += r"\left(" + ", ".join(sp.latex(sp.Symbol(p)) for p in st.params) + r"\right)"
            out["head"] = head
        approx = approx_of(shown.expr if isinstance(shown, (sp.Lambda, DefinedFunction)) else shown, digits or self.digits)
        if approx:
            out["approx"], out["approx_plain"] = approx
        if isinstance(shown, sp.Expr) and not shown.free_symbols and not U.has_units(shown) and shown.is_real:
            try:
                out["num"] = float(shown)  # lets the browser preview cheap arithmetic before the server answers
            except (TypeError, ValueError):
                pass
        return out
