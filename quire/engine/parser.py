"""Turn worksheet text into sympy expressions.

The input language is math notation, not a programming language:

    3 kg * 2 m/s^2          implicit multiplication, ^ for powers
    F = m a                 definition (top-down, like a worksheet)
    f(x) = x^2 + 1          function definition
    solve(x^2 == 4, x)      == builds an equation
    F -> N                  unit conversion of the result
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from .errors import ParseError

TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication, implicit_application)

# Names the parser's generated code needs. Nothing from Python builtins leaks in.
GLOBAL_DICT = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Function": sp.Function,
    "Eq": sp.Eq,
    "Lambda": sp.Lambda,
    "factorial": sp.factorial,
}

IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
PRIME_RE = re.compile(rf"(?<![A-Za-z0-9_.])({IDENT})('+)\s*\(")
INDEX_RE = re.compile(rf"(?<![A-Za-z0-9_.\]])({IDENT})\[")
ASSUME_REL_RE = re.compile(r"^\s*(.+?)\s*(>=|<=|>|<|!=)\s*(-?[0-9][0-9./]*|-?pi|-?[0-9./]*\s*pi)\s*$")
ASSUME_WORD_RE = re.compile(r"^\s*(.+?)\s+((?:[a-z]+\s*)+)$")
ASSUMPTION_WORDS = {
    "real": {"real": True}, "positive": {"positive": True}, "negative": {"negative": True},
    "nonnegative": {"nonnegative": True}, "nonpositive": {"nonpositive": True}, "nonzero": {"nonzero": True},
    "integer": {"integer": True}, "rational": {"rational": True}, "irrational": {"irrational": True},
    "even": {"even": True}, "odd": {"odd": True}, "prime": {"prime": True}, "complex": {"complex": True},
    "finite": {"finite": True},
}
ASSUMPTION_RELATIONS = {">": {"positive": True}, ">=": {"nonnegative": True}, "<": {"negative": True},
                        "<=": {"nonpositive": True}, "!=": {"nonzero": True}}
UNIT_ALIAS = "qunit_"
SYM_ALIAS = "qsym_"
# Names that are functions (beta, gamma, zeta) or Python keywords (lambda) but that people use as variables.
GREEK_NAMES = ("beta", "gamma", "zeta", "lambda")
GREEK_RE = re.compile(r"(?<![A-Za-z0-9_.])(" + "|".join(GREEK_NAMES) + r")\b(?!\s*\()")
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
PHRASE_RE = re.compile(rf"(\s*/\s*|\s*)({IDENT})(?!\s*\()(\s*\^\s*(?:-?\d+|\(\s*-?\d+\s*\)))?")
DEF_RE = re.compile(rf"^\s*({IDENT})\s*(?:\(\s*({IDENT}(?:\s*,\s*{IDENT})*)\s*\))?\s*=(?!=)(.*)$", re.S)
CALL_RE = re.compile(rf"(?<![A-Za-z0-9_.])({IDENT})\(")
IDENT_RE = re.compile(rf"(?<![A-Za-z0-9_.])({IDENT})(?![A-Za-z0-9_(])")
FORBIDDEN = [
    (re.compile(r"__"), "double underscores are not allowed"),
    (re.compile(r"\.\s*[A-Za-z_]"), "attribute access is not allowed"),
    (re.compile(r"!="), "'!=' is not supported; use solve/assume with == or < >"),
    (re.compile(r"[\"';:\\@#$%&{}|~`?]"), "unexpected character"),
    (re.compile(r"\b(import|lambda|def|class|return|yield|for|while|if|else|in|is|not|and|or|None|True|False)\b"),
     "reserved word"),
]

_OPEN = {"(": ")", "[": "]"}
_CLOSE = {")": "(", "]": "["}


def _check_source(src: str) -> None:
    for rx, why in FORBIDDEN:
        m = rx.search(src)
        if m:
            raise ParseError(f"'{m.group(0).strip()}': {why}.")
    depth = []
    for ch in src:
        if ch in _OPEN:
            depth.append(ch)
        elif ch in _CLOSE:
            if not depth or depth[-1] != _CLOSE[ch]:
                raise ParseError(f"Unbalanced '{ch}'.")
            depth.pop()
    if depth:
        raise ParseError(f"Missing closing '{_OPEN[depth[-1]]}'.")


def split_top(s: str, sep: str) -> list[str]:
    """Split on ``sep`` at bracket depth 0."""
    parts, depth, start, i, n = [], 0, 0, 0, len(sep)
    for j, ch in enumerate(s):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        elif depth == 0 and s.startswith(sep, j) and (i <= j):
            parts.append(s[start:j])
            start = j + n
            i = start
    parts.append(s[start:])
    return parts


def _match(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] in _OPEN:
            depth += 1
        elif s[j] in _CLOSE:
            depth -= 1
            if depth == 0:
                return j
    raise ParseError("Missing closing bracket.")


def rewrite_equalities(src: str) -> str:
    """Rewrite ``a == b`` into ``Eq(a, b)`` at every bracket level."""

    def process(s: str) -> str:
        out = []
        for part in split_top(s, ","):
            part = recurse(part)
            sides = split_top(part, "==")
            if len(sides) == 1:
                out.append(part)
            elif len(sides) == 2:
                lead = part[: len(part) - len(part.lstrip())]
                out.append(f"{lead}Eq({sides[0].strip()}, {sides[1].strip()})")
            else:
                raise ParseError("Chained '==' is not supported.")
        return ",".join(out)

    def recurse(s: str) -> str:
        out, i = [], 0
        while i < len(s):
            ch = s[i]
            if ch in _OPEN:
                j = _match(s, i)
                out.append(ch + process(s[i + 1:j]) + _OPEN[ch])
                i = j + 1
            else:
                out.append(ch)
                i += 1
        return "".join(out)

    return process(src)


def split_conversion(src: str) -> tuple[str, str | None]:
    """``expr -> unit`` becomes (expr, unit)."""
    parts = split_top(src, "->")
    if len(parts) == 1:
        return src, None
    if len(parts) > 2:
        raise ParseError("Only one '->' conversion per expression.")
    if not parts[1].strip():
        raise ParseError("Write a unit after '->', e.g. '-> km/hr'.")
    return parts[0], parts[1].strip()


def rewrite_greek(src: str) -> str:
    """beta, gamma, zeta, lambda not followed by '(' are variables, not functions or keywords."""
    return GREEK_RE.sub(lambda m: SYM_ALIAS + m.group(1), src)


def rewrite_primes(src: str) -> str:
    """f'(x) -> dprime_(f, 1, x); f''(x) -> dprime_(f, 2, x)."""
    return PRIME_RE.sub(lambda m: f"dprime_({m.group(1)}, {len(m.group(2))}, ", src)


def rewrite_indexes(src: str, namespace: dict) -> str:
    """a[n] for a name that is not a defined list -> seq_(a, n), an indexed sequence term."""
    out, pos = [], 0
    while True:
        m = INDEX_RE.search(src, pos)
        if not m:
            out.append(src[pos:])
            return "".join(out)
        name = m.group(1)
        if name in namespace and not isinstance(namespace[name], sp.Symbol):
            out.append(src[pos:m.end()])
            pos = m.end()
            continue
        j = _match(src, m.end() - 1)
        out.append(src[pos:m.start()] + f"seq_({name}, {src[m.end():j]})")
        pos = j + 1


def alias_units(src: str, unit_names) -> str:
    """After a number, a run of unit names always means units: ``3 m/s^2``.

    Each such name is rewritten to its ``qunit_`` alias so a worksheet variable
    with the same name (``m = 2 kg``) does not capture it. ``2*m`` is left
    alone: an explicit ``*`` means "the thing named m". A number that is a
    divisor or an exponent (``1/2 g t^2``, ``x^2 m``) does not start a unit
    phrase either, so a user-defined ``g`` survives there.
    """
    out, pos = [], 0
    while True:
        m = NUM_RE.search(src, pos)
        if not m:
            out.append(src[pos:])
            return "".join(out)
        out.append(src[pos:m.end()])
        p = m.end()
        before = src[: m.start()].rstrip()
        if before and before[-1] in "/^":
            pos = p
            continue
        first = True
        while True:
            pm = PHRASE_RE.match(src, p)
            if not pm or pm.group(2) not in unit_names:
                break
            if first and "/" in pm.group(1):
                break  # "2/s^3" is a division by the variable s, not "per second"
            first = False
            out.append(pm.group(1) + UNIT_ALIAS + pm.group(2) + (pm.group(3) or ""))
            p = pm.end()
        pos = p


def identifiers(src: str) -> set[str]:
    return set(IDENT_RE.findall(src))


def called_names(src: str) -> set[str]:
    return set(CALL_RE.findall(src))


def parse(src: str, namespace: dict, unit_names=()) -> sp.Basic:
    """Parse one expression with the given name -> value namespace."""
    src = src.strip()
    if not src:
        raise ParseError("Empty expression.")
    src = rewrite_greek(rewrite_primes(src))
    _check_source(src)
    src = re.sub(r"\]\s+(?=[A-Za-z_(])", "] * ", src)  # "a[n] b[n]" is a product
    src = rewrite_indexes(src, namespace)
    if unit_names:
        src = alias_units(src, unit_names)
    for name in called_names(src):
        value = namespace.get(name)
        if value is None:
            raise ParseError(
                f"'{name}' is not a known function. For multiplication write '{name} * (...)' or '{name} (...)'."
            )
        if not callable(value):
            raise ParseError(f"'{name}' is a value, not a function. For multiplication write '{name} * (...)'.")
    code = rewrite_equalities(src)
    try:
        return parse_expr(code, local_dict=namespace, global_dict=GLOBAL_DICT, transformations=TRANSFORMATIONS,
                          evaluate=True)
    except ParseError:
        raise
    except (SyntaxError, TokenError_):
        raise ParseError(f"Could not read '{src}'.") from None
    except TypeError as exc:
        raise ParseError(_friendly_type_error(str(exc), src)) from None
    except Exception as exc:  # sympy raises many things; show the message.
        msg = str(exc).split("\n")[0]
        raise ParseError(msg or f"Could not read '{src}'.") from None


try:
    from tokenize import TokenError as TokenError_
except ImportError:  # pragma: no cover
    TokenError_ = SyntaxError


def _friendly_type_error(msg: str, src: str) -> str:
    if "positional argument" in msg or "arguments" in msg:
        return "Wrong number of arguments: " + msg
    if "not callable" in msg:
        return "Something is used as a function that is not one: " + msg
    return f"Could not evaluate '{src}': {msg}"


class Statement:
    """A parsed worksheet line."""

    kind: str  # "definition" | "function" | "expression" | "assume"

    def __init__(self, kind: str, name: str | None, params: list[str], body: str, convert_to: str | None,
                 names: list[str] = (), assumptions: dict | None = None):
        self.kind = kind
        self.name = name
        self.params = params
        self.body = body
        self.convert_to = convert_to
        self.names = list(names)
        self.assumptions = assumptions or {}


def _assume(line: str) -> Statement | None:
    """assume x > 0 | assume n positive integer | assume x, y real | assume x > 0, y != 0"""
    if not re.match(r"^\s*assume\b", line):
        return None
    body = re.sub(r"^\s*assume\s*", "", line)
    names: list[str] = []
    assumptions: dict = {}  # name -> {sympy assumption: True}
    pending: list[str] = []

    def take_names(text: str) -> list[str]:
        out = [n for n in text.split() if n]
        for n in out:
            if not re.fullmatch(IDENT, n):
                raise ParseError(f"'{n}' is not a variable name; assume works on single variables, "
                                 f"e.g. 'assume a > 0' (write 'assume a > b' as a definition instead).")
        return out

    for clause in body.split(","):
        clause = clause.strip()
        if not clause:
            continue
        if re.fullmatch(IDENT, clause):
            pending.append(clause)
            continue
        m = ASSUME_REL_RE.match(clause)
        if m:
            op, bound = m.group(2), m.group(3).replace(" ", "")
            val = sp.sympify(bound.replace("pi", "*pi").lstrip("*")) if "pi" in bound else sp.Rational(bound) \
                if "." not in bound else sp.Float(bound)
            if val == 0:
                spec = dict(ASSUMPTION_RELATIONS[op])
            elif op in (">", ">=") and val > 0:
                spec = {"positive": True}
            elif op in ("<", "<=") and val < 0:
                spec = {"negative": True}
            elif op == "!=":
                spec = {}
            else:
                spec = {"real": True}
            these = take_names(m.group(1))
            if val != 0:
                spec["$bound"] = (op, val)
        else:
            m = ASSUME_WORD_RE.match(clause)
            if not m:
                raise ParseError("Write e.g. 'assume x > 0', 'assume n integer' or 'assume x, y real'.")
            these, spec = take_names(m.group(1)), {}
            for w in m.group(2).split():
                if w not in ASSUMPTION_WORDS:
                    raise ParseError(f"Unknown assumption '{w}'. Use: {', '.join(sorted(ASSUMPTION_WORDS))}.")
                spec.update(ASSUMPTION_WORDS[w])
        for n in pending + these:
            names.append(n)
            merged = {**assumptions.get(n, {}), **spec}
            if "$bound" in spec:
                merged["$bound"] = spec["$bound"]
            assumptions[n] = merged
        pending = []
    if pending or not names:
        raise ParseError("Write e.g. 'assume x > 0', 'assume n integer' or 'assume x, y real'.")
    return Statement("assume", None, [], "", None, names, assumptions)


def classify(line: str) -> Statement:
    """Decide whether a line defines something, and split off a '->' conversion."""
    st = _assume(line)
    if st is not None:
        return st
    m = DEF_RE.match(line)
    if m:
        name, params, rhs = m.group(1), m.group(2), m.group(3)
        body, conv = split_conversion(rhs)
        if not body.strip():
            raise ParseError(f"'{name} =' needs a right-hand side.")
        if params:
            plist = [p.strip() for p in params.split(",")]
            if len(set(plist)) != len(plist):
                raise ParseError("Function parameters must be distinct.")
            return Statement("function", name, plist, body, conv)
        return Statement("definition", name, [], body, conv)
    body, conv = split_conversion(line)
    return Statement("expression", None, [], body, conv)
