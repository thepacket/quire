"""Paper-style and LaTeX notation, rewritten to the worksheet's ASCII form before parsing.

    ∫_0^1 x^2 dx          integrate(x^2, x, 0, 1)
    ∫ sin(x) dx           integrate(sin(x), x)
    Σ_{k=1}^{n} k^2       sum(k^2, k, 1, n)
    ∏_{k=1}^{n} k         product(k, k, 1, n)
    d/dx sin(x)           diff(sin(x), x)
    d²/dx² f(x)           diff(f(x), x, 2)
    √(x + 1), √x, x², π, θ, ∞, ≤, ≥, ≠, ×, ÷, ·, ½
    \\frac{a}{b}, \\sqrt{x}, \\int_a^b f\\,dx, \\sum_{k=1}^{n}, \\alpha, \\cdot, \\le, \\infty, \\left( \\right)
"""
from __future__ import annotations

import re

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma",
    "τ": "tau", "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Ξ": "Xi", "Π": "Pi_", "Φ": "Phi",
    "Ψ": "Psi", "Ω": "Omega",
}
SUPERSCRIPTS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
SIMPLE = {"×": "*", "·": "*", "÷": "/", "−": "-", "≤": "<=", "≥": ">=", "≠": "!=", "∞": "oo", "½": "(1/2)", "¼": "(1/4)",
          "¾": "(3/4)", "→": "->", "⁄": "/"}
LATEX_SIMPLE = {
    r"\cdot": "*", r"\times": "*", r"\div": "/", r"\le": "<=", r"\leq": "<=", r"\ge": ">=", r"\geq": ">=", r"\ne": "!=",
    r"\neq": "!=", r"\infty": " oo ", r"\pi": " pi ", r"\left": "", r"\right": "", r"\,": " ", r"\;": " ", r"\!": "",
    r"\quad": " ", r"\mathrm": "", r"\text": "", r"\operatorname": "", r"\displaystyle": "",
}
NOTATION_CHARS = set("∫∑Σ∏√π∞≤≥≠×÷·²³⁴⁵⁶⁷⁸⁹⁰⁻₀₁₂₃₄₅₆₇₈₉½¼¾θαβγδεζηικλμνξρστυφϕχψωΓΔΘΛΞΦΨΩ−→⁄\\{}")
IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


def uses_notation(src: str) -> bool:
    return any(c in NOTATION_CHARS for c in src) or bool(re.search(r"\bd\^?\d?/d[A-Za-z]", src))


def _matching(s: str, i: int, open_: str, close: str) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == open_:
            depth += 1
        elif s[j] == close:
            depth -= 1
            if depth == 0:
                return j
    return -1


def _braced(s: str, i: int):
    """Read a bound at position i: {…}, a parenthesised group, or a single token. Returns (text, next index)."""
    if i >= len(s):
        return "", i
    if s[i] == "{":
        j = _matching(s, i, "{", "}")
        return (s[i + 1:j], j + 1) if j > 0 else (s[i + 1:], len(s))
    if s[i] == "(":
        j = _matching(s, i, "(", ")")
        return (s[i:j + 1], j + 1) if j > 0 else (s[i:], len(s))
    m = re.match(r"-?[A-Za-z0-9_.]+", s[i:])
    if m:
        return m.group(0), i + m.end()
    return "", i


def _latex(src: str) -> str:
    """LaTeX constructs to plain notation (fractions, roots, Greek, operators)."""
    s = src
    s = re.sub(r"\\frac\s*\{\s*d(?:\^(\d))?\s*\}\s*\{\s*d([A-Za-z][A-Za-z0-9_]*)(?:\^(\d))?\s*\}",
               lambda m: f"d{'^' + m.group(1) if m.group(1) else ''}/d{m.group(2)}{'^' + m.group(3) if m.group(3) else ''} ", s)
    # \frac{a}{b} -> ((a)/(b)), innermost first
    while True:
        m = re.search(r"\\d?frac\s*\{", s)
        if not m:
            break
        i = m.end() - 1
        j = _matching(s, i, "{", "}")
        if j < 0:
            break
        k = s.find("{", j + 1)
        if k < 0:
            break
        l = _matching(s, k, "{", "}")
        if l < 0:
            break
        s = s[:m.start()] + f"(({s[i + 1:j]})/({s[k + 1:l]}))" + s[l + 1:]
    # \sqrt[n]{x} and \sqrt{x}
    while True:
        m = re.search(r"\\sqrt\s*(\[([^\]]*)\])?\s*\{", s)
        if not m:
            break
        i = m.end() - 1
        j = _matching(s, i, "{", "}")
        if j < 0:
            break
        inner = s[i + 1:j]
        rep = f"root({inner}, {m.group(2)})" if m.group(2) else f"sqrt({inner})"
        s = s[:m.start()] + rep + s[j + 1:]
    s = re.sub(r"\\int", "∫", s)
    s = re.sub(r"\\sum", "Σ", s)
    s = re.sub(r"\\prod", "∏", s)
    s = re.sub(r"\\(sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|arcsin|arccos|arctan|ln|log|exp|lim|max|min|det)\b",
               lambda m: {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}.get(m.group(1), m.group(1)), s)
    for cmd, rep in LATEX_SIMPLE.items():
        s = s.replace(cmd + " ", rep + " ").replace(cmd + "{", rep + "{") if cmd in (r"\mathrm", r"\text", r"\operatorname") else s.replace(cmd, rep)
    s = re.sub(r"\\(" + "|".join(sorted(set(GREEK.values()), key=len, reverse=True)) + r")\b", r" \1 ", s)
    s = re.sub(r"\\([A-Za-z]+)", r" \1 ", s)  # any other command: drop the backslash, keep it a separate token
    return re.sub(r"[ ]{2,}", " ", s).strip()


def _greek(src: str) -> str:
    s = src
    for ch, rep in GREEK.items():
        s = re.sub(ch + r"(?![A-Za-z])", rep, s)
    return s


def _unicode(src: str) -> str:
    s = src
    for ch, rep in SIMPLE.items():
        s = s.replace(ch, rep)
    # superscript digits: x² -> x^2, e⁻¹ -> e^(-1)
    def sup(m):
        digits = "".join(SUPERSCRIPTS[c] for c in m.group(0))
        return f"^({digits})" if digits.startswith("-") else f"^{digits}"
    s = re.sub("[" + "".join(SUPERSCRIPTS) + "]+", sup, s)
    s = re.sub("[" + "".join(SUBSCRIPTS) + "]+", lambda m: "_" + "".join(SUBSCRIPTS[c] for c in m.group(0)), s)
    # √(...) or √x
    out, i = [], 0
    while i < len(s):
        if s[i] == "√":
            j = i + 1
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "(":
                k = _matching(s, j, "(", ")")
                out.append(f"sqrt{s[j:k + 1]}")
                i = k + 1
            else:
                m = re.match(r"[A-Za-z0-9_.]+", s[j:])
                tok = m.group(0) if m else ""
                out.append(f"sqrt({tok})")
                i = j + len(tok)
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _integrals(s: str) -> str:
    """∫_a^b f dx -> integrate(f, x, a, b); ∫ f dx -> integrate(f, x). Innermost last ∫ first."""
    while "∫" in s:
        i = s.rfind("∫")
        p = i + 1
        lo = hi = None
        while p < len(s) and s[p] in "_^ ":
            if s[p] == "_":
                lo, p = _braced(s, p + 1)
            elif s[p] == "^":
                hi, p = _braced(s, p + 1)
            else:
                p += 1
        rest = s[p:]
        # integrand ends at the last " dvar" (or "dvar" directly after a closing bracket) at depth 0
        depth, end, var = 0, -1, None
        for j in range(len(rest)):
            c = rest[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth < 0:
                    break
            m = re.match(r"(?<![A-Za-z0-9_])d([A-Za-z](?:[A-Za-z0-9_]*)?)(?![A-Za-z0-9_(])", rest[j:]) if depth == 0 else None
            if m and (j == 0 or not rest[j - 1].isalnum()):
                end, var = j, m.group(1)
                stop = j + m.end()
                break
        if end < 0:
            raise ValueError("An integral needs a 'dx' to say which variable: ∫ f dx.")
        integrand = rest[:end].strip()
        if not integrand:
            raise ValueError("Nothing to integrate before the 'dx'.")
        tail = rest[stop:]
        if lo is not None and hi is not None:
            call = f"integrate({integrand}, {var}, {lo}, {hi})"
        elif lo is None and hi is None:
            call = f"integrate({integrand}, {var})"
        else:
            raise ValueError("An integral needs both bounds or none: ∫_a^b f dx.")
        s = s[:i] + call + tail
    return s


def _sums(s: str) -> str:
    """Σ_{k=1}^{n} expr -> sum(expr, k, 1, n); the summand runs to the end of the enclosing group."""
    for sym, fn in (("Σ", "sum"), ("∏", "product")):
        while sym in s:
            i = s.rfind(sym)
            p = i + 1
            lo = hi = None
            while p < len(s) and s[p] in "_^ ":
                if s[p] == "_":
                    lo, p = _braced(s, p + 1)
                elif s[p] == "^":
                    hi, p = _braced(s, p + 1)
                else:
                    p += 1
            if lo is None or hi is None:
                raise ValueError(f"{sym} needs bounds: {sym}_{{k=1}}^{{n}} expr.")
            m = re.match(rf"\s*({IDENT})\s*=\s*(.+)$", lo.strip())
            if not m:
                raise ValueError(f"The lower bound of {sym} must be 'k = a'.")
            var, a = m.group(1), m.group(2)
            rest = s[p:]
            depth, end = 0, len(rest)
            for j, c in enumerate(rest):
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth -= 1
                    if depth < 0:
                        end = j
                        break
                elif c == "," and depth == 0:
                    end = j
                    break
            body = rest[:end].strip()
            if not body:
                raise ValueError(f"Nothing to add after {sym}.")
            s = s[:i] + f"{fn}({body}, {var}, {a}, {hi})" + rest[end:]
    return s


def _derivatives(s: str) -> str:
    """d/dx f -> diff(f, x); d^2/dx^2 f -> diff(f, x, 2). The operand is the next group or term."""
    pat = re.compile(r"(?<![A-Za-z0-9_])d(?:\^(\d))?\s*/\s*d([A-Za-z][A-Za-z0-9_]*)(?:\^(\d))?\s*")
    while True:
        m = pat.search(s)
        if not m:
            return s
        n = m.group(1) or m.group(3) or "1"
        var = m.group(2)
        rest = s[m.end():]
        if rest.startswith("("):
            k = _matching(rest, 0, "(", ")")
            operand, tail = rest[:k + 1], rest[k + 1:]
        elif rest.startswith("["):
            k = _matching(rest, 0, "[", "]")
            operand, tail = rest[1:k], rest[k + 1:]
        else:
            # up to the next top-level + or - (not inside brackets), or the end
            depth, end = 0, len(rest)
            for j, c in enumerate(rest):
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth -= 1
                    if depth < 0:
                        end = j
                        break
                elif c in "+-," and depth == 0 and j > 0:
                    end = j
                    break
            operand, tail = rest[:end].strip(), rest[end:]
        if not operand:
            raise ValueError("d/dx needs something to differentiate: d/dx (x^2).")
        call = f"diff({operand}, {var}{', ' + n if n != '1' else ''})"
        s = s[:m.start()] + call + tail


def normalize(src: str) -> str:
    """Rewrite paper and LaTeX notation into worksheet syntax. Returns src unchanged if no notation is present."""
    if not uses_notation(src):
        return src
    s = src.replace("∑", "Σ")
    if "\\" in s:
        s = _latex(s)
    s = _greek(s)
    s = _integrals(s)
    s = _sums(s)
    s = _unicode(s)
    s = _derivatives(s)
    s = s.replace("{", "(").replace("}", ")")
    return s
