"""Numeric identification: name a number as a combination of known constants.

Integer-relation detection (PSLQ) over a basis of constants, at 50 digits, with
small coefficients. It is evidence, not proof, so callers attach a note.
"""
from __future__ import annotations

import mpmath as mp
import sympy as sp

from .. import hooks

DPS = 50
MAXCOEFF = 400

# (sympy expression, mpmath value factory). Kept small on purpose: spurious relations grow with basis size.
_BASIS = [
    (sp.S.One, lambda: mp.mpf(1)),
    (sp.pi, lambda: mp.pi),
    (sp.pi ** 2, lambda: mp.pi ** 2),
    (sp.log(2), lambda: mp.log(2)),
    (sp.pi * sp.log(2), lambda: mp.pi * mp.log(2)),
    (sp.sqrt(2), lambda: mp.sqrt(2)),
    (sp.sqrt(3), lambda: mp.sqrt(3)),
    (sp.Catalan, lambda: mp.catalan),
    (sp.zeta(3), lambda: mp.zeta(3)),
    (sp.EulerGamma, lambda: mp.euler),
    (sp.pi ** 3, lambda: mp.pi ** 3),
    (sp.log(2) ** 2, lambda: mp.log(2) ** 2),
    (sp.pi ** 4, lambda: mp.pi ** 4),
    (sp.log(3), lambda: mp.log(3)),
    (sp.pi ** 2 * sp.log(2), lambda: mp.pi ** 2 * mp.log(2)),
    (sp.pi * sp.sqrt(2), lambda: mp.pi * mp.sqrt(2)),
    (sp.pi * sp.sqrt(3), lambda: mp.pi * mp.sqrt(3)),
    (sp.sqrt(sp.pi), lambda: mp.sqrt(mp.pi)),
    (sp.E, lambda: mp.e),
    (sp.pi / sp.E, lambda: mp.pi / mp.e),
]


def identify(value, dps: int = DPS) -> sp.Expr | None:
    """Return an exact expression equal to ``value`` (known to ``dps`` digits), or None.

    The coefficient bound shrinks with the available digits so that a short decimal
    cannot be "recognized" as a coincidence.
    """
    dps = int(dps)
    if dps < 15:
        return None
    # Fewer digits: smaller basis and smaller coefficients, so a relation cannot be a coincidence.
    maxcoeff, nbasis = (MAXCOEFF, len(_BASIS)) if dps >= 40 else ((60, 12) if dps >= 25 else (12, 8))
    basis = _BASIS[:nbasis]
    with mp.workdps(dps):
        v = mp.mpf(value)
        if not mp.isfinite(v) or v == 0:
            return None
        vec = [v] + [b() for _, b in basis]
        rel = mp.pslq(vec, tol=mp.mpf(10) ** (-(dps - 4)), maxcoeff=maxcoeff, maxsteps=10000)
        if not rel or rel[0] == 0:
            return None
        # v = -(sum c_i b_i) / c_0
        expr = -sp.Add(*[sp.Integer(c) * b for c, (b, _) in zip(rel[1:], basis) if c]) / sp.Integer(rel[0])
        check = sp.N(expr, dps)
        if abs(mp.mpf(str(check)) - v) > mp.mpf(10) ** (-(dps - 3)) * max(1, abs(v)):
            return None
        return expr


def recognize(x, digits: int = DPS):
    """Worksheet function: recognize(0.6931471805599453) -> log(2)."""
    x = sp.sympify(x)
    if x.free_symbols:
        return x
    if isinstance(x, sp.Float):
        digits = min(int(digits), int(x._prec * 0.30103))  # a decimal is only as precise as it was typed
    digits = max(int(digits), 1)
    with mp.workdps(max(digits, 30)):
        try:
            v = mp.mpf(str(sp.N(x, max(digits, 30))))
        except (TypeError, ValueError):
            return x
    res = identify(v, digits)
    if res is None:
        return x
    hooks.context.setdefault("notes", []).append(f"recognized numerically from {digits} digits; evidence, not proof")
    return res


def register(api):
    api.function("recognize", recognize, signature="recognize(x)",
                 doc="find an exact form for a decimal using known constants (PSLQ); evidence, not proof",
                 category="Simplify & expand", example="recognize(2.17758609030360213)")
