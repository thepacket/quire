"""Sample expressions for plotting. The UI draws; the engine only produces points."""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from . import units as U
from .errors import QuireError
from .parser import identifiers, parse, split_top


def _clean(ys: np.ndarray, n: int) -> list:
    ys = np.asarray(ys)
    if ys.ndim == 0:
        ys = np.full(n, ys)
    if np.iscomplexobj(ys):
        ys = np.where(np.abs(ys.imag) < 1e-12 * np.maximum(1, np.abs(ys.real)), ys.real, np.nan)
    ys = ys.astype(float)
    return [None if (math.isnan(v) or math.isinf(v)) else float(v) for v in ys]


def sample_plot(cell: dict, env: dict, ev) -> dict:
    try:
        return _sample(cell, env, ev)
    except QuireError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"}


def _sample(cell: dict, env: dict, ev) -> dict:
    exprs_src = (cell.get("exprs") or "").strip()
    if not exprs_src:
        return {"ok": True, "series": [], "empty": True}
    parts = [p for p in split_top(exprs_src, ",") if p.strip()]
    var = (cell.get("var") or "").strip()
    if not var:
        free = set()
        for p in parts:
            free |= identifiers(p) - set(ev.base_namespace) - set(env) - {"seq_", "dprime_"}
        if len(free) == 1:
            var = free.pop()
        elif "x" in free or not free:
            var = "x"
        elif "t" in free:
            var = "t"
        else:
            raise QuireError(f"Several unknowns ({', '.join(sorted(free))}); choose the plot variable.")
    ns = ev.namespace(env, [var])
    xs = ns[var]

    xmin = parse(cell.get("xmin") or "-10", ns, ev.unit_names)
    xmax = parse(cell.get("xmax") or "10", ns, ev.unit_names)
    n = max(8, min(int(cell.get("samples") or 400), 5000))
    for b, label in ((xmin, "from"), (xmax, "to")):
        if not isinstance(b, sp.Expr) or b.free_symbols:
            raise QuireError(f"The '{label}' bound must be a number.")
    x_unit = sp.S.One
    if U.has_units(xmin) or U.has_units(xmax):
        _, x_unit = U.split_units(U.to_base(xmin if U.has_units(xmin) else xmax))
    a = float(U.strip_units(xmin)[0])
    b = float(U.strip_units(xmax)[0])
    if not a < b:
        raise QuireError("'from' must be smaller than 'to'.")
    grid = np.linspace(a, b, n)

    series, y_units = [], set()
    for p in parts:
        expr = parse(p, ns, ev.unit_names)
        if isinstance(expr, sp.Lambda):
            expr = expr(xs)
        if callable(expr) and not isinstance(expr, sp.Basic):
            expr = expr(xs)
        if not isinstance(expr, sp.Expr):
            raise QuireError(f"'{p.strip()}' is not something that can be plotted.")
        if x_unit != 1:
            expr = expr.subs(xs, xs * x_unit)
        unknown = expr.free_symbols - {xs}
        if unknown:
            names = ", ".join(sorted(str(s) for s in unknown))
            raise QuireError(f"'{p.strip()}' still depends on {names}. Define them above, or plot against one of them.")
        U.check_dimensions(expr)
        y_units.add(U.unit_label(expr))
        num, _ = U.strip_units(expr)
        f = sp.lambdify(xs, num, modules=["numpy", {"Heaviside": lambda x, *_: np.heaviside(x, 0.5)}])
        with np.errstate(all="ignore"):
            ys = f(grid)
        series.append({"label": sp.latex(expr.subs(xs * x_unit, xs) if x_unit != 1 else expr),
                       "label_plain": p.strip(), "y": _clean(ys, n)})
    if len(y_units) > 1:
        raise QuireError("All plotted expressions must have the same units: " + ", ".join(sorted(y_units)))
    x_label = var + (f" [{U.unit_label(x_unit)}]" if x_unit != 1 else "")
    y_label = f"[{y_units.pop()}]" if y_units and next(iter(y_units)) else ""
    return {"ok": True, "var": var, "x": [float(v) for v in grid], "series": series,
            "xlabel": x_label, "ylabel": y_label}
