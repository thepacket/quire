"""Sample plot cells for the UI. The engine produces points; the browser draws.

Kinds: function (y = f(x), several series), parametric (x(t), y(t)), polar
(r(theta)), scatter (data lists), slope (dy/dx = f(x, y) direction field),
implicit (F(x, y) == 0 contour). Function plots are sampled adaptively and
split at discontinuities.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp

from . import units as U
from .errors import QuireError
from .parser import identifiers, parse, split_top

MAX_POINTS = 3000


def _reduce(op):
    def fn(*args):
        arrs = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
        return op.reduce(np.stack(arrs))
    return fn


_NUMPY_EXTRA = {"Heaviside": lambda x, *_: np.heaviside(x, 0.5), "Max": _reduce(np.maximum), "Min": _reduce(np.minimum)}


def _clean(ys, n=None) -> list:
    ys = np.asarray(ys)
    if ys.ndim == 0:
        ys = np.full(n or 1, ys)
    if np.iscomplexobj(ys):
        ys = np.where(np.abs(ys.imag) < 1e-12 * np.maximum(1, np.abs(ys.real)), ys.real, np.nan)
    ys = ys.astype(float)
    return [None if (math.isnan(v) or math.isinf(v)) else float(v) for v in ys]


def sample_plot(cell: dict, env: dict, ev) -> dict:
    try:
        kind = (cell.get("kind") or "function").strip()
        fn = {"function": _function, "parametric": _parametric, "polar": _polar, "scatter": _scatter,
              "slope": _slope, "implicit": _implicit}.get(kind)
        if fn is None:
            raise QuireError(f"Unknown plot kind '{kind}'.")
        res = fn(cell, env, ev)
        res.setdefault("ok", True)
        res["kind"] = kind
        return res
    except QuireError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"}


# ---------------------------------------------------------------- helpers
def _bound(text, default, ns, ev, label):
    b = parse(text or default, ns, ev.unit_names)
    if not isinstance(b, sp.Expr) or b.free_symbols:
        raise QuireError(f"The '{label}' bound must be a number.")
    return b


def _numeric(expr, var, x_unit=1):
    """(callable of one array, unit label) for an expression in var."""
    if x_unit != 1:
        expr = expr.subs(var, var * x_unit)
    unknown = expr.free_symbols - {var}
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise QuireError(f"'{expr}' still depends on {names}. Define them above, or plot against one of them.")
    U.check_dimensions(expr)
    label = U.unit_label(expr)
    num, _ = U.strip_units(expr)
    f = sp.lambdify(var, num, modules=[_NUMPY_EXTRA, "scipy", "numpy"])
    return f, label


def _call(f, grid):
    with np.errstate(all="ignore"):
        ys = f(grid)
    ys = np.asarray(ys)
    if ys.ndim == 0:
        ys = np.full(grid.shape, float(ys) if not np.iscomplexobj(ys) else ys)
    return ys


def _infer_var(parts, env, ev, preferred=("x", "t", "theta")):
    free = set()
    for p in parts:
        free |= identifiers(p) - set(ev.base_namespace) - set(env) - {"seq_", "dprime_"}
    if len(free) == 1:
        return free.pop()
    for cand in preferred:
        if cand in free or not free:
            return cand
    raise QuireError(f"Several unknowns ({', '.join(sorted(free))}); choose the plot variable.")


def _resolve(text, ns, ev, xs):
    expr = parse(text, ns, ev.unit_names)
    if callable(expr) and not isinstance(expr, sp.Basic):
        expr = expr(xs)
    if isinstance(expr, sp.Lambda):
        expr = expr(xs)
    if not isinstance(expr, sp.Expr):
        raise QuireError(f"'{text.strip()}' is not something that can be plotted.")
    return expr


def _adaptive(f, a, b, n):
    """Uniform samples refined where the curve bends; None marks a discontinuity."""
    xs = np.linspace(a, b, n)
    ys = _call(f, xs)
    if np.iscomplexobj(ys):
        ys = np.where(np.abs(ys.imag) < 1e-12 * np.maximum(1, np.abs(ys.real)), ys.real, np.nan).astype(float)
    ys = ys.astype(float)
    finite = ys[np.isfinite(ys)]
    if finite.size == 0:
        return xs, ys, (None, None)
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    span = (hi - lo) or (abs(hi) or 1.0)
    # refinement: insert midpoints where the middle point is far from the chord
    for _ in range(3):
        if xs.size >= MAX_POINTS:
            break
        scale_x, scale_y = (b - a) or 1.0, span
        with np.errstate(all="ignore"):
            chord = (ys[:-2] + ys[2:]) / 2
            dev = np.abs(ys[1:-1] - chord) / scale_y
        bad = np.where(np.isfinite(dev) & (dev > 0.01))[0] + 1
        if bad.size == 0:
            break
        new_x = np.concatenate([(xs[bad - 1] + xs[bad]) / 2, (xs[bad] + xs[bad + 1]) / 2])
        new_x = np.unique(new_x)[: MAX_POINTS - xs.size]
        xs = np.sort(np.concatenate([xs, new_x]))
        ys = _call(f, xs).astype(float) if not np.iscomplexobj(_call(f, xs[:1])) else \
            np.where(np.abs(_call(f, xs).imag) < 1e-12, _call(f, xs).real, np.nan).astype(float)
    # discontinuities: a jump much larger than the visible range between neighbours
    out = ys.copy()
    jumps = np.abs(np.diff(ys))
    with np.errstate(all="ignore"):
        big = jumps > 3 * span
    for i in np.where(big)[0]:
        # keep the point on each side but break the line by inserting a gap marker later
        out[i] = np.nan if abs(ys[i]) > abs(hi) + 2 * span or abs(ys[i]) < abs(lo) - 2 * span else out[i]
        out[i + 1] = np.nan if abs(ys[i + 1]) > abs(hi) + 2 * span or abs(ys[i + 1]) < abs(lo) - 2 * span else out[i + 1]
        if np.isfinite(out[i]) and np.isfinite(out[i + 1]):
            out[i + 1] = np.nan
    # suggested y-range: ignore the spikes near poles
    pad = 0.08 * span
    suggest = (float(lo - pad), float(hi + pad)) if (finite.max() - finite.min()) > 4 * span else (None, None)
    return xs, out, suggest


# ---------------------------------------------------------------- kinds
def _function(cell, env, ev):
    exprs_src = (cell.get("exprs") or "").strip()
    if not exprs_src:
        return {"series": [], "empty": True}
    parts = [p for p in split_top(exprs_src, ",") if p.strip()]
    var = (cell.get("var") or "").strip() or _infer_var(parts, env, ev)
    ns = ev.namespace(env, [var])
    xs = ns[var]
    xmin, xmax = _bound(cell.get("xmin"), "-10", ns, ev, "from"), _bound(cell.get("xmax"), "10", ns, ev, "to")
    n = max(8, min(int(cell.get("samples") or 400), MAX_POINTS))
    x_unit = sp.S.One
    if U.has_units(xmin) or U.has_units(xmax):
        _, x_unit = U.split_units(U.to_base(xmin if U.has_units(xmin) else xmax))
    a, b = float(U.strip_units(xmin)[0]), float(U.strip_units(xmax)[0])
    if not a < b:
        raise QuireError("'from' must be smaller than 'to'.")
    series, y_units, suggest = [], set(), (None, None)
    for p in parts:
        expr = _resolve(p, ns, ev, xs)
        f, label = _numeric(expr, xs, x_unit)
        y_units.add(label)
        gx, gy, sug = _adaptive(f, a, b, n)
        if sug[0] is not None:
            suggest = (sug[0] if suggest[0] is None else min(suggest[0], sug[0]),
                       sug[1] if suggest[1] is None else max(suggest[1], sug[1]))
        series.append({"type": "line", "label": sp.latex(expr.subs(xs * x_unit, xs) if x_unit != 1 else expr),
                       "label_plain": p.strip(), "x": [float(v) for v in gx], "y": _clean(gy)})
    if len(y_units) > 1:
        raise QuireError("All plotted expressions must have the same units: " + ", ".join(sorted(y_units)))
    return {"var": var, "series": series, "xlabel": var + (f" [{U.unit_label(x_unit)}]" if x_unit != 1 else ""),
            "ylabel": f"[{y_units.pop()}]" if y_units and next(iter(y_units)) else "",
            "ysuggest": list(suggest) if suggest[0] is not None else None}


def _parametric(cell, env, ev):
    xsrc, ysrc = (cell.get("exprs") or "").strip(), (cell.get("expr2") or "").strip()
    if not xsrc or not ysrc:
        return {"series": [], "empty": True}
    var = (cell.get("var") or "").strip() or _infer_var([xsrc, ysrc], env, ev, ("t", "x", "theta"))
    ns = ev.namespace(env, [var])
    ts = ns[var]
    a = float(_bound(cell.get("xmin"), "0", ns, ev, "from"))
    b = float(_bound(cell.get("xmax"), "2 pi", ns, ev, "to"))
    n = max(8, min(int(cell.get("samples") or 600), MAX_POINTS))
    fx, lx = _numeric(_resolve(xsrc, ns, ev, ts), ts)
    fy, ly = _numeric(_resolve(ysrc, ns, ev, ts), ts)
    grid = np.linspace(a, b, n)
    return {"var": var, "series": [{"type": "line", "label": f"({sp.latex(_resolve(xsrc, ns, ev, ts))},\\ {sp.latex(_resolve(ysrc, ns, ev, ts))})",
                                    "label_plain": f"({xsrc}, {ysrc})", "x": _clean(_call(fx, grid), n), "y": _clean(_call(fy, grid), n)}],
            "xlabel": f"[{lx}]" if lx else "", "ylabel": f"[{ly}]" if ly else "", "equal": True}


def _polar(cell, env, ev):
    rsrc = (cell.get("exprs") or "").strip()
    if not rsrc:
        return {"series": [], "empty": True}
    var = (cell.get("var") or "").strip() or _infer_var([rsrc], env, ev, ("theta", "t", "x"))
    ns = ev.namespace(env, [var])
    th = ns[var]
    a = float(_bound(cell.get("xmin"), "0", ns, ev, "from"))
    b = float(_bound(cell.get("xmax"), "2 pi", ns, ev, "to"))
    n = max(8, min(int(cell.get("samples") or 600), MAX_POINTS))
    expr = _resolve(rsrc, ns, ev, th)
    fr, lr = _numeric(expr, th)
    grid = np.linspace(a, b, n)
    r = np.asarray(_call(fr, grid), dtype=float)
    return {"var": var, "series": [{"type": "line", "label": sp.latex(expr), "label_plain": rsrc,
                                    "x": _clean(r * np.cos(grid), n), "y": _clean(r * np.sin(grid), n)}],
            "xlabel": f"[{lr}]" if lr else "", "ylabel": "", "equal": True, "polar": True}


def _list(expr, what):
    if isinstance(expr, sp.MatrixBase):
        expr = list(expr)
    if not isinstance(expr, (list, tuple)):
        raise QuireError(f"{what} must be a list of numbers, e.g. [1, 2, 3].")
    try:
        return [float(U.strip_units(sp.sympify(v))[0]) for v in expr]
    except (TypeError, ValueError):
        raise QuireError(f"{what} must contain numbers only.") from None


def _scatter(cell, env, ev):
    xsrc, ysrc = (cell.get("exprs") or "").strip(), (cell.get("expr2") or "").strip()
    if not xsrc or not ysrc:
        return {"series": [], "empty": True}
    ns = ev.namespace(env)
    xs = _list(parse(xsrc, ns, ev.unit_names), "x data")
    ys = _list(parse(ysrc, ns, ev.unit_names), "y data")
    if len(xs) != len(ys):
        raise QuireError(f"x has {len(xs)} values but y has {len(ys)}.")
    return {"series": [{"type": "points", "label": f"\\text{{{len(xs)} points}}", "label_plain": f"{len(xs)} points",
                        "x": xs, "y": ys}], "xlabel": xsrc, "ylabel": ysrc}


def _slope(cell, env, ev):
    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    xv, yv = (cell.get("var") or "x, y").replace(" ", "").split(",")[:2] if "," in (cell.get("var") or "") else ("x", "y")
    ns = ev.namespace(env, [xv, yv])
    xs, ys = ns[xv], ns[yv]
    expr = _resolve(src, ns, ev, xs)
    unknown = expr.free_symbols - {xs, ys}
    if unknown:
        raise QuireError(f"'{src}' still depends on {', '.join(sorted(str(s) for s in unknown))}.")
    f = sp.lambdify((xs, ys), U.strip_units(expr)[0], modules="numpy")
    a, b = float(_bound(cell.get("xmin"), "-5", ns, ev, "x from")), float(_bound(cell.get("xmax"), "5", ns, ev, "x to"))
    c, d = float(_bound(cell.get("ymin"), "-5", ns, ev, "y from")), float(_bound(cell.get("ymax"), "5", ns, ev, "y to"))
    n = max(5, min(int(cell.get("samples") or 20), 60))
    gx, gy = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
    with np.errstate(all="ignore"):
        s = np.asarray(f(gx, gy), dtype=float)
        if s.shape != gx.shape:
            s = np.full(gx.shape, float(s))
    hx, hy = (b - a) / n * 0.4, (d - c) / n * 0.4
    segs = []
    for i in range(n):
        for j in range(n):
            m = s[i, j]
            if not np.isfinite(m):
                continue
            # unit direction (1, m) scaled to the cell
            dx, dy = 1.0, m * (b - a) / (d - c)  # in normalized screen-ish coordinates
            norm = math.hypot(dx, dy)
            ux, uy = dx / norm, dy / norm
            segs.append([gx[i, j] - ux * hx, gy[i, j] - uy * hy * (d - c) / (b - a), gx[i, j] + ux * hx,
                         gy[i, j] + uy * hy * (d - c) / (b - a)])
    return {"series": [{"type": "segments", "label": f"\\frac{{d{yv}}}{{d{xv}}} = {sp.latex(expr)}", "label_plain": src,
                        "segments": segs}], "xlabel": xv, "ylabel": yv, "xrange": [a, b], "yrange": [c, d]}


def _implicit(cell, env, ev):
    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    xv, yv = (cell.get("var") or "").replace(" ", "").split(",")[:2] if "," in (cell.get("var") or "") else ("x", "y")
    ns = ev.namespace(env, [xv, yv])
    xs, ys = ns[xv], ns[yv]
    expr = parse(src, ns, ev.unit_names)
    if isinstance(expr, sp.Eq):
        expr = expr.lhs - expr.rhs
    if not isinstance(expr, sp.Expr):
        raise QuireError("Write an equation such as x^2 + y^2 == 1.")
    unknown = expr.free_symbols - {xs, ys}
    if unknown:
        raise QuireError(f"'{src}' still depends on {', '.join(sorted(str(s) for s in unknown))}.")
    f = sp.lambdify((xs, ys), U.strip_units(expr)[0], modules="numpy")
    a, b = float(_bound(cell.get("xmin"), "-5", ns, ev, "x from")), float(_bound(cell.get("xmax"), "5", ns, ev, "x to"))
    c, d = float(_bound(cell.get("ymin"), "-5", ns, ev, "y from")), float(_bound(cell.get("ymax"), "5", ns, ev, "y to"))
    n = max(20, min(int(cell.get("samples") or 200), 400))
    X, Y = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
    with np.errstate(all="ignore"):
        Z = np.asarray(f(X, Y), dtype=float)
        if Z.shape != X.shape:
            Z = np.full(X.shape, float(Z))
    segs = _marching_squares(X, Y, Z)
    return {"series": [{"type": "segments", "label": sp.latex(sp.Eq(expr, 0)), "label_plain": src, "segments": segs}],
            "xlabel": xv, "ylabel": yv, "xrange": [a, b], "yrange": [c, d], "equal": True}


def _marching_squares(X, Y, Z):
    """Zero-level segments of Z on the grid (linear interpolation on cell edges)."""
    segs = []
    n, m = Z.shape

    def cross(x1, y1, z1, x2, y2, z2):
        t = z1 / (z1 - z2)
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    for i in range(n - 1):
        for j in range(m - 1):
            corners = [(X[i, j], Y[i, j], Z[i, j]), (X[i, j + 1], Y[i, j + 1], Z[i, j + 1]),
                       (X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]), (X[i + 1, j], Y[i + 1, j], Z[i + 1, j])]
            if any(not np.isfinite(cz) for _, _, cz in corners):
                continue
            pts = []
            for k in range(4):
                x1, y1, z1 = corners[k]
                x2, y2, z2 = corners[(k + 1) % 4]
                if (z1 < 0) != (z2 < 0):
                    pts.append(cross(x1, y1, z1, x2, y2, z2))
            if len(pts) == 2:
                segs.append([float(pts[0][0]), float(pts[0][1]), float(pts[1][0]), float(pts[1][1])])
            elif len(pts) == 4:
                segs.append([float(pts[0][0]), float(pts[0][1]), float(pts[1][0]), float(pts[1][1])])
                segs.append([float(pts[2][0]), float(pts[2][1]), float(pts[3][0]), float(pts[3][1])])
    return segs
