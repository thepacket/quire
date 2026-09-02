"""Sample plot cells for the UI. The engine produces points; the browser draws.

Plot kinds live in a registry (``KINDS``); modules add their own through
``api.plot_kind``. Core kinds: function (y = f(x), several series),
parametric (x(t), y(t)), polar (r(theta)), scatter (data, with error bars),
slope (dy/dx = f(x, y) direction field), implicit (F(x, y) == 0 contour),
shapes (geometry), contour and heatmap (F(x, y) on a grid), surface
(z = F(x, y)) and curve3d (x(t), y(t), z(t)). Two-dimensional kinds are drawn
as SVG in the browser; grids and 3D go through Plotly.

Function plots are sampled adaptively and split at discontinuities. When a
series depends on sliders, it is also compiled to JavaScript so the browser
can redraw it while a slider is dragged, before the server answers.
"""
from __future__ import annotations

import math
import re

import numpy as np
import sympy as sp

from . import units as U
from .errors import QuireError
from .parser import called_names, identifiers, parse, split_top

MAX_POINTS = 3000
KINDS: dict[str, dict] = {}


def register_kind(name: str, fn, *, label: str, f1: str, f2: str | None = None, f3: str | None = None,
                  var: str | None = None, range: str | None = None, yrange: bool = False, ph1: str = "",
                  ph2: str = "", ph3: str = "", renderer: str = "svg", annot: bool = False, module: str = "core",
                  doc: str = "", samples: str = "400"):
    """Describe a plot kind: ``fn(cell, env, ev)`` returns the series; the rest drives the form in the UI.

    f1/f2/f3 label the expression fields (None hides a field); var labels the variable field; range is the
    label of the "from/to" pair (None hides it); yrange shows a second pair; ph* are placeholders that
    double as examples; renderer is "svg" or "plotly"; annot enables the annotation field.
    """
    KINDS[name] = dict(fn=fn, name=name, label=label, f1=f1, f2=f2, f3=f3, var=var, range=range, yrange=yrange,
                       ph1=ph1, ph2=ph2, ph3=ph3, renderer=renderer, annot=annot, module=module, doc=doc,
                       samples=samples)


def describe_kinds(extra: dict | None = None) -> list[dict]:
    """The kind descriptors without their functions (for the catalog)."""
    return [{k: v for k, v in spec.items() if k != "fn"} for spec in [*KINDS.values(), *(extra or {}).values()]]


def _reduce(op):
    def fn(*args):
        arrs = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
        return op.reduce(np.stack(arrs))
    return fn


_NUMPY_EXTRA = {"Heaviside": lambda x, *_: np.heaviside(x, 0.5), "Max": _reduce(np.maximum), "Min": _reduce(np.minimum)}
LAMBDIFY_MODULES = [_NUMPY_EXTRA, "scipy", "numpy"]


def clean(ys, n=None) -> list:
    """A list of floats with None for undefined values (and for complex values with a real imaginary part)."""
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
        spec = getattr(ev, "plot_kinds", {}).get(kind) or KINDS.get(kind)
        if spec is None:
            raise QuireError(f"Unknown plot kind '{kind}'.")
        ev.nominal_values = ev.nominals(env) if hasattr(ev, "nominals") else {}
        res = spec["fn"](cell, env, ev)
        res.setdefault("ok", True)
        res["kind"] = kind
        res["renderer"] = spec["renderer"]
        if spec["annot"] and (cell.get("annot") or "").strip() and res.get("series"):
            res["annotations"] = annotations(cell["annot"], env, ev, res)
        return res
    except QuireError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"}


# ---------------------------------------------------------------- helpers (public: modules use them too)
def bound(text, default, ns, ev, label):
    """A numeric bound from a field (units allowed)."""
    b = parse(text or default, ns, ev.unit_names)
    if not isinstance(b, sp.Expr) or b.free_symbols:
        raise QuireError(f"The '{label}' bound must be a number.")
    return b


def plain_bounds(cell, ns, ev, defaults=("-5", "5", "-5", "5")):
    """x and y ranges as floats, units stripped."""
    a = float(U.strip_units(bound(cell.get("xmin"), defaults[0], ns, ev, "x from"))[0])
    b = float(U.strip_units(bound(cell.get("xmax"), defaults[1], ns, ev, "x to"))[0])
    c = float(U.strip_units(bound(cell.get("ymin"), defaults[2], ns, ev, "y from"))[0])
    d = float(U.strip_units(bound(cell.get("ymax"), defaults[3], ns, ev, "y to"))[0])
    if not a < b or not c < d:
        raise QuireError("'from' must be smaller than 'to'.")
    return a, b, c, d


def samples(cell, default: int, maximum: int, minimum: int = 8) -> int:
    return max(minimum, min(int(cell.get("samples") or default), maximum))


def two_vars(cell, default=("x", "y")):
    v = (cell.get("var") or "").replace(" ", "")
    if "," in v:
        a, b = v.split(",")[:2]
        if a and b:
            return a, b
    return default


def numeric(expr, var, x_unit=1):
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
    f = sp.lambdify(var, num, modules=LAMBDIFY_MODULES)
    return f, label


def numeric2(expr, xs, ys):
    """(callable of two arrays, unit label) for an expression in xs and ys."""
    unknown = expr.free_symbols - {xs, ys}
    if unknown:
        raise QuireError(f"'{expr}' still depends on {', '.join(sorted(str(s) for s in unknown))}.")
    U.check_dimensions(expr)
    label = U.unit_label(expr)
    f = sp.lambdify((xs, ys), U.strip_units(expr)[0], modules=LAMBDIFY_MODULES)
    return f, label


def call(f, grid):
    with np.errstate(all="ignore"):
        ys = f(grid)
    ys = np.asarray(ys)
    if ys.ndim == 0:
        ys = np.full(grid.shape, float(ys) if not np.iscomplexobj(ys) else ys)
    return ys


def call2(f, X, Y):
    with np.errstate(all="ignore"):
        Z = np.asarray(f(X, Y))
    if Z.shape != X.shape:
        Z = np.full(X.shape, complex(Z) if np.iscomplexobj(Z) else float(Z))
    if np.iscomplexobj(Z):
        Z = np.where(np.abs(Z.imag) < 1e-12 * np.maximum(1, np.abs(Z.real)), Z.real, np.nan)
    return Z.astype(float)


def infer_var(parts, env, ev, preferred=("x", "t", "theta")):
    free = set()
    for p in parts:
        free |= identifiers(p) - set(ev.base_namespace) - set(env) - {"seq_", "dprime_"}
    if len(free) == 1:
        return free.pop()
    for cand in preferred:
        if cand in free or not free:
            return cand
    raise QuireError(f"Several unknowns ({', '.join(sorted(free))}); choose the plot variable.")


def resolve(text, ns, ev, *syms):
    """Parse text to an expression in the given symbols (calling worksheet functions with them)."""
    expr = parse(text, ns, ev.unit_names)
    if callable(expr) and not isinstance(expr, sp.Basic):
        expr = expr(*syms)
    if isinstance(expr, sp.Lambda):
        expr = expr(*syms)
    if isinstance(expr, sp.Eq):
        expr = expr.lhs - expr.rhs
    if not isinstance(expr, sp.Expr):
        raise QuireError(f"'{text.strip()}' is not something that can be plotted.")
    nominal = getattr(ev, "nominal_values", None)
    if nominal:
        expr = expr.subs(nominal)  # measured quantities plot at their nominal value
    return expr


def _adaptive(f, a, b, n):
    """Uniform samples refined where the curve bends; None marks a discontinuity."""
    xs = np.linspace(a, b, n)
    ys = call(f, xs)
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
        scale_y = span
        with np.errstate(all="ignore"):
            chord = (ys[:-2] + ys[2:]) / 2
            dev = np.abs(ys[1:-1] - chord) / scale_y
        bad = np.where(np.isfinite(dev) & (dev > 0.01))[0] + 1
        if bad.size == 0:
            break
        new_x = np.concatenate([(xs[bad - 1] + xs[bad]) / 2, (xs[bad] + xs[bad + 1]) / 2])
        new_x = np.unique(new_x)[: MAX_POINTS - xs.size]
        xs = np.sort(np.concatenate([xs, new_x]))
        ys = call(f, xs).astype(float) if not np.iscomplexobj(call(f, xs[:1])) else \
            np.where(np.abs(call(f, xs).imag) < 1e-12, call(f, xs).real, np.nan).astype(float)
    # discontinuities: a jump much larger than the visible range between neighbours
    out = ys.copy()
    jumps = np.abs(np.diff(ys))
    with np.errstate(all="ignore"):
        big = jumps > 3 * span
    for i in np.where(big)[0]:
        out[i] = np.nan if abs(ys[i]) > abs(hi) + 2 * span or abs(ys[i]) < abs(lo) - 2 * span else out[i]
        out[i + 1] = np.nan if abs(ys[i + 1]) > abs(hi) + 2 * span or abs(ys[i + 1]) < abs(lo) - 2 * span else out[i + 1]
        if np.isfinite(out[i]) and np.isfinite(out[i + 1]):
            out[i + 1] = np.nan
    pad = 0.08 * span
    suggest = (float(lo - pad), float(hi + pad)) if (finite.max() - finite.min()) > 4 * span else (None, None)
    return xs, out, suggest


# ---------------------------------------------------------------- sliders compiled to JavaScript
_SLIDER_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*slider\s*\(")


def slider_namespace(texts, bound_names, env, ev):
    """A namespace in which slider names stay symbolic, or None when no slider is involved.

    The definitions the texts depend on are re-run in a scratch environment
    with the slider lines left out, so that a definition such as
    ``g(x) = a sin(x)`` keeps ``a`` as a symbol. Only the dependency chain is
    re-run, under a short time budget; anything that fails means no fast path.
    """
    sliders = env.get("$sliders") or {}
    trace = env.get("$trace") or []
    if not sliders:
        return None
    needed = set()
    for t in texts:
        needed |= identifiers(t) | set(called_names(t))
    needed -= set(bound_names)
    chosen = []
    for rec in reversed(trace):
        if set(rec["defines"]) & needed:
            chosen.append(rec)
            needed |= set(rec["uses"])
    if not any(set(rec["defines"]) & set(sliders) for rec in chosen):
        return None
    scratch = {k: v for k, v in env.items() if k in ("$assume", "$bounds", "$digits")}
    from ..modules.builtin._util import with_budget

    def rerun():
        for rec in reversed(chosen):
            lines = [ln for ln in rec["source"].split("\n")
                     if not (_SLIDER_LINE.match(ln) and _SLIDER_LINE.match(ln).group(1) in sliders)]
            src = "\n".join(lines)
            if src.strip() and not ev.evaluate_math(src, scratch)["ok"]:
                return False
        return True

    try:
        ok, timed_out = with_budget(2.0, rerun)
    except Exception:  # noqa: BLE001
        return None
    if timed_out or not ok:
        return None
    return ev.namespace(scratch, bound_names), set(sliders)


def compile_js(exprs, var, held) -> tuple[list[str], list[str]] | None:
    """JavaScript for expressions in var and slider symbols: (codes, parameter names), or None."""
    hns, snames = held
    codes, params = [], set()
    for expr in exprs:
        num, _ = U.strip_units(expr)
        free = {str(s) for s in num.free_symbols}
        if free - snames - {str(var)}:
            return None
        params |= free & snames
        try:
            code = sp.jscode(num)
        except Exception:  # noqa: BLE001 - a function without a JavaScript form
            return None
        if "Not supported" in code:
            return None
        codes.append(code)
    if not params:
        return None
    return codes, sorted(params)


# ---------------------------------------------------------------- annotations
_STR = re.compile(r'"([^"]*)"')
_ANNOT_HELP = ('use mark(x, "label"), point(x, y, "label"), shade(a, b), band(a, b), hline(y, "label"), '
               'vline(x, "label") or text(x, y, "label")')


def _interp(series, x):
    xs, ys = series["x"], series["y"]
    best = None
    for i in range(len(xs) - 1):
        if xs[i] is None or xs[i + 1] is None or ys[i] is None or ys[i + 1] is None:
            continue
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / ((xs[i + 1] - xs[i]) or 1.0)
            return ys[i] + t * (ys[i + 1] - ys[i])
        if best is None or abs(xs[i] - x) < abs(xs[best] - x):
            best = i
    if best is None:
        raise QuireError("mark(x) needs a curve to sit on.")
    return ys[best]


def annotations(text, env, ev, res) -> list[dict]:
    ns = ev.namespace(env)
    lines = [i for i, s in enumerate(res["series"]) if s.get("type") == "line" and s.get("x")]
    out = []
    for item in split_top(text, ","):
        item = item.strip()
        if not item:
            continue
        labels = _STR.findall(item)
        m = re.match(r"^\s*([a-z_]+)\s*\((.*)\)\s*$", _STR.sub("", item), re.S)
        if not m:
            raise QuireError(f"Annotation '{item}': {_ANNOT_HELP}.")
        name, args_src = m.group(1), m.group(2)
        args = []
        for a in split_top(args_src, ","):
            if a.strip():
                v = parse(a, ns, ev.unit_names)
                if not isinstance(v, sp.Expr) or v.free_symbols:
                    raise QuireError(f"Annotation '{item}': '{a.strip()}' is not a number.")
                args.append(float(U.strip_units(v)[0]))
        label = labels[0] if labels else None
        need = {"mark": 1, "point": 2, "shade": 2, "band": 2, "hline": 1, "vline": 1, "text": 2}.get(name)
        if need is None:
            raise QuireError(f"Annotation '{item}': {_ANNOT_HELP}.")
        if len(args) < need:
            raise QuireError(f"Annotation '{item}' needs {need} number{'s' if need > 1 else ''}.")
        if name == "mark":
            if len(args) >= 2:
                x, y = args[0], args[1]
            else:
                if not lines:
                    raise QuireError("mark(x) needs a curve to sit on; give the y value: mark(x, y).")
                x, y = args[0], _interp(res["series"][lines[0]], args[0])
            out.append({"type": "point", "x": x, "y": y, "label": label if label is not None else f"({x:g}, {y:.4g})"})
        elif name == "point":
            out.append({"type": "point", "x": args[0], "y": args[1], "label": label if label is not None else f"({args[0]:g}, {args[1]:g})"})
        elif name == "shade":
            if not lines:
                raise QuireError("shade(a, b) shades under a curve; this plot has none (use band(a, b) for a vertical band).")
            out.append({"type": "shade", "x0": min(args[:2]), "x1": max(args[:2]), "series": lines[0], "label": label})
        elif name == "band":
            out.append({"type": "band", "x0": min(args[:2]), "x1": max(args[:2]), "label": label})
        elif name == "hline":
            out.append({"type": "hline", "y": args[0], "label": label})
        elif name == "vline":
            out.append({"type": "vline", "x": args[0], "label": label})
        else:
            out.append({"type": "text", "x": args[0], "y": args[1], "label": label or ""})
    return out


# ---------------------------------------------------------------- kinds
def _function(cell, env, ev):
    exprs_src = (cell.get("exprs") or "").strip()
    if not exprs_src:
        return {"series": [], "empty": True}
    parts = [p for p in split_top(exprs_src, ",") if p.strip()]
    var = (cell.get("var") or "").strip() or infer_var(parts, env, ev)
    ns = ev.namespace(env, [var])
    xs = ns[var]
    xmin, xmax = bound(cell.get("xmin"), "-10", ns, ev, "from"), bound(cell.get("xmax"), "10", ns, ev, "to")
    n = samples(cell, 400, MAX_POINTS)
    x_unit = sp.S.One
    if U.has_units(xmin) or U.has_units(xmax):
        _, x_unit = U.split_units(U.to_base(xmin if U.has_units(xmin) else xmax))
    a, b = float(U.strip_units(xmin)[0]), float(U.strip_units(xmax)[0])
    if not a < b:
        raise QuireError("'from' must be smaller than 'to'.")
    series, y_units, suggest = [], set(), (None, None)
    for p in parts:
        expr = resolve(p, ns, ev, xs)
        f, label = numeric(expr, xs, x_unit)
        y_units.add(label)
        gx, gy, sug = _adaptive(f, a, b, n)
        if sug[0] is not None:
            suggest = (sug[0] if suggest[0] is None else min(suggest[0], sug[0]),
                       sug[1] if suggest[1] is None else max(suggest[1], sug[1]))
        series.append({"type": "line", "label": sp.latex(expr.subs(xs * x_unit, xs) if x_unit != 1 else expr),
                       "label_plain": p.strip(), "x": [float(v) for v in gx], "y": clean(gy)})
    if len(y_units) > 1:
        raise QuireError("All plotted expressions must have the same units: " + ", ".join(sorted(y_units)))
    held = slider_namespace(parts, [var], env, ev)
    if held:
        hx = held[0][var]
        for p, s in zip(parts, series):
            try:
                expr = resolve(p, held[0], ev, hx)
                if x_unit != 1:
                    expr = expr.subs(hx, hx * x_unit)
                js = compile_js([expr], hx, held)
            except Exception:  # noqa: BLE001
                js = None
            if js:
                s["js"], s["params"] = js
    return {"var": var, "series": series, "xlabel": var + (f" [{U.unit_label(x_unit)}]" if x_unit != 1 else ""),
            "ylabel": f"[{y_units.pop()}]" if y_units and next(iter(y_units)) else "",
            "ysuggest": list(suggest) if suggest[0] is not None else None}


def _parametric(cell, env, ev):
    xsrc, ysrc = (cell.get("exprs") or "").strip(), (cell.get("expr2") or "").strip()
    if not xsrc or not ysrc:
        return {"series": [], "empty": True}
    var = (cell.get("var") or "").strip() or infer_var([xsrc, ysrc], env, ev, ("t", "x", "theta"))
    ns = ev.namespace(env, [var])
    ts = ns[var]
    a = float(bound(cell.get("xmin"), "0", ns, ev, "from"))
    b = float(bound(cell.get("xmax"), "2 pi", ns, ev, "to"))
    n = samples(cell, 600, MAX_POINTS)
    ex, ey = resolve(xsrc, ns, ev, ts), resolve(ysrc, ns, ev, ts)
    fx, lx = numeric(ex, ts)
    fy, ly = numeric(ey, ts)
    grid = np.linspace(a, b, n)
    s = {"type": "line", "label": f"({sp.latex(ex)},\\ {sp.latex(ey)})", "label_plain": f"({xsrc}, {ysrc})",
         "x": clean(call(fx, grid), n), "y": clean(call(fy, grid), n)}
    held = slider_namespace([xsrc, ysrc], [var], env, ev)
    if held:
        try:
            ht = held[0][var]
            js = compile_js([resolve(xsrc, held[0], ev, ht), resolve(ysrc, held[0], ev, ht)], ht, held)
        except Exception:  # noqa: BLE001
            js = None
        if js:
            s["js"], s["params"] = js
            s["grid"] = [float(v) for v in grid]
    return {"var": var, "series": [s], "xlabel": f"[{lx}]" if lx else "", "ylabel": f"[{ly}]" if ly else "", "equal": True}


def _polar(cell, env, ev):
    rsrc = (cell.get("exprs") or "").strip()
    if not rsrc:
        return {"series": [], "empty": True}
    var = (cell.get("var") or "").strip() or infer_var([rsrc], env, ev, ("theta", "t", "x"))
    ns = ev.namespace(env, [var])
    th = ns[var]
    a = float(bound(cell.get("xmin"), "0", ns, ev, "from"))
    b = float(bound(cell.get("xmax"), "2 pi", ns, ev, "to"))
    n = samples(cell, 600, MAX_POINTS)
    expr = resolve(rsrc, ns, ev, th)
    fr, lr = numeric(expr, th)
    grid = np.linspace(a, b, n)
    r = np.asarray(call(fr, grid), dtype=float)
    s = {"type": "line", "label": sp.latex(expr), "label_plain": rsrc,
         "x": clean(r * np.cos(grid), n), "y": clean(r * np.sin(grid), n)}
    held = slider_namespace([rsrc], [var], env, ev)
    if held:
        try:
            ht = held[0][var]
            hr = resolve(rsrc, held[0], ev, ht)
            js = compile_js([hr * sp.cos(ht), hr * sp.sin(ht)], ht, held)
        except Exception:  # noqa: BLE001
            js = None
        if js:
            s["js"], s["params"] = js
            s["grid"] = [float(v) for v in grid]
    return {"var": var, "series": [s], "xlabel": f"[{lr}]" if lr else "", "ylabel": "", "equal": True, "polar": True}


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
    esrc = (cell.get("expr3") or "").strip()
    if not xsrc or not ysrc:
        return {"series": [], "empty": True}
    ns = ev.namespace(env)
    xs = _list(parse(xsrc, ns, ev.unit_names), "x data")
    ys = _list(parse(ysrc, ns, ev.unit_names), "y data")
    if len(xs) != len(ys):
        raise QuireError(f"x has {len(xs)} values but y has {len(ys)}.")
    s = {"type": "points", "label": f"\\text{{{len(xs)} points}}", "label_plain": f"{len(xs)} points", "x": xs, "y": ys}
    if esrc:
        e = parse(esrc, ns, ev.unit_names)
        if isinstance(e, sp.Expr) and not e.free_symbols:
            errs = [abs(float(U.strip_units(e)[0]))] * len(xs)
        else:
            errs = [abs(v) for v in _list(e, "y errors")]
            if len(errs) != len(xs):
                raise QuireError(f"{len(errs)} error values for {len(xs)} points.")
        s["yerr"] = errs
    return {"series": [s], "xlabel": xsrc, "ylabel": ysrc}


def _slope(cell, env, ev):
    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    xv, yv = two_vars(cell)
    ns = ev.namespace(env, [xv, yv])
    xs, ys = ns[xv], ns[yv]
    expr = resolve(src, ns, ev, xs)
    unknown = expr.free_symbols - {xs, ys}
    if unknown:
        raise QuireError(f"'{src}' still depends on {', '.join(sorted(str(s) for s in unknown))}.")
    f = sp.lambdify((xs, ys), U.strip_units(expr)[0], modules="numpy")
    a, b, c, d = plain_bounds(cell, ns, ev)
    n = samples(cell, 20, 60, 5)
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
            dx, dy = 1.0, m * (b - a) / (d - c)
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
    xv, yv = two_vars(cell)
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
    a, b, c, d = plain_bounds(cell, ns, ev)
    n = samples(cell, 200, 400, 20)
    X, Y = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
    with np.errstate(all="ignore"):
        Z = np.asarray(f(X, Y), dtype=float)
        if Z.shape != X.shape:
            Z = np.full(X.shape, float(Z))
    segs = marching_squares(X, Y, Z)
    return {"series": [{"type": "segments", "label": sp.latex(sp.Eq(expr, 0)), "label_plain": src, "segments": segs}],
            "xlabel": xv, "ylabel": yv, "xrange": [a, b], "yrange": [c, d], "equal": True}


def marching_squares(X, Y, Z, level=0.0):
    """Level-set segments of Z on the grid (linear interpolation on cell edges)."""
    segs = []
    n, m = Z.shape
    Z = Z - level

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


# ---------------------------------------------------------------- grids and 3D
def grid2d(cell, env, ev, default_n, max_n, defaults=("x", "y"), ranges=("-5", "5", "-5", "5")):
    """Sample F(x, y) on a grid: {x, y, z, expr, xv, yv, zlabel, xrange, yrange}."""
    src = (cell.get("exprs") or "").strip()
    xv, yv = two_vars(cell, defaults)
    ns = ev.namespace(env, [xv, yv])
    xs, ys = ns[xv], ns[yv]
    expr = resolve(src, ns, ev, xs, ys)
    f, zlabel = numeric2(expr, xs, ys)
    a, b, c, d = plain_bounds(cell, ns, ev, ranges)
    n = samples(cell, default_n, max_n, 10)
    gx, gy = np.linspace(a, b, n), np.linspace(c, d, n)
    X, Y = np.meshgrid(gx, gy)
    Z = call2(f, X, Y)
    z = [[None if not np.isfinite(v) else float(v) for v in row] for row in Z]
    return {"x": [float(v) for v in gx], "y": [float(v) for v in gy], "z": z, "expr": expr, "xv": xv, "yv": yv,
            "zlabel": zlabel, "xrange": [a, b], "yrange": [c, d]}


def _grid_kind(style, default_n, max_n):
    def fn(cell, env, ev):
        if not (cell.get("exprs") or "").strip():
            return {"series": [], "empty": True}
        g = grid2d(cell, env, ev, default_n, max_n)
        s = {"type": "grid", "style": style, "label": sp.latex(g["expr"]), "label_plain": cell["exprs"].strip(),
             "x": g["x"], "y": g["y"], "z": g["z"]}
        res = {"series": [s], "xlabel": g["xv"], "ylabel": g["yv"], "zlabel": f"[{g['zlabel']}]" if g["zlabel"] else "",
               "xrange": g["xrange"], "yrange": g["yrange"], "three": style == "surface"}
        if style == "contour":
            lv = (cell.get("expr2") or "").strip()
            if lv:
                v = parse(lv, ev.namespace(env), ev.unit_names)
                if isinstance(v, (list, tuple)):
                    res["levels"] = [float(x) for x in v]
                else:
                    res["levels"] = max(2, min(int(v), 60))
            else:
                res["levels"] = 12
        return res
    return fn


def _curve3d(cell, env, ev):
    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    parts = [p for p in split_top(src, ",") if p.strip()]
    if len(parts) != 3:
        raise QuireError("A 3D curve needs three expressions: x(t), y(t), z(t).")
    var = (cell.get("var") or "").strip() or infer_var(parts, env, ev, ("t", "x", "theta"))
    ns = ev.namespace(env, [var])
    ts = ns[var]
    a = float(bound(cell.get("xmin"), "0", ns, ev, "from"))
    b = float(bound(cell.get("xmax"), "2 pi", ns, ev, "to"))
    n = samples(cell, 400, MAX_POINTS)
    grid = np.linspace(a, b, n)
    exprs = [resolve(p, ns, ev, ts) for p in parts]
    fns = [numeric(e, ts) for e in exprs]
    vals = [clean(call(f, grid), n) for f, _ in fns]
    s = {"type": "line3d", "label": "(" + ",\\ ".join(sp.latex(e) for e in exprs) + ")", "label_plain": f"({src})",
         "x": vals[0], "y": vals[1], "z": vals[2]}
    return {"var": var, "series": [s], "xlabel": f"[{fns[0][1]}]" if fns[0][1] else "x", "ylabel": f"[{fns[1][1]}]" if fns[1][1] else "y",
            "zlabel": f"[{fns[2][1]}]" if fns[2][1] else "z", "three": True, "equal": True}


# ---------------------------------------------------------------- shapes
def _shape_series(obj, label):
    """Series for a sympy.geometry entity (or a list of them)."""
    from sympy import geometry as g

    if isinstance(obj, (list, tuple, sp.FiniteSet)):
        out = []
        for k, item in enumerate(obj):
            out.extend(_shape_series(item, f"{label}[{k}]"))
        return out
    if isinstance(obj, g.Point):
        named = re.fullmatch(r"[A-Za-z_]\w*", label) is not None
        return [{"type": "points", "label": sp.latex(sp.Symbol(label)) if named else sp.latex(obj), "label_plain": label,
                 "x": [float(obj.x)], "y": [float(obj.y)], "labels": [label if named else f"({obj.x}, {obj.y})"]}]
    if isinstance(obj, g.Segment):
        p, q = obj.points
        return [{"type": "line", "label": label, "label_plain": label, "x": [float(p.x), float(q.x)], "y": [float(p.y), float(q.y)]}]
    if isinstance(obj, g.Polygon):
        pts = list(obj.vertices) + [obj.vertices[0]]
        return [{"type": "line", "label": label, "label_plain": label, "x": [float(p.x) for p in pts], "y": [float(p.y) for p in pts]}]
    if isinstance(obj, g.Ellipse):  # Circle is an Ellipse
        t = np.linspace(0, 2 * np.pi, 181)
        cx, cy = float(obj.center.x), float(obj.center.y)
        a, b = float(obj.hradius), float(obj.vradius)
        return [{"type": "line", "label": label, "label_plain": label, "x": [cx + a * math.cos(v) for v in t], "y": [cy + b * math.sin(v) for v in t]}]
    if isinstance(obj, g.Line):
        p1, p2 = obj.points
        dx, dy = float(p2.x - p1.x), float(p2.y - p1.y)
        norm = math.hypot(dx, dy) or 1.0
        dx, dy = dx / norm, dy / norm
        L = 1000.0
        if isinstance(obj, g.Ray):
            xs = [float(p1.x), float(p1.x) + L * dx]
            ys = [float(p1.y), float(p1.y) + L * dy]
        else:
            xs = [float(p1.x) - L * dx, float(p1.x) + L * dx]
            ys = [float(p1.y) - L * dy, float(p1.y) + L * dy]
        return [{"type": "line", "label": label, "label_plain": label, "x": xs, "y": ys, "unbounded": True}]
    raise QuireError(f"'{label}' is not a shape that can be drawn.")


def _shapes(cell, env, ev):
    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    ns = ev.namespace(env)
    series = []
    for part in [p for p in split_top(src, ",") if p.strip()]:
        obj = parse(part, ns, ev.unit_names)
        if callable(obj) and not isinstance(obj, sp.Basic):
            raise QuireError(f"'{part.strip()}' is a function, not a shape.")
        series.extend(_shape_series(obj, part.strip()))
    if not series:
        return {"series": [], "empty": True}
    xs = [x for s_ in series if not s_.get("unbounded") for x in s_["x"]]
    ys = [y for s_ in series if not s_.get("unbounded") for y in s_["y"]]
    if not xs:
        xs = [x for s_ in series for x in s_["x"][:1]]
        ys = [y for s_ in series for y in s_["y"][:1]]
    pad = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    xr = [float(cell.get("xmin") or min(xs) - pad), float(cell.get("xmax") or max(xs) + pad)]
    yr = [float(cell.get("ymin") or min(ys) - pad), float(cell.get("ymax") or max(ys) + pad)]
    return {"series": series, "xlabel": "x", "ylabel": "y", "xrange": xr, "yrange": yr, "equal": True}


# ---------------------------------------------------------------- registry of core kinds
register_kind("function", _function, label="y = f(x)", f1="y =", var="variable", range="x", ph1="sin(x), x^2/10",
              annot=True, doc="one or more expressions against a variable; sampled adaptively, gaps at poles")
register_kind("parametric", _parametric, label="parametric", f1="x(t) =", f2="y(t) =", var="variable", range="t",
              ph1="cos(3 t)", ph2="sin(2 t)", annot=True, doc="a curve (x(t), y(t)) with equal scales")
register_kind("polar", _polar, label="polar r(θ)", f1="r(θ) =", var="variable", range="θ", ph1="1 + cos(theta)",
              annot=True, doc="r as a function of the angle")
register_kind("scatter", _scatter, label="scatter (data)", f1="x data", f2="y data", f3="y errors",
              ph1="[1, 2, 3]", ph2="[2, 4, 6.5]", ph3="", annot=True, doc="points from lists, with optional error bars")
register_kind("slope", _slope, label="slope field", f1="dy/dx =", var="variables", range="x", yrange=True, ph1="x - y",
              samples="20", doc="direction field of dy/dx = f(x, y)")
register_kind("implicit", _implicit, label="implicit F(x,y)=0", f1="equation", var="variables", range="x", yrange=True,
              ph1="x^2 + y^2 == 4", samples="200", annot=True, doc="the curve where an equation holds")
register_kind("shapes", _shapes, label="shapes (geometry)", f1="draw", range="x", yrange=True,
              ph1="circle(point(0, 0), 2), triangle(point(-1, -1), point(2, 0), point(0, 1))", annot=True,
              doc="geometry objects; named points are labelled")
register_kind("contour", _grid_kind("contour", 80, 250), label="contour F(x,y)", f1="F(x, y) =", f2="levels",
              var="variables", range="x", yrange=True, ph1="sin(x) cos(y)", ph2="12", renderer="plotly", samples="80",
              doc="level curves of a two-variable function, or of a solution such as u(x, t)")
register_kind("heatmap", _grid_kind("heatmap", 80, 250), label="heatmap F(x,y)", f1="F(x, y) =", var="variables",
              range="x", yrange=True, ph1="exp(-(x^2 + y^2)/4)", renderer="plotly", samples="80",
              doc="a two-variable function as colours")
register_kind("surface", _grid_kind("surface", 50, 150), label="3D surface z = F(x,y)", f1="z =", var="variables",
              range="x", yrange=True, ph1="sin(x) cos(y)", renderer="plotly", samples="50",
              doc="a surface you can rotate")
register_kind("curve3d", _curve3d, label="3D curve", f1="x(t), y(t), z(t) =", var="variable", range="t",
              ph1="cos(t), sin(t), t/5", renderer="plotly", doc="a space curve (x(t), y(t), z(t))")
