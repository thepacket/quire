"""Structural analysis: plane trusses by the method of joints and plane frames by the direct stiffness method.

    T = truss([[0, 0], [4 m, 0], [2 m, 3 m]], [[1, 2], [2, 3], [1, 3]], [[1, pin], [2, roller]], [[3, 0, -10 kN]])
    truss_forces(T)        member forces, + tension, - compression
    truss_reactions(T)     support reactions
    F = frame(nodes, members, supports, loads, E, A, I)   displacements, reactions, member end forces

Node numbers start at 1. Supports: pin (both directions), roller (vertical), roller_x
(horizontal), fixed (frames: both directions and rotation). Loads are [node, Fx, Fy] or,
for frames, [node, Fx, Fy, M]. The "structure" plot kind draws the result.
"""
from __future__ import annotations

import math

import numpy as np
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "structural"
DESCRIPTION = "Plane trusses (method of joints) and frames (stiffness method), with a drawn result."

SUPPORTS = {"pin": (1, 1, 0), "roller": (0, 1, 0), "roller_x": (1, 0, 0), "fixed": (1, 1, 1), "free": (0, 0, 0)}


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _rows(x, what):
    if isinstance(x, sp.MatrixBase):
        x = x.tolist()
    if not isinstance(x, (list, tuple)) or not x:
        raise EvalError(f"{what} must be a list of rows, e.g. [[0, 0], [4 m, 0]].")
    return [list(r) if isinstance(r, (list, tuple)) else [r] for r in x]


def _len(v):
    return U.si_value(v, u.meter, "a coordinate") if U.has_units(v) else float(v)


def _force(v):
    return U.si_value(v, u.newton, "a force") if U.has_units(v) else float(v)


def _moment(v):
    return U.si_value(v, u.newton * u.meter, "a moment") if U.has_units(v) else float(v)


def _index(v, n, what):
    try:
        i = int(v)
    except (TypeError, ValueError):
        raise EvalError(f"{what}: node numbers are integers starting at 1.") from None
    if not 1 <= i <= n:
        raise EvalError(f"{what}: node {i} does not exist (there are {n}).")
    return i - 1


def _support_kind(v):
    name = str(v)
    if name not in SUPPORTS:
        raise EvalError(f"Unknown support '{name}'. Use pin, roller, roller_x or fixed.")
    return name


class Structure:
    """Solved truss or frame: geometry plus member forces, reactions and (frames) displacements."""

    kind = "truss"

    def __init__(self, nodes, members, supports, loads):
        self.nodes, self.members, self.supports, self.loads = nodes, members, supports, loads
        self.forces: list[float] = []          # axial force per member, + tension
        self.reactions: dict[int, list] = {}   # node -> [Rx, Ry, M]
        self.displacements: list[list[float]] = []
        self.end_forces: list[list[float]] = []

    def __repr__(self):
        big = max((abs(f) for f in self.forces), default=0.0)
        return (f"{self.kind}: {len(self.nodes)} nodes, {len(self.members)} members, "
                f"max |N| = {big / 1000:.4g} kN")


def _parse_structure(nodes, members, supports, loads, frame=False):
    nd = [[_len(r[0]), _len(r[1])] for r in _rows(nodes, "nodes")]
    n = len(nd)
    mb = []
    for r in _rows(members, "members"):
        if len(r) < 2:
            raise EvalError("Each member is [node i, node j].")
        mb.append((_index(r[0], n, "members"), _index(r[1], n, "members")))
    sp_ = {}
    for r in _rows(supports, "supports"):
        if len(r) < 2:
            raise EvalError("Each support is [node, pin | roller | roller_x | fixed].")
        sp_[_index(r[0], n, "supports")] = _support_kind(r[1])
    ld = {}
    for r in _rows(loads, "loads"):
        if len(r) < 3:
            raise EvalError("Each load is [node, Fx, Fy] (frames: [node, Fx, Fy, M]).")
        i = _index(r[0], n, "loads")
        fx, fy = _force(r[1]), _force(r[2])
        m = _moment(r[3]) if (frame and len(r) > 3) else 0.0
        old = ld.get(i, [0.0, 0.0, 0.0])
        ld[i] = [old[0] + fx, old[1] + fy, old[2] + m]
    return nd, mb, sp_, ld


def truss(nodes, members, supports, loads):
    """Solve a statically determinate plane truss by the method of joints."""
    nd, mb, sup, ld = _parse_structure(nodes, members, supports, loads)
    n, m = len(nd), len(mb)
    reactions = []  # (node, direction 0=x 1=y)
    for i, kind in sup.items():
        rx, ry, rm = SUPPORTS[kind]
        if rm:
            raise EvalError("A truss has pin and roller supports; use frame(...) for fixed supports.")
        if rx:
            reactions.append((i, 0))
        if ry:
            reactions.append((i, 1))
    r = len(reactions)
    if m + r != 2 * n:
        raise EvalError(f"Not statically determinate: {m} members + {r} reactions ≠ 2 × {n} joints"
                        f"{' (use frame(...) with E, A, I for an indeterminate structure)' if m + r > 2 * n else ' (mechanism)'}.")
    A = np.zeros((2 * n, m + r))
    b = np.zeros(2 * n)
    for k, (i, j) in enumerate(mb):
        dx, dy = nd[j][0] - nd[i][0], nd[j][1] - nd[i][1]
        L = math.hypot(dx, dy)
        if L == 0:
            raise EvalError(f"Member {k + 1} has zero length.")
        cx, cy = dx / L, dy / L
        A[2 * i, k] += cx; A[2 * i + 1, k] += cy      # tension pulls node i towards j
        A[2 * j, k] -= cx; A[2 * j + 1, k] -= cy
    for k, (i, d) in enumerate(reactions):
        A[2 * i + d, m + k] = 1.0
    for i, (fx, fy, _) in ld.items():
        b[2 * i] -= fx
        b[2 * i + 1] -= fy
    if abs(np.linalg.det(A)) < 1e-12 * max(1.0, np.abs(A).max() ** (2 * n)):
        raise EvalError("The truss is a mechanism (singular equilibrium system): check members and supports.")
    x = np.linalg.solve(A, b)
    x[np.abs(x) < 1e-9 * max(1.0, np.abs(x).max())] = 0.0  # numerical dust
    S = Structure(nd, mb, sup, ld)
    S.forces = [float(v) for v in x[:m]]
    for k, (i, d) in enumerate(reactions):
        S.reactions.setdefault(i, [0.0, 0.0, 0.0])[d] = float(x[m + k])
    _note(f"method of joints: {2 * n} equilibrium equations, {m} member forces and {r} reactions")
    return S


def _stiffness(nd, mb, E, A, I):
    n = len(nd)
    K = np.zeros((3 * n, 3 * n))
    elems = []
    for (i, j) in mb:
        dx, dy = nd[j][0] - nd[i][0], nd[j][1] - nd[i][1]
        L = math.hypot(dx, dy)
        if L == 0:
            raise EvalError("A member has zero length.")
        c, s = dx / L, dy / L
        a, b_, d = E * A / L, 12 * E * I / L ** 3, 6 * E * I / L ** 2
        e = 4 * E * I / L, 2 * E * I / L
        k = np.array([[a, 0, 0, -a, 0, 0],
                      [0, b_, d, 0, -b_, d],
                      [0, d, e[0], 0, -d, e[1]],
                      [-a, 0, 0, a, 0, 0],
                      [0, -b_, -d, 0, b_, -d],
                      [0, d, e[1], 0, -d, e[0]]])
        T = np.array([[c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, c, s, 0], [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1]])
        kg = T.T @ k @ T
        dofs = [3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2]
        for p in range(6):
            for q in range(6):
                K[dofs[p], dofs[q]] += kg[p, q]
        elems.append((dofs, k, T, L))
    return K, elems


def frame(nodes, members, supports, loads, E, A, I):
    """Plane frame by the direct stiffness method (rigid joints, nodal loads)."""
    nd, mb, sup, ld = _parse_structure(nodes, members, supports, loads, frame=True)
    Ev = U.si_value(E, u.pascal, "E") if U.has_units(E) else float(E)
    Av = U.si_value(A, u.meter ** 2, "A") if U.has_units(A) else float(A)
    Iv = U.si_value(I, u.meter ** 4, "I") if U.has_units(I) else float(I)
    n = len(nd)
    K, elems = _stiffness(nd, mb, Ev, Av, Iv)
    F = np.zeros(3 * n)
    for i, (fx, fy, m) in ld.items():
        F[3 * i:3 * i + 3] += [fx, fy, m]
    fixed = []
    for i, kind in sup.items():
        for d, flag in enumerate(SUPPORTS[kind]):
            if flag:
                fixed.append(3 * i + d)
    free = [d for d in range(3 * n) if d not in fixed]
    if not free:
        raise EvalError("Every degree of freedom is fixed.")
    Kff = K[np.ix_(free, free)]
    if np.linalg.cond(Kff) > 1e14:
        raise EvalError("The frame is a mechanism (singular stiffness matrix): check the supports.")
    d = np.zeros(3 * n)
    d[free] = np.linalg.solve(Kff, F[free])
    R = K @ d - F
    R[np.abs(R) < 1e-9 * max(1.0, np.abs(R).max())] = 0.0
    S = Structure(nd, mb, sup, ld)
    S.kind = "frame"
    S.displacements = [[float(d[3 * i]), float(d[3 * i + 1]), float(d[3 * i + 2])] for i in range(n)]
    for i in sup:
        S.reactions[i] = [float(R[3 * i]), float(R[3 * i + 1]), float(R[3 * i + 2])]
    for dofs, k, T, L in elems:
        f_local = k @ (T @ d[dofs])
        S.end_forces.append([float(v) for v in f_local])
        S.forces.append(float(-f_local[0]))  # axial: + tension
    _note(f"direct stiffness method: {len(free)} free degrees of freedom, {len(fixed)} restrained")
    return S


def _structure(x):
    if not isinstance(x, Structure):
        raise EvalError("Give the result of truss(...) or frame(...).")
    return x


def truss_forces(S):
    """Member axial forces (+ tension, - compression)."""
    S = _structure(S)
    return [sp.Float(f, 6) * u.newton for f in S.forces]


def truss_reactions(S):
    """[node, Rx, Ry] per support (frames add the moment)."""
    S = _structure(S)
    out = []
    for i in sorted(S.reactions):
        rx, ry, m = S.reactions[i]
        row = [sp.Integer(i + 1), sp.Float(rx, 6) * u.newton, sp.Float(ry, 6) * u.newton]
        if S.kind == "frame":
            row.append(sp.Float(m, 6) * u.newton * u.meter)
        out.append(row)
    return out


def frame_displacements(S):
    """[ux, uy, rotation] per node."""
    S = _structure(S)
    if S.kind != "frame":
        raise EvalError("Displacements come from frame(...).")
    return [[sp.Float(dx, 6) * u.meter, sp.Float(dy, 6) * u.meter, sp.Float(r, 6)] for dx, dy, r in S.displacements]


def frame_end_forces(S):
    """[N_i, V_i, M_i, N_j, V_j, M_j] per member in member coordinates."""
    S = _structure(S)
    if S.kind != "frame":
        raise EvalError("End forces come from frame(...).")
    return [[sp.Float(v, 6) * (u.newton * u.meter if k in (2, 5) else u.newton) for k, v in enumerate(row)] for row in S.end_forces]


def max_member_force(S):
    S = _structure(S)
    return sp.Float(max((abs(f) for f in S.forces), default=0.0), 6) * u.newton


def _structure_plot(cell, env, ev):
    """Plot kind: members coloured by tension/compression, supports, loads, forces."""
    from quire.engine.parser import parse

    src = (cell.get("exprs") or "").strip()
    if not src:
        return {"series": [], "empty": True}
    S = _structure(parse(src, ev.namespace(env), ev.unit_names))
    nd = S.nodes
    xs, ys = [p[0] for p in nd], [p[1] for p in nd]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    big = max((abs(f) for f in S.forces), default=0.0) or 1.0
    comp, tens, slack = [], [], []
    annotations = []
    for k, (i, j) in enumerate(S.members):
        seg = [nd[i][0], nd[i][1], nd[j][0], nd[j][1]]
        f = S.forces[k]
        (tens if f > 1e-9 * big else comp if f < -1e-9 * big else slack).append(seg)
        annotations.append({"type": "text", "x": (nd[i][0] + nd[j][0]) / 2, "y": (nd[i][1] + nd[j][1]) / 2 + 0.03 * span,
                            "label": f"{f / 1000:+.3g} kN" if abs(f) >= 1000 else f"{f:+.3g} N"})
    series = []
    if comp:
        series.append({"type": "segments", "label": r"\text{compression}", "label_plain": "compression", "segments": comp})
    if tens:
        series.append({"type": "segments", "label": r"\text{tension}", "label_plain": "tension", "segments": tens})
    if slack:
        series.append({"type": "segments", "label": r"\text{zero force}", "label_plain": "zero force", "segments": slack})
    if S.kind == "frame" and S.displacements:
        dmax = max((math.hypot(d[0], d[1]) for d in S.displacements), default=0.0)
        scale = 0.08 * span / dmax if dmax > 0 else 0.0
        px, py = [], []
        for (i, j) in S.members:
            px += [nd[i][0] + scale * S.displacements[i][0], nd[j][0] + scale * S.displacements[j][0], None]
            py += [nd[i][1] + scale * S.displacements[i][1], nd[j][1] + scale * S.displacements[j][1], None]
        series.append({"type": "line", "label": rf"\text{{deformed shape}} \times {scale:.3g}", "label_plain": "deformed shape", "x": px, "y": py})
    arrows = []
    for i, (fx, fy, _) in S.loads.items():
        L = math.hypot(fx, fy)
        if L == 0:
            continue
        a = 0.18 * span
        arrows.append([nd[i][0] - fx / L * a, nd[i][1] - fy / L * a, nd[i][0], nd[i][1]])
        annotations.append({"type": "text", "x": nd[i][0] - fx / L * a * 1.25, "y": nd[i][1] - fy / L * a * 1.25,
                            "label": f"{L / 1000:.3g} kN" if L >= 1000 else f"{L:.3g} N"})
    if arrows:
        series.append({"type": "segments", "arrows": True, "label": r"\text{loads}", "label_plain": "loads", "segments": arrows})
    sx, sy = [], []
    for i, kind in S.supports.items():
        sx.append(nd[i][0]); sy.append(nd[i][1])
        annotations.append({"type": "text", "x": nd[i][0], "y": nd[i][1] - 0.07 * span, "label": kind})
    if sx:
        series.append({"type": "points", "marker": "o", "size": 6, "label": r"\text{supports}", "label_plain": "supports", "x": sx, "y": sy})
    series.append({"type": "points", "label": r"\text{nodes}", "label_plain": "nodes", "x": xs, "y": ys,
                   "labels": [str(i + 1) for i in range(len(nd))]})
    pad = 0.25 * span
    return {"series": series, "annotations": annotations, "xlabel": "x [m]", "ylabel": "y [m]", "equal": True,
            "xrange": [min(xs) - pad, max(xs) + pad], "yrange": [min(ys) - pad, max(ys) + pad]}


def register(api):
    api.plot_kind("structure", _structure_plot, label="structure (truss / frame)", f1="structure", ph1="T",
                  doc="a solved truss or frame: members coloured by tension and compression, loads, supports, forces")
    C = "Structural"
    for name in SUPPORTS:
        api.constant(name, sp.Symbol(name), doc="support type for truss / frame", category=C)
    api.function("truss", truss, signature="truss(nodes, members, supports, loads)",
                 doc="solve a determinate plane truss (nodes [[x, y]...], members [[i, j]...], supports [[node, pin | roller]...], loads [[node, Fx, Fy]...])",
                 category=C, example="truss([[0, 0], [4 m, 0], [2 m, 3 m]], [[1, 2], [2, 3], [1, 3]], [[1, pin], [2, roller]], [[3, 0, -10 kN]])")
    api.function("frame", frame, signature="frame(nodes, members, supports, loads, E, A, I)",
                 doc="plane frame by the stiffness method; supports may be fixed; loads [node, Fx, Fy, M]", category=C,
                 example="frame([[0, 0], [0, 3 m], [4 m, 3 m], [4 m, 0]], [[1, 2], [2, 3], [3, 4]], [[1, fixed], [4, fixed]], [[2, 10 kN, 0, 0]], 200 GPa, 0.01 m^2, 2e-5 m^4)")
    api.function("truss_forces", truss_forces, signature="truss_forces(S)", doc="member axial forces, + tension", category=C)
    api.function("truss_reactions", truss_reactions, signature="truss_reactions(S)", doc="[node, Rx, Ry(, M)] per support", category=C)
    api.function("frame_displacements", frame_displacements, signature="frame_displacements(F)", doc="[ux, uy, rotation] per node", category=C)
    api.function("frame_end_forces", frame_end_forces, signature="frame_end_forces(F)", doc="[N, V, M] at both ends of each member", category=C)
    api.function("max_member_force", max_member_force, signature="max_member_force(S)", doc="largest |axial force|", category=C)
