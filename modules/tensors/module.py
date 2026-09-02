"""Differential geometry: metrics, Christoffel symbols, Riemann and Ricci curvature, geodesics.

A metric is a symmetric matrix g_ij in the coordinates [x1, x2, ...]. Index order: christoffel(g, coords)
returns Gamma[k] as the matrix Gamma^k_ij; riemann gives R^rho_{sigma mu nu}.
"""
import sympy as sp

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "tensors"
DESCRIPTION = "Metrics, Christoffel symbols, Riemann/Ricci curvature, scalar curvature, geodesic equations."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _coords(coords):
    cs = list(coords) if isinstance(coords, (list, tuple)) else [coords]
    for c in cs:
        if not isinstance(c, sp.Symbol):
            raise EvalError(f"'{c}' is not a coordinate name.")
    return cs


def _g(g, coords):
    g = sp.Matrix(g)
    n = len(_coords(coords))
    if g.shape != (n, n):
        raise EvalError(f"The metric must be {n}x{n} for {n} coordinates.")
    return g


def inverse_metric(g, coords):
    return sp.ImmutableMatrix(_g(g, coords).inv()).applyfunc(sp.simplify)


def line_element(g, coords):
    """ds^2 = g_ij dx^i dx^j with differentials written as d_x."""
    cs = _coords(coords)
    G = _g(g, coords)
    d = [sp.Symbol(f"d{c.name}") for c in cs]
    return sp.expand(sum(G[i, j] * d[i] * d[j] for i in range(len(cs)) for j in range(len(cs))))


def _gamma(G, cs):
    n = len(cs)
    Ginv = G.inv()
    Gam = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                Gam[k][i][j] = sp.simplify(sum(Ginv[k, l] * (sp.diff(G[l, i], cs[j]) + sp.diff(G[l, j], cs[i]) - sp.diff(G[i, j], cs[l]))
                                               for l in range(n)) / 2)
    return Gam


def christoffel(g, coords):
    """List of matrices: christoffel(g, coords)[k] is Gamma^k_ij."""
    cs = _coords(coords)
    Gam = _gamma(_g(g, coords), cs)
    _note("Gamma^k_ij = 1/2 g^kl (d_j g_li + d_i g_lj - d_l g_ij); index k selects the matrix")
    return [sp.ImmutableMatrix(m) for m in Gam]


def _riemann(G, cs):
    n = len(cs)
    Gam = _gamma(G, cs)
    R = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for s in range(n):
            for m in range(n):
                for v in range(n):
                    val = sp.diff(Gam[r][v][s], cs[m]) - sp.diff(Gam[r][m][s], cs[v])
                    val += sum(Gam[r][m][l] * Gam[l][v][s] - Gam[r][v][l] * Gam[l][m][s] for l in range(n))
                    R[r][s][m][v] = sp.simplify(val)
    return R


def riemann(g, coords):
    """Nonzero components as rows [rho, sigma, mu, nu, R^rho_{sigma mu nu}] (indices from 1)."""
    cs = _coords(coords)
    R = _riemann(_g(g, coords), cs)
    n = len(cs)
    rows = [[sp.Integer(r + 1), sp.Integer(s + 1), sp.Integer(m + 1), sp.Integer(v + 1), R[r][s][m][v]]
            for r in range(n) for s in range(n) for m in range(n) for v in range(n) if R[r][s][m][v] != 0]
    _note("columns: rho, sigma, mu, nu, R^rho_{sigma mu nu}; only nonzero components")
    return sp.ImmutableMatrix(rows) if rows else sp.Symbol("flat")


def ricci(g, coords):
    cs = _coords(coords)
    R = _riemann(_g(g, coords), cs)
    n = len(cs)
    return sp.ImmutableMatrix(n, n, lambda i, j: sp.simplify(sum(R[r][i][r][j] for r in range(n))))


def ricci_scalar(g, coords):
    cs = _coords(coords)
    G = _g(g, coords)
    Ric = ricci(G, cs)
    Ginv = G.inv()
    n = len(cs)
    return sp.simplify(sum(Ginv[i, j] * Ric[i, j] for i in range(n) for j in range(n)))


def gaussian_curvature(g, coords):
    """K = R/2 for a two-dimensional metric."""
    if len(_coords(coords)) != 2:
        raise EvalError("Gaussian curvature is for two-dimensional metrics.")
    return sp.simplify(ricci_scalar(g, coords) / 2)


def geodesic_equations(g, coords, t):
    """d²x^k/dt² + Gamma^k_ij dx^i/dt dx^j/dt = 0 as a list of equations in functions x^k(t)."""
    cs = _coords(coords)
    G = _g(g, coords)
    Gam = _gamma(G, cs)
    fs = [sp.Function(c.name)(t) for c in cs]
    rep = dict(zip(cs, fs))
    n = len(cs)
    eqs = []
    for k in range(n):
        acc = sp.diff(fs[k], t, 2) + sum(Gam[k][i][j].subs(rep) * sp.diff(fs[i], t) * sp.diff(fs[j], t) for i in range(n) for j in range(n))
        eqs.append(sp.Eq(sp.simplify(acc), 0))
    return eqs


def metric_sphere(R, theta, phi):
    return sp.ImmutableMatrix([[R ** 2, 0], [0, R ** 2 * sp.sin(theta) ** 2]])


def metric_polar(r, theta):
    return sp.ImmutableMatrix([[1, 0], [0, r ** 2]])


def metric_schwarzschild(M, t, r, theta, phi):
    """Schwarzschild metric with G = c = 1, signature (-, +, +, +)."""
    f = 1 - 2 * M / r
    return sp.ImmutableMatrix([[-f, 0, 0, 0], [0, 1 / f, 0, 0], [0, 0, r ** 2, 0], [0, 0, 0, r ** 2 * sp.sin(theta) ** 2]])


def metric_minkowski(t, x, y, z):
    return sp.ImmutableMatrix(sp.diag(-1, 1, 1, 1))


def register(api):
    T = "Tensors & curvature"
    api.function("metric_sphere", metric_sphere, signature="metric_sphere(R, theta, phi)", doc="sphere of radius R", category=T,
                 example="metric_sphere(R, theta, phi)")
    api.function("metric_polar", metric_polar, signature="metric_polar(r, theta)", doc="plane in polar coordinates", category=T)
    api.function("metric_schwarzschild", metric_schwarzschild, signature="metric_schwarzschild(M, t, r, theta, phi)", doc="Schwarzschild (G = c = 1)", category=T)
    api.function("metric_minkowski", metric_minkowski, signature="metric_minkowski(t, x, y, z)", doc="flat spacetime", category=T)
    api.function("inverse_metric", inverse_metric, signature="inverse_metric(g, coords)", doc="g^ij", category=T)
    api.function("line_element", line_element, signature="line_element(g, coords)", doc="ds²", category=T)
    api.function("christoffel", christoffel, signature="christoffel(g, [coords])", doc="Γ^k_ij as a list of matrices", category=T,
                 example="christoffel(metric_polar(r, theta), [r, theta])")
    api.function("riemann", riemann, signature="riemann(g, [coords])", doc="nonzero R^ρ_σμν", category=T)
    api.function("ricci", ricci, signature="ricci(g, [coords])", doc="Ricci tensor", category=T,
                 example="ricci(metric_schwarzschild(M, t, r, theta, phi), [t, r, theta, phi])")
    api.function("ricci_scalar", ricci_scalar, signature="ricci_scalar(g, [coords])", doc="scalar curvature", category=T)
    api.function("gaussian_curvature", gaussian_curvature, signature="gaussian_curvature(g, [coords])", doc="K of a surface", category=T,
                 example="gaussian_curvature(metric_sphere(R, theta, phi), [theta, phi])")
    api.function("geodesic_equations", geodesic_equations, signature="geodesic_equations(g, [coords], t)", doc="geodesic ODEs", category=T)
