"""Mechanics of materials and machine design: sections, stresses, beams, buckling, vibration, fatigue."""
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "mechanics"
DESCRIPTION = "Section properties, stress and strain, beam formulas, buckling, vibration, fatigue."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


# ---- sections
def I_rect(b, h):
    """Second moment of area of a rectangle about its centroid, b h^3/12."""
    return sp.simplify(b * h ** 3 / 12)


def I_circle(d):
    return sp.simplify(sp.pi * d ** 4 / 64)


def I_tube(d_outer, d_inner):
    return sp.simplify(sp.pi * (d_outer ** 4 - d_inner ** 4) / 64)


def I_ibeam(b, h, t_web, t_flange):
    """I-beam about the strong axis: outer rectangle minus the two web-side voids."""
    return sp.simplify(b * h ** 3 / 12 - (b - t_web) * (h - 2 * t_flange) ** 3 / 12)


def J_circle(d):
    """Polar moment of a solid circle, pi d^4/32."""
    return sp.simplify(sp.pi * d ** 4 / 32)


def area_circle(d):
    return sp.simplify(sp.pi * d ** 2 / 4)


def section_modulus(I, c):
    return sp.simplify(I / c)


def radius_of_gyration(I, A):
    return sp.simplify(sp.sqrt(I / A))


# ---- stress and strain
def stress(F, A):
    return sp.simplify(F / A)


def strain(dL, L):
    return sp.simplify(dL / L)


def hooke(E, eps):
    return sp.simplify(E * eps)


def elongation(F, L, E, A):
    return sp.simplify(F * L / (E * A))


def bending_stress(M, c, I):
    return sp.simplify(M * c / I)


def torsion_stress(T, r, J):
    return sp.simplify(T * r / J)


def twist_angle(T, L, G, J):
    return sp.simplify(T * L / (G * J))


def thermal_stress(E, alpha, dT):
    return sp.simplify(E * alpha * dT)


def principal_stresses(sx, sy, txy):
    """[sigma_1, sigma_2, theta_p in degrees] for plane stress."""
    avg = (sx + sy) / 2
    r = sp.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    theta = sp.atan2(2 * txy, sx - sy) / 2 * 180 / sp.pi
    return [sp.simplify(avg + r), sp.simplify(avg - r), sp.simplify(theta) * u.degree]


def max_shear(sx, sy, txy):
    return sp.simplify(sp.sqrt(((sx - sy) / 2) ** 2 + txy ** 2))


def von_mises(sx, sy, txy, sz=0):
    return sp.simplify(sp.sqrt(((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) / 2 + 3 * txy ** 2))


def safety_factor(strength, stress_):
    return sp.simplify(strength / stress_)


# ---- beams: [deflection(x), moment(x), shear(x)] for standard cases, x from the left support
def beam_ss_point(P, L, E, I, x):
    """Simply supported, point load P at midspan."""
    a = L / 2
    v = sp.Piecewise((P * x * (3 * L ** 2 - 4 * x ** 2) / (48 * E * I), x <= a),
                     (P * (L - x) * (3 * L ** 2 - 4 * (L - x) ** 2) / (48 * E * I), True))
    M = sp.Piecewise((P * x / 2, x <= a), (P * (L - x) / 2, True))
    V = sp.Piecewise((P / 2, x < a), (-P / 2, True))
    _note("simply supported beam, point load at midspan; deflection is downward positive; max deflection P L^3/(48 E I)")
    return [v, M, V]


def beam_ss_udl(w, L, E, I, x):
    """Simply supported, uniform load w per length."""
    v = w * x * (L ** 3 - 2 * L * x ** 2 + x ** 3) / (24 * E * I)
    M = w * x * (L - x) / 2
    V = w * (L / 2 - x)
    _note("simply supported beam, uniform load; max deflection 5 w L^4/(384 E I) at midspan")
    return [sp.simplify(v), sp.simplify(M), sp.simplify(V)]


def beam_cant_point(P, L, E, I, x):
    """Cantilever fixed at x = 0, point load P at the free end."""
    v = P * x ** 2 * (3 * L - x) / (6 * E * I)
    M = -P * (L - x)
    V = P + 0 * x
    _note("cantilever, point load at the tip; tip deflection P L^3/(3 E I)")
    return [sp.simplify(v), sp.simplify(M), sp.simplify(V)]


def beam_cant_udl(w, L, E, I, x):
    v = w * x ** 2 * (6 * L ** 2 - 4 * L * x + x ** 2) / (24 * E * I)
    M = -w * (L - x) ** 2 / 2
    V = w * (L - x)
    _note("cantilever, uniform load; tip deflection w L^4/(8 E I)")
    return [sp.simplify(v), sp.simplify(M), sp.simplify(V)]


# ---- columns, springs, vibration, fatigue
def euler_buckling(E, I, L, K=1):
    """Critical load pi^2 E I/(K L)^2; K = 1 pinned-pinned, 0.5 fixed-fixed, 2 fixed-free, 0.7 fixed-pinned."""
    return sp.simplify(sp.pi ** 2 * E * I / (K * L) ** 2)


def slenderness(L, r, K=1):
    return sp.simplify(K * L / r)


def spring_series(*k):
    return 1 / sp.Add(*[1 / sp.sympify(v) for v in k])


def spring_parallel(*k):
    return sp.Add(*[sp.sympify(v) for v in k])


def natural_frequency(k, m):
    """f_n = sqrt(k/m)/(2 pi)"""
    return sp.simplify(sp.sqrt(k / m) / (2 * sp.pi))


def damping_ratio(c, k, m):
    return sp.simplify(c / (2 * sp.sqrt(k * m)))


def damped_frequency(k, m, c):
    z = damping_ratio(c, k, m)
    return sp.simplify(natural_frequency(k, m) * sp.sqrt(1 - z ** 2))


def transmissibility(r, zeta):
    """Vibration transmissibility at frequency ratio r = f/f_n."""
    return sp.sqrt((1 + (2 * zeta * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2))


def goodman(sigma_a, sigma_m, S_e, S_ut):
    """Modified Goodman fatigue safety factor 1/(sigma_a/S_e + sigma_m/S_ut)."""
    _note("modified Goodman criterion: n = 1 / (σa/Se + σm/Sut)")
    return sp.simplify(1 / (sigma_a / S_e + sigma_m / S_ut))


def soderberg(sigma_a, sigma_m, S_e, S_y):
    return sp.simplify(1 / (sigma_a / S_e + sigma_m / S_y))


def basquin(S_f, b, N):
    """Stress amplitude at N cycles from the Basquin relation S = S_f (2N)^b."""
    return S_f * (2 * N) ** b


def kinetic_energy(m, v):
    return sp.simplify(m * v ** 2 / 2)


def moment(F, r):
    """Moment F r (scalars) or the cross product r x F for 3-vectors."""
    if isinstance(F, (list, tuple, sp.MatrixBase)):
        return sp.ImmutableMatrix(sp.Matrix(list(r)).cross(sp.Matrix(list(F))))
    return sp.simplify(F * r)


def register(api):
    S = "Mechanics: sections"
    api.function("I_rect", I_rect, signature="I_rect(b, h)", doc="b h³/12 about the centroid", category=S, example="I_rect(50 mm, 100 mm)")
    api.function("I_circle", I_circle, signature="I_circle(d)", doc="π d⁴/64", category=S)
    api.function("I_tube", I_tube, signature="I_tube(d_outer, d_inner)", doc="hollow circle", category=S)
    api.function("I_ibeam", I_ibeam, signature="I_ibeam(b, h, t_web, t_flange)", doc="I-section, strong axis", category=S)
    api.function("J_circle", J_circle, signature="J_circle(d)", doc="polar moment π d⁴/32", category=S)
    api.function("area_circle", area_circle, signature="area_circle(d)", doc="π d²/4", category=S)
    api.function("section_modulus", section_modulus, signature="section_modulus(I, c)", doc="I / c", category=S)
    api.function("radius_of_gyration", radius_of_gyration, signature="radius_of_gyration(I, A)", doc="√(I/A)", category=S)
    T = "Mechanics: stress & strain"
    api.function("stress", stress, signature="stress(F, A)", doc="F / A", category=T, example="stress(10 kN, 200 mm^2) -> MPa")
    api.function("strain", strain, signature="strain(dL, L)", doc="ΔL / L", category=T)
    api.function("hooke", hooke, signature="hooke(E, strain)", doc="σ = E ε", category=T)
    api.function("elongation", elongation, signature="elongation(F, L, E, A)", doc="F L/(E A)", category=T)
    api.function("bending_stress", bending_stress, signature="bending_stress(M, c, I)", doc="M c / I", category=T)
    api.function("torsion_stress", torsion_stress, signature="torsion_stress(T, r, J)", doc="T r / J", category=T)
    api.function("twist_angle", twist_angle, signature="twist_angle(T, L, G, J)", doc="T L/(G J) in radians", category=T)
    api.function("thermal_stress", thermal_stress, signature="thermal_stress(E, alpha, dT)", doc="E α ΔT", category=T)
    api.function("principal_stresses", principal_stresses, signature="principal_stresses(sx, sy, txy)",
                 doc="[σ1, σ2, θp] for plane stress", category=T, example="principal_stresses(80 MPa, -20 MPa, 30 MPa)")
    api.function("max_shear", max_shear, signature="max_shear(sx, sy, txy)", doc="maximum in-plane shear", category=T)
    api.function("von_mises", von_mises, signature="von_mises(sx, sy, txy)", doc="equivalent stress", category=T)
    api.function("safety_factor", safety_factor, signature="safety_factor(strength, stress)", doc="strength / stress", category=T)
    B = "Mechanics: beams & columns"
    api.function("beam_ss_point", beam_ss_point, signature="beam_ss_point(P, L, E, I, x)",
                 doc="[deflection, moment, shear] simply supported, midspan point load", category=B,
                 example="beam_ss_point(10 kN, 4 m, 200 GPa, I_rect(50 mm, 100 mm), x)")
    api.function("beam_ss_udl", beam_ss_udl, signature="beam_ss_udl(w, L, E, I, x)", doc="simply supported, uniform load", category=B)
    api.function("beam_cant_point", beam_cant_point, signature="beam_cant_point(P, L, E, I, x)", doc="cantilever, tip load", category=B)
    api.function("beam_cant_udl", beam_cant_udl, signature="beam_cant_udl(w, L, E, I, x)", doc="cantilever, uniform load", category=B)
    api.function("euler_buckling", euler_buckling, signature="euler_buckling(E, I, L, K)", doc="critical load π²EI/(KL)²", category=B,
                 example="euler_buckling(200 GPa, I_circle(20 mm), 2 m) -> kN")
    api.function("slenderness", slenderness, signature="slenderness(L, r, K)", doc="K L / r", category=B)
    V = "Mechanics: vibration & fatigue"
    api.function("spring_series", spring_series, signature="spring_series(k1, k2, ...)", doc="springs in series", category=V)
    api.function("spring_parallel", spring_parallel, signature="spring_parallel(k1, k2, ...)", doc="springs in parallel", category=V)
    api.function("natural_frequency", natural_frequency, signature="natural_frequency(k, m)", doc="√(k/m)/(2π)", category=V,
                 example="natural_frequency(2000 N/m, 5 kg) -> Hz")
    api.function("damping_ratio", damping_ratio, signature="damping_ratio(c, k, m)", doc="c/(2√(km))", category=V)
    api.function("damped_frequency", damped_frequency, signature="damped_frequency(k, m, c)", doc="damped natural frequency", category=V)
    api.function("transmissibility", transmissibility, signature="transmissibility(r, zeta)", doc="force transmissibility", category=V)
    api.function("goodman", goodman, signature="goodman(sigma_a, sigma_m, S_e, S_ut)", doc="Goodman fatigue safety factor", category=V)
    api.function("soderberg", soderberg, signature="soderberg(sigma_a, sigma_m, S_e, S_y)", doc="Soderberg safety factor", category=V)
    api.function("basquin", basquin, signature="basquin(S_f, b, N)", doc="S = S_f (2N)^b", category=V)
    api.function("kinetic_energy", kinetic_energy, signature="kinetic_energy(m, v)", doc="m v²/2", category=V)
    api.function("moment", moment, signature="moment(F, r)", doc="F r, or r × F for vectors", category=V)
