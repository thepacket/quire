"""Physics: kinematics, special relativity, optics, electromagnetism, waves and photons, nuclear decay.

Angles are degrees unless given with units; results carry SI units.
"""
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "physics"
DESCRIPTION = "Kinematics, relativity, optics, electromagnetism, waves, nuclear decay."

c = u.speed_of_light
g0 = u.gee
h = u.planck
k_e = 1 / (4 * sp.pi * u.vacuum_permittivity)
mu0 = u.vacuum_permeability
eps0 = u.vacuum_permittivity


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _ang(a):
    a = sp.sympify(a)
    return U.strip_angles(a) if U.has_units(a) else a * sp.pi / 180


def _deg(x):
    x = sp.sympify(x)
    if x.is_number and not x.free_symbols and not x.is_Rational:
        return sp.Float(sp.N(x * 180 / sp.pi, 15), 10) * u.degree
    return sp.simplify(x * 180 / sp.pi) * u.degree


# ---- kinematics
def projectile_range(v, theta, g=g0):
    return sp.simplify(v ** 2 * sp.sin(2 * _ang(theta)) / g)


def projectile_height(v, theta, g=g0):
    return sp.simplify((v * sp.sin(_ang(theta))) ** 2 / (2 * g))


def projectile_time(v, theta, g=g0):
    return sp.simplify(2 * v * sp.sin(_ang(theta)) / g)


def free_fall_time(height, g=g0):
    return sp.simplify(sp.sqrt(2 * height / g))


def centripetal_force(m, v, r):
    return sp.simplify(m * v ** 2 / r)


def pendulum_period(L, g=g0):
    return sp.simplify(2 * sp.pi * sp.sqrt(L / g))


def momentum(m, v):
    return sp.simplify(m * v)


def work(F, d, theta=0):
    return sp.simplify(F * d * sp.cos(_ang(theta)))


def power_from_work(W, t):
    return sp.simplify(W / t)


# ---- special relativity
def lorentz_factor(v):
    return sp.simplify(1 / sp.sqrt(1 - (v / c) ** 2))


def time_dilation(dt_proper, v):
    return sp.simplify(dt_proper * lorentz_factor(v))


def length_contraction(L_proper, v):
    return sp.simplify(L_proper / lorentz_factor(v))


def rest_energy(m):
    return sp.simplify(m * c ** 2)


def relativistic_energy(m, v):
    return sp.simplify(lorentz_factor(v) * m * c ** 2)


def relativistic_momentum(m, v):
    return sp.simplify(lorentz_factor(v) * m * v)


def velocity_addition(u1, u2):
    """Relativistic sum of collinear velocities."""
    return sp.simplify((u1 + u2) / (1 + u1 * u2 / c ** 2))


def doppler_relativistic(f, v):
    """Observed frequency for a source approaching at v (negative v: receding)."""
    return sp.simplify(f * sp.sqrt((1 + v / c) / (1 - v / c)))


# ---- optics
def thin_lens_image(f, d_object):
    """Image distance from 1/f = 1/d_o + 1/d_i (negative: virtual image)."""
    return sp.simplify(1 / (1 / f - 1 / d_object))


def magnification(d_image, d_object):
    return sp.simplify(-d_image / d_object)


def lensmaker(n, R1, R2):
    """Focal length of a thin lens in air: 1/f = (n - 1)(1/R1 - 1/R2)."""
    return sp.simplify(1 / ((n - 1) * (1 / R1 - 1 / R2)))


def snell(n1, theta1, n2):
    """Refraction angle in degrees."""
    s = sp.simplify(n1 * sp.sin(_ang(theta1)) / n2)
    if s.is_number and abs(float(s)) > 1:
        raise EvalError("Total internal reflection: n1 sin(theta1)/n2 exceeds 1.")
    return _deg(sp.asin(s))


def critical_angle(n1, n2):
    if sp.sympify(n2) > sp.sympify(n1):
        raise EvalError("Total internal reflection needs n1 > n2.")
    return _deg(sp.asin(sp.sympify(n2) / sp.sympify(n1)))


def brewster_angle(n1, n2):
    return _deg(sp.atan(sp.sympify(n2) / sp.sympify(n1)))


def diffraction_angle(order, wavelength, spacing):
    """Grating equation d sin(theta) = m lambda, angle in degrees."""
    s = sp.simplify(order * wavelength / spacing)
    return _deg(sp.asin(s))


def rayleigh_limit(wavelength, aperture):
    """Angular resolution 1.22 lambda / D in degrees."""
    return _deg(1.22 * wavelength / aperture)


def doppler_sound(f, v_source, v_observer, v_sound=343 * u.m / u.s):
    """Observed frequency; positive v_source moves toward the observer, positive v_observer toward the source."""
    return sp.simplify(f * (v_sound + v_observer) / (v_sound - v_source))


# ---- electromagnetism
def coulomb_force(q1, q2, r):
    return sp.simplify(k_e * q1 * q2 / r ** 2)


def electric_field_point(q, r):
    return sp.simplify(k_e * q / r ** 2)


def electric_potential_point(q, r):
    return sp.simplify(k_e * q / r)


def capacitance_parallel(A, d, eps_r=1):
    return sp.simplify(eps_r * eps0 * A / d)


def capacitor_energy(C, V):
    return sp.simplify(C * V ** 2 / 2)


def inductor_energy(L, I):
    return sp.simplify(L * I ** 2 / 2)


def magnetic_field_wire(I, r):
    return sp.simplify(mu0 * I / (2 * sp.pi * r))


def solenoid_field(turns_per_length, I):
    return sp.simplify(mu0 * turns_per_length * I)


def lorentz_force(q, v, B, theta=90):
    return sp.simplify(q * v * B * sp.sin(_ang(theta)))


def cyclotron_frequency(q, B, m):
    return sp.simplify(q * B / (2 * sp.pi * m))


def larmor_radius(m, v, q, B):
    return sp.simplify(m * v / (q * B))


# ---- waves and photons
def wave_speed(f, wavelength):
    return sp.simplify(f * wavelength)


def photon_energy(wavelength):
    return sp.simplify(h * c / wavelength)


def photon_wavelength(E):
    return sp.simplify(h * c / E)


def de_broglie(m, v):
    return sp.simplify(h / (m * v))


def wien_peak(T):
    return sp.simplify(sp.Float(2.897771955e-3) * u.m * u.K / T)


def blackbody_power(A, T, emissivity=1):
    return sp.simplify(emissivity * u.stefan_boltzmann_constant * A * T ** 4)


def standing_wave_frequency(n, L, v):
    """n-th harmonic of a string or open pipe: n v/(2 L)."""
    return sp.simplify(n * v / (2 * L))


def beat_frequency(f1, f2):
    return sp.Abs(sp.sympify(f1) - sp.sympify(f2))


def sound_level_db(I, I0=sp.Float(1e-12) * u.W / u.m ** 2):
    return 10 * sp.log(sp.simplify(I / I0), 10)


# ---- nuclear
def decay(N0, half_life, t):
    return sp.simplify(N0 * 2 ** (-t / half_life))


def decay_constant(half_life):
    return sp.simplify(sp.log(2) / half_life)


def activity(N, half_life):
    return sp.simplify(N * sp.log(2) / half_life)


def age_from_fraction(fraction_remaining, half_life):
    """Time elapsed for the remaining fraction (radiometric dating)."""
    return sp.simplify(-half_life * sp.log(fraction_remaining) / sp.log(2))


def mass_energy(m):
    return sp.simplify(m * c ** 2)


def binding_energy(Z, N, atomic_mass):
    """Nuclear binding energy from the mass defect (masses in u or kg)."""
    m_p = sp.Float(1.007276466621) * u.amu
    m_n = sp.Float(1.00866491595) * u.amu
    m_e = sp.Float(0.000548579909) * u.amu
    defect = Z * (m_p + m_e) + N * m_n - atomic_mass
    _note("mass defect times c^2; atomic masses include Z electrons")
    return sp.simplify(defect * c ** 2)


def register(api):
    K = "Physics: kinematics"
    api.function("projectile_range", projectile_range, signature="projectile_range(v, theta)", doc="v² sin(2θ)/g", category=K,
                 example="projectile_range(20 m/s, 45)")
    api.function("projectile_height", projectile_height, signature="projectile_height(v, theta)", doc="maximum height", category=K)
    api.function("projectile_time", projectile_time, signature="projectile_time(v, theta)", doc="time of flight", category=K)
    api.function("free_fall_time", free_fall_time, signature="free_fall_time(h)", doc="√(2h/g)", category=K)
    api.function("centripetal_force", centripetal_force, signature="centripetal_force(m, v, r)", doc="m v²/r", category=K)
    api.function("pendulum_period", pendulum_period, signature="pendulum_period(L)", doc="2π√(L/g)", category=K, example="pendulum_period(1 m)")
    api.function("momentum", momentum, signature="momentum(m, v)", doc="m v", category=K)
    api.function("work", work, signature="work(F, d, theta)", doc="F d cos θ", category=K)
    R = "Physics: relativity"
    api.function("lorentz_factor", lorentz_factor, signature="lorentz_factor(v)", doc="γ = 1/√(1 - v²/c²)", category=R,
                 example="lorentz_factor(0.8 c_light)")
    api.function("time_dilation", time_dilation, signature="time_dilation(dt, v)", doc="dilated interval γ dt", category=R)
    api.function("length_contraction", length_contraction, signature="length_contraction(L, v)", doc="L/γ", category=R)
    api.function("rest_energy", rest_energy, signature="rest_energy(m)", doc="m c²", category=R, example="rest_energy(1 g) -> J")
    api.function("relativistic_energy", relativistic_energy, signature="relativistic_energy(m, v)", doc="γ m c²", category=R)
    api.function("relativistic_momentum", relativistic_momentum, signature="relativistic_momentum(m, v)", doc="γ m v", category=R)
    api.function("velocity_addition", velocity_addition, signature="velocity_addition(u1, u2)", doc="relativistic velocity sum", category=R)
    api.function("doppler_relativistic", doppler_relativistic, signature="doppler_relativistic(f, v)", doc="relativistic Doppler shift", category=R)
    O = "Physics: optics"
    api.function("thin_lens_image", thin_lens_image, signature="thin_lens_image(f, d_o)", doc="image distance", category=O,
                 example="thin_lens_image(10 cm, 30 cm)")
    api.function("magnification", magnification, signature="magnification(d_i, d_o)", doc="-d_i/d_o", category=O)
    api.function("lensmaker", lensmaker, signature="lensmaker(n, R1, R2)", doc="thin-lens focal length", category=O)
    api.function("snell", snell, signature="snell(n1, theta1, n2)", doc="refraction angle (degrees)", category=O, example="snell(1, 30, 1.5)")
    api.function("critical_angle", critical_angle, signature="critical_angle(n1, n2)", doc="total internal reflection", category=O)
    api.function("brewster_angle", brewster_angle, signature="brewster_angle(n1, n2)", doc="polarising angle", category=O)
    api.function("diffraction_angle", diffraction_angle, signature="diffraction_angle(m, lambda, d)", doc="grating equation", category=O)
    api.function("rayleigh_limit", rayleigh_limit, signature="rayleigh_limit(lambda, D)", doc="1.22 λ/D in degrees", category=O)
    api.function("doppler_sound", doppler_sound, signature="doppler_sound(f, v_source, v_observer)", doc="acoustic Doppler", category=O)
    E = "Physics: electromagnetism"
    api.function("coulomb_force", coulomb_force, signature="coulomb_force(q1, q2, r)", doc="k q1 q2/r²", category=E,
                 example="coulomb_force(1 uC, 1 uC, 1 cm) -> N")
    api.function("electric_field_point", electric_field_point, signature="electric_field_point(q, r)", doc="k q/r²", category=E)
    api.function("electric_potential_point", electric_potential_point, signature="electric_potential_point(q, r)", doc="k q/r", category=E)
    api.function("capacitance_parallel", capacitance_parallel, signature="capacitance_parallel(A, d, eps_r)", doc="ε A/d", category=E)
    api.function("capacitor_energy", capacitor_energy, signature="capacitor_energy(C, V)", doc="½ C V²", category=E)
    api.function("inductor_energy", inductor_energy, signature="inductor_energy(L, I)", doc="½ L I²", category=E)
    api.function("magnetic_field_wire", magnetic_field_wire, signature="magnetic_field_wire(I, r)", doc="μ₀ I/(2π r)", category=E)
    api.function("solenoid_field", solenoid_field, signature="solenoid_field(n, I)", doc="μ₀ n I", category=E)
    api.function("lorentz_force", lorentz_force, signature="lorentz_force(q, v, B, theta)", doc="q v B sin θ", category=E)
    api.function("cyclotron_frequency", cyclotron_frequency, signature="cyclotron_frequency(q, B, m)", doc="q B/(2π m)", category=E)
    api.function("larmor_radius", larmor_radius, signature="larmor_radius(m, v, q, B)", doc="m v/(q B)", category=E)
    W = "Physics: waves & photons"
    api.function("wave_speed", wave_speed, signature="wave_speed(f, lambda)", doc="f λ", category=W)
    api.function("photon_energy", photon_energy, signature="photon_energy(lambda)", doc="h c/λ", category=W, example="photon_energy(500 nm) -> eV")
    api.function("photon_wavelength", photon_wavelength, signature="photon_wavelength(E)", doc="h c/E", category=W)
    api.function("de_broglie", de_broglie, signature="de_broglie(m, v)", doc="h/(m v)", category=W)
    api.function("wien_peak", wien_peak, signature="wien_peak(T)", doc="peak wavelength b/T", category=W, example="wien_peak(5778 K) -> nm")
    api.function("blackbody_power", blackbody_power, signature="blackbody_power(A, T, emissivity)", doc="ε σ A T⁴", category=W)
    api.function("standing_wave_frequency", standing_wave_frequency, signature="standing_wave_frequency(n, L, v)", doc="n v/(2L)", category=W)
    api.function("beat_frequency", beat_frequency, signature="beat_frequency(f1, f2)", doc="|f1 - f2|", category=W)
    api.function("sound_level_db", sound_level_db, signature="sound_level_db(I)", doc="10 log10(I/I₀)", category=W)
    N = "Physics: nuclear"
    api.function("decay", decay, signature="decay(N0, half_life, t)", doc="N0 2^(-t/T½)", category=N, example="decay(1000, 5730 year, 10000 year)")
    api.function("decay_constant", decay_constant, signature="decay_constant(half_life)", doc="ln 2/T½", category=N)
    api.function("activity", activity, signature="activity(N, half_life)", doc="λ N", category=N)
    api.function("age_from_fraction", age_from_fraction, signature="age_from_fraction(fraction, half_life)", doc="radiometric age", category=N,
                 example="age_from_fraction(0.25, 5730 year) -> year")
    api.function("mass_energy", mass_energy, signature="mass_energy(m)", doc="m c²", category=N)
    api.function("binding_energy", binding_energy, signature="binding_energy(Z, N, atomic_mass)", doc="from the mass defect", category=N,
                 example="binding_energy(2, 2, 4.002602 amu) -> MeV")
