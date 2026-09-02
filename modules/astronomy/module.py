"""Astronomy: celestial constants, orbital mechanics, magnitudes, distances, time and coordinates.

Angles are degrees unless given with units. Right ascension may be given in hours with hour units.
"""
import math

import sympy as sp
from sympy.physics import units as u
from sympy.physics.units import Quantity

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "astronomy"
DESCRIPTION = "Orbits, magnitudes, distances, sidereal time, coordinate transforms, celestial constants."

G = u.gravitational_constant
c = u.speed_of_light


def _q(name, abbrev, factor, base):
    q = Quantity(name, abbrev=abbrev)
    q.set_global_relative_scale_factor(factor, base)
    return q


AU = _q("astronomical_unit", "AU", 149597870700, u.meter)
ly = _q("light_year", "ly", 9460730472580800, u.meter)
pc = _q("parsec", "pc", sp.Float(3.0856775814913673e16), u.meter)
M_sun = _q("solar_mass", "M_sun", sp.Float(1.98892e30), u.kg)
R_sun = _q("solar_radius", "R_sun", sp.Float(6.957e8), u.meter)
L_sun = _q("solar_luminosity", "L_sun", sp.Float(3.828e26), u.watt)
M_earth = _q("earth_mass", "M_earth", sp.Float(5.9722e24), u.kg)
R_earth = _q("earth_radius", "R_earth", sp.Float(6.371e6), u.meter)
M_moon = _q("moon_mass", "M_moon", sp.Float(7.342e22), u.kg)
R_moon = _q("moon_radius", "R_moon", sp.Float(1.7374e6), u.meter)
M_jupiter = _q("jupiter_mass", "M_jup", sp.Float(1.8982e27), u.kg)
arcsec = _q("arcsecond", "arcsec", sp.pi / 648000, u.radian)
arcmin = _q("arcminute", "arcmin", sp.pi / 10800, u.radian)
hour_angle_unit = u.hour

CONSTS = {"AU": (AU, "astronomical unit"), "ly": (ly, "light year"), "pc": (pc, "parsec"),
          "M_sun": (M_sun, "solar mass"), "R_sun": (R_sun, "solar radius"), "L_sun": (L_sun, "solar luminosity"),
          "M_earth": (M_earth, "Earth mass"), "R_earth": (R_earth, "Earth mean radius"),
          "M_moon": (M_moon, "Moon mass"), "R_moon": (R_moon, "Moon radius"), "M_jupiter": (M_jupiter, "Jupiter mass"),
          "arcsec": (arcsec, "arcsecond"), "arcmin": (arcmin, "arcminute")}


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


# ---- orbits
def orbital_velocity(M, r):
    return sp.simplify(sp.sqrt(G * M / r))


def orbital_period(M, a):
    return sp.simplify(2 * sp.pi * sp.sqrt(a ** 3 / (G * M)))


def semi_major_axis(M, T):
    return sp.simplify((G * M * T ** 2 / (4 * sp.pi ** 2)) ** sp.Rational(1, 3))


def escape_velocity(M, r):
    return sp.simplify(sp.sqrt(2 * G * M / r))


def vis_viva(M, r, a):
    return sp.simplify(sp.sqrt(G * M * (2 / r - 1 / a)))


def hohmann(M, r1, r2):
    """[dv1, dv2, total dv, transfer time] for a Hohmann transfer between circular orbits."""
    a_t = (r1 + r2) / 2
    dv1 = sp.sqrt(G * M / r1) * (sp.sqrt(2 * r2 / (r1 + r2)) - 1)
    dv2 = sp.sqrt(G * M / r2) * (1 - sp.sqrt(2 * r1 / (r1 + r2)))
    t = sp.pi * sp.sqrt(a_t ** 3 / (G * M))
    _note("Hohmann transfer: two tangential burns, half an ellipse between the orbits")
    return [sp.simplify(dv1), sp.simplify(dv2), sp.simplify(dv1 + dv2), sp.simplify(t)]


def synodic_period(T1, T2):
    return sp.simplify(1 / sp.Abs(1 / T1 - 1 / T2))


def hill_sphere(m, M, a):
    return sp.simplify(a * (m / (3 * M)) ** sp.Rational(1, 3))


def schwarzschild_radius(M):
    return sp.simplify(2 * G * M / c ** 2)


def surface_gravity(M, R):
    return sp.simplify(G * M / R ** 2)


# ---- brightness and distance
def distance_modulus(m, M):
    """Distance from apparent and absolute magnitude: 10^((m - M + 5)/5) pc."""
    return 10 ** ((sp.sympify(m) - sp.sympify(M) + 5) / 5) * pc


def absolute_magnitude(m, d):
    return sp.sympify(m) - 5 * sp.log(sp.simplify(d / pc), 10) + 5


def apparent_magnitude(M, d):
    return sp.sympify(M) + 5 * sp.log(sp.simplify(d / pc), 10) - 5


def flux_ratio(m1, m2):
    """F1/F2 = 10^(-0.4 (m1 - m2))"""
    return 10 ** (-sp.Rational(2, 5) * (sp.sympify(m1) - sp.sympify(m2)))


def parallax_distance(p):
    """Distance in parsecs from a parallax angle (arcseconds if a plain number)."""
    p = sp.sympify(p)
    if U.has_units(p):
        p = U.convert(p, arcsec) / arcsec
    return sp.simplify(1 / p) * pc


def luminosity(R, T):
    return sp.simplify(4 * sp.pi * R ** 2 * u.stefan_boltzmann_constant * T ** 4)


def luminosity_from_magnitude(M_abs):
    """Luminosity relative to the Sun from absolute magnitude (M_sun = 4.83)."""
    return 10 ** (-sp.Rational(2, 5) * (sp.sympify(M_abs) - sp.Float(4.83))) * L_sun


def redshift_velocity(z):
    """Recession velocity for a redshift z (relativistic formula)."""
    z = sp.sympify(z)
    return sp.simplify(c * ((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1))


def hubble_distance(v, H0=sp.Float(70) * u.km / u.s / (10 ** 6 * pc)):
    return sp.simplify(v / H0)


# ---- time and coordinates
def julian_date(year, month, day, hour=0):
    """Julian Date of a Gregorian calendar date (UT)."""
    y, m, d = int(year), int(month), float(day) + float(hour) / 24
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5
    return sp.Float(jd, 12)


def sidereal_time(jd, longitude=0):
    """Local mean sidereal time in hours for a Julian Date and east longitude in degrees."""
    T = (float(jd) - 2451545.0) / 36525
    gmst = 280.46061837 + 360.98564736629 * (float(jd) - 2451545.0) + 0.000387933 * T ** 2 - T ** 3 / 38710000
    lon = float(_ang(longitude) * 180 / sp.pi)
    lst = (gmst + lon) % 360
    return sp.Float(lst / 15, 8) * u.hour


def alt_az(dec, lat, hour_angle):
    """[altitude, azimuth] in degrees from declination, observer latitude and hour angle (degrees or hours)."""
    dec, lat = _ang(dec), _ang(lat)
    ha = sp.sympify(hour_angle)
    ha = (U.convert(ha, u.hour) / u.hour * 15 * sp.pi / 180) if U.has_units(ha) and ha.has(u.hour) else _ang(ha)
    sin_alt = sp.sin(dec) * sp.sin(lat) + sp.cos(dec) * sp.cos(lat) * sp.cos(ha)
    alt = sp.asin(sin_alt)
    az = sp.atan2(-sp.sin(ha) * sp.cos(dec), sp.sin(dec) * sp.cos(lat) - sp.cos(dec) * sp.sin(lat) * sp.cos(ha))
    az = sp.Mod(az, 2 * sp.pi)
    return [_deg(alt), _deg(az)]


def angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees between two sky positions (degrees)."""
    a1, d1, a2, d2 = (_ang(v) for v in (ra1, dec1, ra2, dec2))
    cosd = sp.sin(d1) * sp.sin(d2) + sp.cos(d1) * sp.cos(d2) * sp.cos(a1 - a2)
    return _deg(sp.acos(cosd))


def equatorial_to_galactic(ra, dec):
    """[l, b] galactic coordinates in degrees from RA and Dec in degrees (J2000)."""
    a, d = _ang(ra), _ang(dec)
    ra_gp, dec_gp, l_ncp = 192.85948 * sp.pi / 180, 27.12825 * sp.pi / 180, 122.93192 * sp.pi / 180
    b = sp.asin(sp.sin(d) * sp.sin(dec_gp) + sp.cos(d) * sp.cos(dec_gp) * sp.cos(a - ra_gp))
    l = l_ncp - sp.atan2(sp.cos(d) * sp.sin(a - ra_gp), sp.sin(d) * sp.cos(dec_gp) - sp.cos(d) * sp.sin(dec_gp) * sp.cos(a - ra_gp))
    return [_deg(sp.Mod(l, 2 * sp.pi)), _deg(b)]


def register(api):
    for name, (q, doc) in CONSTS.items():
        api.unit(name, q, doc=doc, category="Astronomy: constants & units")
    O = "Astronomy: orbits"
    api.function("orbital_velocity", orbital_velocity, signature="orbital_velocity(M, r)", doc="√(GM/r)", category=O,
                 example="orbital_velocity(M_earth, R_earth + 400 km) -> km/s")
    api.function("orbital_period", orbital_period, signature="orbital_period(M, a)", doc="2π√(a³/GM)", category=O,
                 example="orbital_period(M_sun, 1 AU) -> day")
    api.function("semi_major_axis", semi_major_axis, signature="semi_major_axis(M, T)", doc="from the period (Kepler III)", category=O)
    api.function("escape_velocity", escape_velocity, signature="escape_velocity(M, r)", doc="√(2GM/r)", category=O)
    api.function("vis_viva", vis_viva, signature="vis_viva(M, r, a)", doc="speed on an orbit at radius r", category=O)
    api.function("hohmann", hohmann, signature="hohmann(M, r1, r2)", doc="[Δv1, Δv2, Δv, time] of a Hohmann transfer", category=O,
                 example="hohmann(M_earth, R_earth + 400 km, 42164 km)")
    api.function("synodic_period", synodic_period, signature="synodic_period(T1, T2)", doc="1/|1/T1 - 1/T2|", category=O)
    api.function("hill_sphere", hill_sphere, signature="hill_sphere(m, M, a)", doc="a (m/3M)^(1/3)", category=O)
    api.function("schwarzschild_radius", schwarzschild_radius, signature="schwarzschild_radius(M)", doc="2GM/c²", category=O,
                 example="schwarzschild_radius(M_sun) -> km")
    api.function("surface_gravity", surface_gravity, signature="surface_gravity(M, R)", doc="GM/R²", category=O)
    B = "Astronomy: brightness & distance"
    api.function("distance_modulus", distance_modulus, signature="distance_modulus(m, M)", doc="distance from m - M", category=B,
                 example="distance_modulus(10, 5) -> ly")
    api.function("absolute_magnitude", absolute_magnitude, signature="absolute_magnitude(m, d)", doc="M from m and distance", category=B)
    api.function("apparent_magnitude", apparent_magnitude, signature="apparent_magnitude(M, d)", doc="m from M and distance", category=B)
    api.function("flux_ratio", flux_ratio, signature="flux_ratio(m1, m2)", doc="10^(-0.4 Δm)", category=B)
    api.function("parallax_distance", parallax_distance, signature="parallax_distance(p)", doc="1/p parsecs", category=B,
                 example="parallax_distance(0.768) -> ly")
    api.function("luminosity", luminosity, signature="luminosity(R, T)", doc="4π R² σ T⁴", category=B, example="luminosity(R_sun, 5778 K) -> W")
    api.function("luminosity_from_magnitude", luminosity_from_magnitude, signature="luminosity_from_magnitude(M)", doc="relative to the Sun", category=B)
    api.function("redshift_velocity", redshift_velocity, signature="redshift_velocity(z)", doc="recession velocity", category=B)
    api.function("hubble_distance", hubble_distance, signature="hubble_distance(v, H0)", doc="v/H0", category=B)
    T = "Astronomy: time & coordinates"
    api.function("julian_date", julian_date, signature="julian_date(y, m, d, hour)", doc="Julian Date (UT)", category=T,
                 example="julian_date(2000, 1, 1, 12)")
    api.function("sidereal_time", sidereal_time, signature="sidereal_time(jd, longitude)", doc="local mean sidereal time (hours)", category=T)
    api.function("alt_az", alt_az, signature="alt_az(dec, lat, hour_angle)", doc="[altitude, azimuth] in degrees", category=T,
                 example="alt_az(20, 45, 30)")
    api.function("angular_separation", angular_separation, signature="angular_separation(ra1, dec1, ra2, dec2)", doc="sky separation (degrees)", category=T)
    api.function("equatorial_to_galactic", equatorial_to_galactic, signature="equatorial_to_galactic(ra, dec)", doc="[l, b] galactic coordinates", category=T)
