"""Earth science: open-channel and groundwater hydrology, runoff, the standard atmosphere, humidity, seismology."""
from __future__ import annotations

import math

import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "earth"
DESCRIPTION = "Hydrology (Manning, runoff, Darcy), the standard atmosphere and humidity, seismology basics."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _si(v, unit, what):
    return U.si_value(v, unit, what) if U.has_units(v) else float(v)


# ---- hydrology
def manning_velocity(n, R, S):
    """Manning: v = R^(2/3) S^(1/2) / n (SI)."""
    Rv = _si(R, u.meter, "hydraulic radius")
    return sp.Float(Rv ** (2 / 3) * math.sqrt(float(S)) / float(n), 6) * u.meter / u.second


def manning_flow(n, A, R, S):
    """Q = A R^(2/3) S^(1/2) / n."""
    Av = _si(A, u.meter ** 2, "area")
    return sp.Float(Av * float(U.strip_units(manning_velocity(n, R, S))[0]), 6) * u.meter ** 3 / u.second


def hydraulic_radius(A, P):
    return sp.simplify(A / P)


def froude_number(v, depth):
    return sp.simplify(v / sp.sqrt(sp.Float(9.80665) * u.meter / u.second ** 2 * depth))


def critical_depth(q):
    """Critical depth of a rectangular channel from the unit discharge q = Q / b."""
    qv = _si(q, u.meter ** 2 / u.second, "unit discharge")
    return sp.Float((qv ** 2 / 9.80665) ** (1 / 3), 6) * u.meter


def rational_runoff(C, i, A):
    """Rational method: Q = C i A (i in mm/hr, A in ha gives m^3/s after conversion)."""
    return U.convert(sp.sympify(C) * i * A, u.meter ** 3 / u.second) if U.has_units(i) or U.has_units(A) else sp.simplify(C * i * A)


def scs_runoff(P, CN):
    """SCS curve-number runoff depth (mm) from rainfall P (mm)."""
    Pv = _si(P, u.meter, "rainfall") * 1000 if U.has_units(P) else float(P)
    S = 25400 / float(CN) - 254
    Ia = 0.2 * S
    Q = (Pv - Ia) ** 2 / (Pv - Ia + S) if Pv > Ia else 0.0
    _note(f"potential retention S = {S:.1f} mm, initial abstraction {Ia:.1f} mm")
    return sp.Float(Q, 6) * u.millimeter


def time_of_concentration(L, S):
    """Kirpich: tc = 0.0195 L^0.77 S^-0.385 minutes (L in m, S slope)."""
    Lv = _si(L, u.meter, "flow length")
    return sp.Float(0.0195 * Lv ** 0.77 * float(S) ** -0.385, 5) * u.minute


def darcy_flow(K, dh, dl, A):
    """Groundwater flow Q = K A dh/dl."""
    return sp.simplify(K * A * dh / dl)


def hydraulic_conductivity(Q, A, dh, dl):
    return sp.simplify(Q * dl / (A * dh))


def evaporation_hargreaves(T_mean, T_max, T_min, Ra):
    """Hargreaves reference evapotranspiration, mm/day, Ra in MJ/m^2/day."""
    tm, tx, tn = (float(v) for v in (T_mean, T_max, T_min))
    return sp.Float(0.0023 * (tm + 17.8) * math.sqrt(max(tx - tn, 0)) * 0.408 * float(Ra), 5) * u.millimeter / u.day


# ---- atmosphere
def _isa(h_m: float):
    """(T [K], p [Pa], rho [kg/m^3]) of the International Standard Atmosphere up to 32 km."""
    g, R = 9.80665, 287.05287
    if h_m < 11000:
        T = 288.15 - 0.0065 * h_m
        p = 101325 * (T / 288.15) ** (g / (0.0065 * R))
    elif h_m < 20000:
        T = 216.65
        p = 22632.06 * math.exp(-g * (h_m - 11000) / (R * T))
    elif h_m <= 32000:
        T = 216.65 + 0.001 * (h_m - 20000)
        p = 5474.889 * (T / 216.65) ** (-g / (0.001 * R))
    else:
        raise EvalError("The standard atmosphere here goes up to 32 km.")
    return T, p, p / (R * T)


def isa_temperature(h):
    T, _, _ = _isa(_si(h, u.meter, "altitude"))
    return sp.Float(T, 6) * u.kelvin


def isa_pressure(h):
    _, p, _ = _isa(_si(h, u.meter, "altitude"))
    return sp.Float(p, 6) * u.pascal


def isa_density(h):
    _, _, rho = _isa(_si(h, u.meter, "altitude"))
    return sp.Float(rho, 6) * u.kilogram / u.meter ** 3


def pressure_altitude(p):
    """Altitude in the standard atmosphere where the pressure is p (troposphere)."""
    pv = _si(p, u.pascal, "pressure")
    return sp.Float(44330.77 * (1 - (pv / 101325) ** 0.190263), 6) * u.meter


def barometric(p0, h, T):
    """Isothermal barometric formula p0 exp(-M g h / (R T))."""
    return sp.simplify(p0 * sp.exp(-sp.Float(0.0289644) * sp.Float(9.80665) * h / (sp.Float(8.314462618) * T)))


def relative_humidity(T, T_dew):
    a, b = 17.625, 243.04
    t, td = float(T), float(T_dew)
    return sp.Float(100 * math.exp(a * td / (b + td) - a * t / (b + t)), 5)


def saturation_vapour_pressure(T):
    """Magnus, hPa, T in °C."""
    t = float(T)
    return sp.Float(6.1094 * math.exp(17.625 * t / (t + 243.04)), 5) * u.hectopascal if hasattr(u, "hectopascal") else sp.Float(610.94 * math.exp(17.625 * t / (t + 243.04)), 5) * u.pascal


def heat_index(T, RH):
    """NOAA heat index, °C in and out."""
    tf = float(T) * 9 / 5 + 32
    rh = float(RH)
    hi = (-42.379 + 2.04901523 * tf + 10.14333127 * rh - 0.22475541 * tf * rh - 6.83783e-3 * tf ** 2 - 5.481717e-2 * rh ** 2
          + 1.22874e-3 * tf ** 2 * rh + 8.5282e-4 * tf * rh ** 2 - 1.99e-6 * tf ** 2 * rh ** 2)
    if tf < 80:
        hi = 0.5 * (tf + 61.0 + (tf - 68.0) * 1.2 + rh * 0.094)
    return sp.Float((hi - 32) * 5 / 9, 4)


def wind_chill(T, v):
    """Wind chill (°C, v in km/h)."""
    t = float(T)
    vk = _si(v, u.meter / u.second, "wind speed") * 3.6 if U.has_units(v) else float(v)
    return sp.Float(13.12 + 0.6215 * t - 11.37 * vk ** 0.16 + 0.3965 * t * vk ** 0.16, 4)


def lapse_rate(T1, T2, h1, h2):
    return sp.simplify((T1 - T2) / (h2 - h1))


# ---- seismology
def moment_magnitude(M0):
    """Mw = (2/3) (log10 M0[N m] - 9.1)."""
    m0 = _si(M0, u.newton * u.meter, "seismic moment")
    return sp.Float(2 / 3 * (math.log10(m0) - 9.1), 4)


def seismic_moment(Mw):
    return sp.Float(10 ** (1.5 * float(Mw) + 9.1), 5) * u.newton * u.meter


def seismic_energy(M):
    """Radiated energy log10 E[J] = 1.5 M + 4.8 (Gutenberg-Richter)."""
    return sp.Float(10 ** (1.5 * float(M) + 4.8), 5) * u.joule


def magnitude_energy_ratio(M1, M2):
    """How many times more energy magnitude M2 releases than M1."""
    return sp.Float(10 ** (1.5 * (float(M2) - float(M1))), 5)


def local_magnitude(amplitude, distance):
    """Richter local magnitude from a Wood-Anderson amplitude (mm) and epicentral distance (km): log A + 3 log(8 Δt) - 2.92 approximation via Δ."""
    a = _si(amplitude, u.millimeter, "amplitude") * 1000 if U.has_units(amplitude) else float(amplitude)
    d = _si(distance, u.kilometer, "distance") / 1000 if U.has_units(distance) else float(distance)
    return sp.Float(math.log10(a) + 1.11 * math.log10(d) + 0.00189 * d - 2.09, 3)


def epicentral_distance(dt, vp=sp.Float(6.0) * u.kilometer / u.second, vs=sp.Float(3.5) * u.kilometer / u.second):
    """Distance from the S - P arrival gap: d = dt / (1/vs - 1/vp)."""
    return sp.simplify(dt / (1 / vs - 1 / vp))


def gutenberg_richter(a, b, M):
    """Expected number of earthquakes of magnitude >= M per year: 10^(a - b M)."""
    return sp.simplify(sp.Integer(10) ** (a - b * M))


def return_period(a, b, M):
    return sp.simplify(1 / gutenberg_richter(a, b, M))


def register(api):
    H = "Hydrology"
    api.function("manning_velocity", manning_velocity, signature="manning_velocity(n, R, S)", doc="open-channel velocity", category=H, example="manning_velocity(0.013, 0.5 m, 0.002)")
    api.function("manning_flow", manning_flow, signature="manning_flow(n, A, R, S)", doc="open-channel discharge", category=H)
    api.function("hydraulic_radius", hydraulic_radius, signature="hydraulic_radius(A, P)", doc="A / wetted perimeter", category=H)
    api.function("froude_number", froude_number, signature="froude_number(v, depth)", doc="v / sqrt(g y)", category=H)
    api.function("critical_depth", critical_depth, signature="critical_depth(q)", doc="rectangular channel, q = Q/b", category=H)
    api.function("rational_runoff", rational_runoff, signature="rational_runoff(C, i, A)", doc="Q = C i A", category=H, example="rational_runoff(0.6, 50 mm/hr, 20000 m^2)")
    api.function("scs_runoff", scs_runoff, signature="scs_runoff(P, CN)", doc="SCS curve-number runoff depth", category=H, example="scs_runoff(80 mm, 75)")
    api.function("time_of_concentration", time_of_concentration, signature="time_of_concentration(L, S)", doc="Kirpich formula", category=H)
    api.function("darcy_flow", darcy_flow, signature="darcy_flow(K, dh, dl, A)", doc="groundwater flow K A dh/dl", category=H)
    api.function("hydraulic_conductivity", hydraulic_conductivity, signature="hydraulic_conductivity(Q, A, dh, dl)", doc="from a Darcy test", category=H)
    api.function("evaporation_hargreaves", evaporation_hargreaves, signature="evaporation_hargreaves(T_mean, T_max, T_min, Ra)", doc="reference evapotranspiration", category=H)
    A = "Atmosphere"
    api.function("isa_temperature", isa_temperature, signature="isa_temperature(h)", doc="standard atmosphere temperature", category=A, example="isa_temperature(10 km)")
    api.function("isa_pressure", isa_pressure, signature="isa_pressure(h)", doc="standard atmosphere pressure", category=A, example="isa_pressure(5000 m) -> kPa")
    api.function("isa_density", isa_density, signature="isa_density(h)", doc="standard atmosphere density", category=A)
    api.function("pressure_altitude", pressure_altitude, signature="pressure_altitude(p)", doc="altitude of a pressure", category=A)
    api.function("barometric", barometric, signature="barometric(p0, h, T)", doc="isothermal barometric formula", category=A)
    api.function("relative_humidity", relative_humidity, signature="relative_humidity(T, T_dew)", doc="from the dew point", category=A)
    api.function("saturation_vapour_pressure", saturation_vapour_pressure, signature="saturation_vapour_pressure(T)", doc="Magnus (°C)", category=A)
    api.function("heat_index", heat_index, signature="heat_index(T, RH)", doc="NOAA heat index in °C", category=A)
    api.function("wind_chill", wind_chill, signature="wind_chill(T, v)", doc="°C, wind in km/h", category=A, example="wind_chill(-10, 30)")
    api.function("lapse_rate", lapse_rate, signature="lapse_rate(T1, T2, h1, h2)", doc="temperature drop per height", category=A)
    S = "Seismology"
    api.function("moment_magnitude", moment_magnitude, signature="moment_magnitude(M0)", doc="Mw from the seismic moment", category=S, example="moment_magnitude(1e20 N m)")
    api.function("seismic_moment", seismic_moment, signature="seismic_moment(Mw)", doc="M0 from Mw", category=S)
    api.function("seismic_energy", seismic_energy, signature="seismic_energy(M)", doc="radiated energy", category=S, example="seismic_energy(7)")
    api.function("magnitude_energy_ratio", magnitude_energy_ratio, signature="magnitude_energy_ratio(M1, M2)", doc="energy ratio between magnitudes", category=S)
    api.function("local_magnitude", local_magnitude, signature="local_magnitude(A, distance)", doc="ML from amplitude (mm) and distance (km)", category=S)
    api.function("epicentral_distance", epicentral_distance, signature="epicentral_distance(dt, vp, vs)", doc="from the S-P time gap", category=S,
                 example="epicentral_distance(30 s) -> km")
    api.function("gutenberg_richter", gutenberg_richter, signature="gutenberg_richter(a, b, M)", doc="yearly count of magnitude >= M", category=S)
    api.function("return_period", return_period, signature="return_period(a, b, M)", doc="years between magnitude >= M", category=S)
