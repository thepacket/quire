"""AC/DC circuits: impedances, phasors, dividers, RC/RL/RLC responses, power, Bode data.

Transfer functions are expressions in a symbol s, compatible with laplace/ilaplace
and the control module. Units of ohm, F, H, Hz, V and A flow through everything.
"""
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "circuits"
DESCRIPTION = "Impedances, phasors, dividers, RC/RLC responses, AC power, Bode data."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _omega(f):
    return 2 * sp.pi * f


def impedance_R(R):
    return sp.sympify(R)


def impedance_C(C, f):
    """1/(j omega C)"""
    return 1 / (sp.I * _omega(f) * C)


def impedance_L(L, f):
    """j omega L"""
    return sp.I * _omega(f) * L


def z_series(*Z):
    return sp.Add(*[sp.sympify(z) for z in Z])


def z_parallel(*Z):
    return 1 / sp.Add(*[1 / sp.sympify(z) for z in Z])


def phasor(magnitude, angle):
    """magnitude at an angle (degrees, or an angle with units) as a complex number."""
    ang = U.strip_angles(sp.sympify(angle))
    if not U.has_units(sp.sympify(angle)):
        ang = sp.sympify(angle) * sp.pi / 180  # plain numbers are degrees
    return sp.simplify(magnitude * sp.exp(sp.I * ang))


def polar_deg(z):
    """[magnitude, angle in degrees] of a complex value."""
    z = sp.sympify(z)
    mag, unit = U.split_units(z)
    return [sp.simplify(sp.Abs(mag)) * unit, sp.simplify(sp.arg(mag) * 180 / sp.pi) * u.degree]


def divider(Z1, Z2):
    """Voltage divider ratio Vout/Vin = Z2/(Z1 + Z2)."""
    return sp.simplify(Z2 / (Z1 + Z2))


def rc_lowpass(R, C, s):
    return 1 / (1 + R * C * s)


def rc_highpass(R, C, s):
    return R * C * s / (1 + R * C * s)


def rl_lowpass(R, L, s):
    return R / (R + L * s)


def rlc_series(R, L, C, s):
    """Transfer function across the capacitor of a series RLC (second-order low-pass)."""
    return 1 / (L * C * s ** 2 + R * C * s + 1)


def cutoff(R, C):
    """RC corner frequency 1/(2 pi R C)."""
    return sp.simplify(1 / (2 * sp.pi * R * C))


def time_constant(R, C):
    return sp.simplify(R * C)


def resonance(L, C):
    """1/(2 pi sqrt(L C))"""
    return sp.simplify(1 / (2 * sp.pi * sp.sqrt(L * C)))


def q_factor(R, L, C):
    """Series RLC quality factor sqrt(L/C)/R."""
    return sp.simplify(sp.sqrt(L / C) / R)


def rc_step(V, R, C, t):
    """Capacitor voltage after a step V at t = 0."""
    return V * (1 - sp.exp(-t / (R * C)))


def rl_step(V, R, L, t):
    """Inductor current after a step V at t = 0."""
    return V / R * (1 - sp.exp(-R * t / L))


def ac_power(V, I, phi=0):
    """[P, Q, S] from RMS voltage, RMS current and the phase angle (degrees or angle units)."""
    ang = sp.sympify(phi)
    ang = U.strip_angles(ang) if U.has_units(ang) else ang * sp.pi / 180
    S = sp.simplify(V * I)
    return [sp.simplify(S * sp.cos(ang)), sp.simplify(S * sp.sin(ang)), S]


def power_factor(P, S):
    return sp.simplify(P / S)


def db(x):
    """20 log10 of a magnitude ratio."""
    return 20 * sp.log(sp.Abs(x), 10)


def db_power(x):
    return 10 * sp.log(x, 10)


def from_db(x):
    return 10 ** (sp.sympify(x) / 20)


def _hj(H, s, f):
    """H(j 2 pi f) as a dimensionless expression (units must cancel: R C in seconds, f in hertz)."""
    Hj = U.to_base(sp.sympify(H).subs(s, sp.I * 2 * sp.pi * f))
    if U.has_units(Hj):
        num, dim = U.strip_units(Hj)
        if dim != 1:
            raise EvalError(f"H(j2πf) should be dimensionless; it has dimension {dim}. Give f in Hz (or plot with a range in Hz).")
        return num
    return Hj


def bode_gain(H, s, f):
    """|H(j 2 pi f)| in dB as an expression in f (plot it with log x)."""
    return 20 * sp.log(sp.Abs(_hj(H, s, f)), 10)


def bode_phase(H, s, f):
    """Phase of H(j 2 pi f) in degrees as an expression in f."""
    return sp.arg(_hj(H, s, f)) * 180 / sp.pi


def wire_resistance(rho, L, A):
    return sp.simplify(rho * L / A)


def register(api):
    Z = "Circuits: impedance"
    api.function("impedance_C", impedance_C, signature="impedance_C(C, f)", doc="capacitor impedance 1/(jωC)", category=Z,
                 example="impedance_C(10 uF, 50 Hz)")
    api.function("impedance_L", impedance_L, signature="impedance_L(L, f)", doc="inductor impedance jωL", category=Z,
                 example="impedance_L(10 mH, 1 kHz)")
    api.function("z_series", z_series, signature="z_series(Z1, Z2, ...)", doc="impedances in series", category=Z)
    api.function("z_parallel", z_parallel, signature="z_parallel(Z1, Z2, ...)", doc="impedances in parallel", category=Z)
    api.function("parallel", z_parallel, signature="parallel(R1, R2, ...)", doc="resistors (or impedances) in parallel",
                 category=Z, example="parallel(1 kohm, 2 kohm)")
    api.function("phasor", phasor, signature="phasor(magnitude, angle)", doc="complex phasor; angle in degrees",
                 category=Z, example="phasor(230 V, 30)")
    api.function("polar_deg", polar_deg, signature="polar_deg(z)", doc="[magnitude, angle in degrees]", category=Z,
                 example="polar_deg(phasor(230 V, 30))")
    api.function("divider", divider, signature="divider(Z1, Z2)", doc="Vout/Vin = Z2/(Z1 + Z2)", category=Z)
    api.function("wire_resistance", wire_resistance, signature="wire_resistance(rho, L, A)", doc="ρ L / A", category=Z)
    F = "Circuits: filters & responses"
    api.function("rc_lowpass", rc_lowpass, signature="rc_lowpass(R, C, s)", doc="H(s) = 1/(1 + RCs)", category=F,
                 example="rc_lowpass(R, C, s)")
    api.function("rc_highpass", rc_highpass, signature="rc_highpass(R, C, s)", doc="H(s) = RCs/(1 + RCs)", category=F)
    api.function("rl_lowpass", rl_lowpass, signature="rl_lowpass(R, L, s)", doc="H(s) = R/(R + Ls)", category=F)
    api.function("rlc_series", rlc_series, signature="rlc_series(R, L, C, s)", doc="second-order low-pass across C",
                 category=F)
    api.function("cutoff", cutoff, signature="cutoff(R, C)", doc="corner frequency 1/(2πRC)", category=F,
                 example="cutoff(4.7 kohm, 100 nF) -> Hz")
    api.function("time_constant", time_constant, signature="time_constant(R, C)", doc="τ = RC", category=F)
    api.function("resonance", resonance, signature="resonance(L, C)", doc="1/(2π√(LC))", category=F,
                 example="resonance(10 mH, 100 nF) -> kHz")
    api.function("q_factor", q_factor, signature="q_factor(R, L, C)", doc="series RLC quality factor", category=F)
    api.function("rc_step", rc_step, signature="rc_step(V, R, C, t)", doc="capacitor voltage after a step",
                 category=F, example="rc_step(5 V, 1 kohm, 1 uF, t)")
    api.function("rl_step", rl_step, signature="rl_step(V, R, L, t)", doc="inductor current after a step", category=F)
    api.function("bode_gain", bode_gain, signature="bode_gain(H, s, f)", doc="|H(j2πf)| in dB, an expression in f",
                 category=F, example="bode_gain(rc_lowpass(1 kohm, 1 uF, s), s, f)")
    api.function("bode_phase", bode_phase, signature="bode_phase(H, s, f)", doc="phase in degrees, expression in f",
                 category=F)
    P = "Circuits: power"
    api.function("ac_power", ac_power, signature="ac_power(V, I, phi)", doc="[P, Q, S] for RMS V, I and phase angle",
                 category=P, example="ac_power(230 V, 5 A, 30)")
    api.function("power_factor", power_factor, signature="power_factor(P, S)", doc="P / S", category=P)
    api.function("db", db, signature="db(x)", doc="20 log10 |x|", category=P, example="db(1/sqrt(2))")
    api.function("db_power", db_power, signature="db_power(x)", doc="10 log10 x", category=P)
    api.function("from_db", from_db, signature="from_db(x)", doc="10^(x/20)", category=P)
