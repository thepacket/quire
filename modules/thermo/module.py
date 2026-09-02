"""Thermodynamics, heat transfer and fluid mechanics, with real fluid properties from CoolProp.

Fluids are named by tokens: water, air, R134a, R410A, ammonia, CO2, nitrogen, oxygen,
hydrogen, methane, ethanol, propane, helium, argon. Property functions take T and P
with units and return SI quantities.
"""
import math

import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "thermo"
DESCRIPTION = "Ideal gas, cycles, heat transfer, pipe flow, and real fluid properties (CoolProp)."

try:
    from CoolProp.CoolProp import HAPropsSI, PropsSI
except ImportError:  # pragma: no cover
    PropsSI = HAPropsSI = None

FLUIDS = ["water", "air", "R134a", "R410A", "ammonia", "CO2", "nitrogen", "oxygen", "hydrogen", "methane",
          "ethanol", "propane", "helium", "argon"]
_COOL = {"water": "Water", "air": "Air", "R134a": "R134a", "R410A": "R410A", "ammonia": "Ammonia", "CO2": "CO2",
         "nitrogen": "Nitrogen", "oxygen": "Oxygen", "hydrogen": "Hydrogen", "methane": "Methane",
         "ethanol": "Ethanol", "propane": "Propane", "helium": "Helium", "argon": "Argon"}
SIGMA = u.stefan_boltzmann_constant


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _fluid(f):
    name = str(f)
    if name not in _COOL:
        raise EvalError(f"Unknown fluid '{name}'. Use one of: {', '.join(FLUIDS)}.")
    if PropsSI is None:
        raise EvalError("Fluid properties need the CoolProp package (pip install CoolProp).")
    return _COOL[name]


def _props(key, fluid, T, P):
    val = PropsSI(key, "T", U.si_value(T, u.kelvin, "T"), "P", U.si_value(P, u.pascal, "P"), _fluid(fluid))
    _note(f"{str(fluid)} property from CoolProp at the given T and P")
    return val


def fluid_density(fluid, T, P):
    return sp.Float(_props("D", fluid, T, P)) * u.kg / u.m ** 3


def fluid_enthalpy(fluid, T, P):
    return sp.Float(_props("H", fluid, T, P)) * u.J / u.kg


def fluid_entropy(fluid, T, P):
    return sp.Float(_props("S", fluid, T, P)) * u.J / (u.kg * u.K)


def fluid_cp(fluid, T, P):
    return sp.Float(_props("C", fluid, T, P)) * u.J / (u.kg * u.K)


def fluid_viscosity(fluid, T, P):
    return sp.Float(_props("V", fluid, T, P)) * u.Pa * u.s


def fluid_conductivity(fluid, T, P):
    return sp.Float(_props("L", fluid, T, P)) * u.W / (u.m * u.K)


def saturation_temperature(fluid, P):
    val = PropsSI("T", "P", U.si_value(P, u.pascal, "P"), "Q", 0, _fluid(fluid))
    return sp.Float(val) * u.K


def saturation_pressure(fluid, T):
    val = PropsSI("P", "T", U.si_value(T, u.kelvin, "T"), "Q", 0, _fluid(fluid))
    return sp.Float(val) * u.Pa


def latent_heat(fluid, T):
    t = U.si_value(T, u.kelvin, "T")
    f = _fluid(fluid)
    return sp.Float(PropsSI("H", "T", t, "Q", 1, f) - PropsSI("H", "T", t, "Q", 0, f)) * u.J / u.kg


def humidity_ratio(T, P, RH):
    """kg water per kg dry air from temperature, pressure and relative humidity (0..1)."""
    return sp.Float(HAPropsSI("W", "T", U.si_value(T, u.kelvin, "T"), "P", U.si_value(P, u.pascal, "P"), "R", float(RH)))


def wet_bulb(T, P, RH):
    return sp.Float(HAPropsSI("B", "T", U.si_value(T, u.kelvin, "T"), "P", U.si_value(P, u.pascal, "P"), "R", float(RH))) * u.K


def dew_point(T, P, RH):
    return sp.Float(HAPropsSI("D", "T", U.si_value(T, u.kelvin, "T"), "P", U.si_value(P, u.pascal, "P"), "R", float(RH))) * u.K


def to_celsius(T_kelvin):
    """Temperature in kelvin -> the Celsius number (Celsius is an offset scale, not a unit)."""
    return sp.Float(U.si_value(T_kelvin, u.kelvin, "T") - 273.15)


def from_celsius(T_celsius):
    """Celsius number -> kelvin with units."""
    return (sp.sympify(T_celsius) + sp.Float(273.15)) * u.K


# ---- ideal gas and cycles
def ideal_gas_pressure(n, T, V):
    return sp.simplify(n * u.R * T / V)


def ideal_gas_density(P, T, M):
    """rho = P M/(R T) with M the molar mass."""
    return sp.simplify(P * M / (u.R * T))


def isentropic_T2(T1, P1, P2, gamma=sp.Rational(7, 5)):
    return sp.simplify(T1 * (P2 / P1) ** ((gamma - 1) / gamma))


def carnot_efficiency(T_hot, T_cold):
    return sp.simplify(1 - T_cold / T_hot)


def otto_efficiency(r, gamma=sp.Rational(7, 5)):
    return sp.simplify(1 - r ** (1 - gamma))


def brayton_efficiency(r_p, gamma=sp.Rational(7, 5)):
    return sp.simplify(1 - r_p ** ((1 - gamma) / gamma))


def cop_refrigerator(T_hot, T_cold):
    return sp.simplify(T_cold / (T_hot - T_cold))


def heat(m, c, dT):
    return sp.simplify(m * c * dT)


def entropy_change_ideal(n, cp, T1, T2, P1, P2):
    return sp.simplify(n * (cp * sp.log(T2 / T1) - u.R * sp.log(P2 / P1)))


# ---- heat transfer
def conduction(k, A, dT, L):
    return sp.simplify(k * A * dT / L)


def convection(h, A, dT):
    return sp.simplify(h * A * dT)


def radiation(emissivity, A, T_surface, T_surroundings):
    return sp.simplify(emissivity * SIGMA * A * (T_surface ** 4 - T_surroundings ** 4))


def thermal_resistance_wall(L, k, A):
    return sp.simplify(L / (k * A))


def thermal_resistance_conv(h, A):
    return sp.simplify(1 / (h * A))


def lmtd(dT1, dT2):
    return sp.simplify((dT1 - dT2) / sp.log(dT1 / dT2))


def effectiveness_counterflow(NTU, Cr):
    NTU, Cr = sp.sympify(NTU), sp.sympify(Cr)
    if Cr == 1:
        return NTU / (1 + NTU)
    return sp.simplify((1 - sp.exp(-NTU * (1 - Cr))) / (1 - Cr * sp.exp(-NTU * (1 - Cr))))


def dittus_boelter(Re, Pr, heating=True):
    """Nu = 0.023 Re^0.8 Pr^n, n = 0.4 heating / 0.3 cooling (turbulent pipe flow)."""
    n = sp.Rational(2, 5) if heating else sp.Rational(3, 10)
    return 0.023 * sp.sympify(Re) ** sp.Rational(4, 5) * sp.sympify(Pr) ** n


def nusselt_flat_plate(Re, Pr):
    """Laminar flat plate average Nu = 0.664 Re^(1/2) Pr^(1/3)."""
    return 0.664 * sp.sqrt(sp.sympify(Re)) * sp.sympify(Pr) ** sp.Rational(1, 3)


# ---- fluids
def reynolds(rho, v, D, mu):
    return sp.simplify(rho * v * D / mu)


def friction_factor(Re, roughness_ratio=0):
    """Darcy friction factor: 64/Re laminar, Haaland explicit correlation turbulent."""
    Re = sp.sympify(Re)
    if Re.is_number and Re < 2300:
        _note("laminar flow: f = 64/Re")
        return sp.simplify(64 / Re)
    _note("turbulent flow: Haaland explicit approximation of Colebrook")
    return 1 / (-1.8 * sp.log((sp.sympify(roughness_ratio) / 3.7) ** 1.11 + 6.9 / Re, 10)) ** 2


def head_loss(f, L, D, v):
    """Darcy-Weisbach head loss f (L/D) v^2/(2 g)."""
    return sp.simplify(f * L / D * v ** 2 / (2 * u.gee))


def pressure_drop(f, L, D, rho, v):
    return sp.simplify(f * L / D * rho * v ** 2 / 2)


def flow_velocity(Q, D):
    return sp.simplify(4 * Q / (sp.pi * D ** 2))


def bernoulli_p2(P1, v1, z1, v2, z2, rho):
    return sp.simplify(P1 + rho * (v1 ** 2 - v2 ** 2) / 2 + rho * u.gee * (z1 - z2))


def pump_power(rho, Q, head, efficiency=1):
    return sp.simplify(rho * u.gee * Q * head / efficiency)


def drag_force(Cd, rho, v, A):
    return sp.simplify(Cd * rho * v ** 2 * A / 2)


def terminal_velocity(m, Cd, rho, A):
    return sp.simplify(sp.sqrt(2 * m * u.gee / (Cd * rho * A)))


def hydrostatic_pressure(rho, h):
    return sp.simplify(rho * u.gee * h)


def mach(v, T, gamma=sp.Rational(7, 5), M=sp.Float(0.028965) * u.kg / u.mol):
    return sp.simplify(v / sp.sqrt(gamma * u.R * T / M))


def register(api):
    for name in FLUIDS:
        api.constant(name, sp.Symbol(name), doc=f"fluid token for property functions", category="Thermo: fluids")
    P = "Thermo: fluid properties"
    api.function("fluid_density", fluid_density, signature="fluid_density(fluid, T, P)", doc="density (CoolProp)", category=P,
                 example="fluid_density(water, 300 K, 1 atm)")
    api.function("fluid_enthalpy", fluid_enthalpy, signature="fluid_enthalpy(fluid, T, P)", doc="specific enthalpy", category=P)
    api.function("fluid_entropy", fluid_entropy, signature="fluid_entropy(fluid, T, P)", doc="specific entropy", category=P)
    api.function("fluid_cp", fluid_cp, signature="fluid_cp(fluid, T, P)", doc="specific heat at constant pressure", category=P)
    api.function("fluid_viscosity", fluid_viscosity, signature="fluid_viscosity(fluid, T, P)", doc="dynamic viscosity", category=P)
    api.function("fluid_conductivity", fluid_conductivity, signature="fluid_conductivity(fluid, T, P)", doc="thermal conductivity", category=P)
    api.function("saturation_temperature", saturation_temperature, signature="saturation_temperature(fluid, P)", doc="boiling point at P", category=P,
                 example="saturation_temperature(water, 1 atm)")
    api.function("saturation_pressure", saturation_pressure, signature="saturation_pressure(fluid, T)", doc="vapour pressure at T", category=P)
    api.function("latent_heat", latent_heat, signature="latent_heat(fluid, T)", doc="enthalpy of vaporisation", category=P)
    api.function("humidity_ratio", humidity_ratio, signature="humidity_ratio(T, P, RH)", doc="moist air: kg water / kg dry air", category=P)
    api.function("wet_bulb", wet_bulb, signature="wet_bulb(T, P, RH)", doc="wet-bulb temperature", category=P,
                 example="wet_bulb(from_celsius(25), 1 atm, 0.5)")
    api.function("dew_point", dew_point, signature="dew_point(T, P, RH)", doc="dew point", category=P)
    api.function("to_celsius", to_celsius, signature="to_celsius(T)", doc="kelvin -> Celsius number", category=P, example="to_celsius(300 K)")
    api.function("from_celsius", from_celsius, signature="from_celsius(T_C)", doc="Celsius number -> kelvin", category=P, example="from_celsius(25)")
    G = "Thermo: gases & cycles"
    api.function("ideal_gas_pressure", ideal_gas_pressure, signature="ideal_gas_pressure(n, T, V)", doc="n R T / V", category=G,
                 example="ideal_gas_pressure(1 mol, 300 K, 22.4 L) -> kPa")
    api.function("ideal_gas_density", ideal_gas_density, signature="ideal_gas_density(P, T, M)", doc="P M/(R T)", category=G)
    api.function("isentropic_T2", isentropic_T2, signature="isentropic_T2(T1, P1, P2, gamma)", doc="temperature after isentropic compression", category=G)
    api.function("carnot_efficiency", carnot_efficiency, signature="carnot_efficiency(T_hot, T_cold)", doc="1 - Tc/Th", category=G,
                 example="carnot_efficiency(800 K, 300 K)")
    api.function("otto_efficiency", otto_efficiency, signature="otto_efficiency(r, gamma)", doc="1 - r^(1-γ)", category=G)
    api.function("brayton_efficiency", brayton_efficiency, signature="brayton_efficiency(r_p, gamma)", doc="Brayton cycle efficiency", category=G)
    api.function("cop_refrigerator", cop_refrigerator, signature="cop_refrigerator(T_hot, T_cold)", doc="Carnot COP", category=G)
    api.function("heat", heat, signature="heat(m, c, dT)", doc="m c ΔT", category=G)
    api.function("entropy_change_ideal", entropy_change_ideal, signature="entropy_change_ideal(n, cp, T1, T2, P1, P2)", doc="ideal gas ΔS", category=G)
    H = "Thermo: heat transfer"
    api.function("conduction", conduction, signature="conduction(k, A, dT, L)", doc="k A ΔT / L", category=H,
                 example="conduction(0.8 W/(m K), 10 m^2, 20 K, 0.2 m)")
    api.function("convection", convection, signature="convection(h, A, dT)", doc="h A ΔT", category=H)
    api.function("radiation", radiation, signature="radiation(emissivity, A, T_s, T_inf)", doc="ε σ A (Ts⁴ - T∞⁴)", category=H)
    api.function("thermal_resistance_wall", thermal_resistance_wall, signature="thermal_resistance_wall(L, k, A)", doc="L/(k A)", category=H)
    api.function("thermal_resistance_conv", thermal_resistance_conv, signature="thermal_resistance_conv(h, A)", doc="1/(h A)", category=H)
    api.function("lmtd", lmtd, signature="lmtd(dT1, dT2)", doc="log-mean temperature difference", category=H)
    api.function("effectiveness_counterflow", effectiveness_counterflow, signature="effectiveness_counterflow(NTU, Cr)", doc="ε-NTU, counterflow", category=H)
    api.function("dittus_boelter", dittus_boelter, signature="dittus_boelter(Re, Pr)", doc="turbulent pipe Nusselt number", category=H)
    api.function("nusselt_flat_plate", nusselt_flat_plate, signature="nusselt_flat_plate(Re, Pr)", doc="laminar flat plate Nu", category=H)
    Fl = "Thermo: fluid flow"
    api.function("reynolds", reynolds, signature="reynolds(rho, v, D, mu)", doc="ρ v D / μ", category=Fl,
                 example="reynolds(1000 kg/m^3, 2 m/s, 50 mm, 0.001 Pa s)")
    api.function("friction_factor", friction_factor, signature="friction_factor(Re, roughness_ratio)", doc="Darcy friction factor", category=Fl)
    api.function("head_loss", head_loss, signature="head_loss(f, L, D, v)", doc="Darcy-Weisbach head loss", category=Fl)
    api.function("pressure_drop", pressure_drop, signature="pressure_drop(f, L, D, rho, v)", doc="Darcy-Weisbach pressure drop", category=Fl)
    api.function("flow_velocity", flow_velocity, signature="flow_velocity(Q, D)", doc="4 Q/(π D²)", category=Fl)
    api.function("bernoulli_p2", bernoulli_p2, signature="bernoulli_p2(P1, v1, z1, v2, z2, rho)", doc="pressure at point 2", category=Fl)
    api.function("pump_power", pump_power, signature="pump_power(rho, Q, head, efficiency)", doc="ρ g Q H / η", category=Fl)
    api.function("drag_force", drag_force, signature="drag_force(Cd, rho, v, A)", doc="½ Cd ρ v² A", category=Fl)
    api.function("terminal_velocity", terminal_velocity, signature="terminal_velocity(m, Cd, rho, A)", doc="√(2 m g/(Cd ρ A))", category=Fl)
    api.function("hydrostatic_pressure", hydrostatic_pressure, signature="hydrostatic_pressure(rho, h)", doc="ρ g h", category=Fl,
                 example="hydrostatic_pressure(1000 kg/m^3, 10 m) -> kPa")
    api.function("mach", mach, signature="mach(v, T)", doc="Mach number in air", category=Fl)
