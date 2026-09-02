"""Unit table and unit helpers built on sympy.physics.units.

Design notes
------------
* Every unit is a sympy Quantity, so units live inside the same algebra as
  everything else. A definition like ``F = m a`` keeps its units through any
  later symbolic manipulation.
* ``strip_units`` is the bridge to numerics (plotting, nsolve): it converts to
  SI base units and replaces each remaining Quantity by 1.
"""
from __future__ import annotations

import re

import sympy as sp
from sympy.physics import units as u
from sympy.physics.units import Quantity, convert_to
from sympy.physics.units.systems import SI

from .errors import UnitError

BASE_UNITS = [u.meter, u.kilogram, u.second, u.ampere, u.kelvin, u.mole, u.candela]
ANGLE_UNITS = {u.degree, u.radian}


def _prefixed(name: str, abbrev: str, factor, base) -> Quantity:
    q = Quantity(name, abbrev=abbrev)
    q.set_global_relative_scale_factor(factor, base)
    return q


# Derived/prefixed units that sympy does not ship with a short name.
kN = _prefixed("kilonewton", "kN", 1000, u.newton)
kJ = _prefixed("kilojoule", "kJ", 1000, u.joule)
MJ = _prefixed("megajoule", "MJ", 10**6, u.joule)
kW = _prefixed("kilowatt", "kW", 1000, u.watt)
MW = _prefixed("megawatt", "MW", 10**6, u.watt)
kPa = _prefixed("kilopascal", "kPa", 1000, u.pascal)
MPa = _prefixed("megapascal", "MPa", 10**6, u.pascal)
GPa = _prefixed("gigapascal", "GPa", 10**9, u.pascal)
kHz = _prefixed("kilohertz", "kHz", 1000, u.hertz)
MHz = _prefixed("megahertz", "MHz", 10**6, u.hertz)
GHz = _prefixed("gigahertz", "GHz", 10**9, u.hertz)
mA = _prefixed("milliampere", "mA", sp.Rational(1, 1000), u.ampere)
kV = _prefixed("kilovolt", "kV", 1000, u.volt)
mV = _prefixed("millivolt", "mV", sp.Rational(1, 1000), u.volt)
kohm = _prefixed("kiloohm", "kΩ", 1000, u.ohm)
Mohm = _prefixed("megaohm", "MΩ", 10**6, u.ohm)
uF = _prefixed("microfarad", "μF", sp.Rational(1, 10**6), u.farad)
nF = _prefixed("nanofarad", "nF", sp.Rational(1, 10**9), u.farad)
pF = _prefixed("picofarad", "pF", sp.Rational(1, 10**12), u.farad)
mH = _prefixed("millihenry", "mH", sp.Rational(1, 1000), u.henry)
uH = _prefixed("microhenry", "μH", sp.Rational(1, 10**6), u.henry)
mL = _prefixed("milliliter", "mL", sp.Rational(1, 1000), u.liter)
kWh = _prefixed("kilowatt_hour", "kWh", 3600 * 1000, u.joule)
tonne = _prefixed("tonne", "t", 1000, u.kilogram)
rpm = _prefixed("rpm", "rpm", 2 * sp.pi / 60, u.radian / u.second)  # angular rate: 1 rpm = 2π/60 rad/s
kmh = _prefixed("kilometer_per_hour", "km/h", sp.Rational(1000, 3600), u.meter / u.second)
mC = _prefixed("millicoulomb", "mC", sp.Rational(1, 1000), u.coulomb)
uC = _prefixed("microcoulomb", "μC", sp.Rational(1, 10**6), u.coulomb)
nC = _prefixed("nanocoulomb", "nC", sp.Rational(1, 10**9), u.coulomb)
pC = _prefixed("picocoulomb", "pC", sp.Rational(1, 10**12), u.coulomb)
mT = _prefixed("millitesla", "mT", sp.Rational(1, 1000), u.tesla)
uT = _prefixed("microtesla", "μT", sp.Rational(1, 10**6), u.tesla)
keV = _prefixed("kiloelectronvolt", "keV", 1000, u.eV)
MeV = _prefixed("megaelectronvolt", "MeV", 10**6, u.eV)
GeV = _prefixed("gigaelectronvolt", "GeV", 10**9, u.eV)

# name -> (quantity, description). Names must be valid identifiers.
UNIT_TABLE: dict[str, tuple[sp.Expr, str]] = {
    # length
    "m": (u.meter, "meter"), "meter": (u.meter, "meter"),
    "km": (u.km, "kilometer"), "cm": (u.cm, "centimeter"), "mm": (u.mm, "millimeter"),
    "um": (u.um, "micrometer"), "nm": (u.nm, "nanometer"),
    "inch": (u.inch, "inch"), "ft": (u.foot, "foot"), "foot": (u.foot, "foot"),
    "yd": (u.yard, "yard"), "mile": (u.mile, "mile"), "nmi": (u.nautical_mile, "nautical mile"),
    # time
    "s": (u.second, "second"), "second": (u.second, "second"), "ms": (u.ms, "millisecond"),
    "us": (u.us, "microsecond"), "ns": (u.ns, "nanosecond"),
    "minute": (u.minute, "minute"), "hr": (u.hour, "hour"), "hour": (u.hour, "hour"),
    "day": (u.day, "day"), "week": (7 * u.day, "week"),
    "year": (u.year, "year (Julian)"),
    # mass
    "kg": (u.kg, "kilogram"), "g": (u.gram, "gram"), "gram": (u.gram, "gram"), "mg": (u.mg, "milligram"),
    "ug": (u.ug, "microgram"), "tonne": (tonne, "metric ton"), "amu": (u.amu, "atomic mass unit"),
    "lb": (u.pound, "pound (mass)"), "pound": (u.pound, "pound (mass)"),
    # force, energy, power, pressure
    "N": (u.newton, "newton"), "newton": (u.newton, "newton"), "kN": (kN, "kilonewton"),
    "J": (u.joule, "joule"), "joule": (u.joule, "joule"), "kJ": (kJ, "kilojoule"), "MJ": (MJ, "megajoule"),
    "eV": (u.eV, "electronvolt"), "keV": (keV, "kiloelectronvolt"), "MeV": (MeV, "megaelectronvolt"),
    "GeV": (GeV, "gigaelectronvolt"), "kWh": (kWh, "kilowatt-hour"),
    "W": (u.watt, "watt"), "watt": (u.watt, "watt"), "kW": (kW, "kilowatt"), "MW": (MW, "megawatt"),
    "Pa": (u.pascal, "pascal"), "pascal": (u.pascal, "pascal"), "kPa": (kPa, "kilopascal"),
    "MPa": (MPa, "megapascal"), "GPa": (GPa, "gigapascal"),
    "bar": (u.bar, "bar"), "atm": (u.atm, "standard atmosphere"), "psi": (u.psi, "pound per square inch"),
    "mmHg": (u.mmHg, "millimeter of mercury"),
    # frequency, angle
    "Hz": (u.hertz, "hertz"), "hertz": (u.hertz, "hertz"), "kHz": (kHz, "kilohertz"),
    "MHz": (MHz, "megahertz"), "GHz": (GHz, "gigahertz"), "rpm": (rpm, "revolutions per minute (angular rate, 2π/60 rad/s)"),
    "rad": (u.radian, "radian"), "radian": (u.radian, "radian"),
    "deg": (u.degree, "degree of arc"), "degree": (u.degree, "degree of arc"),
    # electrical
    "A": (u.ampere, "ampere"), "ampere": (u.ampere, "ampere"), "mA": (mA, "milliampere"),
    "V": (u.volt, "volt"), "volt": (u.volt, "volt"), "kV": (kV, "kilovolt"), "mV": (mV, "millivolt"),
    "ohm": (u.ohm, "ohm"), "kohm": (kohm, "kiloohm"), "Mohm": (Mohm, "megaohm"),
    "C": (u.coulomb, "coulomb"), "coulomb": (u.coulomb, "coulomb"), "mC": (mC, "millicoulomb"),
    "uC": (uC, "microcoulomb"), "nC": (nC, "nanocoulomb"), "pC": (pC, "picocoulomb"),
    "F": (u.farad, "farad"), "farad": (u.farad, "farad"), "uF": (uF, "microfarad"),
    "nF": (nF, "nanofarad"), "pF": (pF, "picofarad"),
    "H": (u.henry, "henry"), "henry": (u.henry, "henry"), "mH": (mH, "millihenry"), "uH": (uH, "microhenry"),
    "T": (u.tesla, "tesla"), "mT": (mT, "millitesla"), "uT": (uT, "microtesla"), "Wb": (u.weber, "weber"),
    "S": (u.siemens, "siemens"),
    # thermodynamics, amount, light
    "K": (u.kelvin, "kelvin"), "kelvin": (u.kelvin, "kelvin"),
    "mol": (u.mole, "mole"), "mole": (u.mole, "mole"), "cd": (u.candela, "candela"),
    "lux": (u.lux, "lux"),
    # volume, speed, misc
    "L": (u.liter, "liter"), "liter": (u.liter, "liter"), "mL": (mL, "milliliter"),
    "kmh": (kmh, "kilometer per hour"), "percent": (u.percent, "percent (1/100)"),
    "Gy": (u.gray, "gray"), "Bq": (u.becquerel, "becquerel"),
}

# physical constants, exposed as unit-carrying quantities
CONSTANT_TABLE: dict[str, tuple[sp.Expr, str]] = {
    "c_light": (u.speed_of_light, "speed of light in vacuum"),
    "G_grav": (u.gravitational_constant, "Newtonian constant of gravitation"),
    "g_0": (u.gee, "standard acceleration of gravity"),
    "h_planck": (u.planck, "Planck constant"),
    "hbar": (u.hbar, "reduced Planck constant"),
    "k_B": (u.boltzmann, "Boltzmann constant"),
    "N_A": (u.avogadro_number, "Avogadro constant"),
    "R_gas": (u.R, "molar gas constant"),
    "e_charge": (u.elementary_charge, "elementary charge"),
    "epsilon_0": (u.vacuum_permittivity, "vacuum permittivity"),
    "mu_0": (u.vacuum_permeability, "vacuum permeability"),
}


def has_units(expr) -> bool:
    return isinstance(expr, sp.Basic) and bool(expr.atoms(Quantity))


def quantities(expr) -> set:
    return expr.atoms(Quantity) if isinstance(expr, sp.Basic) else set()


def strip_angles(expr):
    """Replace angle units by plain numbers (degree -> pi/180, radian -> 1)."""
    if not isinstance(expr, sp.Basic):
        return expr
    return expr.subs({u.degree: sp.pi / 180, u.radian: 1})


def dimension_of(expr):
    """Dimensional expression (e.g. length/time) or 1 if dimensionless."""
    if not has_units(expr):
        return sp.S.One
    return SI.get_dimensional_expr(expr)


# Dimensions a symbol can be declared with ("assume L length"): name -> a representative SI unit.
DIMENSION_UNITS = {
    "length": u.meter, "mass": u.kilogram, "time": u.second, "current": u.ampere, "temperature": u.kelvin,
    "amount": u.mole, "luminosity": u.candela, "area": u.meter ** 2, "volume": u.meter ** 3,
    "velocity": u.meter / u.second, "speed": u.meter / u.second, "acceleration": u.meter / u.second ** 2,
    "force": u.newton, "energy": u.joule, "work": u.joule, "power": u.watt, "pressure": u.pascal,
    "stress": u.pascal, "frequency": u.hertz, "charge": u.coulomb, "voltage": u.volt, "resistance": u.ohm,
    "capacitance": u.farad, "inductance": u.henry, "density": u.kilogram / u.meter ** 3,
    "momentum": u.kilogram * u.meter / u.second, "torque": u.newton * u.meter, "angle": sp.S.One,
    "dimensionless": sp.S.One,
}


def check_dimensions(expr, symbols=()) -> None:
    """Raise UnitError when a sum mixes incompatible dimensions.

    Only fully bound expressions are checked: a free symbol may receive a
    unit-carrying value later (function parameters, plot variables), so
    ``x m + 3 s`` is judged when x is known, not before. Symbols listed in
    ``symbols`` are dimensionless placeholders whose unit travels beside them
    (declared dimensions, measured values), so they do not block the check.
    Every Add inside the expression is checked term by term in SI base units,
    so Hz*s and km/m cancel and functions such as Abs or log around a sum do
    not confuse the check.
    """
    if not has_units(expr) or (expr.free_symbols - set(symbols)):
        return
    base = to_base(expr)
    dimsys = SI.get_dimension_system()
    for add in base.atoms(sp.Add):
        known = []
        for term in add.args:
            try:
                d = u.Dimension(SI.get_dimensional_expr(term))
            except Exception:  # noqa: BLE001 - a function of a dimensional argument; leave it
                continue
            known.append((term, d))
        for term, d in known[1:]:
            if not dimsys.equivalent_dims(d, known[0][1]):
                raise UnitError(f"Cannot add '{term}' ({d.name}) to a quantity with dimension {known[0][1].name}.")


def to_base(expr):
    """Rewrite every unit in SI base units, quantity by quantity.

    sympy's convert_to on a whole expression mis-scales when the expression contains
    sums, so each Quantity is converted on its own and substituted.
    """
    if not has_units(expr):
        return expr
    try:
        rep = {q: convert_to(q, BASE_UNITS) for q in quantities(expr)}
        return expr.subs(rep)
    except (TypeError, ValueError):
        # e.g. exp(t/second) with a bare symbol t: leave it until t is bound.
        return expr


def tidy_units(expr):
    """Normalise mixed-unit sums to SI base units; leave clean products as written.

    Also collapses units that cancel (a Reynolds number written with kg, m, Pa, s) and
    evaluates physical constants (R, sigma, g_0) so results are numbers with units.
    """
    from sympy.physics.units.quantities import PhysicalConstant

    if not has_units(expr):
        return expr
    qs = quantities(expr)
    if any(isinstance(q, PhysicalConstant) for q in qs):
        return prefer_derived(to_base(expr))
    if isinstance(expr, sp.Add) or any(isinstance(a, sp.Add) and has_units(a) for a in expr.atoms(sp.Add)):
        return prefer_derived(to_base(expr))
    try:
        if SI.get_dimension_system().is_dimensionless(u.Dimension(SI.get_dimensional_expr(expr))):
            return to_base(expr)
    except Exception:  # noqa: BLE001
        pass
    return expr


def strip_units(expr):
    """Return (numeric expression in SI base units, dimension expression)."""
    if not has_units(expr):
        return expr, sp.S.One
    base = to_base(expr)
    dim = SI.get_dimensional_expr(base)
    return base.subs({q: 1 for q in base.atoms(Quantity)}), dim


def split_units(expr):
    """Split a product into (numeric/symbolic part, unit part)."""
    qs = quantities(expr)
    if not qs:
        return expr, sp.S.One
    if isinstance(expr, sp.Add):
        expr = to_base(expr)
        parts = [split_units(t) for t in expr.args]
        units = {u for _, u in parts}
        if len(units) == 1:
            return sp.Add(*[n for n, _ in parts]), units.pop()
        return expr, sp.S.One
    num, unit = expr.as_independent(*qs, as_Add=False)
    return num, unit


# SI units with scale factor 1, preferred for labels of base-unit values.
_PREFERRED = None


def _preferred_units():
    global _PREFERRED
    if _PREFERRED is None:
        _PREFERRED = [u.volt, u.newton, u.joule, u.watt, u.pascal, u.ohm, u.coulomb, u.farad, u.henry, u.tesla,
                      u.weber, u.hertz, u.ampere, u.meter, u.kilogram, u.second, u.kelvin, u.mole, u.candela,
                      u.meter / u.second, u.meter / u.second**2, u.newton * u.meter, u.meter**2, u.meter**3,
                      u.kilogram / u.meter**3, u.watt / u.meter**2, u.joule / u.kelvin, u.volt / u.meter,
                      u.ampere / u.meter, u.newton / u.meter, u.pascal * u.second, u.meter**2 / u.second]
    return _PREFERRED


def _abbrev(unit) -> str:
    rep = {q: sp.Symbol(str(q.abbrev)) for q in unit.atoms(Quantity)}
    return str(unit.subs(rep)).replace("**", "^")


def prefer_derived(expr):
    """Express a base-unit result in the named SI unit of its dimension (W, N, J, Pa, V, ...)."""
    if not has_units(expr):
        return expr
    try:
        dim = u.Dimension(SI.get_dimensional_expr(expr))
        dimsys = SI.get_dimension_system()
        for cand in _preferred_units():
            if isinstance(cand, Quantity) and dimsys.equivalent_dims(dim, u.Dimension(SI.get_dimensional_expr(cand))):
                return convert_to(expr, cand)
    except Exception:  # noqa: BLE001
        pass
    return expr


_BASE_ABBREV = {"mass": "kg", "length": "m", "time": "s", "current": "A", "temperature": "K",
                "amount_of_substance": "mol", "luminous_intensity": "cd"}


def unit_label(expr) -> str:
    """Short plain-text unit label for axes, e.g. 'm/s' or 'V', for values in SI base units.

    Derived from the dimension alone, so a sum or a piecewise expression gets the same
    label as a product would.
    """
    if not has_units(expr):
        return ""
    if isinstance(expr, sp.Piecewise):
        expr = next((e for e, _ in expr.args if has_units(e)), expr.args[0].expr)
    try:
        dim = u.Dimension(SI.get_dimensional_expr(to_base(expr)))
    except Exception:  # noqa: BLE001
        return ""
    dimsys = SI.get_dimension_system()
    if dimsys.is_dimensionless(dim):
        return ""
    for cand in _preferred_units():
        try:
            if dimsys.equivalent_dims(dim, u.Dimension(SI.get_dimensional_expr(cand))):
                return _abbrev(cand)
        except Exception:  # noqa: BLE001
            continue
    order = list(_BASE_ABBREV)
    deps = dict(sorted(((str(getattr(k, "name", k)), v) for k, v in dimsys.get_dimensional_dependencies(dim).items()),
                       key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
    num = [f"{_BASE_ABBREV.get(k, k)}" + (f"^{v}" if v != 1 else "") for k, v in deps.items() if v > 0]
    den = [f"{_BASE_ABBREV.get(k, k)}" + (f"^{-v}" if v != -1 else "") for k, v in deps.items() if v < 0]
    label = "*".join(num) if num else "1"
    if den:
        label += "/" + ("(" + "*".join(den) + ")" if len(den) > 1 else den[0])
    return label


def convert(expr, target):
    """Convert expr to target units, e.g. convert(5*km, m) -> 5000*m."""
    if not isinstance(expr, sp.Basic):
        raise UnitError("Only a single quantity can be converted.")
    target_q = quantities(target)
    if not target_q:
        raise UnitError("The right side of '->' must be a unit, e.g. '-> km/hr'.")
    if target_q <= ANGLE_UNITS and not has_units(expr):
        expr = expr * u.radian  # plain numbers are radians
    try:
        base = strip_angles(to_base(expr))          # angles are dimensionless: rad -> 1, deg -> pi/180
        target_base = strip_angles(to_base(target))
        result = sp.simplify((base / target_base)) * target if not base.free_symbols else \
            sp.powsimp((base / target_base).expand()) * target
        if quantities(result) - target_q:
            result = convert_to(expr, target)  # fall back to sympy's route for odd cases
    except (TypeError, ValueError):
        names = ", ".join(sorted(str(s) for s in expr.free_symbols)) or "the expression"
        raise UnitError(f"Cannot convert while {names} is still symbolic. Give it a value first, "
                        f"or convert inside a function call, e.g. f(2 s) -> {unit_label(target) or 'unit'}.") from None
    leftover = quantities(result) - target_q
    if leftover:
        raise UnitError(
            f"Cannot convert a quantity with dimension {dimension_of(expr)} to "
            f"{unit_label(target) or target} (dimension {dimension_of(target)})."
        )
    return result


def si_value(x, like=None, name: str = "value") -> float:
    """Numeric value of x in SI base units, checking its dimension against ``like`` (a reference quantity).

    ``like=None`` accepts a plain number only; ``like=1`` accepts either. Raises UnitError with the
    dimension it expected, so a formula given the wrong quantity fails instead of guessing.
    """
    x = sp.sympify(x)
    if like is None:
        if has_units(x):
            raise UnitError(f"{name} must be a plain number, got units of {dimension_of(x)}.")
        return float(x)
    if like == 1:
        return float(strip_units(x)[0]) if has_units(x) else float(x)
    want = dimension_of(like)
    if not has_units(x):
        if want == 1:
            return float(x)
        raise UnitError(f"{name} needs units of {unit_label(like)} (dimension {want}); got a plain number.")
    got = dimension_of(x)
    dimsys = SI.get_dimension_system()
    if not dimsys.equivalent_dims(u.Dimension(got), u.Dimension(want)):
        raise UnitError(f"{name} needs units of {unit_label(like)} (dimension {want}); got {got}.")
    return float(strip_units(x)[0])


def quantity(value: float, unit):
    """A Float with units, e.g. quantity(3.0, u.meter / u.second)."""
    return sp.Float(value) * unit
