"""Chemistry and materials: formulas, molar masses, balancing, solutions, kinetics, material data.

Formulas are written as plain names: molar_mass(H2O), moles(10 g, CaCO3). Parentheses
in formulas are not supported by the worksheet syntax; expand them (Ca(OH)2 -> CaO2H2).
"""
import re

import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "chemistry"
DESCRIPTION = "Molar masses, equation balancing, stoichiometry, pH, kinetics, thermochemistry, materials."

try:
    import periodictable as pt
except ImportError:  # pragma: no cover
    pt = None


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _formula(f):
    if pt is None:
        raise EvalError("Chemistry needs the periodictable package (pip install periodictable).")
    name = str(f)
    try:
        return pt.formula(name)
    except Exception:  # noqa: BLE001
        raise EvalError(f"'{name}' is not a chemical formula I can read (write Ca(OH)2 as CaO2H2).") from None


def _element(sym):
    if pt is None:
        raise EvalError("Chemistry needs the periodictable package.")
    name = str(sym)
    el = getattr(pt, name, None)
    if el is None or not hasattr(el, "number"):
        raise EvalError(f"'{name}' is not an element symbol.")
    return el


def molar_mass(formula):
    m = _formula(formula).mass
    return sp.Float(m, 6) * u.gram / u.mol


def atomic_number(element):
    return sp.Integer(_element(element).number)


def element_name(element):
    return sp.Symbol(_element(element).name)


def composition(formula):
    """Mass fractions of each element: rows [element, fraction]."""
    f = _formula(formula)
    rows = [[sp.Symbol(str(el)), sp.Float(fr, 4)] for el, fr in sorted(f.mass_fraction.items(), key=lambda kv: -kv[1])]
    return sp.ImmutableMatrix(rows)


def _atoms(formula):
    return {str(el): int(n) for el, n in _formula(formula).atoms.items()}


def balance(reactants, products):
    """Smallest integer coefficients balancing reactants -> products (nullspace of the element matrix)."""
    reactants, products = list(reactants), list(products)
    species = reactants + products
    atoms = [_atoms(s) for s in species]
    elements = sorted({e for a in atoms for e in a})
    M = sp.Matrix([[a.get(e, 0) * (1 if i < len(reactants) else -1) for i, a in enumerate(atoms)] for e in elements])
    null = M.nullspace()
    if len(null) != 1:
        raise EvalError("The equation has no unique balance (check the species).")
    v = null[0]
    den = sp.ilcm(*[sp.fraction(x)[1] for x in v])
    coeffs = [int(x * den) for x in v]
    if coeffs[0] < 0:
        coeffs = [-c for c in coeffs]
    if any(c <= 0 for c in coeffs):
        raise EvalError("No positive balance exists; a species is on the wrong side.")
    g = sp.igcd(*coeffs)
    coeffs = [c // g for c in coeffs]
    lhs = " + ".join(f"{c if c != 1 else ''}{s}".strip() for c, s in zip(coeffs, reactants))
    rhs = " + ".join(f"{c if c != 1 else ''}{s}".strip() for c, s in zip(coeffs[len(reactants):], products))
    _note(f"balanced: {lhs} -> {rhs}")
    return [sp.Integer(c) for c in coeffs]


def moles(mass, formula):
    return sp.simplify(mass / molar_mass(formula))


def mass_of(n, formula):
    return sp.simplify(n * molar_mass(formula))


def molarity(n, volume):
    return sp.simplify(n / volume)


def dilution_v1(c1, c2, v2):
    """Volume of stock needed: C1 V1 = C2 V2."""
    return sp.simplify(c2 * v2 / c1)


def pH(H_concentration):
    """pH from [H+] in mol/L (or a plain number in mol/L)."""
    c = sp.sympify(H_concentration)
    if U.has_units(c):
        c = U.convert(c, u.mol / u.liter) / (u.mol / u.liter)
    return -sp.log(c, 10)


def pOH(OH_concentration):
    c = sp.sympify(OH_concentration)
    if U.has_units(c):
        c = U.convert(c, u.mol / u.liter) / (u.mol / u.liter)
    return -sp.log(c, 10)


def henderson_hasselbalch(pKa, base, acid):
    return sp.sympify(pKa) + sp.log(sp.sympify(base) / sp.sympify(acid), 10)


def arrhenius(A, Ea, T):
    """k = A exp(-Ea/(R T))"""
    return sp.simplify(A * sp.exp(-Ea / (u.R * T)))


def rate_ratio(Ea, T1, T2):
    """k2/k1 for a temperature change (Arrhenius)."""
    return sp.simplify(sp.exp(-Ea / u.R * (1 / T2 - 1 / T1)))


def half_life(k):
    """First-order half-life ln 2 / k."""
    return sp.simplify(sp.log(2) / k)


def first_order(C0, k, t):
    return C0 * sp.exp(-k * t)


def gibbs(dH, T, dS):
    return sp.simplify(dH - T * dS)


def equilibrium_constant(dG, T):
    return sp.simplify(sp.exp(-dG / (u.R * T)))


def nernst(E0, n, Q, T=sp.Float(298.15) * u.K):
    """E = E0 - (R T/(n F)) ln Q"""
    F = u.elementary_charge * u.avogadro_number
    return sp.simplify(E0 - u.R * T / (n * F) * sp.log(Q))


def ideal_gas_volume(n, T, P):
    return sp.simplify(n * u.R * T / P)


# ---- materials (typical room-temperature values; use datasheets for design)
MATERIALS = {
    # name: (E in GPa, density kg/m3, yield MPa, ultimate MPa, alpha 1e-6/K, k W/mK, Poisson)
    "steel": (200, 7850, 250, 400, 12, 50, 0.30),
    "stainless": (193, 8000, 215, 505, 17, 16, 0.30),
    "aluminum": (69, 2700, 95, 110, 23, 237, 0.33),
    "aluminum_6061": (69, 2700, 276, 310, 23.6, 167, 0.33),
    "copper": (117, 8960, 70, 220, 17, 401, 0.34),
    "brass": (100, 8500, 200, 350, 19, 109, 0.34),
    "titanium": (110, 4500, 830, 900, 8.6, 21.9, 0.34),
    "cast_iron": (170, 7200, 130, 200, 10.8, 55, 0.26),
    "concrete": (30, 2400, 3, 30, 12, 1.7, 0.20),
    "glass": (70, 2500, 33, 33, 9, 1.0, 0.22),
    "nylon": (2.8, 1150, 45, 75, 80, 0.25, 0.40),
    "pvc": (3.0, 1400, 50, 52, 70, 0.19, 0.40),
    "wood_pine": (11, 500, 40, 40, 5, 0.12, 0.30),
    "carbon_fiber": (150, 1600, 600, 600, -0.5, 7, 0.30),
}


def _mat(m):
    name = str(m)
    if name not in MATERIALS:
        raise EvalError(f"Unknown material '{name}'. Known: {', '.join(MATERIALS)}.")
    _note(f"typical room-temperature value for {name}; confirm with a datasheet for design work")
    return MATERIALS[name]


def material_E(m):
    return sp.Integer(_mat(m)[0]) * U.GPa if float(_mat(m)[0]).is_integer() else sp.Float(_mat(m)[0]) * U.GPa


def material_density(m):
    return sp.Integer(_mat(m)[1]) * u.kg / u.m ** 3


def material_yield(m):
    return sp.Integer(_mat(m)[2]) * U.MPa


def material_ultimate(m):
    return sp.Integer(_mat(m)[3]) * U.MPa


def material_alpha(m):
    return sp.Float(_mat(m)[4] * 1e-6) / u.K


def material_k(m):
    return sp.Float(_mat(m)[5]) * u.W / (u.m * u.K)


def material_poisson(m):
    return sp.Float(_mat(m)[6])


def register(api):
    C = "Chemistry"
    api.function("molar_mass", molar_mass, signature="molar_mass(H2O)", doc="molar mass in g/mol", category=C, example="molar_mass(CaCO3)")
    api.function("atomic_number", atomic_number, signature="atomic_number(Fe)", doc="atomic number", category=C)
    api.function("element_name", element_name, signature="element_name(Fe)", doc="element name", category=C)
    api.function("composition", composition, signature="composition(H2O)", doc="mass fraction of each element", category=C)
    api.function("balance", balance, signature="balance([H2, O2], [H2O])", doc="balance a chemical equation", category=C,
                 example="balance([C3H8, O2], [CO2, H2O])")
    api.function("moles", moles, signature="moles(mass, formula)", doc="amount of substance", category=C, example="moles(10 g, NaCl)")
    api.function("mass_of", mass_of, signature="mass_of(n, formula)", doc="mass of n moles", category=C)
    api.function("molarity", molarity, signature="molarity(n, V)", doc="n / V", category=C)
    api.function("dilution_v1", dilution_v1, signature="dilution_v1(c1, c2, v2)", doc="C1 V1 = C2 V2", category=C)
    api.function("pH", pH, signature="pH(H_conc)", doc="-log10 [H+]", category=C, example="pH(0.001)")
    api.function("pOH", pOH, signature="pOH(OH_conc)", doc="-log10 [OH-]", category=C)
    api.function("henderson_hasselbalch", henderson_hasselbalch, signature="henderson_hasselbalch(pKa, base, acid)", doc="buffer pH", category=C)
    api.function("arrhenius", arrhenius, signature="arrhenius(A, Ea, T)", doc="k = A exp(-Ea/RT)", category=C)
    api.function("rate_ratio", rate_ratio, signature="rate_ratio(Ea, T1, T2)", doc="k2/k1 (Arrhenius)", category=C,
                 example="rate_ratio(50 kJ/mol, 298 K, 308 K)")
    api.function("half_life", half_life, signature="half_life(k)", doc="ln 2 / k (first order)", category=C)
    api.function("first_order", first_order, signature="first_order(C0, k, t)", doc="C0 exp(-k t)", category=C)
    api.function("gibbs", gibbs, signature="gibbs(dH, T, dS)", doc="ΔG = ΔH - T ΔS", category=C)
    api.function("equilibrium_constant", equilibrium_constant, signature="equilibrium_constant(dG, T)", doc="exp(-ΔG/RT)", category=C)
    api.function("nernst", nernst, signature="nernst(E0, n, Q, T)", doc="Nernst equation", category=C)
    api.function("ideal_gas_volume", ideal_gas_volume, signature="ideal_gas_volume(n, T, P)", doc="n R T / P", category=C)
    M = "Materials"
    for name in MATERIALS:
        api.constant(name, sp.Symbol(name), doc="material token", category=M)
    api.function("material_E", material_E, signature="material_E(steel)", doc="Young's modulus", category=M, example="material_E(aluminum)")
    api.function("material_density", material_density, signature="material_density(steel)", doc="density", category=M)
    api.function("material_yield", material_yield, signature="material_yield(steel)", doc="yield strength", category=M)
    api.function("material_ultimate", material_ultimate, signature="material_ultimate(steel)", doc="ultimate strength", category=M)
    api.function("material_alpha", material_alpha, signature="material_alpha(steel)", doc="thermal expansion coefficient", category=M)
    api.function("material_k", material_k, signature="material_k(steel)", doc="thermal conductivity", category=M)
    api.function("material_poisson", material_poisson, signature="material_poisson(steel)", doc="Poisson's ratio", category=M)
