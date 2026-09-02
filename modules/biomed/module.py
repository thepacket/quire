"""Biology and medicine: pharmacokinetics and dosing, population and epidemic models, enzyme kinetics."""
from __future__ import annotations

import numpy as np
import sympy as sp
from sympy.physics import units as u

from quire.engine import units as U
from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "biomed"
DESCRIPTION = "Pharmacokinetics and dosing, population and epidemic models, enzyme kinetics."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _num(v, what="a number"):
    v = sp.sympify(v)
    if v.free_symbols:
        raise EvalError(f"{what} must be a number here.")
    return float(U.strip_units(v)[0])


# ---- pharmacokinetics
def pk_iv(dose, V, k, t):
    """Plasma concentration after an IV bolus: C = D/V e^(-k t)."""
    return sp.simplify(dose / V * sp.exp(-k * t))


def pk_oral(dose, F, V, ka, ke, t):
    """Bateman function for a single oral dose: C = F D ka / (V (ka - ke)) (e^(-ke t) - e^(-ka t))."""
    return sp.simplify(F * dose * ka / (V * (ka - ke)) * (sp.exp(-ke * t) - sp.exp(-ka * t)))


def pk_elimination_rate(half_life):
    return sp.simplify(sp.log(2) / half_life)


def pk_half_life(k):
    return sp.simplify(sp.log(2) / k)


def clearance(k, V):
    return sp.simplify(k * V)


def loading_dose(C_target, V, F=1):
    return sp.simplify(C_target * V / F)


def maintenance_dose(C_ss, CL, tau, F=1):
    """Dose per interval tau that holds the average steady-state concentration C_ss."""
    return sp.simplify(C_ss * CL * tau / F)


def steady_state_conc(dose, CL, tau, F=1):
    return sp.simplify(F * dose / (CL * tau))


def accumulation_ratio(k, tau):
    return sp.simplify(1 / (1 - sp.exp(-k * tau)))


def time_to_steady_state(half_life, fraction=0.97):
    """Time to reach a fraction of steady state: -ln(1 - f) / k."""
    return sp.simplify(-sp.log(1 - sp.sympify(fraction)) * half_life / sp.log(2))


def peak_time_oral(ka, ke):
    return sp.simplify(sp.log(ka / ke) / (ka - ke))


def creatinine_clearance(age, weight, creatinine, female=False):
    """Cockcroft-Gault: (140 - age) weight / (72 creatinine) mL/min, x 0.85 for women (creatinine in mg/dL)."""
    age_y = U.si_value(age, u.year, "age") / U.si_value(1 * u.year) if U.has_units(age) else float(age)
    w = U.si_value(weight, u.kilogram, "weight") if U.has_units(weight) else float(weight)
    cr = _num(creatinine, "creatinine")
    val = (140 - age_y) * w / (72 * cr) * (0.85 if female else 1)
    _note("Cockcroft-Gault estimate; creatinine in mg/dL, result in mL/min")
    return sp.Float(val, 6) * u.milliliter / u.minute


def bmi(weight, height):
    return sp.simplify(weight / height ** 2)


def body_surface_area(weight, height):
    """Mosteller: sqrt(height[cm] x weight[kg] / 3600) m^2."""
    w = U.si_value(weight, u.kilogram, "weight") if U.has_units(weight) else float(weight)
    h = U.si_value(height, u.meter, "height") * 100 if U.has_units(height) else float(height)
    return sp.Float(np.sqrt(h * w / 3600), 6) * u.meter ** 2


def pediatric_dose(adult_dose, weight):
    """Clark's rule: adult dose x weight[kg] / 70 kg."""
    w = U.si_value(weight, u.kilogram, "weight") if U.has_units(weight) else float(weight)
    return sp.simplify(adult_dose * sp.Float(w / 70, 6))


# ---- population and epidemics
def exponential_growth(P0, r, t):
    return sp.simplify(P0 * sp.exp(r * t))


def logistic_growth(P0, r, K, t):
    return sp.simplify(K / (1 + (K - P0) / P0 * sp.exp(-r * t)))


def doubling_time(r):
    return sp.simplify(sp.log(2) / r)


def _table(ts, cols):
    return sp.ImmutableMatrix([[sp.Float(t, 6)] + [sp.Float(c[k], 6) for c in cols] for k, t in enumerate(ts)])


def _integrate(rhs, y0, T, n=400):
    from scipy.integrate import solve_ivp

    ts = np.linspace(0, T, n)
    sol = solve_ivp(rhs, (0, T), y0, t_eval=ts, rtol=1e-7, atol=1e-9)
    return ts, sol.y


def lotka_volterra(alpha, beta, delta, gamma, prey0, predator0, T, steps=200):
    """Predator-prey trajectories as a table [t, prey, predator]."""
    a, b, d, g = (_num(v) for v in (alpha, beta, delta, gamma))
    ts, y = _integrate(lambda t, s: [a * s[0] - b * s[0] * s[1], d * s[0] * s[1] - g * s[1]],
                       [_num(prey0), _num(predator0)], _num(T), int(steps))
    _note("Lotka-Volterra: prey' = α prey - β prey predator, predator' = δ prey predator - γ predator")
    return _table(ts, [y[0], y[1]])


def sir_model(beta, gamma, S0, I0, R0, T, steps=200):
    """SIR epidemic as a table [t, S, I, R]."""
    b, g = _num(beta), _num(gamma)
    N = _num(S0) + _num(I0) + _num(R0)
    ts, y = _integrate(lambda t, s: [-b * s[0] * s[1] / N, b * s[0] * s[1] / N - g * s[1], g * s[1]],
                       [_num(S0), _num(I0), _num(R0)], _num(T), int(steps))
    _note(f"SIR with R0 = β/γ = {b / g:.3g}; peak infected {y[1].max():.4g} at t = {ts[int(y[1].argmax())]:.3g}")
    return _table(ts, [y[0], y[1], y[2]])


def basic_reproduction_number(beta, gamma):
    return sp.simplify(beta / gamma)


def herd_immunity_threshold(R0):
    return sp.simplify(1 - 1 / sp.sympify(R0))


def epidemic_final_size(R0):
    """Fraction ever infected in an SIR epidemic: the root of 1 - z - exp(-R0 z)."""
    from scipy.optimize import brentq

    r0 = _num(R0)
    if r0 <= 1:
        return sp.S.Zero
    z = brentq(lambda z: 1 - z - np.exp(-r0 * z), 1e-9, 1 - 1e-12)
    return sp.Float(z, 6)


def table_column(table, k):
    """Column k (1-based) of a table such as sir_model(...) as a list, for plotting."""
    M = sp.Matrix(table)
    k = int(k)
    if not 1 <= k <= M.cols:
        raise EvalError(f"The table has {M.cols} columns.")
    return [M[i, k - 1] for i in range(M.rows)]


# ---- enzyme kinetics and pharmacodynamics
def michaelis_menten(Vmax, Km, S):
    return sp.simplify(Vmax * S / (Km + S))


def hill_equation(Emax, EC50, n, C):
    return sp.simplify(Emax * C ** n / (EC50 ** n + C ** n))


def lineweaver_burk(Vmax, Km, S):
    """1/v = Km/Vmax 1/S + 1/Vmax."""
    return sp.simplify(Km / (Vmax * S) + 1 / Vmax)


def register(api):
    P = "Pharmacokinetics"
    api.function("pk_iv", pk_iv, signature="pk_iv(dose, V, k, t)", doc="IV bolus concentration D/V e^(-k t)", category=P,
                 example="pk_iv(500 mg, 40 L, 0.1/hr, 6 hr) -> mg/L")
    api.function("pk_oral", pk_oral, signature="pk_oral(dose, F, V, ka, ke, t)", doc="single oral dose (Bateman)", category=P,
                 example="pk_oral(500 mg, 0.8, 40 L, 1.2/hr, 0.1/hr, t) -> mg/L")
    api.function("pk_elimination_rate", pk_elimination_rate, signature="pk_elimination_rate(t_half)", doc="ln 2 / t½", category=P)
    api.function("pk_half_life", pk_half_life, signature="pk_half_life(k)", doc="ln 2 / k", category=P)
    api.function("clearance", clearance, signature="clearance(k, V)", doc="CL = k V", category=P)
    api.function("loading_dose", loading_dose, signature="loading_dose(C_target, V, F)", doc="C V / F", category=P)
    api.function("maintenance_dose", maintenance_dose, signature="maintenance_dose(C_ss, CL, tau, F)", doc="C_ss CL τ / F", category=P)
    api.function("steady_state_conc", steady_state_conc, signature="steady_state_conc(dose, CL, tau, F)", doc="average C at steady state", category=P)
    api.function("accumulation_ratio", accumulation_ratio, signature="accumulation_ratio(k, tau)", doc="1 / (1 - e^(-k τ))", category=P)
    api.function("time_to_steady_state", time_to_steady_state, signature="time_to_steady_state(t_half, fraction)", doc="time to reach a fraction of steady state", category=P)
    api.function("peak_time_oral", peak_time_oral, signature="peak_time_oral(ka, ke)", doc="t_max of a single oral dose", category=P)
    api.function("creatinine_clearance", creatinine_clearance, signature="creatinine_clearance(age, weight, creatinine, female)",
                 doc="Cockcroft-Gault estimate (mL/min)", category=P, example="creatinine_clearance(65, 70 kg, 1.2)")
    api.function("bmi", bmi, signature="bmi(weight, height)", doc="body mass index", category=P, example="bmi(70 kg, 1.75 m)")
    api.function("body_surface_area", body_surface_area, signature="body_surface_area(weight, height)", doc="Mosteller formula", category=P)
    api.function("pediatric_dose", pediatric_dose, signature="pediatric_dose(adult_dose, weight)", doc="Clark's rule", category=P)
    B = "Population & epidemics"
    api.function("exponential_growth", exponential_growth, signature="exponential_growth(P0, r, t)", doc="P0 e^(r t)", category=B)
    api.function("logistic_growth", logistic_growth, signature="logistic_growth(P0, r, K, t)", doc="logistic curve with capacity K", category=B,
                 example="logistic_growth(10, 0.5, 1000, t)")
    api.function("doubling_time", doubling_time, signature="doubling_time(r)", doc="ln 2 / r", category=B)
    api.function("lotka_volterra", lotka_volterra, signature="lotka_volterra(alpha, beta, delta, gamma, prey0, predator0, T)",
                 doc="predator-prey table [t, prey, predator]", category=B, example="lotka_volterra(1.1, 0.4, 0.1, 0.4, 10, 10, 30)")
    api.function("sir_model", sir_model, signature="sir_model(beta, gamma, S0, I0, R0, T)", doc="epidemic table [t, S, I, R]", category=B,
                 example="sir_model(0.5, 0.2, 990, 10, 0, 60)")
    api.function("basic_reproduction_number", basic_reproduction_number, signature="basic_reproduction_number(beta, gamma)", doc="R0 = β/γ", category=B)
    api.function("herd_immunity_threshold", herd_immunity_threshold, signature="herd_immunity_threshold(R0)", doc="1 - 1/R0", category=B)
    api.function("epidemic_final_size", epidemic_final_size, signature="epidemic_final_size(R0)", doc="fraction ever infected", category=B)
    api.function("table_column", table_column, signature="table_column(table, k)", doc="column k of a table as a list (for scatter plots)", category=B,
                 example="table_column(sir_model(0.5, 0.2, 990, 10, 0, 60), 3)")
    K = "Enzyme kinetics"
    api.function("michaelis_menten", michaelis_menten, signature="michaelis_menten(Vmax, Km, S)", doc="v = Vmax S / (Km + S)", category=K)
    api.function("hill_equation", hill_equation, signature="hill_equation(Emax, EC50, n, C)", doc="sigmoid dose-response", category=K)
    api.function("lineweaver_burk", lineweaver_burk, signature="lineweaver_burk(Vmax, Km, S)", doc="1/v as a function of S", category=K)
