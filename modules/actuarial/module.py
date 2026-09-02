"""Actuarial mathematics: mortality laws, life tables, survival, expectations, premiums and reserves.

A mortality table is a list of one-year death probabilities q_x starting at age 0
(or at a given first age). Interest i is an annual effective rate.
"""
import sympy as sp

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "actuarial"
DESCRIPTION = "Life tables, survival probabilities, life expectancy, annuities, premiums, reserves."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _q(table):
    if isinstance(table, sp.MatrixBase):
        table = list(table)
    if not isinstance(table, (list, tuple)):
        raise EvalError("A mortality table is a list of q_x values.")
    q = [float(v) for v in table]
    if any(not 0 <= v <= 1 for v in q):
        raise EvalError("q_x values must lie between 0 and 1.")
    return q


def makeham_qx(ages, A=0.0001, B=0.00003, c=1.1):
    """q_x from the Gompertz-Makeham law mu(x) = A + B c^x, for a list of ages (or an integer count from age 0)."""
    import math

    A, B, c = float(A), float(B), float(c)
    xs = list(range(int(ages))) if not isinstance(ages, (list, tuple)) else [int(a) for a in ages]
    out = []
    for x in xs:
        integral = A + B * c ** x * (c - 1) / math.log(c)  # integral of mu from x to x+1
        out.append(sp.Float(1 - math.exp(-integral), 8))
    _note(f"Gompertz-Makeham with A={A:g}, B={B:g}, c={c:g}")
    return out


def life_table(qx, radix=100000, first_age=0):
    """Rows: age, l_x, d_x, q_x, e_x (curtate life expectancy)."""
    q = _q(qx)
    l = float(radix)
    rows = []
    ls = []
    for k, qk in enumerate(q):
        ls.append(l)
        l = l * (1 - qk)
    ls.append(l)
    for k, qk in enumerate(q):
        d = ls[k] * qk
        e = sum(ls[j] for j in range(k + 1, len(ls))) / ls[k] if ls[k] > 0 else 0.0
        rows.append([sp.Integer(int(first_age) + k), sp.Float(ls[k], 8), sp.Float(d, 8), sp.Float(qk, 6), sp.Float(e, 6)])
    _note("columns: age, l_x, d_x, q_x, e_x")
    return sp.ImmutableMatrix(rows)


def survival(qx, x, t, first_age=0):
    """t_p_x: probability that a life aged x survives t more years."""
    q = _q(qx)
    i0 = int(x) - int(first_age)
    if i0 < 0 or i0 + int(t) > len(q):
        raise EvalError("The table does not cover these ages.")
    p = 1.0
    for k in range(i0, i0 + int(t)):
        p *= 1 - q[k]
    return sp.Float(p, 8)


def life_expectancy(qx, x, first_age=0):
    """Curtate expectation of life e_x."""
    q = _q(qx)
    i0 = int(x) - int(first_age)
    p, total = 1.0, 0.0
    for k in range(i0, len(q)):
        p *= 1 - q[k]
        total += p
    return sp.Float(total, 6)


def _v(i):
    return 1 / (1 + float(i))


def annuity_due_factor(qx, x, i, n=None, first_age=0):
    """ä_x (or ä_x:n): present value of 1 per year in advance while alive."""
    q = _q(qx)
    i0 = int(x) - int(first_age)
    v = _v(i)
    p, total = 1.0, 0.0
    horizon = len(q) - i0 if n is None else min(int(n), len(q) - i0)
    for k in range(horizon):
        total += p * v ** k
        p *= 1 - q[i0 + k]
    return sp.Float(total, 8)


def whole_life_insurance(qx, x, i, first_age=0):
    """A_x: present value of 1 paid at the end of the year of death."""
    q = _q(qx)
    i0 = int(x) - int(first_age)
    v = _v(i)
    p, total = 1.0, 0.0
    for k in range(i0, len(q)):
        total += p * q[k] * v ** (k - i0 + 1)
        p *= 1 - q[k]
    return sp.Float(total, 8)


def term_insurance(qx, x, i, n, first_age=0):
    q = _q(qx)
    i0 = int(x) - int(first_age)
    v = _v(i)
    p, total = 1.0, 0.0
    for k in range(i0, min(i0 + int(n), len(q))):
        total += p * q[k] * v ** (k - i0 + 1)
        p *= 1 - q[k]
    return sp.Float(total, 8)


def endowment_insurance(qx, x, i, n, first_age=0):
    """Term insurance plus a pure endowment of 1 at n."""
    term = float(term_insurance(qx, x, i, n, first_age))
    pure = float(survival(qx, x, n, first_age)) * _v(i) ** int(n)
    return sp.Float(term + pure, 8)


def net_annual_premium(qx, x, i, n=None, first_age=0):
    """Whole-life (or n-year term) net annual premium: A / ä."""
    A = float(whole_life_insurance(qx, x, i, first_age)) if n is None else float(term_insurance(qx, x, i, n, first_age))
    a = float(annuity_due_factor(qx, x, i, n, first_age))
    _note("equivalence principle: premium = benefit value / annuity-due factor")
    return sp.Float(A / a, 8)


def reserve(qx, x, i, t, first_age=0):
    """Prospective net premium reserve of a whole-life policy issued at x, after t years."""
    P = float(net_annual_premium(qx, x, i, None, first_age))
    A_t = float(whole_life_insurance(qx, int(x) + int(t), i, first_age))
    a_t = float(annuity_due_factor(qx, int(x) + int(t), i, None, first_age))
    return sp.Float(A_t - P * a_t, 8)


def register(api):
    A = "Actuarial"
    api.function("makeham_qx", makeham_qx, signature="makeham_qx(ages, A, B, c)", doc="mortality rates from the Gompertz-Makeham law",
                 category=A, example="makeham_qx(100)")
    api.function("life_table", life_table, signature="life_table(qx, radix, first_age)", doc="age, l_x, d_x, q_x, e_x", category=A,
                 example="life_table(makeham_qx(100))")
    api.function("survival", survival, signature="survival(qx, x, t)", doc="t_p_x", category=A, example="survival(makeham_qx(100), 40, 20)")
    api.function("life_expectancy", life_expectancy, signature="life_expectancy(qx, x)", doc="curtate expectation e_x", category=A)
    api.function("annuity_due_factor", annuity_due_factor, signature="annuity_due_factor(qx, x, i, n)", doc="ä_x or ä_x:n", category=A)
    api.function("whole_life_insurance", whole_life_insurance, signature="whole_life_insurance(qx, x, i)", doc="A_x", category=A)
    api.function("term_insurance", term_insurance, signature="term_insurance(qx, x, i, n)", doc="n-year term A", category=A)
    api.function("endowment_insurance", endowment_insurance, signature="endowment_insurance(qx, x, i, n)", doc="n-year endowment", category=A)
    api.function("net_annual_premium", net_annual_premium, signature="net_annual_premium(qx, x, i, n)", doc="A / ä", category=A,
                 example="net_annual_premium(makeham_qx(100), 40, 0.04)")
    api.function("reserve", reserve, signature="reserve(qx, x, i, t)", doc="prospective net premium reserve", category=A)
