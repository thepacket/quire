"""Finance: time value of money, annuities and amortization, NPV/IRR, bonds, options.

Rates are per period as decimals (0.05), periods are plain numbers. Money is a plain number.
"""
import numpy as np
import sympy as sp

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "finance"
DESCRIPTION = "Time value of money, annuities, amortization tables, NPV and IRR, bonds, Black-Scholes."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _f(x):
    return sp.Float(float(x))


def fv(pv, rate, n):
    return sp.simplify(sp.sympify(pv) * (1 + sp.sympify(rate)) ** sp.sympify(n))


def pv(fv_, rate, n):
    return sp.simplify(sp.sympify(fv_) / (1 + sp.sympify(rate)) ** sp.sympify(n))


def pmt(rate, n, principal):
    """Level payment of a loan."""
    r, n = sp.sympify(rate), sp.sympify(n)
    if r == 0:
        return sp.simplify(sp.sympify(principal) / n)
    return sp.simplify(sp.sympify(principal) * r / (1 - (1 + r) ** (-n)))


def nper(rate, payment, principal):
    r = sp.sympify(rate)
    return sp.simplify(-sp.log(1 - sp.sympify(principal) * r / sp.sympify(payment)) / sp.log(1 + r))


def rate_solve(n, payment, principal, guess=0.05):
    """Periodic rate that makes the payments repay the principal (numeric)."""
    r = sp.Symbol("r")
    eq = sp.sympify(principal) * r / (1 - (1 + r) ** (-sp.sympify(n))) - sp.sympify(payment)
    try:
        return sp.Float(sp.nsolve(eq, r, float(guess)))
    except Exception as exc:
        raise EvalError(f"rate_solve failed: {str(exc).splitlines()[0]}") from None


def annuity_pv(payment, rate, n, due=False):
    r, n = sp.sympify(rate), sp.sympify(n)
    v = sp.sympify(payment) * (1 - (1 + r) ** (-n)) / r
    return sp.simplify(v * (1 + r) if due else v)


def annuity_fv(payment, rate, n, due=False):
    r, n = sp.sympify(rate), sp.sympify(n)
    v = sp.sympify(payment) * ((1 + r) ** n - 1) / r
    return sp.simplify(v * (1 + r) if due else v)


def perpetuity(payment, rate, growth=0):
    return sp.simplify(sp.sympify(payment) / (sp.sympify(rate) - sp.sympify(growth)))


def amortization(principal, rate, n):
    """Rows: period, payment, interest, principal paid, balance."""
    P, r, n = float(principal), float(rate), int(n)
    pay = P * r / (1 - (1 + r) ** (-n)) if r else P / n
    rows, bal = [], P
    for k in range(1, n + 1):
        interest = bal * r
        prin = pay - interest
        bal -= prin
        rows.append([k, pay, interest, prin, max(bal, 0.0)])
    _note("columns: period, payment, interest, principal, balance")
    return sp.ImmutableMatrix([[sp.Integer(k), _f(a), _f(b), _f(c), _f(d)] for k, a, b, c, d in rows])


def npv(rate, cashflows):
    """Net present value; cashflows[0] is at time 0."""
    r = sp.sympify(rate)
    return sp.simplify(sum(sp.sympify(c) / (1 + r) ** k for k, c in enumerate(cashflows)))


def irr(cashflows):
    cf = [float(c) for c in cashflows]
    coeffs = cf[::-1]  # polynomial in x = 1/(1+r)
    roots = np.roots(coeffs)
    real = [1 / x.real - 1 for x in roots if abs(x.imag) < 1e-9 and x.real > 0]
    if not real:
        raise EvalError("No real IRR for these cash flows.")
    best = min(real, key=lambda v: abs(v))
    _note("internal rate of return: the rate at which NPV = 0")
    return _f(best)


def payback_period(cashflows):
    cf = [float(c) for c in cashflows]
    cum = 0.0
    for k, c in enumerate(cf):
        prev = cum
        cum += c
        if k > 0 and cum >= 0 > prev:
            return _f(k - 1 + (-prev / c if c else 0))
    raise EvalError("The investment is never paid back.")


def effective_rate(nominal, periods_per_year):
    return sp.simplify((1 + sp.sympify(nominal) / sp.sympify(periods_per_year)) ** sp.sympify(periods_per_year) - 1)


def continuous_rate(effective):
    return sp.log(1 + sp.sympify(effective))


def real_rate(nominal, inflation):
    """Fisher: (1 + nominal)/(1 + inflation) - 1"""
    return sp.simplify((1 + sp.sympify(nominal)) / (1 + sp.sympify(inflation)) - 1)


def cagr(begin, end, years):
    return sp.simplify((sp.sympify(end) / sp.sympify(begin)) ** (1 / sp.sympify(years)) - 1)


def rule_of_72(rate):
    return sp.simplify(72 / (100 * sp.sympify(rate)))


def bond_price(face, coupon_rate, ytm, years, freq=2):
    F, c, y, n, m = (sp.sympify(v) for v in (face, coupon_rate, ytm, years, freq))
    N = n * m
    cpn = F * c / m
    return sp.simplify(cpn * (1 - (1 + y / m) ** (-N)) / (y / m) + F / (1 + y / m) ** N)


def bond_duration(face, coupon_rate, ytm, years, freq=2):
    """[Macaulay duration in years, modified duration]."""
    F, c, y, n, m = (float(v) for v in (face, coupon_rate, ytm, years, freq))
    N = int(round(n * m))
    cpn = F * c / m
    times = np.arange(1, N + 1) / m
    cfs = np.full(N, cpn)
    cfs[-1] += F
    disc = (1 + y / m) ** (-times * m)
    price = np.sum(cfs * disc)
    mac = np.sum(times * cfs * disc) / price
    return [_f(mac), _f(mac / (1 + y / m))]


def bond_ytm(price, face, coupon_rate, years, freq=2):
    y = sp.Symbol("y")
    expr = bond_price(face, coupon_rate, y, years, freq) - sp.sympify(price)
    try:
        return sp.Float(sp.nsolve(expr, y, float(coupon_rate) or 0.05))
    except Exception as exc:
        raise EvalError(f"bond_ytm failed: {str(exc).splitlines()[0]}") from None


def _d1d2(S, K, r, sigma, T):
    S, K, r, sigma, T = (sp.sympify(v) for v in (S, K, r, sigma, T))
    d1 = (sp.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * sp.sqrt(T))
    return d1, d1 - sigma * sp.sqrt(T)


def _Phi(x):
    return (1 + sp.erf(x / sp.sqrt(2))) / 2


def black_scholes_call(S, K, r, sigma, T):
    d1, d2 = _d1d2(S, K, r, sigma, T)
    return sp.simplify(sp.sympify(S) * _Phi(d1) - sp.sympify(K) * sp.exp(-sp.sympify(r) * sp.sympify(T)) * _Phi(d2))


def black_scholes_put(S, K, r, sigma, T):
    d1, d2 = _d1d2(S, K, r, sigma, T)
    return sp.simplify(sp.sympify(K) * sp.exp(-sp.sympify(r) * sp.sympify(T)) * _Phi(-d2) - sp.sympify(S) * _Phi(-d1))


def bs_delta_call(S, K, r, sigma, T):
    d1, _ = _d1d2(S, K, r, sigma, T)
    return sp.simplify(_Phi(d1))


def bs_vega(S, K, r, sigma, T):
    d1, _ = _d1d2(S, K, r, sigma, T)
    return sp.simplify(sp.sympify(S) * sp.sqrt(sp.sympify(T)) * sp.exp(-d1 ** 2 / 2) / sp.sqrt(2 * sp.pi))


def register(api):
    T = "Finance: time value of money"
    api.function("fv", fv, signature="fv(pv, rate, n)", doc="future value", category=T, example="fv(1000, 0.05, 10)")
    api.function("pv", pv, signature="pv(fv, rate, n)", doc="present value", category=T)
    api.function("pmt", pmt, signature="pmt(rate, n, principal)", doc="level loan payment", category=T, example="pmt(0.05/12, 360, 300000)")
    api.function("nper", nper, signature="nper(rate, payment, principal)", doc="number of payments", category=T)
    api.function("rate_solve", rate_solve, signature="rate_solve(n, payment, principal)", doc="implied periodic rate", category=T)
    api.function("annuity_pv", annuity_pv, signature="annuity_pv(payment, rate, n, due)", doc="present value of an annuity", category=T)
    api.function("annuity_fv", annuity_fv, signature="annuity_fv(payment, rate, n, due)", doc="future value of an annuity", category=T)
    api.function("perpetuity", perpetuity, signature="perpetuity(payment, rate, growth)", doc="growing perpetuity", category=T)
    api.function("amortization", amortization, signature="amortization(principal, rate, n)", doc="payment schedule table", category=T,
                 example="amortization(10000, 0.06/12, 12)")
    api.function("effective_rate", effective_rate, signature="effective_rate(nominal, m)", doc="effective annual rate", category=T)
    api.function("continuous_rate", continuous_rate, signature="continuous_rate(effective)", doc="ln(1 + i)", category=T)
    api.function("real_rate", real_rate, signature="real_rate(nominal, inflation)", doc="Fisher equation", category=T)
    api.function("cagr", cagr, signature="cagr(begin, end, years)", doc="compound annual growth rate", category=T)
    api.function("rule_of_72", rule_of_72, signature="rule_of_72(rate)", doc="doubling time estimate", category=T)
    I = "Finance: investments"
    api.function("npv", npv, signature="npv(rate, [cashflows])", doc="net present value, first flow at t = 0", category=I,
                 example="npv(0.08, [-1000, 300, 400, 500])")
    api.function("irr", irr, signature="irr([cashflows])", doc="internal rate of return", category=I, example="irr([-1000, 300, 400, 500])")
    api.function("payback_period", payback_period, signature="payback_period([cashflows])", doc="years to recover the outlay", category=I)
    api.function("bond_price", bond_price, signature="bond_price(face, coupon, ytm, years, freq)", doc="bond price", category=I,
                 example="bond_price(1000, 0.05, 0.06, 10)")
    api.function("bond_duration", bond_duration, signature="bond_duration(face, coupon, ytm, years, freq)", doc="[Macaulay, modified]", category=I)
    api.function("bond_ytm", bond_ytm, signature="bond_ytm(price, face, coupon, years, freq)", doc="yield to maturity", category=I)
    api.function("black_scholes_call", black_scholes_call, signature="black_scholes_call(S, K, r, sigma, T)", doc="European call price", category=I,
                 example="black_scholes_call(100, 100, 0.05, 0.2, 1)")
    api.function("black_scholes_put", black_scholes_put, signature="black_scholes_put(S, K, r, sigma, T)", doc="European put price", category=I)
    api.function("bs_delta_call", bs_delta_call, signature="bs_delta_call(S, K, r, sigma, T)", doc="call delta", category=I)
    api.function("bs_vega", bs_vega, signature="bs_vega(S, K, r, sigma, T)", doc="vega", category=I)
