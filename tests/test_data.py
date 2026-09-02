import math
from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def val(ev, *sources):
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)]
    r = ev.evaluate(cells)[-1]
    assert r["ok"], r["error"]
    return r["outputs"][-1]["plain"]


def num(ev, *sources):
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)]
    r = ev.evaluate(cells)[-1]
    assert r["ok"], r["error"]
    out = r["outputs"][-1]
    return float((out.get("approx_plain") or out["plain"]).replace("*", " ").split()[0])


def test_probability_symbolic(ev):
    assert val(ev, "X = normal(mu, sigma)", "pdf(X, x)") == "sqrt(2)*exp(-(mu - x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)"
    assert val(ev, "expected(normal(mu, sigma)^2)") == "mu**2 + sigma**2"
    assert val(ev, "variance_of(poisson(lambda))") == "lambda"
    assert val(ev, "expected(uniform(a, b))") == "a/2 + b/2"
    assert val(ev, "pmf(binomial_dist(10, 1/2), 3)") == "15/128"
    assert val(ev, "prob(binomial_dist(10, 1/2) >= 8)") == "7/128"
    assert val(ev, "X = normal(0, 1)", "Y = normal(0, 1)", "variance_of(X + Y)") == "2"
    assert val(ev, "mgf(exponential(lambda), t)") == "lambda/(lambda - t)"


def test_probability_numeric(ev):
    assert abs(num(ev, "cdf(normal(0, 1), 1.96)") - 0.975) < 1e-3
    assert abs(num(ev, "prob(normal(0, 1) > 1.96)") - 0.025) < 1e-3
    assert abs(num(ev, "quantile_of(normal(0, 1), 0.975)") - 1.95996) < 1e-4
    assert val(ev, "bayes_update([1/2, 1/2], [0.8, 0.3])") == "[0.727272727272727, 0.272727272727273]"
    s = val(ev, "sample_from(poisson(3), 5, 2)")
    assert s.startswith("[") and s.count(",") == 4


def test_statistics(ev):
    assert val(ev, "mean([1, 2, 3, 4])") == "5/2"
    assert "r_squared" in val(ev, "linear_regression([0, 1, 2, 3], [1, 3, 5, 7.2])")
    lo_hi = val(ev, "confidence_interval([5.1, 4.9, 5.3, 5.0, 5.2], 0.95)")
    assert lo_hi.startswith("[4.9") and "5.2" in lo_hi
    t = val(ev, "t_test([5.1, 4.9, 5.3, 5.0, 5.2], 5)")
    assert t.startswith("[1.4")
    chi = val(ev, "chi2_test([[10, 20], [30, 40]])")
    assert chi.startswith("[0.4")
    assert abs(num(ev, "r_squared([0, 1, 2, 3], [1, 3, 5, 7.2], 2 x + 1, x)") - 0.999) < 0.002
    assert val(ev, "histogram([1, 2, 2, 3, 3, 3, 4], 4)")[0] == "["


def test_finance(ev):
    assert abs(num(ev, "pmt(0.05/12, 360, 300000)") - 1610.46) < 0.01
    assert abs(num(ev, "fv(1000, 0.05, 10)") - 1628.89) < 0.01
    assert abs(num(ev, "npv(0.08, [-1000, 300, 400, 500])") - 17.63) < 0.01
    assert abs(num(ev, "irr([-1000, 300, 400, 500])") - 0.0896) < 1e-3
    assert abs(num(ev, "bond_price(1000, 0.05, 0.06, 10)") - 925.61) < 0.01
    assert abs(num(ev, "bond_ytm(925.61, 1000, 0.05, 10)") - 0.06) < 1e-4
    assert abs(num(ev, "black_scholes_call(100, 100, 0.05, 0.2, 1)") - 10.4506) < 1e-3
    assert "erf" in val(ev, "black_scholes_call(S_0, K, r, sigma, tau)")
    assert val(ev, "median_of(exponential(lambda))") == "log(2)/lambda"
    assert val(ev, "polyfit([0, 1, 2, 3], [1, 2, 5, 10], x, 2)") == "1.0*x**2 + 1.0"
    assert abs(num(ev, "effective_rate(0.12, 12)") - 0.126825) < 1e-5
    table = val(ev, "amortization(1000, 0.1, 3)")
    assert table.startswith("Matrix([[1, 402.1")


def test_actuarial(ev):
    p = num(ev, "q = makeham_qx(100)", "survival(q, 40, 20)")
    assert 0.5 < p < 1.0
    e = num(ev, "q = makeham_qx(100)", "life_expectancy(q, 40)")
    assert 20 < e < 60
    prem = num(ev, "q = makeham_qx(100)", "net_annual_premium(q, 40, 0.04)")
    assert 0 < prem < 0.1
    res = num(ev, "q = makeham_qx(100)", "reserve(q, 40, 0.04, 10)")
    assert 0 < res < 1
    assert num(ev, "q = makeham_qx(100)", "reserve(q, 40, 0.04, 0)") < 1e-9


def test_data_no_conflicts(ev):
    assert ev.registry.conflicts() == []
