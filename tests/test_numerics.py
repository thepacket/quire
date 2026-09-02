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
    return r["outputs"][-1]


def test_series_family(ev):
    assert val(ev, "taylor(exp(x), x, 0, 4)")["plain"] == "x**3/6 + x**2/2 + x + 1"
    assert val(ev, "pade(exp(x), x, 0, 2, 2)")["plain"] == "(x**2/12 + x/2 + 1)/(x**2/12 - x/2 + 1)"
    assert val(ev, "series_solve(D(y, x) == x + y, y, x, 0, 1)")["plain"] == "x**5/60 + x**4/12 + x**3/3 + x**2 + x + 1"
    out = val(ev, "chebyshev_approx(exp(x), x, -1, 1, 3)")
    assert "Chebyshev" in out["notes"][0]


def test_iterative_family(ev):
    assert abs(float(val(ev, "newton_raphson(x^3 - 2 x - 5, x, 2)")["plain"]) - 2.0945514815423265) < 1e-9
    assert "iterations" in val(ev, "newton_raphson(x^2 - 2, x, 1)")["notes"][0]
    assert abs(float(val(ev, "fixed_point(cos(x), x, 1)")["plain"]) - 0.7390851332151607) < 1e-8
    assert abs(float(val(ev, "bisection(x^2 - 2, x, 1, 2)")["plain"]) - math.sqrt(2)) < 1e-9
    assert abs(float(val(ev, "secant(x^2 - 2, x, 1, 2)")["plain"]) - math.sqrt(2)) < 1e-9
    steps = val(ev, "newton_raphson_steps(x^2 - 2, x, 1)")
    assert steps["plain"].startswith("Matrix(") and "columns" in steps["notes"][0]
    gs = val(ev, "gauss_seidel_iter(matrix([[4, 1], [2, 5]]), [1, 2])")["plain"]
    assert "0.1666666666" in gs and "0.3333333333" in gs


def test_iterative_errors(ev):
    r = ev.evaluate([{"id": 1, "type": "math", "source": "bisection(x^2 + 1, x, 0, 1)"}])[0]
    assert not r["ok"] and "sign change" in r["error"]
    r = ev.evaluate([{"id": 1, "type": "math", "source": "jacobi_iter(matrix([[1, 4], [5, 1]]), [1, 2])"}])[0]
    assert not r["ok"]


def test_discretized_family(ev):
    assert val(ev, "finite_difference(sin(x), x, x, h)")["plain"] == "(sin(h - x) + sin(h + x))/(2*h)"
    fd = float(val(ev, "y = fdm_solve(D(y, x, 2) == -y, y, x, 0, 0, pi/2, 1, 40)", "y(1)")["plain"])
    assert abs(fd - math.sin(1)) < 1e-3
    bvp = float(val(ev, "y = bvp_solve(D(y, x, 2) == -exp(y), y, x, 0, 0, 1, 0)", "y(0.5)")["plain"])
    assert 0.1 < bvp < 0.2
    fem = float(val(ev, "u = fem_solve(1, x, 0, 1, 0, 0, 10)", "u(0.5)")["plain"])
    assert abs(fem - 0.125) < 1e-9
    heat = float(val(ev, "w = heat_fdm(sin(pi x), x, 1, 1, 0.1, 40, 400)", "w(0.5, 0.1)")["plain"])
    assert abs(heat - math.exp(-math.pi ** 2 * 0.1)) < 5e-3
    r = ev.evaluate([{"id": 1, "type": "math", "source": "heat_fdm(sin(pi x), x, 1, 1, 1, 40, 10)"}])[0]
    assert not r["ok"] and "unstable" in r["error"]


def test_time_stepping_family(ev):
    exact = math.exp(-4)
    e = float(val(ev, "y = euler(-2 y, y, x, 0, 1, 2, 0.1)", "y(2)")["plain"])
    r4 = float(val(ev, "y = rk4(-2 y, y, x, 0, 1, 2, 0.1)", "y(2)")["plain"])
    assert abs(r4 - exact) < 1e-5 and abs(e - exact) > 1e-3
    assert "Runge-Kutta" in val(ev, "rk4(-2 y, y, x, 0, 1, 2, 0.1)")["notes"][0]
    sys_ = float(val(ev, "s = rk4([v, -y], [y, v], t, 0, [1, 0], 4, 0.05)", "s[0](pi)")["plain"])
    assert abs(sys_ + 1) < 1e-4
    r = ev.evaluate([{"id": 1, "type": "math", "source": "y = euler(-2 y, y, x, 0, 1, 2, 0.1)\ny(5)"}])[0]
    assert not r["ok"] and "only defined" in r["error"]


def test_numeric_functions_plot(ev):
    res = ev.evaluate([{"id": 1, "type": "math", "source": "y = rk4(-2 y, y, x, 0, 1, 2, 0.25)"},
                       {"id": 2, "type": "plot", "exprs": "y(x), exp(-2 x)", "xmin": "0", "xmax": "2", "samples": "9"}])[1]
    assert res["ok"] and len(res["series"]) == 2
    a, b = res["series"][0]["y"], res["series"][1]["y"]
    assert all(abs(p - q) < 2e-3 for p, q in zip(a, b))


def test_units_rejected(ev):
    r = ev.evaluate([{"id": 1, "type": "math", "source": "newton_raphson(x - 2 m, x, 1)"}])[0]
    assert not r["ok"] and "plain numbers" in r["error"]
