from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
from quire.engine.notation import normalize
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def out(ev, *sources):
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)]
    r = ev.evaluate(cells)[-1]
    assert r["ok"], r["error"]
    return r["outputs"][-1]


@pytest.mark.parametrize("src,expected", [
    ("∫_0^1 x² dx", "integrate(x^2, x, 0, 1)"),
    ("∫ sin(x) dx", "integrate(sin(x), x)"),
    ("∫∫ x y dx dy", "integrate(integrate(x y, x), y)"),
    ("Σ_{k=1}^{n} k²", "sum(k^2, k, 1, n)"),
    ("∑_{k=1}^{∞} 1/k²", "sum(1/k^2, k, 1, oo)"),
    ("∏_{k=1}^{n} k", "product(k, k, 1, n)"),
    ("d/dx sin(x)", "diff(sin(x), x)"),
    ("d²/dx² (x³ + 1)", "diff((x^3 + 1), x, 2)"),
    ("√(x + 1) + √2", "sqrt(x + 1) + sqrt(2)"),
    ("π r²", "pi r^2"),
    ("θ ≤ π/2", "theta <= pi/2"),
    ("x₁ + x₂", "x_1 + x_2"),
    ("\\frac{a}{b} + \\sqrt{x}", "((a)/(b)) + sqrt(x)"),
    ("\\int_0^1 x^2\\,dx", "integrate(x^2, x, 0, 1)"),
    ("\\sum_{k=1}^{n} \\frac{1}{k^2}", "sum(((1)/(k^2)), k, 1, n)"),
    ("\\frac{d}{dx} x^2", "diff(x^2, x)"),
    ("\\alpha \\cdot \\beta \\le \\infty", "alpha * beta <= oo"),
    ("e^{-x}", "e^(-x)"),
])
def test_normalize(src, expected):
    assert normalize(src) == expected


def test_notation_evaluates(ev):
    assert out(ev, "∫_0^1 x² dx")["plain"] == "1/3"
    assert out(ev, "Σ_{k=1}^{n} k")["plain"] == "n**2/2 + n/2"
    assert out(ev, "d/dx sin(x) x")["plain"] == "x*cos(x) + sin(x)"
    assert out(ev, "√(x + 1)²")["plain"] == "x + 1"
    assert out(ev, "\\frac{1}{\\sqrt{2\\pi}} e^{-x^2/2}")["plain"] == "sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))"
    assert out(ev, "f(x) = ∫_0^x t² dt", "f(3)")["plain"] == "9"
    assert out(ev, "θ = 30 deg", "sin(θ)")["plain"] == "1/2"


def test_reading_is_reported(ev):
    o = out(ev, "∫_0^1 x² dx")
    assert "\\int" in o["reading"] and o["plain"] == "1/3"
    assert "reading" not in out(ev, "integrate(x^2, x, 0, 1)")


def test_notation_errors(ev):
    r = ev.evaluate([{"id": 1, "type": "math", "source": "∫ x²"}])[0]
    assert not r["ok"] and "dx" in r["error"]
    r = ev.evaluate([{"id": 1, "type": "math", "source": "Σ k²"}])[0]
    assert not r["ok"] and "bounds" in r["error"]


def test_did_you_mean(ev):
    r = ev.evaluate([{"id": 1, "type": "math", "source": "intergate(x, x)"}])[0]
    assert not r["ok"] and "Did you mean integrate" in r["error"]
