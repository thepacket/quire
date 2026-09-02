"""Measured values with automatic propagation, dimensions on symbols, user constants for recognize."""
from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def run(ev, *sources):
    return ev.evaluate([{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)])


def test_measured_definitions_and_propagation(ev):
    rs = run(ev, "x = 12.3 ± 0.2 m\ny = 4.0 m ± 5 cm\nz = 2.5 +- 0.1\nw = 7 +/- 1 kg",
             "x^2", "x y -> cm^2", "x/y", "nominal(x^2)", "uncertainty(x y)", "z^2", "x + 3 s", "x + 2 m")
    o = rs[0]["outputs"]
    assert [v["plain"] for v in o] == ["12.3 ± 0.2 meter", "4 ± 0.05 meter", "2.5 ± 0.1", "7 ± 1 kilogram"]
    assert o[0]["latex"] == "12.3 \\pm 0.2 \\, \\text{m}" and o[0]["head"] == "x"
    sq = rs[1]["outputs"][0]
    assert sq["plain"] == "151.29 ± 4.9 meter**2" and sq["measured"]["sigma"] == pytest.approx(4.92)
    assert sq["notes"] == ["uncertainty by linear propagation from x"]
    assert rs[2]["outputs"][0]["plain"] == "492000 ± 1e4 centimeter**2"     # sigma = sqrt((4*0.2)^2 + (12.3*0.05)^2) m^2
    assert rs[2]["outputs"][0]["measured"]["sigma"] == pytest.approx(10090.7, rel=1e-4)
    assert rs[3]["outputs"][0]["plain"] == "3.075 ± 0.063"
    assert rs[4]["outputs"][0]["plain"] == "151.29*meter**2"
    assert rs[5]["outputs"][0]["plain"].startswith("1.0090713")
    assert rs[6]["outputs"][0]["plain"] == "6.25 ± 0.5"
    assert not rs[7]["ok"] and "length" in rs[7]["error"] and "time" in rs[7]["error"]
    assert rs[8]["outputs"][0]["plain"] == "14.3 ± 0.2 meter"


def test_measured_in_text_and_plots(ev):
    rs = ev.evaluate([{"id": 1, "type": "math", "source": "x = 2.0 ± 0.1 m"},
                      {"id": 2, "type": "text", "source": "area {{x^2}}"},
                      {"id": 3, "type": "plot", "kind": "function", "exprs": "x t", "var": "t", "xmin": "0", "xmax": "2"}])
    assert rs[1]["values"][0]["plain"] == "4 ± 0.4 meter**2"
    assert rs[2]["ok"] and rs[2]["ylabel"] == "[m]" and abs(rs[2]["series"][0]["y"][-1] - 4) < 1e-9   # nominal value
    r = run(ev, "x = 1 ± 0.1 m ± 2")[0]
    assert not r["ok"]
    r = run(ev, "x = 1 ± 0.1 kg", "x + 1 m")[1]
    assert not r["ok"]


def test_dimensions_on_symbols(ev):
    rs = run(ev, "assume m mass, a acceleration\nassume L is a length\nassume R in kohm", "F = m a", "F -> N", "F -> kN",
             "L^2 -> cm^2", "L + 3 s", "m a + 2 N", "assume q banana", "R I -> V", "assume I current\nR I -> V")
    o = rs[0]["outputs"]
    assert rs[0]["defines"] == ["m", "a", "L", "R"] and o[0]["plain"] == "m mass; a acceleration"
    assert o[1]["latex"] == "L : \\text{length}\\ [\\text{m}]"
    assert rs[1]["outputs"][0]["plain"] == "kilogram*meter*a*m/second**2"
    assert rs[2]["outputs"][0]["plain"] == "newton*a*m" and rs[3]["outputs"][0]["plain"] == "kilonewton*a*m/1000"
    assert rs[4]["outputs"][0]["plain"] == "10000*centimeter**2*L**2"
    assert not rs[5]["ok"] and "time" in rs[5]["error"]
    assert rs[6]["ok"]
    assert not rs[7]["ok"] and "banana" in rs[7]["error"]
    assert not rs[8]["ok"] and "impedance" in rs[8]["error"]           # I has no dimension yet: R I is not a voltage
    assert rs[9]["outputs"][1]["plain"] == "1000*volt*I*R"


def test_recognize_with_user_constants(ev):
    rs = run(ev, "recognize(2.6457513110645907, [sqrt(7)])", "recognize(0.37796447300922723, [sqrt(7)])",
             "recognize(1.6180339887498948482045868343656, [sqrt(5)])", "recognize(0.123456789)")
    assert rs[0]["outputs"][0]["plain"] == "sqrt(7)" and rs[1]["outputs"][0]["plain"] == "sqrt(7)/7"
    assert rs[2]["outputs"][0]["plain"] == "1/2 + sqrt(5)/2"
    assert rs[3]["outputs"][0]["plain"].startswith("0.123456789")   # too few digits: left as it was
