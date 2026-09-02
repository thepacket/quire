from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def plot(ev, spec, *pre):
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(pre)] + [dict(spec, id=99, type="plot")]
    return ev.evaluate(cells)[-1]


def test_adaptive_sampling_and_gaps(ev):
    r = plot(ev, {"kind": "function", "exprs": "tan(x)", "xmin": "-5", "xmax": "5"})
    assert r["ok"]
    ys = r["series"][0]["y"]
    assert len(ys) > 400 and ys.count(None) >= 3          # refined, and broken at the poles
    assert r["ysuggest"] and abs(r["ysuggest"][1]) < 100  # the spikes do not set the visible range


def test_parametric_polar_equal_aspect(ev):
    r = plot(ev, {"kind": "parametric", "exprs": "cos(t)", "expr2": "sin(t)", "xmin": "0", "xmax": "2 pi"})
    assert r["ok"] and r["equal"] and abs(r["series"][0]["x"][0] - 1) < 1e-9
    r = plot(ev, {"kind": "polar", "exprs": "1 + cos(theta)"})
    assert r["ok"] and r["polar"] and abs(r["series"][0]["x"][0] - 2) < 1e-9


def test_scatter_from_data(ev):
    r = plot(ev, {"kind": "scatter", "exprs": "xs", "expr2": "ys"}, "xs = [1, 2, 3]", "ys = [2, 4, 6]")
    assert r["ok"] and r["series"][0]["type"] == "points" and r["series"][0]["y"] == [2.0, 4.0, 6.0]
    r = plot(ev, {"kind": "scatter", "exprs": "[1, 2]", "expr2": "[1]"})
    assert not r["ok"] and "values" in r["error"]


def test_slope_field_and_implicit(ev):
    r = plot(ev, {"kind": "slope", "exprs": "x - y", "xmin": "-2", "xmax": "2", "ymin": "-2", "ymax": "2", "samples": "8"})
    assert r["ok"] and r["series"][0]["type"] == "segments" and len(r["series"][0]["segments"]) == 64
    r = plot(ev, {"kind": "implicit", "exprs": "x^2 + y^2 == 1", "xmin": "-2", "xmax": "2", "ymin": "-2", "ymax": "2", "samples": "80"})
    segs = r["series"][0]["segments"]
    assert r["ok"] and len(segs) > 100
    assert all(abs((s[0] ** 2 + s[1] ** 2) - 1) < 0.05 for s in segs)  # segment endpoints lie on the circle


def test_slider_definition(ev):
    r = ev.evaluate([{"id": 1, "type": "math", "source": "a = slider(2, 0, 5, 0.5)\nb = 2 a"}])[0]
    assert r["ok"] and r["outputs"][0]["slider"] == {"value": 2.0, "min": 0.0, "max": 5.0, "step": 0.5, "line": 0}
    assert "slider" not in r["outputs"][1] and r["outputs"][1]["plain"] == "4"


def test_units_on_axes_still_work(ev):
    r = plot(ev, {"kind": "function", "exprs": "v t", "xmin": "0 s", "xmax": "2 s", "var": "t"}, "v = 3 m/s")
    assert r["ok"] and r["xlabel"] == "t [s]" and r["ylabel"] == "[m]"
