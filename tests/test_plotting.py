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
    assert r["ok"] and r["outputs"][0]["slider"] == {"value": 2.0, "min": 0.0, "max": 5.0, "step": 0.5, "line": 0, "name": "a"}
    assert "slider" not in r["outputs"][1] and r["outputs"][1]["plain"] == "4"


def test_units_on_axes_still_work(ev):
    r = plot(ev, {"kind": "function", "exprs": "v t", "xmin": "0 s", "xmax": "2 s", "var": "t"}, "v = 3 m/s")
    assert r["ok"] and r["xlabel"] == "t [s]" and r["ylabel"] == "[m]"


def test_grid_kinds_go_through_plotly(ev):
    r = plot(ev, {"kind": "contour", "exprs": "x^2 + y^2", "expr2": "8", "samples": "20"})
    assert r["ok"] and r["renderer"] == "plotly" and r["levels"] == 8
    s = r["series"][0]
    assert s["type"] == "grid" and s["style"] == "contour" and len(s["z"]) == 20 and abs(s["z"][0][0] - 50) < 1e-9
    r = plot(ev, {"kind": "heatmap", "exprs": "u(x, t)", "var": "x, t", "xmin": "0", "xmax": "1", "ymin": "0", "ymax": "0.1",
                  "samples": "21"}, "u = heat_fdm(sin(pi x), x, 1, 1, 0.1, 40, 400)")
    assert r["ok"] and r["series"][0]["style"] == "heatmap"
    assert abs(r["series"][0]["z"][0][10] - 1) < 1e-2                 # u(0.5, 0) = sin(pi/2)
    assert r["series"][0]["z"][-1][10] < r["series"][0]["z"][0][10]    # and it cools down
    r = plot(ev, {"kind": "surface", "exprs": "sin(x) cos(y)", "samples": "12"})
    assert r["ok"] and r["three"] and r["series"][0]["style"] == "surface"
    r = plot(ev, {"kind": "curve3d", "exprs": "cos(t), sin(t), t/5", "xmin": "0", "xmax": "4 pi", "samples": "50"})
    assert r["ok"] and r["series"][0]["type"] == "line3d" and abs(r["series"][0]["z"][-1] - 4 * 3.141592653589793 / 5) < 1e-9
    r = plot(ev, {"kind": "curve3d", "exprs": "cos(t), sin(t)"})
    assert not r["ok"] and "three" in r["error"]


def test_annotations_and_error_bars(ev):
    r = plot(ev, {"kind": "function", "exprs": "sin(x)", "xmin": "0", "xmax": "6",
                  "annot": 'mark(pi/2, "peak"), shade(0, pi), hline(0.5, "half"), vline(1), band(4, 5), text(2, -0.5, "note"), point(3, 0, "root")'})
    assert r["ok"]
    kinds = [a["type"] for a in r["annotations"]]
    assert kinds == ["point", "shade", "hline", "vline", "band", "text", "point"]
    peak = r["annotations"][0]
    assert peak["label"] == "peak" and abs(peak["y"] - 1) < 1e-3       # y read off the curve
    assert r["annotations"][1]["series"] == 0 and r["annotations"][6]["label"] == "root"
    r = plot(ev, {"kind": "function", "exprs": "sin(x)", "annot": "circle(1)"})
    assert not r["ok"] and "mark(x" in r["error"]
    r = plot(ev, {"kind": "scatter", "exprs": "[1, 2, 3]", "expr2": "[2, 4, 6]", "expr3": "[0.1, 0.2, 0.3]"})
    assert r["ok"] and r["series"][0]["yerr"] == [0.1, 0.2, 0.3]
    r = plot(ev, {"kind": "scatter", "exprs": "[1, 2, 3]", "expr2": "[2, 4, 6]", "expr3": "0.5"})
    assert r["ok"] and r["series"][0]["yerr"] == [0.5, 0.5, 0.5]


def test_named_points_are_labelled(ev):
    r = plot(ev, {"kind": "shapes", "exprs": "A, segment(A, point(3, 3))"}, "A = point(1, 2)")
    assert r["ok"] and r["series"][0]["labels"] == ["A"] and r["series"][1]["type"] == "line"


def test_slider_series_compiled_to_javascript(ev):
    rs = ev.evaluate([{"id": 1, "type": "math", "source": "a = slider(1, 0, 3)"},
                      {"id": 2, "type": "math", "source": "b = 2"},
                      {"id": 3, "type": "math", "source": "g(x) = a sin(x) + b"},
                      {"id": 4, "type": "plot", "kind": "function", "exprs": "g(x), x^2", "xmin": "0", "xmax": "6"}])
    assert rs[0]["outputs"][0]["slider"]["name"] == "a"
    r = rs[-1]
    assert r["ok"] and r["series"][0]["js"] == ["a*Math.sin(x) + 2"] and r["series"][0]["params"] == ["a"]
    assert "js" not in r["series"][1]                                   # no slider involved
    r = plot(ev, {"kind": "polar", "exprs": "1 + k cos(theta)"}, "k = slider(1, 0, 3)")
    assert r["ok"] and r["series"][0]["js"] == ["(k*Math.cos(theta) + 1)*Math.cos(theta)", "(k*Math.cos(theta) + 1)*Math.sin(theta)"]
    assert len(r["series"][0]["grid"]) == 600
    r = plot(ev, {"kind": "parametric", "exprs": "cos(k t)", "expr2": "sin(t)"}, "k = slider(3, 1, 7, 1)")
    assert r["ok"] and r["series"][0]["params"] == ["k"]
    r = plot(ev, {"kind": "function", "exprs": "gamma(a x)"}, "a = slider(1, 0, 3)")
    assert r["ok"] and "js" not in r["series"][0]                       # no JavaScript form: server only


def test_module_plot_kinds(ev):
    kinds = {k["name"]: k for k in ev.registry.catalog()["plot_kinds"]}
    assert {"function", "contour", "surface", "bloch", "phase", "bode", "root_locus"} <= set(kinds)
    assert kinds["bloch"]["module"] == "quantum" and kinds["bloch"]["renderer"] == "plotly"
    assert any(e["kind"] == "plot" and e["category"] == "Plots" for e in ev.registry.catalog()["entries"])
    r = plot(ev, {"kind": "bloch", "exprs": "plus(), bloch_state(pi/3, pi/4)"})
    assert r["ok"] and r["series"][0]["type"] == "sphere"
    v = r["series"][1]
    assert abs(v["x"] - 1) < 1e-9 and abs(v["y"]) < 1e-9 and abs(v["z"]) < 1e-9
    assert abs(r["series"][2]["z"] - 0.5) < 1e-9
    r = plot(ev, {"kind": "bloch", "exprs": "bloch_state(theta, 0)"})
    assert not r["ok"] and "theta" in r["error"]
    r = plot(ev, {"kind": "phase", "exprs": "y, -x", "xmin": "-2", "xmax": "2", "ymin": "-2", "ymax": "2", "samples": "6"})
    assert r["ok"]
    types = [s["type"] for s in r["series"]]
    assert types == ["segments", "line", "points"] and r["series"][0]["arrows"] and len(r["series"][0]["segments"]) == 36
    assert r["series"][2]["x"] == [0.0] and r["series"][2]["y"] == [0.0]  # the equilibrium at the origin
    r = plot(ev, {"kind": "bode", "exprs": "1/(s + 1)", "xmin": "0.001", "xmax": "1000", "samples": "50"})
    assert r["ok"] and len(r["subplots"]) == 2 and r["subplots"][0]["logx"]
    mag, phase = r["subplots"][0]["series"][0]["y"], r["subplots"][1]["series"][0]["y"]
    assert abs(mag[0]) < 0.01 and mag[-1] < -70 and abs(phase[-1] + 90) < 0.1
    r = plot(ev, {"kind": "root_locus", "exprs": "(s + 3)/(s (s + 1))", "expr2": "5", "samples": "40"})
    assert r["ok"] and [s.get("marker") for s in r["series"]] == [None, "x", "o"]
    assert len(r["series"][0]["x"]) == 80 and sorted(r["series"][1]["x"]) == [-1.0, 0.0] and r["series"][2]["x"] == [-3.0]
