from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
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


def test_steps_diff(ev):
    o = out(ev, "steps_diff(x^2 sin(x), x)")
    assert o["plain"] == "x*(x*cos(x) + 2*sin(x))"
    texts = [s["text"] for s in o["steps"]]
    assert any("product rule" in t for t in texts) and any("power rule" in t for t in texts)
    assert out(ev, "d = steps_diff(x^3, x)", "d + 1")["plain"] == "3*x**2 + 1"


def test_steps_integrate(ev):
    o = out(ev, "steps_integrate(x exp(x), x)")
    assert o["plain"] == "(x - 1)*exp(x)" and any("parts" in s["text"] for s in o["steps"])
    o = out(ev, "steps_integrate(x^2 + 1/x, x, 1, 2)")
    assert o["plain"] == "log(2) + 7/3" and "evaluate from 1 to 2" in o["steps"][-1]["text"]
    o = out(ev, "steps_integrate(sin(2 x), x)")
    assert any("substitution" in s["text"] for s in o["steps"])


def test_steps_algebra(ev):
    o = out(ev, "steps_partial_fractions((3 x + 5)/(x^2 + 3 x + 2), x)")
    assert o["plain"] == "1/(x + 2) + 2/(x + 1)" and "factor the denominator" in o["steps"][0]["text"]
    o = out(ev, "steps_gauss(matrix([[2, 1], [1, 3]]), [3, 5])")
    assert o["plain"] == "Matrix([[1, 0, 4/5], [0, 1, 7/5]])" and len(o["steps"]) == 5
    o = out(ev, "steps_solve(x^2 - 5 x + 6 == 0, x)")
    assert o["plain"] == "[2, 3]" and any("discriminant" in s["text"] for s in o["steps"])
    assert out(ev, "steps_solve(2 x - 3 == 7, x)")["steps"][1]["text"].startswith("add 10")


def test_verify(ev):
    assert out(ev, "m = 2 kg", "v = 3 m/s", "dimension(1/2 m v^2)")["plain"] == "length**2*mass/time**2"
    assert out(ev, "m = 2 kg", "a = 3 m/s^2", "F = 5 N", "check_units(F == m a)")["plain"] == "True"
    assert out(ev, "m = 2 kg", "check_units(m == 3 m/s)")["plain"] == "False"
    assert out(ev, "check_identity(sin(x)^2 + cos(x)^2 == 1)")["plain"] == "True"
    assert out(ev, "check_identity((x + y)^2 == x^2 + y^2)")["plain"] == "False"
    assert "Eq(x," in out(ev, "counterexample((x + y)^2 == x^2 + y^2)")["plain"]
    o = out(ev, "propagate(V/I, [[V, 12.0, 0.1], [I, 2.0, 0.05]])")
    assert o["plain"].startswith("[6.0") and "0.158" in o["plain"]
    assert out(ev, "propagate(a b, [[a, a, s_a], [b, b, s_b]])")["plain"] == "[a*b, sqrt(a**2*s_b**2 + b**2*s_a**2)]"
    assert out(ev, "significant(pi 1000, 3)")["plain"] == "3.14e+3"
    assert out(ev, "check_ode(D(y, x, 2) + y == 0, y, x, sin(x) + cos(x))")["plain"] == "True"
    assert out(ev, "check_solution(x^2 - 5 x + 6 == 0, x, 4)")["plain"] == "False"


def test_data_files(ev):
    o = out(ev, "read_csv(measurements)")
    assert o["plain"].startswith("Matrix([[0, 20.1") and "temperature" in o["notes"][0]
    assert out(ev, "column(measurements, temperature)")["plain"] == "[20.1, 22.4, 24.9, 27.2, 29.8, 32.1]"
    assert out(ev, "table_size(measurements)")["plain"] == "[6, 3]"
    assert "measurements" in out(ev, "data_files()")["plain"]
    fit = out(ev, "linfit(column(measurements, time), column(measurements, temperature))")["plain"]
    assert fit.startswith("[2.4")
    r = ev.evaluate([{"id": 1, "type": "math", "source": "column(nonexistent, 1)"}])[0]
    assert not r["ok"] and "No data file" in r["error"]


def test_upload_names():
    from quire.server import App

    class Fake(App):
        def __init__(self, tmp):
            self.worksheets = tmp

    import base64
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        app = Fake(Path(d))
        r = app.upload("My Data (2024).csv", base64.b64encode(b"a,b\n1,2\n").decode())
        assert r["name"] == "My_Data_2024" and Path(r["path"]).read_text() == "a,b\n1,2\n"
        with pytest.raises(ValueError):
            app.upload("evil.exe", base64.b64encode(b"x").decode())


def test_tooling_no_conflicts(ev):
    assert ev.registry.conflicts() == []
