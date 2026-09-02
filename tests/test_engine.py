from pathlib import Path

import pytest
import sympy as sp

from quire.engine.evaluator import Evaluator
from quire.engine.parser import alias_units, rewrite_equalities, split_top
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def run(ev, *sources):
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)]
    return ev.evaluate(cells)


def last_plain(ev, *sources):
    r = run(ev, *sources)[-1]
    assert r["ok"], r["error"]
    return r["outputs"][-1]["plain"]


# --- parser helpers -------------------------------------------------------

def test_split_top_respects_brackets():
    assert split_top("f(a, b), c", ",") == ["f(a, b)", " c"]
    assert split_top("x -> m", "->") == ["x ", " m"]


def test_rewrite_equalities_nested():
    assert rewrite_equalities("solve(x^2 == 4, x)") == "solve(Eq(x^2, 4), x)"
    assert rewrite_equalities("a == b") == "Eq(a, b)"
    assert rewrite_equalities("[x == 1, y == 2]") == "[Eq(x, 1), Eq(y, 2)]"


def test_alias_units_only_after_numbers():
    units = {"m", "s", "kg"}
    assert alias_units("3 m/s^2", units) == "3 qunit_m/qunit_s^2"
    assert alias_units("2*m", units) == "2*m"
    assert alias_units("m a", units) == "m a"
    assert alias_units("2 kg m/s^2 + x", units) == "2 qunit_kg qunit_m/qunit_s^2 + x"
    assert alias_units("1.5e3 N(x)", units | {"N"}) == "1.5e3 N(x)"
    assert alias_units("1/2 g t^2", units | {"g"}) == "1/2 g t^2"
    assert alias_units("x^2 m", units) == "x^2 m"
    assert alias_units("2*3 kg", units) == "2*3 qunit_kg"


def test_user_g_survives_in_fraction(ev):
    r = run(ev, "g = 9.81 m/s^2", "y(t) = 10 m/s t - 1/2 g t^2", "y(1 s)")
    assert r[2]["ok"], r[2]["error"]
    assert r[2]["outputs"][0]["plain"] == "5.095*meter"


def test_decimal_inputs_display_numerically(ev):
    out = run(ev, "theta = 40 deg", "v = 9.81 m/s", "v sin(theta) -> m/s")[2]["outputs"][0]
    assert out["plain"] == "6.30575*meter/second" and out.get("approx") is None


# --- worksheet semantics --------------------------------------------------

def test_definitions_flow_downward(ev):
    assert last_plain(ev, "m = 2 kg", "a = 3 m/s^2", "F = m a -> N") == "6*newton"


def test_undefined_names_stay_symbolic(ev):
    assert last_plain(ev, "E = 1/2 M v^2", "diff(E, v)") == "M*v"


def test_shadowed_unit_after_number_is_still_unit(ev):
    r = run(ev, "m = 2 kg", "2 km + 3 m")
    assert r[0]["warning"] is None  # single-letter units never clash with variables
    assert r[1]["outputs"][0]["plain"] == "2003*meter"


def test_single_letter_units_are_variables_outside_unit_position(ev):
    assert last_plain(ev, "laplace(t^2, t, s)") == "2/s**3"
    assert last_plain(ev, "E = 1/2 M v^2\ndiff(E, v)") == "M*v"
    assert last_plain(ev, "5 N") == "5*newton"
    assert last_plain(ev, "3 m -> cm") == "300*centimeter"
    r = run(ev, "kg = 3")[0]
    assert r["warning"] and "kilogram" in r["warning"]


def test_assumptions(ev):
    assert last_plain(ev, "assume x > 0", "sqrt(x^2)") == "x"
    assert last_plain(ev, "assume x > 0, y != 0", "sqrt(x^2) y / y") == "x"
    assert last_plain(ev, "assume n positive integer", "integrate(x^n, x, 0, 1)") == "1/(n + 1)"
    r = run(ev, "assume x > 0", "assume x < 0")[1]
    assert not r["ok"] and "contradict" in r["error"]


def test_prime_and_sequence_notation(ev):
    assert last_plain(ev, "f(x) = x^3", "f'(2)") == "12"
    assert last_plain(ev, "f(x) = sin(x)", "f''(x)") == "-sin(x)"
    assert last_plain(ev, "rsolve(a[n+1] == 2 a[n], a, n, 1)") == "2**n"
    assert last_plain(ev, "xs = [10, 20]", "xs[1]") == "20"


def test_function_definition_and_call(ev):
    assert last_plain(ev, "f(x) = x^2 + 1", "f(3) + f(y)") == "y**2 + 11"


def test_function_body_units_checked_at_call(ev):
    r = run(ev, "v_0 = 20 m/s", "y(t) = v_0 t - 1/2 g_0 t^2", "y(1 s) -> m")
    assert all(c["ok"] for c in r), [c["error"] for c in r]
    assert r[2]["outputs"][0]["plain"] == "15.096675*meter"


def test_dimension_error_is_friendly(ev):
    r = run(ev, "2 km + 3 s")[0]
    assert not r["ok"]
    assert "Cannot add" in r["error"] and "time" in r["error"]


def test_conversion_error(ev):
    r = run(ev, "3 m -> s")[0]
    assert not r["ok"] and "Cannot convert" in r["error"]


def test_angles(ev):
    assert last_plain(ev, "sin(30 deg)") == "1/2"
    assert last_plain(ev, "asin(1/2) -> deg") == "30*degree"


def test_solve_and_calculus(ev):
    assert last_plain(ev, "solve(x^2 == 4, x)") == "[-2, 2]"
    assert last_plain(ev, "integrate(x^2, x, 0, 1)") == "1/3"
    assert last_plain(ev, "limit(sin(x)/x, x, 0)") == "1"
    assert last_plain(ev, "sum(k, k, 1, n)") == "n**2/2 + n/2"


def test_numeric_approximation(ev):
    out = run(ev, "sqrt(2) meter")[0]["outputs"][0]
    assert out["approx_plain"] == "1.41421 meter"
    assert run(ev, "3")[0]["outputs"][0].get("approx") is None


def test_uses_and_defines(ev):
    r = run(ev, "a = 1", "b = 2", "c = a + b")
    assert r[2]["defines"] == ["c"] and r[2]["uses"] == ["a", "b"]


def test_unknown_function_message(ev):
    r = run(ev, "foo(2)")[0]
    assert not r["ok"] and "not a known function" in r["error"]


def test_value_called_like_function(ev):
    r = run(ev, "x = 3", "x(2)")[1]
    assert not r["ok"] and "is a value" in r["error"]


def test_unit_function_name_collision(ev):
    # N is both newton (after a number) and the numeric-evaluation function.
    assert last_plain(ev, "5 N") == "5*newton"
    assert last_plain(ev, "N(pi, 5)") == "3.1416"


@pytest.mark.parametrize("src", ["open('x')", "x.__class__", "__import__('os')", "lambda: 1", "import os"])
def test_hostile_input_rejected(ev, src):
    r = run(ev, src)[0]
    assert not r["ok"]


def test_multi_line_cell(ev):
    r = run(ev, "a = 2\nb = a^2\na + b")[0]
    assert [o["plain"] for o in r["outputs"]] == ["2", "4", "6"]


# --- plotting -------------------------------------------------------------

def test_plot_sampling_with_units(ev):
    cells = [
        {"id": 1, "type": "math", "source": "v_0 = 20 m/s\ny(t) = v_0 t - 1/2 g_0 t^2"},
        {"id": 2, "type": "plot", "exprs": "y(t)", "xmin": "0 s", "xmax": "2 s", "samples": "9"},
    ]
    r = ev.evaluate(cells)[1]
    assert r["ok"], r["error"]
    assert r["xlabel"] == "t [s]" and r["ylabel"] == "[m]"
    assert r["series"][0]["y"][0] == 0.0 and abs(r["series"][0]["y"][-1] - (40 - 2 * 9.80665)) < 1e-9


def test_plot_infers_variable(ev):
    r = ev.evaluate([{"id": 1, "type": "plot", "exprs": "sin(t), cos(t)", "xmin": "0", "xmax": "1"}])[0]
    assert r["ok"] and r["var"] == "t" and len(r["series"]) == 2


def test_plot_reports_unknowns(ev):
    r = ev.evaluate([{"id": 1, "type": "plot", "exprs": "a x", "xmin": "0", "xmax": "1", "var": "x"}])[0]
    assert not r["ok"] and "depends on a" in r["error"]


# --- modules --------------------------------------------------------------

def test_modules_loaded(ev):
    names = {m.name for m in ev.registry.modules}
    assert {"core", "stats", "ode"} <= names
    assert all(m.error is None for m in ev.registry.modules)


def test_stats_module(ev):
    assert last_plain(ev, "mean([1, 2, 3, 4])") == "5/2"
    assert last_plain(ev, "linfit([0, 1, 2], [1, 3, 5])") == "[2, 1]"


def test_ode_module(ev):
    assert last_plain(ev, "dsolve(D(y, x) == -y, y, x, 0, 1)") == "exp(-x)"
    r = run(ev, "sol = odesolve(-y, y, x, 0, 1, 5)", "sol(1)")[1]
    assert r["ok"] and abs(float(r["outputs"][0]["plain"]) - 0.367879) < 1e-5


def test_broken_module_is_reported(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "module.py").write_text("def register(api):\n    raise RuntimeError('boom')\n")
    reg = load_registry([tmp_path])
    m = [m for m in reg.modules if m.name == "bad"][0]
    assert m.error and "boom" in m.error


def test_catalog_shape(ev):
    cat = ev.registry.catalog()
    entry = next(e for e in cat["entries"] if e["name"] == "integrate")
    assert entry["kind"] == "function" and entry["category"] == "Calculus" and entry["example"]


# --- deferred conversion, symbolic exponentials, worker ------------------

def test_function_conversion_applies_when_called(ev):
    r = run(ev, "V_s = 5 V\ntau = 0.47 s", "v_c(t) = V_s (1 - exp(-t/tau))",
            "i(t) = diff(v_c(t), t) 100 uF -> mA", "i(1 s)")
    assert all(c["ok"] for c in r), [c["error"] for c in r]
    assert r[3]["outputs"][0]["plain"].endswith("*milliampere")


def test_symbolic_exponential_with_units_is_left_alone(ev):
    r = run(ev, "tau = 0.47 s", "v = 5 V (1 - exp(-t/tau))")[1]
    assert r["ok"], r["error"]
    assert "exp" in r["outputs"][0]["plain"]


def test_dsolve_with_units(ev):
    r = run(ev, "V_s = 5 V\ntau = 0.47 s", "dsolve(D(v, t) == -v/tau + V_s/tau, v, t, 0, 0)")[1]
    assert r["ok"], r["error"]


def test_worker_timeout_and_recovery():
    from quire.engine.worker import EvalWorker

    w = EvalWorker([ROOT / "modules"], timeout=2)
    try:
        assert len(w.catalog()["entries"]) > 100
        res = w.evaluate([
            {"id": 1, "type": "math", "source": "a = 1"},
            {"id": 2, "type": "math", "source": "N(pi, 10000000)"},
            {"id": 3, "type": "math", "source": "a + 1"},
        ])
        assert res[0]["ok"] and not res[1]["ok"] and "stopped" in res[1]["error"]
        assert "Not evaluated" in res[2]["error"]
        again = w.evaluate([{"id": 1, "type": "math", "source": "2 + 2"}])
        assert again[0]["outputs"][0]["plain"] == "4"
    finally:
        w.proc.kill()


def test_solve_strips_units_inside_exponentials(ev):
    r = run(ev, "V_s = 5 V\ntau = 0.47 s", "v_c(t) = V_s (1 - exp(-t/tau))", "solve(v_c(t) == 0.95 V_s, t)")[2]
    assert r["ok"], r["error"]
    assert r["outputs"][0]["plain"] == "[1.40799416857038*second]"


def test_solve_keeps_units_when_exact(ev):
    assert last_plain(ev, "solve(x^2 == 4 m^2, x)") == "[-2*meter, 2*meter]"


def test_solve_without_exact_solution_points_to_nsolve(ev):
    r = run(ev, "solve(cos(x) == x, x)")[0]
    assert not r["ok"] and "nsolve" in r["error"]


def test_bare_unit_shows_the_one(ev):
    out = run(ev, "R = 10 kohm\nC = 100 uF", "tau = R C -> s")[1]["outputs"][0]
    assert out["latex"].startswith("1\\,")


def test_plot_labels_prefer_named_si_units(ev):
    from quire.engine import units as U
    assert U.unit_label(3 * U.u.volt) == "V"
    assert U.unit_label(2 * U.mA) == "A"
    assert U.unit_label(U.u.km / U.u.hour) == "m/s"
    assert U.unit_label(U.u.kg * U.u.m**2 / U.u.s**2) == "J"


def test_bound_visible_in_same_cell(ev):
    r = run(ev, "assume s > 1\nintegrate(x^(s - 1)/(exp(x) - 1), x, 0, oo)")[0]
    assert r["ok"] and r["outputs"][1]["plain"] == "gamma(s)*zeta(s)"


def test_recognized_integral_carries_note(ev):
    r = run(ev, "integrate(ln(1 + x^2)/(1 + x^2), x, 0, oo)")[0]
    assert r["ok"] and r["outputs"][0]["plain"] == "pi*log(2)" and "recognized" in r["outputs"][0]["notes"][0]
    assert last_plain(ev, "recognize(0.2722)") == "0.272200000000000"


def test_no_module_name_conflicts(ev):
    assert ev.registry.conflicts() == []


def test_password_guard():
    from quire.server import App

    class Fake(App):  # no worker needed to test the auth check
        def __init__(self, password):
            self.password = password

    import base64

    assert Fake(None).authorized(None)
    a = Fake("s3cret")
    assert not a.authorized(None)
    assert not a.authorized("Basic " + base64.b64encode(b"user:wrong").decode())
    assert a.authorized("Basic " + base64.b64encode(b"anyone:s3cret").decode())


def test_digits_setting(ev):
    r = run(ev, "sqrt(2)", "digits 12", "sqrt(2)", "pi 1.0", "N(pi, 20)", "nsolve(cos(x) == x, x, 1)")
    assert r[0]["outputs"][0]["approx_plain"] == "1.41421"
    assert r[1]["outputs"][0]["plain"] == "digits 12"
    assert r[2]["outputs"][0]["approx_plain"] == "1.41421356237"
    assert r[3]["outputs"][0]["plain"] == "3.14159265359"
    assert r[4]["outputs"][0]["plain"] == "3.14159265359"   # display precision governs
    assert r[5]["outputs"][0]["plain"] == "0.739085133215"
    bad = run(ev, "digits 100")[0]
    assert not bad["ok"] and "between" in bad["error"]
