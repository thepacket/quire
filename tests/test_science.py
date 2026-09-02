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
    """Leading number of the last output: the approximation when the exact form is not a decimal."""
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)]
    r = ev.evaluate(cells)[-1]
    assert r["ok"], r["error"]
    out = r["outputs"][-1]
    text = out.get("approx_plain") or out["plain"]
    return float(text.replace("*", " ").split()[0])


def test_physics(ev):
    assert abs(num(ev, "N(projectile_range(20 m/s, 45)) -> m") - 40.78) < 0.01
    assert abs(num(ev, "N(lorentz_factor(0.8 c_light))") - 5 / 3) < 1e-9
    assert abs(num(ev, "rest_energy(1 g) -> J") - 8.98755e13) < 1e9
    assert abs(num(ev, "N(velocity_addition(0.5 c_light, 0.5 c_light)) -> c_light") - 0.8) < 1e-9
    assert val(ev, "thin_lens_image(10 cm, 30 cm) -> cm") == "15*centimeter"
    assert abs(num(ev, "N(snell(1, 30, 1.5))") - 19.47) < 0.01
    assert abs(num(ev, "N(photon_energy(500 nm)) -> eV") - 2.4797) < 1e-3
    assert abs(num(ev, "N(wien_peak(5778 K)) -> nm") - 501.5) < 0.1
    assert abs(num(ev, "N(age_from_fraction(0.25, 5730 year)) -> year") - 11460) < 1e-6
    assert abs(num(ev, "N(binding_energy(2, 2, 4.002602 amu)) -> MeV") - 28.3) < 0.1
    r = ev.evaluate([{"id": 1, "type": "math", "source": "snell(1.5, 60, 1)"}])[0]
    assert not r["ok"] and "internal reflection" in r["error"]


def test_astronomy(ev):
    assert abs(num(ev, "N(orbital_velocity(M_earth, R_earth + 400 km)) -> km/s") - 7.67) < 0.01
    assert abs(num(ev, "N(orbital_period(M_sun, 1 AU)) -> day") - 365.25) < 0.1
    assert abs(num(ev, "N(escape_velocity(M_earth, R_earth)) -> km/s") - 11.19) < 0.01
    assert abs(num(ev, "N(schwarzschild_radius(M_sun)) -> km") - 2.953) < 0.002
    assert abs(num(ev, "N(distance_modulus(10, 5)) -> ly") - 326.2) < 0.2
    assert abs(num(ev, "N(parallax_distance(0.768)) -> ly") - 4.246) < 0.005
    assert abs(num(ev, "N(luminosity(R_sun, 5778 K)) -> L_sun") - 1.0) < 0.01
    assert val(ev, "julian_date(2000, 1, 1, 12)") == "2451545.00000"
    assert abs(num(ev, "sidereal_time(2451545, 0)") - 18.697) < 0.01
    lb = val(ev, "equatorial_to_galactic(266.405, -28.936)")  # galactic centre
    assert "degree" in lb
    h = val(ev, "h = hohmann(M_earth, R_earth + 400 km, 42164 km)", "N(h[2]) -> km/s")
    assert abs(float(h.split("*")[0]) - 3.85) < 0.05


def test_geodesy(ev):
    assert abs(num(ev, "haversine(45.5, -73.6, 51.5, -0.1) -> km") - 5217) < 10
    assert abs(num(ev, "vincenty(45.5, -73.6, 51.5, -0.1) -> km") - 5227) < 15
    assert abs(num(ev, "bearing(45.5, -73.6, 51.5, -0.1)") - 54.7) < 0.1
    z = val(ev, "utm(45.5, -73.6)")
    assert z.startswith("[18,")
    back = val(ev, "e = geodetic_to_ecef(45.5, -73.6, 100 m)", "ecef_to_geodetic(e[0], e[1], e[2])")
    assert "45.5" in back and "-73.6" in back


def test_quantum_information(ev):
    assert val(ev, "bloch_mixed(depolarizing(ket(0), 0.5))") == "[0, 0, 0.500000000000000]"
    assert val(ev, "kraus_valid([matrix([[1, 0], [0, sqrt(1 - g)]]), matrix([[0, sqrt(g)], [0, 0]])])") == "True"
    assert abs(num(ev, "concurrence(bell_state(0))") - 1) < 1e-9
    assert abs(num(ev, "concurrence(tensor(plus(), ket(0)))")) < 1e-9
    assert val(ev, "trace_distance(ket(0), ket(1))") == "1"
    assert val(ev, "teleport_check()") == "True"
    g = val(ev, "grover_iterate(3, 5)")
    assert "|101>" in g
    assert val(ev, "grover_iterations(10)") == "25"
    q = val(ev, "qft(1)")
    assert q == "Matrix([[sqrt(2)/2, sqrt(2)/2], [sqrt(2)/2, -sqrt(2)/2]])"
    assert val(ev, "amplitude_damping(ket(1), 0.3)") == "Matrix([[0.300000000000000, 0], [0, 0.700000000000000]])"


def test_science_no_conflicts(ev):
    assert ev.registry.conflicts() == []
