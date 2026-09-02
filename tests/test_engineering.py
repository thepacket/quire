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
    return float(val(ev, *sources).split("*")[0])


def test_circuits(ev):
    assert val(ev, "impedance_C(10 uF, 50 Hz) -> ohm") == "-1000*I*ohm/pi"
    assert val(ev, "parallel(1 kohm, 2 kohm) -> ohm") == "2000*ohm/3"
    assert abs(num(ev, "cutoff(4.7 kohm, 100 nF) -> Hz") - 338.628) < 1e-2
    assert val(ev, "polar_deg(phasor(230 V, 30))") == "[230*volt, 30*degree]"
    assert val(ev, "ac_power(230 V, 5 A, 30)")[:20] == "[575*sqrt(3)*ampere*"
    assert abs(num(ev, "N(resonance(10 mH, 100 nF), 6) -> kHz") - 5.0329) < 1e-3
    gain = val(ev, "H = rc_lowpass(1 kohm, 1 uF, s)", "bode_gain(H, s, 1000 Hz / (2 pi))")
    assert "log" in gain


def test_control(ev):
    assert val(ev, "tf_poles(tf([1], [1, 3, 2], s), s)") == "[-2, -1]"
    assert val(ev, "feedback(K/(s (s + 1)), 1)") == "K/(K + s**2 + s)"
    assert val(ev, "step_response(1/(s + 1), s, t)") == "1 - exp(-t)"
    assert val(ev, "is_stable(1/(s^2 + s + 1), s)") == "True"
    assert val(ev, "assume wn > 0", "damping(second_order(wn, zeta, s), s)") == "[wn, zeta]"
    routh = val(ev, "routh(s^3 + 2 s^2 + 3 s + K, s)")
    assert routh == "Matrix([[1, 3], [2, K], [3 - K/2, 0], [K, 0]])"
    assert abs(num(ev, "overshoot(0.5)") - 16.3034) < 1e-3
    assert val(ev, "controllability(matrix([[0, 1], [-2, -3]]), matrix([0, 1]))") == "True"
    assert val(ev, "state_space_tf(matrix([[0, 1], [-2, -3]]), matrix([0, 1]), matrix([[1, 0]]), matrix([0]), s)") == "Matrix([[1/(s**2 + 3*s + 2)]])"


def test_mechanics(ev):
    assert val(ev, "I_rect(50 mm, 100 mm) -> mm^4") == "12500000*millimeter**4/3"
    assert val(ev, "stress(10 kN, 200 mm^2) -> MPa") == "50*megapascal"
    assert val(ev, "euler_buckling(200 GPa, I_circle(20 mm), 2 m) -> kN") == "pi**3*kilonewton/8"
    assert val(ev, "natural_frequency(2000 N/m, 5 kg) -> Hz") == "10*hertz/pi"
    assert val(ev, "goodman(100 MPa, 50 MPa, 200 MPa, 500 MPa)") == "5/3"
    assert val(ev, "von_mises(80 MPa, -20 MPa, 30 MPa)") == "10*sqrt(111)*megapascal"
    d = val(ev, "b = beam_ss_point(10 kN, 4 m, 200 GPa, I_rect(50 mm, 100 mm), x)", "b[0]")
    assert "Piecewise" in d and "meter" in d
    mid = num(ev, "b = beam_ss_point(10 kN, 4 m, 200 GPa, I_rect(50 mm, 100 mm), x)", "subs(b[0], x, 2 m) -> mm")
    assert abs(mid - 10 * 4 ** 3 / (48 * 200e6 * (0.05 * 0.1 ** 3 / 12)) * 1000) < 1e-6


def test_thermo(ev):
    assert abs(num(ev, "fluid_density(water, 300 K, 1 atm)") - 996.557) < 1e-2
    assert abs(num(ev, "to_celsius(saturation_temperature(water, 1 atm))") - 99.974) < 1e-2
    assert abs(num(ev, "wet_bulb(from_celsius(25), 1 atm, 0.5)") - 291.033) < 1e-2
    assert val(ev, "carnot_efficiency(800 K, 300 K)") == "5/8"
    assert val(ev, "reynolds(1000 kg/m^3, 2 m/s, 50 mm, 0.001 Pa s)") == "100000.000000000"
    assert val(ev, "conduction(0.8 W/(m K), 10 m^2, 20 K, 0.2 m)") == "800.0*watt"
    rad = val(ev, "radiation(0.9, 1 m^2, 400 K, 300 K)")
    assert "stefan" not in rad and rad.endswith("*watt")
    assert abs(num(ev, "radiation(0.9, 1 m^2, 400 K, 300 K)") - 0.9 * 5.670374e-8 * (400 ** 4 - 300 ** 4)) < 1e-3
    assert abs(num(ev, "hydrostatic_pressure(1000 kg/m^3, 10 m) -> kPa") - 98.0665) < 1e-6
    assert abs(num(ev, "friction_factor(100000, 0.001)") - 0.02197) < 1e-4


def test_signals(ev):
    peak = val(ev, "x = sample_signal(sin(2 pi 50 t) + 0.5 sin(2 pi 120 t), t, 1000, 0.5)", "sp = spectrum(x, 1000)", "sp[1][25]")
    assert abs(float(peak) - 1.0) < 1e-9
    assert val(ev, "aliased_frequency(900 Hz, 1000 Hz)") == "100*hertz"
    assert val(ev, "fourier_partial_sum(t, t, 2 pi, 3)") == "2*sin(t) - sin(2*t) + 2*sin(3*t)/3"
    assert val(ev, "z_transfer([1, 1], [1, -0.5], z)") == "(z + 1)/(z - 0.5)"
    dc = float(val(ev, "ba = butter(4, 100, 1000)", "fr = freq_response(ba[0], ba[1], 1000)", "fr[1][0]"))
    assert abs(dc) < 1e-9


def test_chemistry_and_materials(ev):
    assert val(ev, "molar_mass(CaCO3)") == "100.086*gram/mole"
    assert val(ev, "balance([C3H8, O2], [CO2, H2O])") == "[1, 5, 3, 4]"
    assert val(ev, "balance([Fe, O2], [Fe2O3])") == "[4, 3, 2]"
    assert abs(num(ev, "moles(10 g, NaCl) -> mol") - 0.171116) < 1e-5
    assert val(ev, "pH(0.001)") == "3.00000"
    ratio = float(val(ev, "rate_ratio(50 kJ/mol, 298 K, 308 K)"))
    assert abs(ratio - math.exp(50000 / 8.314462618 * (1 / 298 - 1 / 308))) < 1e-4
    assert val(ev, "material_E(aluminum) -> GPa") == "69*gigapascal"
    assert val(ev, "atomic_number(Fe)") == "26"
    r = ev.evaluate([{"id": 1, "type": "math", "source": "molar_mass(Xyz)"}])[0]
    assert not r["ok"]


def test_unit_notation_from_engineering(ev):
    assert val(ev, "0.8 W/(m K)") == "0.8*watt/(kelvin*meter)"
    assert val(ev, "s^3 + 2 s^2 + 3 s") == "s**3 + 2*s**2 + 3*s"
    assert val(ev, "3 m/s^2") == "3*meter/second**2"
    assert val(ev, "0.001 Pa s") == "0.001*pascal*second"


def test_no_conflicts_with_engineering_modules(ev):
    assert ev.registry.conflicts() == []
