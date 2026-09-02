"""Structural, chemistry (equilibria, titration, periodic table), biomed, acoustics, earth and ml modules."""
import math
from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def run(ev, *sources):
    rs = ev.evaluate([{"id": i, "type": "math", "source": s} for i, s in enumerate(sources)])
    for r in rs:
        assert r["ok"], r["error"]
    return rs


def last(ev, *sources):
    return run(ev, *sources)[-1]["outputs"][-1]


def num(text):
    return float(text.replace("*", " ").split()[0])


def test_no_conflicts(ev):
    assert ev.registry.conflicts() == []
    assert {"structural", "biomed", "acoustics", "earth", "ml"} <= {m.name for m in ev.registry.modules if not m.error}


def test_truss_by_joints(ev):
    rs = run(ev, "T = truss([[0, 0], [4 m, 0], [2 m, 3 m]], [[1, 2], [2, 3], [1, 3]], [[1, pin], [2, roller]], [[3, 0, -10 kN]])",
             "truss_forces(T) -> kN", "truss_reactions(T)", "max_member_force(T) -> kN")
    forces = rs[1]["outputs"][0]["plain"]
    assert forces.startswith("[3.33333*kilonewton, -6.00925*kilonewton, -6.00925*kilonewton]")  # bottom chord in tension
    assert rs[2]["outputs"][0]["plain"] == "[[1, 0, 5000.0*newton], [2, 0, 5000.0*newton]]"
    assert num(rs[3]["outputs"][0]["plain"]) == pytest.approx(6.00925, rel=1e-5)
    r = ev.evaluate([{"id": 1, "type": "math", "source": "truss([[0, 0], [4 m, 0]], [[1, 2]], [[1, pin]], [[2, 0, -1 kN]])"}])[0]
    assert not r["ok"] and "determinate" in r["error"]


def test_frame_by_stiffness(ev):
    rs = run(ev, "F = frame([[0, 0], [0, 3 m], [4 m, 3 m], [4 m, 0]], [[1, 2], [2, 3], [3, 4]], [[1, fixed], [4, fixed]], [[2, 10 kN, 0, 0]], 200 GPa, 0.01 m^2, 2e-5 m^4)",
             "frame_displacements(F)", "truss_reactions(F)", "frame_end_forces(F)")
    sway = num(rs[1]["outputs"][0]["plain"].split("], [")[1])
    assert 0.003 < sway < 0.006                                             # a few mm of sway under 10 kN
    reactions = rs[2]["outputs"][0]["plain"]
    assert "[[1, " in reactions and "meter*newton" in reactions
    rx = [num(part.split(", ")[1]) for part in reactions.strip("[]").split("], [")]
    assert sum(rx) == pytest.approx(-10000, rel=1e-6)                       # horizontal equilibrium
    r = ev.evaluate([{"id": 0, "type": "math", "source": "T = truss([[0, 0], [4 m, 0], [2 m, 3 m]], [[1, 2], [2, 3], [1, 3]], [[1, pin], [2, roller]], [[3, 0, -10 kN]])"},
                     {"id": 1, "type": "plot", "kind": "structure", "exprs": "T"}])[-1]
    assert r["ok"] and [s["label_plain"] for s in r["series"]] == ["compression", "tension", "loads", "supports", "nodes"]
    assert r["equal"] and len(r["annotations"]) == 6 and r["annotations"][0]["label"] == "+3.33 kN"


def test_chemistry_equilibria_titration_elements(ev):
    rs = run(ev, "equilibrium(1.8e-5, [[0.1, -1], [0, 1], [0, 1]])", "pHc = titration(0.1, 25, 0.1, 1.8e-5)", "pHc(0)\npHc(25)\npHc(50)",
             "strong = titration(0.1, 25, 0.1)\nstrong(0)", "element(Fe)", "periodic_table(1, 3)", "equivalence_volume(0.1, 25 mL, 0.2)")
    c = rs[0]["outputs"][0]["plain"]
    assert c.startswith("[0.0986673, 0.00133267, 0.00133267]")
    ph = [num(o["plain"]) for o in rs[2]["outputs"]]
    assert ph[0] == pytest.approx(2.875, abs=0.01) and ph[1] == pytest.approx(8.72, abs=0.01) and ph[2] == pytest.approx(12.52, abs=0.01)
    assert num(rs[3]["outputs"][1]["plain"]) == pytest.approx(1.0, abs=0.01)
    assert "[number, 26]" in rs[4]["outputs"][0]["plain"] and "iron" in rs[4]["outputs"][0]["plain"]
    assert rs[5]["outputs"][0]["plain"].startswith("Matrix([[1, H, hydrogen, 1.008")
    assert rs[6]["outputs"][0]["plain"] == "12.5*milliliter"
    r = ev.evaluate([{"id": 0, "type": "math", "source": "pHc = titration(0.1, 25, 0.1, 1.8e-5)"},
                     {"id": 1, "type": "plot", "kind": "function", "exprs": "pHc(V)", "var": "V", "xmin": "0", "xmax": "50"}])[-1]
    assert r["ok"] and r["series"][0]["y"][0] < 3 and r["series"][0]["y"][-1] > 12


def test_biomed(ev):
    rs = run(ev, "pk_iv(500 mg, 40 L, 0.1/hr, 6 hr) -> mg/L", "pk_oral(500 mg, 0.8, 40 L, 1.2/hr, 0.1/hr, 3 hr) -> mg/L",
             "creatinine_clearance(65, 70 kg, 1.2)", "S = sir_model(0.5, 0.2, 990, 10, 0, 60)\ntable_column(S, 3)", "epidemic_final_size(2.5)",
             "bmi(70 kg, 1.75 m)", "body_surface_area(70 kg, 175 cm)", "logistic_growth(10, 0.5, 1000, 20)", "herd_immunity_threshold(2.5)",
             "michaelis_menten(10, 2, 2)")
    assert num(rs[0]["outputs"][0]["plain"]) == pytest.approx(12.5 * math.exp(-0.6), rel=1e-6)
    assert 7 < num(rs[1]["outputs"][0]["plain"]) < 8
    assert num(rs[2]["outputs"][0]["plain"]) == pytest.approx(60.76, abs=0.01)
    infected = rs[3]["outputs"][1]["plain"]
    assert infected.startswith("[10.0000, ") and "SIR" in rs[3]["outputs"][0]["notes"][0]
    assert num(rs[4]["outputs"][0]["plain"]) == pytest.approx(0.8926, abs=1e-3)
    assert num(rs[5]["outputs"][0]["plain"]) == pytest.approx(22.857, abs=1e-3)
    assert num(rs[6]["outputs"][0]["plain"]) == pytest.approx(1.8447, abs=1e-3)
    assert 990 < num(rs[7]["outputs"][0].get("approx_plain") or rs[7]["outputs"][0]["plain"]) < 1000
    assert rs[8]["outputs"][0]["plain"] == "0.600000000000000" or num(rs[8]["outputs"][0]["plain"]) == pytest.approx(0.6)
    assert rs[9]["outputs"][0]["plain"] == "5"


def test_acoustics(ev):
    rs = run(ev, "note_frequency(Cs4)", "note_name(450 Hz)", "cents(1, 3/2)", "scale(C4, major)", "room_modes(5 m, 4 m, 2.7 m, 4)",
             "sabine_rt60(54 m^3, 12 m^2)", "spl(1 Pa)", "spl_sum([80, 80])", "b = biquad(lowpass, 1 kHz, 44.1 kHz)",
             "dominant_frequencies(tone(440 Hz, 8 kHz, 0.5 s), 8 kHz)", "speed_of_sound(20)", "just_intonation(7)", "pythagorean(12)",
             "filter_response(biquad(lowpass, 1 kHz, 44.1 kHz), 44.1 kHz, 5)", "apply_filter(biquad(lowpass, 1 kHz, 8 kHz), [1, 0, 0, 0])")
    assert num(rs[0]["outputs"][0]["plain"]) == pytest.approx(277.183, abs=1e-3)
    assert rs[1]["outputs"][0]["plain"] == "A4" and "+38.9 cents" in rs[1]["outputs"][0]["notes"][0]
    assert num(rs[2]["outputs"][0]["plain"]) == pytest.approx(701.955, abs=1e-3)
    assert rs[3]["outputs"][0]["plain"].startswith("[261.626*hertz, 293.665*hertz") and rs[3]["outputs"][0]["plain"].count("hertz") == 8
    modes = rs[4]["outputs"][0]["plain"]
    assert modes.startswith("Matrix([[34.300, 1, 0, 0], [42.875, 0, 1, 0]")                  # axial modes of the two long dimensions
    assert num(rs[5]["outputs"][0]["plain"]) == pytest.approx(0.7245, abs=1e-4)
    assert num(rs[6]["outputs"][0]["plain"]) == pytest.approx(93.98, abs=0.01)
    assert num(rs[7]["outputs"][0]["plain"]) == pytest.approx(83.01, abs=0.01)
    assert rs[8]["outputs"][0]["plain"].startswith("[0.0050662636")
    assert rs[9]["outputs"][0]["plain"].startswith("[[440.0*hertz, 1.00000]")
    assert num(rs[10]["outputs"][0]["plain"]) == pytest.approx(343.2, abs=0.1)
    assert rs[11]["outputs"][0]["plain"] == "3/2" and rs[12]["outputs"][0]["plain"] == "531441/524288"
    resp = rs[13]["outputs"][0]["plain"]
    assert resp.startswith("Matrix([[0.0, ") and "[4410.00, -26.3" in resp                        # 1 kHz low-pass is well down at 4.4 kHz
    assert rs[14]["outputs"][0]["plain"].startswith("[0.0976")                                    # first impulse-response sample


def test_earth(ev):
    rs = run(ev, "manning_velocity(0.013, 0.5 m, 0.002)", "rational_runoff(0.6, 50 mm/hr, 20000 m^2)", "scs_runoff(80 mm, 75)",
             "isa_pressure(5000 m) -> kPa", "isa_temperature(10 km)", "isa_density(0 m)", "pressure_altitude(50 kPa) -> km",
             "wind_chill(-10, 30)", "moment_magnitude(1e20 N m)", "epicentral_distance(30 s) -> km", "seismic_energy(7)",
             "magnitude_energy_ratio(6, 7)", "relative_humidity(25, 16.7)", "critical_depth(2 m^2/s)", "return_period(4, 1, 6)")
    assert num(rs[0]["outputs"][0]["plain"]) == pytest.approx(2.167, abs=1e-3)
    assert num(rs[1]["outputs"][0]["plain"]) == pytest.approx(0.1667, abs=1e-3)
    assert num(rs[2]["outputs"][0]["plain"]) == pytest.approx(26.92, abs=0.01)
    assert num(rs[3]["outputs"][0]["plain"]) == pytest.approx(54.02, abs=0.01)
    assert num(rs[4]["outputs"][0]["plain"]) == pytest.approx(223.15, abs=0.01)
    assert num(rs[5]["outputs"][0]["plain"]) == pytest.approx(1.225, abs=1e-3)
    assert num(rs[6]["outputs"][0]["plain"]) == pytest.approx(5.57, abs=0.02)
    assert num(rs[7]["outputs"][0]["plain"]) == pytest.approx(-19.5, abs=0.1)
    assert num(rs[8]["outputs"][0]["plain"]) == pytest.approx(7.267, abs=1e-3)
    assert num(rs[9]["outputs"][0]["plain"]) == pytest.approx(252.0, abs=0.1)
    assert num(rs[10]["outputs"][0]["plain"]) == pytest.approx(1.995e15, rel=1e-3)
    assert num(rs[11]["outputs"][0]["plain"]) == pytest.approx(31.62, abs=0.01)
    assert num(rs[12]["outputs"][0]["plain"]) == pytest.approx(60, abs=0.5)
    assert num(rs[13]["outputs"][0]["plain"]) == pytest.approx(0.742, abs=1e-3)
    assert rs[14]["outputs"][0]["plain"] == "100"


def test_ml(ev):
    rs = run(ev, "w = multi_regression([[1, 2], [2, 1], [3, 4], [4, 3]], [5, 4, 11, 10])\npredict(w, [2, 2])",
             "w2 = logistic_regression([[1], [2], [3], [6], [7], [8]], [0, 0, 0, 1, 1, 1])\nlogistic_predict(w2, [4.5])\nlogistic_predict(w2, [8])",
             "C = kmeans([[1, 1], [1.2, 0.8], [0.9, 1.1], [8, 8], [8.2, 7.9], [7.8, 8.1]], 2)\nkmeans_labels([[1, 1], [8, 8]], C)",
             "pca([[1, 2], [2, 4], [3, 6.1], [4, 8]], 1)", "pca_transform([[1, 2], [2, 4], [3, 6], [4, 8]], 1)", "explained_variance([[1, 2], [2, 4], [3, 6], [4, 8]])",
             "knn_predict([[0], [1], [10], [11]], [0, 0, 1, 1], [9], 3)", "confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])", "accuracy([0, 1, 1, 0], [0, 1, 0, 0])",
             "r2_score([1, 2, 3], [1, 2, 3.1])", "rmse([1, 2], [1, 4])", "standardize([[1], [3]])")
    w = rs[0]["outputs"][0]["plain"]
    assert w.startswith("[0, 1.00000, 2.00000]") and abs(num(rs[0]["outputs"][1]["plain"]) - 6) < 1e-6   # y = x1 + 2 x2
    assert 0.4 < num(rs[1]["outputs"][1]["plain"]) < 0.6 and num(rs[1]["outputs"][2]["plain"]) > 0.9
    assert rs[2]["outputs"][0]["plain"].startswith("Matrix([[1.03333, 0.966667], [8.00000, 8.00000]])") and rs[2]["outputs"][1]["plain"] == "[1, 2]"
    assert rs[3]["outputs"][0]["notes"] == ["explained variance: 100.0%"]
    assert rs[4]["outputs"][0]["plain"].startswith("Matrix([[-3.35410], [-1.11803], [1.11803], [3.35410]])")
    assert rs[5]["outputs"][0]["plain"].startswith("[1.00000, ")
    assert rs[6]["outputs"][0]["plain"] == "1" and rs[7]["outputs"][0]["plain"] == "Matrix([[2, 0], [1, 1]])" and rs[8]["outputs"][0]["plain"] == "3/4"
    assert num(rs[9]["outputs"][0]["plain"]) == pytest.approx(0.995, abs=1e-3) and num(rs[10]["outputs"][0]["plain"]) == pytest.approx(math.sqrt(2), abs=1e-5)
    assert rs[11]["outputs"][0]["plain"] == "Matrix([[-1.00000], [1.00000]])"
    r = ev.evaluate([{"id": 1, "type": "math", "source": "kmeans([[1, 1]], 3)"}])[0]
    assert not r["ok"] and "between 1" in r["error"]
