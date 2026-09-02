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


def test_geometry(ev):
    assert val(ev, "T1 = triangle(point(0, 0), point(4, 0), point(0, 3))", "area(T1)") == "6"
    assert val(ev, "T1 = triangle(point(0, 0), point(4, 0), point(0, 3))", "perimeter(T1)") == "12"
    assert val(ev, "intersect(circle(point(0, 0), 1), line(point(0, 0), point(1, 1)))") == "[Point2D(-sqrt(2)/2, -sqrt(2)/2), Point2D(sqrt(2)/2, sqrt(2)/2)]"
    assert val(ev, "distance(point(0, 0), line(point(0, 1), point(1, 0)))") == "sqrt(2)/2"
    assert val(ev, "equation_of(circle(point(1, 0), 2), x, y)") == "Eq(y**2 + (x - 1)**2 - 4, 0)"
    tri = val(ev, "triangle_from_sides(3, 4, 5)")
    assert tri.startswith("[36.869897") and "90*degree, 6]" in tri
    assert val(ev, "rotate(point(1, 0), 90)") == "Point2D(0, 1)"
    assert val(ev, "angle_between(line(point(0, 0), point(1, 0)), line(point(0, 0), point(1, 1)))") == "45*degree"


def test_shapes_plot(ev):
    r = ev.evaluate([{"id": 1, "type": "plot", "kind": "shapes",
                      "exprs": "circle(point(0, 0), 2), triangle(point(-1, -1), point(2, 0), point(0, 1)), point(1, 1)"}])[0]
    assert r["ok"] and r["equal"] and [s["type"] for s in r["series"]] == ["line", "line", "points"]


def test_optimization(ev):
    cp = val(ev, "critical_points(x^3 - 3 x + y^2, [x, y])")
    assert "min" in cp and "saddle" in cp
    assert val(ev, "lagrange(x y, [x + y == 10], [x, y])") == "Matrix([[5, 5, 5, 25]])"
    m = val(ev, "minimize((x - 1)^2 + (y - 2)^2, [x, y])")
    assert m.startswith("[1.0") or m.startswith("[0.99999")
    lp = val(ev, "linprog([-1, -2], [[1, 1], [1, 3]], [4, 6])")
    assert lp.startswith("[3.0") and "-5.0" in lp
    fit = val(ev, "curve_fit(a exp(-b x), [a, b], x, [0, 1, 2, 3], [2, 1.2, 0.7, 0.45])")
    assert fit.startswith("[2.0") or fit.startswith("[1.9")


def test_discrete(ev):
    assert val(ev, "combinations_of([a, b, c, d], 2)") == "[[a, b], [a, c], [a, d], [b, c], [b, d], [c, d]]"
    assert val(ev, "derangements(4)") == "9"
    assert val(ev, "series_coefficients(1/(1 - x - x^2), x, 8)") == "[1, 1, 2, 3, 5, 8, 13, 21]"
    assert val(ev, "generating_function(1, k, x)").startswith("Piecewise") or val(ev, "generating_function(1, k, x)") == "1/(1 - x)"
    assert val(ev, "shortest_path([[A, B, 4], [A, C, 2], [C, B, 1], [B, D, 5]], A, D)") == "[8, [A, C, B, D]]"
    assert val(ev, "minimum_spanning_tree([[A, B, 4], [A, C, 2], [C, B, 1], [B, D, 5]])").startswith("[8,")
    assert val(ev, "truth_table(IMPLIES(p, q), [p, q])") == "Matrix([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]])"
    assert val(ev, "is_tautology(OR(p, NOT(p)))") == "True"
    assert val(ev, "cardinality(powerset_of([1, 2, 3]))") == "8"


def test_crypto(ev):
    assert val(ev, "continued_fraction(sqrt(2), 6)") == "[1, 2, 2, 2, 2, 2]"
    assert val(ev, "convergents(pi, 4)") == "[3, 22/7, 333/106, 355/113]"
    assert val(ev, "pell(61)") == "[1766319049, 226153980]"
    assert val(ev, "crt([2, 3, 2], [3, 5, 7])") == "[23, 105]"
    assert val(ev, "discrete_log(23, 5, 8)") == "6"
    assert val(ev, "rsa_keygen(61, 53, 17)") == "[3233, 17, 2753]"
    assert val(ev, "rsa_decrypt(rsa_encrypt(65, 17, 3233), 2753, 3233)") == "65"
    assert val(ev, "diffie_hellman(23, 5, 6, 15)") == "[8, 19, 2]"
    assert val(ev, "ec_add([5, 1], [5, 1], 2, 2, 17)") == "[6, 3]"
    assert val(ev, "ec_order([5, 1], 2, 2, 17)") == "19"
    assert val(ev, "ec_multiply(19, [5, 1], 2, 2, 17)") == "O"


def test_tensors(ev):
    assert val(ev, "gaussian_curvature(metric_sphere(R, theta, phi), [theta, phi])") == "R**(-2)"
    assert val(ev, "ricci(metric_schwarzschild(M, t, r, theta, phi), [t, r, theta, phi])") == "Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])"
    assert val(ev, "christoffel(metric_polar(r, theta), [r, theta])[1]") == "Matrix([[0, 1/r], [1/r, 0]])"
    assert val(ev, "ricci_scalar(metric_polar(r, theta), [r, theta])") == "0"


def test_math_modules_no_conflicts(ev):
    assert ev.registry.conflicts() == []
