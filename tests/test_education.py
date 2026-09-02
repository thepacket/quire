"""Education: graded check cells, hidden references on share links, step-by-step derivations."""
from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator, grade
from quire.modules.registry import load_registry
from quire.server import App

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def check(ev, reference, answer, *pre, hint="try again"):
    cells = [{"id": i, "type": "math", "source": s} for i, s in enumerate(pre)]
    cells.append({"id": 99, "type": "check", "prompt": "q", "reference": reference, "answer": answer, "hint": hint})
    return ev.evaluate(cells)[-1]


def test_grading_by_identity(ev):
    assert check(ev, "x^2 - 1", "(x - 1)(x + 1)")["correct"] is True
    r = check(ev, "x^2 - 1", "x^2 + 1")
    assert r["correct"] is False and r["hint"] == "try again" and "Not the expected" in r["feedback"]
    assert check(ev, "sin(x)^2", "1 - cos(x)^2")["correct"] is True
    assert check(ev, "solve(x^2 == 4, x)", "[2, -2]")["correct"] is True
    assert check(ev, "solve(x^2 == 4, x)", "[2]")["feedback"] == "Expected 2 values, got 1."
    assert check(ev, "3 m/s", "10.8 km/hr")["correct"] is True
    assert "dimension length/time" in check(ev, "3 m/s", "3 kg")["feedback"]
    assert check(ev, "2 x", "2 x + y")["feedback"] == "Your answer still contains y."
    assert check(ev, "pi/4", "0.785")["correct"] is True                      # rounding tolerance for decimals
    assert "Close" in check(ev, "pi/4", "0.8")["feedback"]
    assert check(ev, "pi/4", "1")["feedback"] == "That value is not right."
    assert check(ev, "F", "20 N", "m = 2 kg\na = 10 m/s^2\nF = m a")["correct"] is True   # definitions above apply
    assert check(ev, "x^2 == 1", "2 x^2 == 2")["correct"] is True
    assert check(ev, "x^2 == 1", "x^2 == 2")["correct"] is False
    assert check(ev, "matrix([[1, 0], [0, 1]])", "matrix([[2, 0], [0, 2]])/2")["correct"] is True
    assert check(ev, "true", "1 == 1")["correct"] is True or check(ev, "1", "1")["correct"] is True


def test_check_cell_states(ev):
    r = check(ev, "", "1")
    assert r["ok"] and r["correct"] is None and "reference" in r["feedback"]
    r = check(ev, "2", "")
    assert r["ok"] and r["correct"] is None and r["feedback"] == ""
    r = check(ev, "1/0)", "1")
    assert not r["ok"] and "reference" in r["error"]
    r = check(ev, "2", "?")
    assert r["ok"] and r["correct"] is False and "could not read" in r["feedback"]
    r = check(ev, "2", "2", hint="")
    assert "hint" not in r and r["answer_latex"] == "2"
    # a check never defines anything and the reference stays private to the result
    rs = ev.evaluate([{"id": 1, "type": "check", "prompt": "", "reference": "q = 5", "answer": "q = 5"}, {"id": 2, "type": "math", "source": "q"}])
    assert rs[1]["outputs"][0]["plain"] == "q" and "reference" not in rs[0]


def test_grade_function_directly():
    import sympy as sp

    assert grade([1, 2], sp.FiniteSet(2, 1)) == (True, "Correct.")
    assert grade(sp.Integer(1), [1])[0] is False
    assert grade(sp.Matrix([[1, 2]]), sp.Matrix([[1], [2]]))[1] == "Expected a 2×1 matrix."


def test_share_link_hides_the_reference(tmp_path):
    app = App.__new__(App)
    app.worksheets, app.examples, app.mirror, app.mirror_error = tmp_path / "ws", ROOT / "examples", None, None
    app.evaluate = lambda cells: {"results": [dict(c, reference=c.get("reference")) for c in cells]}
    app.save("lesson", {"quire": 1, "title": "L", "cells": [{"id": "c", "type": "check", "prompt": "p", "reference": "x^2", "hint": "h", "answer": ""}]})
    token = app.share("lesson")["token"]
    shared = app.shared_doc(token)["cells"][0]
    assert shared["reference"] == "" and shared["hint"] == "" and shared["locked"] and shared["has_hint"]
    results = app.shared_evaluate(token, {"answers": {"c": "x*x", "zzz": "ignored"}})["results"]
    assert results[0]["answer"] == "x*x" and "reference" not in results[0]      # graded on the server, never echoed


def test_more_step_by_step(ev):
    def steps(src):
        r = ev.evaluate([{"id": 1, "type": "math", "source": src}])[0]
        assert r["ok"], r["error"]
        o = r["outputs"][0]
        return o["plain"], [s["text"] for s in o["steps"]]
    plain, texts = steps("steps_limit((x^2 - 1)/(x - 1), x, 1)")
    assert plain == "2" and texts[0].startswith("0/0: factor")
    plain, texts = steps("steps_limit(sin(x)/x, x, 0)")
    assert plain == "1" and "L'Hôpital" in texts[0]
    plain, texts = steps("steps_limit((3 x^2 + 1)/(x^2 - 2), x, oo)")
    assert plain == "3" and "leading terms" in texts[0]
    plain, texts = steps("steps_series(exp(x), x, 0, 4)")
    assert plain == "x**3/6 + x**2/2 + x + 1" and len(texts) == 6 and texts[2] == "k = 1: f'(0) = 1"
    plain, texts = steps("steps_system([2 x + y == 5, x - y == 1], [x, y])")
    assert plain == "[Eq(x, 2), Eq(y, 1)]" and texts[0].startswith("write the augmented") and texts[-1] == "read off the solution"
    plain, texts = steps("steps_system([x^2 + y == 5, x - y == 1], [x, y])")
    assert "Eq(x, -3), Eq(y, -4)" in plain and "Eq(x, 2), Eq(y, 1)" in plain and texts[0].startswith("solve the equation for")
    plain, texts = steps("steps_inverse(matrix([[2, 1], [1, 1]]))")
    assert plain == "Matrix([[1, -1], [-1, 2]])" and texts[0].startswith("determinant") and texts[-1] == "the right block is A^-1"
    plain, texts = steps("steps_inverse(matrix([[1, 2], [2, 4]]))")
    assert plain == "no_inverse"
    plain, texts = steps("steps_separable(x y, y, x)")
    assert plain == "Eq(y, exp(C + x**2/2))" and texts[2].startswith("separate the variables")
    plain, texts = steps("steps_separable(y^2, y, x)")
    assert plain == "Eq(y, -1/(C + x))"
    r = ev.evaluate([{"id": 1, "type": "math", "source": "steps_separable(x + y, y, x)"}])[0]
    assert not r["ok"] and "separate" in r["error"]
