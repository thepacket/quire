"""Incremental evaluation: unchanged cells replay, changes above invalidate exactly the dependants."""
from pathlib import Path

import pytest

from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


def cells(*sources):
    return [{"id": i + 1, "type": "math", "source": s} for i, s in enumerate(sources)]


def plain(r, k=-1):
    return r["outputs"][k]["plain"]


def test_replay_and_invalidation(ev):
    cache = {}
    doc = cells("a = 2", "b = a + 1", "c + 1", "sqrt(2) a")
    first = ev.evaluate(doc, cache)
    assert all(not r.get("cached") and isinstance(r["ms"], float) for r in first)
    again = ev.evaluate(doc, cache)
    assert all(r["cached"] and r["ms"] == 0 for r in again)
    assert [plain(r) for r in again] == ["2", "3", "c + 1", "2*sqrt(2)"]
    doc[0]["source"] = "a = 3"                                   # only what depends on a re-runs
    rs = ev.evaluate(doc, cache)
    assert [bool(r.get("cached")) for r in rs] == [False, False, True, False]
    assert [plain(r) for r in rs] == ["3", "4", "c + 1", "3*sqrt(2)"]
    doc[0]["source"] = "a = 3  "                                 # same value, different text: a re-runs, b does not
    rs = ev.evaluate(doc, cache)
    assert [bool(r.get("cached")) for r in rs] == [False, True, True, True]


def test_defining_a_name_above_invalidates_its_users(ev):
    cache = {}
    doc = cells("a = 1", "c + 1")
    ev.evaluate(doc, cache)
    doc.insert(1, {"id": 9, "type": "math", "source": "c = 10"})
    rs = ev.evaluate(doc, cache)
    assert plain(rs[2]) == "11" and not rs[2].get("cached")
    del doc[1]                                                   # and removing it again
    rs = ev.evaluate(doc, cache)
    assert plain(rs[1]) == "c + 1" and 9 not in cache


def test_settings_and_assumptions_are_part_of_the_key(ev):
    cache = {}
    doc = cells("digits 3", "1/3.0", "assume x > 0", "sqrt(x^2)")
    rs = ev.evaluate(doc, cache)
    assert plain(rs[1]) == "0.333" and plain(rs[3]) == "x"
    doc[0]["source"] = "digits 5"
    rs = ev.evaluate(doc, cache)
    assert plain(rs[1]) == "0.33333" and not rs[3].get("cached")   # digits changes how every cell below shows
    doc[2]["source"] = "assume x real"
    rs = ev.evaluate(doc, cache)
    assert plain(rs[3]) == "Abs(x)" and not rs[3].get("cached")


def test_text_plots_and_sliders(ev):
    cache = {}
    doc = [{"id": 1, "type": "math", "source": "a = slider(1, 0, 3)"},
           {"id": 2, "type": "math", "source": "g(x) = a sin(x)"},
           {"id": 3, "type": "plot", "kind": "function", "exprs": "g(x)", "xmin": "0", "xmax": "6"},
           {"id": 4, "type": "text", "source": "twice {{2 a}} and plain"}]
    rs = ev.evaluate(doc, cache)
    assert rs[2]["series"][0]["params"] == ["a"] and rs[3]["values"][0]["plain"] == "2"
    rs = ev.evaluate(doc, cache)
    assert all(r["cached"] for r in rs)
    doc[0]["source"] = "a = slider(2, 0, 3)"                     # a slider move re-runs the chain, the plot included
    rs = ev.evaluate(doc, cache)
    assert [bool(r.get("cached")) for r in rs] == [False, False, False, False]
    assert rs[2]["series"][0]["js"] == ["a*Math.sin(x)"] and rs[3]["values"][0]["plain"] == "4"
    doc[2]["xmax"] = "7"                                         # a plot field change re-runs the plot only
    rs = ev.evaluate(doc, cache)
    assert [bool(r.get("cached")) for r in rs] == [True, True, False, True]


def test_imports_are_never_cached(ev):
    docs = {"k": ({"cells": [{"type": "math", "source": "k = 1"}]}, 1)}
    e = Evaluator(load_registry([ROOT / "modules"]), loader=lambda n: docs.get(n))
    cache = {}
    doc = cells("import k", "k + 1")
    assert plain(e.evaluate(doc, cache)[1]) == "2"
    docs["k"] = ({"cells": [{"type": "math", "source": "k = 5"}]}, 2)
    rs = e.evaluate(doc, cache)
    assert not rs[0].get("cached") and plain(rs[1]) == "6"


def test_worker_reports_cached_cells():
    from quire.engine.worker import EvalWorker

    w = EvalWorker([ROOT / "modules"], timeout=20)
    try:
        doc = cells("a = 2", "a^2")
        w.evaluate(doc)
        rs = w.evaluate(doc)
        assert all(r["cached"] for r in rs) and plain(rs[1]) == "4"
        doc[0]["source"] = "a = 3"
        rs = w.evaluate(doc)
        assert not rs[1].get("cached") and plain(rs[1]) == "9"
    finally:
        w.proc.kill()
