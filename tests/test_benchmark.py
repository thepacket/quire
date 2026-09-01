"""Every corpus problem must pass; known gaps are listed in XFAIL with a reason."""
from pathlib import Path

import pytest

from bench.problems import PROBLEMS
from bench.run import run_one
from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent
XFAIL: dict[str, str] = {}


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


@pytest.mark.parametrize("domain,cells,expected", PROBLEMS, ids=[" | ".join(c)[:60] for _, c, _ in PROBLEMS])
def test_problem(ev, domain, cells, expected):
    key = " | ".join(cells)
    ok, detail = run_one(ev, cells, expected)
    if key in XFAIL:
        if ok:
            pytest.fail(f"unexpectedly passes now, remove from XFAIL: {key}")
        pytest.xfail(XFAIL[key])
    assert ok, f"{key} -> {detail} (expected {expected})"
