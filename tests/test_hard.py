"""The hard corpus: every problem must pass unless listed in KNOWN_GAPS (then it must still fail)."""
from pathlib import Path

import pytest

from bench.hard import HARD, KNOWN_GAPS
from bench.run import run_one
from quire.engine.evaluator import Evaluator
from quire.modules.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ev():
    return Evaluator(load_registry([ROOT / "modules"]))


@pytest.mark.parametrize("domain,cells,expected", HARD, ids=[" | ".join(c)[:60] for _, c, _ in HARD])
def test_hard(ev, domain, cells, expected):
    key = " | ".join(cells)
    ok, detail = run_one(ev, cells, expected, timeout=25)
    if key in KNOWN_GAPS:
        if ok:
            pytest.fail(f"now passes; remove from KNOWN_GAPS: {key}")
        pytest.xfail(KNOWN_GAPS[key])
    assert ok, f"{key} -> {detail} (expected {expected})"
