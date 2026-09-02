"""Run the symbolic coverage corpus through the worksheet pipeline and report by domain.

    .venv/bin/python -m bench.run            summary + failures
    .venv/bin/python -m bench.run -v         every problem
"""
from __future__ import annotations

import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import sympy as sp

from quire.engine.evaluator import DefinedFunction, Evaluator
from quire.modules.registry import load_registry

from .problems import PROBLEMS
from .hard import HARD

ROOT = Path(__file__).resolve().parent.parent


class Timeout(BaseException):  # not Exception: must not be swallowed by the engine's error handling
    pass


def same(a, b) -> bool:
    if isinstance(a, DefinedFunction):
        a = a.expr
    if isinstance(b, DefinedFunction):
        b = b.expr
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        rest = list(b)
        for x in a:
            for j, y in enumerate(rest):
                if same(x, y):
                    del rest[j]
                    break
            else:
                return False
        return True
    if isinstance(a, sp.MatrixBase) and isinstance(b, sp.MatrixBase):
        return a.shape == b.shape and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, sp.Eq) and isinstance(b, sp.Eq):
        return same(a.lhs, b.lhs) and same(a.rhs, b.rhs)
    if isinstance(a, sp.Basic) and isinstance(b, sp.Basic):
        if a == b:
            return True
        if isinstance(a, sp.Set) or isinstance(b, sp.Set):
            return a == b
        try:
            if a.is_number and b.is_number and abs(complex(a) - complex(b)) < 1e-12:
                return True
            if sp.simplify(a - b) == 0:
                return True
            return bool(a.equals(b))
        except Exception:  # noqa: BLE001
            return False
    return a == b


def run_one(ev: Evaluator, cells, expected, timeout=20):
    """Return (passed, detail)."""
    env: dict = {}
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))
    signal.alarm(timeout)
    try:
        res = None
        for src in cells:
            res = ev.evaluate_math(src, env)
            if not res["ok"]:
                break
        if isinstance(expected, dict):
            if "error" in expected:
                ok = not res["ok"] and expected["error"] in (res["error"] or "")
                return ok, res["error"] or res["outputs"][-1]["plain"]
            if not res["ok"]:
                return False, "ERROR " + res["error"]
            plain = res["outputs"][-1]["plain"]
            if "contains" in expected:
                return expected["contains"] in plain, plain
            return True, plain
        if not res["ok"]:
            return False, "ERROR " + res["error"]
        got = ev.last_values[-1]
        got_plain = res["outputs"][-1]["plain"]
        exp_res = ev.evaluate_math(expected, env)
        if not exp_res["ok"]:
            return False, f"expected side failed: {exp_res['error']}"
        want = ev.last_values[-1]
        return same(got, want), got_plain
    except Timeout:
        return False, "TIMEOUT"
    except Exception as exc:  # noqa: BLE001
        return False, f"CRASH {type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)


def main(argv):
    verbose = "-v" in argv
    problems = HARD if "--hard" in argv else PROBLEMS
    ev = Evaluator(load_registry([ROOT / "modules"]))
    totals = defaultdict(lambda: [0, 0])
    failures = []
    t0 = time.time()
    for domain, cells, expected in problems:
        ok, detail = run_one(ev, cells, expected)
        totals[domain][1] += 1
        totals[domain][0] += ok
        if verbose or not ok:
            mark = "ok " if ok else "FAIL"
            line = f"{mark} [{domain}] {' | '.join(cells)}  ->  {detail[:110]}"
            if not ok:
                line += f"   (expected {expected})"
                failures.append(line)
            elif verbose:
                print(line)
    print()
    print(f"{'domain':26s} {'pass':>5s} {'total':>6s}")
    for d, (ok, n) in totals.items():
        print(f"{d:26s} {ok:5d} {n:6d}")
    total_ok = sum(v[0] for v in totals.values())
    total = sum(v[1] for v in totals.values())
    print(f"{'ALL':26s} {total_ok:5d} {total:6d}   ({100 * total_ok / total:.0f}%, {time.time() - t0:.1f}s)")
    if failures:
        print("\nFailures:")
        print("\n".join(failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
