import sympy as sp

from ...engine.errors import EvalError


def sym(x, what="a variable name"):
    if not isinstance(x, sp.Symbol):
        raise EvalError(f"Expected {what}, got '{x}'.")
    return x


def as_residual(e):
    e = sp.sympify(e)
    return e.lhs - e.rhs if isinstance(e, sp.Eq) else e


def as_list(x):
    if isinstance(x, sp.MatrixBase):
        return list(x)
    if isinstance(x, (list, tuple, set, frozenset, sp.FiniteSet)):
        return list(x)
    raise EvalError("Expected a list, e.g. [1, 2, 3].")


def matrix(A):
    return A if isinstance(A, sp.MatrixBase) else sp.ImmutableMatrix(A)


def pretty_solutions(res, syms):
    """Turn sympy's dict / tuple solutions into lists of equations: [x = 2, y = 1]."""
    if isinstance(res, dict):
        return [sp.Eq(k, v, evaluate=False) for k, v in res.items()]
    if isinstance(res, list) and res and isinstance(res[0], dict):
        return [pretty_solutions(d, syms) for d in res]
    if isinstance(res, list) and res and isinstance(res[0], (tuple, sp.Tuple)) and len(syms) > 1:
        return [[sp.Eq(s, v, evaluate=False) for s, v in zip(syms, t)] for t in res]
    if res is sp.S.EmptySet:
        return []
    if isinstance(res, sp.FiniteSet):
        items = list(res)
        if items and isinstance(items[0], (tuple, sp.Tuple)) and len(syms) > 1:
            return [[sp.Eq(s, v, evaluate=False) for s, v in zip(syms, t)] for t in items]
        return items
    return res


def _bound_samples(name: str, bounds: dict):
    """Sample values satisfying the worksheet bounds for a symbol (assume x > 1, x < 3, ...)."""
    lo, hi = None, None
    for op, val in bounds.get(name, []):
        v = sp.nsimplify(val) if not isinstance(val, sp.Basic) else val
        if op in (">", ">="):
            lo = v if lo is None else sp.Max(lo, v)
        elif op in ("<", "<="):
            hi = v if hi is None else sp.Min(hi, v)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None:
        return [lo + (hi - lo) * f for f in (sp.Rational(1, 7), sp.Rational(1, 2), sp.Rational(6, 7))]
    if lo is not None:
        return [lo + sp.Rational(1, 3), lo + 2, lo + 50]
    return [hi - sp.Rational(1, 3), hi - 2, hi - 50]


def prune_piecewise(res, bounds: dict):
    """Decide piecewise conditions using the worksheet's numeric bounds on symbols.

    A condition that is False at every sample point inside the declared region is
    dropped; one that is True at every sample point ends the piecewise there. If no
    bounded symbol appears, the result is returned unchanged.
    """
    if not isinstance(res, sp.Piecewise) or not bounds:
        return res
    names = {s.name for s in res.free_symbols}
    if not names & set(bounds):
        return res
    samples = {}
    for s in res.free_symbols:
        pts = _bound_samples(s.name, bounds)
        if pts:
            samples[s] = pts
    if not samples:
        return res
    kept = []
    for expr, cond in res.args:
        if cond == sp.true:
            kept.append((expr, cond))
            break
        verdicts = set()
        for i in range(3):
            point = {s: pts[i] for s, pts in samples.items()}
            try:
                v = cond.subs(point)
                verdicts.add(bool(v) if v in (sp.true, sp.false) else None)
            except (TypeError, ValueError):
                verdicts.add(None)
        if verdicts == {False}:
            continue
        if verdicts == {True}:
            kept.append((expr, sp.true))
            break
        kept.append((expr, cond))
    if len(kept) == 1:
        return kept[0][0]
    return sp.Piecewise(*kept) if kept else res


class Budget(BaseException):
    """Raised inside with_budget when the time budget is exhausted (BaseException: not swallowed by handlers)."""


def with_budget(seconds: float, fn, *args, **kwargs):
    """Run fn with a SIGALRM time budget. Returns (result, timed_out).

    Only the main thread can use signals; elsewhere fn runs unbounded. An outer
    alarm (e.g. the benchmark runner's) is saved and re-armed afterwards.
    """
    import signal
    import threading
    import time

    if threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs), False
    old_handler = signal.getsignal(signal.SIGALRM)
    remaining = signal.alarm(0)
    start = time.monotonic()

    def handler(*_):
        raise Budget()

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args, **kwargs), False
    except Budget:
        return None, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if remaining:
            left = max(1, int(remaining - (time.monotonic() - start)))
            signal.alarm(left)
