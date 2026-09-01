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
