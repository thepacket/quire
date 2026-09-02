"""Discrete mathematics: sets, counting, graphs, propositional logic, generating functions.

Logic is written with functions because the worksheet syntax has no &, | or ~:
    truth_table(IMPLIES(p, q), [p, q])   is_tautology(OR(p, NOT(p)))
Graphs are edge lists [[a, b, weight], ...] with node names as symbols or numbers.
"""
import itertools

import numpy as np
import sympy as sp
from sympy.logic import boolalg as B

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "discrete"
DESCRIPTION = "Sets, permutations and combinations, graphs, logic and truth tables, generating functions."


def _note(t):
    hooks.context.setdefault("notes", []).append(t)


def _items(xs):
    if isinstance(xs, sp.FiniteSet):
        return list(xs)
    if isinstance(xs, sp.MatrixBase):
        return list(xs)
    if isinstance(xs, (list, tuple)):
        return list(xs)
    raise EvalError("Expected a list or a set.")


# ---- sets
def set_of(*xs):
    xs = xs[0] if len(xs) == 1 and isinstance(xs[0], (list, tuple)) else xs
    return sp.FiniteSet(*[sp.sympify(x) for x in xs])


def union_of(*sets):
    return sp.Union(*[set_of(s) if not isinstance(s, sp.Set) else s for s in sets])


def intersect_of(*sets):
    return sp.Intersection(*[set_of(s) if not isinstance(s, sp.Set) else s for s in sets])


def difference_of(a, b):
    return sp.Complement(set_of(a) if not isinstance(a, sp.Set) else a, set_of(b) if not isinstance(b, sp.Set) else b)


def symmetric_difference(a, b):
    return sp.SymmetricDifference(set_of(a) if not isinstance(a, sp.Set) else a, set_of(b) if not isinstance(b, sp.Set) else b)


def powerset_of(xs):
    return set_of(xs).powerset()


def cartesian(a, b):
    return [[x, y] for x in _items(a) for y in _items(b)]


def is_subset(a, b):
    return set_of(a).is_subset(set_of(b))


def cardinality(s):
    return sp.Integer(len(_items(s)))


# ---- counting
def permutations_count(n, k=None):
    n = sp.sympify(n)
    k = n if k is None else sp.sympify(k)
    return sp.simplify(sp.factorial(n) / sp.factorial(n - k))


def combinations_count(n, k):
    return sp.binomial(sp.sympify(n), sp.sympify(k))


def derangements(n):
    n = int(n)
    return sp.Integer(round(sp.factorial(n) * sum(sp.Rational((-1) ** k, sp.factorial(k)) for k in range(n + 1))))


def permutations_of(xs, k=None, limit=120):
    items = _items(xs)
    k = len(items) if k is None else int(k)
    out = [list(p) for p in itertools.islice(itertools.permutations(items, k), int(limit) + 1)]
    if len(out) > int(limit):
        raise EvalError(f"More than {int(limit)} permutations; use permutations_count.")
    return out


def combinations_of(xs, k, limit=120):
    items = _items(xs)
    out = [list(c) for c in itertools.islice(itertools.combinations(items, int(k)), int(limit) + 1)]
    if len(out) > int(limit):
        raise EvalError(f"More than {int(limit)} combinations; use combinations_count.")
    return out


def multiset_count(n, k):
    """Combinations with repetition C(n + k - 1, k)."""
    return sp.binomial(sp.sympify(n) + sp.sympify(k) - 1, sp.sympify(k))


def pigeonhole(items, boxes):
    """Smallest number guaranteed in some box: ceil(items/boxes)."""
    return sp.ceiling(sp.sympify(items) / sp.sympify(boxes))


# ---- generating functions and sequences
def generating_function(a_k, k, x):
    """sum a_k x^k for k >= 0 in closed form (when sympy finds one)."""
    return sp.simplify(sp.summation(sp.sympify(a_k) * x ** k, (k, 0, sp.oo)))


def series_coefficients(f, x, n):
    """First n coefficients of the power series of f about 0."""
    ser = sp.series(sp.sympify(f), x, 0, int(n)).removeO()
    return [ser.coeff(x, k) for k in range(int(n))]


def sequence_terms(a_k, k, n):
    return [sp.sympify(a_k).subs(k, i) for i in range(int(n))]


# ---- graphs
def _node(v):
    return sp.Symbol(v.__name__) if callable(v) and not isinstance(v, sp.Basic) else v


def _graph(edges, directed=False):
    nodes = []
    edges = [[_node(x) for x in list(e)[:2]] + list(e)[2:] for e in edges]
    for e in edges:
        for v in list(e)[:2]:
            if v not in nodes:
                nodes.append(v)
    idx = {v: i for i, v in enumerate(nodes)}
    M = np.zeros((len(nodes), len(nodes)))
    for e in edges:
        e = list(e)
        a, b = idx[e[0]], idx[e[1]]
        w = float(e[2]) if len(e) > 2 else 1.0
        M[a, b] = w
        if not directed:
            M[b, a] = w
    return nodes, idx, M


def adjacency(edges, directed=False):
    nodes, _, M = _graph(edges, directed)
    _note("rows and columns in order: " + ", ".join(str(n) for n in nodes))
    return sp.ImmutableMatrix([[sp.nsimplify(v) for v in row] for row in M])


def shortest_path(edges, source, target, directed=False):
    """[distance, path] by Dijkstra."""
    from scipy.sparse.csgraph import dijkstra

    nodes, idx, M = _graph(edges, directed)
    source, target = _node(source), _node(target)
    if source not in idx or target not in idx:
        raise EvalError("source or target is not a node of the graph.")
    dist, pred = dijkstra(M, directed=directed, indices=idx[source], return_predecessors=True)
    d = dist[idx[target]]
    if not np.isfinite(d):
        raise EvalError("No path between the nodes.")
    path, cur = [], idx[target]
    while cur != -9999 and cur != idx[source]:
        path.append(nodes[cur])
        cur = pred[cur]
    path.append(source)
    return [sp.nsimplify(float(d)), list(reversed(path))]


def minimum_spanning_tree(edges):
    """[total weight, [[a, b, w], ...]] (Kruskal via scipy)."""
    from scipy.sparse.csgraph import minimum_spanning_tree as mst

    nodes, _, M = _graph(edges)
    T = mst(M).toarray()
    out, total = [], 0.0
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if T[i, j]:
                out.append([nodes[i], nodes[j], sp.nsimplify(T[i, j])])
                total += T[i, j]
    return [sp.nsimplify(total), out]


def is_connected(edges):
    from scipy.sparse.csgraph import connected_components

    _, _, M = _graph(edges)
    return int(connected_components(M, directed=False)[0]) == 1


def degrees(edges):
    nodes, idx, M = _graph(edges)
    return sp.ImmutableMatrix([[n, sp.Integer(int(np.count_nonzero(M[idx[n]])))] for n in nodes])


def has_euler_circuit(edges):
    nodes, idx, M = _graph(edges)
    return is_connected(edges) and all(int(np.count_nonzero(M[idx[n]])) % 2 == 0 for n in nodes)


# ---- logic
def AND(*args):
    return B.And(*args)


def OR(*args):
    return B.Or(*args)


def NOT(a):
    return B.Not(a)


def XOR(a, b):
    return B.Xor(a, b)


def IMPLIES(a, b):
    return B.Implies(a, b)


def IFF(a, b):
    return B.Equivalent(a, b)


def truth_table(expr, vars_):
    """Rows: values of the variables (0/1) followed by the expression's value."""
    vs = list(vars_)
    rows = []
    for values in itertools.product([False, True], repeat=len(vs)):
        val = expr.subs(dict(zip(vs, values)))
        rows.append([sp.Integer(int(bool(v))) for v in values] + [sp.Integer(int(bool(val)))])
    _note("columns: " + ", ".join(str(v) for v in vs) + ", result")
    return sp.ImmutableMatrix(rows)


def is_tautology(expr):
    return B.simplify_logic(expr) == B.true


def is_satisfiable(expr):
    return bool(B.satisfiable(expr))


def simplify_logic(expr):
    return B.simplify_logic(expr)


def to_cnf(expr):
    return B.to_cnf(expr, simplify=True)


def to_dnf(expr):
    return B.to_dnf(expr, simplify=True)


def register(api):
    S = "Discrete: sets & counting"
    api.function("set_of", set_of, signature="set_of([a, b, c])", doc="a finite set", category=S, example="set_of([1, 2, 3])")
    api.function("union_of", union_of, signature="union_of(A, B)", doc="A ∪ B", category=S)
    api.function("intersect_of", intersect_of, signature="intersect_of(A, B)", doc="A ∩ B", category=S)
    api.function("difference_of", difference_of, signature="difference_of(A, B)", doc="A \\\\ B", category=S)
    api.function("symmetric_difference", symmetric_difference, signature="symmetric_difference(A, B)", doc="A △ B", category=S)
    api.function("powerset_of", powerset_of, signature="powerset_of(A)", doc="all subsets", category=S)
    api.function("cartesian", cartesian, signature="cartesian(A, B)", doc="ordered pairs", category=S)
    api.function("is_subset", is_subset, signature="is_subset(A, B)", doc="A ⊆ B ?", category=S)
    api.function("cardinality", cardinality, signature="cardinality(A)", doc="number of elements", category=S)
    api.function("permutations_count", permutations_count, signature="permutations_count(n, k)", doc="n!/(n-k)!", category=S)
    api.function("combinations_count", combinations_count, signature="combinations_count(n, k)", doc="C(n, k)", category=S)
    api.function("multiset_count", multiset_count, signature="multiset_count(n, k)", doc="combinations with repetition", category=S)
    api.function("derangements", derangements, signature="derangements(n)", doc="permutations with no fixed point", category=S)
    api.function("permutations_of", permutations_of, signature="permutations_of([a, b, c], k)", doc="list them", category=S)
    api.function("combinations_of", combinations_of, signature="combinations_of([a, b, c], k)", doc="list them", category=S,
                 example="combinations_of([a, b, c, d], 2)")
    api.function("pigeonhole", pigeonhole, signature="pigeonhole(items, boxes)", doc="ceil(items/boxes)", category=S)
    G = "Discrete: generating functions"
    api.function("generating_function", generating_function, signature="generating_function(a_k, k, x)", doc="sum a_k x^k in closed form",
                 category=G, example="generating_function(1, k, x)")
    api.function("series_coefficients", series_coefficients, signature="series_coefficients(f, x, n)", doc="first n coefficients", category=G,
                 example="series_coefficients(1/(1 - x - x^2), x, 8)")
    api.function("sequence_terms", sequence_terms, signature="sequence_terms(a_k, k, n)", doc="first n terms", category=G)
    Gr = "Discrete: graphs"
    api.function("adjacency", adjacency, signature="adjacency([[a, b, w], ...])", doc="adjacency matrix", category=Gr)
    api.function("shortest_path", shortest_path, signature="shortest_path(edges, s, t)", doc="[distance, path] (Dijkstra)", category=Gr,
                 example="shortest_path([[A, B, 4], [A, C, 2], [C, B, 1], [B, D, 5]], A, D)")
    api.function("minimum_spanning_tree", minimum_spanning_tree, signature="minimum_spanning_tree(edges)", doc="[weight, edges]", category=Gr)
    api.function("is_connected", is_connected, signature="is_connected(edges)", doc="connected?", category=Gr)
    api.function("degrees", degrees, signature="degrees(edges)", doc="degree of each node", category=Gr)
    api.function("has_euler_circuit", has_euler_circuit, signature="has_euler_circuit(edges)", doc="all degrees even and connected?", category=Gr)
    L = "Discrete: logic"
    for name, fn, sig, doc in [("AND", AND, "AND(p, q, ...)", "conjunction"), ("OR", OR, "OR(p, q, ...)", "disjunction"), ("NOT", NOT, "NOT(p)", "negation"),
                               ("XOR", XOR, "XOR(p, q)", "exclusive or"), ("IMPLIES", IMPLIES, "IMPLIES(p, q)", "p → q"), ("IFF", IFF, "IFF(p, q)", "p ↔ q")]:
        api.function(name, fn, signature=sig, doc=doc, category=L)
    api.function("truth_table", truth_table, signature="truth_table(expr, [p, q])", doc="truth table", category=L,
                 example="truth_table(IMPLIES(p, q), [p, q])")
    api.function("is_tautology", is_tautology, signature="is_tautology(expr)", doc="always true?", category=L, example="is_tautology(OR(p, NOT(p)))")
    api.function("is_satisfiable", is_satisfiable, signature="is_satisfiable(expr)", doc="some assignment true?", category=L)
    api.function("simplify_logic", simplify_logic, signature="simplify_logic(expr)", doc="simplified form", category=L)
    api.function("to_cnf", to_cnf, signature="to_cnf(expr)", doc="conjunctive normal form", category=L)
    api.function("to_dnf", to_dnf, signature="to_dnf(expr)", doc="disjunctive normal form", category=L)
