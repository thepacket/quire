"""Quantum computing: linear algebra over complex Hilbert spaces, symbolically.

States are written in Dirac notation and displayed that way (a|0> + b|1>); every
function also accepts a column vector. Operators are matrices, so det, eigenvals,
dagger and friends from the core apply. Qubit 0 is the leftmost symbol in |q0 q1 ...>.

  States        ket, qubit, bloch_state, plus, minus, bell_state, ghz, to_dirac, vec, amplitudes, normalize,
                is_normalized, same_state, global_phase, bloch, bloch_vector
  Composition   tensor, is_entangled, schmidt, density, partial_trace, purity, entropy, fidelity
  Gates         X Y Z H S T, CNOT CZ SWAP TOFFOLI, Rx Ry Rz phase U3, controlled, gate_on, circuit, apply
  Operators     dagger, is_unitary, is_hermitian, commutator, anticommutator, expectation, qvariance,
                uncertainty (Robertson bound)
  Measurement   measure, born, collapse, sample
"""
from __future__ import annotations

import math
import random

import sympy as sp
from sympy.physics.quantum import Ket, TensorProduct

from quire.engine.errors import EvalError
from quire.modules import hooks

NAME = "quantum"
DESCRIPTION = "Qubits, gates, entanglement and measurement as symbolic linear algebra."


def _note(text):
    hooks.context.setdefault("notes", []).append(text)


# ---------------------------------------------------------------- representations
def _label(index: int, n: int) -> str:
    return format(index, f"0{n}b")


def _vec(state):
    """Column vector of a state given in Dirac form or as a vector."""
    if isinstance(state, sp.MatrixBase):
        if state.shape[1] != 1:
            if state.shape[0] == 1:
                return sp.ImmutableMatrix(state.T)
            raise EvalError("A state must be a column vector.")
        n = int(round(math.log2(state.shape[0])))
        if 2 ** n != state.shape[0]:
            raise EvalError("A state vector must have 2^n entries.")
        return sp.ImmutableMatrix(state)
    if isinstance(state, (list, tuple)):
        return _vec(sp.ImmutableMatrix(list(state)))
    expr = sp.expand(sp.sympify(state))
    terms = []  # (label, coefficient)
    for term in sp.Add.make_args(expr):
        kets = term.atoms(Ket)
        if len(kets) != 1:
            raise EvalError("Expected a state such as a ket(0) + b ket(1), or a column vector.")
        k = kets.pop()
        coeff = term.subs(k, 1)
        if coeff.has(Ket):
            raise EvalError("Expected a state such as a ket(0) + b ket(1), or a column vector.")
        terms.append((str(k.label[0]), coeff))
    n = len(terms[0][0])
    if any(len(lab) != n for lab, _ in terms):
        raise EvalError("All kets in a state must have the same number of qubits.")
    v = [sp.S.Zero] * (2 ** n)
    for lab, c in terms:
        if not set(lab) <= {"0", "1"}:
            raise EvalError(f"Ket label '{lab}' must be a string of 0s and 1s.")
        v[int(lab, 2)] += c
    return sp.ImmutableMatrix(v)


def _n(v) -> int:
    return int(round(math.log2(v.shape[0])))


def to_dirac(state):
    """Dirac form of a state vector: sum of amplitude * |bits>."""
    v = _vec(state)
    n = _n(v)
    terms = [sp.simplify(c) * Ket(_label(i, n)) for i, c in enumerate(v) if sp.simplify(c) != 0]
    return sp.Add(*terms) if terms else sp.S.Zero * Ket(_label(0, n))


def vec(state):
    return _vec(state)


def ket(*bits):
    """ket(0, 1) -> |01>; ket(011) also works."""
    if len(bits) == 1 and not isinstance(bits[0], (list, tuple)) and sp.sympify(bits[0]).is_Integer:
        s = str(int(bits[0]))
    else:
        s = "".join(str(int(b)) for b in (bits[0] if len(bits) == 1 and isinstance(bits[0], (list, tuple)) else bits))
    if not s or not set(s) <= {"0", "1"}:
        raise EvalError("ket takes 0s and 1s, e.g. ket(0, 1).")
    return Ket(s)


def qubit(alpha, beta):
    return sp.sympify(alpha) * Ket("0") + sp.sympify(beta) * Ket("1")


def amplitudes(state):
    return list(_vec(state))


def bra(state):
    return _vec(state).H


def inner(u, v):
    """<u|v>"""
    return sp.simplify((_vec(u).H * _vec(v))[0, 0])


def outer(u, v):
    """|u><v|"""
    return sp.ImmutableMatrix(_vec(u) * _vec(v).H)


def norm_sq(state):
    v = _vec(state)
    return sp.simplify(sum(sp.Abs(c) ** 2 for c in v))


def is_normalized(state):
    return sp.simplify(norm_sq(state) - 1) == 0


def normalize(state):
    v = _vec(state)
    nrm = sp.sqrt(norm_sq(v))
    if nrm == 0:
        raise EvalError("The zero vector cannot be normalized.")
    out = (v / nrm).applyfunc(sp.simplify)
    return to_dirac(out) if not isinstance(state, sp.MatrixBase) else sp.ImmutableMatrix(out)


def global_phase(u, v):
    """theta such that u = exp(i theta) v, or an error if the states differ physically."""
    a, b = _vec(u), _vec(v)
    ratio = None
    for x, y in zip(a, b):
        if sp.simplify(y) == 0:
            if sp.simplify(x) != 0:
                raise EvalError("The states differ (not by a global phase).")
            continue
        r = sp.simplify(x / y)
        if ratio is None:
            ratio = r
        elif sp.simplify(r - ratio) != 0:
            raise EvalError("The states differ (not by a global phase).")
    if ratio is None:
        raise EvalError("The states differ (not by a global phase).")
    # symbols inside a phase factor are angles: treat them as real
    real = {s_: sp.Symbol(s_.name, real=True) for s_ in ratio.free_symbols if s_.is_real is None}
    if sp.simplify(sp.Abs(ratio.subs(real)) - 1) != 0:
        raise EvalError("The states differ (not by a global phase).")
    return sp.simplify(sp.arg(ratio.subs(real))).subs({v: k for k, v in real.items()})


def same_state(u, v):
    try:
        global_phase(u, v)
        return True
    except EvalError:
        return False


def bloch_state(theta, phi):
    theta, phi = sp.sympify(theta), sp.sympify(phi)
    return sp.cos(theta / 2) * Ket("0") + sp.exp(sp.I * phi) * sp.sin(theta / 2) * Ket("1")


def bloch(state):
    """[theta, phi] Bloch angles of a single-qubit state (global phase removed)."""
    v = _vec(state)
    if v.shape[0] != 2:
        raise EvalError("bloch needs a single-qubit state.")
    a, b = v
    theta = 2 * sp.acos(sp.Abs(a) / sp.sqrt(norm_sq(v)))
    phi = sp.simplify(sp.arg(b) - sp.arg(a)) if sp.simplify(b) != 0 else sp.S.Zero
    return [sp.simplify(theta), phi]


def bloch_vector(state):
    return [expectation(g, state) for g in (X, Y, Z)]


def plus():
    return (Ket("0") + Ket("1")) / sp.sqrt(2)


def minus():
    return (Ket("0") - Ket("1")) / sp.sqrt(2)


def bell_state(k=0):
    k = int(k)
    return [(Ket("00") + Ket("11")) / sp.sqrt(2), (Ket("00") - Ket("11")) / sp.sqrt(2),
            (Ket("01") + Ket("10")) / sp.sqrt(2), (Ket("01") - Ket("10")) / sp.sqrt(2)][k % 4]


def ghz(n=3):
    n = int(n)
    return (Ket("0" * n) + Ket("1" * n)) / sp.sqrt(2)


# ---------------------------------------------------------------- composition
def tensor(*parts):
    if all(isinstance(p, sp.MatrixBase) and p.shape[1] > 1 for p in parts):
        out = parts[0]
        for p in parts[1:]:
            out = TensorProduct(out, p)
        return sp.ImmutableMatrix(out)
    out = _vec(parts[0])
    for p in parts[1:]:
        out = sp.ImmutableMatrix(TensorProduct(out, _vec(p)))
    return to_dirac(out)


def density(state):
    v = _vec(state)
    return sp.ImmutableMatrix(v * v.H).applyfunc(sp.simplify)


def _rho(x):
    if isinstance(x, sp.MatrixBase) and x.shape[0] == x.shape[1] and x.shape[0] > 1 and x.shape[1] > 1:
        return sp.ImmutableMatrix(x)
    return density(x)


def partial_trace(rho_or_state, *keep):
    """Reduced density matrix on the qubits listed in keep (indices from 0, leftmost)."""
    rho = _rho(rho_or_state)
    n = _n(rho)
    keep = [int(k) for k in keep]
    trace_out = [q for q in range(n) if q not in keep]
    m = len(keep)
    out = sp.zeros(2 ** m, 2 ** m)
    for i in range(2 ** n):
        for j in range(2 ** n):
            bi, bj = _label(i, n), _label(j, n)
            if all(bi[q] == bj[q] for q in trace_out):
                ki = int("".join(bi[q] for q in keep) or "0", 2)
                kj = int("".join(bj[q] for q in keep) or "0", 2)
                out[ki, kj] += rho[i, j]
    return sp.ImmutableMatrix(out).applyfunc(sp.simplify)


def purity(rho_or_state):
    rho = _rho(rho_or_state)
    return sp.simplify((rho * rho).trace())


def entropy(rho_or_state):
    """von Neumann entropy in bits."""
    rho = _rho(rho_or_state)
    s = sp.S.Zero
    for lam, mult in rho.eigenvals().items():
        lam = sp.simplify(lam)
        if lam != 0:
            s -= mult * lam * sp.log(lam, 2)
    return sp.simplify(s)


def schmidt(state, *first):
    """Schmidt coefficients across the split (qubits in first | the rest)."""
    v = _vec(state)
    n = _n(v)
    first = [int(q) for q in first] or [0]
    rest = [q for q in range(n) if q not in first]
    M = sp.zeros(2 ** len(first), 2 ** len(rest))
    for i, c in enumerate(v):
        b = _label(i, n)
        M[int("".join(b[q] for q in first), 2), int("".join(b[q] for q in rest) or "0", 2)] = c
    return [sp.simplify(s) for s in sorted(M.singular_values(), key=lambda s: -float(sp.N(sp.re(s))))]


def is_entangled(state, *first):
    coeffs = schmidt(state, *first)
    return len([c for c in coeffs if sp.simplify(c) != 0]) > 1


def fidelity(a, b):
    """|<a|b>|^2 for pure states."""
    return sp.simplify(sp.Abs(inner(a, b)) ** 2)


# ---------------------------------------------------------------- gates and operators
def _m(rows):
    return sp.ImmutableMatrix(rows)


X = _m([[0, 1], [1, 0]])
Y = _m([[0, -sp.I], [sp.I, 0]])
Z = _m([[1, 0], [0, -1]])
H = _m([[1, 1], [1, -1]]) / sp.sqrt(2)
S = _m([[1, 0], [0, sp.I]])
T = _m([[1, 0], [0, sp.exp(sp.I * sp.pi / 4)]])
CNOT = _m([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = _m([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
SWAP = _m([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
TOFFOLI = sp.ImmutableMatrix(sp.eye(8)).copy()
TOFFOLI = sp.ImmutableMatrix(sp.Matrix(TOFFOLI).tolist()[:6] + [[0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]])


def Rx(theta):
    t = sp.sympify(theta) / 2
    return _m([[sp.cos(t), -sp.I * sp.sin(t)], [-sp.I * sp.sin(t), sp.cos(t)]])


def Ry(theta):
    t = sp.sympify(theta) / 2
    return _m([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])


def Rz(theta):
    t = sp.sympify(theta) / 2
    return _m([[sp.exp(-sp.I * t), 0], [0, sp.exp(sp.I * t)]])


def phase(phi):
    return _m([[1, 0], [0, sp.exp(sp.I * sp.sympify(phi))]])


def U3(theta, phi, lam):
    theta, phi, lam = map(sp.sympify, (theta, phi, lam))
    return _m([[sp.cos(theta / 2), -sp.exp(sp.I * lam) * sp.sin(theta / 2)],
               [sp.exp(sp.I * phi) * sp.sin(theta / 2), sp.exp(sp.I * (phi + lam)) * sp.cos(theta / 2)]])


def dagger(A):
    return sp.ImmutableMatrix(sp.Matrix(A).H)


def _real_params(A):
    """Gate parameters are angles: treat free symbols as real for structural checks."""
    A = sp.Matrix(A)
    real = {s_: sp.Symbol(s_.name, real=True) for s_ in A.free_symbols if s_.is_real is None}
    if real:
        _note("symbolic parameters treated as real angles")
    return A.subs(real)


def is_unitary(A):
    A = _real_params(A)
    return sp.simplify(A.H * A - sp.eye(A.shape[0])) == sp.zeros(A.shape[0])


def is_hermitian(A):
    A = _real_params(A)
    return sp.simplify(A.H - A) == sp.zeros(*A.shape)


def commutator(A, B):
    A, B = sp.Matrix(A), sp.Matrix(B)
    return sp.ImmutableMatrix(A * B - B * A)


def anticommutator(A, B):
    A, B = sp.Matrix(A), sp.Matrix(B)
    return sp.ImmutableMatrix(A * B + B * A)


def controlled(G):
    """Controlled version of a k-qubit gate: control first, then the targets."""
    G = sp.Matrix(G)
    d = G.shape[0]
    out = sp.eye(2 * d)
    out[d:, d:] = G
    return sp.ImmutableMatrix(out)


def gate_on(G, n, *targets):
    """Embed a gate acting on the given qubits into the full 2^n-dimensional operator."""
    G = sp.Matrix(G)
    n = int(n)
    targets = [int(t) for t in targets]
    k = int(round(math.log2(G.shape[0])))
    if 2 ** k != G.shape[0] or G.shape[0] != G.shape[1]:
        raise EvalError("A gate must be a square 2^k x 2^k matrix.")
    if len(targets) != k:
        raise EvalError(f"This gate acts on {k} qubit(s); give {k} qubit index(es).")
    if len(set(targets)) != k or any(t < 0 or t >= n for t in targets):
        raise EvalError(f"Qubit indices must be distinct and between 0 and {n - 1}.")
    others = [q for q in range(n) if q not in targets]
    U = sp.zeros(2 ** n, 2 ** n)
    for i in range(2 ** n):
        bi = _label(i, n)
        sub_i = int("".join(bi[t] for t in targets), 2)
        for j in range(2 ** n):
            bj = _label(j, n)
            if all(bi[q] == bj[q] for q in others):
                sub_j = int("".join(bj[t] for t in targets), 2)
                U[i, j] = G[sub_i, sub_j]
    return sp.ImmutableMatrix(U)


def apply(G, state, *targets):
    """Apply a gate to a state; targets pick the qubits (omit when sizes match)."""
    v = _vec(state)
    n = _n(v)
    G = sp.Matrix(G)
    if G.shape[0] != v.shape[0]:
        if not targets:
            raise EvalError(f"The gate acts on {int(round(math.log2(G.shape[0])))} qubit(s) but the state has {n}; "
                            f"say which: apply(gate, state, 0).")
        G = gate_on(G, n, *targets)
    out = sp.ImmutableMatrix(G * v).applyfunc(sp.simplify)
    return sp.ImmutableMatrix(out) if isinstance(state, sp.MatrixBase) else to_dirac(out)


def circuit(n, steps):
    """Total unitary of a list of steps [[gate, q0, q1, ...], ...] applied in order to n qubits."""
    n = int(n)
    U = sp.eye(2 ** n)
    for step in steps:
        step = list(step)
        G, targets = step[0], step[1:]
        if not targets and sp.Matrix(G).shape[0] == 2 ** n:
            full = sp.Matrix(G)
        else:
            full = gate_on(G, n, *targets)
        U = full * U
    _note(f"circuit on {n} qubit(s), {len(list(steps))} gate(s); the product is one unitary matrix")
    return sp.ImmutableMatrix(U).applyfunc(sp.simplify)


# ---------------------------------------------------------------- observables and measurement
def expectation(A, state):
    v = _vec(state)
    return sp.simplify((v.H * sp.Matrix(A) * v)[0, 0])


def qvariance(A, state):
    A = sp.Matrix(A)
    return sp.simplify(expectation(A * A, state) - expectation(A, state) ** 2)


def uncertainty(A, B, state):
    """[sigma_A sigma_B, |<[A, B]>| / 2]: the Robertson bound holds when the first is >= the second."""
    sa = sp.sqrt(qvariance(A, state))
    sb = sp.sqrt(qvariance(B, state))
    bound = sp.Abs(expectation(commutator(A, B), state)) / 2
    _note("Robertson relation: sigma_A sigma_B >= |<[A, B]>| / 2")
    return [sp.simplify(sa * sb), sp.simplify(bound)]


def _total(v):
    """Normalization to divide by: symbolic amplitudes are taken as already normalized."""
    total = norm_sq(v)
    if total.free_symbols:
        _note("symbolic amplitudes: the state is assumed normalized (|alpha|^2 + |beta|^2 = 1)")
        return sp.S.One
    return total


def born(state, basis_state):
    """P = |<e|psi>|^2"""
    v = _vec(state)
    return sp.simplify(sp.Abs(inner(basis_state, v)) ** 2 / _total(v))


def measure(state, *qubits):
    """Outcome probabilities: all qubits, or only the listed ones. Rows: [outcome, probability]."""
    v = _vec(state)
    n = _n(v)
    total = _total(v)
    qubits = [int(q) for q in qubits]
    if not qubits:
        rows = [[Ket(_label(i, n)), sp.simplify(sp.Abs(c) ** 2 / total)] for i, c in enumerate(v)]
    else:
        probs = {}
        for i, c in enumerate(v):
            key = "".join(_label(i, n)[q] for q in qubits)
            probs[key] = probs.get(key, 0) + sp.Abs(c) ** 2 / total
        rows = [[Ket(k), sp.simplify(p)] for k, p in sorted(probs.items())]
    _note("Born rule: P(i) = |<e_i|psi>|^2" + ("" if not qubits else f"; qubit(s) {qubits} measured, the rest unmeasured"))
    return sp.ImmutableMatrix(rows)


def collapse(state, qubit_index, outcome):
    """State after measuring one qubit and observing outcome 0 or 1 (renormalized)."""
    v = _vec(state)
    n = _n(v)
    q, o = int(qubit_index), str(int(outcome))
    kept = [c if _label(i, n)[q] == o else sp.S.Zero for i, c in enumerate(v)]
    w = sp.ImmutableMatrix(kept)
    if sp.simplify(norm_sq(w)) == 0:
        raise EvalError(f"Outcome {o} on qubit {q} has probability zero.")
    _note(f"projective measurement of qubit {q} gave {o}; the state is projected and renormalized (irreversible)")
    return normalize(w) if isinstance(state, sp.MatrixBase) else to_dirac(normalize(w))


def sample(state, shots=1000, seed=1):
    """Simulated measurement counts over all qubits. Rows: [outcome, count]."""
    v = _vec(state)
    n = _n(v)
    total = float(norm_sq(v))
    probs = [float(sp.Abs(c) ** 2) / total for c in v]
    rng = random.Random(int(seed))
    counts = [0] * len(v)
    for _ in range(int(shots)):
        r = rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r < acc:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    _note(f"{int(shots)} simulated shots (seed {int(seed)})")
    return sp.ImmutableMatrix([[Ket(_label(i, n)), sp.Integer(c)] for i, c in enumerate(counts) if c])


# ---------------------------------------------------------------- quantum information
def kraus_apply(kraus, rho_or_state, qubit_index=0):
    """Apply a channel given by Kraus operators: sum K rho K†. Smaller operators act on the given qubit."""
    rho = _rho(rho_or_state)
    n = _n(rho)
    out = sp.zeros(*rho.shape)
    for K in kraus:
        K = sp.Matrix(K)
        if K.shape[0] != rho.shape[0]:
            k = int(round(math.log2(K.shape[0])))
            K = sp.Matrix(gate_on(K, n, *range(int(qubit_index), int(qubit_index) + k)))
        out += K * rho * K.H
    return sp.ImmutableMatrix(out).applyfunc(sp.simplify)


def kraus_valid(kraus):
    """Do the operators satisfy sum K† K = I (trace preservation)?"""
    ks = [_real_params(K) for K in kraus]
    n = ks[0].shape[1]
    total = sum((K.H * K for K in ks), sp.zeros(n))
    diff = sp.simplify(total - sp.eye(n))
    if diff == sp.zeros(n):
        return True
    # symbolic probabilities: check at sample values in (0, 1)
    import random

    syms = sorted(diff.free_symbols, key=lambda s_: s_.name)
    if not syms:
        return False
    rng = random.Random(3)
    for _ in range(5):
        point = {s_: sp.Rational(rng.randint(1, 99), 100) for s_ in syms}
        if any(abs(complex(sp.N(e.subs(point)))) > 1e-10 for e in diff):
            return False
    _note("checked at sample values of the parameters in (0, 1)")
    return True


def bit_flip(rho_or_state, p, qubit_index=0):
    p = sp.sympify(p)
    return kraus_apply([sp.sqrt(1 - p) * sp.eye(2), sp.sqrt(p) * X], rho_or_state, qubit_index)


def phase_flip(rho_or_state, p, qubit_index=0):
    p = sp.sympify(p)
    return kraus_apply([sp.sqrt(1 - p) * sp.eye(2), sp.sqrt(p) * Z], rho_or_state, qubit_index)


def depolarizing(rho_or_state, p, qubit_index=0):
    p = sp.sympify(p)
    return kraus_apply([sp.sqrt(1 - 3 * p / 4) * sp.eye(2), sp.sqrt(p / 4) * X, sp.sqrt(p / 4) * Y, sp.sqrt(p / 4) * Z],
                       rho_or_state, qubit_index)


def amplitude_damping(rho_or_state, gamma, qubit_index=0):
    g = sp.sympify(gamma)
    return kraus_apply([_m([[1, 0], [0, sp.sqrt(1 - g)]]), _m([[0, sp.sqrt(g)], [0, 0]])], rho_or_state, qubit_index)


def phase_damping(rho_or_state, lam, qubit_index=0):
    lam = sp.sympify(lam)
    return kraus_apply([_m([[1, 0], [0, sp.sqrt(1 - lam)]]), _m([[0, 0], [0, sp.sqrt(lam)]])], rho_or_state, qubit_index)


def fidelity_mixed(rho, sigma):
    """Uhlmann fidelity (tr sqrt(sqrt(rho) sigma sqrt(rho)))^2; states or density matrices."""
    r, s_ = _rho(rho), _rho(sigma)
    if r.rank() == 1:  # pure rho: <psi|sigma|psi>
        v = _vec(rho) if not (isinstance(rho, sp.MatrixBase) and rho.shape[0] == rho.shape[1] and rho.shape[0] > 1) else None
        if v is not None:
            return sp.simplify((v.H * s_ * v)[0, 0])
    import numpy as np
    from scipy.linalg import sqrtm

    R, S = np.array(r.evalf().tolist(), dtype=complex), np.array(s_.evalf().tolist(), dtype=complex)
    sr = sqrtm(R)
    val = np.trace(sqrtm(sr @ S @ sr)).real ** 2
    _note("Uhlmann fidelity evaluated numerically")
    return sp.Float(float(val))


def trace_distance(rho, sigma):
    d = _rho(rho) - _rho(sigma)
    eig = d.eigenvals()
    return sp.simplify(sum(sp.Abs(l) * m for l, m in eig.items()) / 2)


def bloch_mixed(rho_or_state):
    """[x, y, z] Bloch vector of a single-qubit density matrix (length < 1 when mixed)."""
    rho = _rho(rho_or_state)
    return [sp.simplify((rho * g).trace()) for g in (X, Y, Z)]


def concurrence(rho_or_state):
    """Wootters concurrence of a two-qubit state: 1 for Bell states, 0 for product states."""
    import numpy as np

    rho = _rho(rho_or_state)
    if rho.shape[0] != 4:
        raise EvalError("concurrence needs a two-qubit state.")
    R = np.array(rho.evalf().tolist(), dtype=complex)
    yy = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))
    Rt = yy @ R.conj() @ yy
    eig = np.sort(np.sqrt(np.abs(np.linalg.eigvals(R @ Rt))))[::-1]
    return sp.Float(max(0.0, eig[0] - eig[1] - eig[2] - eig[3]))


def qft(n):
    """Quantum Fourier transform on n qubits as a unitary matrix."""
    n = int(n)
    N = 2 ** n
    w = sp.exp(2 * sp.pi * sp.I / N)
    return sp.ImmutableMatrix(N, N, lambda j, k: w ** (j * k) / sp.sqrt(N))


def grover_iterate(n, marked, iterations=None):
    """State after Grover iterations on n qubits with the marked basis states (list of integers)."""
    n = int(n)
    N = 2 ** n
    marked = [int(m) for m in (marked if isinstance(marked, (list, tuple)) else [marked])]
    if iterations is None:
        iterations = int(round(sp.pi / 4 * math.sqrt(N / len(marked))))
    oracle = sp.diag(*[-1 if i in marked else 1 for i in range(N)])
    s = sp.ImmutableMatrix([1] * N) / sp.sqrt(N)
    diffuser = 2 * s * s.H - sp.eye(N)
    v = s
    for _ in range(int(iterations)):
        v = diffuser * (oracle * v)
    v = sp.ImmutableMatrix(v).applyfunc(sp.simplify)
    p = sp.simplify(sum(sp.Abs(v[m]) ** 2 for m in marked))
    _note(f"Grover: {int(iterations)} iteration(s); probability of a marked outcome {sp.N(p, 6)}")
    return to_dirac(v)


def grover_iterations(n, n_marked=1):
    """Optimal number of Grover iterations, round(pi/4 sqrt(N/M))."""
    return sp.Integer(int(round(sp.pi / 4 * math.sqrt(2 ** int(n) / int(n_marked)))))


def teleport_check():
    """Verify teleportation: Alice's Bell measurement plus Bob's corrections reproduce alpha|0> + beta|1>."""
    a, b = sp.symbols("alpha beta")
    psi = qubit(a, b)
    state = tensor(psi, bell_state(0))            # qubits: 0 = message, 1 = Alice, 2 = Bob
    state = apply(CNOT, state, 0, 1)
    state = apply(H, state, 0)
    v = _vec(state)
    results = []
    for m0 in (0, 1):
        for m1 in (0, 1):
            proj = [c if _label(i, 3)[0] == str(m0) and _label(i, 3)[1] == str(m1) else 0 for i, c in enumerate(v)]
            bob = sp.ImmutableMatrix([proj[i] for i in range(8) if _label(i, 3)[:2] == f"{m0}{m1}"]) * 2
            if m1:
                bob = X * bob
            if m0:
                bob = Z * bob
            results.append(sp.simplify(bob - sp.ImmutableMatrix([a, b])) == sp.zeros(2, 1))
    _note("all four measurement outcomes give Bob the original state after the corrections")
    return all(results)


# ---------------------------------------------------------------- registration
def register(api):
    St = "Quantum states"
    api.function("ket", ket, signature="ket(0, 1)", doc="basis state |01>", category=St, example="ket(0, 1)")
    api.function("qubit", qubit, signature="qubit(alpha, beta)", doc="alpha |0> + beta |1>", category=St,
                 example="qubit(alpha, beta)")
    api.function("bloch_state", bloch_state, signature="bloch_state(theta, phi)",
                 doc="cos(θ/2)|0> + e^{iφ} sin(θ/2)|1>", category=St, example="bloch_state(pi/2, 0)")
    api.function("plus", plus, signature="plus()", doc="|+> = (|0> + |1>)/√2", category=St)
    api.function("minus", minus, signature="minus()", doc="|-> = (|0> - |1>)/√2", category=St)
    api.function("bell_state", bell_state, signature="bell_state(k)", doc="Bell states k = 0..3", category=St, example="bell_state(0)")
    api.function("ghz", ghz, signature="ghz(n)", doc="GHZ state on n qubits", category=St)
    api.function("to_dirac", to_dirac, signature="to_dirac(v)", doc="Dirac form of a state vector", category=St)
    api.function("vec", vec, signature="vec(state)", doc="column vector of a state", category=St,
                 example="vec(bell_state(0))")
    api.function("amplitudes", amplitudes, signature="amplitudes(state)", doc="list of amplitudes", category=St)
    api.function("bra", bra, signature="bra(state)", doc="<psi| as a row vector", category=St)
    api.function("inner", inner, signature="inner(u, v)", doc="<u|v>", category=St, example="inner(plus(), minus())")
    api.function("outer", outer, signature="outer(u, v)", doc="|u><v|", category=St)
    api.function("norm_sq", norm_sq, signature="norm_sq(state)", doc="sum of |amplitude|^2", category=St,
                 example="norm_sq(qubit(alpha, beta))")
    api.function("is_normalized", is_normalized, signature="is_normalized(state)", doc="total probability 1?",
                 category=St)
    api.function("normalize", normalize, signature="normalize(state)", doc="scale to unit norm", category=St,
                 example="normalize(ket(0) + ket(1))")
    api.function("same_state", same_state, signature="same_state(u, v)", doc="equal up to a global phase?",
                 category=St, example="same_state(ket(0), exp(i theta) ket(0))")
    api.function("global_phase", global_phase, signature="global_phase(u, v)", doc="θ with u = e^{iθ} v",
                 category=St)
    api.function("bloch", bloch, signature="bloch(state)", doc="[θ, φ] on the Bloch sphere", category=St,
                 example="bloch(plus())")
    api.function("bloch_vector", bloch_vector, signature="bloch_vector(state)", doc="[<X>, <Y>, <Z>]", category=St)

    Co = "Composite systems"
    api.function("tensor", tensor, signature="tensor(a, b, ...)", doc="tensor product of states or operators",
                 category=Co, example="tensor(plus(), ket(0))")
    api.function("is_entangled", is_entangled, signature="is_entangled(state, q0, ...)",
                 doc="not a product across the split (qubits listed | rest)", category=Co, example="is_entangled(bell_state(0))")
    api.function("schmidt", schmidt, signature="schmidt(state, q0, ...)", doc="Schmidt coefficients across a split",
                 category=Co, example="schmidt(bell_state(0))")
    api.function("density", density, signature="density(state)", doc="density matrix |psi><psi|", category=Co)
    api.function("partial_trace", partial_trace, signature="partial_trace(rho, keep...)",
                 doc="reduced density matrix on the kept qubits", category=Co, example="partial_trace(bell_state(0), 0)")
    api.function("purity", purity, signature="purity(rho)", doc="tr(rho^2): 1 for pure states", category=Co)
    api.function("entropy", entropy, signature="entropy(rho)", doc="von Neumann entropy in bits", category=Co,
                 example="entropy(partial_trace(bell_state(0), 0))")
    api.function("fidelity", fidelity, signature="fidelity(a, b)", doc="|<a|b>|^2", category=Co)

    G = "Quantum gates"
    for name, g, doc in [("X", X, "Pauli X (NOT)"), ("Y", Y, "Pauli Y"), ("Z", Z, "Pauli Z"), ("H", H, "Hadamard"),
                         ("S", S, "phase gate (√Z)"), ("T", T, "π/8 gate (√S)"), ("CNOT", CNOT, "controlled NOT"),
                         ("CZ", CZ, "controlled Z"), ("SWAP", SWAP, "swap two qubits"),
                         ("TOFFOLI", TOFFOLI, "controlled-controlled NOT")]:
        api.constant(name, g, doc=doc, category=G, example=f"apply({name}, ket(0))" if g.shape[0] == 2 else None)
    api.function("Rx", Rx, signature="Rx(theta)", doc="rotation about x", category=G, example="Rx(theta)")
    api.function("Ry", Ry, signature="Ry(theta)", doc="rotation about y", category=G)
    api.function("Rz", Rz, signature="Rz(theta)", doc="rotation about z", category=G)
    api.function("phase", phase, signature="phase(phi)", doc="phase gate diag(1, e^{iφ})", category=G)
    api.function("U3", U3, signature="U3(theta, phi, lam)", doc="general single-qubit unitary", category=G)
    api.function("controlled", controlled, signature="controlled(G)", doc="controlled version of a gate",
                 category=G, example="controlled(H)")
    api.function("gate_on", gate_on, signature="gate_on(G, n, q0, ...)", doc="gate embedded in an n-qubit operator",
                 category=G, example="gate_on(H, 2, 0)")
    api.function("apply", apply, signature="apply(G, state, q0, ...)", doc="apply a gate to the given qubits",
                 category=G, example="apply(CNOT, apply(H, ket(0, 0), 0), 0, 1)")
    api.function("circuit", circuit, signature="circuit(n, [[gate, q...], ...])",
                 doc="unitary of a gate sequence on n qubits", category=G, example="circuit(2, [[H, 0], [CNOT, 0, 1]])")

    Op = "Operators & observables"
    api.function("dagger", dagger, signature="dagger(A)", doc="conjugate transpose A†", category=Op)
    api.function("is_unitary", is_unitary, signature="is_unitary(U)", doc="U† U = I ?", category=Op,
                 example="is_unitary(H)")
    api.function("is_hermitian", is_hermitian, signature="is_hermitian(A)", doc="A = A† ?", category=Op)
    api.function("commutator", commutator, signature="commutator(A, B)", doc="[A, B] = AB - BA", category=Op,
                 example="commutator(X, Z)")
    api.function("anticommutator", anticommutator, signature="anticommutator(A, B)", doc="{A, B} = AB + BA",
                 category=Op)
    api.function("expectation", expectation, signature="expectation(A, state)", doc="<psi|A|psi>", category=Op,
                 example="expectation(Z, plus())")
    api.function("qvariance", qvariance, signature="qvariance(A, state)", doc="<A^2> - <A>^2", category=Op)
    api.function("uncertainty", uncertainty, signature="uncertainty(A, B, state)",
                 doc="[σ_A σ_B, |<[A,B]>|/2]: Robertson bound", category=Op, example="uncertainty(X, Y, ket(0))")

    QI = "Quantum information"
    api.function("kraus_apply", kraus_apply, signature="kraus_apply([K1, K2], rho)", doc="channel sum K rho K†", category=QI)
    api.function("kraus_valid", kraus_valid, signature="kraus_valid([K1, K2])", doc="sum K† K = I ?", category=QI)
    api.function("bit_flip", bit_flip, signature="bit_flip(rho, p, qubit)", doc="bit-flip channel", category=QI, example="bit_flip(ket(0), 0.1)")
    api.function("phase_flip", phase_flip, signature="phase_flip(rho, p)", doc="phase-flip channel", category=QI)
    api.function("depolarizing", depolarizing, signature="depolarizing(rho, p, qubit)", doc="depolarizing channel", category=QI,
                 example="depolarizing(plus(), p)")
    api.function("amplitude_damping", amplitude_damping, signature="amplitude_damping(rho, gamma)", doc="energy relaxation", category=QI)
    api.function("phase_damping", phase_damping, signature="phase_damping(rho, lambda)", doc="dephasing", category=QI)
    api.function("fidelity_mixed", fidelity_mixed, signature="fidelity_mixed(rho, sigma)", doc="Uhlmann fidelity", category=QI)
    api.function("trace_distance", trace_distance, signature="trace_distance(rho, sigma)", doc="½ tr|ρ - σ|", category=QI)
    api.function("bloch_mixed", bloch_mixed, signature="bloch_mixed(rho)", doc="Bloch vector of a density matrix", category=QI,
                 example="bloch_mixed(depolarizing(ket(0), 0.5))")
    api.function("concurrence", concurrence, signature="concurrence(rho)", doc="two-qubit entanglement measure", category=QI,
                 example="concurrence(bell_state(0))")
    api.function("qft", qft, signature="qft(n)", doc="quantum Fourier transform matrix", category=QI, example="qft(2)")
    api.function("grover_iterate", grover_iterate, signature="grover_iterate(n, marked, iterations)", doc="Grover search state", category=QI,
                 example="grover_iterate(3, 5)")
    api.function("grover_iterations", grover_iterations, signature="grover_iterations(n, n_marked)", doc="optimal iteration count", category=QI)
    api.function("teleport_check", teleport_check, signature="teleport_check()", doc="verify the teleportation protocol symbolically", category=QI)
    Me = "Measurement"
    api.function("measure", measure, signature="measure(state, q...)",
                 doc="outcome probabilities (Born rule), all or listed qubits", category=Me, example="measure(bell_state(0))")
    api.function("born", born, signature="born(state, e)", doc="P = |<e|psi>|^2", category=Me,
                 example="born(plus(), ket(0))")
    api.function("collapse", collapse, signature="collapse(state, qubit, outcome)",
                 doc="post-measurement state", category=Me, example="collapse(bell_state(0), 0, 1)")
    api.function("sample", sample, signature="sample(state, shots, seed)", doc="simulated measurement counts",
                 category=Me, example="sample(bell_state(0), 1000)")
