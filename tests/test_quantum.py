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


def test_states_and_normalization(ev):
    assert val(ev, "norm_sq(qubit(alpha, beta))") == "Abs(alpha)**2 + Abs(beta)**2"
    assert val(ev, "measure(qubit(alpha, beta))") == "Matrix([[|0>, Abs(alpha)**2], [|1>, Abs(beta)**2]])"
    assert val(ev, "normalize(ket(0) + ket(1))") == "sqrt(2)*|0>/2 + sqrt(2)*|1>/2"
    assert val(ev, "is_normalized(plus())") == "True"
    assert val(ev, "vec(bell_state(0))") == "Matrix([[sqrt(2)/2], [0], [0], [sqrt(2)/2]])"


def test_global_phase_and_bloch(ev):
    assert val(ev, "same_state(ket(0), exp(i theta) ket(0))") == "True"
    assert val(ev, "same_state(ket(0), ket(1))") == "False"
    assert val(ev, "global_phase(i ket(1), ket(1))") == "pi/2"
    assert val(ev, "bloch(plus())") == "[pi/2, 0]"
    assert val(ev, "bloch_vector(ket(0))") == "[0, 0, 1]"
    assert val(ev, "assume theta real", "simplify(bloch_vector(bloch_state(theta, 0))[0])") == "sin(theta)"


def test_tensor_and_entanglement(ev):
    assert val(ev, "tensor(ket(0), ket(1))") == "|01>"
    assert val(ev, "is_entangled(tensor(plus(), ket(0)))") == "False"
    assert val(ev, "is_entangled(bell_state(0))") == "True"
    assert val(ev, "schmidt(bell_state(0))") == "[sqrt(2)/2, sqrt(2)/2]"
    assert val(ev, "partial_trace(bell_state(0), 0)") == "Matrix([[1/2, 0], [0, 1/2]])"
    assert val(ev, "entropy(partial_trace(bell_state(0), 0))") == "1"
    assert val(ev, "purity(bell_state(0))") == "1"
    assert val(ev, "purity(partial_trace(bell_state(0), 1))") == "1/2"
    assert val(ev, "is_entangled(ghz(3), 0)") == "True"


def test_gates_and_circuits(ev):
    assert val(ev, "apply(H, ket(0))") == "sqrt(2)*|0>/2 + sqrt(2)*|1>/2"
    assert val(ev, "apply(CNOT, apply(H, ket(0, 0), 0), 0, 1)") == "sqrt(2)*|00>/2 + sqrt(2)*|11>/2"
    assert val(ev, "U = circuit(2, [[H, 0], [CNOT, 0, 1]])", "apply(U, ket(0, 0))") == "sqrt(2)*|00>/2 + sqrt(2)*|11>/2"
    assert val(ev, "is_unitary(H)") == "True" and val(ev, "is_unitary(Rx(theta))") == "True"
    assert val(ev, "controlled(X) - CNOT") == "Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])"
    assert val(ev, "apply(TOFFOLI, ket(1, 1, 0))") == "|111>"
    assert val(ev, "apply(CNOT, ket(1, 0, 1), 0, 2)") == "|100>"
    assert val(ev, "apply(X, ket(0, 1), 1)") == "|00>"
    assert val(ev, "simplify(dagger(T) T)") == "Matrix([[1, 0], [0, 1]])"


def test_observables_and_uncertainty(ev):
    assert val(ev, "is_hermitian(Y)") == "True"
    assert val(ev, "commutator(X, Y)") == "Matrix([[2*I, 0], [0, -2*I]])"
    assert val(ev, "expectation(Z, plus())") == "0"
    assert val(ev, "uncertainty(X, Y, ket(0))") == "[1, 1]"
    assert val(ev, "qvariance(Z, plus())") == "1"


def test_measurement(ev):
    assert val(ev, "measure(bell_state(0))") == "Matrix([[|00>, 1/2], [|01>, 0], [|10>, 0], [|11>, 1/2]])"
    assert val(ev, "measure(bell_state(0), 0)") == "Matrix([[|0>, 1/2], [|1>, 1/2]])"
    assert val(ev, "collapse(bell_state(0), 0, 1)") == "|11>"
    assert val(ev, "born(plus(), ket(0))") == "1/2"
    r = ev.evaluate([{"id": 1, "type": "math", "source": "collapse(bell_state(0), 0, 1)"}])[0]
    assert "irreversible" in r["outputs"][0]["notes"][0]
    counts = val(ev, "sample(bell_state(0), 100, 3)")
    assert "|00>" in counts and "|11>" in counts and "|01>" not in counts


def test_interference(ev):
    assert val(ev, "apply(H, apply(H, ket(0)))") == "|0>"
    assert val(ev, "apply(H, apply(Z, apply(H, ket(0))))") == "|1>"
    assert val(ev, "assume phi real", "born(apply(H, apply(phase(phi), apply(H, ket(0)))), ket(1))") in ("sin(phi/2)**2", "1/2 - cos(phi)/2")


def test_greek_names_are_variables_unless_called(ev):
    assert val(ev, "beta(2, 3)") == "1/12"
    assert val(ev, "alpha beta + gamma") == "alpha*beta + gamma"
    assert val(ev, "gamma(5) + zeta") == "zeta + 24"
    assert val(ev, "lambda = 500 nm", "lambda -> m") == "meter/2000000"
    assert val(ev, "beta = 3", "2 beta") == "6"
