import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator

def assert_no_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
        """
        helper: checks for measurements in quantum circuit

        args:
            circuit: QuantumCircuit
                the input quantum circuit
        returns:
            QuantumCircuit: the same circuit as passed in
        throws:
            ValueError: if the input circuit contains measurements
        """
        for instruction in circuit.data:
            if instruction.operation.name == "measure":
                raise ValueError("circuit contains measurements")
        return circuit


def is_circuit_equiv(qasm1: QuantumCircuit, qasm2: QuantumCircuit):
    """
    compare two quantum circuits to check if they are equal to each other,
    up to some global phase

    this piggybacks on the qiskit Operation's equiv(), which finds the entire unitary
    of the circuit. thus, it's really slow and memory-intensive, as the matrices
    will be 2^n by 2^n, but it is exact

    try to use with <= 12 qubits. anything over that might throw due to memory
    issues

    args:
        qasm1, qasm2: QuantumCircuit
            the input quantum circuits to check
    returns:
        bool: True if equal, False ow
    throws:
        ValueError: if either of the input circuits contain measurements
    """
    assert_no_measurements(qasm1)
    assert_no_measurements(qasm2)

    if qasm1.num_qubits != qasm2.num_qubits:
        return False

    op1 = Operator(qasm1)
    op2 = Operator(qasm2)
    return op1.equiv(op2)


def is_circuit_equiv_random_sv(qasm1: QuantumCircuit, qasm2: QuantumCircuit, tol: float = 1e-8, trials: int = 5) -> bool:
    """
    compare two quantum circuits to check if they are equal to each other,
    up to some global phase

    this is done by basically fuzzing both circuits, and checking that the
    resulting states are within a certain tolerance of each other
    (up to a global phase)

    this scales a lot better then the official implementation, but it is probabilistic,
    albeit barely. the odds of a random space lining up with an eigenvector
    get exponentially smaller the more qubits we have and is basically 0

    this still scales really badly, because there will be 2^n different states
    but it still scales much better than finding the unitary of the entire circuit
    (which is what qiskit's does)

    still, try to only use with <= 28 qubits. anything over that might throw
    due to memory issues

    args:
        qasm1, qasm2: QuantumCircuit
            the input quantum circuits to check
        tol: float
            tolerance for equality
        trials: int
            the number of test circuits to pump through
    returns:
        bool: True if equal within tolerance, False ow
    throws:
        ValueError: if either of the input circuits contain measurements
    """
    assert_no_measurements(qasm1)
    assert_no_measurements(qasm2)

    if qasm1.num_qubits != qasm2.num_qubits:
        return False
    n = qasm1.num_qubits
    for _ in range(trials):
        psi = np.random.randn(2**n) + 1j * np.random.randn(2**n)
        psi /= np.linalg.norm(psi)
        sv = Statevector(psi)
        out1 = sv.evolve(qasm1)
        out2 = sv.evolve(qasm2)
        # check for global phase
        inner = np.vdot(out1.data, out2.data)
        if abs(abs(inner) - 1.0) > tol:
            return False
    
    return True