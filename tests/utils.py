import os
import time
from qiskit.circuit.random import random_circuit
from qiskit.qasm2 import dump as dump_qasm_2
from qiskit import transpile
from pathlib import Path


def build_random_qasm(
        num_qubits: int,
        depth: int,
        basis_gates: list[str] | None = None,
        dir_name: str = "randomly-generated-circuits",
        file_name: str | None = None
    ) -> None:
    """
    builds a random qasm file with the given params.
    basically a wrapper for qiskit.circuit.random's random_circuit,
    except it also saves the respective .qasm file
    
    args:
        num_qubits: int
            the number of qubits
        depth: int
            the depth of the generated circuit
        basis_gates: list[str] | None = None
            the list of gates to be included in the circuit
            if left as None, the gates will be selected from
            qiskit.circuit.library.standard_gates
        dir_name: str = "randomly-generated-circuits"
            the name of the directory to store the randomly generated circuits
        file_name: str | None
            the name to give the generated file. if left blank,
            a generated name with the num_qubits, depth, and time of
            creation will be used
    output:
        None
            writes file to folder randomly-generated-circuits
    """
    
    circ = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
    )
    circ = transpile(circ, basis_gates=basis_gates, optimization_level=0)
    os.makedirs(dir_name, exist_ok=True)
    if file_name == None:
        file_name = f"random_{num_qubits}q_{depth}d-{int(time.time())}"
    output_path = Path(dir_name) / f"{file_name}.qasm"
    dump_qasm_2(circ, output_path)