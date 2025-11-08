import os
import time
from qiskit.circuit.random import random_circuit
from qiskit.qasm2 import dumps as dump_qasm_2


def build_random_qasm(
        num_qubits: int,
        depth: int,
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
    os.makedirs(dir_name, exist_ok=True)
    circ_str = dump_qasm_2(circ)
    if file_name == None:
        file_name = f"random_{num_qubits}q_{depth}d-{int(time.time())}"
    filename = f"{dir_name}/{file_name}.qasm"
    with open(filename, "w") as f:
        f.write(circ_str)