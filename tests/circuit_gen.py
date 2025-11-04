import os
import math
import random
import time

def build_random_qasm(qubits: int, complexity: int, gateset: str) -> None:
    """
    builds a random qasm file with provided gate set
    
    args:
        qubits: int
            the number of qubits
        complexity: int
            2^complexity = number of quantum gates
        gateset: Literal["NAM", "CLIFFORDT", "IBMO", "IBMN", "ION"]
            the target gateset
    output:
        none
        writes file to folder randomly-generated-circuits
    """
    GATE_SETS = {
    "NAM": ["rz", "h", "x", "cx"],
    "CLIFFORDT": ["t", "tdg", "s", "sdg", "h", "x", "cx"],
    "IBMO": ["u1", "u2", "u3", "cx"],
    "IBMN": ["rz", "sx", "x", "cx"],
    "ION": ["rx", "ry", "rz", "rxx"],
    }
    gateset = gateset.upper()

    # validate gate set
    if gateset not in GATE_SETS:
        raise ValueError(f"unknown gateset {gateset}. must be one of {list(GATE_SETS)}")
    
    os.makedirs("randomly-generated-circuits", exist_ok=True)
    num_gates = 2 ** complexity
    qasm_lines = [
        "OPENQASM 2.0;",
        "include \"qelib1.inc\";",
        f"qreg q[{qubits}];"
    ]
    available_gates = GATE_SETS[gateset]
    for _ in range(num_gates):
        gate = random.choice(available_gates)
        # gate acts on 2 qubits
        if gate in {"cx", "rxx"}:
            if qubits < 2:
                continue
            q1, q2 = random.sample(range(qubits), 2)
            if gate == "rxx":
                theta = round(random.uniform(0, 2 * math.pi), 4)
                qasm_lines.append(f"{gate}({theta}) q[{q1}], q[{q2}];")
            else:
                qasm_lines.append(f"{gate} q[{q1}], q[{q2}];")
        
        # parameterized single-qubit gates
        elif gate in {"rz", "rx", "ry", "u1", "u2", "u3"}:
            q = random.randrange(qubits)
            if gate == "u1":
                theta = round(random.uniform(0, 2*math.pi), 4)
                qasm_lines.append(f"{gate}({theta}) q[{q}];")
            elif gate == "u2":
                phi = round(random.uniform(0, 2*math.pi), 4)
                lam = round(random.uniform(0, 2*math.pi), 4)
                qasm_lines.append(f"{gate}({phi},{lam}) q[{q}];")
            elif gate == "u3":
                theta = round(random.uniform(0, 2*math.pi), 4)
                phi = round(random.uniform(0, 2*math.pi), 4)
                lam = round(random.uniform(0, 2*math.pi), 4)
                qasm_lines.append(f"{gate}({theta},{phi},{lam}) q[{q}];")
            else:
                theta = round(random.uniform(0, 2*math.pi), 4)
                qasm_lines.append(f"{gate}({theta}) q[{q}];")
        
        # non-parameterized single-qubit gates
        else:
            q = random.randrange(qubits)
            qasm_lines.append(f"{gate} q[{q}];")
    
    filename = f"randomly-generated-circuits/{gateset}_{qubits}q_{complexity}c-{int(time.time())}.qasm"
    with open(filename, "w") as f:
        f.write("\n".join(qasm_lines))
    
    print(f"successfully created qasm2 file {filename}")