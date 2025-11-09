import pytest
import subprocess
import shutil
from utils import build_random_qasm, is_circuit_equiv_random_sv
from qiskit.qasm2 import load as load_qasm_2, LEGACY_CUSTOM_INSTRUCTIONS
from pathlib import Path
import time
import random

@pytest.mark.parametrize("num_qubits,depth", [
  (2, 5),
  (3, 7),
  (4, 10),
  (5, 15),
  (6, 22)
])
def test_optimizer_cli_equivalence(num_qubits: int, depth: int, target_gateset: str | None = None):
  """
  tests the optimizer by generating a random circuit, running it
  through the optimizer, then checking equivalence

  this is done with a random gateset as well, unless specified through
  the target_gateset param
  """

  GATE_SETS = {
    "NAM": ["rz", "h", "x", "cx"],
    "CLIFFORDT": ["t", "tdg", "s", "sdg", "h", "x", "cx"],
    "IBMO": ["u1", "u2", "u3", "cx"],
    "IBMN": ["rz", "sx", "x", "cx"],
    "ION": ["rx", "ry", "rz", "rxx"],
  }

  seed = int(time.time())
  random.seed(seed)
  print(f"[INFO] Running with seed={seed}")
  
  if target_gateset:
    key = target_gateset.upper()
    if key not in GATE_SETS:
      raise ValueError(
        f"invalid target gateset {target_gateset}."
        f"valid options are: {','.join(GATE_SETS.keys())}"
      )
    gateset_name = key
    gateset = GATE_SETS[key]
  else:
    gateset_name = random.choice(list(GATE_SETS.keys()))
    gateset = GATE_SETS[gateset_name]

  input_path = Path("input.qasm")
  output_path = Path("out.qasm")

  try:
    # clean up from prior runs
    for path in [input_path, output_path]:
      if path.exists():
        path.unlink()

    build_random_qasm(num_qubits=num_qubits, depth=depth, seed=seed, basis_gates=gateset, dir_name=".", file_name="input")

    # sanity check
    assert input_path.exists(), "failed to generate QASM file"

    result = subprocess.run(
      [
        "wisq", "--mode", "opt", "-ot", "10", "-ap", "0.5",
         "--target_gateset", "NAM", input_path.as_posix()
      ],
      capture_output=True,
      text=True
    )

    if result.returncode != 0:
      fail_dir = Path("failed_tests")
      fail_dir.mkdir(exist_ok=True)
      name_header = f"{gateset_name}_{num_qubits}q_{depth}d_{seed}s"
      shutil.copy(input_path, fail_dir / f"input_{name_header}.qasm")
      shutil.copy(output_path, fail_dir / f"output_{name_header}.qasm")
      pytest.fail(
        f"'wisq' failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
      )

    assert output_path.exists(), "optimizer did not produce output file"

    circ_in = load_qasm_2(input_path.as_posix(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
    circ_out = load_qasm_2(output_path.as_posix(), custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)

    if not is_circuit_equiv_random_sv(circ_in, circ_out):
      fail_dir = Path("failed_tests")
      fail_dir.mkdir(exist_ok=True)
      name_header = f"{gateset_name}_{num_qubits}q_{depth}d_{seed}s"
      shutil.copy(input_path, fail_dir / f"input_{name_header}.qasm")
      shutil.copy(output_path, fail_dir / f"output_{name_header}.qasm")
      pytest.fail(
        f"circuits not equivalent for {gateset_name} gateset, {num_qubits} qubits, {depth} depth\n"
        f"copied input/output file to {fail_dir}/input_{name_header}.qasm "
        f"and {fail_dir}/output_{name_header}.qasm"
      )
  finally:
    for path in [input_path, output_path]:
      if path.exists():
        path.unlink()