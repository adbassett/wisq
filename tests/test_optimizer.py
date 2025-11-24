import pytest
import subprocess
import shutil
from utils import build_random_qasm, is_circuit_equiv
from qiskit.qasm2 import load as load_qasm_2, LEGACY_CUSTOM_INSTRUCTIONS
from pathlib import Path
import time
import random
from wisq.guoq import GATE_SETS
import csv

def load_csv_args(path="args_limited.csv"):
  rows = []
  with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      rows.append(row)
  return rows


def optimizer_cli_equivalence_test(
  num_qubits: int = 4,
  depth: int = 10,
  quick: bool = False,
  target_gateset: str | None = None,
  csv_row: dict | None = None
):
  """
  tests the optimizer by generating a random circuit, running it
  through the optimizer, then checking equivalence

  this is done with a random gateset as well, unless specified through
  the target_gateset param
  """

  seed = int(time.time())
  random.seed(seed)
  print(f"[INFO] Running with seed={seed}")
  
  if target_gateset:
    key = target_gateset.upper()
    gateset_name = key
    gateset = GATE_SETS[key]
  else:
    gateset_name = random.choice(list(GATE_SETS.keys()))
    gateset = GATE_SETS[gateset_name]

  input_path = Path("input.qasm")
  output_path = Path("out.qasm")

  try:
    build_random_qasm(num_qubits=num_qubits, depth=depth, seed=seed, basis_gates=gateset, dir_name=".", file_name="input")

    # sanity check
    assert input_path.exists(), "failed to generate QASM file"

    # build args list
    args = []
    if csv_row is not None:
      a = {k: (v if v != "" else None) for k, v in csv_row.items()}
      args = ["wisq"]
      args += ["--mode", "opt"]  # for now, only output .qasm
      args += ["--target_gateset", a["target_gateset"]]
      args += ["--optimization_objective", a["optimization_objective"]]
      args += ["--opt_timeout", "10" if quick else a["opt_timeout"]]
      args += ["--approx_epsilon", a["approx_epsilon"]]
      args += ["--architecture", a["architecture"]]
      if a["advanced_args"]:  
          args += ["--advanced_args", a["advanced_args"]]
      if a["verbose"] and a["verbose"].lower() == "true":
          args.append("--verbose")
      if a["guoq_help"] and a["guoq_help"].lower() == "true":
          args.append("--guoq_help")
      args.append(input_path.as_posix())
    else:
      args = [
        "wisq", "--mode", "opt", "-ot", "10" if quick else "60", "-ap", "0.5",
         "--target_gateset", gateset, input_path.as_posix()
      ]
    
    result = subprocess.run(
      args,
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

    if not is_circuit_equiv(circ_in, circ_out):
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

@pytest.mark.parametrize(
  "csv_row",
  load_csv_args()  
)
def test_optimizer_cli_equivalence_csv(csv_row):
  """
  tests the optimizer using arguments from args_limited.csv
  uses fixed num_qubits & depth (for now)
  """
  print(csv_row)
  optimizer_cli_equivalence_test(
    num_qubits=4,
    depth=10,
    quick=False,
    target_gateset=csv_row["target_gateset"],
    csv_row=csv_row
  )