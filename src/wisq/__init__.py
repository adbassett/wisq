import argparse
import ast
from qiskit import QuantumCircuit
from .architecture import square_sparse_layout, compact_layout
from .dascot import (
    extract_gates_from_file,
    extract_qubits_from_gates,
    dump,
    run_dascot,
    run_sat_scmr,
)
from .guoq import run_guoq, print_help, CLIFFORDT, FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE
from .utils import create_scratch_dir
import os
import shutil
import json


OPT_MODE = "opt"
FULL_FT_MODE = "full_ft"
SCMR_MODE = "scmr"

DEFAULT_EXT = {
    OPT_MODE: "qasm",
    FULL_FT_MODE: "json",
    SCMR_MODE: "json",
}


class Guoq_Help_Action(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
    ):
        super(Guoq_Help_Action, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print_help()
        parser.exit()


def map_and_route(
    input_path: str, arch_name: str, output_path: str, timeout: int, mode="dascot"
):
    """
    Apply a surface code mapping and routing pass to the given circuit

    Args:
        input_path: Path to the input circuit file.
        output_path: Path to the output circuit file.
        arch_name: A description of the target architecture. Valid options are the
        strings "square_sparse_layout" or "compact_layout" representing built-in architectures,
        or the path to a text file containing the description of a custom architecture
        timeout: Total timeout in seconds for both mapping and routing.


    Writes a JSON representing
    the scheduled circuit after mapping and routing to output_path.
    """
    circ = QuantumCircuit.from_qasm_file(input_path)
    gates, ops = extract_gates_from_file(input_path)
    id_to_op = {i: ops[i] for i in range(len(ops))}
    total_qubits = len(extract_qubits_from_gates(gates))
    circ = QuantumCircuit.from_qasm_file(input_path)
    if arch_name == "square_sparse_layout":
        arch = square_sparse_layout(total_qubits, magic_states="all_sides")
    elif arch_name == "compact_layout":
        arch = compact_layout(total_qubits, magic_states="all_sides")
    else:
        with open(arch_name) as f:
            arch = ast.literal_eval(f.read())
    if mode == "dascot":
        map, steps = run_dascot(circ, gates, arch, output_path, timeout)
    elif mode == "sat":
        map, steps = run_sat_scmr(circ, gates, arch, output_path, timeout)
    dump(arch, map, steps, id_to_op, output_path, gates)


def optimize(
    input_path: str,
    output_path: str,
    target_gateset: str,
    optimization_objective: str,
    timeout: int,
    approximation_epsilon: float = 0,
    advanced_args: dict = None,
    verbose: bool = False,
    path_to_synthetiq: str = None,
    serverless: bool = False,
) -> None:
    """
    Use the default GUOQ parameters to optimize a circuit. Recommended for most users. Advanced users can use `advanced_args` to override default values.

    Args:
        input_path: Path to the input circuit file.
        output_path: Path to the output circuit file.
        target_gateset: Target gateset to optimize the circuit to.
        timeout: Timeout in seconds. If set to 0, only transpiles and performs no optimization.
        approximation_epsilon: Approximation epsilon to use.
        advanced_args: Dictionary containing advanced arguments to pass to GUOQ, overriding default values except `-out` and `-job`. `guoq.print_help` displays available options.
        For example, if we want to override the default for `--rules` and use the `--remove-size-preserving-rules` flag, the dictionary would be `{"--rules": "file.txt", "--remove-size-preserving-rules": None}`.
        verbose: Whether to print verbose output.
    """
    run_guoq(
        input_path,
        output_path,
        target_gateset,
        optimization_objective,
        timeout,
        approximation_epsilon=approximation_epsilon,
        args=advanced_args,
        verbose=verbose,
        path_to_synthetiq=path_to_synthetiq,
        serverless=serverless,
    )


from qiskit import QuantumCircuit, converters, qasm2, dagcircuit
def _compose_evenly(qcs: list[QuantumCircuit], parts: int) -> list[QuantumCircuit]:
    while len(qcs) > parts:
        pair_costs = [
            (len(qcs[i].data) + len(qcs[i+1].data), i)
            for i in range(len(qcs) - 1)
        ]
        _, i = min(pair_costs, key=lambda x: x[0])  # pick cheapest adjacent pair
        qcs[i].compose(qcs[i+1], inplace=True)
        qcs.pop(i+1)
    return qcs


def naive_split(qasm_circuit: QuantumCircuit, parts: int) -> list[QuantumCircuit]:
    """
    splitting method: just split evenly into n parts

    Args:
        qasm_circuit (QuantumCircuit): the quantum circuit to split into n parts
        parts (int): the number of parts to split the quantum circuit into

    Returns:
        list[QuantumCircuit]: list of generated subcircuits
    """
    instructions = qasm_circuit.data
    total = len(instructions)
    chunk_size = (total + parts - 1) // parts
    # prepare subcircuits
    subcircuits = [
        QuantumCircuit(qasm_circuit.qubits, qasm_circuit.clbits) for _ in range(parts)
    ]
    # fill subcircuits
    for i in range(parts):
        start = i * chunk_size
        end = start + chunk_size
        for inst, qargs, cargs in instructions[start:end]:
            subcircuits[i].append(inst, qargs, cargs)
    return subcircuits


def topo_split_by_qubit_connectivity(qasm_circuit: QuantumCircuit, max_parts: int) -> list[QuantumCircuit]:
    """
    splitting method: splits into disconnected qubit sets, no shared qubits

    Args:
        qasm_circuit (QuantumCircuit): the quantum circuit to split into n parts
        max_parts (int): the max number of parts to split the quantum circuit into

    Returns:
        list[QuantumCircuit]: list of generated subcircuits
    """
    dag = converters.circuit_to_dag(qasm_circuit)
    sub_dags = dag.separable_circuits()
    qc_list = [converters.dag_to_circuit(d) for d in sub_dags]
    return _compose_evenly(qc_list, max_parts)


def topo_split_by_layers(qasm_circuit: QuantumCircuit, max_parts: int) -> list[QuantumCircuit]:
    """
    splitting method: splits into layers where gates act on disjoint qubits, returned by DAG.serial_layers

    Args:
        qasm_circuit (QuantumCircuit): the quantum circuit to split into n parts
        max_parts (int): the max number of parts to split the quantum circuit into

    Returns:
        list[QuantumCircuit]: list of generated subcircuits
    """
    dag = converters.circuit_to_dag(qasm_circuit)
    layer_dags = []
    for layer in dag.layers():
        new_dag = dagcircuit.DAGCircuit()
        for qr in dag.qregs.values():
            new_dag.add_qreg(qr)
        for cr in dag.cregs.values():
            new_dag.add_creg(cr)
        for node in layer["graph"].op_nodes():
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        layer_dags.append(new_dag)
    qc_list = [converters.dag_to_circuit(d) for d in layer_dags]
    return _compose_evenly(qc_list, max_parts)


def topo_split_by_depth(qasm_circuit: QuantumCircuit, max_parts: int) -> list[QuantumCircuit]:
    """
    splitting method: splits into subcircuits with max layers

    Args:
        qasm_circuit (QuantumCircuit): the quantum circuit to split into n parts
        max_parts (int): the max number of parts to split the quantum circuit into

    Returns:
        list[QuantumCircuit]: list of generated subcircuits
    """
    import math
    dag = converters.circuit_to_dag(qasm_circuit)
    max_depth = math.ceil(dag.depth() / max_parts)
    sub_dags = []
    current = None
    depth = 0
    for layer in dag.layers():
        if current is None:
            current = dagcircuit.DAGCircuit()
            # add same registers as qc
            for qreg in qasm_circuit.qregs:
                current.add_qreg(qreg)
            for creg in qasm_circuit.cregs:
                current.add_creg(creg)
            depth = 0
        # merge layer into current
        for node in layer["graph"].op_nodes():
            current.apply_operation_back(node.op, qargs=node.qargs, cargs=node.cargs)
        depth += 1
        if depth >= max_depth:
            sub_dags.append(current)
            current = None
    # leftovers
    if current is not None:
        sub_dags.append(current)
    qc_list = [converters.dag_to_circuit(d) for d in sub_dags]
    return qc_list



from typing import Callable
def optimize_parallel(
    input_path: str,
    max_threads: int,
    output_path: str,
    target_gateset: str,
    optimization_objective: str,
    timeout: int,
    approximation_epsilon: float = 0,
    advanced_args: dict | None = None,
    verbose: bool = False,
    path_to_synthetiq: str | None = None,
    split_method: Callable[[QuantumCircuit, int], list[QuantumCircuit]] = naive_split
) -> None:
    """
    parallel wrapper for optimizer

    args:
        circuit: str
            the input path of the circuit
        num_threads: int
            the number of threads to divide the optimizer between
    returns:
        none: writes the output file
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    import platform, sys, tempfile
    from .guoq import start_resynth_server, is_server_ready

    # Start resynthesis server if needed
    resynth_proc = None
    if not advanced_args or advanced_args.get("-resynth", None) != "NONE":
        if optimization_objective in ["FT", "T"] and path_to_synthetiq is None:
            system = platform.system().lower()
            processor = platform.processor().lower() or platform.machine().lower()
            print(system, processor)
            if system == "linux" and processor in ["x86_64"]:
                path_to_synthetiq = f"./bin/main_linux_{processor}"
            elif system == "darwin" and processor in ["arm", "i386"]:
                path_to_synthetiq = f"./bin/main_mac_{processor}"
            else:
                print(
                    "Unsupported platform for pre-compiled Synthetiq. Please follow the instructions here to compile Synthetiq for your platform: https://github.com/eth-sri/synthetiq/tree/bbe3c1299a97295f5af38eec647f6bbe9fdd9234. Then try again using the `--abs_path_to_synthetiq/-apts` option to pass in the absolute path to the Synthetiq `bin/main` binary."
                )
                sys.exit(1)
        resynth_proc = start_resynth_server(
            bqskit=(advanced_args is not None and "BQSKIT" in advanced_args.values())
            or optimization_objective in ["TWO_Q", "FIDELITY"],
            verbose=verbose,
            path_to_synthetiq=path_to_synthetiq,
        )
        # Wait for server to spin up
        while not is_server_ready():
            continue
    try:
        # split the file
        circuit_full = qasm2.load(input_path)
        circuits = split_method(circuit_full, max_threads)
        num_threads = len(circuits)
        temp_dir = Path(tempfile.mkdtemp(prefix="parallel_opt"))
        part_paths = []
        for i, part in enumerate(circuits):
            p = temp_dir / f"part_{i}.qasm"
            qasm2.dump(part, p)
            part_paths.append(p)
        # define worker threads
        def worker(qasm_path):
            thread_temp_dir = Path(tempfile.mkdtemp(prefix="parallel_opt_part_"))
            output_part = thread_temp_dir / f"{qasm_path.stem}_out.qasm"
            optimize(
                input_path=str(qasm_path),
                output_path=str(output_part),
                target_gateset=target_gateset,
                optimization_objective=optimization_objective,
                timeout=timeout,
                approximation_epsilon=approximation_epsilon,
                advanced_args=advanced_args,  # type:ignore
                verbose=verbose,
                path_to_synthetiq=path_to_synthetiq,  # type:ignore
                serverless=True,
            )
            return output_part
        remaining = list(part_paths)
        optimized_part_paths = []
        # we will retry for paths that don't exist after each thread
        if verbose:
            print(f"optimization for {len(remaining)} parts...")
        failed = []
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {pool.submit(worker, p): p for p in remaining}
            for f in as_completed(futures):
                part = futures[f]
                out_path = f.result()
                # check if the output file exists
                if out_path.exists():
                    optimized_part_paths.append(out_path)
                else:
                    failed.append(part)
        if failed:
            raise RuntimeError(f"failed to optimize {len(failed)} parts: {failed}")
        # get the circuits themselves, in order
        optimized_circuits_list = [
            qasm2.load(p) for p in sorted(
                optimized_part_paths,
                key=lambda path: int(path.stem.split("_")[-2])
            )
        ]
        # convert back to a single circuit
        full_quantum_circ = optimized_circuits_list[0]
        for circuit in optimized_circuits_list[1:]:
            full_quantum_circ.compose(circuit, inplace=True)
        # write final output
        qasm2.dump(full_quantum_circ, output_path)  # type:ignore
        if verbose:
            print(f"Parallel optimization complete -> {output_path}")
    finally:
        # kill resynthesis server if necessary
        if resynth_proc is not None:
            resynth_proc.terminate()
            resynth_proc.join()


def compile_fault_tolerant(
    input_path,
    output_path,
    opt_timeout,
    arch_name,
    approximation_epsilon=1e-10,
    verbose=False,
    mr_timeout=1800,
    mr_solver="dascot",
    path_to_synthetiq=None,
):
    """
    Compiles a circuit to a fault-tolerant architecture using the Clifford + T gate set.
    The input is a QASM circuit and architecture and the output is a JSON representing
    the scheduled circuit after optimizing, mapping, and routing.
    """
    scratch_dir_path, _ = create_scratch_dir(output_path)

    try:
        transpiled_and_optimized_path = os.path.join(
            scratch_dir_path, "after_guoq.qasm"
        )
        print(
            f"Decomposing to Clifford + T (if needed) and optimizing the input circuit with a timeout of {opt_timeout} seconds..."
        )
        optimize(
            input_path,
            transpiled_and_optimized_path,
            CLIFFORDT,
            FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE,
            opt_timeout,
            approximation_epsilon,
            verbose=verbose,
            path_to_synthetiq=path_to_synthetiq,
        )
        print(
            f"Done optimizing. Mapping and routing the optimized circuit with a timeout of {mr_timeout} seconds..."
        )
        map_and_route(
            transpiled_and_optimized_path,
            arch_name,
            output_path,
            mr_timeout,
            mode=mr_solver,
        )
    finally:
        if os.path.exists(scratch_dir_path):
            shutil.rmtree(scratch_dir_path)


def main():
    parser = argparse.ArgumentParser(
        prog="wisq",
        description="A compiler for quantum circuits. Optimize your circuits and/or map them to a surface code architecture. See README for example usage and documentation.",
    )
    opt = parser.add_argument_group(title="optimization config")
    scmr = parser.add_argument_group(title="mapping and routing config")
    parser.add_argument("input_path", help="path to the input circuit")
    parser.add_argument(
        "--output_path",
        "-op",
        help="path to write the output. Default is out.qasm or out.json",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default=FULL_FT_MODE,
        help="""
        control which compilation passes to apply. |
        opt: circuit optimization only |
        scmr: surface code mapping and routing only |
        full_ft (default): circuit optimization, then surface code mapping and routing
        """,
        choices=[OPT_MODE, FULL_FT_MODE, SCMR_MODE],
    )
    opt.add_argument(
        "--target_gateset",
        "-tg",
        help="target gateset for circuit optimization (default: Clifford + T)",
        default=CLIFFORDT,
        choices=guoq.GATE_SETS.keys(),
    )
    opt.add_argument(
        "--optimization_objective",
        "-obj",
        help="objective function used to guide the optimization (default: fault-tolerant objective function)",
        default=FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE,
        choices=["TWO_Q", "FIDELITY", "FT", "TOTAL", "T"],
    )
    opt.add_argument(
        "--opt_timeout",
        "-ot",
        help="integer representing timeout for optimization in seconds (default: 3600)",
        type=int,
        default=3600,
    )
    opt.add_argument(
        "--approx_epsilon",
        "-ap",
        help="the approximation epsilon for optimization (represented as a plain decimal or in scientific notation, e.g., 1e-8); output of optimization pass is equivalent to input circuit up to epsilon error (default: 0)",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--verbose", "-v", help="print verbose output", action="store_true"
    )
    scmr.add_argument(
        "--architecture",
        "-arch",
        help="target architecture for mapping and routing, can be one of the strings {'square_sparse_layout', 'compact_layout'} or the path to a file specifying a custom architecture. See README for format. (default: 'square_sparse')",
        default="square_sparse_layout",
    )
    scmr.add_argument(
        "--mr_timeout",
        "-tmr",
        type=int,
        help="integer representing timeout for mapping and routing in seconds (default: 1800)",
        default=1800,
    )
    scmr.add_argument(
        "--mr_solver",
        "-smr",
        help="solver to use for mapping and routing (default: 'dascot')",
        default="dascot",
    )
    parser.add_argument(
        "--guoq_help", "-gh", help="print GUOQ options", action=Guoq_Help_Action
    )
    parser.add_argument(
        "--advanced_args",
        "-aa",
        help="file path to JSON with advanced GUOQ args. See `optimize` for example",
    )
    parser.add_argument(
        "--abs_path_to_synthetiq",
        "-apts",
        help="absolute path to Synthetiq `main` binary",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path) and "wisq-circuits" in args.input_path:
        args.input_path = os.path.join(
            os.path.dirname(__file__),
            args.input_path,
        )

    if args.output_path is None:
        args.output_path = f"out.{DEFAULT_EXT[args.mode]}"

    if args.advanced_args is not None:
        with open(args.advanced_args, "r") as f:
            args.advanced_args = json.load(f)

    if args.mode == OPT_MODE:
        optimize(
            input_path=args.input_path,
            output_path=args.output_path,
            target_gateset=args.target_gateset,
            optimization_objective=args.optimization_objective,
            timeout=args.opt_timeout,
            approximation_epsilon=args.approx_epsilon,
            advanced_args=args.advanced_args,
            verbose=args.verbose,
            path_to_synthetiq=args.abs_path_to_synthetiq,
        )
    elif args.mode == FULL_FT_MODE:
        compile_fault_tolerant(
            input_path=args.input_path,
            output_path=args.output_path,
            opt_timeout=args.opt_timeout,
            arch_name=args.architecture,
            approximation_epsilon=args.approx_epsilon,
            verbose=args.verbose,
            mr_timeout=args.mr_timeout,
            mr_solver=args.mr_solver,
            path_to_synthetiq=args.abs_path_to_synthetiq,
        )
    elif args.mode == SCMR_MODE:
        map_and_route(
            input_path=args.input_path,
            output_path=args.output_path,
            arch_name=args.architecture,
            timeout=args.mr_timeout,
            mode=args.mr_solver,
        )


if __name__ == "__main__":
    main()
