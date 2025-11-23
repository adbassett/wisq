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


def parse_qasm_file(path: str) -> tuple[list[str], list[str]]:
    """
    helper: get header lines and a list of gate lines
    args:
        path: str
            the path of the qasm file
    returns:
        header, gates: tuple[list[str], list[str]]
            a tuple containing the header and the gates of the file
    """
    header = []
    gates = []
    in_header = True
    with open(path, "r") as f:
        for line in f:
            l = line.strip()
            if in_header:
                header.append(line)
                if l.startswith(("qreg", "creg")):
                    in_header = False
            else:
                if l:
                    if l.startswith("qreg") or l.startswith("creg"):
                        header.append(line)
                    else:
                        gates.append(line)
    return header, gates


def write_qasm_file(path: str, header: list[str], gate_lines: list[str]) -> None:
    """
    helper: creates a qasm file from a list of header lines and gate lines
    args:
        path: str
            the path to write the output file to
        header: list[str]
            a list of header lines
        gate_lines: list[str]
            a list of actual quantum gates
    returns:
        none
            writes file to path given
    """
    with open(path, "w") as f:
        f.writelines(header)
        f.writelines(gate_lines)


from collections import defaultdict, deque
class ParseGate:
    """
    helper: class to keep track of the name of the gate and the qubits
    attached to it
    """
    def __init__(self, name: str, qubits: tuple[int, ...], raw: str):
        self.name = name
        self.qubits = tuple(qubits)
        self.raw = raw


import re
GATE_PATTERN = re.compile(r"^\s*([A-Za-z]\w*(?:\([^\)]*\))?)\s+(.*?);\s*$")
QUBIT_PATTERN = re.compile(r"[A-Za-z]\w*\[(\d+)\]")
def parse_gate(text: str) -> ParseGate:
    """
    helper: partse a gate into a ParseGate object
    """
    m = GATE_PATTERN.match(text)
    if not m:
        raise ValueError(f"invalid gate format: {text!r}")
    gate_token = m.group(1)
    rest = m.group(2)
    qubits = tuple(int(x) for x in QUBIT_PATTERN.findall(rest))
    return ParseGate(gate_token, qubits, text)

SELF_INVERSE_GATES = {"x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"}
INVERSE_PAIRS = {
    "s": "sdg", "sdg": "s",
    "t": "tdg", "tdg": "t",
    "rx": "rx", "ry": "ry", "rz": "rz",
}

from collections import defaultdict
def build_full_graph(gates: list[ParseGate]) -> tuple[dict[int, list[int]], dict[int, list[int]], list[int]]:
    """
    helper: builds graph with predecessor and successors
    returns (succ_adj, pred_adj, indeg_list)
    """
    n = len(gates)
    succ = defaultdict(list)
    pred = defaultdict(list)
    indeg = [0] * n
    last_gate_by_qubit = {}
    for i, g in enumerate(gates):
        touched = set()
        for q in g.qubits:
            if q in last_gate_by_qubit:
                prev = last_gate_by_qubit[q]
                if i not in succ[prev]:
                    succ[prev].append(i)
                if prev not in pred[i]:
                    pred[i].append(prev)
                touched.add(prev)
            last_gate_by_qubit[q] = i
    for i in range(n):
        indeg[i] = len(pred[i])
    return succ, pred, indeg


def find_asap_times(succ: dict[int, list[int]], pred: dict[int, list[int]], indeg: list[int]) -> list[int]:
    n = len(indeg)
    asap = [0] * n
    q = deque([i for i, d in enumerate(indeg) if d == 0])
    topo = []
    local_indeq = indeg[:]
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succ.get(u, []):
            asap[v] = max(asap[v], asap[u] + 1)
            local_indeq[v] -= 1
            if local_indeq[v] == 0:
                q.append(v)
    return asap


def find_alap_times(succ: dict[int, list[int]], pred: dict[int, list[int]], asap: list[int]) -> list[int]:
    n = len(asap)
    h = max(asap) + 1  # upper bound
    alap = [h] * n
    order = sorted(range(n), key=lambda i: asap[i], reverse=True)  # asap descending
    for u in order:
        if not succ.get(u):
            alap[u] = asap[u]  # leaf
        else:
            alap[u] = min(alap[v] - 1 for v in succ[u])
            alap[u] = max(alap[u], asap[u])
    return alap

import heapq
def pack_schedule(gates: list[ParseGate]) -> list[ParseGate]:
    """
    return reorder gates after packing into time steps
    """
    succ, pred, indeg = build_full_graph(gates)
    n = len(gates)
    asap = find_asap_times(succ, pred, indeg)
    alap = find_alap_times(succ, pred, asap)
    slack = [alap[i] - asap[i] for i in range(n)]  # range of which we can place each gate
    remaining_preds = [len(pred[i]) for i in range(n)]
    scheduled = [False] * n
    succ_map = succ
    qubit_free_at = defaultdict(int)  # track when qubit becomes free
    def priority_for(i):
        # (slack, -num_successors, asap, gate_index)
        return (slack[i], -len(succ_map.get(i, [])), asap[i], i)
    ready_heap = []
    # add gates with no more preds
    for i in range(n):
        if remaining_preds[i] == 0:
            heapq.heappush(ready_heap, (priority_for(i), i))
    time = 0
    schedule_time = [None]*n
    unscheduled_count = n
    while unscheduled_count > 0:
        any_scheduled = False
        temp_list = []
        while ready_heap:
            pr, idx = heapq.heappop(ready_heap)
            if all(qubit_free_at[q] <= time for q in gates[idx].qubits):
                # time to schedule
                schedule_time[idx] = time
                scheduled[idx] = True
                any_scheduled = True
                unscheduled_count -= 1
                # mark as busy
                for q in gates[idx].qubits:
                    qubit_free_at[q] = time + 1
                # push successors whose preds are now satisfied
                for v in succ_map.get(idx, []):
                    remaining_preds[v] -= 1
                    if remaining_preds[v] == 0:
                        heapq.heappush(ready_heap, (priority_for(v)), v)
            else:
                # cannot schedule
                temp_list.append((pr, idx))
        # re-add leftovers
        for item in temp_list:
            heapq.heappush(ready_heap, item)
        
        if not any_scheduled:
            # jump to next free qubit
            next_free_qubit = min(qubit_free_at.values()) if qubit_free_at else time+1
            next_ready_asap = None
            if ready_heap:
                # check earliest asap
                next_ready_asap = min(asap[idx] for _, idx in ready_heap)
            candidates = [t for t in (next_free_qubit, next_ready_asap) if t is not None and t > time]
            if candidates:
                time = min(candidates)
            else:
                # fallback
                time += 1
        else:
            # TODO: is advance by 1 fallback necessary?
            time += 1
    ordered = sorted(range(n), key=lambda i: (schedule_time[i], asap[i], i))
    return[gates[i] for i in ordered]


def minimize_qubit_lifetime(gates: list[ParseGate]) -> list[ParseGate]:
    return pack_schedule(gates)


def get_gate_qubits_used(gates: list[ParseGate]) -> set[int]:
    """
    helper: get all qubits used in gate sequence
    """
    qubits = set()
    for gate in gates:
        qubits.update(gate.qubits)
    return qubits


def peephole_optimzie(gates: list[ParseGate]) -> list[ParseGate]:
    """
    helper: apply local pattern matching to quickly cancel/merge consecutive gates
    """
    optimized = []
    i = 0
    while i < len(gates):
        current = gates[i]
        if i + 1 < len(gates):  # look ahead
            next_gate = gates[i+1]
            # check for overlap
            if set(current.qubits) & set(next_gate.qubits):
                current_base = current.name.split("(")[0]
                next_base = current.name.split("(")[0]

                # cancel self inverse gates
                if (current.qubits == next_gate.qubits and
                    current_base in SELF_INVERSE_GATES and
                    current_base == next_base
                    ):
                    i += 2
                    continue

                # cancel inverse pairs
                if (current.qubits == next_gate.qubits and
                    INVERSE_PAIRS.get(current_base) == next_base):
                    i += 2
                    continue
        optimized.append(current)
        i += 1
    return optimized


def build_dependency_graph(gates: list[ParseGate]):
    """
    helper: builds a DAG for each of the gates and the qubits,
    representing the dependencies of each
    """
    n = len(gates)
    deps_count = [0] * n
    adj = defaultdict(list)
    last_gate = {}
    for i, g in enumerate(gates):
        touched_gates = set()
        for q in g.qubits:
            if q in last_gate:
                touched_gates.add(last_gate[q])
            last_gate[q] = i

        # add dependencies
        for prev in touched_gates:
            adj[prev].append(i)
            deps_count[i] += 1

    return adj, deps_count


def reorder_qasm(gates: list[ParseGate]) -> list[ParseGate]:
    """
    re-orders the circuit so that every gate is scheduled as soon as possible
    """
    adj, deps_count = build_dependency_graph(gates)
    ready = deque([i for i, d in enumerate(deps_count) if d == 0])
    order = []
    while ready:
        i = ready.popleft()
        order.append(gates[i])
        for nxt in adj[i]:
            deps_count[nxt] -= 1
            if deps_count[nxt] == 0:
                ready.append(nxt)

    return order


def split_qasm_by_connectivity(
        qasm_path: str,
        num_parts: int,
        max_qubits_per_partition: int = 3,
) -> tuple[list[str], list[list[ParseGate]]]:
    """
    helper: splits qasm circuit into parts based on qubit connectivity
    limits subcircuits to a max length of max_qubits_per_partition, mainly
    for debugging and optimization purposes
    """
    header, gate_lines = parse_qasm_file(qasm_path)
    gates = [parse_gate(line.strip()) for line in gate_lines]
    partitions: list[list[ParseGate]] = []
    current_partition: list[ParseGate] = []
    current_qubits: set[int] = set()

    for gate in gates:
        gate_qubits = set(gate.qubits)
        new_qubits = gate_qubits - current_qubits
        # if we exceed max qubits, force a split
        if (len(current_qubits) + len(new_qubits) > max_qubits_per_partition and
            current_partition and
            len(partitions) < num_parts - 1):
            partitions.append(current_partition)
            current_partition = []
            current_qubits = set()
        
        current_partition.append(gate)
        current_qubits.update(gate_qubits)
    
    if current_partition:
        partitions.append(current_partition)
    return header, partitions


def split_qasm(qasm_path: str, num_parts: int):
    """
    helper: naieve split within a qasm file
    args:
        qasm_path: str
            the path of the qasm file to be parsed
        num_parts: int
            the number of parts to split the qasm circuit into
    returns:
        header, parts: tuple[list[str], list[list[str]]]
            header: the header present for the qasm file
            parts: a list of parts, each part being a list of qasm lines

    """
    header, gates = parse_qasm_file(qasm_path)
    total = len(gates)
    part_size = max(1, total // num_parts)
    parts = []
    for i in range(num_parts):
        start = i * part_size
        end = (i+1) * part_size if i < num_parts - 1 else total
        parts.append(gates[start:end])

    return header, parts


def extract_boundary_gates(
        gates: list[ParseGate],
        boundary_depth: int = 3,
) -> tuple[list[ParseGate], list[ParseGate], list[ParseGate]]:
    """
    extract gates near partition boundaries for seperate optimization
    returns (start_boundary, middle_boundary, end_boundary)
    """
    if len(gates) <= 2 * boundary_depth:
        return [], gates, []
    
    start = gates[:boundary_depth]
    end = gates[-boundary_depth:]
    middle = gates[boundary_depth:-boundary_depth]
    return start, middle, end


def reorder_qasm_file(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        lines = f.readlines()

    header_lines = []
    gate_lines = []

    for line in lines:
        l = line.strip()
        if not l or l.startswith("//"):
            header_lines.append(line)
            continue
        elif GATE_PATTERN.match(line):
            gate_lines.append(line.rstrip("\n"))
        else:
            header_lines.append(line)

    gates = [parse_gate(t) for t in gate_lines]
    reordered = reorder_qasm(gates)

    with open(output_path, "w") as f:
        f.writelines(header_lines)
        for g in reordered:
            if g.raw.strip().endswith(";"):
                f.write(g.raw.strip() + "\n")
            else:
                qubit_str = ",".join(f"q[{q}]" for q in g.qubits)
                f.write(f"{g.name} {qubit_str};\n")


def optimize_boundary_region(
    header: list[str], 
    gates: list[ParseGate],
    target_gateset: str,
    optimization_objective: str,
    timeout: int,
    approximation_epsilon: float = 0,
    advanced_args: dict | None = None,
    verbose: bool = False,
    path_to_synthetiq: str | None = None
) -> list[ParseGate]:
    """
    helper: optimize a specific region of gates, most most likely circuit boundaries
    """
    from pathlib import Path
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp(prefix="boundary_opt_"))
    temp_input = temp_dir / "boundary_input.qasm"
    temp_output = temp_dir / "boundary_output.qasm"
    
    write_qasm_file(str(temp_input), header, [g.raw + "\n" for g in gates])
    
    optimize(
        input_path=str(temp_input),
        output_path=str(temp_output),
        target_gateset=target_gateset,
        optimization_objective=optimization_objective,
        timeout=timeout,
        approximation_epsilon=approximation_epsilon,
        advanced_args=advanced_args,
        verbose=verbose,
        path_to_synthetiq=path_to_synthetiq,
        serverless=True,
    )
    
    _, optimized_gates = parse_qasm_file(str(temp_output))
    return [parse_gate(g.strip()) for g in optimized_gates if g.strip()]


def optimize_parallel(
    input_path: str,
    num_threads: int,
    output_path: str,
    target_gateset: str,
    optimization_objective: str,
    timeout: int,
    approximation_epsilon: float = 0,
    advanced_args: dict | None = None,
    verbose: bool = False,
    path_to_synthetiq: str | None = None,
    boundary_optimization: bool = True,
    peephole_passes: int = 2,
        ) -> str:
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

    # start up the server, if needed
    # Start resynthesis server if needed
    resynth_proc = None
    if not advanced_args or advanced_args.get("-resynth", None) != "NONE":
        if optimization_objective in ["FT", "T"] and path_to_synthetiq is None:
            system = platform.system().lower()
            processor = platform.processor().lower()
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
        # header, parts = split_qasm(input_path, num_threads)
        header, parts = split_qasm_by_connectivity(input_path, num_threads)
        temp_dir = Path(tempfile.mkdtemp(prefix="parallel_opt"))
        part_paths = []

        for i, part in enumerate(parts):
            p = temp_dir / f"part_{i}.qasm"
            write_qasm_file(str(p), header, part)
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
            raise RuntimeError(f"failed to optimize {len(remaining)} parts: {failed}")

        # sort through the order
        optimized_gate_lists = [
            parse_qasm_file(p)[1] for p in sorted(
                optimized_part_paths,
                key=lambda path: int(path.stem.split("_")[-2])
                )
        ]

        # convert back to ParseGate objects
        all_gates = [
            parse_gate(g.strip())
            for part in optimized_gate_lists
            for g in part
            if g.strip()
        ]
        
        # apply peephole optimization multiple passes
        if verbose:
            print(f"Applying {peephole_passes} peephole optimization passes...")
        for pass_num in range(peephole_passes):
            before_count = len(all_gates)
            all_gates = peephole_optimzie(all_gates)
            if verbose:
                gates_removed = before_count - len(all_gates)
                if gates_removed > 0:
                    print(f"  Pass {pass_num + 1}: Removed {gates_removed} gates")
        
        # apply boundary region optimization
        if boundary_optimization and len(all_gates) > 4:
            if verbose:
                print("Optimizing circuit boundaries...")
            
            boundary_depth = min(3, len(all_gates) // 4)
            start_gates, middle_gates, end_gates = extract_boundary_gates(
                all_gates, boundary_depth
            )
            
            temp_dir_boundary = Path(tempfile.mkdtemp(prefix="boundary_opt_"))
            
            # optimize start boundary
            if start_gates:
                start_gates = optimize_boundary_region(
                    header, start_gates, target_gateset,
                    optimization_objective, timeout // 3,
                    advanced_args, path_to_synthetiq
                )
            
            # optimize end boundary
            if end_gates:
                end_gates = optimize_boundary_region(
                    header, end_gates, target_gateset,
                    optimization_objective, timeout // 3,
                    advanced_args, path_to_synthetiq
                )
            
            all_gates = start_gates + middle_gates + end_gates
            
            if verbose:
                print("Boundary optimization complete")
        
        # final topological reordering for circuit depth
        all_gates = reorder_qasm(all_gates)
        
        # write final output
        final_gate_lines = [g.raw + "\n" if hasattr(g, 'raw') else str(g) + "\n" for g in all_gates]
        write_qasm_file(output_path, header, final_gate_lines)
        
        if verbose:
            print(f"Parallel optimization complete -> {output_path}")
            print(f"Final circuit has {len(all_gates)} gates")
        
        return output_path
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
