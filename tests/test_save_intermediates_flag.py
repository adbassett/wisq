import contextlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


# Ensure the repository's src directory is on sys.path so that `import wisq`
# resolves without requiring the package to be installed in the environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Provide a lightweight stub for qiskit so the wisq module can be imported in
# environments where qiskit is not installed (e.g., CI).
if "qiskit" not in sys.modules:
    qiskit_stub = types.ModuleType("qiskit")

    class _QuantumCircuit:
        """Minimal stub that mimics the qiskit.QuantumCircuit API used here."""

        @staticmethod
        def from_qasm_file(path):
            # Return a descriptive object so tests can validate usage if needed.
            return {"qasm_path": path}

    qiskit_stub.QuantumCircuit = _QuantumCircuit
    converters_stub = types.ModuleType("qiskit.converters")

    def _not_implemented(*_args, **_kwargs):  # pragma: no cover - defensive stub
        raise NotImplementedError("converter stubs are not implemented")

    converters_stub.circuit_to_dag = _not_implemented
    converters_stub.dag_to_circuit = _not_implemented
    qiskit_stub.converters = converters_stub
    dagcircuit_stub = types.ModuleType("qiskit.dagcircuit")
    dagnode_stub = types.ModuleType("qiskit.dagcircuit.dagnode")

    class _DAGNode:  # pragma: no cover - structural stub
        pass

    class _DAGOpNode(_DAGNode):  # pragma: no cover - structural stub
        pass

    class _DAGInNode(_DAGNode):  # pragma: no cover
        pass

    class _DAGOutNode(_DAGNode):  # pragma: no cover
        pass

    dagnode_stub.DAGNode = _DAGNode
    dagnode_stub.DAGOpNode = _DAGOpNode
    dagnode_stub.DAGInNode = _DAGInNode
    dagnode_stub.DAGOutNode = _DAGOutNode
    dagcircuit_stub.dagnode = dagnode_stub
    qiskit_stub.dagcircuit = dagcircuit_stub
    sys.modules["qiskit"] = qiskit_stub
    sys.modules["qiskit.converters"] = converters_stub
    sys.modules["qiskit.dagcircuit"] = dagcircuit_stub
    sys.modules["qiskit.dagcircuit.dagnode"] = dagnode_stub


# Provide minimal stubs for internal wisq submodules that pull in heavy
# dependencies. These stubs expose only the attributes needed for import.
if "wisq.architecture" not in sys.modules:
    architecture_stub = types.ModuleType("wisq.architecture")

    def _make_architecture(num_qubits, magic_states="all_sides"):
        return {
            "width": max(1, num_qubits),
            "height": max(1, num_qubits),
            "alg_qubits": list(range(num_qubits)),
            "magic_states": list(range(num_qubits, num_qubits + 1)),
        }

    architecture_stub.square_sparse_layout = _make_architecture
    architecture_stub.compact_layout = _make_architecture
    sys.modules["wisq.architecture"] = architecture_stub

if "wisq.dascot" not in sys.modules:
    dascot_stub = types.ModuleType("wisq.dascot")

    def _placeholder(*_args, **_kwargs):  # pragma: no cover - defensive stub
        raise NotImplementedError("dascot functionality is not available in tests")

    dascot_stub.extract_gates_from_file = _placeholder
    dascot_stub.extract_qubits_from_gates = _placeholder
    dascot_stub.dump = _placeholder
    dascot_stub.run_dascot = _placeholder
    dascot_stub.run_sat_scmr = _placeholder
    sys.modules["wisq.dascot"] = dascot_stub

if "wisq.guoq" not in sys.modules:
    guoq_stub = types.ModuleType("wisq.guoq")

    def run_guoq(*_args, **_kwargs):  # pragma: no cover - stub behaviour
        return None

    def print_help():  # pragma: no cover
        return None

    guoq_stub.run_guoq = run_guoq
    guoq_stub.print_help = print_help
    guoq_stub.CLIFFORDT = "CLIFFORDT"
    guoq_stub.FAULT_TOLERANT_OPTIMIZATION_OBJECTIVE = "FT"
    guoq_stub.GATE_SETS = {"CLIFFORDT": None}
    sys.modules["wisq.guoq"] = guoq_stub


import wisq  # noqa: E402  (import after stub injection)
from wisq import utils


class SaveIntermediatesUtilitiesTest(unittest.TestCase):
    """Exhaustive tests for utility helpers that manage intermediate files."""

    def test_get_intermediate_filepath_appends_filename(self):
        base_dir = "/tmp/test_intermediates"
        expected = os.path.join(base_dir, "decomposed_circuit.qasm")
        self.assertEqual(utils.get_intermediate_filepath(base_dir), expected)

    def test_get_intermediate_filepath_returns_none_for_none(self):
        self.assertIsNone(utils.get_intermediate_filepath(None))

    def test_has_intermediate_file_false_for_none(self):
        self.assertFalse(utils.has_intermediate_file(None))

    def test_has_intermediate_file_true_when_valid_qasm_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intermediate_path = utils.get_intermediate_filepath(tmpdir)
            with open(intermediate_path, "w", encoding="utf-8") as fh:
                fh.write("OPENQASM 2.0;\nqreg q[1];")
            self.assertTrue(utils.has_intermediate_file(tmpdir))

    def test_has_intermediate_file_false_for_invalid_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intermediate_path = utils.get_intermediate_filepath(tmpdir)
            with open(intermediate_path, "w", encoding="utf-8") as fh:
                fh.write("not valid qasm content")
            # Should return False because content doesn't look like QASM
            self.assertFalse(utils.has_intermediate_file(tmpdir))

    def test_normalize_intermediates_path_expands_tilde(self):
        result = utils.normalize_intermediates_path("~/test_dir")
        self.assertTrue(result.startswith(os.path.expanduser("~")))
        self.assertIn("test_dir", result)

    def test_normalize_intermediates_path_makes_absolute(self):
        result = utils.normalize_intermediates_path("relative/path")
        self.assertTrue(os.path.isabs(result))

    def test_normalize_intermediates_path_rejects_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                utils.normalize_intermediates_path(filepath)
            self.assertIn("is a file", str(ctx.exception))
        finally:
            os.unlink(filepath)

    def test_normalize_intermediates_path_none_returns_none(self):
        self.assertIsNone(utils.normalize_intermediates_path(None))

    def test_save_intermediate_file_creates_directory_and_copies_contents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intermediates_dir = os.path.join(tmpdir, "sub", "dir")
            source_path = os.path.join(tmpdir, "source.qasm")
            payload = "OPENQASM 2.0;\n// test circuit\n"
            with open(source_path, "w", encoding="utf-8") as fh:
                fh.write(payload)

            utils.save_intermediate_file(source_path, intermediates_dir)

            saved_path = utils.get_intermediate_filepath(intermediates_dir)
            self.assertTrue(os.path.exists(saved_path))
            with open(saved_path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), payload)

    def test_save_intermediate_file_creates_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intermediates_dir = os.path.join(tmpdir, "ints")
            source_path = os.path.join(tmpdir, "source.qasm")
            with open(source_path, "w", encoding="utf-8") as fh:
                fh.write("OPENQASM 2.0;\ncx q[0],q[1];")

            utils.save_intermediate_file(source_path, intermediates_dir, input_circuit="test.qasm")

            metadata_path = utils.get_metadata_filepath(intermediates_dir)
            self.assertTrue(os.path.exists(metadata_path))
            with open(metadata_path, "r", encoding="utf-8") as fh:
                content = fh.read()
                self.assertIn("WISQ Intermediate", content)
                self.assertIn("test.qasm", content)

    def test_save_intermediate_file_noop_when_directory_is_none(self):
        with tempfile.NamedTemporaryFile("w", suffix=".qasm", delete=False) as fh:
            fh.write("OPENQASM 2.0;\n")
            source_path = fh.name
        try:
            result = utils.save_intermediate_file(source_path, None)
            self.assertIsNone(result)
        finally:
            os.unlink(source_path)

    def test_save_intermediate_file_raises_on_missing_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                utils.save_intermediate_file("/nonexistent/file.qasm", tmpdir)

    def test_get_default_intermediates_dir_returns_absolute_path(self):
        result = utils.get_default_intermediates_dir()
        self.assertTrue(os.path.isabs(result))
        self.assertIn("wisq_intermediates", result)


class CompileFaultTolerantIntermediatesTest(unittest.TestCase):
    """High-coverage tests for compile_fault_tolerant intermediate handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_path = os.path.join(self.temp_dir.name, "input.qasm")
        self.output_path = os.path.join(self.temp_dir.name, "out.json")
        with open(self.input_path, "w", encoding="utf-8") as fh:
            fh.write("OPENQASM 2.0;\nqreg q[1];\n")

    def _assert_no_scratch_directories(self):
        scratch_dirs = [
            entry
            for entry in os.listdir(self.temp_dir.name)
            if entry.startswith("wisq_tmp_")
        ]
        self.assertEqual(
            scratch_dirs,
            [],
            "Scratch directories should be cleaned up even when exceptions occur.",
        )

    def test_no_intermediate_directory_runs_optimize_and_does_not_save(self):
        with contextlib.ExitStack() as stack:
            mock_optimize = stack.enter_context(
                mock.patch.object(wisq, "optimize")
            )
            mock_map = stack.enter_context(
                mock.patch.object(wisq, "map_and_route")
            )
            mock_save = stack.enter_context(
                mock.patch.object(wisq, "save_intermediate_file")
            )
            stack.enter_context(
                mock.patch.object(wisq, "print_intermediate_status")
            )
            mock_map.return_value = None

            wisq.compile_fault_tolerant(
                input_path=self.input_path,
                output_path=self.output_path,
                opt_timeout=10,
                arch_name="square_sparse_layout",
                save_intermediates_dir=None,
            )

            mock_optimize.assert_called_once()
            mock_map.assert_called_once()
            mock_save.assert_not_called()
            self._assert_no_scratch_directories()

    def test_missing_intermediate_file_triggers_optimize_and_save(self):
        with contextlib.ExitStack() as stack:
            mock_optimize = stack.enter_context(
                mock.patch.object(wisq, "optimize")
            )
            mock_map = stack.enter_context(
                mock.patch.object(wisq, "map_and_route")
            )
            mock_save = stack.enter_context(
                mock.patch.object(wisq, "save_intermediate_file")
            )
            stack.enter_context(
                mock.patch.object(wisq, "has_intermediate_file", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(wisq, "print_intermediate_status")
            )
            stack.enter_context(
                mock.patch.object(wisq, "normalize_intermediates_path", side_effect=lambda x: x)
            )
            mock_map.return_value = None

            intermediates_dir = os.path.join(self.temp_dir.name, "ints")
            wisq.compile_fault_tolerant(
                input_path=self.input_path,
                output_path=self.output_path,
                opt_timeout=10,
                arch_name="square_sparse_layout",
                save_intermediates_dir=intermediates_dir,
            )

            mock_optimize.assert_called_once()
            mock_save.assert_called_once()
            mock_map.assert_called_once()
            self._assert_no_scratch_directories()

    def test_existing_intermediate_file_skips_optimize_and_uses_getter(self):
        intermediates_dir = os.path.join(self.temp_dir.name, "ints")
        expected_path = os.path.join(intermediates_dir, "decomposed_circuit.qasm")

        with contextlib.ExitStack() as stack:
            mock_optimize = stack.enter_context(
                mock.patch.object(wisq, "optimize")
            )
            mock_map = stack.enter_context(
                mock.patch.object(wisq, "map_and_route")
            )
            mock_save = stack.enter_context(
                mock.patch.object(wisq, "save_intermediate_file")
            )
            stack.enter_context(
                mock.patch.object(wisq, "has_intermediate_file", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(wisq, "print_intermediate_status")
            )
            stack.enter_context(
                mock.patch.object(wisq, "normalize_intermediates_path", side_effect=lambda x: x)
            )
            stack.enter_context(
                mock.patch.object(
                    wisq, "get_intermediate_filepath", return_value=expected_path
                )
            )
            mock_map.return_value = None

            wisq.compile_fault_tolerant(
                input_path=self.input_path,
                output_path=self.output_path,
                opt_timeout=10,
                arch_name="square_sparse_layout",
                save_intermediates_dir=intermediates_dir,
            )

            mock_optimize.assert_not_called()
            mock_save.assert_not_called()
            mock_map.assert_called_once_with(
                expected_path,
                "square_sparse_layout",
                self.output_path,
                1800,
                mode="dascot",
            )
            self._assert_no_scratch_directories()

    def test_exception_during_map_and_route_still_cleans_scratch_directory(self):
        intermediates_dir = os.path.join(self.temp_dir.name, "ints")

        with contextlib.ExitStack() as stack:
            mock_optimize = stack.enter_context(
                mock.patch.object(wisq, "optimize")
            )
            stack.enter_context(
                mock.patch.object(wisq, "save_intermediate_file")
            )
            stack.enter_context(
                mock.patch.object(wisq, "has_intermediate_file", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(wisq, "print_intermediate_status")
            )
            stack.enter_context(
                mock.patch.object(wisq, "normalize_intermediates_path", side_effect=lambda x: x)
            )
            stack.enter_context(
                mock.patch.object(
                    wisq, "map_and_route", side_effect=RuntimeError("boom")
                )
            )
            mock_optimize.return_value = None

            with self.assertRaises(RuntimeError):
                wisq.compile_fault_tolerant(
                    input_path=self.input_path,
                    output_path=self.output_path,
                    opt_timeout=10,
                    arch_name="square_sparse_layout",
                    save_intermediates_dir=intermediates_dir,
                )

        # Even after the exception, scratch directories must be removed.
        self._assert_no_scratch_directories()

    def test_save_intermediates_dir_preserves_relative_paths(self):
        """Ensure relative directories are supported and saved correctly."""
        rel_dir = "relative_intermediates"
        with contextlib.ExitStack() as stack:
            mock_optimize = stack.enter_context(
                mock.patch.object(wisq, "optimize")
            )
            mock_map = stack.enter_context(
                mock.patch.object(wisq, "map_and_route")
            )
            stack.enter_context(
                mock.patch.object(wisq, "has_intermediate_file", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(wisq, "print_intermediate_status")
            )
            mock_map.return_value = None

            def _fake_optimize(_input_path, output_path, *_args, **_kwargs):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as fh:
                    fh.write("OPENQASM 2.0;\n")

            mock_optimize.side_effect = _fake_optimize

            wisq.compile_fault_tolerant(
                input_path=self.input_path,
                output_path=self.output_path,
                opt_timeout=10,
                arch_name="square_sparse_layout",
                save_intermediates_dir=rel_dir,
            )

            mock_optimize.assert_called_once()
            mock_map.assert_called_once()
            # Cleanup the relative directory if the compile created it.
        if os.path.isdir(rel_dir):
            for root, _, files in os.walk(rel_dir, topdown=False):
                for file_name in files:
                    os.remove(os.path.join(root, file_name))
                os.rmdir(root)


if __name__ == "__main__":
    unittest.main()
