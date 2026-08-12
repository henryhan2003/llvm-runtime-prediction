from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "pipeline_rodinia.py"
MANIFEST_PATH = PROJECT_ROOT / "configs" / "rodinia_openmp.json"
RODINIA_ROOT = PROJECT_ROOT / "datasets" / "rodinia_3.1"

MODULE_SPEC = importlib.util.spec_from_file_location("pipeline_rodinia", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
pipeline = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = pipeline
MODULE_SPEC.loader.exec_module(pipeline)


class RodiniaManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmarks = pipeline.load_manifest(MANIFEST_PATH)

    def test_manifest_has_unique_benchmarks_and_expected_disabled_entry(self) -> None:
        identifiers = [item.benchmark_id for item in self.benchmarks]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertIn("pathfinder", identifiers)
        mummergpu = next(item for item in self.benchmarks if item.benchmark_id == "mummergpu")
        self.assertFalse(mummergpu.enabled)
        self.assertIn("CUDA", mummergpu.disabled_reason)

    def test_every_enabled_benchmark_has_smoke_and_official_inputs(self) -> None:
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            profiles = {item.profile for item in benchmark.inputs}
            self.assertEqual({"smoke", "official"}, profiles, benchmark.benchmark_id)

    def test_all_declared_sources_and_required_inputs_exist(self) -> None:
        openmp_root = RODINIA_ROOT / "openmp"
        missing: list[str] = []
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            workdir = openmp_root / benchmark.workdir
            for source in benchmark.sources:
                if not (workdir / source).resolve().is_file():
                    missing.append(f"{benchmark.benchmark_id}: source {source}")
            for source in benchmark.ast_owned_sources:
                if not (workdir / source).resolve().is_file():
                    missing.append(f"{benchmark.benchmark_id}: AST-owned source {source}")
            for input_spec in benchmark.inputs:
                for required in (*benchmark.required_files, *input_spec.required_files):
                    if not (workdir / required).resolve().exists():
                        missing.append(
                            f"{benchmark.benchmark_id}[{input_spec.input_id}]: required {required}"
                        )
        self.assertEqual([], missing)

    def test_manifest_forces_consistent_btree_linker_and_skips_empty_getopt(self) -> None:
        btree = next(item for item in self.benchmarks if item.benchmark_id == "b+tree")
        self.assertIn("C_C=gcc", btree.build_commands[0])
        self.assertIn("OMP_LIB=-lgomp", btree.build_commands[0])
        self.assertIn("-B", btree.build_commands[0])

        kmeans = next(item for item in self.benchmarks if item.benchmark_id == "kmeans")
        self.assertNotIn("kmeans_openmp/getopt.c", kmeans.sources)

    def test_textually_included_c_files_are_owned_by_the_ast_extractor(self) -> None:
        expected = {
            "heartwall": {"define.c", "kernel.c"},
            "myocyte": {"cam.c", "solver.c", "embedded_fehlberg_7_8.c"},
            "srad_v1": {"define.c", "graphics.c", "resize.c", "timer.c"},
        }
        for benchmark_id, included_sources in expected.items():
            benchmark = next(
                item for item in self.benchmarks if item.benchmark_id == benchmark_id
            )
            self.assertTrue(
                included_sources.issubset(benchmark.ast_owned_sources),
                benchmark_id,
            )

    def test_each_ast_owned_source_belongs_to_exactly_one_translation_unit(self) -> None:
        openmp_root = RODINIA_ROOT / "openmp"
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            workdir = openmp_root / benchmark.workdir
            declared = {
                (workdir / source).resolve()
                for source in benchmark.ast_owned_sources
            }
            ownership_counts = {source: 0 for source in declared}
            for source in benchmark.sources:
                owned = pipeline.translation_unit_owned_sources(
                    (workdir / source).resolve(),
                    declared,
                )
                for owned_source in owned:
                    ownership_counts[owned_source] += 1
            unexpected = {
                str(source): count
                for source, count in ownership_counts.items()
                if count != 1
            }
            self.assertEqual({}, unexpected, benchmark.benchmark_id)


class AggregationTests(unittest.TestCase):
    def test_ast_average_is_weighted_by_literal_count(self) -> None:
        rows = [
            {"integer_literal_count": 1, "avg_integer_literal": 10, "max_integer_literal": 10},
            {"integer_literal_count": 3, "avg_integer_literal": 2, "max_integer_literal": 4},
        ]
        result = pipeline.aggregate_ast(rows)
        self.assertEqual(4, result["integer_literal_count"])
        self.assertEqual(4, result["avg_integer_literal"])
        self.assertEqual(10, result["max_integer_literal"])

    def test_cfg_average_degree_is_recomputed_after_merging(self) -> None:
        rows = [
            {"basic_block_count": 2, "cfg_edge_count": 1, "avg_out_degree": 0.5},
            {"basic_block_count": 3, "cfg_edge_count": 4, "avg_out_degree": 4 / 3},
        ]
        result = pipeline.aggregate_cfg(rows)
        self.assertEqual(5, result["basic_block_count"])
        self.assertEqual(5, result["cfg_edge_count"])
        self.assertEqual(1, result["avg_out_degree"])

    def test_null_device_placeholder_targets_linux(self) -> None:
        self.assertEqual(["output", "/dev/null"], pipeline.expand_tokens(["output", "{null_device}"]))


class ProcessProbeTests(unittest.TestCase):
    def test_probe_metrics_parser_preserves_precision_and_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "process.metrics"
            metrics_path.write_text(
                "probe_version=1\n"
                "elapsed_ns=123456789\n"
                "returncode=0\n"
                "user_us=12000\n"
                "system_us=3000\n"
                "max_rss_bytes=4194304\n"
                "major_faults=1\n"
                "minor_faults=23\n"
                "fs_inputs=4\n"
                "fs_outputs=5\n",
                encoding="utf-8",
            )
            elapsed, returncode, metrics = pipeline.parse_process_probe_metrics(
                metrics_path
            )

        self.assertEqual(0, returncode)
        self.assertAlmostEqual(0.123456789, elapsed)
        self.assertEqual(0.012, metrics["process_cpu_user_sec"])
        self.assertEqual(0.003, metrics["process_cpu_system_sec"])
        self.assertEqual(4194304, metrics["process_max_rss_bytes"])
        self.assertEqual("linux-process-probe-v1", metrics["process_metrics_backend"])

    def test_probe_metrics_parser_rejects_incomplete_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics_path = Path(temporary) / "process.metrics"
            metrics_path.write_text(
                "probe_version=1\nelapsed_ns=10\n",
                encoding="utf-8",
            )
            self.assertEqual(
                (None, None, {}),
                pipeline.parse_process_probe_metrics(metrics_path),
            )

    @unittest.skipUnless(
        os.name == "posix"
        and Path("/bin/true").is_file()
        and any(shutil.which(name) for name in ("cc", "gcc", "clang")),
        "Linux/POSIX compiler required",
    )
    def test_linux_probe_measures_child_instead_of_python_parent(self) -> None:
        pipeline._PROCESS_PROBE_PATH = None
        result = pipeline.run_process(
            ["/bin/true"],
            Path("/"),
            os.environ.copy(),
            timeout=5,
            measure_process=True,
            discard_stdout=True,
        )
        self.assertTrue(result.success, result.error_message)
        self.assertEqual(
            "linux-process-probe-v1",
            result.metrics["process_metrics_backend"],
        )
        self.assertGreater(result.metrics["process_max_rss_bytes"], 0)
        self.assertLess(result.metrics["process_max_rss_bytes"], 32 * 1024 * 1024)
        self.assertGreater(result.elapsed_sec, 0)
        self.assertLess(result.elapsed_sec, 1)


class MeasurementTests(unittest.TestCase):
    def test_time_budget_still_collects_minimum_run_count(self) -> None:
        benchmark = pipeline.BenchmarkSpec(
            benchmark_id="toy",
            workdir="toy",
            executable="toy",
            build_commands=(),
            clean_commands=(),
            sources=(),
            ast_owned_sources=(),
            include_dirs=(),
            static_flags=(),
            required_files=(),
            environment={},
            inputs=(),
            enabled=True,
            disabled_reason="",
        )
        input_spec = pipeline.InputSpec(
            input_id="slow",
            profile="smoke",
            args=(),
            threads=1,
            parameters={},
            required_files=(),
            environment={},
        )
        result = pipeline.ProcessResult(
            command=["toy"],
            returncode=0,
            elapsed_sec=6.0,
            timed_out=False,
            error_message="",
            metrics={
                "process_cpu_user_sec": 5.0,
                "process_cpu_system_sec": 0.1,
                "process_max_rss_bytes": 1024,
                "process_major_page_faults": 0,
                "process_minor_page_faults": 1,
                "process_fs_inputs": 0,
                "process_fs_outputs": 0,
                "process_metrics_backend": "test",
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "toy").mkdir()
            with mock.patch.object(pipeline, "run_process", return_value=result) as mocked:
                summary, rows, errors = pipeline.measure_input(
                    "session",
                    benchmark,
                    input_spec,
                    root,
                    warmup=0,
                    runs=5,
                    measurement_seconds=10,
                    min_measured_runs=3,
                    timeout=30,
                )
        self.assertEqual([], errors)
        self.assertEqual(3, mocked.call_count)
        self.assertEqual(3, len(rows))
        self.assertEqual(3, summary["measured_runs"])
        self.assertEqual(18, summary["measurement_seconds_actual"])

    def test_measurement_records_before_during_after_windows(self) -> None:
        benchmark = pipeline.BenchmarkSpec(
            benchmark_id="toy",
            workdir="toy",
            executable="toy",
            build_commands=(),
            clean_commands=(),
            sources=(),
            ast_owned_sources=(),
            include_dirs=(),
            static_flags=(),
            required_files=(),
            environment={},
            inputs=(),
            enabled=True,
            disabled_reason="",
        )
        input_spec = pipeline.InputSpec(
            input_id="small",
            profile="smoke",
            args=(),
            threads=1,
            parameters={},
            required_files=(),
            environment={},
        )
        result = pipeline.ProcessResult(
            command=["toy"],
            returncode=0,
            elapsed_sec=1.0,
            timed_out=False,
            error_message="",
            metrics={"process_metrics_backend": "test"},
        )
        windows: dict[str, tuple[float, float]] = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "toy").mkdir()
            with mock.patch.object(pipeline, "run_process", return_value=result), mock.patch.object(
                pipeline.time, "time", side_effect=[10.0, 15.0, 15.0, 25.0, 25.0, 30.0]
            ), mock.patch.object(pipeline.time, "sleep") as sleeping:
                pipeline.measure_input(
                    "session",
                    benchmark,
                    input_spec,
                    root,
                    warmup=0,
                    runs=1,
                    measurement_seconds=0,
                    min_measured_runs=1,
                    timeout=30,
                    phase_windows=windows,
                    prometheus_context_seconds=5,
                )
        self.assertEqual(
            {"before": (10.0, 15.0), "during": (15.0, 25.0), "after": (25.0, 30.0)},
            windows,
        )
        self.assertEqual([mock.call(5), mock.call(5)], sleeping.call_args_list)

    def test_new_safety_defaults_are_enabled(self) -> None:
        args = pipeline.parse_args(["--benchmark", "pathfinder"])
        self.assertEqual(3, args.min_measured_runs)
        self.assertFalse(args.allow_oversubscription)
        self.assertEqual("auto", args.prometheus_exporter)
        self.assertEqual(5.0, args.prometheus_context_seconds)

    def test_explicit_environment_id_is_available_to_rodinia(self) -> None:
        args = pipeline.parse_args(
            ["--benchmark", "pathfinder", "--env-id", "server_3970x_ubuntu20_clang14_gcc9"]
        )
        self.assertEqual("server_3970x_ubuntu20_clang14_gcc9", args.env_id)

    def test_dry_run_rejects_oversubscription_unless_explicitly_allowed(self) -> None:
        environment = {"cpu_logical_core_count": 2}
        common = [
            "--benchmark",
            "pathfinder",
            "--input-profile",
            "official",
            "--dry-run",
            "--no-static",
        ]
        with mock.patch.object(pipeline, "collect_environment", return_value=environment):
            self.assertEqual(1, pipeline.main(common))
            self.assertEqual(0, pipeline.main([*common, "--allow-oversubscription"]))


if __name__ == "__main__":
    unittest.main()
