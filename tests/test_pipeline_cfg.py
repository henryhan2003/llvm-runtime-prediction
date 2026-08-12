from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "pipeline_new.py"


class _Metric:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def labels(self, *args: object, **kwargs: object) -> "_Metric":
        return self

    def set(self, *args: object, **kwargs: object) -> None:
        pass

    def inc(self, *args: object, **kwargs: object) -> None:
        pass

    def observe(self, *args: object, **kwargs: object) -> None:
        pass


prometheus_stub = types.ModuleType("prometheus_client")
prometheus_stub.Counter = _Metric
prometheus_stub.Gauge = _Metric
prometheus_stub.Histogram = _Metric
prometheus_stub.start_http_server = lambda *args, **kwargs: None
sys.modules.setdefault("prometheus_client", prometheus_stub)

MODULE_SPEC = importlib.util.spec_from_file_location("pipeline_new_cfg_tests", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
pipeline = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = pipeline
MODULE_SPEC.loader.exec_module(pipeline)


class DotParsingTests(unittest.TestCase):
    def test_llvm_edge_ports_are_parsed(self) -> None:
        text = """
        Node0x1 [shape=record,label="entry"];
        Node0x2 [shape=record,label="true"];
        Node0x3 [shape=record,label="false"];
        Node0x1:s0 -> Node0x2;
        Node0x1:s1 -> Node0x3;
        """
        nodes, edges = pipeline.parse_dot_edges(text)
        self.assertEqual({"Node0x1", "Node0x2", "Node0x3"}, nodes)
        self.assertEqual(
            {("Node0x1", "Node0x2"), ("Node0x1", "Node0x3")},
            edges,
        )

    def test_hidden_dot_file_is_not_counted_twice(self) -> None:
        text = """
        Node0x1 [shape=record,label="entry"];
        Node0x2 [shape=record,label="exit"];
        Node0x1 -> Node0x2;
        """
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dot_path = directory / ".foo.dot"
            dot_path.write_text(text, encoding="utf-8")
            self.assertEqual(1, len(pipeline.unique_dot_files(directory)))
            features = pipeline.extract_cfg_features(directory)
        self.assertEqual(1, features["cfg_function_count"])
        self.assertEqual(2, features["basic_block_count"])
        self.assertEqual(1, features["cfg_edge_count"])


class GraphFeatureTests(unittest.TestCase):
    def test_single_loop_uses_dominator_backedge(self) -> None:
        nodes = {"entry", "header", "body", "exit"}
        edges = {
            ("entry", "header"),
            ("header", "body"),
            ("header", "exit"),
            ("body", "header"),
        }
        features = pipeline.graph_features(nodes, edges)
        self.assertEqual(2, features["cyclomatic_complexity"])
        self.assertEqual(1, features["branch_block_count"])
        self.assertEqual(1, features["loop_backedge_count"])
        self.assertEqual(1, features["natural_loop_count"])
        self.assertEqual(1, features["max_cfg_loop_depth"])
        self.assertEqual(1, features["scc_count"])

    def test_nested_natural_loops_have_depth_two(self) -> None:
        nodes = {"entry", "outer", "inner", "body", "inner_exit", "exit"}
        edges = {
            ("entry", "outer"),
            ("outer", "inner"),
            ("outer", "exit"),
            ("inner", "body"),
            ("inner", "inner_exit"),
            ("body", "inner"),
            ("inner_exit", "outer"),
        }
        features = pipeline.graph_features(nodes, edges)
        self.assertEqual(2, features["loop_backedge_count"])
        self.assertEqual(2, features["natural_loop_count"])
        self.assertEqual(2, features["max_cfg_loop_depth"])
        self.assertEqual(1, features["scc_count"])


class AstOwnershipTests(unittest.TestCase):
    def test_included_c_subtrees_are_kept_and_system_headers_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            main_source = directory / "main.c"
            included_source = directory / "included.c"
            ast_path = directory / "main.ast"
            main_source.write_text("int main(void) { return included(); }\n", encoding="utf-8")
            included_source.write_text(
                "int included(void) { return 1; }\n"
                "int included_helper(void) { return 2; }\n",
                encoding="utf-8",
            )
            ast_path.write_text(
                "\n".join(
                    [
                        "TranslationUnitDecl 0x0 <<invalid sloc>> <invalid sloc>",
                        "|-FunctionDecl 0x1 </usr/include/stdio.h:1:1, line:2:1> system_fn 'int ()'",
                        "| `-CompoundStmt 0x2 <col:1, col:2>",
                        f"|-FunctionDecl 0x3 <{included_source}:1:1, col:33> included 'int ()'",
                        "| `-CompoundStmt 0x4 <col:20, col:33>",
                        "|-FunctionDecl 0x5 <line:2:1, col:40> included_helper 'int ()'",
                        "| `-CompoundStmt 0x6 <col:27, col:40>",
                        f"`-FunctionDecl 0x7 <{main_source}:1:1, col:46> main 'int ()'",
                        "  `-CompoundStmt 0x8 <col:16, col:46>",
                    ]
                ),
                encoding="utf-8",
            )

            features = pipeline.extract_ast_features(
                ast_path,
                main_source,
                [main_source, included_source],
            )
            owned_text = pipeline.source_owned_ast_text(
                ast_path.read_text(encoding="utf-8"),
                main_source,
                [main_source, included_source],
            )

        self.assertEqual(3, features["function_count"])
        self.assertNotIn("system_fn", owned_text)
        self.assertIn("included_helper", owned_text)


class EnvironmentMetadataTests(unittest.TestCase):
    def test_llvm_opt_version_selects_the_version_line(self) -> None:
        output = "LLVM (http://llvm.org/):\n  LLVM version 14.0.6\n  Optimized build."
        with mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/opt"), mock.patch.object(
            pipeline, "command_output", return_value=output
        ):
            self.assertEqual("14.0.6", pipeline.llvm_opt_version())

    def test_linux_cache_topology_deduplicates_shared_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def cache(cpu: int, index: int, level: int, kind: str, size: str, shared: str) -> None:
                directory = root / f"cpu{cpu}" / "cache" / f"index{index}"
                directory.mkdir(parents=True)
                (directory / "level").write_text(str(level), encoding="ascii")
                (directory / "type").write_text(kind, encoding="ascii")
                (directory / "size").write_text(size, encoding="ascii")
                (directory / "shared_cpu_list").write_text(shared, encoding="ascii")

            for cpu in (0, 1):
                cache(cpu, 0, 1, "Data", "32K", str(cpu))
                cache(cpu, 1, 1, "Instruction", "32K", str(cpu))
                cache(cpu, 2, 2, "Unified", "512K", str(cpu))
                cache(cpu, 3, 3, "Unified", "16M", "0-1")
            topology = pipeline.linux_cpu_cache_topology(root)

        self.assertEqual(2, topology["cpu_l1d_cache_instance_count"])
        self.assertEqual(64 * 1024, topology["cpu_l1d_cache_bytes_total"])
        self.assertEqual(1, topology["cpu_l3_cache_instance_count"])
        self.assertEqual(16 * 1024 * 1024, topology["cpu_l3_cache_bytes_total"])


class PrometheusPhaseTests(unittest.TestCase):
    @staticmethod
    def result(
        start: float,
        end: float,
        cpu: float,
        memory: float,
        disk_read: float,
        disk_write: float,
        network: float,
    ) -> dict[str, object]:
        return {
            "prometheus_window_start_timestamp": start,
            "prometheus_window_end_timestamp": end,
            "prometheus_window_duration_seconds": end - start,
            "prometheus_sample_count": 5,
            "host_cpu_usage_pct_mean": cpu,
            "host_cpu_usage_pct_max": cpu + 2,
            "host_memory_used_bytes_mean": memory,
            "host_memory_used_bytes_max": memory + 10,
            "host_disk_read_bytes_delta": disk_read,
            "host_disk_write_bytes_delta": disk_write,
            "host_network_bytes_delta": network,
            "prometheus_collection_success": True,
            "prometheus_collection_error": "",
        }

    def test_three_phase_summary_subtracts_interpolated_background(self) -> None:
        phases = {
            "before": self.result(0, 5, 10, 1000, 50, 100, 150),
            "during": self.result(5, 15, 30, 1600, 500, 700, 900),
            "after": self.result(15, 20, 14, 1200, 100, 150, 200),
        }
        summary = pipeline.summarize_prometheus_phases(phases, 5)
        self.assertEqual(12, summary["prometheus_background_cpu_usage_pct_mean"])
        self.assertEqual(18, summary["host_cpu_usage_pct_background_adjusted_mean"])
        self.assertEqual(1100, summary["prometheus_background_memory_used_bytes_mean"])
        self.assertEqual(500, summary["host_memory_used_bytes_background_adjusted_mean"])
        self.assertEqual(350, summary["host_disk_read_bytes_background_adjusted_delta"])
        self.assertTrue(summary["prometheus_three_phase_collection_success"])


if __name__ == "__main__":
    unittest.main()
