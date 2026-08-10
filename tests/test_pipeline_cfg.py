from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
