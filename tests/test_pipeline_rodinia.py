from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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
            for input_spec in benchmark.inputs:
                for required in (*benchmark.required_files, *input_spec.required_files):
                    if not (workdir / required).resolve().exists():
                        missing.append(
                            f"{benchmark.benchmark_id}[{input_spec.input_id}]: required {required}"
                        )
        self.assertEqual([], missing)


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


if __name__ == "__main__":
    unittest.main()
