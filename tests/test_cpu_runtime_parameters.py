from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cpu_runtime_parameters as runtime
from test_pipeline_cfg import pipeline as polybench
import pipeline_rodinia as openmp


def probe_values(version="2"):
    values = dict(probe_version=version, elapsed_ns="123456789", returncode="0",
                  user_us="12000", system_us="3000", max_rss_bytes="4194304",
                  major_faults="1", minor_faults="23", fs_inputs="0", fs_outputs="8")
    if version == "2":
        values.update(io_status="ok", io_error="", child_pid="456")
        values.update({source: str(i * 10) for i, source in enumerate(runtime.IO_PROBE_FIELDS)})
    return values


def record(value, **overrides):
    return {"phase": "measure", "success": True, "process_metrics_success": True,
            "process_io_collection_status": "ok", "process_io_collection_error": "",
            **{field: value for field in runtime.IO_NUMERIC_FIELDS}, **overrides}


class ParserTests(unittest.TestCase):
    def parse(self, module, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics"
            path.write_text("\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf-8")
            return module.parse_process_probe_metrics(path)

    def test_v2_cpu_and_io_are_identical_in_both_pipelines(self):
        results = [self.parse(module, probe_values()) for module in [polybench, openmp]]
        for elapsed, code, metrics in results:
            self.assertAlmostEqual(.123456789, elapsed)
            self.assertEqual(0, code)
            self.assertEqual(.012, metrics["process_cpu_user_sec"])
            self.assertEqual(.003, metrics["process_cpu_system_sec"])
            self.assertEqual(4194304, metrics["process_max_rss_bytes"])
            self.assertEqual(0, metrics["process_fs_inputs"])
            self.assertEqual("linux-process-probe-v2", metrics["process_metrics_backend"])
            self.assertEqual("ok", metrics["process_io_collection_status"])
            self.assertEqual(0, metrics["process_io_rchar_bytes"])
            self.assertEqual(40, metrics["process_storage_read_bytes"])
            self.assertEqual("", metrics["task_file_transfer_bytes"])
        self.assertEqual(456, results[0][2]["process_id"])
        for field in runtime.RUN_FIELDS:
            self.assertEqual(results[0][2][field], results[1][2][field])

    def test_v1_is_accepted_without_inventing_io(self):
        for module in [polybench, openmp]:
            _, _, metrics = self.parse(module, probe_values("1"))
            self.assertEqual("linux-process-probe-v1", metrics["process_metrics_backend"])
            self.assertEqual("not_collected_probe_v1", metrics["process_io_collection_status"])
            self.assertTrue(all(metrics[field] == "" for field in runtime.IO_NUMERIC_FIELDS))

    def test_permission_failure_preserves_cpu_and_leaves_io_blank(self):
        values = probe_values()
        values.update(io_status="permission_denied", io_error="Permission denied")
        for module in [polybench, openmp]:
            _, code, metrics = self.parse(module, values)
            self.assertEqual(0, code)
            self.assertEqual(.012, metrics["process_cpu_user_sec"])
            self.assertEqual("permission_denied", metrics["process_io_collection_status"])
            self.assertTrue(all(metrics[field] == "" for field in runtime.IO_NUMERIC_FIELDS))

    def test_missing_negative_or_noninteger_io_is_not_a_true_zero(self):
        for bad in ["-1", "NaN", "1.2", ""]:
            values = probe_values()
            values["io_rchar"] = bad
            metrics = runtime.parse_probe_extensions(values)
            self.assertEqual("invalid_probe_io", metrics["process_io_collection_status"])
            self.assertTrue(all(metrics[f] == "" for f in runtime.IO_NUMERIC_FIELDS))
        values = probe_values()
        del values["io_read_bytes"]
        self.assertEqual("invalid_probe_io", runtime.parse_probe_extensions(values)["process_io_collection_status"])

    def test_invalid_core_probe_is_still_rejected(self):
        values = probe_values()
        values["elapsed_ns"] = "-1"
        for module in [polybench, openmp]:
            self.assertEqual((None, None, {}), self.parse(module, values))


class AggregationTests(unittest.TestCase):
    def test_zero_and_means_are_preserved(self):
        result = runtime.summarize([record(0), record(20)])
        self.assertEqual(10., result["process_io_rchar_bytes"])
        self.assertEqual(2, result["process_io_sampled_runs"])
        self.assertEqual("ok", result["process_io_collection_status"])
        self.assertEqual(0., runtime.summarize([record(0)])["process_storage_read_bytes"])

    def test_warmup_and_failed_runs_do_not_enter_summary(self):
        result = runtime.summarize([record(10), record(999, phase="warmup"), record(999, success=False)])
        self.assertEqual(10., result["process_io_rchar_bytes"])
        self.assertEqual(1, result["process_io_sampled_runs"])

    def test_partial_data_invalidates_summary_instead_of_averaging_subset(self):
        result = runtime.summarize([record(10), record(0, process_io_collection_status="permission_denied")])
        self.assertEqual("partial", result["process_io_collection_status"])
        self.assertEqual(1, result["process_io_sampled_runs"])
        self.assertEqual(1, result["process_io_missing_runs"])
        self.assertTrue(all(result[f] == "" for f in runtime.IO_NUMERIC_FIELDS))

    def test_no_runs_are_not_reported_as_zero_usage(self):
        result = runtime.summarize([])
        self.assertEqual("not_run", result["process_io_collection_status"])
        self.assertEqual("", result["process_storage_read_bytes"])
        self.assertEqual("", result["task_file_transfer_bytes"])

    def test_polybench_summary_keeps_separate_cpu_times(self):
        rows = [record(10, process_cpu_user_sec=.2, process_cpu_system_sec=.1,
                       process_max_rss_bytes=1000, process_fs_inputs=0, process_fs_outputs=8),
                record(20, process_cpu_user_sec=.4, process_cpu_system_sec=.3,
                       process_max_rss_bytes=3000, process_fs_inputs=0, process_fs_outputs=16)]
        result = polybench.summarize_process_records(rows, 3, 3, 2)
        self.assertAlmostEqual(.3, result["process_cpu_user_sec"])
        self.assertAlmostEqual(.2, result["process_cpu_system_sec"])
        self.assertEqual(2000, result["process_max_rss_bytes"])
        self.assertEqual(12, result["process_fs_outputs"])
        self.assertEqual(15, result["process_io_rchar_bytes"])

    def test_both_csv_schemas_include_all_runtime_fields_once(self):
        for fields in [polybench.PROCESS_RUN_FIELDS, polybench.MEASUREMENT_SUMMARY_FIELDS,
                       openmp.RUN_FIELDS, openmp.SUMMARY_FIELDS]:
            self.assertEqual(len(fields), len(set(fields)))
            self.assertTrue(set(runtime.RUN_FIELDS).issubset(fields))

    def test_schema_marks_dynamic_supervision_not_static_prediction_inputs(self):
        schema = runtime.schema_description()
        self.assertIn("not_direct_static_inputs", schema["prediction_role"])
        self.assertIn("missing, not zero", schema["file_transfer_and_bandwidth"])


class OpenMPIntegrationTests(unittest.TestCase):
    def test_openmp_summary_serializes_new_io_fields(self):
        spec = openmp.load_manifest(ROOT / "configs/rodinia_openmp.json")[0]
        item = spec.inputs[0]
        metrics = {**record(10), "process_cpu_user_sec": .1, "process_cpu_system_sec": .01,
                   "process_max_rss_bytes": 4096, "process_fs_inputs": 0, "process_fs_outputs": 8}
        measured = openmp.ProcessResult(["toy"], 0, .1, False, "", metrics)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(openmp, "run_process", return_value=measured):
            summary, rows, errors = openmp.measure_input("test", spec, item, Path(directory),
                warmup=1, runs=2, measurement_seconds=0, min_measured_runs=1, timeout=5)
            self.assertFalse(errors)
            self.assertEqual(2, summary["process_io_sampled_runs"])
            self.assertEqual(10, summary["process_io_rchar_bytes"])
            self.assertEqual(3, len(rows))
            for row in rows:
                self.assertEqual("ok", row["process_io_collection_status"])
                self.assertEqual(runtime.TRANSFER_STATUS, row["task_transfer_metrics_status"])
            path = Path(directory) / "summary.csv"
            openmp.write_csv(path, [summary], openmp.SUMMARY_FIELDS)
            self.assertIn("process_storage_read_bytes", path.read_text(encoding="utf-8-sig"))


@unittest.skipUnless(sys.platform.startswith("linux") and any(shutil.which(c) for c in ["cc", "gcc", "clang"]),
                     "Linux with a C compiler is required for the real /proc exit snapshot")
class LinuxProbeTests(unittest.TestCase):
    def test_both_pipeline_interfaces_capture_short_lived_child_io(self):
        import check_cpu_runtime_parameters as check
        report = check.run_checks()
        self.assertEqual({"polybench", "openmp"}, set(report))
        for item in report.values():
            self.assertEqual("linux-process-probe-v2", item["metrics"]["process_metrics_backend"])

    def test_child_failure_keeps_metrics_and_exit_status(self):
        completed, elapsed, timed_out, metrics = polybench.run_monitored_command(["/bin/sh", "-c", "exit 7"], timeout=5)
        self.assertEqual(7, completed.returncode)
        self.assertFalse(timed_out)
        self.assertGreater(elapsed, 0)
        self.assertEqual("linux-process-probe-v2", metrics["process_metrics_backend"])

    def test_timeout_does_not_fabricate_io_values(self):
        completed, _, timed_out, metrics = polybench.run_monitored_command(["/bin/sleep", "5"], timeout=.05)
        self.assertTrue(timed_out)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue(all(metrics[field] == "" for field in runtime.IO_NUMERIC_FIELDS))


if __name__ == "__main__":
    unittest.main()
