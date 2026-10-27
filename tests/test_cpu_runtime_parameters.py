from __future__ import annotations

import hashlib
import json
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
    if version in {"2", "3", "4"}:
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

    def test_v3_self_snapshot_is_supported_and_has_a_distinct_backend(self):
        for module in [polybench, openmp]:
            _, code, metrics = self.parse(module, probe_values("3"))
            self.assertEqual(0, code)
            self.assertEqual("linux-process-probe-v3", metrics["process_metrics_backend"])
            self.assertEqual("ok", metrics["process_io_collection_status"])
            self.assertEqual(40, metrics["process_storage_read_bytes"])

    def test_v4_is_supported_without_changing_legacy_parser_results(self):
        for module in [polybench, openmp]:
            _, code, metrics = self.parse(module, probe_values("4"))
            self.assertEqual(0, code)
            self.assertEqual("linux-process-probe-v4", metrics["process_metrics_backend"])
            self.assertEqual("ok", metrics["process_io_collection_status"])
            self.assertEqual(40, metrics["process_storage_read_bytes"])

    def test_obsolete_collector_is_missing_not_a_true_zero(self):
        values = probe_values("4")
        values.update(io_status="incompatible_self_io_collector", io_error="rebuild probe and library together")
        for module in [polybench, openmp]:
            _, _, metrics = self.parse(module, values)
            self.assertEqual("incompatible_self_io_collector", metrics["process_io_collection_status"])
            self.assertTrue(all(metrics[field] == "" for field in runtime.IO_NUMERIC_FIELDS))

    def test_missing_normal_exit_snapshot_stays_missing(self):
        values = probe_values("3")
        values.update(io_status="missing_self_io_snapshot", io_error="no normal-exit callback")
        for module in [polybench, openmp]:
            _, code, metrics = self.parse(module, values)
            self.assertEqual(0, code)
            self.assertEqual("missing_self_io_snapshot", metrics["process_io_collection_status"])
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
        self.assertEqual(4, schema["schema_version"])
        self.assertIn("syscall", schema["io_collector_implementation"])
        self.assertIn("not_direct_static_inputs", schema["prediction_role"])
        self.assertIn("missing, not zero", schema["file_transfer_and_bandwidth"])


class OpenMPIntegrationTests(unittest.TestCase):
    def test_linux_command_selects_self_collector_without_replacing_target(self):
        with mock.patch.object(runtime.sys, "platform", "linux"):
            cmd = runtime.probe_command(Path("/probe"), Path("/tmp/report"), ["/task", "arg"], Path("/repo"))
        self.assertEqual([str(Path("/probe")), str(Path("/tmp/report")), "--io-collector",
                          str(Path("/repo") / "build/cpu_runtime_io_collector.so"),
                          "--", "/task", "arg"], cmd)

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


class OpenMPRepairValidationTests(unittest.TestCase):
    def write_batch(self, root, mutate_summary=None, mutate_runs=None):
        import check_cpu_runtime_parameters as checker
        metrics = {**record(10), "process_cpu_user_sec": .1, "process_cpu_system_sec": .01,
                   "process_max_rss_bytes": 4096, "process_major_page_faults": 0,
                   "process_minor_page_faults": 10, "process_fs_inputs": 0, "process_fs_outputs": 8,
                   "process_metrics_backend": "linux-process-probe-v4",
                   **{field: "" for field in runtime.TRANSFER_FIELDS},
                   "task_transfer_metrics_status": runtime.TRANSFER_STATUS}
        rows = []
        for phase, count in [("warmup", 1), ("measure", 3)]:
            for i in range(1, count + 1):
                rows.append({**metrics, "session_id": "repair", "run_id": f"{phase}-{i}",
                             "program_id": "myocyte", "input_id": "SIZE_01", "input_profile": "SIZE_01",
                             "phase": phase, "run_index": i, "runtime_sec": 1.,
                             "success": True, "timed_out": False, "returncode": 0, "error_message": ""})
        summary = {**metrics, "session_id": "repair", "program_id": "myocyte", "input_id": "SIZE_01",
                   "build_success": True, "run_success": True, "static_status": "success", "error_message": "",
                   "warmup_runs_requested": 1, "warmup_runs_completed": 1,
                   "measured_runs": 3, "runtime_sec_median": 1., "measurement_seconds_actual": 3.,
                   "measurement_seconds_requested": 3., "process_io_sampled_runs": 3, "process_io_missing_runs": 0,
                   "process_max_rss_bytes_peak": 4096, "prometheus_three_phase_collection_success": True,
                   "prometheus_three_phase_collection_error": "",
                   **{field: 0 for field in [*openmp.AST_FIELDS, *openmp.CFG_FIELDS, *openmp.IR_FIELDS]}}
        for phase in ["before", "during", "after"]:
            summary.update({**{field: 0 for field in openmp.SUMMARY_FIELDS
                                if field.startswith("host_") and f"_{phase}_" in field},
                            f"prometheus_{phase}_collection_success": True,
                            f"prometheus_{phase}_collection_error": "", f"prometheus_{phase}_sample_count": 6,
                            f"host_cpu_usage_pct_{phase}_mean": 2.})
        if mutate_summary:
            mutate_summary(summary)
        if mutate_runs:
            mutate_runs(rows)
        openmp.write_csv(root / "rodinia_openmp_summary_repair.csv", [summary], openmp.SUMMARY_FIELDS)
        openmp.write_csv(root / "rodinia_openmp_runs_repair.csv", rows, openmp.RUN_FIELDS)
        (root / "rodinia_openmp_environment_repair.json").write_text(
            json.dumps({"session_id": "repair", "cpu_runtime_parameter_schema": {"schema_version": 4}}), encoding="utf-8")
        return checker

    def test_valid_repair_reconciles_and_preserves_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            report = checker.check_openmp_results(root, 1, ["myocyte"])
            self.assertEqual("PASS", report["status"], report)
            self.assertEqual(4, report["run_records"])
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_complete_twenty_size_repair_is_accepted(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root)
            with (root / "rodinia_openmp_summary_repair.csv").open(encoding="utf-8-sig", newline="") as stream:
                template = next(csv.DictReader(stream))
            with (root / "rodinia_openmp_runs_repair.csv").open(encoding="utf-8-sig", newline="") as stream:
                invocations = list(csv.DictReader(stream))
            summaries, runs = [], []
            for i in range(1, 21):
                size = f"SIZE_{i:02d}"
                summaries.append({**template, "input_id": size, "input_profile": size})
                runs.extend({**row, "input_id": size, "input_profile": size,
                             "run_id": f"{size}-{row['run_id']}"} for row in invocations)
            openmp.write_csv(root / "rodinia_openmp_summary_repair.csv", summaries, openmp.SUMMARY_FIELDS)
            openmp.write_csv(root / "rodinia_openmp_runs_repair.csv", runs, openmp.RUN_FIELDS)
            report = checker.check_openmp_results(root, 20, ["myocyte"])
            self.assertEqual("PASS", report["status"], report)
            self.assertEqual(80, report["run_records"])

    def test_incomplete_io_is_rejected_even_if_program_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root, mutate_runs=lambda rows: rows[1].update(process_io_collection_status="invalid_self_io_snapshot"))
            report = checker.check_openmp_results(root, 1, ["myocyte"])
            self.assertEqual("FAIL", report["status"])
            self.assertTrue(any("I/O incomplete" in error for error in report["errors"]))

    def test_wrong_program_or_configuration_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root)
            self.assertEqual("FAIL", checker.check_openmp_results(root, 20, ["myocyte"])["status"])
            self.assertEqual("FAIL", checker.check_openmp_results(root, 1, ["bfs"])["status"])

    def test_summary_mean_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root, mutate_summary=lambda row: row.update(process_io_rchar_bytes=999))
            report = checker.check_openmp_results(root, 1)
            self.assertEqual("FAIL", report["status"])
            self.assertTrue(any("aggregation mismatch" in error for error in report["errors"]))

    def test_old_or_mixed_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root, mutate_runs=lambda rows: rows[0].update(process_metrics_backend="linux-process-probe-v3"))
            report = checker.check_openmp_results(root, 1)
            self.assertEqual("FAIL", report["status"])
            self.assertTrue(any("old or mixed" in error for error in report["errors"]))

    def test_duplicate_run_and_monitoring_failure_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root, mutate_summary=lambda row: row.update(prometheus_before_collection_success=False),
                                       mutate_runs=lambda rows: rows[1].update(run_id=rows[0]["run_id"]))
            report = checker.check_openmp_results(root, 1)
            self.assertEqual("FAIL", report["status"])
            self.assertTrue(any("duplicate run_id" in error for error in report["errors"]))
            self.assertTrue(any("before monitoring failure" in error for error in report["errors"]))

    def test_nonfinite_warmup_and_missing_transfer_status_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checker = self.write_batch(root, mutate_runs=lambda rows: rows[0].update(runtime_sec="NaN"))
            self.assertEqual("FAIL", checker.check_openmp_results(root, 1)["status"])
            checker = self.write_batch(root, mutate_summary=lambda row: row.update(task_transfer_metrics_status=""))
            self.assertEqual("FAIL", checker.check_openmp_results(root, 1)["status"])


@unittest.skipUnless(sys.platform.startswith("linux") and any(shutil.which(c) for c in ["cc", "gcc", "clang"]),
                     "Linux with a C compiler is required for the real /proc exit snapshot")
class LinuxProbeTests(unittest.TestCase):
    def test_both_pipeline_interfaces_capture_short_lived_child_io(self):
        import check_cpu_runtime_parameters as check
        report = check.run_checks()
        self.assertEqual({"polybench", "openmp", "polybench_native", "openmp_native",
                          "polybench_symbol_collision", "openmp_symbol_collision"}, set(report))
        for name, item in report.items():
            self.assertEqual("linux-process-probe-v4", item["metrics"]["process_metrics_backend"])
            if name.endswith("_symbol_collision"):
                self.assertEqual({"read": 91, "write": 92}, item["legacy_negative_control_exit_codes"])

    def test_child_failure_keeps_metrics_and_exit_status(self):
        completed, elapsed, timed_out, metrics = polybench.run_monitored_command(["/bin/sh", "-c", "exit 7"], timeout=5)
        self.assertEqual(7, completed.returncode)
        self.assertFalse(timed_out)
        self.assertGreater(elapsed, 0)
        self.assertEqual("linux-process-probe-v4", metrics["process_metrics_backend"])

    def test_direct_exit_does_not_fabricate_normal_exit_snapshot(self):
        completed, _, timed_out, metrics = polybench.run_monitored_command(
            [sys.executable, "-c", "import os; os._exit(0)"], timeout=5)
        self.assertEqual(0, completed.returncode)
        self.assertFalse(timed_out)
        self.assertEqual("missing_self_io_snapshot", metrics["process_io_collection_status"])
        self.assertTrue(all(metrics[field] == "" for field in runtime.IO_NUMERIC_FIELDS))

    def test_timeout_does_not_fabricate_io_values(self):
        completed, _, timed_out, metrics = polybench.run_monitored_command(["/bin/sleep", "5"], timeout=.05)
        self.assertTrue(timed_out)
        self.assertNotEqual(0, completed.returncode)
        self.assertTrue(all(metrics[field] == "" for field in runtime.IO_NUMERIC_FIELDS))


if __name__ == "__main__":
    unittest.main()
