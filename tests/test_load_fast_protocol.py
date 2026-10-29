from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import test_resource_load as fixtures
import polybench_preparation_cache as cache

load, runner, checker, core = fixtures.load, fixtures.runner, fixtures.checker, fixtures.core


class CacheTests(fixtures.Fixture):
    def setup_cache(self):
        source = self.root / "sources"
        source.mkdir()
        (source / "toy.c").write_text("int main(void) { return 0; }\n")
        (source / "toy.h").write_text("#define N 10\n")
        fake = types.SimpleNamespace(__file__=__file__, PROJECT_ROOT=self.root, COMPAT_INCLUDE_DIR=self.root / "compat",
                                     AST_DIR=self.root / "original-ast", IR_DIR=self.root / "original-ir",
                                     CFG_DIR=self.root / "original-cfg", BUILD_DIR=self.root / "original-build",
                                     compiler_version=lambda c: "fixture compiler 1", llvm_opt_version=lambda: "fixture opt 1",
                                     program_extra_flags=lambda p, f: ["-D" + p.input_profile])
        program = core.Program(source / "toy.c", source, "LARGE_DATASET")
        args = argparse.Namespace(preparation_cache_dir=self.root / "cache", extra_flag=None, opt_level="O2",
                                  c_compiler="fixture-cc", cxx_compiler="fixture-cxx", run_c_compiler="fixture-cc", run_cxx_compiler="fixture-cxx")
        def produce():
            paths = {"ast_path": fake.AST_DIR / "toy.ast", "ir_path": fake.IR_DIR / "toy.ll",
                     "cfg_path": fake.CFG_DIR / "toy", "exe_path": fake.BUILD_DIR / "toy"}
            for name, path in paths.items():
                if name == "cfg_path":
                    path.mkdir(parents=True)
                    (path / ".main.dot").write_text("digraph {}")
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("fixture " + name)
            return {**paths, "ast_success": True, "ir_success": True, "cfg_success": True, "build_success": True,
                    "cfg_kind": "dot", "errors": [], "ast_features": {"ast_node_count": 10},
                    "ir_features": {"ir_instruction_count": 5}, "cfg_features": {"basic_block_count": 1}}
        return fake, program, args, produce

    def test_four_loads_prepare_once_and_share_verified_paths_without_runtime_cache(self):
        fake, program, args, produce = self.setup_cache()
        previous = fake.AST_DIR
        producer = mock.Mock(side_effect=produce)
        with mock.patch.object(cache.shutil, "which", return_value=sys.executable):
            payloads = [cache.get_or_prepare(fake, program, args, producer) for _ in range(4)]
        producer.assert_called_once()
        self.assertEqual(previous, fake.AST_DIR)
        self.assertTrue(all(p == payloads[0] for p in payloads))
        self.assertEqual(1, len(list(args.preparation_cache_dir.glob("*/preparation.json"))))
        metadata = args._preparation_provenance[(program.program_id, program.input_profile)]
        manifest = json.loads(Path(metadata["manifest_file"]).read_text())
        self.assertTrue(metadata["cache_hit"])
        self.assertFalse(manifest["contains_runtime_or_load_observations"])
        self.assertNotIn("run_success", manifest["payload"])

    def test_header_profile_toolchain_and_flags_invalidate_key(self):
        fake, program, args, produce = self.setup_cache()
        with mock.patch.object(cache.shutil, "which", return_value=sys.executable):
            first, _ = cache.signature(fake, program, args)
            program.source_path.with_suffix(".h").write_text("#define N 20\n")
            header, _ = cache.signature(fake, program, args)
            profile, _ = cache.signature(fake, core.Program(program.source_path, program.source_root, "SMALL_DATASET"), args)
            fake.compiler_version = lambda c: "fixture compiler 2"
            compiler, _ = cache.signature(fake, program, args)
            args.opt_level = "O1"
            optimize, _ = cache.signature(fake, program, args)
        self.assertEqual(5, len({first, header, profile, compiler, optimize}))

    def test_corrupt_artifact_is_rejected_not_silently_rebuilt(self):
        fake, program, args, produce = self.setup_cache()
        with mock.patch.object(cache.shutil, "which", return_value=sys.executable):
            payload = cache.get_or_prepare(fake, program, args, produce)
            payload["exe_path"].write_text("corrupt")
            with self.assertRaisesRegex(RuntimeError, "missing/changed"):
                cache.get_or_prepare(fake, program, args, lambda: self.fail("Corrupt entry must not be rebuilt silently"))

    def test_failed_preparation_is_not_published_and_globals_are_restored(self):
        fake, program, args, produce = self.setup_cache()
        previous = fake.BUILD_DIR
        def failed():
            result = produce()
            result["build_success"] = False
            return result
        with mock.patch.object(cache.shutil, "which", return_value=sys.executable):
            with self.assertRaisesRegex(RuntimeError, "Preparation failed"):
                cache.get_or_prepare(fake, program, args, failed)
        self.assertEqual(previous, fake.BUILD_DIR)
        self.assertFalse(list(args.preparation_cache_dir.glob("*/preparation.json")))

    def test_runtime_fields_cannot_be_saved_in_preparation_payload(self):
        fake, program, args, produce = self.setup_cache()
        def invalid():
            return {**produce(), "runtime_sec_median": 1.}
        with mock.patch.object(cache.shutil, "which", return_value=sys.executable):
            with self.assertRaisesRegex(ValueError, "pre-execution"):
                cache.get_or_prepare(fake, program, args, invalid)

    def test_no_cache_preserves_direct_preparation(self):
        producer = mock.Mock(return_value={"legacy": True})
        self.assertEqual({"legacy": True}, cache.get_or_prepare(None, None, argparse.Namespace(), producer))
        producer.assert_called_once()

    def test_pipeline_reuses_preparation_without_reusing_any_run_rows(self):
        fake, program, _, produce = self.setup_cache()
        args = core.build_arg_parser().parse_args(["--preparation-cache-dir", str(self.root / "pipeline-cache"), "--no-run"])
        args.env_id = "MOCK_ONLY"
        def prepare(p, a):
            for name in ["AST_DIR", "IR_DIR", "CFG_DIR", "BUILD_DIR"]:
                setattr(fake, name, getattr(core, name))
            return produce()
        with mock.patch.object(core, "prepare_program", side_effect=prepare) as preparation, \
             mock.patch.object(core, "compiler_version", return_value="fixture compiler"), \
             mock.patch.object(core, "llvm_opt_version", return_value="fixture opt"), \
             mock.patch.object(core, "collect_static_environment", return_value={}), \
             mock.patch.object(cache.shutil, "which", return_value=sys.executable), \
             mock.patch.object(core, "run_monitored_command") as native:
            outputs = [core.process_program(program, args) for _ in range(4)]
        native.assert_not_called()
        preparation.assert_called_once()
        self.assertTrue(all(out[0] == outputs[0][0] for out in outputs))
        self.assertTrue(all(out[1] == [] for out in outputs))


class FastProtocolTests(fixtures.Fixture):
    def test_plan_is_explicit_bounded_and_shares_six_preparations_for_24_trials(self):
        args = runner.build_parser().parse_args(["--protocol", "fast-v2"])
        with mock.patch.object(runner.subprocess, "Popen") as spawn:
            plan = runner.create_manifest(args, self.root)
        spawn.assert_not_called()
        self.assertEqual(24, len(plan["trials"]))
        self.assertEqual(6, plan["preparation_cache"]["expected_unique_preparations"])
        self.assertEqual(2, args.min_measured_runs)
        self.assertEqual(10, args.max_measured_runs)
        self.assertEqual(3, args.measurement_seconds)
        self.assertEqual(6, args.context_seconds)
        caches = set()
        for trial in plan["trials"]:
            config = load.read_config(trial["config_path"])
            self.assertEqual("fast-v2", config["collection_protocol"])
            self.assertEqual(3, config["min_raw_scrapes"])
            self.assertEqual(6, config["stable_window_seconds"])
            command = trial["command"]
            caches.add(command[command.index("--preparation-cache-dir") + 1])
        self.assertEqual(1, len(caches))

    def test_strict_pilot_cannot_unlock_fast_batch(self):
        args = runner.build_parser().parse_args(["--stage", "batch", "--protocol", "fast-v2"])
        args.pilot_validation = self.root / "pilot.json"
        report = {"status": "PASS", "stage": "pilot", "dataset": "polybench", "hostname": runner.socket.gethostname(),
                  "cpu_four_levels_validated": True, "temperature_policies": ["abort"]}
        load.write_json(args.pilot_validation, report)
        with self.assertRaisesRegex(ValueError, "Batch protocol"):
            runner.check_batch_gate(args)
        report["collection_protocols"] = ["fast-v2"]
        load.write_json(args.pilot_validation, report)
        runner.check_batch_gate(args)

    def test_other_datasets_do_not_silently_use_polybench_cache(self):
        args = runner.build_parser().parse_args(["--dataset", "cuda", "--protocol", "fast-v2"])
        with self.assertRaisesRegex(ValueError, "PolyBench"):
            runner.apply_protocol_defaults(args)

    def test_fast_boundaries_reject_invalid_policy_and_mismatched_pipeline_args(self):
        for fields in [{"collection_protocol": "fast-v2", "minimum_measured_runs": 1},
                       {"collection_protocol": "fast-v2", "minimum_measured_runs": 2, "maximum_measured_runs": 1},
                       {"collection_protocol": "strict-v1", "minimum_measured_runs": 2}]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                load.read_config(self.config(**fields))
        path = self.config(collection_protocol="fast-v2", minimum_measured_runs=2, maximum_measured_runs=10)
        args = self.args(path)
        args.min_measured_runs, args.max_measured_runs = 3, 10
        with self.assertRaisesRegex(ValueError, "must match"):
            load.validate_args(args, self.root / "fresh")

    def test_short_during_becomes_explicit_missing_but_query_failures_do_not(self):
        args = self.args(self.config(collection_protocol="fast-v2", minimum_measured_runs=2, maximum_measured_runs=10, min_raw_scrapes=3))
        session = load.LoadSession(args, core, {})
        session.interval = 1
        session.monitor = {"selector": "{}"}
        session.record["monitoring_preflight"] = {"scrape_interval_seconds": 1}
        session.record["phase_windows"] = {"during": (10, 10.01)}
        with mock.patch.object(load, "fresh_scrape_times", return_value={"cpu": [], "memory": []}):
            with self.assertRaises(load.InsufficientScrapes):
                session.validate_phase("during", 10, 10.01)
        self.assertEqual("MISSING_SHORT_WINDOW", session.record["phase_quality"]["during"]["status"])
        with mock.patch.object(load, "fresh_scrape_times", side_effect=OSError("Prometheus disconnected")):
            with self.assertRaises(OSError):
                session.validate_phase("during", 10, 10.01)
        self.assertEqual("FAIL", session.record["phase_quality"]["during"]["status"])


class BoundedMeasurementTests(fixtures.Fixture):
    def measured(self, duration):
        clock = fixtures.FakeClock()
        guard = fixtures.FakeGuard(clock)
        def native(*args, **kwargs):
            clock.sleep(duration)
            return subprocess.CompletedProcess(["toy"], 0, "", ""), duration, False, {}
        with mock.patch.object(core.time, "time", side_effect=clock.now), mock.patch.object(core.time, "perf_counter", side_effect=clock.now), \
             mock.patch.object(core.time, "sleep", side_effect=clock.sleep), mock.patch.object(core, "run_monitored_command", side_effect=native):
            return core.run_program(Path("toy"), 1, 5, 300, "fixture", 3, 2, 6, guard, 10)

    def test_long_runs_stop_at_two_and_median_is_mean_of_two(self):
        records, summary, success, error, windows = self.measured(10)
        self.assertTrue(success)
        self.assertEqual(2, sum(r["phase"] == "measure" for r in records))
        self.assertEqual(20, windows["during"][1] - windows["during"][0])
        self.assertEqual(10, summary["runtime_sec_median"])

    def test_short_runs_stop_at_ten_and_do_not_pad_during_with_idle(self):
        records, summary, success, error, windows = self.measured(.001)
        self.assertEqual(10, sum(r["phase"] == "measure" for r in records))
        self.assertAlmostEqual(.01, windows["during"][1] - windows["during"][0])
        self.assertEqual(.001, summary["runtime_sec_median"])

    def test_two_observation_median_uses_both_values_not_one_fast_run(self):
        self.assertEqual(12., core.summarize([10., 14.])["runtime_sec_median"])

    def test_intermediate_runs_stop_after_budget_not_after_ten(self):
        records, summary, success, error, windows = self.measured(.6)
        self.assertLessEqual(sum(r["phase"] == "measure" for r in records), 6)
        self.assertGreaterEqual(windows["during"][1] - windows["during"][0], 3)


class FastValidationTests(fixtures.Fixture):
    def make_trial(self, short=False):
        trial, record, sidecar = fixtures.ResultValidationTests.make_trial(self)
        config = {**record["config"], "collection_protocol": "fast-v2", "minimum_measured_runs": 2,
                  "maximum_measured_runs": 10, "min_raw_scrapes": 3}
        record["config"] = config
        load.write_json(trial["config_path"], config)
        trial["config_sha256"] = hashlib.sha256(Path(trial["config_path"]).read_bytes()).hexdigest()
        self.rewrite(trial, record, sidecar, short)
        return trial, record, sidecar

    def rewrite(self, trial, record, sidecar, short=False):
        source = Path(record["outputs"]["summary_file"])
        row = checker.rows(source)[0]
        details = checker.rows(record["outputs"]["runs_file"])[:2]
        details[0]["runtime_sec"], details[1]["runtime_sec"] = "1", "3"
        ids = [r["run_id"] for r in details]
        record["outputs"]["run_id_count"] = 2
        record["outputs"]["run_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        row["measured_runs"] = 2
        if short:
            details[0]["runtime_sec"], details[1]["runtime_sec"] = "0.001", "0.003"
            row["runtime_sec_median"], row["runtime_sec_mean"] = .002, .002
            record["phase_windows"]["during"] = (17, 17.01)
            record["phase_windows"]["after"] = (17.01, 23.01)
            record["phase_quality"]["during"] = {"status": "MISSING_SHORT_WINDOW", "distinct_raw_scrapes": {"cpu": 0, "memory": 0},
                                                   "scrape_timestamps": {"cpu": [], "memory": []}}
            timestamps = [17.01 + i for i in range(6)]
            record["phase_quality"]["after"]["scrape_timestamps"] = {"cpu": timestamps, "memory": timestamps}
            for phase in ["during", "after"]:
                for edge, stamp in zip(["start", "end"], record["phase_windows"][phase]):
                    row[f"prometheus_{phase}_window_{edge}_timestamp"] = stamp
            row["prometheus_during_collection_success"] = False
            row["prometheus_three_phase_collection_success"] = False
            for name in ["cpu_usage_pct_mean", "cpu_usage_pct_max", "memory_used_bytes_mean", "memory_used_bytes_max",
                         "disk_read_bytes_delta", "disk_write_bytes_delta", "network_bytes_delta"]:
                prefix, suffix = name.rsplit("_", 1)
                row[f"host_{prefix}_during_{suffix}"] = ""
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        with Path(record["outputs"]["runs_file"]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
        load.write_json(sidecar, record)

    def test_two_valid_measurements_are_accepted_only_with_explicit_fast_protocol(self):
        trial, record, sidecar = self.make_trial()
        self.assertEqual(2, checker.validate_trial(trial)["measured_runs"])
        record["config"]["minimum_measured_runs"] = 3
        record["config"]["collection_protocol"] = "strict-v1"
        load.write_json(trial["config_path"], record["config"])
        trial["config_sha256"] = hashlib.sha256(Path(trial["config_path"]).read_bytes()).hexdigest()
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "Too few"):
            checker.validate_trial(trial)

    def test_short_during_missing_is_accepted_but_not_called_complete(self):
        trial, record, sidecar = self.make_trial(short=True)
        validated = checker.validate_trial(trial)
        self.assertEqual("MISSING_SHORT_WINDOW", validated["during_monitoring_status"])
        self.assertEqual("fast-v2", validated["collection_protocol"])

    def test_finalize_and_full_report_accept_short_window_with_explicit_incompleteness(self):
        trial, record, sidecar = self.make_trial(short=True)
        row = checker.rows(record["outputs"]["summary_file"])[0]
        for key in list(row):
            if key.endswith("success"):
                row[key] = checker.true(row[key])
        row["host_cpu_usage_pct_before_mean"] = 25.
        details = checker.rows(record["outputs"]["runs_file"])
        for run in details:
            run["success"] = True
            run.update(record["identity"])
        args = self.args(Path(trial["config_path"]))
        session = load.LoadSession(args, core, record["identity"])
        session.record, session.path = record, sidecar
        args._load_sessions = [session]
        load.finalize(args, record["outputs"]["summary_file"], record["outputs"]["runs_file"], [row], details)
        self.assertEqual("PASS", session.record["status"])
        self.assertEqual("SHORT_DURING_MISSING", session.record["monitoring_acceptance"])
        manifest = self.root / "manifest.json"
        load.write_json(manifest, {"status": "VALIDATING", "stage": "batch", "dataset": "polybench",
                                  "trials": [trial], "load_families": ["cpu"]})
        report = checker.validate(manifest)
        self.assertEqual("PASS", report["status"])
        self.assertEqual(0, report["complete_three_phase_configurations"])
        self.assertEqual(1, report["short_during_missing_configurations"])
        self.assertFalse(row["prometheus_three_phase_collection_success"])

    def test_runtime_sum_must_fit_the_recorded_window(self):
        trial, record, sidecar = self.make_trial(short=True)
        record["phase_windows"]["during"] = (17, 17.00001)
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            checker.validate_trial(trial)

    def test_measurements_above_recorded_cap_are_rejected(self):
        trial, record, sidecar = self.make_trial()
        details = checker.rows(record["outputs"]["runs_file"])
        details = [{**details[0], "run_id": f"fixture-{i}", "runtime_sec": ".001"} for i in range(11)]
        path = Path(record["outputs"]["runs_file"])
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
        ids = [r["run_id"] for r in details]
        record["outputs"]["run_id_count"] = 11
        record["outputs"]["run_ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        summary = Path(record["outputs"]["summary_file"])
        row = checker.rows(summary)[0]
        row["measured_runs"] = 11
        with summary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            checker.validate_trial(trial)

    def test_missing_before_is_not_relaxed(self):
        trial, record, sidecar = self.make_trial(short=True)
        record["phase_quality"]["before"]["status"] = "FAIL"
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "Collection failed"):
            checker.validate_trial(trial)

    def test_validator_verifies_cache_artifacts_and_program_input_identity(self):
        trial, record, sidecar = self.make_trial()
        fake, program, args, produce = CacheTests.setup_cache(self)
        program = core.Program(program.source_path, program.source_root, "SIZE_01")
        with mock.patch.object(cache.shutil, "which", return_value=sys.executable):
            payload = cache.get_or_prepare(fake, program, args, produce)
        record["preparation"] = args._preparation_provenance[(program.program_id, program.input_profile)]
        record["identity"]["program_id"] = program.program_id
        path = Path(record["outputs"]["summary_file"])
        row = checker.rows(path)[0]
        row["program_id"] = program.program_id
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        load.write_json(sidecar, record)
        self.assertEqual("toy.c", checker.validate_trial(trial)["program_id"])
        payload["exe_path"].write_text("changed executable")
        with self.assertRaisesRegex(RuntimeError, "missing/changed"):
            checker.validate_trial(trial)
        manifest = self.root / "manifest.json"
        load.write_json(manifest, {"status": "VALIDATING", "stage": "batch", "dataset": "polybench",
                                  "trials": [trial], "load_families": ["cpu"]})
        report = checker.validate(manifest)
        self.assertEqual("FAIL", report["status"])
        self.assertTrue(any("missing/changed" in error for error in report["errors"]))

    def test_fake_zero_filling_is_rejected(self):
        trial, record, sidecar = self.make_trial(short=True)
        path = Path(record["outputs"]["summary_file"])
        row = checker.rows(path)[0]
        row["host_cpu_usage_pct_during_mean"] = 0
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        with self.assertRaisesRegex(ValueError, "Collection failed"):
            checker.validate_trial(trial)

    def test_long_window_missing_is_not_relaxed(self):
        trial, record, sidecar = self.make_trial(short=True)
        record["phase_windows"]["during"] = (17, 27)
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "Collection failed"):
            checker.validate_trial(trial)
