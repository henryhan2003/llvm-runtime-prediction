from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resource_load as load
import run_load_experiments as runner
import check_load_experiment as checker
from test_pipeline_cfg import pipeline as core
from test_pipeline_rodinia import pipeline as omp
from test_pipeline_rodinia_cuda import pipeline as cuda


class Fixture(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "tmp"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent, prefix="load-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, **fields):
        path = self.root / "config.json"
        path.write_text(json.dumps(fields), encoding="utf-8")
        return path

    def args(self, config=None):
        return argparse.Namespace(load_config=config, load_record_dir=self.root / "records", _load_results_dir=self.root / "results",
            _load_sessions=[], prometheus_exporter="node", prometheus_job="node", prometheus_instance="127.0.0.1:9100",
            prometheus_url="http://127.0.0.1:9090", prometheus_query_step=1., prometheus_query_timeout=1.,
            prometheus_context_seconds=12.)


class ConfigAndSafetyTests(Fixture):
    def test_temperature_peak_records_source_without_assuming_it_is_cpu(self):
        cpu, disk = self.root / "cpu", self.root / "disk"
        for directory, device, label, value in [(cpu, "k10temp", "Tdie", "70000"), (disk, "nvme", "Composite", "85500")]:
            directory.mkdir()
            (directory / "name").write_text(device)
            (directory / "temp1_label").write_text(label)
            (directory / "temp1_input").write_text(value)
        def paths(path, pattern):
            return [cpu / "temp1_input", disk / "temp1_input"] if path.as_posix() == "/sys/class/hwmon" else []
        with mock.patch.object(load, "memory_info", return_value={"MemAvailable": 1000}), \
             mock.patch.object(Path, "glob", autospec=True, side_effect=paths):
            sample = load.safety_snapshot()
        self.assertEqual(85.5, sample["temperature_max_celsius"])
        self.assertEqual({"device": "nvme", "label": "Composite", "path": str(disk / "temp1_input")}, sample["temperature_max_source"])

    def test_missing_temperature_is_not_fabricated_as_zero(self):
        with mock.patch.object(load, "memory_info", return_value={"MemAvailable": 1000}), mock.patch.object(Path, "glob", return_value=[]):
            sample = load.safety_snapshot()
        self.assertIsNone(sample["temperature_max_celsius"])
        self.assertIsNone(sample["temperature_max_source"])

    def test_config_rejects_unsafe_mixed_unknown_and_nonfinite_controls(self):
        bad = [{"target_cpu_fraction": .8, "load_type": "cpu"}, {"max_cpu_fraction": 1},
            {"load_type": "cpu", "target_cpu_fraction": .25, "memory_budget_fraction": .5},
            {"permission_confirmed": "false"}, {"target_cpu_fraction": float("nan")},
            {"new_cpu_feature": .2}, {"min_raw_scrapes": 1}, {"max_duration_seconds": 99999},
            {"temperature_policy": "off"}]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                load.read_config(self.config(**fields))

    def test_temperature_policy_defaults_to_abort_and_record_only_is_explicit(self):
        self.assertEqual("abort", load.read_config(self.config())["temperature_policy"])
        session = load.LoadSession(self.args(self.config()), core, {})
        with self.assertRaisesRegex(RuntimeError, "temperature cap"):
            session.enforce_temperature_limit({"temperature_max_celsius": 85.5})
        session = load.LoadSession(self.args(self.config(temperature_policy="record-only")), core, {})
        session.enforce_temperature_limit({"temperature_max_celsius": 85.5})
        self.assertEqual(85, session.config["max_temperature_celsius"])

    def test_memory_budget_is_capacity_and_reserve_bounded(self):
        config = load.read_config(self.config(load_type="memory", memory_budget_fraction=.75))
        memory = {"MemTotal": 64 * load.GIB, "EffectiveTotal": 64 * load.GIB, "MemAvailable": 20 * load.GIB}
        amount, reserve, budget = load.allocation_bytes(config, memory)
        self.assertLessEqual(amount, 6 * load.GIB)
        self.assertGreaterEqual(memory["MemAvailable"] - amount, reserve)
        self.assertLessEqual(budget, 8 * load.GIB)
        memory["MemAvailable"] = load.GIB
        with self.assertRaises(RuntimeError):
            load.allocation_bytes(config, memory)

    def test_nested_cpu_quota_uses_smallest_ancestor_limit(self):
        parent, child = self.root / "parent", self.root / "child"
        parent.mkdir()
        child.mkdir()
        (parent / "cpu.max").write_text("800000 100000")
        (child / "cpu.max").write_text("250000 100000")
        with mock.patch.object(load, "cgroup_directories", return_value=[(parent, True), (child, True)]):
            self.assertEqual(2.5, load.cpu_quota_cores())

    def test_cgroup_memory_headroom_overrides_large_host_free_memory(self):
        directory = self.root / "cgroup"
        directory.mkdir()
        (directory / "memory.max").write_text(str(2 * load.GIB))
        (directory / "memory.current").write_text(str(int(1.5 * load.GIB)))
        original_read = Path.read_text
        def read(path, *args, **kwargs):
            if path.as_posix() == "/proc/meminfo":
                return "MemTotal: 67108864 kB\nMemAvailable: 52428800 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
            return original_read(path, *args, **kwargs)
        with mock.patch.object(load, "cgroup_directories", return_value=[(directory, True)]), \
             mock.patch.object(Path, "read_text", autospec=True, side_effect=read):
            memory = load.memory_info()
        self.assertEqual(2 * load.GIB, memory["EffectiveTotal"])
        self.assertEqual(int(.5 * load.GIB), memory["MemAvailable"])

    def test_duration_parser_and_explicit_target_required(self):
        self.assertEqual(90, load.duration_seconds("1m30s"))
        self.assertEqual(.9, load.duration_seconds("900ms"))
        for value in ["0oops", "nan", "-1s"]:
            with self.assertRaises(ValueError):
                load.duration_seconds(value)
        args = self.args(self.config())
        args.prometheus_instance = ""
        with self.assertRaises(ValueError):
            load.validate_args(args, self.root / "fresh")

    def test_old_mode_does_not_create_load_or_extra_monitoring(self):
        args = self.args()
        with mock.patch.object(load.subprocess, "Popen") as process:
            load.validate_args(args, self.root)
            with load.session(args, core, {}) as controller:
                self.assertIsNone(controller)
            load.finalize(args, "unused.csv", "unused-runs.csv", [], [])
        process.assert_not_called()

    def test_permission_failure_writes_sidecar_and_never_spawns(self):
        session = load.LoadSession(self.args(self.config()), core, {"program_id": "toy", "input_id": "MINI_DATASET"})
        with mock.patch.object(load.platform, "system", return_value="Linux"), mock.patch.object(load, "safety_snapshot", return_value={}), \
             mock.patch.object(load.subprocess, "Popen") as process:
            with self.assertRaisesRegex(RuntimeError, "permission"):
                session.__enter__()
        process.assert_not_called()
        saved = json.loads(session.path.read_text())
        self.assertEqual("FAIL", saved["status"])
        self.assertTrue(saved["cleanup_success"])

    def test_linux_requirement_fails_without_spawning(self):
        session = load.LoadSession(self.args(self.config(permission_confirmed=True)), core, {})
        with mock.patch.object(load.platform, "system", return_value="Windows"), mock.patch.object(load.subprocess, "Popen") as process:
            with self.assertRaisesRegex(RuntimeError, "Linux"):
                session.__enter__()
        process.assert_not_called()

    def test_refuses_overwriting_existing_result_csv(self):
        (self.root / "old.csv").write_text("old")
        with self.assertRaisesRegex(ValueError, "fresh"):
            load.validate_args(self.args(self.config()), self.root)

    def test_watchdog_aborts_temperature_and_cleans_only_owned_process(self):
        session = load.LoadSession(self.args(self.config()), core, {})
        session.reserve = 10
        session.record["memory_before"] = {"SwapTotal": 0, "SwapFree": 0}
        sample = {"memory_available_bytes": 100, "temperature_max_celsius": 90}
        with mock.patch.object(session.stop_event, "wait", return_value=False), \
             mock.patch.object(load, "safety_snapshot", return_value=sample), mock.patch.object(session, "stop_owned_process") as cleanup:
            session.watchdog()
        self.assertIn("temperature", session.error)
        cleanup.assert_called_once()

    def test_record_only_watchdog_retains_hot_sample_without_temperature_abort(self):
        session = load.LoadSession(self.args(self.config(temperature_policy="record-only")), core, {})
        session.reserve = 10
        session.record["memory_before"] = {"SwapTotal": 100, "SwapFree": 100}
        sample = {"memory_available_bytes": 100, "temperature_max_celsius": 90}
        with mock.patch.object(session.stop_event, "wait", side_effect=[False, True]), \
             mock.patch.object(load, "safety_snapshot", return_value=sample), \
             mock.patch.object(load, "memory_info", return_value={"SwapTotal": 100, "SwapFree": 100}), \
             mock.patch.object(session, "stop_owned_process") as cleanup:
            session.watchdog()
        self.assertEqual("", session.error)
        self.assertEqual([sample], session.record["safety_samples"])
        cleanup.assert_not_called()

    def test_record_only_does_not_disable_memory_swap_or_duration_guards(self):
        for case in ["memory", "swap", "duration"]:
            with self.subTest(case=case):
                session = load.LoadSession(self.args(self.config(temperature_policy="record-only")), core, {})
                session.reserve = 10
                session.record["memory_before"] = {"SwapTotal": load.GIB, "SwapFree": load.GIB}
                sample = {"memory_available_bytes": 5 if case == "memory" else 100, "temperature_max_celsius": 90}
                if case == "duration":
                    session.started = 0
                memory = {"SwapTotal": load.GIB, "SwapFree": 0 if case == "swap" else load.GIB}
                with mock.patch.object(session.stop_event, "wait", return_value=False), \
                     mock.patch.object(load, "safety_snapshot", return_value=sample), \
                     mock.patch.object(load, "memory_info", return_value=memory), \
                     mock.patch.object(load.time, "monotonic", return_value=5000), \
                     mock.patch.object(session, "stop_owned_process") as cleanup:
                    session.watchdog()
                self.assertIn({"memory": "memory", "swap": "swap", "duration": "duration"}[case], session.error)
                cleanup.assert_called_once()

    def test_cleanup_signals_exact_owned_group_not_process_names(self):
        session = load.LoadSession(self.args(self.config()), core, {})
        session.process = mock.Mock(pid=876543)
        session.process.poll.return_value = -15
        with mock.patch.object(load.os, "killpg", create=True) as kill, mock.patch.object(load.signal, "SIGKILL", 9, create=True):
            session.stop_owned_process()
        self.assertTrue(kill.called)
        self.assertTrue(all(call.args[0] == 876543 for call in kill.call_args_list))

    def test_successful_cpu_lifecycle_is_mocked_and_cleans_worker(self):
        args = self.args(self.config(load_type="cpu", target_cpu_fraction=.25, permission_confirmed=True))
        session = load.LoadSession(args, core, {"program_id": "toy", "input_id": "MINI_DATASET"})
        process = mock.Mock(pid=876543)
        process.poll.return_value = None
        memory = {"MemTotal": 64 * load.GIB, "EffectiveTotal": 64 * load.GIB,
                  "MemAvailable": 50 * load.GIB, "SwapTotal": 0, "SwapFree": 0}
        snapshot = {"memory_available_bytes": 50 * load.GIB, "temperature_max_celsius": 40}
        metrics = {"host_cpu_usage_pct_mean": 5., "host_memory_used_bytes_mean": 14 * load.GIB}
        fcntl = types.SimpleNamespace(flock=mock.Mock(), LOCK_EX=1, LOCK_NB=2)
        def stop():
            process.poll.return_value = -15
        with mock.patch.dict(sys.modules, {"fcntl": fcntl}), mock.patch.object(load.platform, "system", return_value="Linux"), \
             mock.patch.object(load, "monitoring_preflight", return_value={"scrape_interval_seconds": 1., "selector": "{}"}), \
             mock.patch.object(load, "memory_info", return_value=memory), mock.patch.object(load, "safety_snapshot", return_value=snapshot), \
             mock.patch.object(load, "cpu_quota_cores", return_value=None), \
             mock.patch.object(load.tempfile, "gettempdir", return_value=str(self.root)), \
             mock.patch.object(load.os, "open", return_value=987), mock.patch.object(load.os, "close"), \
             mock.patch.object(load.os, "sched_getaffinity", return_value=set(range(8)), create=True), mock.patch.object(load.os, "cpu_count", return_value=8), \
             mock.patch.object(load.shutil, "which", return_value="/usr/bin/stress-ng"), \
             mock.patch.object(load.subprocess, "Popen", return_value=process) as spawn, mock.patch.object(load.threading, "Thread"), \
             mock.patch.object(load.threading, "current_thread", return_value=object()), mock.patch.object(load.time, "sleep"), \
             mock.patch.object(session, "observation", return_value=(metrics, {"cpu": [1, 2, 3], "memory": [1, 2, 3]})), \
             mock.patch.object(session, "stabilize"), mock.patch.object(session, "stop_owned_process", side_effect=stop):
            with session as controller:
                now = controller.record["stable_at"]
                controller.record_windows({"before": (now, now), "during": (now, now), "after": (now, now)})
        self.assertEqual("MEASUREMENT_COMPLETED", session.record["status"])
        self.assertTrue(session.record["cleanup_success"])
        self.assertIn("--cpu-load", session.record["command"])
        self.assertTrue(spawn.call_args.kwargs["start_new_session"])
        self.assertIn("resource_load_supervisor.py", " ".join(spawn.call_args.args[0]))


class SamplingTests(Fixture):
    def test_lookback_evaluation_points_are_not_counted_as_fresh_scrapes(self):
        args = self.args()
        fake = types.SimpleNamespace(parsed_series_values=core.parsed_series_values)
        fake.prometheus_query_range = mock.Mock(return_value=[{"metric": {}, "values": [[100, "90"], [101, "90"], [102, "90"]]}])
        result = load.fresh_scrape_times(fake, args, "{}", 100, 103, 1)
        self.assertEqual({"cpu": [], "memory": []}, result)
        fake.prometheus_query_range.return_value = [{"metric": {}, "values": [[100, "100"], [101, "101"], [102, "101"]]}]
        result = load.fresh_scrape_times(fake, args, "{}", 100, 103, 1)
        self.assertEqual([100, 101], result["cpu"])

    def test_guard_marks_short_during_window_missing_not_zero(self):
        windows = {"before": (1., 10.), "during": (10., 10.1), "after": (10.1, 20.)}
        def guard(phase, start, end):
            if phase == "during":
                raise RuntimeError("Insufficient distinct raw scrapes")
        result = core.empty_prometheus_result(1, 10, "")
        result.update(prometheus_collection_success=True, host_cpu_usage_pct_mean=25., host_cpu_usage_pct_max=26.,
                      host_memory_used_bytes_mean=100., host_memory_used_bytes_max=101.)
        with mock.patch.object(core, "collect_prometheus_host_metrics", return_value=result):
            summary = core.collect_prometheus_three_phase_metrics("http://localhost", windows, 1, 1, {}, 9, guard)
        self.assertFalse(summary["prometheus_during_collection_success"])
        self.assertFalse(summary["prometheus_three_phase_collection_success"])
        self.assertEqual("", summary["host_cpu_usage_pct_during_mean"])
        self.assertEqual("", summary["host_disk_read_bytes_during_delta"])
        self.assertEqual(25., summary["host_cpu_usage_pct_before_mean"])

    def test_phase_order_must_be_after_stabilization(self):
        session = load.LoadSession(self.args(self.config()), core, {})
        session.record["stable_at"] = 100.
        with self.assertRaisesRegex(RuntimeError, "ordering"):
            session.record_windows({"before": (90., 99.), "during": (101., 102.), "after": (102., 103.)})


class FakeClock:
    def __init__(self):
        self.value = 100.

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeGuard:
    def __init__(self, clock):
        self.clock = clock
        self.events = []
        self.windows = None

    def check(self):
        pass

    def before_window(self, seconds):
        start = self.clock.now()
        self.events.append("before")
        self.clock.sleep(seconds)
        return start, self.clock.now()

    def before_invocation(self, phase, index):
        self.events.append(phase)

    def record_windows(self, windows):
        self.windows = windows


class PipelineIntegrationTests(Fixture):
    def test_polybench_load_before_warmup_and_control_time_not_in_runtime(self):
        clock = FakeClock()
        guard = FakeGuard(clock)
        def native(*args, **kwargs):
            clock.sleep(.25)
            return subprocess.CompletedProcess(["toy"], 0, "", ""), .25, False, {}
        with mock.patch.object(core.time, "time", side_effect=clock.now), mock.patch.object(core.time, "perf_counter", side_effect=clock.now), \
             mock.patch.object(core.time, "sleep", side_effect=clock.sleep), mock.patch.object(core, "run_monitored_command", side_effect=native):
            records, summary, success, error, windows = core.run_program(Path("toy"), 1, 3, 10, "test", 0, 3, 12, guard)
        self.assertTrue(success)
        self.assertEqual(["before", "warmup", "measure", "measure", "measure"], guard.events)
        self.assertEqual(.25, summary["runtime_sec_median"])
        self.assertEqual(3, len([r for r in records if r["phase"] == "measure"]))
        self.assertEqual((100., 112.), windows["before"])
        self.assertEqual(.75, windows["during"][1] - windows["during"][0])
        self.assertEqual(windows, guard.windows)

    def test_shared_openmp_cuda_measurement_preserves_timing_and_phase_scope(self):
        clock = FakeClock()
        guard = FakeGuard(clock)
        spec = omp.BenchmarkSpec("toy", "toy", "toy", (), (), (), (), (), (), (), {}, (), True, "")
        item = omp.InputSpec("SIZE_01", "SIZE_01", (), 1, {}, (), {})
        def native(*args, **kwargs):
            clock.sleep(.25)
            return omp.ProcessResult(["toy"], 0, .25, False, "", {})
        windows = {}
        with mock.patch.object(omp.time, "time", side_effect=clock.now), mock.patch.object(omp.time, "sleep", side_effect=clock.sleep), \
             mock.patch.object(omp, "run_process", side_effect=native):
            summary, details, errors = omp.measure_input("session", spec, item, self.root, 1, 3, 0, 3, 10, windows, 12, guard)
        self.assertFalse(errors)
        self.assertEqual(.75, summary["measurement_seconds_actual"])
        self.assertEqual(.25, summary["runtime_sec_median"])
        self.assertEqual((100., 112.), windows["before"])
        self.assertEqual(["before", "warmup", "measure", "measure", "measure"], guard.events)
        self.assertEqual(windows, guard.windows)
        self.assertTrue(hasattr(cuda.base, "resource_load"))

    def test_polybench_interrupt_kills_and_reaps_target(self):
        process = mock.Mock()
        process.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with mock.patch.object(core, "ensure_process_probe", return_value=Path("probe")), \
             mock.patch.object(core.cpu_runtime, "probe_command", return_value=["probe"]), \
             mock.patch.object(core.subprocess, "Popen", return_value=process), mock.patch.object(core, "_kill_process_group") as kill:
            with self.assertRaises(KeyboardInterrupt):
                core.run_monitored_command(["toy"], timeout=10)
        kill.assert_called_once_with(process)
        self.assertEqual(2, process.communicate.call_count)

    def test_openmp_interrupt_kills_target_and_cancels_timeout(self):
        process = mock.Mock()
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with mock.patch.object(omp, "ensure_process_probe", return_value=Path("probe")), \
             mock.patch.object(omp.cpu_runtime, "probe_command", return_value=["probe"]), \
             mock.patch.object(omp.subprocess, "Popen", return_value=process), mock.patch.object(omp.threading, "Timer") as timer, \
             mock.patch.object(omp, "_kill_process_group") as kill:
            with self.assertRaises(KeyboardInterrupt):
                omp._run_measured_posix(["toy"], self.root, {}, 10, True, self.root / "native.log")
        kill.assert_called_once_with(process)
        timer.return_value.cancel.assert_called_once()

    def test_all_three_clis_have_opt_in_flags(self):
        self.assertIsNone(core.build_arg_parser().parse_args([]).load_config)
        self.assertIsNone(omp.parse_args(["--benchmark", "bfs"]).load_config)
        self.assertIsNone(cuda.parse_args(["--benchmark", "bfs"]).load_config)


class PlanningTests(Fixture):
    def test_execute_updates_global_state_before_trials_and_validation_without_real_load(self):
        root = self.root / "execution"
        path = root / "experiment_manifest.json"
        observed = []
        def run_trial(trial, timeout):
            manifest = json.loads(path.read_text())
            self.assertEqual("RUNNING", manifest["status"])
            current = next(t for t in manifest["trials"] if t["trial_id"] == trial["trial_id"])
            self.assertEqual("RUNNING", current["status"])
            observed.append(trial["trial_id"])
            return 0
        def validate(manifest_path):
            manifest = json.loads(Path(manifest_path).read_text())
            self.assertEqual("VALIDATING", manifest["status"])
            self.assertEqual(24, len(manifest["trials"]))
            self.assertTrue(all(t["status"] == "COMPLETED" and t["exit_code"] == 0 for t in manifest["trials"]))
            return {"status": "PASS"}
        with mock.patch.object(runner.platform, "system", return_value="Linux"), \
             mock.patch.object(load, "monitoring_preflight"), \
             mock.patch.object(runner.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 * load.GIB)), \
             mock.patch.object(runner, "run_trial", side_effect=run_trial), \
             mock.patch.object(checker, "validate", side_effect=validate), \
             mock.patch.object(runner.signal, "SIGHUP", 1, create=True), \
             mock.patch.object(runner.signal, "getsignal"), mock.patch.object(runner.signal, "signal"), \
             mock.patch.object(runner.subprocess, "Popen") as spawn:
            code = runner.main(["--execute", "--permission-confirmed", "--output-dir", str(root)])
        spawn.assert_not_called()
        self.assertEqual(0, code)
        self.assertEqual(24, len(observed))
        self.assertEqual("PASS", json.loads(path.read_text())["status"])

    def test_failed_trial_preserves_fail_state_and_never_reaches_validation(self):
        root = self.root / "failed-execution"
        with mock.patch.object(runner.platform, "system", return_value="Linux"), \
             mock.patch.object(load, "monitoring_preflight"), \
             mock.patch.object(runner.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10 * load.GIB)), \
             mock.patch.object(runner, "run_trial", return_value=1) as run, \
             mock.patch.object(checker, "validate") as validate, \
             mock.patch.object(runner.signal, "SIGHUP", 1, create=True), \
             mock.patch.object(runner.signal, "getsignal"), mock.patch.object(runner.signal, "signal"), \
             mock.patch.object(runner.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "Trial failed"):
                runner.main(["--execute", "--permission-confirmed", "--output-dir", str(root)])
        spawn.assert_not_called()
        validate.assert_not_called()
        run.assert_called_once()
        manifest = json.loads((root / "experiment_manifest.json").read_text())
        self.assertEqual("FAIL", manifest["status"])
        self.assertEqual("FAIL", manifest["trials"][0]["status"])
        self.assertTrue(all(t["status"] == "PLANNED" for t in manifest["trials"][1:]))

    def test_interrupted_trial_terminates_only_its_pipeline_group(self):
        process = mock.Mock(pid=765432)
        process.wait.side_effect = [KeyboardInterrupt(), 0]
        with mock.patch.object(runner.subprocess, "Popen", return_value=process), \
             mock.patch.object(runner.os, "killpg", create=True) as kill:
            with self.assertRaises(KeyboardInterrupt):
                runner.run_trial({"directory": str(self.root), "command": ["never-executed"]}, 10)
        kill.assert_called_once_with(765432, runner.signal.SIGTERM)

    def test_pilot_plan_has_24_unique_trials_and_unchanged_input_ids(self):
        args = runner.build_parser().parse_args([])
        plan = runner.create_manifest(args, self.root)
        self.assertEqual(24, len(plan["trials"]))
        self.assertEqual(24, len({t["trial_id"] for t in plan["trials"]}))
        self.assertEqual({"gemm", "jacobi-2d"}, {t["program"] for t in plan["trials"]})
        for trial in plan["trials"]:
            self.assertFalse(load.read_config(trial["config_path"])["permission_confirmed"])
            self.assertEqual("abort", load.read_config(trial["config_path"])["temperature_policy"])
            command = trial["command"]
            self.assertEqual("3", command[command.index("--min-measured-runs") + 1])
            self.assertEqual(str(ROOT / "datasets/PolyBenchC-4.2.1"), command[command.index("--source-dir") + 1])
            self.assertIn("--load-config", command)
        self.assertNotEqual([t["profile_id"] for t in plan["trials"][:4]], [t["profile_id"] for t in plan["trials"][4:8]])

    def test_record_only_plan_preserves_other_limits_and_never_starts_load(self):
        args = runner.build_parser().parse_args(["--temperature-policy", "record-only"])
        with mock.patch.object(runner.subprocess, "Popen") as spawn:
            plan = runner.create_manifest(args, self.root)
        spawn.assert_not_called()
        for trial in plan["trials"]:
            config = load.read_config(trial["config_path"])
            self.assertEqual("record-only", config["temperature_policy"])
            self.assertEqual(.75, config["max_cpu_fraction"])
            self.assertEqual(1200, config["max_duration_seconds"])
            self.assertFalse(config["permission_confirmed"])

    def test_batch_gate_requires_matching_temperature_policy(self):
        args = runner.build_parser().parse_args(["--stage", "batch", "--temperature-policy", "record-only"])
        args.pilot_validation = self.root / "validation.json"
        report = {"status": "PASS", "stage": "pilot", "dataset": "polybench",
                  "hostname": runner.socket.gethostname(), "cpu_four_levels_validated": True}
        load.write_json(args.pilot_validation, report)
        with self.assertRaisesRegex(ValueError, "temperature policy"):
            runner.check_batch_gate(args)
        report["temperature_policies"] = ["record-only"]
        load.write_json(args.pilot_validation, report)
        runner.check_batch_gate(args)

    def test_catalog_covers_all_twenty_sizes_without_executing(self):
        for dataset, count in [("polybench", 600), ("openmp", 380), ("cuda", 440)]:
            catalog = runner.workload_catalog(dataset, "batch", [])
            self.assertEqual(count, len(catalog))
            self.assertEqual({f"SIZE_{i:02d}" for i in range(1, 21)}, {row[1] for row in catalog})

    def test_batch_gate_cannot_use_missing_failed_or_other_host_pilot(self):
        args = runner.build_parser().parse_args(["--stage", "batch"])
        with self.assertRaises(ValueError):
            runner.check_batch_gate(args)
        args.pilot_validation = self.root / "validation.json"
        load.write_json(args.pilot_validation, {"status": "FAIL"})
        with self.assertRaises(ValueError):
            runner.check_batch_gate(args)
        load.write_json(args.pilot_validation, {"status": "PASS", "stage": "pilot", "dataset": "polybench", "hostname": "another-machine"})
        with self.assertRaises(ValueError):
            runner.check_batch_gate(args)

    def test_plan_only_manifest_is_not_a_passed_real_experiment(self):
        args = runner.build_parser().parse_args([])
        plan = runner.create_manifest(args, self.root)
        path = self.root / "manifest.json"
        load.write_json(path, plan)
        report = checker.validate(path)
        self.assertEqual("FAIL", report["status"])
        self.assertEqual(0, report["validated_configurations"])
        self.assertTrue(any("Plan-only" in error for error in report["errors"]))


class ResultValidationTests(Fixture):
    def make_trial(self):
        results = self.root / "results"
        sidecars = results / "load_records"
        sidecars.mkdir(parents=True)
        config_path = self.config(load_type="cpu", profile_id="CPU_LOW", target_cpu_fraction=.25, permission_confirmed=True)
        config = load.read_config(config_path)
        summary_path, runs_path = results / "summary.csv", results / "runs.csv"
        windows = {"before": (10., 16.), "during": (17., 23.), "after": (23., 29.)}
        summary = {"program_id": "toy", "input_id": "SIZE_01", "env_id": "MOCK_ONLY", "run_success": True,
            "prometheus_three_phase_collection_success": True, "measured_runs": 3, "runtime_sec_median": 2.,
            "runtime_sec_mean": 2., "host_cpu_usage_pct_before_mean": 25.}
        quality = {}
        for phase, (start, end) in windows.items():
            summary.update({f"prometheus_{phase}_window_start_timestamp": start,
                f"prometheus_{phase}_window_end_timestamp": end, f"prometheus_{phase}_collection_success": True})
            timestamps = [start + i for i in range(6)]
            quality[phase] = {"status": "PASS", "distinct_raw_scrapes": {"cpu": 6, "memory": 6},
                              "scrape_timestamps": {"cpu": timestamps, "memory": timestamps}}
        ids = ["fixture1", "fixture2", "fixture3"]
        with summary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        with runs_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["run_id", "phase", "success", "runtime_sec"])
            writer.writeheader()
            writer.writerows({"run_id": name, "phase": "measure", "success": True, "runtime_sec": value} for name, value in zip(ids, [1, 2, 3]))
        record = {"test_fixture_not_observed_experiment": True, "config": config, "status": "PASS", "cleanup_success": True,
            "before_load_control_validated": True, "identity": {"program_id": "toy", "input_id": "SIZE_01"},
            "outputs": {"summary_file": str(summary_path), "runs_file": str(runs_path), "run_id_count": 3,
                "run_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()},
            "phase_quality": quality, "phase_windows": windows, "stable_at": 9., "target_first_invocation_at": 16.5,
            "load_stopped_at": 30., "monitoring_preflight": {"scrape_interval_seconds": 1.}}
        sidecar = sidecars / "fixture.json"
        load.write_json(sidecar, record)
        trial = {"status": "COMPLETED", "config_path": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "results_dir": str(results), "profile_id": "CPU_LOW", "program": "toy", "input_profile": "SIZE_01",
            "trial_id": "mock-only", "load_family": "cpu"}
        return trial, record, sidecar

    def test_validator_accepts_consistent_mock_and_recalculates_median(self):
        trial, record, sidecar = self.make_trial()
        validated = checker.validate_trial(trial)
        self.assertEqual(2., validated["runtime_sec_median"])
        self.assertEqual(25., validated["host_cpu_usage_pct_before_mean"])

    def test_revalidate_historical_state_only_failure_without_changing_observations(self):
        trial, record, sidecar = self.make_trial()
        manifest_path = self.root / "manifest.json"
        manifest = {"status": "PLANNED", "stage": "batch", "dataset": "polybench",
                    "trials": [trial], "load_families": ["cpu"]}
        tracked = [sidecar, Path(trial["config_path"]),
                   Path(record["outputs"]["summary_file"]), Path(record["outputs"]["runs_file"])]
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
        load.write_json(manifest_path, manifest)
        first_report = checker.validate(manifest_path)
        self.assertEqual(["Plan-only manifest contains no real load experiment results"], first_report["errors"])
        self.assertEqual(1, first_report["validated_configurations"])
        # The historical launcher persisted FAIL after its first final validation.
        manifest["status"] = first_report["status"]
        load.write_json(manifest_path, manifest)
        before = manifest_path.read_bytes()
        with mock.patch.object(load.subprocess, "Popen") as spawn:
            report = checker.validate(manifest_path)
        spawn.assert_not_called()
        self.assertEqual("PASS", report["status"])
        self.assertEqual([], report["errors"])
        self.assertEqual(before, manifest_path.read_bytes())
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked})

    def test_validator_accepts_legacy_sidecar_without_explicit_temperature_policy(self):
        trial, record, sidecar = self.make_trial()
        for key in ["temperature_policy", "collection_protocol", "minimum_measured_runs", "maximum_measured_runs"]:
            record["config"].pop(key)
        load.write_json(sidecar, record)
        validated = checker.validate_trial(trial)
        self.assertEqual("abort", validated["temperature_policy"])
        self.assertEqual("", validated["temperature_peak_observed_celsius_ANALYSIS_ONLY"])

    def test_record_only_validation_reports_policy_without_certifying_thermal_safety(self):
        trial, record, sidecar = self.make_trial()
        record["config"]["temperature_policy"] = "record-only"
        record["safety_samples"] = [{"temperature_max_celsius": None}, {"temperature_max_celsius": 85.5}]
        load.write_json(sidecar, record)
        load.write_json(trial["config_path"], record["config"])
        trial["config_sha256"] = hashlib.sha256(Path(trial["config_path"]).read_bytes()).hexdigest()
        validated = checker.validate_trial(trial)
        self.assertEqual("record-only", validated["temperature_policy"])
        self.assertEqual(85.5, validated["temperature_peak_observed_celsius_ANALYSIS_ONLY"])
        manifest = self.root / "manifest.json"
        load.write_json(manifest, {"status": "COMPLETED", "stage": "batch", "dataset": "polybench",
                                  "trials": [trial], "load_families": ["cpu"]})
        report = checker.validate(manifest)
        self.assertEqual("PASS", report["status"])
        self.assertEqual(["record-only"], report["temperature_policies"])
        self.assertEqual(1, report["temperature_record_only_configurations"])
        self.assertFalse(report["thermal_safety_and_absence_of_throttling_validated"])

    def test_validator_rejects_load_cleanup_before_after_window(self):
        trial, record, sidecar = self.make_trial()
        record["load_stopped_at"] = 20.
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "released"):
            checker.validate_trial(trial)

    def test_validator_rejects_fake_sample_count_and_input_identity(self):
        trial, record, sidecar = self.make_trial()
        record["phase_quality"]["during"]["scrape_timestamps"]["cpu"] = [18.]
        load.write_json(sidecar, record)
        with self.assertRaisesRegex(ValueError, "counts"):
            checker.validate_trial(trial)
        trial["input_profile"] = "SIZE_20"
        with self.assertRaisesRegex(ValueError, "Manifest"):
            checker.validate_trial(trial)


if __name__ == "__main__":
    unittest.main()
