"""Plan or execute isolated load trials using the existing three pipelines.

Default mode only writes commands/configuration; it never creates pressure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import socket
import subprocess
import sys
import time

try:
    import resource_load as load
except ImportError:
    from scripts import resource_load as load

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workload_catalog(dataset, stage, selected):
    if dataset == "polybench":
        unified = json.loads((ROOT / "configs/input_scales_20.json").read_text(encoding="utf-8"))
        sources = {e["program_id"]: str(Path(e["header_path"]).with_suffix(".c")) for e in unified["entries"]
                   if e["dataset"] == "PolyBench/C"}
        ids = selected or (json.loads((ROOT / "configs/resource_load_profiles.json").read_text())["pilot_programs"]
                           if stage == "pilot" else sorted(sources))
        if any(name not in sources for name in ids):
            raise ValueError("Unknown PolyBench program")
        sizes = ["MINI_DATASET", "MEDIUM_DATASET", "EXTRALARGE_DATASET"] if stage == "pilot" else [f"SIZE_{i:02d}" for i in range(1, 21)]
        return [(name, size, {"source": sources[name], "all_sources": sorted(sources.values())}) for name in ids for size in sizes]
    if stage == "pilot":
        raise ValueError("The initial pilot is PolyBench only; validate it before Rodinia batch execution")
    manifest_path = ROOT / f"configs/rodinia_{dataset}_20sizes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmarks = [b for b in manifest["benchmarks"] if b.get("enabled", True)]
    if selected:
        if set(selected) - {b["id"] for b in benchmarks}:
            raise ValueError("Unknown or disabled Rodinia benchmark")
        benchmarks = [b for b in benchmarks if b["id"] in selected]
    return [(b["id"], item["id"], {"manifest": str(manifest_path.relative_to(ROOT))})
            for b in benchmarks for item in b["inputs"] if item["id"].startswith("SIZE_")]


def command_for(args, program, size, info, config_path, results):
    common = ["--warmup", str(args.warmup), "--measurement-seconds", str(args.measurement_seconds),
        "--min-measured-runs", "3", "--timeout", str(args.timeout), "--prometheus-exporter", "node",
        "--prometheus-url", args.prometheus_url, "--prometheus-job", args.prometheus_job,
        "--prometheus-instance", args.prometheus_instance, "--prometheus-context-seconds", str(args.context_seconds),
        "--prometheus-query-step", "1", "--prometheus-query-delay", "2", "--results-dir", str(results),
        "--load-config", str(config_path)]
    if args.dataset == "polybench":
        source_root = ROOT / "datasets/PolyBenchC-4.2.1"
        specific = ["--source-dir", str(source_root), "--dataset", "PolyBenchC-4.2.1",
            "--polybench-profile" if args.stage == "pilot" else "--polybench-size", size,
            "--prometheus-final-wait", "0"]
        # Keep the original full-root program IDs; do not switch --source-dir to a subdirectory.
        for path in info["all_sources"]:
            if path != info["source"]:
                specific += ["--exclude", (ROOT / path).relative_to(source_root).as_posix()]
        return [args.python, str(ROOT / "scripts/pipeline_new.py"), *common, *specific]
    specific = ["--manifest", str(ROOT / info["manifest"]), "--benchmark", program,
                "--input-profile", size, "--collect-static"]
    if args.dataset == "cuda":
        specific += ["--cuda-arch", args.cuda_arch, "--gpu-index", str(args.gpu_index)]
    return [args.python, str(ROOT / f"scripts/pipeline_rodinia{'_cuda' if args.dataset == 'cuda' else ''}.py"), *common, *specific]


def create_manifest(args, root):
    profiles_path = ROOT / "configs/resource_load_profiles.json"
    definitions = json.loads(profiles_path.read_text(encoding="utf-8"))
    workload = workload_catalog(args.dataset, args.stage, args.program)
    safety = {**definitions["safety"], "permission_confirmed": args.permission_confirmed,
              "temperature_policy": args.temperature_policy}
    if args.max_memory_gib is not None:
        safety["max_memory_bytes"] = int(args.max_memory_gib * load.GIB)
    trials = []
    for family in args.load_types:
        profiles = definitions[family + "_profiles"]
        for index, (program, size, info) in enumerate(workload):
            # Rotate/reverse level order to reduce systematic time-order/thermal confounding.
            order = profiles[index % 4:] + profiles[:index % 4]
            if index % 2:
                order = list(reversed(order))
            for profile in order:
                trial_id = f"{family}__{program}__{size}__{profile['profile_id']}"
                directory = root / trial_id
                directory.mkdir(parents=True)
                config_path = directory / "load_config.json"
                load.write_json(config_path, {**safety, **profile})
                load.read_config(config_path)
                results = directory / "results"
                trials.append({"trial_id": trial_id, "dataset": args.dataset, "load_family": family,
                    "program": program, "input_profile": size, "profile_id": profile["profile_id"],
                    "directory": str(directory), "config_path": str(config_path), "config_sha256": file_digest(config_path),
                    "results_dir": str(results), "command": command_for(args, program, size, info, config_path, results), "status": "PLANNED"})
    tracked = [profiles_path, ROOT / "configs/input_scales_20.json", ROOT / "scripts/resource_load.py",
               ROOT / "scripts/resource_load_memory_worker.py", ROOT / "scripts/resource_load_supervisor.py",
               ROOT / "scripts/pipeline_new.py", ROOT / "scripts/pipeline_rodinia.py", ROOT / "scripts/pipeline_rodinia_cuda.py",
               ROOT / "scripts/run_load_experiments.py", ROOT / "scripts/check_load_experiment.py"]
    return {"schema_version": 1, "stage": args.stage, "status": "PLANNED", "dataset": args.dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "execution_host": socket.gethostname() if args.execute else None,
        "trials": trials, "load_families": args.load_types, "prediction_features_changed": False,
        "scripts_and_config_sha256": {str(p): file_digest(p) for p in tracked},
        "note": "Control fields are experimental metadata only; pressure is created only with --execute and explicit permission."}


def check_batch_gate(args):
    if args.stage != "batch":
        return
    if args.pilot_validation is None:
        raise ValueError("Batch execution requires --pilot-validation from a successful real PolyBench load pilot")
    report = json.loads(args.pilot_validation.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("stage") != "pilot" or report.get("dataset") != "polybench":
        raise ValueError("Pilot validation has not passed")
    if report.get("hostname") != socket.gethostname() or not report.get("cpu_four_levels_validated"):
        raise ValueError("Pilot must validate all four CPU levels on this host")
    if set(report.get("temperature_policies", ["abort"])) != {args.temperature_policy}:
        raise ValueError("Batch temperature policy must match the validated pilot")
    if "memory" in args.load_types and not report.get("memory_four_levels_validated"):
        raise ValueError("Memory batch execution requires a validated four-level memory pilot")


def run_trial(trial, timeout):
    path = Path(trial["directory"]) / "pipeline.log"
    with path.open("wb") as log:
        process = subprocess.Popen(trial["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["polybench", "openmp", "cuda"], default="polybench")
    parser.add_argument("--stage", choices=["pilot", "batch"], default="pilot")
    parser.add_argument("--program", action="append", default=[])
    parser.add_argument("--load-types", nargs="+", choices=["cpu", "memory"], default=["cpu"])
    parser.add_argument("--execute", action="store_true", help="Actually run Linux benchmarks and pressure; default is plan-only")
    parser.add_argument("--preflight-only", action="store_true", help="Read-only Linux Prometheus/host check; creates no load")
    parser.add_argument("--permission-confirmed", action="store_true", help="Only set after obtaining shared-server authorization")
    parser.add_argument("--pilot-validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-memory-gib", type=float, default=None)
    parser.add_argument("--temperature-policy", choices=["abort", "record-only"], default="abort",
                        help="abort: retain the configured temperature gate (default); record-only: explicitly disable "
                             "script temperature aborts, retaining telemetry and all other guards. Does not change hardware protections.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measurement-seconds", type=float, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--trial-timeout", type=int, default=1800)
    parser.add_argument("--context-seconds", type=float, default=12)
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--prometheus-job", default="node")
    parser.add_argument("--prometheus-instance", default="127.0.0.1:9100")
    parser.add_argument("--prometheus-query-timeout", type=float, default=10)
    parser.add_argument("--cuda-arch", default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.temperature_policy == "record-only":
        print("[warning] Temperature policy: record-only. Script temperature aborts are disabled; "
              "completion is not proof of thermal safety or absence of throttling.", file=sys.stderr, flush=True)
    if args.warmup < 0 or not math_is_positive(args.context_seconds) or not math_is_positive(args.prometheus_query_timeout) or \
        not (math_is_positive(args.measurement_seconds) or args.measurement_seconds == 0) or args.timeout <= 0 or args.trial_timeout <= 0:
        parser.error("Invalid timing parameters")
    if len(set(args.load_types)) != len(args.load_types):
        parser.error("Duplicate load families")
    if args.max_memory_gib is not None and (not math_is_positive(args.max_memory_gib)):
        parser.error("--max-memory-gib must be finite and positive")
    if args.preflight_only:
        if args.execute or platform.system() != "Linux":
            parser.error("--preflight-only requires Linux and cannot be combined with --execute")
        result = load.monitoring_preflight(args)
        result.update(stress_ng_path=shutil.which("stress-ng"), memory=load.memory_info(),
                      logical_processors=os.cpu_count(), affinity_cpus=sorted(os.sched_getaffinity(0)),
                      safety=load.safety_snapshot(), synthetic_load_started=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.execute:
        if platform.system() != "Linux" or not args.permission_confirmed:
            parser.error("Execution requires Linux and explicit --permission-confirmed; plan-only is safe on Windows")
        try:
            check_batch_gate(args)
        except ValueError as exc:
            parser.error(str(exc))
    root = (args.output_dir or ROOT / "results" / ("load_aware_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + os.urandom(3).hex())).resolve()
    if root.exists():
        parser.error("Output directory already exists; choose a fresh directory")
    root.mkdir(parents=True)
    manifest = create_manifest(args, root)
    path = root / "experiment_manifest.json"
    load.write_json(path, manifest)
    print(f"[plan] {len(manifest['trials'])} trials: {path}", flush=True)
    if not args.execute:
        print("[plan] No stressor, benchmark, SSH connection, or monitoring service was started.")
        return 0
    previous_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Experiment launcher interrupted by signal {signum}")
    for signum in [signal.SIGTERM, signal.SIGHUP]:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    try:
        # Read-only early check; each trial also verifies freshness, capacity, and safe baseline.
        load.monitoring_preflight(args)
        for trial in manifest["trials"]:
            if shutil.disk_usage(root).free < load.GIB:
                raise RuntimeError("Less than 1 GiB free disk space remains; experiment stopped without deleting anything")
            print(f"[run] {trial['trial_id']}", flush=True)
            trial["status"] = "RUNNING"
            trial["started_at"] = time.time()
            load.write_json(path, manifest)
            code = run_trial(trial, args.trial_timeout)
            trial["exit_code"] = code
            trial["finished_at"] = time.time()
            trial["status"] = "COMPLETED" if code == 0 else "FAIL"
            load.write_json(path, manifest)
            if code:
                raise RuntimeError(f"Trial failed; partial results and sidecars retained: {trial['directory']}")
        import check_load_experiment
        report = check_load_experiment.validate(path)
        manifest["status"] = report["status"]
        load.write_json(path, manifest)
        print(f"[{report['status']}] {root / 'validation_report.json'}", flush=True)
        return 0 if report["status"] == "PASS" else 1
    except BaseException as exc:
        manifest["status"] = "FAIL"
        manifest["error"] = str(exc)
        load.write_json(path, manifest)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def math_is_positive(value):
    import math
    return math.isfinite(value) and value > 0


if __name__ == "__main__":
    raise SystemExit(main())
