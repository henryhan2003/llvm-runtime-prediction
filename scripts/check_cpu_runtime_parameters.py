#!/usr/bin/env python3
"""Verify the Linux CPU/I/O probe or reconcile a completed v4 OpenMP repair batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    import cpu_runtime_parameters as runtime
    import pipeline_new as polybench
    import pipeline_rodinia as openmp
except ImportError:
    from scripts import cpu_runtime_parameters as runtime
    from scripts import pipeline_new as polybench
    from scripts import pipeline_rodinia as openmp


IO_TASK = """import os, sys
path = sys.argv[1]
data = bytes(range(256)) * 256
with open(path, 'wb', buffering=0) as stream:
    assert stream.write(data) == len(data)
    os.fsync(stream.fileno())
with open(path, 'rb', buffering=0) as stream:
    assert stream.read() == data
"""

NATIVE_IO_TASK = r"""#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    unsigned char input[65536], output[65536];
    memset(input, 42, sizeof(input));
    FILE *stream = fopen(argv[1], "wb");
    if (!stream) return 3;
    if (fwrite(input, 1, sizeof(input), stream) != sizeof(input)) return 4;
    if (fflush(stream) || fsync(fileno(stream)) || fclose(stream)) return 5;
    stream = fopen(argv[1], "rb");
    if (!stream) return 6;
    if (fread(output, 1, sizeof(output), stream) != sizeof(output)) return 7;
    if (fclose(stream)) return 8;
    return memcmp(input, output, sizeof(input)) != 0;
}
"""

COLLIDING_IO_SYMBOLS = r"""#include <stdlib.h>
/* Match Rodinia myocyte's non-POSIX signatures in a separate translation unit. */
void read(char *filename, float *data, int rows, int cols, int major) {
    (void)filename; (void)data; (void)rows; (void)cols; (void)major;
    _Exit(91);
}
void write(char *filename, float *data, int rows, int cols, int major, int range) {
    (void)filename; (void)data; (void)rows; (void)cols; (void)major; (void)range;
    _Exit(92);
}
"""

LEGACY_IO_CONTROL = r"""#include <unistd.h>
__attribute__((destructor))
static void legacy_io(void) {
    char byte = 0;
#ifdef CHECK_WRITE
    (void)write(-1, &byte, 1);
#else
    (void)read(-1, &byte, 1);
#endif
}
"""


def build_c_task(compiler, sources, binary, flags=()):
    built = subprocess.run([compiler, "-O2", "-std=c11", "-Wall", "-Wextra", *flags,
                            *(str(source) for source in sources), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    if built.returncode:
        raise RuntimeError(f"Cannot build native I/O self-check: {built.stderr}")


def check_metrics(metrics):
    if metrics.get("process_metrics_backend") != "linux-process-probe-v4":
        raise AssertionError("Expected rebuilt v4 probe; an older executable is still in use")
    for field in ["process_cpu_user_sec", "process_cpu_system_sec", "process_max_rss_bytes",
                  "process_fs_inputs", "process_fs_outputs"]:
        if not isinstance(metrics.get(field), (int, float)) or metrics[field] < 0:
            raise AssertionError(f"Missing or invalid CPU/rusage field: {field}")
    if metrics["process_max_rss_bytes"] <= 0:
        raise AssertionError("Peak RSS must be positive")
    if metrics.get("process_io_collection_status") != "ok":
        raise AssertionError(f"Process I/O unavailable: {metrics.get('process_io_collection_status')} "
                             f"{metrics.get('process_io_collection_error')}")
    if metrics["process_io_rchar_bytes"] < 65536 or metrics["process_io_wchar_bytes"] < 65536:
        raise AssertionError("Failed to capture the short-lived child's known 64 KiB logical I/O")
    for field in runtime.IO_NUMERIC_FIELDS:
        if not isinstance(metrics.get(field), (int, float)) or metrics[field] < 0:
            raise AssertionError(f"Missing/negative process I/O field: {field}")
    if metrics.get("task_file_transfer_bytes") != "" or metrics.get("task_network_bandwidth_bytes_per_sec") != "":
        raise AssertionError("Unmeasured task transfers must remain blank, not fake zeros")


def run_checks():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Run this check on the Linux collection server")
    result = {}
    with tempfile.TemporaryDirectory(prefix="cpu-runtime-check-") as directory:
        root = Path(directory)
        compiler = next((path for c in ["cc", "gcc", "clang"] if (path := shutil.which(c))), None)
        if compiler is None:
            raise RuntimeError("A C compiler is required for the native program self-check")
        source = root / "native_io.c"
        binary = root / "native_io"
        source.write_text(NATIVE_IO_TASK, encoding="utf-8")
        build_c_task(compiler, [source], binary)
        collision_source = root / "collision.c"
        collision_source.write_text(COLLIDING_IO_SYMBOLS, encoding="utf-8")
        collision_binary = root / "native_io_collision"
        build_c_task(compiler, [source, collision_source], collision_binary, ["-rdynamic"])
        control_source = root / "legacy_io.c"
        control_source.write_text(LEGACY_IO_CONTROL, encoding="utf-8")
        controls = {}
        for symbol, code in [("read", 91), ("write", 92)]:
            library = root / f"legacy_{symbol}.so"
            flags = ["-fPIC", "-shared", *(["-DCHECK_WRITE"] if symbol == "write" else [])]
            build_c_task(compiler, [control_source], library, flags)
            env = os.environ.copy()
            env["LD_PRELOAD"] = str(library) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            control = subprocess.run([str(collision_binary), str(root / f"control_{symbol}.bin")],
                                     env=env, capture_output=True, text=True, timeout=15)
            if control.returncode != code:
                raise AssertionError(f"{symbol} interposition negative control: expected {code}, got {control.returncode}")
            controls[symbol] = control.returncode
        names = ["polybench", "openmp", "polybench_native", "openmp_native",
                 "polybench_symbol_collision", "openmp_symbol_collision"]
        for name in names:
            if name.endswith("_symbol_collision"):
                command = [str(collision_binary), str(root / (name + ".bin"))]
            elif name.endswith("_native"):
                command = [str(binary), str(root / (name + ".bin"))]
            else:
                command = [sys.executable, "-c", IO_TASK, str(root / (name + ".bin"))]
            if name.startswith("polybench"):
                completed, elapsed, timed_out, metrics = polybench.run_monitored_command(command, root, 15)
                if completed.returncode != 0 or timed_out:
                    raise AssertionError(f"{name}: execution failed, code={completed.returncode}: {completed.stderr}")
            else:
                measured = openmp.run_process(command, root, os.environ.copy(), 15, True, True)
                if not measured.success:
                    raise AssertionError(f"OpenMP execution failed: {measured.error_message}")
                elapsed, metrics = measured.elapsed_sec, measured.metrics
            try:
                check_metrics(metrics)
            except AssertionError as exc:
                raise AssertionError(f"{name}: {exc}") from exc
            if elapsed <= 0:
                raise AssertionError("Runtime must be positive")
            result[name] = {"elapsed_sec": elapsed, "metrics": metrics}
            if name.endswith("_symbol_collision"):
                result[name]["legacy_negative_control_exit_codes"] = controls
    return result


def check_openmp_results(directory, expected_configurations, expected_programs=()):
    """Reconcile a completed repair batch without running any benchmark."""
    paths = {}
    for kind in ["summary", "runs", "environment"]:
        suffix = "json" if kind == "environment" else "csv"
        found = list(directory.glob(f"rodinia_openmp_{kind}_*.{suffix}"))
        if len(found) != 1:
            raise ValueError(f"Expected one {kind} file, found {len(found)}")
        paths[kind] = found[0]
    hashes = {kind: hashlib.sha256(path.read_bytes()).hexdigest() for kind, path in paths.items()}
    with paths["summary"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with paths["runs"].open(encoding="utf-8-sig", newline="") as stream:
        run_rows = list(csv.DictReader(stream))
    environment = json.loads(paths["environment"].read_text(encoding="utf-8-sig"))
    errors = []
    truth = lambda value: str(value).lower() in {"true", "1"}
    identities = [(row.get("program_id"), row.get("input_id")) for row in rows]
    if any(not all(identity) for identity in identities):
        errors.append("Missing program_id or input_id")
    if len(rows) != expected_configurations or len(set(identities)) != expected_configurations:
        errors.append(f"Expected {expected_configurations} unique configurations, got {len(rows)} rows")
    if expected_programs and {key[0] for key in identities} != set(expected_programs):
        errors.append("Unexpected program set")
    if expected_configurations == 20 and len({key[0] for key in identities}) == 1:
        if {key[1] for key in identities} != {f"SIZE_{i:02d}" for i in range(1, 21)}:
            errors.append("Repair batch does not contain exactly SIZE_01 through SIZE_20")
    if environment.get("cpu_runtime_parameter_schema", {}).get("schema_version") != 4:
        errors.append("Environment does not identify CPU runtime schema v4")
    expected_session = environment.get("session_id")
    run_ids = set()
    groups = {}
    for run in run_rows:
        key = (run.get("program_id"), run.get("input_id"))
        groups.setdefault(key, []).append(run)
        if not run.get("run_id") or run["run_id"] in run_ids:
            errors.append(f"{key}: missing or duplicate run_id")
        run_ids.add(run.get("run_id"))
        if (not truth(run.get("success")) or run.get("returncode") != "0" or
                str(run.get("timed_out")).lower() not in {"false", "0"} or run.get("error_message")):
            errors.append(f"{key}: unsuccessful invocation")
        if run.get("phase") not in {"warmup", "measure"}:
            errors.append(f"{key}: unknown phase")
        if run.get("session_id") != expected_session:
            errors.append(f"{key}: session mismatch")
    if set(groups) != set(identities):
        errors.append("Run identities differ from summary identities")
    for row in rows:
        key = (row.get("program_id"), row.get("input_id"))
        try:
            for flag in ["build_success", "run_success", "prometheus_three_phase_collection_success"]:
                if not truth(row.get(flag)):
                    raise ValueError(f"{flag} is not true")
            if row.get("static_status") != "success" or row.get("session_id") != expected_session:
                raise ValueError("static extraction failure or session mismatch")
            if row.get("error_message") or row.get("prometheus_three_phase_collection_error"):
                raise ValueError("nonempty error field")
            invocations = groups.get(key, [])
            measured = [run for run in invocations if run["phase"] == "measure"]
            warmups = [run for run in invocations if run["phase"] == "warmup"]
            if len(measured) < 3 or len(measured) != int(row["measured_runs"]):
                raise ValueError("measured run count mismatch")
            if len(warmups) != int(row["warmup_runs_completed"]) or len(warmups) != int(row["warmup_runs_requested"]):
                raise ValueError("warmup count mismatch")
            for phase, phase_rows in [("measure", measured), ("warmup", warmups)]:
                if sorted(int(run["run_index"]) for run in phase_rows) != list(range(1, len(phase_rows) + 1)):
                    raise ValueError(f"{phase} run indices are not contiguous")
            for item in [row, *invocations]:
                if item.get("process_metrics_backend") != "linux-process-probe-v4":
                    raise ValueError("old or mixed probe backend")
                if item.get("process_io_collection_status") != "ok" or item.get("process_io_collection_error"):
                    raise ValueError("process I/O incomplete")
                for field in [*openmp.PROCESS_FIELDS, *runtime.IO_NUMERIC_FIELDS]:
                    value = float(item[field])
                    if not math.isfinite(value) or value < 0 or (field == "process_max_rss_bytes" and value <= 0):
                        raise ValueError(f"invalid {field}")
                if item.get("task_file_transfer_bytes", "") or item.get("task_network_bandwidth_bytes_per_sec", ""):
                    raise ValueError("unmeasured task transfer values must remain blank")
                if item.get("task_transfer_metrics_status") != runtime.TRANSFER_STATUS:
                    raise ValueError("task transfer missing-state marker is incorrect")
                if "runtime_sec" in item and (not math.isfinite(float(item["runtime_sec"])) or float(item["runtime_sec"]) <= 0):
                    raise ValueError("invalid invocation runtime")
            values = [float(run["runtime_sec"]) for run in measured]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("invalid invocation runtime")
            references = {"runtime_sec_median": statistics.median(values),
                          "measurement_seconds_actual": math.fsum(values),
                          "process_max_rss_bytes_peak": max(float(run["process_max_rss_bytes"]) for run in measured)}
            references.update({field: statistics.fmean(float(run[field]) for run in measured)
                               for field in [*openmp.PROCESS_FIELDS, *runtime.IO_NUMERIC_FIELDS]})
            for field, reference in references.items():
                if not math.isclose(float(row[field]), reference, rel_tol=1e-9, abs_tol=1e-11):
                    raise ValueError(f"summary aggregation mismatch: {field}")
            budget = float(row["measurement_seconds_requested"])
            if (not math.isfinite(budget) or budget <= 0 or int(row["process_io_sampled_runs"]) != len(measured) or
                    int(row["process_io_missing_runs"]) != 0 or math.fsum(values) < budget):
                raise ValueError("I/O coverage or measurement budget mismatch")
            for field in [*openmp.AST_FIELDS, *openmp.CFG_FIELDS, *openmp.IR_FIELDS]:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"nonfinite static feature: {field}")
            for phase in ["before", "during", "after"]:
                if not truth(row.get(f"prometheus_{phase}_collection_success")) or row.get(f"prometheus_{phase}_collection_error"):
                    raise ValueError(f"{phase} monitoring failure")
                if int(row[f"prometheus_{phase}_sample_count"]) < 2:
                    raise ValueError(f"too few {phase} query points")
                for field in row:
                    if field.startswith("host_") and f"_{phase}_" in field:
                        value = float(row[field])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(f"invalid host metric: {field}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{key}: {exc}")
    if hashes != {kind: hashlib.sha256(path.read_bytes()).hexdigest() for kind, path in paths.items()}:
        errors.append("Source files changed during validation")
    return {"status": "FAIL" if errors else "PASS", "configurations": len(rows),
            "programs": sorted({key[0] for key in identities}), "run_records": len(run_rows),
            "source_sha256": hashes, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path (otherwise stdout only)")
    parser.add_argument("--openmp-results-dir", type=Path, help="Validate an existing v4 OpenMP batch instead of running self-check tasks")
    parser.add_argument("--expected-configurations", type=int)
    parser.add_argument("--expected-program", action="append", default=[])
    args = parser.parse_args()
    report = {"platform": platform.platform(), "created_at": datetime.now().astimezone().isoformat()}
    try:
        if args.openmp_results_dir:
            if not args.expected_configurations or args.expected_configurations < 1:
                raise ValueError("--expected-configurations must be positive")
            validation = check_openmp_results(args.openmp_results_dir, args.expected_configurations, args.expected_program)
            report.update(status=validation["status"], validation=validation)
        else:
            report.update(status="PASS", checks=run_checks())
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
