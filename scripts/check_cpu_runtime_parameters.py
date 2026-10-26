#!/usr/bin/env python3
"""Verify the real Linux CPU and I/O probe through both CPU pipeline interfaces."""

from __future__ import annotations

import argparse
import json
import os
import platform
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


def check_metrics(metrics):
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
        for name in ["polybench", "openmp"]:
            command = [sys.executable, "-c", IO_TASK, str(root / (name + ".bin"))]
            if name == "polybench":
                completed, elapsed, timed_out, metrics = polybench.run_monitored_command(command, root, 15)
                if completed.returncode != 0 or timed_out:
                    raise AssertionError(f"PolyBench execution failed: {completed.stderr}")
            else:
                measured = openmp.run_process(command, root, os.environ.copy(), 15, True, True)
                if not measured.success:
                    raise AssertionError(f"OpenMP execution failed: {measured.error_message}")
                elapsed, metrics = measured.elapsed_sec, measured.metrics
            check_metrics(metrics)
            if elapsed <= 0:
                raise AssertionError("Runtime must be positive")
            result[name] = {"elapsed_sec": elapsed, "metrics": metrics}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path (otherwise stdout only)")
    args = parser.parse_args()
    report = {"platform": platform.platform(), "created_at": datetime.now().astimezone().isoformat()}
    try:
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
