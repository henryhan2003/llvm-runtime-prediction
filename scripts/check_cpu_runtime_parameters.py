#!/usr/bin/env python3
"""Verify the real Linux CPU and I/O probe through both CPU pipeline interfaces."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
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


def check_metrics(metrics):
    if metrics.get("process_metrics_backend") != "linux-process-probe-v3":
        raise AssertionError("Expected rebuilt v3 probe; an older executable is still in use")
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
        built = subprocess.run([compiler, "-O2", "-std=c11", "-Wall", "-Wextra", str(source), "-o", str(binary)],
                               capture_output=True, text=True, timeout=30)
        if built.returncode:
            raise RuntimeError(f"Cannot build native I/O self-check: {built.stderr}")
        for name in ["polybench", "openmp", "polybench_native", "openmp_native"]:
            command = ([str(binary), str(root / (name + ".bin"))] if name.endswith("_native") else
                       [sys.executable, "-c", IO_TASK, str(root / (name + ".bin"))])
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
