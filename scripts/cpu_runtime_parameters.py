"""Process-scoped CPU runtime labels shared by the two CPU collection pipelines."""

from __future__ import annotations

import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


IO_PROBE_FIELDS = {
    "io_rchar": "process_io_rchar_bytes",
    "io_wchar": "process_io_wchar_bytes",
    "io_syscr": "process_io_read_syscalls",
    "io_syscw": "process_io_write_syscalls",
    "io_read_bytes": "process_storage_read_bytes",
    "io_write_bytes": "process_storage_write_bytes",
    "io_cancelled_write_bytes": "process_storage_cancelled_write_bytes",
}
IO_NUMERIC_FIELDS = list(IO_PROBE_FIELDS.values())
TRANSFER_FIELDS = ["task_file_transfer_bytes", "task_network_bandwidth_bytes_per_sec",
                   "task_transfer_metrics_status"]
RUN_FIELDS = [*IO_NUMERIC_FIELDS, "process_io_collection_status", "process_io_collection_error",
              *TRANSFER_FIELDS]
SUMMARY_FIELDS = [*RUN_FIELDS, "process_io_sampled_runs", "process_io_missing_runs"]
TRANSFER_STATUS = "not_collected_no_workflow_transfer_instrumentation"


def io_collector_binary(root: Path) -> Path:
    return root / "build" / "cpu_runtime_io_collector.so"


def ensure_io_collector(root: Path) -> Path | None:
    if not sys.platform.startswith("linux"):
        return None
    source = root / "scripts" / "cpu_runtime_io_collector.c"
    binary = io_collector_binary(root)
    if not source.is_file():
        raise RuntimeError(f"self I/O collector source not found: {source}")
    if any(character.isspace() for character in str(binary)) or ":" in str(binary):
        raise RuntimeError("LD_PRELOAD collector path must not contain whitespace or colons")
    if binary.is_file() and binary.stat().st_mtime >= source.stat().st_mtime:
        return binary
    compiler = next((path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))), None)
    if compiler is None:
        raise RuntimeError("cannot build self I/O collector: cc, gcc, and clang are unavailable")
    binary.parent.mkdir(parents=True, exist_ok=True)
    temporary = binary.with_name(f".{binary.name}.{os.getpid()}")
    try:
        completed = subprocess.run([compiler, "-O2", "-std=c11", "-Wall", "-Wextra", "-fPIC", "-shared",
                                    str(source), "-o", str(temporary)], capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=30)
        if completed.returncode != 0:
            raise RuntimeError(f"cannot build self I/O collector: {completed.stderr[-4000:]}")
        temporary.replace(binary)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot build self I/O collector: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return binary


def probe_command(probe: Path, report: Path, command: list[str], root: Path) -> list[str]:
    if sys.platform.startswith("linux"):
        return [str(probe), str(report), "--io-collector", str(io_collector_binary(root)), "--", *command]
    return [str(probe), str(report), "--", *command]


def empty_metrics(reason: str = "not_collected") -> dict[str, object]:
    return {**{field: "" for field in IO_NUMERIC_FIELDS},
            "process_io_collection_status": reason, "process_io_collection_error": "",
            "task_file_transfer_bytes": "", "task_network_bandwidth_bytes_per_sec": "",
            "task_transfer_metrics_status": TRANSFER_STATUS}


def parse_probe_extensions(values: dict[str, str]) -> dict[str, object]:
    metrics = empty_metrics("not_collected_probe_v1")
    if values.get("probe_version") not in {"2", "3"}:
        return metrics
    status = values.get("io_status", "invalid_probe_io")
    metrics["process_io_collection_status"] = status
    metrics["process_io_collection_error"] = values.get("io_error", "")
    if status != "ok":
        return metrics
    try:
        counters = {field: int(values[source]) for source, field in IO_PROBE_FIELDS.items()}
        if any(value < 0 for value in counters.values()):
            raise ValueError("negative process I/O counter")
    except (KeyError, ValueError) as exc:
        metrics["process_io_collection_status"] = "invalid_probe_io"
        metrics["process_io_collection_error"] = str(exc)
        return metrics
    metrics.update(counters)
    return metrics


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    # Failed runs and warmups must not become resource-demand supervision labels.
    successful = [r for r in records if r.get("success") is True and r.get("phase", "measure") == "measure"]
    def is_complete(record):
        return record.get("process_io_collection_status") == "ok" and all(
            isinstance(record.get(f), (int, float)) and not isinstance(record[f], bool)
            and math.isfinite(record[f]) and record[f] >= 0 for f in IO_NUMERIC_FIELDS)

    complete = [r for r in successful if is_complete(r)]
    summary = empty_metrics("not_run" if not successful else "unavailable")
    summary["process_io_sampled_runs"] = len(complete)
    summary["process_io_missing_runs"] = len(successful) - len(complete)
    if successful and len(complete) == len(successful):
        summary.update({field: statistics.fmean(r[field] for r in complete) for field in IO_NUMERIC_FIELDS})
        summary["process_io_collection_status"] = "ok"
    elif complete:
        # Keep each good run in the detail CSV, but do not silently average a biased subset.
        summary["process_io_collection_status"] = "partial"
    errors = sorted({str(r.get("process_io_collection_error") or r.get("process_io_collection_status") or "not_collected")
                     for r in successful if not is_complete(r)})
    summary["process_io_collection_error"] = " | ".join(errors)[:2000]
    return summary


def schema_description() -> dict[str, object]:
    return {
        "schema_version": 3,
        "prediction_role": "post_execution_supervision_labels_only_not_direct_static_inputs",
        "scope": "launched command; CPU covers its threads; proc I/O includes waited-for children; RSS is not summed process-tree memory",
        "aggregation": "arithmetic mean per successful measured invocation; warmup excluded; missing I/O invalidates its summary mean",
        "existing_runtime_parameters": {
            "uCPU": {"field": "process_cpu_user_sec", "unit": "CPU-seconds"},
            "sCPU": {"field": "process_cpu_system_sec", "unit": "CPU-seconds"},
            "memory_usage": {"field": "process_max_rss_bytes", "unit": "bytes", "definition": "mean of per-invocation peak RSS"},
            "read_blocks": {"field": "process_fs_inputs", "source": "rusage.ru_inblock", "unit": "kernel filesystem input accounting units; not bytes or syscalls"},
            "write_blocks": {"field": "process_fs_outputs", "source": "rusage.ru_oublock", "unit": "kernel filesystem output accounting units; not bytes or syscalls"},
        },
        "io_fields": {field: {"source": "/proc/PID/io:" + source.removeprefix("io_"),
                              "unit": "syscalls" if "syscalls" in field else "bytes"}
                      for source, field in IO_PROBE_FIELDS.items()},
        "io_snapshot": "Linux dynamic-loader normal-exit callback reads /proc/self/io before address-space teardown; PID-checked sidecar; no periodic sampling",
        "runtime_boundary": "fork/exec through wait4; includes preload startup and exit callback overhead, excludes parent report parsing",
        "file_transfer_and_bandwidth": "missing, not zero: no task-attributed workflow file-transfer instrumentation; never copied from host Prometheus",
        "io_limits": "normal dynamically linked program exit only; static binaries, _exit, fatal signals or callbacks bypassing loader cleanup yield missing, not zero; late cleanup after the callback is outside the I/O snapshot; rchar/wchar include redirected output and loader overhead; mmap is not fully represented; collector report writes are excluded from I/O snapshot",
        "comparison": "v3 self-collection changes instrumentation and timing overhead; do not silently mix old and new batches",
    }
