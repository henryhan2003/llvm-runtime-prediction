"""Process-scoped CPU runtime labels shared by the two CPU collection pipelines."""

from __future__ import annotations

import math
import statistics


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


def empty_metrics(reason: str = "not_collected") -> dict[str, object]:
    return {**{field: "" for field in IO_NUMERIC_FIELDS},
            "process_io_collection_status": reason, "process_io_collection_error": "",
            "task_file_transfer_bytes": "", "task_network_bandwidth_bytes_per_sec": "",
            "task_transfer_metrics_status": TRANSFER_STATUS}


def parse_probe_extensions(values: dict[str, str]) -> dict[str, object]:
    metrics = empty_metrics("not_collected_probe_v1")
    if values.get("probe_version") != "2":
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
        "schema_version": 2,
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
        "io_snapshot": "Linux waitid(WNOWAIT) retains the exited child until /proc/PID/io is read; no periodic sampling",
        "runtime_boundary": "fork/exec to child exit notification; excludes post-exit procfs read and report serialization",
        "file_transfer_and_bandwidth": "missing, not zero: no task-attributed workflow file-transfer instrumentation; never copied from host Prometheus",
        "io_limits": "rchar/wchar include successful read/write-family syscall bytes including redirected output, not just regular files; mmap access is not fully represented; storage bytes depend on cache/writeback",
        "comparison": "v2 timing/probe identity is distinct; do not silently mix old and new collection batches",
    }
