#!/usr/bin/env python3
"""Build and measure Rodinia 3.1 CUDA benchmarks on Linux.

Benchmark-specific build and run commands live in configs/rodinia_cuda.json.
The runtime collector reuses the Rodinia process probe and Prometheus host
collector, then adds selected-device telemetry sampled directly with
nvidia-smi before, during, and after each measurement window.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import pipeline_rodinia as base
except ImportError:
    from scripts import pipeline_rodinia as base


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "rodinia_cuda.json"
DEFAULT_RODINIA_ROOT = PROJECT_ROOT / "datasets" / "rodinia_3.1"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
CUDA_12_TEXTURE_REFERENCE_BENCHMARKS = frozenset({"leukocyte"})

GPU_PHASES = ("before", "during", "after")
GPU_SAMPLE_KEYS = (
    "gpu_utilization_pct",
    "gpu_memory_io_utilization_pct",
    "gpu_memory_used_bytes",
    "gpu_power_draw_watts",
    "gpu_temperature_celsius",
    "gpu_sm_clock_mhz",
    "gpu_memory_clock_mhz",
)
GPU_ENVIRONMENT_FIELDS = [
    "gpu_physical_index",
    "gpu_runtime_index",
    "gpu_uuid",
    "gpu_name",
    "gpu_compute_capability",
    "gpu_total_memory_bytes",
    "gpu_driver_version",
    "gpu_power_limit_watts",
    "cuda_visible_devices",
    "cuda_arch",
    "cuda_toolkit_root",
    "cuda_toolkit_version",
    "nvcc_path",
    "nvcc_version",
]
GPU_PHASE_FIELDS = [
    field
    for phase in GPU_PHASES
    for field in (
        f"gpu_{phase}_window_start_timestamp",
        f"gpu_{phase}_window_end_timestamp",
        f"gpu_{phase}_window_duration_seconds",
        f"gpu_{phase}_sample_count",
        f"gpu_utilization_pct_{phase}_mean",
        f"gpu_utilization_pct_{phase}_max",
        f"gpu_memory_io_utilization_pct_{phase}_mean",
        f"gpu_memory_io_utilization_pct_{phase}_max",
        f"gpu_memory_used_bytes_{phase}_mean",
        f"gpu_memory_used_bytes_{phase}_max",
        f"gpu_power_draw_watts_{phase}_mean",
        f"gpu_power_draw_watts_{phase}_max",
        f"gpu_temperature_celsius_{phase}_mean",
        f"gpu_temperature_celsius_{phase}_max",
        f"gpu_sm_clock_mhz_{phase}_mean",
        f"gpu_memory_clock_mhz_{phase}_mean",
        f"gpu_energy_joules_{phase}_estimate",
        f"gpu_{phase}_collection_success",
        f"gpu_{phase}_collection_error",
    )
]
GPU_BACKGROUND_FIELDS = [
    "gpu_sampling_scheme",
    "gpu_metric_scope",
    "gpu_sample_interval_seconds_requested",
    "gpu_background_correction_method",
    "gpu_background_stability_warning",
    "gpu_background_utilization_pct_mean",
    "gpu_background_memory_used_bytes_mean",
    "gpu_background_power_draw_watts_mean",
    "gpu_background_utilization_before_after_abs_diff_pct",
    "gpu_background_memory_before_after_abs_diff_bytes",
    "gpu_background_power_before_after_abs_diff_watts",
    "gpu_utilization_pct_background_adjusted_mean",
    "gpu_utilization_pct_background_adjusted_max",
    "gpu_memory_used_bytes_background_adjusted_mean",
    "gpu_memory_used_bytes_background_adjusted_max",
    "gpu_power_draw_watts_background_adjusted_mean",
    "gpu_power_draw_watts_background_adjusted_max",
    "gpu_energy_joules_background_adjusted_estimate",
    "gpu_three_phase_collection_success",
    "gpu_three_phase_collection_error",
]
GPU_RESULT_FIELDS = [*GPU_PHASE_FIELDS, *GPU_BACKGROUND_FIELDS]

CUDA_STATIC_FIELDS = [
    "cuda_source_file_count",
    "cuda_kernel_definition_count",
    "cuda_kernel_launch_count",
    "cuda_device_function_count",
    "cuda_host_function_count",
    "cuda_shared_memory_declaration_count",
    "cuda_constant_memory_declaration_count",
    "cuda_texture_reference_count",
    "cuda_memory_allocation_call_count",
    "cuda_pinned_host_memory_allocation_call_count",
    "cuda_managed_memory_allocation_call_count",
    "cuda_memory_free_call_count",
    "cuda_memcpy_call_count",
    "cuda_symbol_copy_call_count",
    "cuda_host_to_device_copy_count",
    "cuda_device_to_host_copy_count",
    "cuda_device_to_device_copy_count",
    "cuda_memset_call_count",
    "cuda_synchronization_call_count",
    "cuda_atomic_call_count",
    "cuda_stream_call_count",
    "cuda_event_call_count",
    "cuda_error_check_call_count",
]

SUMMARY_FIELDS = [
    "session_id",
    "collected_at",
    "dataset",
    "parallel_model",
    "program_id",
    "input_id",
    "input_profile",
    "input_size_parameters",
    "host_thread_count",
    "gpu_idle_preflight_passed",
    "gpu_preexisting_compute_process_count",
    "gpu_preexisting_compute_processes",
    "workdir",
    "executable",
    "source_files",
    "source_file_count",
    "ast_owned_source_files",
    "ast_owned_source_count",
    "build_command",
    "run_command",
    "warmup_runs_requested",
    "warmup_runs_completed",
    "measurement_seconds_requested",
    "min_measured_runs_requested",
    "measurement_seconds_actual",
    "measured_runs",
    "runtime_sec_median",
    "runtime_sec_mean",
    "runtime_sec_std",
    "runtime_sec_cv",
    "runtime_sec_min",
    "runtime_sec_max",
    "process_metric_aggregation",
    *base.PROCESS_FIELDS,
    "process_max_rss_bytes_peak",
    "process_metrics_backend",
    *base.pipeline_core().PROMETHEUS_RESULT_FIELDS,
    *GPU_RESULT_FIELDS,
    *base.ENVIRONMENT_FIELDS,
    *GPU_ENVIRONMENT_FIELDS,
    "static_status",
    "static_source_count",
    "static_source_success_count",
    "cuda_static_source_metrics_success",
    *base.AST_FIELDS,
    *base.CFG_FIELDS,
    *base.IR_FIELDS,
    *CUDA_STATIC_FIELDS,
    "build_success",
    "run_success",
    "error_message",
]

RUN_FIELDS = [*base.RUN_FIELDS, "gpu_physical_index", "gpu_uuid", "cuda_arch"]

CUDA_LOCAL_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s+"([^"\n]+\.(?:c|cc|cpp|cxx|cu|h|hh|hpp|hxx|cuh))"',
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen=True)
class GpuDevice:
    physical_index: int
    uuid: str
    name: str
    compute_capability: str
    total_memory_bytes: int
    driver_version: str
    power_limit_watts: float | None


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    values: dict[str, float]


def _float_or_none(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() in {"N/A", "NA", "[N/A]", "NOT SUPPORTED"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _run_text(command: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def query_gpu_devices(nvidia_smi: str) -> list[GpuDevice]:
    fields = "index,uuid,name,compute_cap,memory.total,driver_version,power.limit"
    result = _run_text(
        [nvidia_smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    has_compute_capability = True
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if "compute_cap" not in message.lower():
            raise RuntimeError(f"nvidia-smi GPU query failed: {message}")
        # Older nvidia-smi releases (including the 470 driver series) do not
        # expose compute_cap as a selectable field. Retain the other identity
        # fields and require an explicit --cuda-arch for those systems.
        fields = "index,uuid,name,memory.total,driver_version,power.limit"
        result = _run_text(
            [nvidia_smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        )
        has_compute_capability = False
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"nvidia-smi GPU query failed: {message}")
    devices: list[GpuDevice] = []
    for row in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
        expected_fields = 7 if has_compute_capability else 6
        if len(row) != expected_fields:
            continue
        if has_compute_capability:
            compute_capability = row[3].strip()
            memory_index = 4
            driver_index = 5
            power_index = 6
        else:
            compute_capability = ""
            memory_index = 3
            driver_index = 4
            power_index = 5
        memory_mib = _float_or_none(row[memory_index]) or 0.0
        devices.append(
            GpuDevice(
                physical_index=int(row[0].strip()),
                uuid=row[1].strip(),
                name=row[2].strip(),
                compute_capability=compute_capability,
                total_memory_bytes=int(memory_mib * 1024 * 1024),
                driver_version=row[driver_index].strip(),
                power_limit_watts=_float_or_none(row[power_index]),
            )
        )
    if not devices:
        raise RuntimeError("nvidia-smi did not return any NVIDIA GPU")
    return devices


def query_compute_processes(nvidia_smi: str, gpu_uuid: str) -> list[dict[str, str]]:
    fields = "gpu_uuid,pid,process_name,used_gpu_memory"
    result = _run_text(
        [nvidia_smi, f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"]
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if "no running" in message.lower() and "process" in message.lower():
            return []
        raise RuntimeError(f"nvidia-smi compute-process query failed: {message}")
    processes: list[dict[str, str]] = []
    for row in csv.reader(line for line in result.stdout.splitlines() if line.strip()):
        if len(row) >= 4 and row[0].strip() == gpu_uuid:
            processes.append(
                {
                    "gpu_uuid": row[0].strip(),
                    "pid": row[1].strip(),
                    "process_name": row[2].strip(),
                    "used_gpu_memory_mib": row[3].strip(),
                }
            )
    return processes


def _parse_gpu_sample_line(line: str, timestamp: float) -> GpuSample:
    row = next(csv.reader([line]))
    if len(row) != len(GPU_SAMPLE_KEYS):
        raise ValueError(f"expected {len(GPU_SAMPLE_KEYS)} nvidia-smi values, got {len(row)}")
    values: dict[str, float] = {}
    for key, raw in zip(GPU_SAMPLE_KEYS, row):
        parsed = _float_or_none(raw)
        if parsed is not None:
            values[key] = parsed
    if "gpu_memory_used_bytes" in values:
        values["gpu_memory_used_bytes"] *= 1024 * 1024
    return GpuSample(timestamp=timestamp, values=values)


class NvidiaSmiSampler:
    QUERY_FIELDS = (
        "utilization.gpu,utilization.memory,memory.used,power.draw,"
        "temperature.gpu,clocks.current.sm,clocks.current.memory"
    )

    def __init__(
        self,
        executable: str,
        physical_index: int,
        interval_seconds: float,
    ) -> None:
        self.executable = executable
        self.physical_index = physical_index
        self.interval_seconds = interval_seconds
        self.samples: list[GpuSample] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU sampler has already been started")
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def wait_for_first_sample(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._lock:
                if self.samples:
                    return True
                if self.errors:
                    return False
            if self._thread is not None and not self._thread.is_alive():
                return False
            time.sleep(min(0.05, self.interval_seconds / 4))
        with self._lock:
            return bool(self.samples)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 4))
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    def _sample_loop(self) -> None:
        command = [
            self.executable,
            f"--id={self.physical_index}",
            f"--query-gpu={self.QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={max(1, round(self.interval_seconds * 1000))}",
        ]
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._lock:
                self._process = process
            assert process.stdout is not None
            while not self._stop.is_set():
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue
                try:
                    sample = _parse_gpu_sample_line(line, time.time())
                except ValueError as exc:
                    message = str(exc)[:500]
                    with self._lock:
                        if not self.errors or self.errors[-1] != message:
                            self.errors.append(message)
                    continue
                with self._lock:
                    self.samples.append(sample)
            if not self._stop.is_set() and process.poll() not in (None, 0):
                stderr = process.stderr.read().strip() if process.stderr else ""
                raise RuntimeError(stderr or f"nvidia-smi exited with code {process.returncode}")
        except (OSError, RuntimeError) as exc:
            message = str(exc)[:500]
            with self._lock:
                if not self.errors or self.errors[-1] != message:
                    self.errors.append(message)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            with self._lock:
                self._process = None

    def snapshot(self) -> tuple[list[GpuSample], list[str]]:
        with self._lock:
            return list(self.samples), list(self.errors)


def _values(samples: Iterable[GpuSample], key: str) -> list[float]:
    return [sample.values[key] for sample in samples if key in sample.values]


def _mean(samples: Iterable[GpuSample], key: str) -> float | None:
    values = _values(samples, key)
    return statistics.fmean(values) if values else None


def _max(samples: Iterable[GpuSample], key: str) -> float | None:
    values = _values(samples, key)
    return max(values) if values else None


def _average_available(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return statistics.fmean(values) if values else None


def _difference(left: float | None, right: float | None) -> float | str:
    return abs(left - right) if left is not None and right is not None else ""


def _adjust(value: float | None, background: float | None) -> float | str:
    if value is None or background is None:
        return ""
    return max(0.0, value - background)


def empty_gpu_result(
    phase_windows: dict[str, tuple[float, float]] | None,
    error: str,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        field: "" for field in GPU_RESULT_FIELDS
    }
    result.update(
        {
            "gpu_sampling_scheme": "nvidia_smi_before_during_after_v1",
            "gpu_metric_scope": "selected physical GPU; not process-exclusive",
            "gpu_sample_interval_seconds_requested": sample_interval_seconds,
            "gpu_background_correction_method": "mean(before,after) subtracted and clipped at zero",
            "gpu_three_phase_collection_success": False,
            "gpu_three_phase_collection_error": error,
        }
    )
    for phase in GPU_PHASES:
        start, end = (phase_windows or {}).get(phase, (0.0, 0.0))
        result.update(
            {
                f"gpu_{phase}_window_start_timestamp": start,
                f"gpu_{phase}_window_end_timestamp": end,
                f"gpu_{phase}_window_duration_seconds": max(0.0, end - start),
                f"gpu_{phase}_sample_count": 0,
                f"gpu_{phase}_collection_success": False,
                f"gpu_{phase}_collection_error": error,
            }
        )
    return result


def summarize_gpu_samples(
    samples: list[GpuSample],
    phase_windows: dict[str, tuple[float, float]],
    sample_interval_seconds: float,
    sampler_errors: Iterable[str] = (),
) -> dict[str, Any]:
    error_text = " | ".join(dict.fromkeys(error for error in sampler_errors if error))
    result = empty_gpu_result(phase_windows, error_text, sample_interval_seconds)
    by_phase: dict[str, list[GpuSample]] = {}
    phase_success: dict[str, bool] = {}
    for phase in GPU_PHASES:
        start, end = phase_windows.get(phase, (0.0, 0.0))
        selected = [sample for sample in samples if start <= sample.timestamp <= end]
        by_phase[phase] = selected
        duration = max(0.0, end - start)
        has_utilization = any("gpu_utilization_pct" in sample.values for sample in selected)
        has_memory = any("gpu_memory_used_bytes" in sample.values for sample in selected)
        phase_ok = bool(selected) and has_utilization and has_memory
        phase_success[phase] = phase_ok
        if phase_ok:
            phase_error = ""
        elif not selected:
            phase_error = error_text or "no nvidia-smi sample in phase window"
        else:
            phase_error = "nvidia-smi samples lack GPU utilization or used-memory values"
        result.update(
            {
                f"gpu_{phase}_sample_count": len(selected),
                f"gpu_{phase}_collection_success": phase_ok,
                f"gpu_{phase}_collection_error": phase_error,
            }
        )
        for key in GPU_SAMPLE_KEYS:
            mean_value = _mean(selected, key)
            if mean_value is not None:
                result[f"{key}_{phase}_mean"] = mean_value
            if key not in {"gpu_sm_clock_mhz", "gpu_memory_clock_mhz"}:
                max_value = _max(selected, key)
                if max_value is not None:
                    result[f"{key}_{phase}_max"] = max_value
        power_mean = _mean(selected, "gpu_power_draw_watts")
        result[f"gpu_energy_joules_{phase}_estimate"] = (
            power_mean * duration if power_mean is not None else ""
        )

    before = by_phase["before"]
    during = by_phase["during"]
    after = by_phase["after"]
    before_util = _mean(before, "gpu_utilization_pct")
    after_util = _mean(after, "gpu_utilization_pct")
    before_memory = _mean(before, "gpu_memory_used_bytes")
    after_memory = _mean(after, "gpu_memory_used_bytes")
    before_power = _mean(before, "gpu_power_draw_watts")
    after_power = _mean(after, "gpu_power_draw_watts")
    background_util = _average_available(before_util, after_util)
    background_memory = _average_available(before_memory, after_memory)
    background_power = _average_available(before_power, after_power)
    during_duration = max(
        0.0,
        phase_windows.get("during", (0.0, 0.0))[1]
        - phase_windows.get("during", (0.0, 0.0))[0],
    )
    adjusted_power = _adjust(_mean(during, "gpu_power_draw_watts"), background_power)
    result.update(
        {
            "gpu_background_utilization_pct_mean": background_util if background_util is not None else "",
            "gpu_background_memory_used_bytes_mean": background_memory if background_memory is not None else "",
            "gpu_background_power_draw_watts_mean": background_power if background_power is not None else "",
            "gpu_background_utilization_before_after_abs_diff_pct": _difference(before_util, after_util),
            "gpu_background_memory_before_after_abs_diff_bytes": _difference(before_memory, after_memory),
            "gpu_background_power_before_after_abs_diff_watts": _difference(before_power, after_power),
            "gpu_utilization_pct_background_adjusted_mean": _adjust(
                _mean(during, "gpu_utilization_pct"), background_util
            ),
            "gpu_utilization_pct_background_adjusted_max": _adjust(
                _max(during, "gpu_utilization_pct"), background_util
            ),
            "gpu_memory_used_bytes_background_adjusted_mean": _adjust(
                _mean(during, "gpu_memory_used_bytes"), background_memory
            ),
            "gpu_memory_used_bytes_background_adjusted_max": _adjust(
                _max(during, "gpu_memory_used_bytes"), background_memory
            ),
            "gpu_power_draw_watts_background_adjusted_mean": adjusted_power,
            "gpu_power_draw_watts_background_adjusted_max": _adjust(
                _max(during, "gpu_power_draw_watts"), background_power
            ),
            "gpu_energy_joules_background_adjusted_estimate": (
                adjusted_power * during_duration if isinstance(adjusted_power, float) else ""
            ),
        }
    )
    stability_warnings: list[str] = []
    if before_util is not None and after_util is not None and abs(before_util - after_util) > 10:
        stability_warnings.append("GPU utilization baseline changed by more than 10 percentage points")
    if before_power is not None and after_power is not None and abs(before_power - after_power) > 20:
        stability_warnings.append("GPU power baseline changed by more than 20 W")
    if before_memory is not None and after_memory is not None and abs(before_memory - after_memory) > 256 * 1024 * 1024:
        stability_warnings.append("GPU used-memory baseline changed by more than 256 MiB")
    result["gpu_background_stability_warning"] = " | ".join(stability_warnings)
    all_phases_ok = all(phase_success.values()) and not error_text
    result["gpu_three_phase_collection_success"] = all_phases_ok
    result["gpu_three_phase_collection_error"] = "" if all_phases_ok else (
        error_text or "one or more GPU phase windows contain no samples"
    )
    return result


def _strip_c_comments_and_literals(source: str) -> str:
    pattern = re.compile(
        r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
        re.DOTALL,
    )
    return pattern.sub(" ", source)


def extract_cuda_source_features(paths: Iterable[Path]) -> dict[str, int | bool]:
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    texts: list[str] = []
    cuda_file_count = 0
    for path in unique_paths:
        if not path.is_file():
            continue
        if path.suffix.lower() == ".cu":
            cuda_file_count += 1
        texts.append(_strip_c_comments_and_literals(path.read_text(encoding="utf-8", errors="ignore")))
    source = "\n".join(texts)

    def count(pattern: str, flags: int = 0) -> int:
        return len(re.findall(pattern, source, flags))

    return {
        "cuda_static_source_metrics_success": bool(texts),
        "cuda_source_file_count": cuda_file_count,
        "cuda_kernel_definition_count": count(r"\b__global__\b[^;{]*\{", re.DOTALL),
        "cuda_kernel_launch_count": count(r"<<<"),
        "cuda_device_function_count": count(r"\b__device__\b[^;{]*\{", re.DOTALL),
        "cuda_host_function_count": count(r"\b__host__\b[^;{]*\{", re.DOTALL),
        "cuda_shared_memory_declaration_count": count(r"\b__shared__\b"),
        "cuda_constant_memory_declaration_count": count(r"\b__constant__\b"),
        "cuda_texture_reference_count": count(r"\btexture\s*<|\bcudaBindTexture\s*\("),
        "cuda_memory_allocation_call_count": count(
            r"\bcudaMalloc(?:Pitch)?\s*\("
        ),
        "cuda_pinned_host_memory_allocation_call_count": count(r"\bcudaMallocHost\s*\("),
        "cuda_managed_memory_allocation_call_count": count(r"\bcudaMallocManaged\s*\("),
        "cuda_memory_free_call_count": count(r"\bcudaFree(?:Host)?\s*\("),
        "cuda_memcpy_call_count": count(r"\bcudaMemcpy(?:2D|3D)?(?:Async)?\s*\("),
        "cuda_symbol_copy_call_count": count(
            r"\bcudaMemcpy(?:To|From)Symbol(?:Async)?\s*\("
        ),
        "cuda_host_to_device_copy_count": count(r"\bcudaMemcpyHostToDevice\b"),
        "cuda_device_to_host_copy_count": count(r"\bcudaMemcpyDeviceToHost\b"),
        "cuda_device_to_device_copy_count": count(r"\bcudaMemcpyDeviceToDevice\b"),
        "cuda_memset_call_count": count(r"\bcudaMemset(?:2D|Async)?\s*\("),
        "cuda_synchronization_call_count": count(
            r"\b(?:cudaDeviceSynchronize|cudaThreadSynchronize|cudaStreamSynchronize|__syncthreads)\s*\("
        ),
        "cuda_atomic_call_count": count(r"\batomic[A-Za-z0-9_]*\s*\("),
        "cuda_stream_call_count": count(r"\bcudaStream[A-Za-z0-9_]*\s*\("),
        "cuda_event_call_count": count(r"\bcudaEvent[A-Za-z0-9_]*\s*\("),
        "cuda_error_check_call_count": count(
            r"\b(?:cudaGetLastError|cudaPeekAtLastError|cudaGetErrorString)\s*\("
        ),
    }


def translation_unit_owned_sources(
    source_path: Path,
    declared_source_paths: Iterable[Path],
) -> list[Path]:
    declared = {path.resolve() for path in declared_source_paths}
    result: list[Path] = [source_path.resolve()]
    visited: set[Path] = set()
    pending = [source_path.resolve()]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        if not current.is_file():
            continue
        source = current.read_text(encoding="utf-8", errors="ignore")
        includes = [
            (current.parent / match).resolve()
            for match in CUDA_LOCAL_INCLUDE_RE.findall(source)
        ]
        for included in reversed(includes):
            if included in declared and included not in result:
                result.append(included)
            if included.is_file() and included not in visited:
                pending.append(included)
    return result


def translation_unit_ownership(
    spec: base.BenchmarkSpec,
    cuda_root: Path,
) -> dict[Path, list[Path]]:
    workdir = cuda_root / spec.workdir
    translation_units = {(workdir / source).resolve() for source in spec.sources}
    declared = [(workdir / source).resolve() for source in spec.ast_owned_sources]
    claimed_includes: set[Path] = set()
    ownership: dict[Path, list[Path]] = {}
    for relative_source in spec.sources:
        source_path = (workdir / relative_source).resolve()
        discovered = translation_unit_owned_sources(source_path, declared)
        owned = [source_path]
        for candidate in discovered[1:]:
            if candidate in translation_units or candidate in claimed_includes:
                continue
            claimed_includes.add(candidate)
            owned.append(candidate)
        ownership[source_path] = owned
    return ownership


def _zero_cuda_static() -> dict[str, int | bool]:
    return {"cuda_static_source_metrics_success": False, **{field: 0 for field in CUDA_STATIC_FIELDS}}


def empty_static(status: str) -> dict[str, Any]:
    return {**base.empty_static(status), **_zero_cuda_static()}


def collect_static_features(
    spec: base.BenchmarkSpec,
    cuda_root: Path,
    c_compiler: str,
    cxx_compiler: str,
    opt_level: str,
    cuda_dir: Path,
    cuda_arch: str,
) -> tuple[dict[str, Any], list[str]]:
    try:
        core = base.pipeline_core()
    except Exception as exc:
        return empty_static("unavailable"), [f"static pipeline import failed: {exc}"]

    core.AST_DIR = PROJECT_ROOT / "ast" / "rodinia_cuda"
    core.IR_DIR = PROJECT_ROOT / "llvm_ir" / "rodinia_cuda"
    core.CFG_DIR = PROJECT_ROOT / "cfg" / "rodinia_cuda"
    core.ensure_dirs()
    workdir = cuda_root / spec.workdir
    declared = [(workdir / source).resolve() for source in spec.ast_owned_sources]
    ownership = translation_unit_ownership(spec, cuda_root)
    source_features = extract_cuda_source_features(declared)
    common_flags = list(spec.static_flags)
    for include_dir in spec.include_dirs:
        common_flags.extend(["-I", str((workdir / include_dir).resolve())])

    ast_rows: list[dict[str, Any]] = []
    cfg_rows: list[dict[str, Any]] = []
    ir_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    source_success_count = 0
    for relative_source in spec.sources:
        source_path = (workdir / relative_source).resolve()
        if not source_path.is_file():
            errors.append(f"missing static source: {source_path}")
            continue
        program = core.Program(source_path=source_path, source_root=cuda_root)
        owned_paths = ownership.get(source_path, [source_path])
        flags = list(common_flags)
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        is_cuda_translation_unit = (
            source_path.suffix.lower() == ".cu"
            or any(
                Path(match).suffix.lower() in {".cu", ".cuh"}
                for match in CUDA_LOCAL_INCLUDE_RE.findall(source_text)
            )
        )
        if is_cuda_translation_unit:
            flags.extend(
                [
                    "-x",
                    "cuda",
                    f"--cuda-path={cuda_dir}",
                    f"--cuda-gpu-arch={cuda_arch}",
                    "--cuda-host-only",
                    "-nocudalib",
                    "--no-cuda-version-check",
                    "-Wno-unknown-cuda-version",
                ]
            )
        source_ok = True
        try:
            ast_path, ast_ok, ast_error = core.generate_ast(
                program, c_compiler, cxx_compiler, flags
            )
            if ast_ok:
                ast_rows.append(core.extract_ast_features(ast_path, source_path, owned_paths))
            else:
                source_ok = False
                errors.append(f"AST {program.program_id}: {ast_error}")
            ir_path, ir_ok, ir_error = core.generate_ir(
                program, c_compiler, cxx_compiler, opt_level, flags
            )
            if ir_ok:
                ir_rows.append(core.extract_ir_features(ir_path))
            else:
                source_ok = False
                errors.append(f"IR {program.program_id}: {ir_error}")
            cfg_path, cfg_ok, cfg_error, _ = core.generate_cfg(
                program, ir_path, c_compiler, cxx_compiler, flags
            )
            if cfg_ok:
                cfg_rows.append(core.extract_cfg_features(cfg_path))
            else:
                source_ok = False
                errors.append(f"CFG {program.program_id}: {cfg_error}")
        except Exception as exc:
            source_ok = False
            errors.append(f"STATIC {program.program_id}: {exc}")
        if source_ok:
            source_success_count += 1

    if not spec.sources:
        status = "no_sources"
    elif source_success_count == len(spec.sources):
        status = "success"
    else:
        status = "partial"
    return (
        {
            "static_status": status,
            "static_source_count": len(spec.sources),
            "static_source_success_count": source_success_count,
            **base.aggregate_ast(ast_rows),
            **base.aggregate_cfg(cfg_rows),
            **base.aggregate_ir(ir_rows),
            **source_features,
        },
        errors,
    )


def collect_cuda_source_features_only(
    spec: base.BenchmarkSpec,
    cuda_root: Path,
    status: str,
) -> dict[str, Any]:
    workdir = cuda_root / spec.workdir
    declared = [(workdir / source).resolve() for source in spec.ast_owned_sources]
    return {
        **base.empty_static(status),
        **extract_cuda_source_features(declared),
    }


def _toolkit_version(nvcc_version: str) -> str:
    match = re.search(r"\brelease\s+([0-9.]+)", nvcc_version)
    return match.group(1) if match else ""


def incompatible_benchmarks_for_toolkit(
    benchmarks: Iterable[base.BenchmarkSpec],
    toolkit_version: str,
) -> list[base.BenchmarkSpec]:
    match = re.match(r"(\d+)", toolkit_version)
    if not match or int(match.group(1)) < 12:
        return []
    return [
        benchmark
        for benchmark in benchmarks
        if benchmark.benchmark_id in CUDA_12_TEXTURE_REFERENCE_BENCHMARKS
    ]


def _full_version(executable: str) -> str:
    try:
        result = _run_text([executable, "--version"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return (result.stdout or result.stderr).strip()


def collect_environment(
    c_compiler: str,
    cxx_compiler: str,
    env_id: str,
    nvcc: str,
    cuda_dir: Path,
    cuda_arch: str,
    device: GpuDevice,
) -> dict[str, Any]:
    environment = base.collect_environment(c_compiler, cxx_compiler, env_id)
    nvcc_version = _full_version(nvcc)
    environment.update(
        {
            "environment_schema_version": 3,
            "runtime_compiler": "nvcc / gcc / g++ (Rodinia CUDA manifest)",
            "runtime_compiler_version": (
                f"{nvcc_version.splitlines()[-1] if nvcc_version else 'unknown'} | "
                f"{base._compiler_version('gcc')} | {base._compiler_version('g++')}"
            ),
            "gpu_physical_index": device.physical_index,
            "gpu_runtime_index": 0,
            "gpu_uuid": device.uuid,
            "gpu_name": device.name,
            "gpu_compute_capability": device.compute_capability,
            "gpu_total_memory_bytes": device.total_memory_bytes,
            "gpu_driver_version": device.driver_version,
            "gpu_power_limit_watts": device.power_limit_watts or "",
            "cuda_visible_devices": device.uuid,
            "cuda_arch": cuda_arch,
            "cuda_toolkit_root": str(cuda_dir),
            "cuda_toolkit_version": _toolkit_version(nvcc_version),
            "nvcc_path": nvcc,
            "nvcc_version": nvcc_version,
        }
    )
    return environment


def _cuda_include_dir(cuda_dir: Path) -> Path:
    candidates = [
        cuda_dir / "include",
        cuda_dir / "targets" / "x86_64-linux" / "include",
    ]
    for candidate in candidates:
        if (candidate / "cuda_runtime.h").is_file():
            return candidate
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def _cuda_library_dir(cuda_dir: Path) -> Path:
    candidates = [
        cuda_dir / "lib64",
        cuda_dir / "lib",
        cuda_dir / "targets" / "x86_64-linux" / "lib",
        cuda_dir / "targets" / "x86_64-linux" / "lib64",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("libcudart.*")):
            return candidate
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


def infer_cuda_dir(nvcc_path: str) -> Path:
    nvcc = Path(nvcc_path).absolute()
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).resolve()
        try:
            nvcc.relative_to(prefix)
        except ValueError:
            pass
        else:
            return prefix
    return nvcc.resolve().parent.parent


def token_context(nvcc: str, cuda_dir: Path, cuda_arch: str) -> dict[str, str]:
    include_dir = _cuda_include_dir(cuda_dir)
    lib_dir = _cuda_library_dir(cuda_dir)
    return {
        "null_device": "/dev/null",
        "nvcc": nvcc,
        "cuda_dir": str(cuda_dir),
        "cuda_include_dir": str(include_dir),
        "cuda_lib_dir": str(lib_dir),
        "cuda_arch": cuda_arch,
        "compat_include": str(SCRIPT_DIR / "compat_include"),
    }


def configure_cuda_environment(context: dict[str, str]) -> None:
    cuda_dir = context["cuda_dir"]
    os.environ["CUDA_HOME"] = cuda_dir
    os.environ["CUDA_PATH"] = cuda_dir
    for name, value in (
        ("CPATH", context["cuda_include_dir"]),
        ("LIBRARY_PATH", context["cuda_lib_dir"]),
        ("LD_LIBRARY_PATH", context["cuda_lib_dir"]),
    ):
        existing = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
        os.environ[name] = os.pathsep.join([value, *[item for item in existing if item != value]])


def expand_tokens(tokens: Iterable[str], context: dict[str, str]) -> list[str]:
    try:
        return [token.format_map(context) for token in tokens]
    except KeyError as exc:
        raise ValueError(f"unknown manifest command token: {exc.args[0]}") from exc


def build_benchmark(
    spec: base.BenchmarkSpec,
    cuda_root: Path,
    context: dict[str, str],
    clean: bool,
    skip_build: bool,
    build_timeout: float,
) -> tuple[bool, str, list[str]]:
    workdir = cuda_root / spec.workdir
    environment = os.environ.copy()
    environment.update(spec.environment)
    errors: list[str] = []
    build_texts: list[str] = []
    if clean and not skip_build:
        for raw in spec.clean_commands:
            command = expand_tokens(raw, context)
            result = base.run_process(command, workdir, environment, build_timeout, False, False)
            if not result.success:
                print(f"[clean] warning for {spec.benchmark_id}: {result.error_message}")
    if not skip_build:
        for raw in spec.build_commands:
            command = expand_tokens(raw, context)
            build_texts.append(shlex.join(command))
            print(f"[build] {spec.benchmark_id}: {shlex.join(command)}")
            result = base.run_process(command, workdir, environment, build_timeout, False, False)
            if not result.success:
                errors.append(f"BUILD: {result.error_message}")
                return False, " && ".join(build_texts), errors
    else:
        build_texts.append("skipped (--skip-build)")
    executable = (workdir / spec.executable).resolve()
    if not executable.is_file():
        errors.append(f"BUILD: executable was not produced: {executable}")
        return False, " && ".join(build_texts), errors
    return True, " && ".join(build_texts), errors


def resolve_cuda_arch(requested: str, device: GpuDevice) -> str:
    value = requested.strip().lower()
    if value == "auto":
        digits = re.sub(r"\D", "", device.compute_capability)
        if not digits:
            raise ValueError(
                "this nvidia-smi version does not expose compute capability; "
                "pass --cuda-arch sm_XX explicitly"
            )
        value = f"sm_{digits}"
    if not re.fullmatch(r"sm_\d{2,3}", value):
        raise ValueError("CUDA architecture must be auto or sm_XX, for example sm_86")
    return value


def validate_nvcc_arch(nvcc: str, cuda_arch: str) -> str:
    try:
        result = _run_text([nvcc, "--list-gpu-code"])
    except (OSError, subprocess.TimeoutExpired):
        return ""
    supported = set(re.findall(r"sm_\d{2,3}", result.stdout))
    if result.returncode == 0 and supported and cuda_arch not in supported:
        return (
            f"nvcc does not list {cuda_arch} as supported; available SASS targets: "
            f"{', '.join(sorted(supported))}"
        )
    return ""


def _dummy_device(index: int) -> GpuDevice:
    return GpuDevice(index, "dry-run", "dry-run GPU", "", 0, "", None)


def print_manifest(benchmarks: tuple[base.BenchmarkSpec, ...]) -> None:
    print("Rodinia CUDA benchmark manifest")
    for benchmark in benchmarks:
        profiles = ",".join(sorted({item.profile for item in benchmark.inputs})) or "-"
        state = "enabled" if benchmark.enabled else f"disabled: {benchmark.disabled_reason}"
        print(f"  {benchmark.benchmark_id:<22} {state:<70} profiles={profiles}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and measure Rodinia 3.1 CUDA benchmarks on an NVIDIA Linux host."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rodinia-root", type=Path, default=DEFAULT_RODINIA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--input-profile", choices=("smoke", "official", "all"), default="smoke")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--measurement-seconds", type=float, default=10.0)
    parser.add_argument("--min-measured-runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--build-timeout", type=float, default=1200.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuda-dir", type=Path, default=None)
    parser.add_argument("--cuda-arch", default="auto")
    parser.add_argument("--gpu-index", type=int, default=0, help="Physical nvidia-smi GPU index.")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--gpu-metrics", choices=("nvidia-smi", "off"), default="nvidia-smi")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Allow formal measurement while another compute process uses the selected GPU.",
    )
    parser.add_argument("--c-compiler", default="clang")
    parser.add_argument("--cxx-compiler", default="clang++")
    parser.add_argument("--opt-level", choices=("O0", "O1", "O2", "O3"), default="O2")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090")
    parser.add_argument("--prometheus-exporter", choices=("auto", "node", "off"), default="auto")
    parser.add_argument("--prometheus-job", default=None)
    parser.add_argument("--prometheus-instance", default="")
    parser.add_argument("--prometheus-query-step", type=float, default=1.0)
    parser.add_argument("--prometheus-query-timeout", type=float, default=10.0)
    parser.add_argument("--prometheus-query-delay", type=float, default=2.0)
    parser.add_argument("--prometheus-context-seconds", type=float, default=5.0)
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument(
        "--collect-static",
        dest="collect_static",
        action="store_true",
        help="Also attempt generic LLVM AST/IR/CFG extraction for CUDA translation units.",
    )
    static_group.add_argument(
        "--no-static",
        dest="collect_static",
        action="store_false",
        help="Collect CUDA source metrics only (default).",
    )
    parser.set_defaults(collect_static=False)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs < 1 or args.min_measured_runs < 1 or args.measurement_seconds < 0:
        parser.error("warmup must be >= 0, runs/min-measured-runs >= 1, and measurement-seconds >= 0")
    if args.timeout <= 0 or args.build_timeout <= 0 or args.gpu_sample_interval <= 0:
        parser.error("timeouts and GPU sample interval must be positive")
    if (
        args.prometheus_query_step <= 0
        or args.prometheus_query_timeout <= 0
        or args.prometheus_query_delay < 0
        or args.prometheus_context_seconds < 0
    ):
        parser.error("Prometheus step/timeout must be positive and wait values must be >= 0")
    if (
        args.gpu_metrics != "off"
        and not args.build_only
        and not args.dry_run
        and args.prometheus_context_seconds <= 0
    ):
        parser.error("three-phase GPU metrics require --prometheus-context-seconds > 0")
    if args.all and args.benchmark:
        parser.error("use either --all or --benchmark, not both")
    return args


def _not_run_metrics(context_seconds: float, gpu_interval: float) -> dict[str, Any]:
    return {
        "warmup_runs_completed": 0,
        "measurement_seconds_actual": 0,
        "measured_runs": 0,
        "runtime_sec_median": 0,
        "runtime_sec_mean": 0,
        "runtime_sec_std": 0,
        "runtime_sec_cv": 0,
        "runtime_sec_min": 0,
        "runtime_sec_max": 0,
        "process_metric_aggregation": "",
        **{field: "" for field in base.PROCESS_FIELDS},
        "process_max_rss_bytes_peak": "",
        "process_metrics_backend": "not_run",
        **base.pipeline_core().empty_prometheus_three_phase_result(
            None, "Benchmark measurement was not executed", context_seconds
        ),
        **empty_gpu_result(None, "Benchmark measurement was not executed", gpu_interval),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"[pipeline] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        benchmarks = base.load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[pipeline] invalid manifest: {exc}", file=sys.stderr)
        return 2
    if args.list:
        print_manifest(benchmarks)
        return 0
    try:
        selected = base.selected_benchmarks(benchmarks, args.benchmark, args.all)
    except ValueError as exc:
        print(f"[pipeline] {exc}", file=sys.stderr)
        return 2

    rodinia_root = args.rodinia_root.resolve()
    cuda_root = rodinia_root / "cuda"
    if not cuda_root.is_dir():
        print(f"[pipeline] Rodinia CUDA directory not found: {cuda_root}", file=sys.stderr)
        return 2
    if os.name != "posix" and not args.dry_run:
        print("[pipeline] CUDA runtime collection must execute on Linux; use --dry-run on Windows", file=sys.stderr)
        return 2

    nvcc = shutil.which(args.nvcc) or args.nvcc
    nvidia_smi = shutil.which(args.nvidia_smi) or args.nvidia_smi
    if args.cuda_dir is not None:
        cuda_dir = args.cuda_dir.resolve()
    elif shutil.which(args.nvcc):
        cuda_dir = infer_cuda_dir(shutil.which(args.nvcc) or args.nvcc)
    else:
        cuda_dir = Path("/usr/local/cuda")

    device = _dummy_device(args.gpu_index)
    compatibility_skips: list[dict[str, str]] = []
    if args.dry_run:
        if args.cuda_arch == "auto":
            print("[pipeline] dry-run uses placeholder sm_70; pass --cuda-arch sm_XX to inspect exact commands")
            cuda_arch = "sm_70"
        else:
            try:
                cuda_arch = resolve_cuda_arch(args.cuda_arch, device)
            except ValueError as exc:
                print(f"[pipeline] {exc}", file=sys.stderr)
                return 2
    else:
        if not shutil.which(args.nvcc) and not Path(args.nvcc).is_file():
            print(f"[pipeline] nvcc not found: {args.nvcc}", file=sys.stderr)
            return 2
        if not shutil.which(args.nvidia_smi) and not Path(args.nvidia_smi).is_file():
            print(f"[pipeline] nvidia-smi not found: {args.nvidia_smi}", file=sys.stderr)
            return 2
        try:
            devices = query_gpu_devices(nvidia_smi)
            device = next(item for item in devices if item.physical_index == args.gpu_index)
            cuda_arch = resolve_cuda_arch(args.cuda_arch, device)
        except (RuntimeError, StopIteration, ValueError) as exc:
            message = (
                f"GPU index {args.gpu_index} was not returned by nvidia-smi"
                if isinstance(exc, StopIteration)
                else str(exc)
            )
            print(f"[pipeline] GPU preflight failed: {message}", file=sys.stderr)
            return 2
        arch_error = validate_nvcc_arch(nvcc, cuda_arch)
        if arch_error:
            print(f"[pipeline] CUDA preflight failed: {arch_error}", file=sys.stderr)
            return 2
        toolkit_version = _toolkit_version(_full_version(nvcc))
        incompatible = incompatible_benchmarks_for_toolkit(selected, toolkit_version)
        if incompatible:
            names = ", ".join(benchmark.benchmark_id for benchmark in incompatible)
            reason = (
                f"CUDA Toolkit {toolkit_version or '12+'} removed the texture-reference API "
                "used by these Rodinia 3.1 sources"
            )
            if not args.all:
                print(f"[pipeline] incompatible benchmark(s): {names}: {reason}", file=sys.stderr)
                return 2
            for benchmark in incompatible:
                compatibility_skips.append(
                    {"benchmark": benchmark.benchmark_id, "reason": reason}
                )
                print(f"[pipeline] compatibility skip: {benchmark.benchmark_id}: {reason}")
            incompatible_ids = {benchmark.benchmark_id for benchmark in incompatible}
            selected = [
                benchmark
                for benchmark in selected
                if benchmark.benchmark_id not in incompatible_ids
            ]
        if not args.build_only:
            try:
                busy_processes = query_compute_processes(nvidia_smi, device.uuid)
            except RuntimeError as exc:
                print(f"[pipeline] GPU process preflight failed: {exc}", file=sys.stderr)
                return 2
            if busy_processes and not args.allow_busy_gpu:
                details = "; ".join(
                    f"pid={item['pid']} name={item['process_name']} memory={item['used_gpu_memory_mib']} MiB"
                    for item in busy_processes
                )
                print(
                    "[pipeline] selected GPU already has compute processes: " + details,
                    file=sys.stderr,
                )
                print("[pipeline] wait for an idle GPU or use --allow-busy-gpu only for non-formal testing", file=sys.stderr)
                return 2
        print(
            f"[gpu] physical={device.physical_index}, runtime=0, name={device.name}, "
            f"compute_capability={device.compute_capability}, arch={cuda_arch}"
        )

    context = token_context(nvcc, cuda_dir, cuda_arch)
    configure_cuda_environment(context)
    if args.dry_run:
        any_failure = False
        for spec in selected:
            print(f"[pipeline] {spec.benchmark_id} in {cuda_root / spec.workdir}")
            for raw in spec.build_commands:
                print(f"  build: {shlex.join(expand_tokens(raw, context))}")
            for item in base.input_specs(spec, args.input_profile):
                errors = base.validate_paths(spec, item, cuda_root, False)
                command = [f"./{spec.executable}", *base.expand_tokens(item.args)]
                print(f"  run[{item.input_id}]: {shlex.join(command)}")
                for error in errors:
                    print(f"  ERROR: {error}")
                    any_failure = True
        print("[pipeline] dry run completed")
        return 1 if any_failure else 0

    if not args.build_only:
        try:
            process_probe = base.ensure_process_probe()
        except RuntimeError as exc:
            print(f"[pipeline] {exc}", file=sys.stderr)
            return 2
        print(f"[pipeline] process metrics probe: {process_probe}")

    os.environ["CUDA_VISIBLE_DEVICES"] = device.uuid
    args.env_id = args.env_id or base.pipeline_core().default_env_id()
    args.prometheus_exporter = "node" if args.prometheus_exporter == "auto" else args.prometheus_exporter
    if args.prometheus_job is None:
        args.prometheus_job = "node" if args.prometheus_exporter == "node" else ""
    host_queries = base.pipeline_core().prometheus_host_queries(
        args.prometheus_exporter, args.prometheus_job, args.prometheus_instance
    )
    context_seconds = (
        args.prometheus_context_seconds
        if host_queries or args.gpu_metrics == "nvidia-smi"
        else 0.0
    )
    environment = collect_environment(
        args.c_compiler,
        args.cxx_compiler,
        args.env_id,
        nvcc,
        cuda_dir,
        cuda_arch,
        device,
    )
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    collected_at = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    any_failure = False

    for spec in selected:
        inputs = base.input_specs(spec, args.input_profile)
        if not inputs:
            print(f"[pipeline] warning: {spec.benchmark_id} has no {args.input_profile} input")
            continue
        print(f"[pipeline] processing {spec.benchmark_id} in {cuda_root / spec.workdir}")
        build_success, build_command, build_errors = build_benchmark(
            spec, cuda_root, context, args.clean, args.skip_build, args.build_timeout
        )
        if args.collect_static:
            static_features, static_errors = collect_static_features(
                spec,
                cuda_root,
                args.c_compiler,
                args.cxx_compiler,
                args.opt_level,
                cuda_dir,
                cuda_arch,
            )
        else:
            static_features = collect_cuda_source_features_only(
                spec, cuda_root, "generic_llvm_not_requested"
            )
            static_errors = []

        for item in inputs:
            errors = [*build_errors, *static_errors]
            path_errors = base.validate_paths(spec, item, cuda_root, build_success)
            input_busy_processes: list[dict[str, str]] = []
            gpu_idle_preflight_passed: bool | str = ""
            gpu_preexisting_process_count: int | str = ""
            gpu_preexisting_processes = ""
            if build_success and not path_errors and not args.build_only:
                try:
                    input_busy_processes = query_compute_processes(nvidia_smi, device.uuid)
                except RuntimeError as exc:
                    path_errors.append(f"GPU_PREFLIGHT: {exc}")
                    gpu_idle_preflight_passed = False
                else:
                    gpu_idle_preflight_passed = not input_busy_processes
                    gpu_preexisting_process_count = len(input_busy_processes)
                    gpu_preexisting_processes = json.dumps(
                        input_busy_processes,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if input_busy_processes and not args.allow_busy_gpu:
                        details = "; ".join(
                            f"pid={process['pid']} name={process['process_name']} "
                            f"memory={process['used_gpu_memory_mib']} MiB"
                            for process in input_busy_processes
                        )
                        path_errors.append(
                            "GPU_BUSY: selected GPU gained another compute process before "
                            f"this input ({details})"
                        )
            errors.extend(path_errors)
            run_display = [f"./{spec.executable}", *base.expand_tokens(item.args)]
            row: dict[str, Any] = {
                "session_id": session_id,
                "collected_at": collected_at,
                "dataset": "Rodinia-3.1",
                "parallel_model": "CUDA",
                "program_id": spec.benchmark_id,
                "input_id": item.input_id,
                "input_profile": item.profile,
                "input_size_parameters": json.dumps(
                    item.parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "host_thread_count": item.threads,
                "gpu_idle_preflight_passed": gpu_idle_preflight_passed,
                "gpu_preexisting_compute_process_count": gpu_preexisting_process_count,
                "gpu_preexisting_compute_processes": gpu_preexisting_processes,
                "workdir": spec.workdir,
                "executable": spec.executable,
                "source_files": ";".join(spec.sources),
                "source_file_count": len(spec.sources),
                "ast_owned_source_files": ";".join(spec.ast_owned_sources),
                "ast_owned_source_count": len(spec.ast_owned_sources),
                "build_command": build_command,
                "run_command": shlex.join(run_display),
                "warmup_runs_requested": args.warmup,
                "measurement_seconds_requested": args.measurement_seconds,
                "min_measured_runs_requested": args.min_measured_runs,
                "build_success": build_success,
                **environment,
                **static_features,
            }
            if build_success and not path_errors and not args.build_only:
                print(
                    f"[run] {spec.benchmark_id}[{item.input_id}] warmup={args.warmup}, "
                    f"budget={args.measurement_seconds:g}s"
                )
                phase_windows: dict[str, tuple[float, float]] = {}
                sampler: NvidiaSmiSampler | None = None
                if args.gpu_metrics == "nvidia-smi":
                    sampler = NvidiaSmiSampler(
                        nvidia_smi, device.physical_index, args.gpu_sample_interval
                    )
                    sampler.start()
                    if not sampler.wait_for_first_sample(
                        max(2.0, args.gpu_sample_interval * 4)
                    ):
                        sampler.stop()
                        _, startup_errors = sampler.snapshot()
                        errors.append(
                            "GPU_METRICS: nvidia-smi sampler did not produce an initial sample"
                            + (f" ({' | '.join(startup_errors)})" if startup_errors else "")
                        )
                        sampler = None
                try:
                    measured, details, run_errors = base.measure_input(
                        session_id,
                        spec,
                        item,
                        cuda_root,
                        args.warmup,
                        args.runs,
                        args.measurement_seconds,
                        args.min_measured_runs,
                        args.timeout,
                        phase_windows,
                        context_seconds,
                    )
                finally:
                    if sampler is not None:
                        sampler.stop()
                row.update(measured)
                for detail in details:
                    detail.update(
                        {
                            "gpu_physical_index": device.physical_index,
                            "gpu_uuid": device.uuid,
                            "cuda_arch": cuda_arch,
                        }
                    )
                run_rows.extend(details)
                errors.extend(run_errors)

                if sampler is not None:
                    samples, sampler_errors = sampler.snapshot()
                    gpu_result = summarize_gpu_samples(
                        samples, phase_windows, args.gpu_sample_interval, sampler_errors
                    )
                    row.update(gpu_result)
                    if not gpu_result["gpu_three_phase_collection_success"]:
                        errors.append(
                            "GPU_METRICS: " + str(gpu_result["gpu_three_phase_collection_error"])
                        )
                else:
                    row.update(
                        empty_gpu_result(
                            phase_windows,
                            (
                                "nvidia-smi GPU metric collection is disabled"
                                if args.gpu_metrics == "off"
                                else "nvidia-smi sampler failed to start"
                            ),
                            args.gpu_sample_interval,
                        )
                    )

                if host_queries and phase_windows:
                    if args.prometheus_query_delay > 0:
                        time.sleep(args.prometheus_query_delay)
                    try:
                        row.update(
                            base.pipeline_core().collect_prometheus_three_phase_metrics(
                                args.prometheus_url,
                                phase_windows,
                                args.prometheus_query_step,
                                args.prometheus_query_timeout,
                                host_queries,
                                context_seconds,
                            )
                        )
                        if not row.get("prometheus_three_phase_collection_success"):
                            errors.append(
                                "PROMETHEUS: "
                                + str(row.get("prometheus_three_phase_collection_error", "unknown error"))
                            )
                    except Exception as exc:
                        message = str(exc)[:500]
                        row.update(
                            base.pipeline_core().empty_prometheus_three_phase_result(
                                phase_windows, message, context_seconds
                            )
                        )
                        errors.append(f"PROMETHEUS: {message}")
                else:
                    row.update(
                        base.pipeline_core().empty_prometheus_three_phase_result(
                            phase_windows or None,
                            "Prometheus host metric collection is disabled",
                            context_seconds,
                        )
                    )
            else:
                row.update(_not_run_metrics(context_seconds, args.gpu_sample_interval))
                row["run_success"] = "not_run" if args.build_only and build_success else False

            row["error_message"] = " | ".join(error for error in errors if error)[-12000:]
            summary_rows.append(row)
            metrics_failed = (
                args.gpu_metrics != "off"
                and not args.build_only
                and not row.get("gpu_three_phase_collection_success")
            )
            host_metrics_failed = (
                bool(host_queries)
                and not args.build_only
                and not row.get("prometheus_three_phase_collection_success")
            )
            if (
                not build_success
                or (not args.build_only and not row.get("run_success"))
                or metrics_failed
                or host_metrics_failed
            ):
                any_failure = True

    results_dir = args.results_dir.resolve()
    summary_path = results_dir / f"rodinia_cuda_summary_{session_id}.csv"
    runs_path = results_dir / f"rodinia_cuda_runs_{session_id}.csv"
    environment_path = results_dir / f"rodinia_cuda_environment_{session_id}.json"
    base.write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    base.write_csv(runs_path, run_rows, RUN_FIELDS)
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "collected_at": collected_at,
                "manifest": str(manifest_path),
                "rodinia_root": str(rodinia_root),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "environment": environment,
                "benchmarks": [spec.benchmark_id for spec in selected],
                "compatibility_skips": compatibility_skips,
                "gpu_metric_scope_note": (
                    "nvidia-smi telemetry is scoped to the selected physical GPU, not to one PID; "
                    "the idle-GPU preflight is therefore part of the measurement protocol"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[pipeline] summary: {summary_path}")
    print(f"[pipeline] run details: {runs_path}")
    print(f"[pipeline] environment: {environment_path}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
