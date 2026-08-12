#!/usr/bin/env python3
"""Collect PolyBenchC static and runtime features on Windows or Linux."""

from __future__ import annotations

import argparse
import ctypes
import csv
import fnmatch
import json
import math
import os
import platform
import re
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from ctypes import wintypes
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
except ImportError as exc:
    raise SystemExit(
        "Prometheus monitoring requires prometheus-client. "
        "Install it with: python -m pip install prometheus-client"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets" / "test"
AST_DIR = PROJECT_ROOT / "ast"
IR_DIR = PROJECT_ROOT / "llvm_ir"
CFG_DIR = PROJECT_ROOT / "cfg"
RESULTS_DIR = PROJECT_ROOT / "results"
BUILD_DIR = PROJECT_ROOT / "build" / "pipeline"
COMPAT_INCLUDE_DIR = PROJECT_ROOT / "scripts" / "compat_include"
PROCESS_PROBE_SOURCE = SCRIPT_DIR / "rodinia_process_probe.c"
PROCESS_PROBE_BINARY = PROJECT_ROOT / "build" / "polybench_process_probe"
_PROCESS_PROBE_PATH: Path | None = None

SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx"}
POLYBENCH_DATASET_PROFILES = (
    "MINI_DATASET",
    "SMALL_DATASET",
    "MEDIUM_DATASET",
    "LARGE_DATASET",
    "EXTRALARGE_DATASET",
)


PIPELINE_RUNNING = Gauge(
    "llvm_pipeline_running",
    "Whether the LLVM dataset pipeline is currently running.",
    ["dataset"],
)
PIPELINE_PROGRAMS_TOTAL = Gauge(
    "llvm_pipeline_programs_total",
    "Number of benchmark configurations in the current pipeline run.",
    ["dataset"],
)
PIPELINE_PROGRAMS_COMPLETED = Gauge(
    "llvm_pipeline_programs_completed",
    "Number of benchmark configurations completed in the current pipeline run.",
    ["dataset"],
)
PIPELINE_PROGRESS_RATIO = Gauge(
    "llvm_pipeline_progress_ratio",
    "Fraction of benchmark configurations completed in the current pipeline run.",
    ["dataset"],
)
PIPELINE_PROGRAM_RESULTS = Counter(
    "llvm_pipeline_program_results_total",
    "Number of benchmark configurations processed by result.",
    ["dataset", "status"],
)
PIPELINE_PROGRAM_DURATION = Histogram(
    "llvm_pipeline_program_duration_seconds",
    "Time spent processing one benchmark configuration.",
    ["dataset", "status"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
)
PIPELINE_BENCHMARK_RUNS = Counter(
    "llvm_pipeline_benchmark_runs_total",
    "Number of benchmark executable runs by phase and result.",
    ["dataset", "phase", "status"],
)
PIPELINE_BENCHMARK_DURATION = Histogram(
    "llvm_pipeline_benchmark_duration_seconds",
    "Observed duration of one benchmark executable run.",
    ["dataset", "phase"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
PIPELINE_STAGE_ERRORS = Counter(
    "llvm_pipeline_stage_errors_total",
    "Number of pipeline stage failures.",
    ["dataset", "stage"],
)
PIPELINE_LAST_RUN_SUCCESS = Gauge(
    "llvm_pipeline_last_run_success",
    "Whether all benchmark configurations completed successfully.",
    ["dataset"],
)

LEGACY_PROMETHEUS_RESULT_FIELDS = [
    "prometheus_window_start_timestamp",
    "prometheus_window_end_timestamp",
    "prometheus_window_duration_seconds",
    "prometheus_sample_count",
    "host_cpu_usage_pct_mean",
    "host_cpu_usage_pct_max",
    "host_memory_used_bytes_mean",
    "host_memory_used_bytes_max",
    "host_disk_read_bytes_delta",
    "host_disk_write_bytes_delta",
    "host_network_bytes_delta",
    "prometheus_collection_success",
    "prometheus_collection_error",
]

PROMETHEUS_PHASES = ("before", "during", "after")
PROMETHEUS_PHASE_FIELDS = [
    field
    for phase in PROMETHEUS_PHASES
    for field in (
        f"prometheus_{phase}_window_start_timestamp",
        f"prometheus_{phase}_window_end_timestamp",
        f"prometheus_{phase}_window_duration_seconds",
        f"prometheus_{phase}_sample_count",
        f"host_cpu_usage_pct_{phase}_mean",
        f"host_cpu_usage_pct_{phase}_max",
        f"host_memory_used_bytes_{phase}_mean",
        f"host_memory_used_bytes_{phase}_max",
        f"host_disk_read_bytes_{phase}_delta",
        f"host_disk_write_bytes_{phase}_delta",
        f"host_network_bytes_{phase}_delta",
        f"prometheus_{phase}_collection_success",
        f"prometheus_{phase}_collection_error",
    )
]
PROMETHEUS_BACKGROUND_FIELDS = [
    "prometheus_sampling_scheme",
    "prometheus_context_seconds_requested",
    "prometheus_background_correction_method",
    "prometheus_background_cpu_usage_pct_mean",
    "prometheus_background_memory_used_bytes_mean",
    "prometheus_background_disk_read_bytes_per_sec",
    "prometheus_background_disk_write_bytes_per_sec",
    "prometheus_background_network_bytes_per_sec",
    "prometheus_background_cpu_before_after_abs_diff_pct",
    "prometheus_background_memory_before_after_abs_diff_bytes",
    "prometheus_background_disk_read_rate_abs_diff_bytes_per_sec",
    "prometheus_background_disk_write_rate_abs_diff_bytes_per_sec",
    "prometheus_background_network_rate_abs_diff_bytes_per_sec",
    "host_cpu_usage_pct_background_adjusted_mean",
    "host_cpu_usage_pct_background_adjusted_max",
    "host_memory_used_bytes_background_adjusted_mean",
    "host_memory_used_bytes_background_adjusted_max",
    "host_disk_read_bytes_background_adjusted_delta",
    "host_disk_write_bytes_background_adjusted_delta",
    "host_network_bytes_background_adjusted_delta",
    "prometheus_three_phase_collection_success",
    "prometheus_three_phase_collection_error",
]
PROMETHEUS_RESULT_FIELDS = [
    *LEGACY_PROMETHEUS_RESULT_FIELDS,
    *PROMETHEUS_PHASE_FIELDS,
    *PROMETHEUS_BACKGROUND_FIELDS,
]

PROMETHEUS_HOST_QUERY_TEMPLATES = {
    "windows": {
        "cpu": "windows_cpu_time_total{selector}",
        "memory_used": (
            "windows_memory_physical_total_bytes{selector} "
            "- windows_memory_available_bytes{selector}"
        ),
        "disk_read": "windows_logical_disk_read_bytes_total{selector}",
        "disk_write": "windows_logical_disk_write_bytes_total{selector}",
        "network": "windows_net_bytes_total{selector}",
    },
    "node": {
        "cpu": "node_cpu_seconds_total{selector}",
        "memory_used": (
            "node_memory_MemTotal_bytes{selector} "
            "- node_memory_MemAvailable_bytes{selector}"
        ),
        "disk_read": "node_disk_read_bytes_total{selector}",
        "disk_write": "node_disk_written_bytes_total{selector}",
        "network": (
            "node_network_receive_bytes_total{selector} "
            "+ node_network_transmit_bytes_total{selector}"
        ),
    },
}

PROCESS_RUN_FIELDS = [
    "process_id",
    "process_cpu_user_sec",
    "process_cpu_system_sec",
    "process_cpu_total_sec",
    "process_max_rss_bytes",
    "process_major_page_faults",
    "process_minor_page_faults",
    "process_fs_inputs",
    "process_fs_outputs",
    "process_metrics_backend",
    # Retained for compatibility with previously collected Windows CSV files.
    "process_cpu_kernel_sec",
    "process_peak_working_set_bytes",
    "process_peak_private_bytes",
    "process_page_faults",
    "process_read_bytes",
    "process_write_bytes",
    "process_other_bytes",
    "process_metrics_success",
    "process_metrics_error",
]

MEASUREMENT_SUMMARY_FIELDS = [
    "measurement_target_seconds",
    "measurement_actual_seconds",
    "min_measured_runs_requested",
    "successful_runs",
    "failed_runs",
    "process_metrics_sampled_runs",
    "process_metric_aggregation",
    "process_cpu_user_sec",
    "process_cpu_system_sec",
    "process_max_rss_bytes",
    "process_major_page_faults",
    "process_minor_page_faults",
    "process_fs_inputs",
    "process_fs_outputs",
    "process_max_rss_bytes_peak",
    "process_metrics_backend",
    "process_cpu_user_sec_total",
    "process_cpu_kernel_sec_total",
    "process_cpu_total_sec_total",
    "process_cpu_total_sec_mean",
    "process_peak_working_set_bytes_max",
    "process_peak_private_bytes_max",
    "process_page_faults_total",
    "process_read_bytes_total",
    "process_write_bytes_total",
    "process_other_bytes_total",
]

STATIC_ENVIRONMENT_FIELDS = [
    "environment_schema_version",
    "env_id",
    "host_name",
    "os_system",
    "os_release",
    "os_version",
    "cpu_model",
    "cpu_vendor",
    "cpu_identifier",
    "cpu_architecture",
    "cpu_socket_count",
    "cpu_physical_core_count",
    "cpu_logical_core_count",
    "cpu_threads_per_core",
    "cpu_nominal_frequency_mhz",
    "cpu_l1_cache_bytes",
    "cpu_l2_cache_bytes",
    "cpu_l3_cache_bytes",
    "cpu_cache_legacy_scope",
    "cpu_l1_cache_bytes_per_instance",
    "cpu_l1_cache_instance_count",
    "cpu_l1_cache_bytes_total",
    "cpu_l1d_cache_bytes_per_instance",
    "cpu_l1d_cache_instance_count",
    "cpu_l1d_cache_bytes_total",
    "cpu_l1i_cache_bytes_per_instance",
    "cpu_l1i_cache_instance_count",
    "cpu_l1i_cache_bytes_total",
    "cpu_l2_cache_bytes_per_instance",
    "cpu_l2_cache_instance_count",
    "cpu_l2_cache_bytes_total",
    "cpu_l3_cache_bytes_per_instance",
    "cpu_l3_cache_instance_count",
    "cpu_l3_cache_bytes_total",
    "memory_total_bytes",
    "representation_compiler",
    "representation_compiler_version",
    "runtime_compiler",
    "runtime_compiler_version",
    "opt_version",
]


AST_FIELDS = [
    "ast_node_count",
    "function_count",
    "call_expr_count",
    "recursive_call_flag",
    "for_count",
    "while_count",
    "do_while_count",
    "max_loop_depth",
    "loop_count_total",
    "if_count",
    "switch_count",
    "conditional_operator_count",
    "binary_operator_count",
    "arithmetic_operator_count",
    "comparison_operator_count",
    "array_subscript_count",
    "pointer_deref_count",
    "malloc_free_count",
    "integer_literal_count",
    "max_integer_literal",
    "avg_integer_literal",
    "float_literal_count",
    "pragma_omp_count",
    "cuda_kernel_decl_count",
]

CFG_FIELDS = [
    "cfg_function_count",
    "basic_block_count",
    "cfg_edge_count",
    "cyclomatic_complexity",
    "branch_block_count",
    "merge_block_count",
    "avg_in_degree",
    "avg_out_degree",
    "max_out_degree",
    "loop_backedge_count",
    "natural_loop_count",
    "max_cfg_loop_depth",
    "scc_count",
    "cfg_depth",
    "unreachable_block_count",
    "critical_edge_count",
]

IR_FIELDS = [
    "ir_instruction_count",
    "ir_line_count",
    "load_count",
    "store_count",
    "memory_inst_count",
    "getelementptr_count",
    "alloca_count",
    "memcpy_count",
    "memset_count",
    "call_count_ir",
    "external_call_count",
    "branch_count",
    "switch_count_ir",
    "phi_count",
    "select_count",
    "icmp_count",
    "fcmp_count",
    "add_count",
    "sub_count",
    "mul_count",
    "div_count",
    "rem_count",
    "fadd_count",
    "fsub_count",
    "fmul_count",
    "fdiv_count",
    "vector_inst_count",
    "atomic_count",
    "barrier_or_sync_count",
    "constant_count_ir",
    "max_constant_ir",
    "avg_constant_ir",
    "load_store_ratio",
    "memory_arithmetic_ratio",
    "branch_instruction_ratio",
]

INPUT_SCALE_FIELDS = [
    "input_size_profile",
    "input_size_parameters",
    "graph_nodes",
    "graph_edges",
    "image_width",
    "image_height",
    "iterations",
]

STATUS_FIELDS = [
    "ast_success",
    "ir_success",
    "cfg_success",
    "build_success",
    "run_success",
    "error_message",
]


@dataclass(frozen=True)
class Program:
    source_path: Path
    source_root: Path
    input_profile: str = ""

    @property
    def relative_path(self) -> Path:
        return self.source_path.relative_to(self.source_root)

    @property
    def program_id(self) -> str:
        return self.relative_path.as_posix()

    @property
    def output_stem(self) -> Path:
        stem = self.relative_path.with_suffix("")
        if not self.input_profile:
            return stem
        profile_name = self.input_profile.lower().removesuffix("_dataset")
        return stem.parent / f"{stem.name}__{profile_name}"

    @property
    def language(self) -> str:
        suffix = self.source_path.suffix.lower()
        if suffix == ".c":
            return "C"
        return "C++"


def ensure_dirs() -> None:
    for directory in [AST_DIR, IR_DIR, CFG_DIR, RESULTS_DIR, BUILD_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout,
    )


def command_text(cmd: list[str]) -> str:
    text = command_output(cmd)
    return text.splitlines()[0] if text and not text.startswith("unavailable:") else text


def command_output(cmd: list[str]) -> str:
    try:
        result = run_command(cmd, timeout=10)
    except Exception as exc:
        return f"unavailable: {exc}"
    return (result.stdout or result.stderr or "").strip() or "unavailable"


def llvm_opt_version() -> str:
    if not shutil.which("opt"):
        return "unavailable"
    output = command_output(["opt", "--version"])
    match = re.search(r"\bLLVM version\s+([^\s]+)", output, re.IGNORECASE)
    return match.group(1) if match else command_text(["opt", "--version"])


_COMPILER_VERSION_CACHE: dict[str, str] = {}


def compiler_version(command: str) -> str:
    if command not in _COMPILER_VERSION_CACHE:
        _COMPILER_VERSION_CACHE[command] = command_text([command, "--version"])
    return _COMPILER_VERSION_CACHE[command]


def windows_registry_cpu_info() -> dict[str, object]:
    if os.name != "nt":
        return {}
    try:
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            def registry_value(name: str) -> object:
                try:
                    return winreg.QueryValueEx(key, name)[0]
                except OSError:
                    return ""

            return {
                "cpu_model": str(registry_value("ProcessorNameString")).strip(),
                "cpu_vendor": str(registry_value("VendorIdentifier")).strip(),
                "cpu_identifier": str(registry_value("Identifier")).strip(),
                "cpu_nominal_frequency_mhz": registry_value("~MHz"),
            }
    except (ImportError, OSError):
        return {}


def windows_cpu_topology() -> dict[str, object]:
    empty = {
        "cpu_socket_count": "",
        "cpu_physical_core_count": "",
        "cpu_l1_cache_bytes": "",
        "cpu_l2_cache_bytes": "",
        "cpu_l3_cache_bytes": "",
        **_empty_cache_topology(),
    }
    if os.name != "nt":
        return empty
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_topology = kernel32.GetLogicalProcessorInformationEx
        get_topology.argtypes = [
            wintypes.DWORD,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.DWORD),
        ]
        get_topology.restype = wintypes.BOOL

        relation_all = 0xFFFF
        required_bytes = wintypes.DWORD(0)
        get_topology(relation_all, None, ctypes.byref(required_bytes))
        if required_bytes.value == 0:
            return empty
        buffer = ctypes.create_string_buffer(required_bytes.value)
        if not get_topology(relation_all, buffer, ctypes.byref(required_bytes)):
            return empty

        raw = buffer.raw[: required_bytes.value]
        offset = 0
        physical_cores = 0
        sockets = 0
        cache_instances: dict[tuple[int, int], list[int]] = {}
        while offset + 8 <= len(raw):
            relationship = int.from_bytes(raw[offset : offset + 4], "little")
            record_size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
            if record_size < 8 or offset + record_size > len(raw):
                break
            if relationship == 0:
                physical_cores += 1
            elif relationship == 2 and record_size >= 20:
                cache_level = raw[offset + 8]
                size = int.from_bytes(raw[offset + 12 : offset + 16], "little")
                cache_type = int.from_bytes(raw[offset + 16 : offset + 20], "little")
                if cache_level in (1, 2, 3):
                    cache_instances.setdefault((cache_level, cache_type), []).append(size)
            elif relationship == 3:
                sockets += 1
            offset += record_size

        cache_bytes = {
            level: sum(
                size
                for (instance_level, _), sizes in cache_instances.items()
                if instance_level == level
                for size in sizes
            )
            for level in (1, 2, 3)
        }
        cache_topology = _empty_cache_topology()

        def write_group(prefix: str, values: list[int]) -> None:
            cache_topology[f"{prefix}_bytes_per_instance"] = values[0] if values else ""
            cache_topology[f"{prefix}_instance_count"] = len(values) if values else ""
            cache_topology[f"{prefix}_bytes_total"] = sum(values) if values else ""

        l1_data = cache_instances.get((1, 2), [])
        l1_instruction = cache_instances.get((1, 1), [])
        all_l1 = [
            size
            for (level, _), sizes in cache_instances.items()
            if level == 1
            for size in sizes
        ]
        write_group("cpu_l1_cache", all_l1)
        write_group("cpu_l1d_cache", l1_data)
        write_group("cpu_l1i_cache", l1_instruction)
        write_group(
            "cpu_l2_cache",
            [size for (level, _), sizes in cache_instances.items() if level == 2 for size in sizes],
        )
        write_group(
            "cpu_l3_cache",
            [size for (level, _), sizes in cache_instances.items() if level == 3 for size in sizes],
        )
        cache_topology["cpu_cache_legacy_scope"] = "system_cache_total"
        return {
            "cpu_socket_count": sockets or "",
            "cpu_physical_core_count": physical_cores or "",
            "cpu_l1_cache_bytes": cache_bytes[1] or "",
            "cpu_l2_cache_bytes": cache_bytes[2] or "",
            "cpu_l3_cache_bytes": cache_bytes[3] or "",
            **cache_topology,
        }
    except (AttributeError, OSError):
        return empty


def _parse_hardware_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)B?\s*", value, re.IGNORECASE)
    if not match:
        return 0
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def _empty_cache_topology() -> dict[str, object]:
    fields = {
        "cpu_cache_legacy_scope": "unavailable",
        "cpu_l1_cache_bytes_per_instance": "",
        "cpu_l1_cache_instance_count": "",
        "cpu_l1_cache_bytes_total": "",
        "cpu_l1d_cache_bytes_per_instance": "",
        "cpu_l1d_cache_instance_count": "",
        "cpu_l1d_cache_bytes_total": "",
        "cpu_l1i_cache_bytes_per_instance": "",
        "cpu_l1i_cache_instance_count": "",
        "cpu_l1i_cache_bytes_total": "",
        "cpu_l2_cache_bytes_per_instance": "",
        "cpu_l2_cache_instance_count": "",
        "cpu_l2_cache_bytes_total": "",
        "cpu_l3_cache_bytes_per_instance": "",
        "cpu_l3_cache_instance_count": "",
        "cpu_l3_cache_bytes_total": "",
    }
    return fields


def linux_cpu_cache_topology(
    cpu_root: Path = Path("/sys/devices/system/cpu"),
) -> dict[str, object]:
    topology = _empty_cache_topology()
    unique_instances: dict[tuple[int, str, str], int] = {}
    for index_dir in cpu_root.glob("cpu[0-9]*/cache/index*"):
        try:
            level = int((index_dir / "level").read_text().strip())
            cache_type = (index_dir / "type").read_text().strip().lower()
            size = _parse_hardware_size((index_dir / "size").read_text().strip())
            shared_cpu_list = (index_dir / "shared_cpu_list").read_text().strip()
        except (OSError, ValueError):
            continue
        if level not in (1, 2, 3) or not size:
            continue
        key = (level, cache_type, shared_cpu_list or index_dir.parent.parent.name)
        unique_instances[key] = size

    groups: dict[tuple[int, str], list[int]] = {}
    for (level, cache_type, _), size in unique_instances.items():
        groups.setdefault((level, cache_type), []).append(size)

    def write_group(prefix: str, values: list[int]) -> None:
        topology[f"{prefix}_bytes_per_instance"] = values[0] if values else ""
        topology[f"{prefix}_instance_count"] = len(values) if values else ""
        topology[f"{prefix}_bytes_total"] = sum(values) if values else ""

    l1_data = groups.get((1, "data"), [])
    l1_instruction = groups.get((1, "instruction"), [])
    l1_unified = groups.get((1, "unified"), [])
    all_l1 = [*l1_data, *l1_instruction, *l1_unified]
    write_group("cpu_l1_cache", all_l1)
    write_group("cpu_l1d_cache", l1_data)
    write_group("cpu_l1i_cache", l1_instruction)
    write_group("cpu_l2_cache", [size for (level, _), values in groups.items() if level == 2 for size in values])
    write_group("cpu_l3_cache", [size for (level, _), values in groups.items() if level == 3 for size in values])
    topology["cpu_cache_legacy_scope"] = "cpu0_visible_cache_sum"
    return topology


def linux_cpu_info() -> dict[str, object]:
    if os.name != "posix":
        return {}
    records: list[dict[str, str]] = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            sections = cpuinfo.read_text(encoding="utf-8", errors="ignore").split("\n\n")
        except OSError:
            sections = []
        for section in sections:
            record: dict[str, str] = {}
            for line in section.splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    record[key.strip()] = value.strip()
            if record:
                records.append(record)

    try:
        logical_cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logical_cores = os.cpu_count() or len(records) or 1
    sockets = {record.get("physical id") for record in records if record.get("physical id")}
    physical_cores = {
        (record.get("physical id", "0"), record.get("core id"))
        for record in records
        if record.get("core id") is not None
    }
    physical_count = min(len(physical_cores) or logical_cores, logical_cores)

    cache_bytes: dict[int, int] = {}
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    if cache_root.exists():
        for index_dir in cache_root.glob("index*"):
            try:
                level = int((index_dir / "level").read_text().strip())
                size = _parse_hardware_size((index_dir / "size").read_text().strip())
            except (OSError, ValueError):
                continue
            cache_bytes[level] = cache_bytes.get(level, 0) + size

    first = records[0] if records else {}
    try:
        nominal_frequency = float(first.get("cpu MHz", 0) or 0)
    except ValueError:
        nominal_frequency = 0.0
    return {
        "cpu_model": first.get("model name", platform.processor()),
        "cpu_vendor": first.get("vendor_id", ""),
        "cpu_identifier": first.get("model", ""),
        "cpu_socket_count": len(sockets) or 1,
        "cpu_physical_core_count": physical_count,
        "cpu_logical_core_count": logical_cores,
        "cpu_l1_cache_bytes": cache_bytes.get(1, 0),
        "cpu_l2_cache_bytes": cache_bytes.get(2, 0),
        "cpu_l3_cache_bytes": cache_bytes.get(3, 0),
        **linux_cpu_cache_topology(),
        "cpu_nominal_frequency_mhz": nominal_frequency,
    }


def total_physical_memory_bytes() -> int | str:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("length", wintypes.DWORD),
                ("memory_load", wintypes.DWORD),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        memory = MemoryStatusEx()
        memory.length = ctypes.sizeof(memory)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                return int(memory.total_physical)
        except (AttributeError, OSError):
            return ""
    elif hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            return ""
    return ""


def collect_static_environment() -> dict[str, object]:
    if os.name == "nt":
        processor_info = windows_registry_cpu_info()
        topology = windows_cpu_topology()
        logical_cores: int | str = os.cpu_count() or ""
    else:
        processor_info = linux_cpu_info()
        topology = {
            field: processor_info.get(field, "")
            for field in (
                "cpu_socket_count",
                "cpu_physical_core_count",
                "cpu_l1_cache_bytes",
                "cpu_l2_cache_bytes",
                "cpu_l3_cache_bytes",
                *_empty_cache_topology().keys(),
            )
        }
        logical_cores = processor_info.get("cpu_logical_core_count", os.cpu_count() or "")
    physical_cores = topology["cpu_physical_core_count"]
    threads_per_core: float | str = ""
    if isinstance(logical_cores, int) and isinstance(physical_cores, int) and physical_cores:
        threads_per_core = logical_cores / physical_cores
    return {
        "environment_schema_version": 2,
        "host_name": platform.node() or "unknown-host",
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "cpu_model": processor_info.get("cpu_model") or platform.processor(),
        "cpu_vendor": processor_info.get("cpu_vendor", ""),
        "cpu_identifier": processor_info.get("cpu_identifier", ""),
        "cpu_architecture": platform.machine(),
        **topology,
        "cpu_logical_core_count": logical_cores,
        "cpu_threads_per_core": threads_per_core,
        "cpu_nominal_frequency_mhz": processor_info.get("cpu_nominal_frequency_mhz", ""),
        "memory_total_bytes": total_physical_memory_bytes(),
    }


def default_env_id() -> str:
    configured = os.environ.get("LLVM_RUNTIME_ENV_ID", "").strip()
    return configured or f"{platform.node() or 'unknown-host'}-{platform.machine() or 'unknown-arch'}"


def matches_any_pattern(relative_path: Path, patterns: Iterable[str]) -> bool:
    path_text = relative_path.as_posix()
    return any(fnmatch.fnmatch(path_text, pattern) for pattern in patterns)


def discover_sources(source_dir: Path, extensions: Iterable[str], exclude_patterns: Iterable[str] = ()) -> list[Program]:
    extensions = {ext.lower() for ext in extensions}
    source_dir = source_dir.resolve()
    programs = []
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative_path = path.resolve().relative_to(source_dir)
        if matches_any_pattern(relative_path, exclude_patterns):
            continue
        programs.append(Program(path.resolve(), source_dir))
    return sorted(programs, key=lambda item: item.program_id)


def default_include_dirs(source_root: Path, source_path: Path) -> list[Path]:
    dirs = [COMPAT_INCLUDE_DIR, source_path.parent, source_root]
    polybench_util = PROJECT_ROOT / "datasets" / "PolyBenchC-4.2.1" / "utilities"
    if polybench_util.exists():
        dirs.append(polybench_util)
    return list(dict.fromkeys(path.resolve() for path in dirs if path.exists()))


def extra_link_sources(source_root: Path, source_path: Path) -> list[Path]:
    search_roots = (source_root.resolve(), *source_path.resolve().parents)
    for root in dict.fromkeys(search_roots):
        polybench_runtime = root / "utilities" / "polybench.c"
        if polybench_runtime.exists() and source_path.resolve() != polybench_runtime.resolve():
            return [polybench_runtime]
    return []


def include_flags(paths: Iterable[Path]) -> list[str]:
    flags: list[str] = []
    for path in paths:
        flags.extend(["-I", str(path)])
    return flags


def forced_include_flags() -> list[str]:
    compat_header = COMPAT_INCLUDE_DIR / "polybench_windows_compat.h"
    return ["-include", str(compat_header)] if compat_header.exists() else []


def compiler_for(program: Program, c_compiler: str, cxx_compiler: str) -> str:
    return c_compiler if program.language == "C" else cxx_compiler


def output_path(base_dir: Path, program: Program, suffix: str) -> Path:
    path = base_dir / program.output_stem
    return path.with_suffix(suffix)


def generate_ast(
    program: Program,
    c_compiler: str,
    cxx_compiler: str,
    extra_flags: list[str],
) -> tuple[Path, bool, str]:
    ast_path = output_path(AST_DIR, program, ".ast")
    ast_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = compiler_for(program, c_compiler, cxx_compiler)
    cmd = [
        compiler,
        *extra_flags,
        *include_flags(default_include_dirs(program.source_root, program.source_path)),
        "-Xclang",
        "-ast-dump",
        "-fsyntax-only",
        str(program.source_path),
    ]
    result = run_command(cmd)
    ast_path.write_text(result.stdout, encoding="utf-8", errors="ignore")
    return ast_path, result.returncode == 0 and bool(result.stdout.strip()), result.stderr.strip()


def generate_ir(
    program: Program,
    c_compiler: str,
    cxx_compiler: str,
    opt_level: str,
    extra_flags: list[str],
) -> tuple[Path, bool, str]:
    ir_path = output_path(IR_DIR, program, ".ll")
    ir_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = compiler_for(program, c_compiler, cxx_compiler)
    cmd = [
        compiler,
        f"-{opt_level}",
        *extra_flags,
        *include_flags(default_include_dirs(program.source_root, program.source_path)),
        "-emit-llvm",
        "-S",
        str(program.source_path),
        "-o",
        str(ir_path),
    ]
    result = run_command(cmd)
    return ir_path, result.returncode == 0 and ir_path.exists(), result.stderr.strip()


def generate_cfg(
    program: Program,
    ir_path: Path,
    c_compiler: str,
    cxx_compiler: str,
    extra_flags: list[str],
) -> tuple[Path, bool, str, str]:
    program_cfg_dir = CFG_DIR / program.output_stem
    program_cfg_dir.mkdir(parents=True, exist_ok=True)
    opt_path = shutil.which("opt")
    if opt_path and ir_path.exists():
        cmd = [opt_path, "-passes=dot-cfg", "-disable-output", str(ir_path)]
        result = run_command(cmd, cwd=program_cfg_dir)
        dot_files = unique_dot_files(program_cfg_dir)
        if result.returncode == 0 and dot_files:
            return program_cfg_dir, True, result.stderr.strip(), "dot"
        return program_cfg_dir, False, result.stderr.strip(), "dot"

    cfg_path = output_path(CFG_DIR, program, ".cfg.txt")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = compiler_for(program, c_compiler, cxx_compiler)
    cmd = [
        compiler,
        "-fsyntax-only",
        *extra_flags,
        *include_flags(default_include_dirs(program.source_root, program.source_path)),
        "-Xclang",
        "-analyze",
        "-Xclang",
        "-analyzer-checker=debug.DumpCFG",
        str(program.source_path),
    ]
    result = run_command(cmd)
    text = "\n".join(part for part in [result.stdout, result.stderr] if part)
    cfg_path.write_text(text, encoding="utf-8", errors="ignore")
    return cfg_path, bool(text.strip()), result.stderr.strip(), "clang-dumpcfg"


def compile_executable(
    program: Program,
    run_c_compiler: str,
    run_cxx_compiler: str,
    opt_level: str,
    extra_flags: list[str],
) -> tuple[Path, bool, str]:
    exe_path = output_path(BUILD_DIR, program, ".exe" if os.name == "nt" else "")
    exe_path.parent.mkdir(parents=True, exist_ok=True)
    compiler = compiler_for(program, run_c_compiler, run_cxx_compiler)
    cmd = [
        compiler,
        f"-{opt_level}",
        *extra_flags,
        *include_flags(default_include_dirs(program.source_root, program.source_path)),
        *forced_include_flags(),
        str(program.source_path),
        *(str(path) for path in extra_link_sources(program.source_root, program.source_path)),
        "-lm",
        "-o",
        str(exe_path),
    ]
    result = run_command(cmd)
    return exe_path, result.returncode == 0 and exe_path.exists(), result.stderr.strip()


def count_regex(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


def literal_numbers(text: str, pattern: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(pattern, text):
        try:
            values.append(float(match))
        except ValueError:
            pass
    return values


def max_depth_from_ast_lines(lines: list[str], tokens: tuple[str, ...]) -> int:
    stack: list[int] = []
    best = 0
    for line in lines:
        indent = len(line) - len(line.lstrip(" |`-"))
        while stack and indent <= stack[-1]:
            stack.pop()
        if any(token in line for token in tokens):
            stack.append(indent)
            best = max(best, len(stack))
    return best


AST_SOURCE_LOCATION_RE = re.compile(
    r"(?:[A-Za-z]:)?[^<>,\s']+\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx|inc):\d+",
    re.IGNORECASE,
)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve()).replace("\\", "/").casefold()
        if normalized not in seen:
            seen.add(normalized)
            result.append(path.resolve())
    return result


def _ast_line_mentions_owned_source(line: str, source_paths: list[Path]) -> bool:
    normalized_line = line.replace("\\", "/").casefold()
    for source_path in source_paths:
        normalized_path = str(source_path).replace("\\", "/").casefold()
        if f"{normalized_path}:" in normalized_line:
            return True
        if re.search(
            rf"(?:^|[/<]){re.escape(source_path.name.casefold())}:\d+",
            normalized_line,
        ):
            return True
    return False


def source_owned_ast_text(
    ast_text: str,
    source_path: Path,
    owned_source_paths: Iterable[Path] | None = None,
) -> str:
    """Keep complete top-level AST subtrees belonging to project sources.

    Clang abbreviates consecutive source locations as ``line:`` or ``col:``.
    ``current_location_owned`` preserves that context while top-level
    declarations are traversed.
    """

    source_paths = _unique_paths([source_path, *(owned_source_paths or ())])
    retained: list[str] = []
    active_owned = False
    current_location_owned = False
    found_owned_declaration = False

    for line in ast_text.splitlines():
        is_top_level = line.startswith(("|-", "`-"))
        if is_top_level:
            if _ast_line_mentions_owned_source(line, source_paths):
                current_location_owned = True
                active_owned = True
                found_owned_declaration = True
            elif AST_SOURCE_LOCATION_RE.search(line) or any(
                marker in line for marker in ("<built-in>", "<invalid sloc>", "<scratch space>")
            ):
                current_location_owned = False
                active_owned = False
            elif re.search(r"<(?:line|col):\d+", line):
                active_owned = current_location_owned
            else:
                active_owned = False
        if active_owned:
            retained.append(line)

    if not found_owned_declaration:
        return ast_text
    return "\n".join(retained)


def detect_recursive_call(source_text: str, function_names: list[str]) -> int:
    for name in set(function_names):
        definition = re.search(rf"\b{name}\s*\([^;]*\)\s*\{{", source_text)
        if not definition:
            continue
        open_index = source_text.find("{", definition.start())
        depth = 0
        close_index = len(source_text)
        for index in range(open_index, len(source_text)):
            if source_text[index] == "{":
                depth += 1
            elif source_text[index] == "}":
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        body = source_text[open_index + 1 : close_index]
        if re.search(rf"\b{name}\s*\(", body):
            return 1
    return 0


def extract_ast_features(
    ast_path: Path,
    source_path: Path,
    owned_source_paths: Iterable[Path] | None = None,
) -> dict[str, float | int]:
    raw_text = ast_path.read_text(encoding="utf-8", errors="ignore") if ast_path.exists() else ""
    source_paths = _unique_paths([source_path, *(owned_source_paths or ())])
    text = source_owned_ast_text(raw_text, source_path, source_paths)
    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in source_paths
        if path.is_file()
    )
    lines = text.splitlines()
    integer_values = literal_numbers(text, r"IntegerLiteral[^\n]*\s(-?\d+)\b")
    float_values = literal_numbers(text, r"FloatingLiteral[^\n]*\s(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b")
    loop_count_total = (
        count_regex(r"\bForStmt\b", text)
        + count_regex(r"\bWhileStmt\b", text)
        + count_regex(r"\bDoStmt\b", text)
    )
    malloc_free_count = count_regex(r"\bCallExpr\b[^\n]*(malloc|calloc|realloc|free)\b", text)
    if malloc_free_count == 0:
        malloc_free_count = count_regex(r"\b(malloc|calloc|realloc|free)\s*\(", source_text)
    function_names = re.findall(r"FunctionDecl[^\n]*\b([A-Za-z_]\w*)\s+'", text)
    recursive = detect_recursive_call(source_text, function_names)
    return {
        "ast_node_count": len(lines),
        "function_count": count_regex(r"\bFunctionDecl\b", text),
        "call_expr_count": count_regex(r"\bCallExpr\b", text),
        "recursive_call_flag": recursive,
        "for_count": count_regex(r"\bForStmt\b", text),
        "while_count": count_regex(r"\bWhileStmt\b", text),
        "do_while_count": count_regex(r"\bDoStmt\b", text),
        "max_loop_depth": max_depth_from_ast_lines(lines, ("ForStmt", "WhileStmt", "DoStmt")),
        "loop_count_total": loop_count_total,
        "if_count": count_regex(r"\bIfStmt\b", text),
        "switch_count": count_regex(r"\bSwitchStmt\b", text),
        "conditional_operator_count": count_regex(r"\bConditionalOperator\b", text),
        "binary_operator_count": count_regex(r"\bBinaryOperator\b", text),
        "arithmetic_operator_count": count_regex(r"BinaryOperator[^\n]*'([+\-*/%]|[+\-*/%]=)'", text),
        "comparison_operator_count": count_regex(r"BinaryOperator[^\n]*'(==|!=|<=|>=|<|>)'", text),
        "array_subscript_count": count_regex(r"\bArraySubscriptExpr\b", text),
        "pointer_deref_count": count_regex(r"\bUnaryOperator\b[^\n]*'\*'", text),
        "malloc_free_count": malloc_free_count,
        "integer_literal_count": len(integer_values),
        "max_integer_literal": max(integer_values) if integer_values else 0,
        "avg_integer_literal": statistics.mean(integer_values) if integer_values else 0,
        "float_literal_count": len(float_values),
        "pragma_omp_count": count_regex(r"#\s*pragma\s+omp\b", source_text),
        "cuda_kernel_decl_count": count_regex(r"\b__global__\b", source_text),
    }


def ir_instruction_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.endswith(":") or stripped.startswith(("define ", "declare ", "attributes ", "source_filename", "target ")):
            continue
        if stripped == "}":
            continue
        lines.append(stripped)
    return lines


def count_ir_op(text: str, op: str) -> int:
    return count_regex(rf"(^|[\s=,]){re.escape(op)}([\s,]|$)", text)


def extract_ir_features(ir_path: Path) -> dict[str, float | int]:
    text = ir_path.read_text(encoding="utf-8", errors="ignore") if ir_path.exists() else ""
    instruction_lines = ir_instruction_lines(text)
    arithmetic_ops = ["add", "sub", "mul", "sdiv", "udiv", "fadd", "fsub", "fmul", "fdiv", "srem", "urem", "frem"]
    memory_count = sum(count_ir_op(text, op) for op in ["load", "store", "alloca", "getelementptr"])
    memory_count += count_regex(r"llvm\.mem(?:cpy|set)", text)
    arithmetic_count = sum(count_ir_op(text, op) for op in arithmetic_ops)
    control_count = sum(count_ir_op(text, op) for op in ["br", "switch", "indirectbr", "invoke", "ret"])
    constants = []
    for match in re.findall(r"(?<![%@.\w])-?\b\d+\b(?!\.\d)", text):
        try:
            value = int(match)
            if abs(value) < 100000000:
                constants.append(value)
        except ValueError:
            pass
    load_count = count_ir_op(text, "load")
    store_count = count_ir_op(text, "store")
    return {
        "ir_instruction_count": len(instruction_lines),
        "ir_line_count": len(text.splitlines()),
        "load_count": load_count,
        "store_count": store_count,
        "memory_inst_count": memory_count,
        "getelementptr_count": count_ir_op(text, "getelementptr"),
        "alloca_count": count_ir_op(text, "alloca"),
        "memcpy_count": count_regex(r"llvm\.memcpy", text),
        "memset_count": count_regex(r"llvm\.memset", text),
        "call_count_ir": count_ir_op(text, "call") + count_ir_op(text, "invoke"),
        "external_call_count": max(
            count_regex(r"\b(?:call|invoke)\b[^\n]*@", text) - count_regex(r"\b(?:call|invoke)\b[^\n]*@llvm\.", text),
            0,
        ),
        "branch_count": count_ir_op(text, "br"),
        "switch_count_ir": count_ir_op(text, "switch"),
        "phi_count": count_ir_op(text, "phi"),
        "select_count": count_ir_op(text, "select"),
        "icmp_count": count_ir_op(text, "icmp"),
        "fcmp_count": count_ir_op(text, "fcmp"),
        "add_count": count_ir_op(text, "add"),
        "sub_count": count_ir_op(text, "sub"),
        "mul_count": count_ir_op(text, "mul"),
        "div_count": count_ir_op(text, "sdiv") + count_ir_op(text, "udiv"),
        "rem_count": count_ir_op(text, "srem") + count_ir_op(text, "urem"),
        "fadd_count": count_ir_op(text, "fadd"),
        "fsub_count": count_ir_op(text, "fsub"),
        "fmul_count": count_ir_op(text, "fmul"),
        "fdiv_count": count_ir_op(text, "fdiv"),
        "vector_inst_count": count_regex(r"<\s*\d+\s+x\s+[^>]+>", text),
        "atomic_count": count_ir_op(text, "atomicrmw") + count_ir_op(text, "cmpxchg"),
        "barrier_or_sync_count": count_regex(r"(__syncthreads|barrier|omp_|cudaDeviceSynchronize|clFinish)", text),
        "constant_count_ir": len(constants),
        "max_constant_ir": max(constants) if constants else 0,
        "avg_constant_ir": statistics.mean(constants) if constants else 0,
        "load_store_ratio": safe_ratio(load_count, store_count),
        "memory_arithmetic_ratio": safe_ratio(memory_count, arithmetic_count),
        "branch_instruction_ratio": safe_ratio(control_count, len(instruction_lines)),
    }


def unique_dot_files(directory: Path) -> list[Path]:
    paths = [*directory.glob(".*.dot"), *directory.glob("*.dot")]
    return sorted({path.resolve(): path for path in paths}.values(), key=lambda path: path.name)


def parse_dot_edges(text: str) -> tuple[set[str], set[tuple[str, str]]]:
    node_pattern = r"Node(?:0x[0-9A-Fa-f]+|\d+)"
    nodes = set(re.findall(rf"\b({node_pattern})\s*\[", text))
    edges = set(
        re.findall(
            rf"\b({node_pattern})(?::[A-Za-z0-9_]+)?\s*->\s*"
            rf"({node_pattern})(?::[A-Za-z0-9_]+)?",
            text,
        )
    )
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
    return nodes, edges


def parse_clang_cfg_blocks(text: str) -> tuple[set[int], set[tuple[int, int]]]:
    blocks: set[int] = set()
    edges: set[tuple[int, int]] = set()
    current: int | None = None
    for line in text.splitlines():
        block_match = re.match(r"\s*\[B(\d+)", line)
        if block_match:
            current = int(block_match.group(1))
            blocks.add(current)
            continue
        if current is not None:
            succ_match = re.search(r"Succs\s*\(\d+\):\s*(.*)", line)
            if succ_match:
                for target in re.findall(r"B(\d+)", succ_match.group(1)):
                    dst = int(target)
                    blocks.add(dst)
                    edges.add((current, dst))
    return blocks, edges


def strongly_connected_components(nodes: set, edges: set[tuple]) -> list[set]:
    adjacency: dict[object, set[object]] = {node: set() for node in nodes}
    reverse: dict[object, set[object]] = {node: set() for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)
        reverse.setdefault(dst, set()).add(src)

    visited: set[object] = set()
    finish_order: list[object] = []
    for start in sorted(nodes, key=str):
        if start in visited:
            continue
        visited.add(start)
        stack: list[tuple[object, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            stack.append((node, True))
            for dst in sorted(adjacency.get(node, set()), key=str, reverse=True):
                if dst not in visited:
                    visited.add(dst)
                    stack.append((dst, False))

    components: list[set] = []
    assigned: set[object] = set()
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component = {start}
        assigned.add(start)
        stack = [start]
        while stack:
            node = stack.pop()
            for src in reverse.get(node, set()):
                if src not in assigned:
                    assigned.add(src)
                    component.add(src)
                    stack.append(src)
        components.append(component)
    return components


def weak_component_count(nodes: set, edges: set[tuple]) -> int:
    if not nodes:
        return 0
    adjacency: dict[object, set[object]] = {node: set() for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
    remaining = set(nodes)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            node = stack.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def dominator_backedges(
    nodes: set,
    edges: set[tuple],
    in_degree: dict,
) -> set[tuple]:
    if not nodes:
        return set()
    predecessors: dict[object, set[object]] = {node: set() for node in nodes}
    for src, dst in edges:
        predecessors.setdefault(dst, set()).add(src)
    entries = {node for node in nodes if in_degree.get(node, 0) == 0}
    if not entries:
        entries = {min(nodes, key=str)}

    dominators: dict[object, set] = {
        node: ({node} if node in entries else set(nodes)) for node in nodes
    }
    changed = True
    while changed:
        changed = False
        for node in sorted(nodes - entries, key=str):
            preds = predecessors.get(node, set())
            if preds:
                common = set(nodes)
                for pred in preds:
                    common.intersection_update(dominators[pred])
                updated = {node, *common}
            else:
                updated = {node}
            if updated != dominators[node]:
                dominators[node] = updated
                changed = True
    return {(src, dst) for src, dst in edges if dst in dominators.get(src, set())}


def natural_loops(nodes: set, edges: set[tuple], backedges: set[tuple]) -> list[set]:
    predecessors: dict[object, set[object]] = {node: set() for node in nodes}
    for src, dst in edges:
        predecessors.setdefault(dst, set()).add(src)
    loops: set[frozenset] = set()
    for tail, header in backedges:
        members = {header, tail}
        stack = [] if tail == header else [tail]
        while stack:
            node = stack.pop()
            for pred in predecessors.get(node, set()):
                if pred not in members:
                    members.add(pred)
                    stack.append(pred)
        loops.add(frozenset(members))
    return [set(loop) for loop in loops]


def loop_nesting_depth(loops: list[set]) -> int:
    if not loops:
        return 0

    def depth(loop: set, seen: set[int]) -> int:
        index = id(loop)
        if index in seen:
            return 1
        parents = [candidate for candidate in loops if loop < candidate]
        if not parents:
            return 1
        return 1 + max(depth(parent, seen | {index}) for parent in parents)

    return max(depth(loop, set()) for loop in loops)


def graph_features(nodes: set, edges: set[tuple]) -> dict[str, float | int]:
    nodes = set(nodes)
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
    node_count = len(nodes)
    edge_count = len(edges)
    out_degree = {node: 0 for node in nodes}
    in_degree = {node: 0 for node in nodes}
    for src, dst in edges:
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1
    branch_blocks = sum(1 for value in out_degree.values() if value > 1)
    merge_blocks = sum(1 for value in in_degree.values() if value > 1)
    backedges = dominator_backedges(nodes, edges, in_degree)
    loops = natural_loops(nodes, edges, backedges)
    components = strongly_connected_components(nodes, edges)
    cyclic_components = sum(
        1
        for component in components
        if len(component) > 1 or any(src == dst and src in component for src, dst in edges)
    )
    connected_components = weak_component_count(nodes, edges)
    depth = approximate_depth(nodes, edges)
    return {
        "basic_block_count": node_count,
        "cfg_edge_count": edge_count,
        "cyclomatic_complexity": (
            max(edge_count - node_count + 2 * connected_components, 0) if node_count else 0
        ),
        "branch_block_count": branch_blocks,
        "merge_block_count": merge_blocks,
        "avg_in_degree": statistics.mean(in_degree.values()) if in_degree else 0,
        "avg_out_degree": statistics.mean(out_degree.values()) if out_degree else 0,
        "max_out_degree": max(out_degree.values()) if out_degree else 0,
        "loop_backedge_count": len(backedges),
        "natural_loop_count": len(loops),
        "max_cfg_loop_depth": loop_nesting_depth(loops),
        "scc_count": cyclic_components,
        "cfg_depth": depth,
        "unreachable_block_count": 0,
        "critical_edge_count": sum(1 for src, dst in edges if out_degree.get(src, 0) > 1 and in_degree.get(dst, 0) > 1),
    }


def approximate_depth(nodes: set, edges: set[tuple]) -> int:
    if not nodes:
        return 0
    components = strongly_connected_components(nodes, edges)
    component_index = {
        node: index for index, component in enumerate(components) for node in component
    }
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(components))}
    in_degree = {index: 0 for index in range(len(components))}
    for src, dst in edges:
        src_component = component_index[src]
        dst_component = component_index[dst]
        if src_component != dst_component and dst_component not in adjacency[src_component]:
            adjacency[src_component].add(dst_component)
            in_degree[dst_component] += 1
    queue = [component for component, degree in in_degree.items() if degree == 0]
    distance = {component: 1 for component in queue}
    while queue:
        component = queue.pop(0)
        for dst in adjacency[component]:
            distance[dst] = max(distance.get(dst, 1), distance[component] + 1)
            in_degree[dst] -= 1
            if in_degree[dst] == 0:
                queue.append(dst)
    return max(distance.values(), default=1)


def aggregate_cfg_graphs(graphs: list[tuple[set, set[tuple]]]) -> dict[str, float | int]:
    rows = [graph_features(nodes, edges) for nodes, edges in graphs if nodes]
    result = {field: 0 for field in CFG_FIELDS}
    result["cfg_function_count"] = len(rows)
    if not rows:
        return result
    sum_fields = [
        "basic_block_count",
        "cfg_edge_count",
        "cyclomatic_complexity",
        "branch_block_count",
        "merge_block_count",
        "loop_backedge_count",
        "natural_loop_count",
        "scc_count",
        "unreachable_block_count",
        "critical_edge_count",
    ]
    for field in sum_fields:
        result[field] = sum(row[field] for row in rows)
    blocks = result["basic_block_count"]
    edges = result["cfg_edge_count"]
    result["avg_in_degree"] = safe_ratio(edges, blocks)
    result["avg_out_degree"] = safe_ratio(edges, blocks)
    for field in ["max_out_degree", "max_cfg_loop_depth", "cfg_depth"]:
        result[field] = max(row[field] for row in rows)
    return result


def extract_cfg_features(cfg_path: Path) -> dict[str, float | int]:
    if cfg_path.is_dir():
        graphs: list[tuple[set, set[tuple]]] = []
        for dot_file in unique_dot_files(cfg_path):
            text = dot_file.read_text(encoding="utf-8", errors="ignore")
            nodes, edges = parse_dot_edges(text)
            if nodes:
                graphs.append((nodes, edges))
        return aggregate_cfg_graphs(graphs)

    text = cfg_path.read_text(encoding="utf-8", errors="ignore") if cfg_path.exists() else ""
    sections = re.split(r"\n(?=[A-Za-z_]\w*\s*\()", text)
    graphs = []
    for index, section in enumerate(sections):
        nodes, edges = parse_clang_cfg_blocks(section)
        if not nodes:
            continue
        graphs.append((nodes, edges))
    return aggregate_cfg_graphs(graphs)


def strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\r\n]*", "", text)


def preprocessor_defines(text: str) -> dict[str, str]:
    defines: dict[str, str] = {}
    clean_text = strip_c_comments(text)
    pattern = re.compile(
        r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\b(?:\s+([^\r\n]+))?$",
        flags=re.MULTILINE,
    )
    for name, raw_value in pattern.findall(clean_text):
        value = raw_value.strip() if raw_value else "1"
        defines[name] = value
    return defines


def compiler_defines(extra_flags: Iterable[str]) -> dict[str, str]:
    flags = list(extra_flags)
    defines: dict[str, str] = {}
    index = 0
    while index < len(flags):
        flag = flags[index].strip()
        token = ""
        if flag in {"-D", "/D"} and index + 1 < len(flags):
            index += 1
            token = flags[index].strip()
        elif flag.startswith("-D") or flag.startswith("/D"):
            token = flag[2:].strip()

        if token:
            name, separator, value = token.partition("=")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                defines[name] = value.strip() if separator else "1"
        index += 1
    return defines


def polybench_profile_definitions(header_text: str) -> dict[str, dict[str, str]]:
    clean_text = strip_c_comments(header_text)
    profiles: dict[str, dict[str, str]] = {}
    for profile in POLYBENCH_DATASET_PROFILES:
        block_match = re.search(
            rf"^\s*#\s*ifdef\s+{profile}\b(?P<body>.*?)^\s*#\s*endif\b",
            clean_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not block_match:
            continue
        values = preprocessor_defines(block_match.group("body"))
        if values:
            profiles[profile] = values
    return profiles


def available_polybench_profiles(program: Program) -> tuple[str, ...]:
    header_path = program.source_path.with_suffix(".h")
    if not header_path.exists():
        return ()
    header_text = header_path.read_text(encoding="utf-8", errors="ignore")
    profiles = polybench_profile_definitions(header_text)
    return tuple(profile for profile in POLYBENCH_DATASET_PROFILES if profile in profiles)


def expand_program_inputs(
    programs: list[Program],
    requested_profiles: Iterable[str],
) -> tuple[list[Program], list[Program]]:
    selected_profiles = tuple(dict.fromkeys(requested_profiles))
    available_by_path = {
        program.source_path: available_polybench_profiles(program)
        for program in programs
    }
    has_polybench_suite_layout = any(
        (program.source_root / "utilities" / "polybench.c").exists()
        for program in programs
    )

    expanded: list[Program] = []
    skipped: list[Program] = []
    for program in programs:
        available_profiles = available_by_path[program.source_path]
        if not available_profiles:
            if has_polybench_suite_layout:
                skipped.append(program)
            else:
                expanded.append(program)
            continue

        missing = [profile for profile in selected_profiles if profile not in available_profiles]
        if missing:
            raise ValueError(
                f"{program.program_id} does not define requested PolyBench profiles: "
                f"{', '.join(missing)}"
            )
        expanded.extend(
            Program(program.source_path, program.source_root, profile)
            for profile in selected_profiles
        )
    return expanded, skipped


def program_extra_flags(program: Program, extra_flags: Iterable[str]) -> list[str]:
    flags = list(extra_flags)
    if not program.input_profile:
        return flags
    flags.extend(f"-U{profile}" for profile in POLYBENCH_DATASET_PROFILES)
    flags.append(f"-D{program.input_profile}")
    return flags


def default_polybench_profile(header_text: str, profiles: dict[str, dict[str, str]]) -> str:
    comment_match = re.search(
        r"Default\s+to\s+(MINI_DATASET|SMALL_DATASET|MEDIUM_DATASET|LARGE_DATASET|EXTRALARGE_DATASET)",
        header_text,
        flags=re.IGNORECASE,
    )
    if comment_match:
        profile = comment_match.group(1).upper()
        if profile in profiles:
            return profile
    return "LARGE_DATASET" if "LARGE_DATASET" in profiles else next(iter(profiles), "")


def simple_integer(value: str) -> int | None:
    token = value.strip()
    while token.startswith("(") and token.endswith(")"):
        token = token[1:-1].strip()
    match = re.fullmatch(r"([+-]?(?:0[xX][0-9A-Fa-f]+|\d+))[uUlL]*", token)
    if not match:
        return None
    return int(match.group(1), 0)


def first_integer(defines: dict[str, int], *names: str) -> int:
    for name in names:
        if name in defines:
            return defines[name]
    return 0


def resolve_polybench_input_size(
    source_path: Path,
    source_text: str,
    extra_flags: Iterable[str],
) -> tuple[str, dict[str, str], Path | None]:
    header_path = source_path.with_suffix(".h")
    if not header_path.exists():
        return "", {}, None

    header_text = header_path.read_text(encoding="utf-8", errors="ignore")
    profiles = polybench_profile_definitions(header_text)
    if not profiles:
        return "", {}, None

    include_match = re.search(
        rf'^\s*#\s*include\s*"{re.escape(header_path.name)}"',
        source_text,
        flags=re.MULTILINE,
    )
    source_prefix = source_text[:include_match.start()] if include_match else source_text
    source_defines = preprocessor_defines(source_prefix)
    flag_defines = compiler_defines(extra_flags)
    active_profiles = [
        profile
        for profile in POLYBENCH_DATASET_PROFILES
        if profile in source_defines or profile in flag_defines
    ]
    if active_profiles:
        effective_profile = active_profiles[-1]
        profile_label = effective_profile
        if len(active_profiles) > 1:
            profile_label = f"AMBIGUOUS[{','.join(active_profiles)}];EFFECTIVE={effective_profile}"
    else:
        effective_profile = default_polybench_profile(header_text, profiles)
        profile_label = effective_profile

    values = dict(profiles.get(effective_profile, {}))
    dimension_names = {name for profile_values in profiles.values() for name in profile_values}
    for name in dimension_names:
        if name in source_defines:
            values[name] = source_defines[name]
        if name in flag_defines:
            values[name] = flag_defines[name]
    return profile_label, values, header_path


def infer_input_scale(source_path: Path, extra_flags: Iterable[str] = ()) -> dict[str, object]:
    source = source_path.read_text(encoding="utf-8", errors="ignore")
    profile, profile_values, _ = resolve_polybench_input_size(source_path, source, extra_flags)

    source_defines = preprocessor_defines(source)
    resolved_defines = {**source_defines, **profile_values, **compiler_defines(extra_flags)}
    integer_defines = {
        name: value
        for name, raw_value in resolved_defines.items()
        if (value := simple_integer(raw_value)) is not None
    }
    parameters = ";".join(f"{name}={value}" for name, value in profile_values.items())

    return {
        "input_size_profile": profile,
        "input_size_parameters": parameters,
        "graph_nodes": first_integer(integer_defines, "NODES"),
        "graph_edges": first_integer(integer_defines, "EDGES"),
        "image_width": first_integer(integer_defines, "WIDTH", "W"),
        "image_height": first_integer(integer_defines, "HEIGHT", "H"),
        "iterations": first_integer(integer_defines, "ITERATIONS", "ITERS", "TSTEPS", "TMAX"),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float(numerator)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "runtime_sec_median": 0,
            "runtime_sec_mean": 0,
            "runtime_sec_std": 0,
            "runtime_cv": 0,
            "runtime_min": 0,
            "runtime_max": 0,
        }
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    return {
        "runtime_sec_median": statistics.median(values),
        "runtime_sec_mean": mean,
        "runtime_sec_std": std,
        "runtime_cv": safe_ratio(std, mean),
        "runtime_min": min(values),
        "runtime_max": max(values),
    }


def resolve_prometheus_exporter(value: str) -> str:
    if value != "auto":
        return value
    return "windows" if os.name == "nt" else "node"


def prometheus_host_queries(
    exporter: str,
    job: str,
    instance: str,
) -> dict[str, str]:
    if exporter == "off":
        return {}
    templates = PROMETHEUS_HOST_QUERY_TEMPLATES[exporter]
    labels: list[str] = []
    if job:
        labels.append(f"job={json.dumps(job)}")
    if instance:
        labels.append(f"instance={json.dumps(instance)}")
    selector = "{" + ",".join(labels) + "}"
    return {
        name: template.format(selector=selector)
        for name, template in templates.items()
    }


def prometheus_query_range(
    base_url: str,
    query: str,
    start_timestamp: float,
    end_timestamp: float,
    step_seconds: float,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    params = urlencode(
        {
            "query": query,
            "start": f"{start_timestamp:.6f}",
            "end": f"{end_timestamp:.6f}",
            "step": f"{step_seconds:g}",
        }
    )
    endpoint = f"{base_url.rstrip('/')}/api/v1/query_range?{params}"
    request = Request(endpoint, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if payload.get("status") != "success":
        raise RuntimeError(payload.get("error") or "Prometheus query failed")
    return payload.get("data", {}).get("result", [])


def parsed_series_values(series: dict[str, object]) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for timestamp, raw_value in series.get("values", []):
        try:
            value = float(raw_value)
            if math.isfinite(value):
                parsed.append((float(timestamp), value))
        except (TypeError, ValueError):
            continue
    return parsed


def gauge_values_by_timestamp(result: list[dict[str, object]]) -> list[float]:
    totals: dict[float, float] = {}
    for series in result:
        for timestamp, value in parsed_series_values(series):
            totals[timestamp] = totals.get(timestamp, 0.0) + value
    return [totals[timestamp] for timestamp in sorted(totals)]


def counter_delta(result: list[dict[str, object]]) -> float:
    total_delta = 0.0
    for series in result:
        values = parsed_series_values(series)
        for (_, previous), (_, current) in zip(values, values[1:]):
            total_delta += current - previous if current >= previous else current
    return total_delta


def cpu_usage_samples(result: list[dict[str, object]]) -> list[float]:
    total_by_timestamp: dict[float, float] = {}
    idle_by_timestamp: dict[float, float] = {}
    for series in result:
        metric = series.get("metric", {})
        mode = metric.get("mode") if isinstance(metric, dict) else None
        for timestamp, value in parsed_series_values(series):
            total_by_timestamp[timestamp] = total_by_timestamp.get(timestamp, 0.0) + value
            if mode == "idle":
                idle_by_timestamp[timestamp] = idle_by_timestamp.get(timestamp, 0.0) + value

    timestamps = sorted(set(total_by_timestamp) & set(idle_by_timestamp))
    samples: list[float] = []
    for previous_timestamp, current_timestamp in zip(timestamps, timestamps[1:]):
        total_delta = total_by_timestamp[current_timestamp] - total_by_timestamp[previous_timestamp]
        idle_delta = idle_by_timestamp[current_timestamp] - idle_by_timestamp[previous_timestamp]
        if total_delta <= 0 or idle_delta < 0:
            continue
        usage = 100.0 * (1.0 - idle_delta / total_delta)
        samples.append(min(100.0, max(0.0, usage)))
    return samples


def empty_prometheus_result(
    start_timestamp: float,
    end_timestamp: float,
    error: str,
) -> dict[str, object]:
    return {
        "prometheus_window_start_timestamp": start_timestamp,
        "prometheus_window_end_timestamp": end_timestamp,
        "prometheus_window_duration_seconds": max(0.0, end_timestamp - start_timestamp),
        "prometheus_sample_count": 0,
        "host_cpu_usage_pct_mean": "",
        "host_cpu_usage_pct_max": "",
        "host_memory_used_bytes_mean": "",
        "host_memory_used_bytes_max": "",
        "host_disk_read_bytes_delta": "",
        "host_disk_write_bytes_delta": "",
        "host_network_bytes_delta": "",
        "prometheus_collection_success": False,
        "prometheus_collection_error": error,
    }


def collect_prometheus_host_metrics(
    base_url: str,
    start_timestamp: float,
    end_timestamp: float,
    step_seconds: float,
    timeout_seconds: float,
    queries: dict[str, str],
) -> dict[str, object]:
    results = {
        name: prometheus_query_range(
            base_url,
            query,
            start_timestamp,
            end_timestamp,
            step_seconds,
            timeout_seconds,
        )
        for name, query in queries.items()
    }
    cpu_samples = cpu_usage_samples(results["cpu"])
    memory_samples = gauge_values_by_timestamp(results["memory_used"])
    sample_count = max(len(cpu_samples), len(memory_samples))
    if sample_count == 0:
        return empty_prometheus_result(
            start_timestamp,
            end_timestamp,
            "No Prometheus samples were found in this program window",
        )
    return {
        "prometheus_window_start_timestamp": start_timestamp,
        "prometheus_window_end_timestamp": end_timestamp,
        "prometheus_window_duration_seconds": max(0.0, end_timestamp - start_timestamp),
        "prometheus_sample_count": sample_count,
        "host_cpu_usage_pct_mean": statistics.mean(cpu_samples) if cpu_samples else "",
        "host_cpu_usage_pct_max": max(cpu_samples) if cpu_samples else "",
        "host_memory_used_bytes_mean": statistics.mean(memory_samples) if memory_samples else "",
        "host_memory_used_bytes_max": max(memory_samples) if memory_samples else "",
        "host_disk_read_bytes_delta": counter_delta(results["disk_read"]),
        "host_disk_write_bytes_delta": counter_delta(results["disk_write"]),
        "host_network_bytes_delta": counter_delta(results["network"]),
        "prometheus_collection_success": True,
        "prometheus_collection_error": "",
    }


def _phase_prometheus_result(
    phase: str,
    result: dict[str, object],
) -> dict[str, object]:
    return {
        f"prometheus_{phase}_window_start_timestamp": result["prometheus_window_start_timestamp"],
        f"prometheus_{phase}_window_end_timestamp": result["prometheus_window_end_timestamp"],
        f"prometheus_{phase}_window_duration_seconds": result["prometheus_window_duration_seconds"],
        f"prometheus_{phase}_sample_count": result["prometheus_sample_count"],
        f"host_cpu_usage_pct_{phase}_mean": result["host_cpu_usage_pct_mean"],
        f"host_cpu_usage_pct_{phase}_max": result["host_cpu_usage_pct_max"],
        f"host_memory_used_bytes_{phase}_mean": result["host_memory_used_bytes_mean"],
        f"host_memory_used_bytes_{phase}_max": result["host_memory_used_bytes_max"],
        f"host_disk_read_bytes_{phase}_delta": result["host_disk_read_bytes_delta"],
        f"host_disk_write_bytes_{phase}_delta": result["host_disk_write_bytes_delta"],
        f"host_network_bytes_{phase}_delta": result["host_network_bytes_delta"],
        f"prometheus_{phase}_collection_success": result["prometheus_collection_success"],
        f"prometheus_{phase}_collection_error": result["prometheus_collection_error"],
    }


def _numeric_metric(result: dict[str, object], field: str) -> float | None:
    value = result.get(field)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_prometheus_phases(
    phase_results: dict[str, dict[str, object]],
    context_seconds: float,
) -> dict[str, object]:
    before = phase_results["before"]
    during = phase_results["during"]
    after = phase_results["after"]
    output = {
        **during,
        **_phase_prometheus_result("before", before),
        **_phase_prometheus_result("during", during),
        **_phase_prometheus_result("after", after),
        "prometheus_sampling_scheme": "before_during_after_v1",
        "prometheus_context_seconds_requested": context_seconds,
        "prometheus_background_correction_method": (
            "linear-background estimate: mean(before,after); counter deltas use "
            "mean background rate multiplied by during duration; adjusted values clipped at zero"
        ),
    }

    failures = [
        f"{phase}: {phase_results[phase].get('prometheus_collection_error', '')}"
        for phase in PROMETHEUS_PHASES
        if phase_results[phase].get("prometheus_collection_success") is not True
    ]
    output["prometheus_three_phase_collection_success"] = not failures
    output["prometheus_three_phase_collection_error"] = " | ".join(failures)

    def background(field: str) -> tuple[float | None, float | None, float | None]:
        before_value = _numeric_metric(before, field)
        after_value = _numeric_metric(after, field)
        if before_value is None or after_value is None:
            return None, before_value, after_value
        return (before_value + after_value) / 2.0, before_value, after_value

    cpu_background, cpu_before, cpu_after = background("host_cpu_usage_pct_mean")
    memory_background, memory_before, memory_after = background("host_memory_used_bytes_mean")
    output["prometheus_background_cpu_usage_pct_mean"] = cpu_background if cpu_background is not None else ""
    output["prometheus_background_memory_used_bytes_mean"] = memory_background if memory_background is not None else ""
    output["prometheus_background_cpu_before_after_abs_diff_pct"] = (
        abs(cpu_before - cpu_after) if cpu_before is not None and cpu_after is not None else ""
    )
    output["prometheus_background_memory_before_after_abs_diff_bytes"] = (
        abs(memory_before - memory_after)
        if memory_before is not None and memory_after is not None
        else ""
    )

    during_cpu_mean = _numeric_metric(during, "host_cpu_usage_pct_mean")
    during_memory_mean = _numeric_metric(during, "host_memory_used_bytes_mean")
    output["host_cpu_usage_pct_background_adjusted_mean"] = (
        max(0.0, during_cpu_mean - cpu_background)
        if during_cpu_mean is not None and cpu_background is not None
        else ""
    )
    output["host_memory_used_bytes_background_adjusted_mean"] = (
        max(0.0, during_memory_mean - memory_background)
        if during_memory_mean is not None and memory_background is not None
        else ""
    )

    before_cpu_max = _numeric_metric(before, "host_cpu_usage_pct_max")
    after_cpu_max = _numeric_metric(after, "host_cpu_usage_pct_max")
    during_cpu_max = _numeric_metric(during, "host_cpu_usage_pct_max")
    before_memory_max = _numeric_metric(before, "host_memory_used_bytes_max")
    after_memory_max = _numeric_metric(after, "host_memory_used_bytes_max")
    during_memory_max = _numeric_metric(during, "host_memory_used_bytes_max")
    output["host_cpu_usage_pct_background_adjusted_max"] = (
        max(0.0, during_cpu_max - max(before_cpu_max, after_cpu_max))
        if None not in (during_cpu_max, before_cpu_max, after_cpu_max)
        else ""
    )
    output["host_memory_used_bytes_background_adjusted_max"] = (
        max(0.0, during_memory_max - max(before_memory_max, after_memory_max))
        if None not in (during_memory_max, before_memory_max, after_memory_max)
        else ""
    )

    during_duration = _numeric_metric(during, "prometheus_window_duration_seconds")
    for metric, output_prefix in (
        ("host_disk_read_bytes_delta", "disk_read"),
        ("host_disk_write_bytes_delta", "disk_write"),
        ("host_network_bytes_delta", "network"),
    ):
        before_delta = _numeric_metric(before, metric)
        after_delta = _numeric_metric(after, metric)
        during_delta = _numeric_metric(during, metric)
        before_duration = _numeric_metric(before, "prometheus_window_duration_seconds")
        after_duration = _numeric_metric(after, "prometheus_window_duration_seconds")
        rates = (
            before_delta / before_duration
            if before_delta is not None and before_duration and before_duration > 0
            else None,
            after_delta / after_duration
            if after_delta is not None and after_duration and after_duration > 0
            else None,
        )
        background_rate = (
            (rates[0] + rates[1]) / 2.0
            if rates[0] is not None and rates[1] is not None
            else None
        )
        output[f"prometheus_background_{output_prefix}_bytes_per_sec"] = (
            background_rate if background_rate is not None else ""
        )
        output[f"prometheus_background_{output_prefix}_rate_abs_diff_bytes_per_sec"] = (
            abs(rates[0] - rates[1])
            if rates[0] is not None and rates[1] is not None
            else ""
        )
        output[f"host_{output_prefix}_bytes_background_adjusted_delta"] = (
            max(0.0, during_delta - background_rate * during_duration)
            if during_delta is not None
            and background_rate is not None
            and during_duration is not None
            else ""
        )
    return output


def empty_prometheus_three_phase_result(
    windows: dict[str, tuple[float, float]] | None,
    error: str,
    context_seconds: float,
) -> dict[str, object]:
    windows = windows or {phase: (0.0, 0.0) for phase in PROMETHEUS_PHASES}
    return summarize_prometheus_phases(
        {
            phase: empty_prometheus_result(*windows.get(phase, (0.0, 0.0)), error)
            for phase in PROMETHEUS_PHASES
        },
        context_seconds,
    )


def collect_prometheus_three_phase_metrics(
    base_url: str,
    windows: dict[str, tuple[float, float]],
    step_seconds: float,
    timeout_seconds: float,
    queries: dict[str, str],
    context_seconds: float,
) -> dict[str, object]:
    phase_results: dict[str, dict[str, object]] = {}
    for phase in PROMETHEUS_PHASES:
        try:
            phase_results[phase] = collect_prometheus_host_metrics(
                base_url,
                *windows[phase],
                step_seconds,
                timeout_seconds,
                queries,
            )
        except Exception as exc:
            phase_results[phase] = empty_prometheus_result(
                *windows[phase],
                str(exc)[:500],
            )
    return summarize_prometheus_phases(phase_results, context_seconds)


def empty_process_metrics(error: str = "") -> dict[str, object]:
    metrics = {field: "" for field in PROCESS_RUN_FIELDS}
    metrics["process_metrics_success"] = False
    metrics["process_metrics_error"] = error
    return metrics


def windows_process_metrics(process: subprocess.Popen[str]) -> dict[str, object]:
    metrics = empty_process_metrics()
    metrics["process_id"] = process.pid
    metrics["process_metrics_backend"] = "windows-native-v1"
    if os.name != "nt":
        metrics["process_metrics_error"] = "Windows process metrics requested on a non-Windows host"
        return metrics

    class FileTime(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
            ("quota_nonpaged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    def filetime_seconds(value: FileTime) -> float:
        return ((int(value.high) << 32) | int(value.low)) / 10_000_000.0

    errors: list[str] = []
    handle = wintypes.HANDLE(int(process._handle))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    creation = FileTime()
    exit_time = FileTime()
    kernel = FileTime()
    user = FileTime()
    get_process_times = kernel32.GetProcessTimes
    get_process_times.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    get_process_times.restype = wintypes.BOOL
    if get_process_times(handle, creation, exit_time, kernel, user):
        user_seconds = filetime_seconds(user)
        kernel_seconds = filetime_seconds(kernel)
        metrics["process_cpu_user_sec"] = user_seconds
        metrics["process_cpu_system_sec"] = kernel_seconds
        metrics["process_cpu_kernel_sec"] = kernel_seconds
        metrics["process_cpu_total_sec"] = user_seconds + kernel_seconds
    else:
        errors.append(f"GetProcessTimes error {ctypes.get_last_error()}")

    memory = ProcessMemoryCountersEx()
    memory.cb = ctypes.sizeof(memory)
    get_process_memory_info = psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    if get_process_memory_info(handle, memory, memory.cb):
        metrics["process_max_rss_bytes"] = int(memory.peak_working_set_size)
        metrics["process_peak_working_set_bytes"] = int(memory.peak_working_set_size)
        metrics["process_peak_private_bytes"] = int(memory.peak_pagefile_usage)
        metrics["process_page_faults"] = int(memory.page_fault_count)
    else:
        errors.append(f"GetProcessMemoryInfo error {ctypes.get_last_error()}")

    io = IoCounters()
    get_process_io_counters = kernel32.GetProcessIoCounters
    get_process_io_counters.argtypes = [wintypes.HANDLE, ctypes.POINTER(IoCounters)]
    get_process_io_counters.restype = wintypes.BOOL
    if get_process_io_counters(handle, io):
        metrics["process_read_bytes"] = int(io.read_transfer_count)
        metrics["process_write_bytes"] = int(io.write_transfer_count)
        metrics["process_other_bytes"] = int(io.other_transfer_count)
    else:
        errors.append(f"GetProcessIoCounters error {ctypes.get_last_error()}")

    metrics["process_metrics_success"] = not errors
    metrics["process_metrics_error"] = " | ".join(errors)
    return metrics


def ensure_process_probe() -> Path:
    global _PROCESS_PROBE_PATH
    if not PROCESS_PROBE_SOURCE.is_file():
        raise RuntimeError(f"process probe source not found: {PROCESS_PROBE_SOURCE}")
    if _PROCESS_PROBE_PATH and _PROCESS_PROBE_PATH.is_file():
        if _PROCESS_PROBE_PATH.stat().st_mtime >= PROCESS_PROBE_SOURCE.stat().st_mtime:
            return _PROCESS_PROBE_PATH
    if os.name != "posix":
        raise RuntimeError("the process probe can only be built on POSIX systems")
    if PROCESS_PROBE_BINARY.is_file():
        if PROCESS_PROBE_BINARY.stat().st_mtime >= PROCESS_PROBE_SOURCE.stat().st_mtime:
            _PROCESS_PROBE_PATH = PROCESS_PROBE_BINARY
            return PROCESS_PROBE_BINARY
    compiler = next(
        (path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))),
        None,
    )
    if not compiler:
        raise RuntimeError("cannot build process probe: cc, gcc, and clang are all unavailable")

    PROCESS_PROBE_BINARY.parent.mkdir(parents=True, exist_ok=True)
    temporary_binary = PROCESS_PROBE_BINARY.with_name(
        f".{PROCESS_PROBE_BINARY.name}.{os.getpid()}"
    )
    command = [
        compiler,
        "-O2",
        "-std=c11",
        "-Wall",
        "-Wextra",
        str(PROCESS_PROBE_SOURCE),
        "-o",
        str(temporary_binary),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        temporary_binary.unlink(missing_ok=True)
        raise RuntimeError(f"cannot build process probe: {exc}") from exc
    if result.returncode != 0:
        temporary_binary.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"cannot build process probe: {detail[-4000:]}")
    temporary_binary.replace(PROCESS_PROBE_BINARY)
    if not os.access(PROCESS_PROBE_BINARY, os.X_OK):
        raise RuntimeError(f"compiled process probe is not executable: {PROCESS_PROBE_BINARY}")
    _PROCESS_PROBE_PATH = PROCESS_PROBE_BINARY
    return PROCESS_PROBE_BINARY


def parse_process_probe_metrics(
    path: Path,
) -> tuple[float | None, int | None, dict[str, object]]:
    if not path.is_file():
        return None, None, {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None, None, {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    required = {
        "probe_version",
        "elapsed_ns",
        "returncode",
        "user_us",
        "system_us",
        "max_rss_bytes",
        "major_faults",
        "minor_faults",
        "fs_inputs",
        "fs_outputs",
    }
    if not required.issubset(values) or values["probe_version"] != "1":
        return None, None, {}
    try:
        elapsed_ns = int(values["elapsed_ns"])
        returncode = int(values["returncode"])
        user_us = int(values["user_us"])
        system_us = int(values["system_us"])
        max_rss_bytes = int(values["max_rss_bytes"])
        major_faults = int(values["major_faults"])
        minor_faults = int(values["minor_faults"])
        fs_inputs = int(values["fs_inputs"])
        fs_outputs = int(values["fs_outputs"])
    except ValueError:
        return None, None, {}
    if min(
        elapsed_ns,
        user_us,
        system_us,
        max_rss_bytes,
        major_faults,
        minor_faults,
        fs_inputs,
        fs_outputs,
    ) < 0:
        return None, None, {}

    metrics = empty_process_metrics()
    metrics.update(
        {
            "process_cpu_user_sec": user_us / 1_000_000,
            "process_cpu_system_sec": system_us / 1_000_000,
            "process_cpu_kernel_sec": system_us / 1_000_000,
            "process_cpu_total_sec": (user_us + system_us) / 1_000_000,
            "process_max_rss_bytes": max_rss_bytes,
            "process_peak_working_set_bytes": max_rss_bytes,
            "process_major_page_faults": major_faults,
            "process_minor_page_faults": minor_faults,
            "process_page_faults": major_faults + minor_faults,
            "process_fs_inputs": fs_inputs,
            "process_fs_outputs": fs_outputs,
            "process_metrics_backend": "linux-process-probe-v1",
            "process_metrics_success": True,
            "process_metrics_error": "",
        }
    )
    return elapsed_ns / 1_000_000_000, returncode, metrics


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def run_monitored_command(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
) -> tuple[subprocess.CompletedProcess[str], float, bool, dict[str, object]]:
    process_probe: Path | None = None
    metrics_directory: tempfile.TemporaryDirectory[str] | None = None
    metrics_path: Path | None = None
    actual_command = cmd
    if os.name == "posix":
        try:
            process_probe = ensure_process_probe()
        except RuntimeError as exc:
            metrics = empty_process_metrics(str(exc))
            result = subprocess.CompletedProcess(cmd, 125, "", str(exc))
            return result, 0.0, False, metrics
        metrics_directory = tempfile.TemporaryDirectory(prefix="polybench-process-")
        metrics_path = Path(metrics_directory.name) / "metrics.txt"
        actual_command = [str(process_probe), str(metrics_path), "--", *cmd]

    start = time.perf_counter()
    process = subprocess.Popen(
        actual_command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
        start_new_session=os.name == "posix",
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
        stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - start
    returncode = process.returncode
    if os.name == "nt":
        process_metrics = windows_process_metrics(process)
    else:
        probe_elapsed, child_returncode, process_metrics = parse_process_probe_metrics(
            metrics_path or Path()
        )
        if probe_elapsed is not None:
            elapsed = probe_elapsed
        if child_returncode is not None:
            returncode = child_returncode
        if not process_metrics:
            message = "process probe did not produce valid metrics"
            process_metrics = empty_process_metrics(message)
            if not timed_out:
                returncode = 125
                stderr = " | ".join(part for part in (stderr.strip(), message) if part)
    if metrics_directory is not None:
        metrics_directory.cleanup()
    result = subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return result, elapsed, timed_out, process_metrics


def numeric_record_values(records: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def summarize_process_records(
    records: list[dict[str, object]],
    target_seconds: float,
    actual_seconds: float,
    min_measured_runs: int,
) -> dict[str, object]:
    successful = [record for record in records if record.get("success") is True]
    sampled = [record for record in records if record.get("process_metrics_success") is True]

    def total(field: str) -> float | str:
        values = numeric_record_values(sampled, field)
        return sum(values) if values else ""

    def mean(field: str) -> float | str:
        values = numeric_record_values(sampled, field)
        return statistics.mean(values) if values else ""

    cpu_totals = numeric_record_values(sampled, "process_cpu_total_sec")
    peak_working_sets = numeric_record_values(sampled, "process_peak_working_set_bytes")
    peak_private_bytes = numeric_record_values(sampled, "process_peak_private_bytes")
    max_rss_values = numeric_record_values(sampled, "process_max_rss_bytes")
    backends = sorted(
        {
            str(record.get("process_metrics_backend"))
            for record in sampled
            if record.get("process_metrics_backend")
        }
    )
    return {
        "measurement_target_seconds": target_seconds,
        "measurement_actual_seconds": actual_seconds,
        "min_measured_runs_requested": min_measured_runs,
        "successful_runs": len(successful),
        "failed_runs": len(records) - len(successful),
        "process_metrics_sampled_runs": len(sampled),
        "process_metric_aggregation": "mean of measured runs",
        "process_cpu_user_sec": mean("process_cpu_user_sec"),
        "process_cpu_system_sec": mean("process_cpu_system_sec"),
        "process_max_rss_bytes": mean("process_max_rss_bytes"),
        "process_major_page_faults": mean("process_major_page_faults"),
        "process_minor_page_faults": mean("process_minor_page_faults"),
        "process_fs_inputs": mean("process_fs_inputs"),
        "process_fs_outputs": mean("process_fs_outputs"),
        "process_max_rss_bytes_peak": max(max_rss_values) if max_rss_values else "",
        "process_metrics_backend": ",".join(backends) if backends else "unavailable",
        "process_cpu_user_sec_total": total("process_cpu_user_sec"),
        "process_cpu_kernel_sec_total": total("process_cpu_kernel_sec"),
        "process_cpu_total_sec_total": sum(cpu_totals) if cpu_totals else "",
        "process_cpu_total_sec_mean": statistics.mean(cpu_totals) if cpu_totals else "",
        "process_peak_working_set_bytes_max": max(peak_working_sets) if peak_working_sets else "",
        "process_peak_private_bytes_max": max(peak_private_bytes) if peak_private_bytes else "",
        "process_page_faults_total": total("process_page_faults"),
        "process_read_bytes_total": total("process_read_bytes"),
        "process_write_bytes_total": total("process_write_bytes"),
        "process_other_bytes_total": total("process_other_bytes"),
    }


def run_program(
    exe_path: Path,
    warmup: int,
    runs: int,
    timeout: int,
    dataset: str,
    measurement_seconds: float,
    min_measured_runs: int,
    prometheus_context_seconds: float,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    bool,
    str,
    dict[str, tuple[float, float]],
]:
    records: list[dict[str, object]] = []
    measured: list[float] = []
    error = ""
    success = True

    def execute(phase: str, index: int) -> None:
        nonlocal error, success
        result, elapsed, timed_out, process_metrics = run_monitored_command(
            [str(exe_path)],
            timeout=timeout,
        )
        ok = not timed_out and result.returncode == 0
        status = "timeout" if timed_out else ("success" if ok else "failed")
        PIPELINE_BENCHMARK_DURATION.labels(dataset=dataset, phase=phase).observe(elapsed)
        PIPELINE_BENCHMARK_RUNS.labels(dataset=dataset, phase=phase, status=status).inc()
        if phase == "measure" and ok:
            measured.append(elapsed)
        if not ok:
            success = False
            error = (
                f"timeout after {timeout}s"
                if timed_out
                else (result.stderr or result.stdout or f"return code {result.returncode}").strip()
            )
        records.append(
            {
                "phase": phase,
                "run_index": index,
                "runtime_sec": elapsed,
                "return_code": "timeout" if timed_out else result.returncode,
                "success": ok,
                **process_metrics,
            }
        )

    for index in range(1, warmup + 1):
        execute("warmup", index)

    before_start_timestamp = time.time()
    if prometheus_context_seconds > 0:
        time.sleep(prometheus_context_seconds)
    before_end_timestamp = time.time()

    measurement_start_timestamp = time.time()
    measurement_start = time.perf_counter()
    measured_run_count = 0
    while True:
        elapsed_window = time.perf_counter() - measurement_start
        if measurement_seconds > 0:
            if (
                measured_run_count >= min_measured_runs
                and elapsed_window >= measurement_seconds
            ):
                break
        elif measured_run_count >= runs:
            break
        measured_run_count += 1
        execute("measure", measured_run_count)
    measurement_end_timestamp = time.time()
    actual_seconds = time.perf_counter() - measurement_start

    after_start_timestamp = time.time()
    if prometheus_context_seconds > 0:
        time.sleep(prometheus_context_seconds)
    after_end_timestamp = time.time()

    measured_records = [record for record in records if record["phase"] == "measure"]
    summary = {
        **summarize(measured),
        **summarize_process_records(
            measured_records,
            measurement_seconds,
            actual_seconds,
            min_measured_runs,
        ),
    }
    return (
        records,
        summary,
        success,
        error,
        {
            "before": (before_start_timestamp, before_end_timestamp),
            "during": (measurement_start_timestamp, measurement_end_timestamp),
            "after": (after_start_timestamp, after_end_timestamp),
        },
    )


def fallback_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    try:
        handle = target.open("w", newline="", encoding="utf-8-sig")
    except PermissionError:
        target = fallback_path(path)
        print(f"[pipeline] {path} is locked; writing {target} instead")
        handle = target.open("w", newline="", encoding="utf-8-sig")
    with handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    try:
        target.write_text(text, encoding=encoding)
    except PermissionError:
        target = fallback_path(path)
        print(f"[pipeline] {path} is locked; writing {target} instead")
        target.write_text(text, encoding=encoding)
    return target


def unique_fields(fields: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            result.append(field)
    return result


def environment_manifest(
    args: argparse.Namespace,
    static_environment: dict[str, object],
) -> dict[str, object]:
    return {
        "env_id": args.env_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        **static_environment,
        "c_compiler": args.c_compiler,
        "cxx_compiler": args.cxx_compiler,
        "run_c_compiler": args.run_c_compiler,
        "run_cxx_compiler": args.run_cxx_compiler,
        "c_compiler_version": compiler_version(args.c_compiler),
        "cxx_compiler_version": compiler_version(args.cxx_compiler),
        "run_c_compiler_version": compiler_version(args.run_c_compiler),
        "run_cxx_compiler_version": compiler_version(args.run_cxx_compiler),
        "opt_version": llvm_opt_version(),
        "opt_level": args.opt_level,
        "warmup_runs": args.warmup,
        "measurement_seconds": args.measurement_seconds,
        "min_measured_runs": args.min_measured_runs,
        "fixed_measured_runs": args.runs,
        "timeout_sec": args.timeout,
        "polybench_profiles": args.polybench_profile or list(POLYBENCH_DATASET_PROFILES),
        "prometheus_url": args.prometheus_url,
        "prometheus_exporter": args.prometheus_exporter,
        "prometheus_job": args.prometheus_job,
        "prometheus_instance": args.prometheus_instance,
        "prometheus_query_step_sec": args.prometheus_query_step,
        "prometheus_context_seconds": args.prometheus_context_seconds,
        "prometheus_sampling_scheme": "before_during_after_v1",
    }


def process_program(
    program: Program,
    args: argparse.Namespace,
    static_environment: dict[str, object] | None = None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    dict[str, tuple[float, float]] | None,
]:
    error_messages: list[str] = []
    extra_flags = program_extra_flags(program, args.extra_flag or [])
    input_id = program.input_profile or args.input_id
    representation_compiler = compiler_for(program, args.c_compiler, args.cxx_compiler)
    runtime_compiler = compiler_for(program, args.run_c_compiler, args.run_cxx_compiler)
    environment = {
        **(static_environment or collect_static_environment()),
        "representation_compiler": representation_compiler,
        "representation_compiler_version": compiler_version(representation_compiler),
        "runtime_compiler": runtime_compiler,
        "runtime_compiler_version": compiler_version(runtime_compiler),
        "opt_version": llvm_opt_version(),
    }
    ast_path, ast_success, ast_error = generate_ast(program, args.c_compiler, args.cxx_compiler, extra_flags)
    if ast_error and not ast_success:
        error_messages.append(f"AST: {ast_error[:300]}")
    if not ast_success:
        PIPELINE_STAGE_ERRORS.labels(dataset=args.dataset, stage="ast").inc()
    ast_features = extract_ast_features(ast_path, program.source_path)

    ir_path, ir_success, ir_error = generate_ir(program, args.c_compiler, args.cxx_compiler, args.opt_level, extra_flags)
    if ir_error and not ir_success:
        error_messages.append(f"IR: {ir_error[:300]}")
    if not ir_success:
        PIPELINE_STAGE_ERRORS.labels(dataset=args.dataset, stage="ir").inc()
    ir_features = extract_ir_features(ir_path)

    cfg_path, cfg_success, cfg_error, cfg_kind = generate_cfg(program, ir_path, args.c_compiler, args.cxx_compiler, extra_flags)
    if cfg_error and not cfg_success:
        error_messages.append(f"CFG: {cfg_error[:300]}")
    if not cfg_success:
        PIPELINE_STAGE_ERRORS.labels(dataset=args.dataset, stage="cfg").inc()
    cfg_features = extract_cfg_features(cfg_path)

    exe_path, build_success, build_error = compile_executable(program, args.run_c_compiler, args.run_cxx_compiler, args.opt_level, extra_flags)
    if build_error and not build_success:
        error_messages.append(f"BUILD: {build_error[:300]}")
    if not build_success:
        PIPELINE_STAGE_ERRORS.labels(dataset=args.dataset, stage="build").inc()

    run_records: list[dict[str, object]] = []
    runtime_summary = summarize([])
    measurement_summary = summarize_process_records(
        [],
        args.measurement_seconds,
        0.0,
        args.min_measured_runs,
    )
    measurement_windows: dict[str, tuple[float, float]] | None = None
    run_success = False
    run_error = ""
    if build_success and not args.no_run:
        run_records, runtime_summary, run_success, run_error, measurement_windows = run_program(
            exe_path,
            args.warmup,
            args.runs,
            args.timeout,
            args.dataset,
            args.measurement_seconds,
            args.min_measured_runs,
            args.prometheus_context_seconds if args.prometheus_exporter != "off" else 0.0,
        )
        measurement_summary = {
            field: runtime_summary.get(field, "")
            for field in MEASUREMENT_SUMMARY_FIELDS
        }
        if run_error and not run_success:
            error_messages.append(f"RUN: {run_error[:300]}")
        if not run_success:
            PIPELINE_STAGE_ERRORS.labels(dataset=args.dataset, stage="run").inc()

    input_scale = infer_input_scale(program.source_path, extra_flags)
    base = {
        "program_id": program.program_id,
        "input_id": input_id,
        "env_id": args.env_id,
        "dataset": args.dataset,
        "relative_path": program.program_id,
        "source_path": str(program.source_path),
        "language": program.language,
        "task_type": args.task_type,
        "parallel_model": args.parallel_model,
        "compile_config_id": args.compile_config_id,
        "opt_level": args.opt_level,
        "ast_path": str(ast_path),
        "ir_path": str(ir_path),
        "cfg_path": str(cfg_path),
        "cfg_kind": cfg_kind,
        "exe_path": str(exe_path) if build_success else "",
        "ast_success": ast_success,
        "ir_success": ir_success,
        "cfg_success": cfg_success,
        "build_success": build_success,
        "run_success": run_success,
        "error_message": " | ".join(error_messages),
        **environment,
    }
    static_row = {**base, **input_scale, **ast_features, **cfg_features, **ir_features}
    run_id_base = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"{args.env_id}:{program.program_id}:{input_id}:{time.time()}",
    ).hex[:12]
    run_rows = []
    for record in run_records:
        run_rows.append(
            {
                "run_id": f"{run_id_base}-{record['phase']}-{record['run_index']}",
                "program_id": program.program_id,
                "env_id": args.env_id,
                "input_id": input_id,
                "compile_config_id": args.compile_config_id,
                **record,
            }
        )
    summary_row = {
        "program_id": program.program_id,
        "env_id": args.env_id,
        "input_id": input_id,
        "compile_config_id": args.compile_config_id,
        "warmup_runs": args.warmup,
        "measured_runs": len([record for record in run_records if record["phase"] == "measure"]),
        "timeout_sec": args.timeout,
        **runtime_summary,
        **measurement_summary,
        "run_success": run_success,
    }
    return static_row, run_rows, summary_row, measurement_windows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AST/CFG/LLVM IR and collect runtime prediction features.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Directory containing source files.")
    parser.add_argument("--dataset", default="test", help="Dataset label written to CSV.")
    parser.add_argument("--task-type", default="mixed_test", help="Task type label written to CSV.")
    parser.add_argument("--parallel-model", default="serial", help="Parallel model label written to CSV.")
    parser.add_argument(
        "--env-id",
        default=None,
        help=(
            "Environment id written to CSV. Defaults to LLVM_RUNTIME_ENV_ID when set, "
            "otherwise hostname-architecture. Pass the same value to every dataset pipeline."
        ),
    )
    parser.add_argument("--input-id", default="default", help="Input id written to CSV.")
    parser.add_argument("--compile-config-id", default="release_O2", help="Compile config id written to CSV.")
    parser.add_argument("--opt-level", default="O2", choices=["O0", "O1", "O2", "O3"], help="Compiler optimization level.")
    parser.add_argument("--c-compiler", default="clang", help="C compiler command.")
    parser.add_argument("--cxx-compiler", default="clang++", help="C++ compiler command.")
    parser.add_argument("--run-c-compiler", default="gcc", help="C compiler used to build runnable executables.")
    parser.add_argument("--run-cxx-compiler", default="g++", help="C++ compiler used to build runnable executables.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs. Use 5 for formal collection.")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Fixed measured runs used only when --measurement-seconds is 0.",
    )
    parser.add_argument(
        "--measurement-seconds",
        type=float,
        default=10.0,
        help="Measured execution time budget per program. Defaults to 10 seconds.",
    )
    parser.add_argument(
        "--min-measured-runs",
        type=int,
        default=3,
        help="Minimum measured runs even when the time budget has already been reached.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Per-run timeout in seconds.")
    parser.add_argument("--extra-flag", action="append", help="Extra compiler flag. Can be repeated.")
    parser.add_argument(
        "--polybench-profile",
        action="append",
        choices=POLYBENCH_DATASET_PROFILES,
        help=(
            "PolyBench input profile to collect. Can be repeated. "
            "When omitted, all five profiles are collected."
        ),
    )
    parser.add_argument("--exclude", action="append", default=[], help="Glob pattern for source files to skip, relative to --source-dir. Can be repeated.")
    parser.add_argument("--no-run", action="store_true", help="Only generate representations and static features.")
    parser.add_argument("--prometheus-address", default="127.0.0.1", help="Address used by the Prometheus metrics HTTP server.")
    parser.add_argument("--prometheus-port", type=int, default=8000, help="Port used by the Prometheus metrics HTTP server.")
    parser.add_argument("--prometheus-url", default="http://127.0.0.1:9090", help="Prometheus HTTP API base URL.")
    parser.add_argument(
        "--prometheus-exporter",
        choices=("auto", "windows", "node", "off"),
        default="auto",
        help=(
            "Host exporter queried through Prometheus. 'auto' selects Windows Exporter "
            "on Windows and Node Exporter on Linux; 'off' disables host queries."
        ),
    )
    parser.add_argument(
        "--prometheus-job",
        default=None,
        help="Prometheus job label. Defaults to 'windows' or 'node' for the selected exporter.",
    )
    parser.add_argument(
        "--prometheus-instance",
        default="",
        help="Optional Prometheus instance label used to select one monitored host.",
    )
    parser.add_argument("--prometheus-query-step", type=float, default=1.0, help="Prometheus query_range step in seconds.")
    parser.add_argument("--prometheus-query-timeout", type=float, default=10.0, help="Timeout for one Prometheus API request.")
    parser.add_argument(
        "--prometheus-context-seconds",
        type=float,
        default=5.0,
        help=(
            "Quiet host-observation duration immediately before and after each measured "
            "benchmark window. Use at least two scrape intervals."
        ),
    )
    parser.add_argument(
        "--prometheus-query-delay",
        type=float,
        default=2.0,
        help="Seconds to wait for the final host-exporter scrape before querying Prometheus.",
    )
    parser.add_argument(
        "--prometheus-final-wait",
        type=float,
        default=20.0,
        help="Seconds to keep metrics available after the pipeline finishes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.runs < 1 or args.min_measured_runs < 1:
        parser.error("--runs and --min-measured-runs must be >= 1")
    if args.measurement_seconds < 0:
        parser.error("--measurement-seconds must be >= 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if (
        args.prometheus_query_delay < 0
        or args.prometheus_final_wait < 0
        or args.prometheus_context_seconds < 0
    ):
        parser.error("Prometheus wait values must be >= 0")
    if args.prometheus_query_step <= 0 or args.prometheus_query_timeout <= 0:
        parser.error("Prometheus query step and timeout must be > 0")

    args.env_id = args.env_id or default_env_id()
    args.prometheus_exporter = resolve_prometheus_exporter(args.prometheus_exporter)
    if args.prometheus_job is None:
        args.prometheus_job = {
            "windows": "windows",
            "node": "node",
        }.get(args.prometheus_exporter, "")
    host_metric_queries = prometheus_host_queries(
        args.prometheus_exporter,
        args.prometheus_job,
        args.prometheus_instance,
    )

    ensure_dirs()
    static_environment = collect_static_environment()
    source_dir = Path(args.source_dir).resolve()
    source_programs = discover_sources(source_dir, SOURCE_EXTENSIONS, args.exclude)
    if not source_programs:
        print(f"No source files found in {source_dir}")
        return 1
    try:
        programs, skipped_programs = expand_program_inputs(
            source_programs,
            args.polybench_profile or POLYBENCH_DATASET_PROFILES,
        )
    except ValueError as exc:
        print(f"Invalid PolyBench input configuration: {exc}")
        return 1
    if not programs:
        print(f"No benchmark programs with input profiles found in {source_dir}")
        return 1
    if skipped_programs:
        print(
            f"[pipeline] skipped {len(skipped_programs)} PolyBench helper/template "
            "source files without benchmark input profiles"
        )
    profiled_programs = len({program.program_id for program in programs if program.input_profile})
    if profiled_programs:
        print(
            f"[pipeline] expanded {profiled_programs} PolyBench programs into "
            f"{len(programs)} input configurations"
        )

    if os.name == "posix" and not args.no_run:
        try:
            process_probe = ensure_process_probe()
        except RuntimeError as exc:
            print(f"[pipeline] cannot initialize Linux process metrics: {exc}", file=sys.stderr)
            return 1
        print(f"[pipeline] process metrics probe: {process_probe}")

    start_http_server(args.prometheus_port, addr=args.prometheus_address)
    print(
        f"[prometheus] metrics available at "
        f"http://{args.prometheus_address}:{args.prometheus_port}/metrics"
    )
    if host_metric_queries:
        selector_description = f"job={args.prometheus_job or '*'}"
        if args.prometheus_instance:
            selector_description += f", instance={args.prometheus_instance}"
        print(
            f"[prometheus] host queries use {args.prometheus_exporter} exporter "
            f"({selector_description})"
        )
    else:
        print("[prometheus] host metric queries disabled")
    metric_labels = {"dataset": args.dataset}
    PIPELINE_RUNNING.labels(**metric_labels).set(1)
    PIPELINE_PROGRAMS_TOTAL.labels(**metric_labels).set(len(programs))
    PIPELINE_PROGRAMS_COMPLETED.labels(**metric_labels).set(0)
    PIPELINE_PROGRESS_RATIO.labels(**metric_labels).set(0)
    PIPELINE_LAST_RUN_SUCCESS.labels(**metric_labels).set(0)
    pipeline_success = True

    try:
        static_rows: list[dict[str, object]] = []
        run_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        prometheus_measurement_windows: list[
            tuple[dict[str, object], dict[str, tuple[float, float]]]
        ] = []
        for completed, program in enumerate(programs, start=1):
            profile_suffix = f" [{program.input_profile}]" if program.input_profile else ""
            print(f"[pipeline] processing {program.program_id}{profile_suffix}")
            program_start = time.perf_counter()
            static_row, program_run_rows, summary_row, measurement_windows = process_program(
                program,
                args,
                static_environment,
            )
            static_rows.append(static_row)
            run_rows.extend(program_run_rows)
            summary_rows.append(summary_row)
            if measurement_windows is not None:
                if host_metric_queries:
                    prometheus_measurement_windows.append((summary_row, measurement_windows))
                else:
                    summary_row.update(
                        empty_prometheus_three_phase_result(
                            measurement_windows,
                            "Prometheus host metric collection is disabled",
                            args.prometheus_context_seconds,
                        )
                    )
            else:
                summary_row.update(
                    empty_prometheus_three_phase_result(
                        None,
                        "Benchmark measurement was not executed",
                        args.prometheus_context_seconds,
                    )
                )

            program_success = bool(
                static_row["ast_success"]
                and static_row["ir_success"]
                and static_row["cfg_success"]
                and static_row["build_success"]
                and (args.no_run or static_row["run_success"])
            )
            status = "success" if program_success else "failed"
            pipeline_success = pipeline_success and program_success
            PIPELINE_PROGRAM_RESULTS.labels(dataset=args.dataset, status=status).inc()
            PIPELINE_PROGRAM_DURATION.labels(dataset=args.dataset, status=status).observe(
                time.perf_counter() - program_start
            )
            PIPELINE_PROGRAMS_COMPLETED.labels(**metric_labels).set(completed)
            PIPELINE_PROGRESS_RATIO.labels(**metric_labels).set(completed / len(programs))

        if prometheus_measurement_windows and args.prometheus_query_delay > 0:
            print(
                f"[prometheus] waiting {args.prometheus_query_delay:g} seconds "
                "for the final host scrape"
            )
            time.sleep(args.prometheus_query_delay)
        if prometheus_measurement_windows:
            print(
                "[prometheus] collecting host metrics for "
                f"{len(prometheus_measurement_windows)} program windows"
            )
        for summary_row, measurement_windows in prometheus_measurement_windows:
            try:
                metrics = collect_prometheus_three_phase_metrics(
                    args.prometheus_url,
                    measurement_windows,
                    max(args.prometheus_query_step, 0.1),
                    max(args.prometheus_query_timeout, 0.1),
                    host_metric_queries,
                    args.prometheus_context_seconds,
                )
            except Exception as exc:
                metrics = empty_prometheus_three_phase_result(
                    measurement_windows,
                    str(exc)[:500],
                    args.prometheus_context_seconds,
                )
                print(
                    f"[prometheus] warning: could not collect metrics for "
                    f"{summary_row['program_id']} [{summary_row['input_id']}]: {exc}"
                )
            summary_row.update(metrics)

        static_fields = [
            "program_id",
            "input_id",
            *STATIC_ENVIRONMENT_FIELDS,
            "dataset",
            "relative_path",
            "source_path",
            "language",
            "task_type",
            "parallel_model",
            "compile_config_id",
            "opt_level",
            "ast_path",
            "ir_path",
            "cfg_path",
            "cfg_kind",
            "exe_path",
            *INPUT_SCALE_FIELDS,
            *AST_FIELDS,
            *CFG_FIELDS,
            *IR_FIELDS,
            *STATUS_FIELDS,
        ]
        summary_fields = [
            "program_id",
            "env_id",
            "input_id",
            "compile_config_id",
            "warmup_runs",
            "measured_runs",
            "timeout_sec",
            *MEASUREMENT_SUMMARY_FIELDS,
            *PROMETHEUS_RESULT_FIELDS,
            "runtime_sec_median",
            "runtime_sec_mean",
            "runtime_sec_std",
            "runtime_cv",
            "runtime_min",
            "runtime_max",
            "run_success",
        ]
        run_fields = [
            "run_id",
            "program_id",
            "env_id",
            "input_id",
            "compile_config_id",
            "phase",
            "run_index",
            "runtime_sec",
            "return_code",
            "success",
            *PROCESS_RUN_FIELDS,
        ]
        combined_rows = []
        summary_by_program = {
            (row["program_id"], row["input_id"]): row
            for row in summary_rows
        }
        for row in static_rows:
            summary_key = (row["program_id"], row["input_id"])
            combined_rows.append({**row, **summary_by_program.get(summary_key, {})})

        static_fields = unique_fields(static_fields)
        summary_fields = unique_fields(summary_fields)
        run_fields = unique_fields(run_fields)
        results_fields = unique_fields(static_fields + [field for field in summary_fields if field != "program_id"])

        result_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = RESULTS_DIR / f"results_{result_stamp}.csv"
        if result_path.exists():
            result_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            result_path = RESULTS_DIR / f"results_{result_stamp}.csv"

        output_paths = [
            write_csv(RESULTS_DIR / "program_static_features.csv", static_rows, static_fields),
            write_csv(RESULTS_DIR / "run_records.csv", run_rows, run_fields),
            write_csv(RESULTS_DIR / "run_summary.csv", summary_rows, summary_fields),
            write_csv(result_path, combined_rows, results_fields),
        ]

        manifest = environment_manifest(args, static_environment)
        output_paths.append(write_text(
            RESULTS_DIR / "environment_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        ))
        PIPELINE_LAST_RUN_SUCCESS.labels(**metric_labels).set(1 if pipeline_success else 0)
        benchmark_source_count = len(source_programs) - len(skipped_programs)
        print(
            f"[pipeline] processed {len(programs)} input configurations from "
            f"{benchmark_source_count} source programs"
        )
        print("[pipeline] wrote:")
        for path in output_paths:
            print(f"  {path}")
        return 0
    except Exception:
        pipeline_success = False
        PIPELINE_LAST_RUN_SUCCESS.labels(**metric_labels).set(0)
        PIPELINE_STAGE_ERRORS.labels(dataset=args.dataset, stage="pipeline").inc()
        raise
    finally:
        PIPELINE_RUNNING.labels(**metric_labels).set(0)
        if args.prometheus_final_wait > 0:
            print(
                f"[prometheus] keeping final metrics available for "
                f"{args.prometheus_final_wait:g} seconds"
            )
            time.sleep(args.prometheus_final_wait)


if __name__ == "__main__":
    raise SystemExit(main())
