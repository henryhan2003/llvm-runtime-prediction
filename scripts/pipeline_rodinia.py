#!/usr/bin/env python3
"""Build and measure Rodinia 3.1 OpenMP benchmarks on Linux.

The pipeline treats one Rodinia benchmark as one program, even when the
benchmark contains multiple translation units. Build and run behavior is
described by configs/rodinia_openmp.json so benchmark-specific commands do
not leak into the PolyBench pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "rodinia_openmp.json"
DEFAULT_RODINIA_ROOT = PROJECT_ROOT / "datasets" / "rodinia_3.1"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

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

ENVIRONMENT_FIELDS = [
    "env_id",
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
    "memory_total_bytes",
    "representation_compiler",
    "representation_compiler_version",
    "runtime_compiler",
    "runtime_compiler_version",
]

PROCESS_FIELDS = [
    "process_cpu_user_sec",
    "process_cpu_system_sec",
    "process_max_rss_bytes",
    "process_major_page_faults",
    "process_minor_page_faults",
    "process_fs_inputs",
    "process_fs_outputs",
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
    "thread_count",
    "workdir",
    "executable",
    "source_files",
    "source_file_count",
    "build_command",
    "run_command",
    "warmup_runs_requested",
    "warmup_runs_completed",
    "measurement_seconds_requested",
    "measurement_seconds_actual",
    "measured_runs",
    "runtime_sec_median",
    "runtime_sec_mean",
    "runtime_sec_std",
    "runtime_sec_cv",
    "runtime_sec_min",
    "runtime_sec_max",
    "process_metric_aggregation",
    *PROCESS_FIELDS,
    "process_max_rss_bytes_peak",
    "process_metrics_backend",
    *ENVIRONMENT_FIELDS,
    "static_status",
    "static_source_count",
    "static_source_success_count",
    *AST_FIELDS,
    *CFG_FIELDS,
    *IR_FIELDS,
    "build_success",
    "run_success",
    "error_message",
]

RUN_FIELDS = [
    "session_id",
    "run_id",
    "program_id",
    "input_id",
    "input_profile",
    "phase",
    "run_index",
    "started_at",
    "runtime_sec",
    "success",
    "returncode",
    "timed_out",
    *PROCESS_FIELDS,
    "process_metrics_backend",
    "error_message",
]


@dataclass(frozen=True)
class InputSpec:
    input_id: str
    profile: str
    args: tuple[str, ...]
    threads: int
    parameters: dict[str, Any]
    required_files: tuple[str, ...]
    environment: dict[str, str]


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    workdir: str
    executable: str
    build_commands: tuple[tuple[str, ...], ...]
    clean_commands: tuple[tuple[str, ...], ...]
    sources: tuple[str, ...]
    include_dirs: tuple[str, ...]
    static_flags: tuple[str, ...]
    required_files: tuple[str, ...]
    environment: dict[str, str]
    inputs: tuple[InputSpec, ...]
    enabled: bool
    disabled_reason: str


@dataclass
class ProcessResult:
    command: list[str]
    returncode: int | None
    elapsed_sec: float
    timed_out: bool
    error_message: str
    metrics: dict[str, float | int | str]

    @property
    def success(self) -> bool:
        return not self.timed_out and self.returncode == 0


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a JSON array of strings")
    return tuple(value)


def _commands(value: Any, label: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    commands: list[tuple[str, ...]] = []
    for index, command in enumerate(value):
        parsed = _string_tuple(command, f"{label}[{index}]")
        if not parsed:
            raise ValueError(f"{label}[{index}] cannot be empty")
        commands.append(parsed)
    return tuple(commands)


def load_manifest(path: Path) -> tuple[BenchmarkSpec, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    raw_benchmarks = data.get("benchmarks")
    if not isinstance(raw_benchmarks, list):
        raise ValueError("manifest benchmarks must be a JSON array")

    benchmarks: list[BenchmarkSpec] = []
    seen: set[str] = set()
    for raw in raw_benchmarks:
        if not isinstance(raw, dict):
            raise ValueError("each benchmark entry must be a JSON object")
        benchmark_id = str(raw.get("id", "")).strip()
        if not benchmark_id or benchmark_id in seen:
            raise ValueError(f"invalid or duplicate benchmark id: {benchmark_id!r}")
        seen.add(benchmark_id)

        raw_inputs = raw.get("inputs", [])
        if not isinstance(raw_inputs, list):
            raise ValueError(f"{benchmark_id}.inputs must be a JSON array")
        inputs: list[InputSpec] = []
        input_ids: set[str] = set()
        for item in raw_inputs:
            if not isinstance(item, dict):
                raise ValueError(f"{benchmark_id}.inputs entries must be objects")
            input_id = str(item.get("id", "")).strip()
            profile = str(item.get("profile", "")).strip()
            if not input_id or input_id in input_ids or profile not in {"smoke", "official"}:
                raise ValueError(f"invalid input entry in {benchmark_id}: {input_id!r}")
            input_ids.add(input_id)
            parameters = item.get("parameters", {})
            environment = item.get("environment", {})
            if not isinstance(parameters, dict) or not isinstance(environment, dict):
                raise ValueError(f"{benchmark_id}.{input_id} parameters/environment must be objects")
            inputs.append(
                InputSpec(
                    input_id=input_id,
                    profile=profile,
                    args=_string_tuple(item.get("args", []), f"{benchmark_id}.{input_id}.args"),
                    threads=int(item.get("threads", 1)),
                    parameters=parameters,
                    required_files=_string_tuple(
                        item.get("required_files", []),
                        f"{benchmark_id}.{input_id}.required_files",
                    ),
                    environment={str(key): str(value) for key, value in environment.items()},
                )
            )

        enabled = bool(raw.get("enabled", True))
        if enabled and not inputs:
            raise ValueError(f"enabled benchmark {benchmark_id} has no inputs")
        environment = raw.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError(f"{benchmark_id}.environment must be an object")
        benchmarks.append(
            BenchmarkSpec(
                benchmark_id=benchmark_id,
                workdir=str(raw.get("workdir", benchmark_id)),
                executable=str(raw.get("executable", "")),
                build_commands=_commands(raw.get("build", []), f"{benchmark_id}.build"),
                clean_commands=_commands(raw.get("clean", []), f"{benchmark_id}.clean"),
                sources=_string_tuple(raw.get("sources", []), f"{benchmark_id}.sources"),
                include_dirs=_string_tuple(
                    raw.get("include_dirs", []), f"{benchmark_id}.include_dirs"
                ),
                static_flags=_string_tuple(
                    raw.get("static_flags", ["-fopenmp"]),
                    f"{benchmark_id}.static_flags",
                ),
                required_files=_string_tuple(
                    raw.get("required_files", []), f"{benchmark_id}.required_files"
                ),
                environment={str(key): str(value) for key, value in environment.items()},
                inputs=tuple(inputs),
                enabled=enabled,
                disabled_reason=str(raw.get("disabled_reason", "")),
            )
        )
    return tuple(benchmarks)


def expand_tokens(tokens: Iterable[str]) -> list[str]:
    context = {"null_device": "/dev/null"}
    return [token.format_map(context) for token in tokens]


def command_text(command: Iterable[str]) -> str:
    return shlex.join(list(command))


def _read_log(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > limit:
        data = data[-limit:]
    return data.decode("utf-8", errors="replace").strip()


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _parse_gnu_time(path: Path) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    if not path.exists():
        return metrics
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    try:
        metrics = {
            "command_elapsed_sec": float(values["elapsed_sec"]),
            "process_cpu_user_sec": float(values["user_sec"]),
            "process_cpu_system_sec": float(values["system_sec"]),
            "process_max_rss_bytes": int(values["max_rss_kb"]) * 1024,
            "process_major_page_faults": int(values["major_faults"]),
            "process_minor_page_faults": int(values["minor_faults"]),
            "process_fs_inputs": int(values["fs_inputs"]),
            "process_fs_outputs": int(values["fs_outputs"]),
            "process_metrics_backend": "gnu-time",
        }
    except (KeyError, ValueError):
        return {}
    return metrics


def run_process(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    timeout: float,
    measure_process: bool,
    discard_stdout: bool,
) -> ProcessResult:
    gnu_time = Path("/usr/bin/time")
    use_gnu_time = measure_process and os.name == "posix" and gnu_time.exists()
    with tempfile.TemporaryDirectory(prefix="rodinia-pipeline-") as temporary:
        temporary_dir = Path(temporary)
        log_path = temporary_dir / "process.log"
        metrics_path = temporary_dir / "time.metrics"
        actual_command = list(command)
        if use_gnu_time:
            time_format = (
                "elapsed_sec=%e\nuser_sec=%U\nsystem_sec=%S\nmax_rss_kb=%M\n"
                "major_faults=%F\nminor_faults=%R\n"
                "fs_inputs=%I\nfs_outputs=%O"
            )
            actual_command = [
                str(gnu_time),
                "-f",
                time_format,
                "-o",
                str(metrics_path),
                "--",
                *command,
            ]

        started = time.perf_counter()
        timed_out = False
        returncode: int | None = None
        error_message = ""
        try:
            with log_path.open("wb") as log_file:
                stdout_target: Any = subprocess.DEVNULL if discard_stdout else log_file
                process = subprocess.Popen(
                    actual_command,
                    cwd=str(cwd),
                    env=environment,
                    stdout=stdout_target,
                    stderr=log_file,
                    start_new_session=os.name == "posix",
                )
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_process_group(process)
                    process.wait()
                    returncode = process.returncode
        except OSError as exc:
            error_message = str(exc)
        elapsed = time.perf_counter() - started
        log_text = _read_log(log_path)
        if timed_out:
            error_message = f"timeout after {timeout:g}s"
        elif returncode not in (None, 0):
            error_message = log_text or f"command exited with code {returncode}"
        elif error_message:
            error_message = error_message
        metrics = _parse_gnu_time(metrics_path) if use_gnu_time else {}
        command_elapsed = metrics.pop("command_elapsed_sec", None)
        if not timed_out and isinstance(command_elapsed, (int, float)) and command_elapsed > 0:
            elapsed = float(command_elapsed)
        return ProcessResult(
            command=command,
            returncode=returncode,
            elapsed_sec=elapsed,
            timed_out=timed_out,
            error_message=error_message[-4000:],
            metrics=metrics,
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)B?\s*", value, re.IGNORECASE)
    if not match:
        return 0
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    return int(float(match.group(1)) * scale[match.group(2).upper()])


def _compiler_version(command: str) -> str:
    executable = shutil.which(command)
    if not executable:
        return "not found"
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _cpu_cache_sizes() -> dict[int, int]:
    sizes: dict[int, int] = {}
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    if not cache_root.exists():
        return sizes
    for index_dir in cache_root.glob("index*"):
        try:
            level = int((index_dir / "level").read_text().strip())
            size = _parse_size((index_dir / "size").read_text().strip())
        except (OSError, ValueError):
            continue
        sizes[level] = sizes.get(level, 0) + size
    return sizes


def collect_environment(c_compiler: str, cxx_compiler: str) -> dict[str, Any]:
    cpu_records: list[dict[str, str]] = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for section in cpuinfo.read_text(encoding="utf-8", errors="ignore").split("\n\n"):
            record: dict[str, str] = {}
            for line in section.splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    record[key.strip()] = value.strip()
            if record:
                cpu_records.append(record)
    first = cpu_records[0] if cpu_records else {}
    sockets = {record.get("physical id") for record in cpu_records if record.get("physical id")}
    physical_cores = {
        (record.get("physical id", "0"), record.get("core id"))
        for record in cpu_records
        if record.get("core id") is not None
    }
    logical_cores = os.cpu_count() or len(cpu_records) or 1
    physical_count = len(physical_cores) or logical_cores
    socket_count = len(sockets) or 1
    threads_per_core = logical_cores / physical_count if physical_count else 1

    memory_total = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        match = re.search(
            r"^MemTotal:\s+(\d+)\s+kB",
            meminfo.read_text(encoding="utf-8", errors="ignore"),
            re.MULTILINE,
        )
        if match:
            memory_total = int(match.group(1)) * 1024
    caches = _cpu_cache_sizes()
    c_version = _compiler_version(c_compiler)
    cxx_version = _compiler_version(cxx_compiler)
    node = platform.node() or "unknown-host"
    architecture = platform.machine()
    return {
        "env_id": f"{node}-{architecture}",
        "os_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "cpu_model": first.get("model name", platform.processor()),
        "cpu_vendor": first.get("vendor_id", ""),
        "cpu_identifier": first.get("model", ""),
        "cpu_architecture": architecture,
        "cpu_socket_count": socket_count,
        "cpu_physical_core_count": physical_count,
        "cpu_logical_core_count": logical_cores,
        "cpu_threads_per_core": threads_per_core,
        "cpu_nominal_frequency_mhz": float(first.get("cpu MHz", 0) or 0),
        "cpu_l1_cache_bytes": caches.get(1, 0),
        "cpu_l2_cache_bytes": caches.get(2, 0),
        "cpu_l3_cache_bytes": caches.get(3, 0),
        "memory_total_bytes": memory_total,
        "representation_compiler": f"{c_compiler} / {cxx_compiler}",
        "representation_compiler_version": f"{c_version} | {cxx_version}",
        "runtime_compiler": "gcc / g++ (Rodinia Makefiles)",
        "runtime_compiler_version": f"{_compiler_version('gcc')} | {_compiler_version('g++')}",
    }


def _zero_fields(fields: Iterable[str]) -> dict[str, int | float]:
    return {field: 0 for field in fields}


def _weighted_average(rows: list[dict[str, Any]], average: str, count: str) -> float:
    denominator = sum(float(row.get(count, 0) or 0) for row in rows)
    if denominator == 0:
        return 0.0
    numerator = sum(
        float(row.get(average, 0) or 0) * float(row.get(count, 0) or 0) for row in rows
    )
    return numerator / denominator


def aggregate_ast(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    result = _zero_fields(AST_FIELDS)
    special = {"recursive_call_flag", "max_loop_depth", "max_integer_literal", "avg_integer_literal"}
    for field in AST_FIELDS:
        if field not in special:
            result[field] = sum(float(row.get(field, 0) or 0) for row in rows)
    result["recursive_call_flag"] = int(any(row.get("recursive_call_flag", 0) for row in rows))
    result["max_loop_depth"] = max((float(row.get("max_loop_depth", 0) or 0) for row in rows), default=0)
    result["max_integer_literal"] = max(
        (float(row.get("max_integer_literal", 0) or 0) for row in rows), default=0
    )
    result["avg_integer_literal"] = _weighted_average(
        rows, "avg_integer_literal", "integer_literal_count"
    )
    return result


def aggregate_cfg(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    result = _zero_fields(CFG_FIELDS)
    special = {"avg_in_degree", "avg_out_degree", "max_out_degree", "max_cfg_loop_depth", "cfg_depth"}
    for field in CFG_FIELDS:
        if field not in special:
            result[field] = sum(float(row.get(field, 0) or 0) for row in rows)
    blocks = float(result["basic_block_count"])
    edges = float(result["cfg_edge_count"])
    result["avg_in_degree"] = edges / blocks if blocks else 0
    result["avg_out_degree"] = edges / blocks if blocks else 0
    for field in ("max_out_degree", "max_cfg_loop_depth", "cfg_depth"):
        result[field] = max((float(row.get(field, 0) or 0) for row in rows), default=0)
    return result


def aggregate_ir(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    result = _zero_fields(IR_FIELDS)
    special = {
        "max_constant_ir",
        "avg_constant_ir",
        "load_store_ratio",
        "memory_arithmetic_ratio",
        "branch_instruction_ratio",
    }
    for field in IR_FIELDS:
        if field not in special:
            result[field] = sum(float(row.get(field, 0) or 0) for row in rows)
    result["max_constant_ir"] = max(
        (float(row.get("max_constant_ir", 0) or 0) for row in rows), default=0
    )
    result["avg_constant_ir"] = _weighted_average(rows, "avg_constant_ir", "constant_count_ir")
    stores = float(result["store_count"])
    arithmetic = sum(
        float(result[field])
        for field in (
            "add_count",
            "sub_count",
            "mul_count",
            "div_count",
            "rem_count",
            "fadd_count",
            "fsub_count",
            "fmul_count",
            "fdiv_count",
        )
    )
    instructions = float(result["ir_instruction_count"])
    result["load_store_ratio"] = float(result["load_count"]) / stores if stores else 0
    result["memory_arithmetic_ratio"] = (
        float(result["memory_inst_count"]) / arithmetic if arithmetic else 0
    )
    result["branch_instruction_ratio"] = (
        float(result["branch_count"]) / instructions if instructions else 0
    )
    return result


def empty_static(status: str) -> dict[str, Any]:
    return {
        "static_status": status,
        "static_source_count": 0,
        "static_source_success_count": 0,
        **_zero_fields(AST_FIELDS),
        **_zero_fields(CFG_FIELDS),
        **_zero_fields(IR_FIELDS),
    }


def collect_static_features(
    spec: BenchmarkSpec,
    openmp_root: Path,
    c_compiler: str,
    cxx_compiler: str,
    opt_level: str,
) -> tuple[dict[str, Any], list[str]]:
    try:
        import pipeline_new as core
    except Exception as exc:
        return empty_static("unavailable"), [f"static pipeline import failed: {exc}"]

    core.AST_DIR = PROJECT_ROOT / "ast" / "rodinia_openmp"
    core.IR_DIR = PROJECT_ROOT / "llvm_ir" / "rodinia_openmp"
    core.CFG_DIR = PROJECT_ROOT / "cfg" / "rodinia_openmp"
    core.ensure_dirs()

    workdir = openmp_root / spec.workdir
    extra_flags = list(spec.static_flags)
    for include_dir in spec.include_dirs:
        extra_flags.extend(["-I", str((workdir / include_dir).resolve())])

    ast_rows: list[dict[str, Any]] = []
    cfg_rows: list[dict[str, Any]] = []
    ir_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    source_success_count = 0

    for relative_source in spec.sources:
        source_path = (workdir / relative_source).resolve()
        if not source_path.exists():
            errors.append(f"missing static source: {source_path}")
            continue
        program = core.Program(source_path=source_path, source_root=openmp_root)
        source_ok = True
        try:
            ast_path, ast_ok, ast_error = core.generate_ast(
                program, c_compiler, cxx_compiler, extra_flags
            )
            if ast_ok:
                ast_rows.append(core.extract_ast_features(ast_path, source_path))
            else:
                source_ok = False
                errors.append(f"AST {program.program_id}: {ast_error}")

            ir_path, ir_ok, ir_error = core.generate_ir(
                program, c_compiler, cxx_compiler, opt_level, extra_flags
            )
            if ir_ok:
                ir_rows.append(core.extract_ir_features(ir_path))
            else:
                source_ok = False
                errors.append(f"IR {program.program_id}: {ir_error}")

            cfg_path, cfg_ok, cfg_error, _ = core.generate_cfg(
                program, ir_path, c_compiler, cxx_compiler, extra_flags
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

    status = "success" if source_success_count == len(spec.sources) else "partial"
    if not spec.sources:
        status = "no_sources"
    features = {
        "static_status": status,
        "static_source_count": len(spec.sources),
        "static_source_success_count": source_success_count,
        **aggregate_ast(ast_rows),
        **aggregate_cfg(cfg_rows),
        **aggregate_ir(ir_rows),
    }
    return features, errors


def validate_paths(
    spec: BenchmarkSpec,
    input_spec: InputSpec,
    openmp_root: Path,
    require_executable: bool,
) -> list[str]:
    workdir = openmp_root / spec.workdir
    missing: list[str] = []
    if not workdir.is_dir():
        return [f"missing workdir: {workdir}"]
    paths = [*spec.required_files, *input_spec.required_files]
    for relative_path in paths:
        path = (workdir / relative_path).resolve()
        if not path.exists():
            missing.append(f"missing required file: {path}")
    for relative_source in spec.sources:
        source = (workdir / relative_source).resolve()
        if not source.exists():
            missing.append(f"missing source file: {source}")
    if require_executable:
        executable = (workdir / spec.executable).resolve()
        if not executable.is_file():
            missing.append(f"missing executable: {executable}")
    return missing


def build_benchmark(
    spec: BenchmarkSpec,
    openmp_root: Path,
    clean: bool,
    skip_build: bool,
    build_timeout: float,
) -> tuple[bool, str, list[str]]:
    workdir = openmp_root / spec.workdir
    environment = os.environ.copy()
    environment.update(spec.environment)
    errors: list[str] = []
    build_texts: list[str] = []

    if clean and not skip_build:
        for raw_command in spec.clean_commands:
            command = expand_tokens(raw_command)
            result = run_process(
                command, workdir, environment, build_timeout, False, False
            )
            if not result.success:
                print(f"[clean] warning for {spec.benchmark_id}: {result.error_message}")

    if not skip_build:
        for raw_command in spec.build_commands:
            command = expand_tokens(raw_command)
            build_texts.append(command_text(command))
            print(f"[build] {spec.benchmark_id}: {command_text(command)}")
            result = run_process(
                command, workdir, environment, build_timeout, False, False
            )
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


def _run_row(
    session_id: str,
    run_id: str,
    spec: BenchmarkSpec,
    input_spec: InputSpec,
    phase: str,
    index: int,
    started_at: str,
    result: ProcessResult,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "session_id": session_id,
        "run_id": run_id,
        "program_id": spec.benchmark_id,
        "input_id": input_spec.input_id,
        "input_profile": input_spec.profile,
        "phase": phase,
        "run_index": index,
        "started_at": started_at,
        "runtime_sec": result.elapsed_sec,
        "success": result.success,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "error_message": result.error_message,
    }
    row.update({field: result.metrics.get(field, "") for field in PROCESS_FIELDS})
    row["process_metrics_backend"] = result.metrics.get("process_metrics_backend", "unavailable")
    return row


def measure_input(
    session_id: str,
    spec: BenchmarkSpec,
    input_spec: InputSpec,
    openmp_root: Path,
    warmup: int,
    runs: int,
    measurement_seconds: float,
    timeout: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    workdir = openmp_root / spec.workdir
    executable = (workdir / spec.executable).resolve()
    args = expand_tokens(input_spec.args)
    command = [str(executable), *args]
    environment = os.environ.copy()
    environment.update(spec.environment)
    environment.update(input_spec.environment)
    environment.setdefault("OMP_NUM_THREADS", str(input_spec.threads))
    environment.setdefault("OMP_DYNAMIC", "FALSE")
    all_rows: list[dict[str, Any]] = []
    measured_results: list[ProcessResult] = []
    errors: list[str] = []
    warmup_completed = 0

    def execute(phase: str, index: int) -> ProcessResult:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        run_id = f"{session_id}-{spec.benchmark_id}-{input_spec.input_id}-{phase}-{index}"
        result = run_process(command, workdir, environment, timeout, True, True)
        all_rows.append(
            _run_row(
                session_id,
                run_id,
                spec,
                input_spec,
                phase,
                index,
                started_at,
                result,
            )
        )
        return result

    for index in range(1, warmup + 1):
        result = execute("warmup", index)
        if not result.success:
            errors.append(f"WARMUP: {result.error_message}")
            break
        warmup_completed += 1

    measured_runtime_total = 0.0
    if warmup_completed == warmup:
        index = 1
        while True:
            if measurement_seconds > 0:
                if measured_results and measured_runtime_total >= measurement_seconds:
                    break
            elif index > runs:
                break
            result = execute("measure", index)
            if not result.success:
                errors.append(f"RUN: {result.error_message}")
                break
            measured_results.append(result)
            measured_runtime_total += result.elapsed_sec
            index += 1

    runtimes = [result.elapsed_sec for result in measured_results]
    mean = statistics.fmean(runtimes) if runtimes else 0.0
    std = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0
    summary: dict[str, Any] = {
        "warmup_runs_completed": warmup_completed,
        "measurement_seconds_actual": sum(runtimes),
        "measured_runs": len(measured_results),
        "runtime_sec_median": statistics.median(runtimes) if runtimes else 0.0,
        "runtime_sec_mean": mean,
        "runtime_sec_std": std,
        "runtime_sec_cv": std / mean if mean else 0.0,
        "runtime_sec_min": min(runtimes) if runtimes else 0.0,
        "runtime_sec_max": max(runtimes) if runtimes else 0.0,
        "process_metric_aggregation": "mean of measured runs",
    }
    for field in PROCESS_FIELDS:
        values = [
            float(result.metrics[field])
            for result in measured_results
            if field in result.metrics and isinstance(result.metrics[field], (int, float))
        ]
        summary[field] = statistics.fmean(values) if values else ""
    rss_values = [
        int(result.metrics["process_max_rss_bytes"])
        for result in measured_results
        if "process_max_rss_bytes" in result.metrics
    ]
    summary["process_max_rss_bytes_peak"] = max(rss_values) if rss_values else ""
    backends = sorted(
        {
            str(result.metrics.get("process_metrics_backend", "unavailable"))
            for result in measured_results
        }
    )
    summary["process_metrics_backend"] = ",".join(backends) if backends else "unavailable"
    summary["run_success"] = bool(measured_results) and not errors
    return summary, all_rows, errors


def selected_benchmarks(
    benchmarks: tuple[BenchmarkSpec, ...],
    requested: list[str],
    select_all: bool,
) -> list[BenchmarkSpec]:
    by_id = {benchmark.benchmark_id: benchmark for benchmark in benchmarks}
    names = [name.strip() for item in requested for name in item.split(",") if name.strip()]
    if select_all:
        return [benchmark for benchmark in benchmarks if benchmark.enabled]
    if not names:
        raise ValueError("choose at least one --benchmark NAME or use --all")
    unknown = [name for name in names if name not in by_id]
    if unknown:
        raise ValueError(f"unknown benchmark(s): {', '.join(unknown)}")
    selected: list[BenchmarkSpec] = []
    for name in dict.fromkeys(names):
        benchmark = by_id[name]
        if not benchmark.enabled:
            reason = benchmark.disabled_reason or "disabled by manifest"
            raise ValueError(f"benchmark {name} is disabled: {reason}")
        selected.append(benchmark)
    return selected


def input_specs(spec: BenchmarkSpec, profile: str) -> list[InputSpec]:
    if profile == "all":
        return list(spec.inputs)
    return [item for item in spec.inputs if item.profile == profile]


def print_manifest(benchmarks: tuple[BenchmarkSpec, ...]) -> None:
    print("Rodinia OpenMP benchmark manifest")
    for benchmark in benchmarks:
        profiles = ",".join(sorted({item.profile for item in benchmark.inputs})) or "-"
        state = "enabled" if benchmark.enabled else f"disabled: {benchmark.disabled_reason}"
        print(f"  {benchmark.benchmark_id:<16} {state:<52} profiles={profiles}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and measure Rodinia 3.1 OpenMP benchmarks on Linux."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rodinia-root", type=Path, default=DEFAULT_RODINIA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--benchmark", action="append", default=[], help="Benchmark id; repeat or use commas.")
    parser.add_argument("--all", action="store_true", help="Run every enabled benchmark.")
    parser.add_argument("--list", action="store_true", help="List the manifest and exit.")
    parser.add_argument("--input-profile", choices=("smoke", "official", "all"), default="smoke")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup run count; use 5 for formal collection.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs when --measurement-seconds is 0.")
    parser.add_argument("--measurement-seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout for one benchmark run.")
    parser.add_argument("--build-timeout", type=float, default=600.0)
    parser.add_argument("--clean", action="store_true", help="Run the benchmark clean command before building.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing executable.")
    parser.add_argument("--build-only", action="store_true", help="Build and optionally extract static features only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print commands without executing.")
    parser.add_argument("--c-compiler", default="clang")
    parser.add_argument("--cxx-compiler", default="clang++")
    parser.add_argument("--opt-level", choices=("O0", "O1", "O2", "O3"), default="O2")
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument("--collect-static", dest="collect_static", action="store_true")
    static_group.add_argument("--no-static", dest="collect_static", action="store_false")
    parser.set_defaults(collect_static=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.runs < 1 or args.measurement_seconds < 0:
        parser.error("warmup must be >= 0, runs >= 1, and measurement-seconds >= 0")
    if args.timeout <= 0 or args.build_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.all and args.benchmark:
        parser.error("use either --all or --benchmark, not both")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"[pipeline] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    try:
        benchmarks = load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[pipeline] invalid manifest: {exc}", file=sys.stderr)
        return 2
    if args.list:
        print_manifest(benchmarks)
        return 0
    try:
        selected = selected_benchmarks(benchmarks, args.benchmark, args.all)
    except ValueError as exc:
        print(f"[pipeline] {exc}", file=sys.stderr)
        return 2

    rodinia_root = args.rodinia_root.resolve()
    openmp_root = rodinia_root / "openmp"
    if not openmp_root.is_dir():
        print(f"[pipeline] Rodinia OpenMP directory not found: {openmp_root}", file=sys.stderr)
        return 2
    if os.name != "posix" and not args.dry_run:
        print("[pipeline] this runtime pipeline must execute on Linux; use --dry-run on Windows", file=sys.stderr)
        return 2

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    collected_at = datetime.now().astimezone().isoformat(timespec="seconds")
    environment_features = collect_environment(args.c_compiler, args.cxx_compiler)
    summary_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    any_failure = False

    for spec in selected:
        inputs = input_specs(spec, args.input_profile)
        if not inputs:
            print(f"[pipeline] warning: {spec.benchmark_id} has no {args.input_profile} input")
            continue
        workdir = openmp_root / spec.workdir
        print(f"[pipeline] processing {spec.benchmark_id} in {workdir}")

        if args.dry_run:
            for command in spec.build_commands:
                print(f"  build: {command_text(expand_tokens(command))}")
            for item in inputs:
                errors = validate_paths(spec, item, openmp_root, False)
                run_cmd = [f"./{spec.executable}", *expand_tokens(item.args)]
                print(f"  run[{item.input_id}]: {command_text(run_cmd)}")
                for error in errors:
                    print(f"  ERROR: {error}")
                    any_failure = True
            continue

        build_success, build_command, build_errors = build_benchmark(
            spec,
            openmp_root,
            args.clean,
            args.skip_build,
            args.build_timeout,
        )
        if args.collect_static:
            static_features, static_errors = collect_static_features(
                spec,
                openmp_root,
                args.c_compiler,
                args.cxx_compiler,
                args.opt_level,
            )
        else:
            static_features, static_errors = empty_static("not_requested"), []

        for item in inputs:
            run_command_display = [f"./{spec.executable}", *expand_tokens(item.args)]
            errors = [*build_errors, *static_errors]
            path_errors = validate_paths(spec, item, openmp_root, build_success)
            errors.extend(path_errors)
            base_row: dict[str, Any] = {
                "session_id": session_id,
                "collected_at": collected_at,
                "dataset": "Rodinia-3.1",
                "parallel_model": "OpenMP",
                "program_id": spec.benchmark_id,
                "input_id": item.input_id,
                "input_profile": item.profile,
                "input_size_parameters": json.dumps(
                    item.parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "thread_count": item.threads,
                "workdir": spec.workdir,
                "executable": spec.executable,
                "source_files": ";".join(spec.sources),
                "source_file_count": len(spec.sources),
                "build_command": build_command,
                "run_command": command_text(run_command_display),
                "warmup_runs_requested": args.warmup,
                "measurement_seconds_requested": args.measurement_seconds,
                "build_success": build_success,
                **environment_features,
                **static_features,
            }
            if build_success and not path_errors and not args.build_only:
                print(
                    f"[run] {spec.benchmark_id}[{item.input_id}] "
                    f"warmup={args.warmup}, budget={args.measurement_seconds:g}s"
                )
                measured, detail_rows, run_errors = measure_input(
                    session_id,
                    spec,
                    item,
                    openmp_root,
                    args.warmup,
                    args.runs,
                    args.measurement_seconds,
                    args.timeout,
                )
                base_row.update(measured)
                run_rows.extend(detail_rows)
                errors.extend(run_errors)
            else:
                base_row.update(
                    {
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
                        **{field: "" for field in PROCESS_FIELDS},
                        "process_max_rss_bytes_peak": "",
                        "process_metrics_backend": "not_run",
                        "run_success": "not_run" if args.build_only and build_success else False,
                    }
                )
            base_row["error_message"] = " | ".join(error for error in errors if error)[-12000:]
            summary_rows.append(base_row)
            if not build_success or (not args.build_only and not base_row.get("run_success")):
                any_failure = True

    if args.dry_run:
        print("[pipeline] dry run completed")
        return 1 if any_failure else 0

    results_dir = args.results_dir.resolve()
    summary_path = results_dir / f"rodinia_openmp_summary_{session_id}.csv"
    runs_path = results_dir / f"rodinia_openmp_runs_{session_id}.csv"
    environment_path = results_dir / f"rodinia_openmp_environment_{session_id}.json"
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_csv(runs_path, run_rows, RUN_FIELDS)
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "collected_at": collected_at,
                "manifest": str(manifest_path),
                "rodinia_root": str(rodinia_root),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "environment": environment_features,
                "benchmarks": [spec.benchmark_id for spec in selected],
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
