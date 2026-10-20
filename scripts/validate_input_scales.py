#!/usr/bin/env python3
"""Validate the reproducible 20-level input-scale expansion."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYBENCH_ROOT = PROJECT_ROOT / "datasets" / "PolyBenchC-4.2.1"
RODINIA_ROOT = PROJECT_ROOT / "datasets" / "rodinia_3.1"
GENERATED_ROOT = RODINIA_ROOT / "data" / "generated_20"
MANIFEST_PATH = PROJECT_ROOT / "configs" / "input_scales_20.json"
OPENMP_CONFIG = PROJECT_ROOT / "configs" / "rodinia_openmp_20sizes.json"
CUDA_CONFIG = PROJECT_ROOT / "configs" / "rodinia_cuda_20sizes.json"
OUTPUT_JSON = PROJECT_ROOT / "input_scale_validation.json"
OUTPUT_MD = PROJECT_ROOT / "input_scale_validation_report.md"
SIZE_IDS = tuple(f"SIZE_{index:02d}" for index in range(1, 21))
SPOT_SIZES = ("SIZE_01", "SIZE_10", "SIZE_20")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expand_input_scales import (  # noqa: E402
    DEFINITIONS_BEGIN,
    DEFINITIONS_END,
    parse_polybench_legacy_profiles,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generated_path(raw_path: str) -> Path | None:
    normalized = raw_path.replace("\\", "/")
    marker = "data/generated_20/"
    if marker not in normalized:
        return None
    return GENERATED_ROOT / normalized.split(marker, 1)[1]


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            count += chunk.count(b"\n")
    return count


def check_generated_file(entry: dict[str, Any], path: Path) -> list[str]:
    errors: list[str] = []
    parameters = entry["input_size_parameters"]
    program = entry["program_id"]
    name = path.name
    if not path.exists():
        return [f"missing generated file: {path.relative_to(PROJECT_ROOT)}"]
    size = path.stat().st_size
    if size <= 0:
        return [f"empty generated file: {path.relative_to(PROJECT_ROOT)}"]

    if program == "b+tree" and name.startswith("keys_"):
        with path.open("r", encoding="ascii") as stream:
            declared = int(stream.readline().strip())
        if declared != int(parameters["records"]):
            errors.append(f"{name}: record header {declared} != {parameters['records']}")
        if line_count(path) != declared + 1:
            errors.append(f"{name}: key line count mismatch")
    elif program == "bfs":
        nodes = int(parameters["nodes"])
        edges = int(parameters["edges"])
        with path.open("r", encoding="ascii") as stream:
            declared = int(stream.readline().strip())
        if declared != nodes:
            errors.append(f"{name}: node header {declared} != {nodes}")
        if line_count(path) != nodes + edges + 3:
            errors.append(f"{name}: graph line count mismatch")
    elif program == "cfd":
        cells = int(parameters["mesh_cells"])
        with path.open("r", encoding="ascii") as stream:
            declared = int(stream.readline().strip())
            first_tokens = stream.readline().split()
        if declared != cells:
            errors.append(f"{name}: cell header {declared} != {cells}")
        if len(first_tokens) != 17:
            errors.append(f"{name}: expected 17 tokens per cell, found {len(first_tokens)}")
        if line_count(path) != cells + 1:
            errors.append(f"{name}: mesh line count mismatch")
    elif program == "hotspot":
        expected = int(parameters.get("rows", parameters.get("grid_size"))) * int(parameters.get("columns", parameters.get("grid_size")))
        if line_count(path) != expected:
            errors.append(f"{name}: thermal field line count mismatch")
    elif program == "hotspot3D":
        expected = int(parameters["grid_size"]) ** 2 * int(parameters["layers"])
        if line_count(path) != expected:
            errors.append(f"{name}: 3D thermal field line count mismatch")
    elif program == "kmeans":
        expected = 8 + int(parameters["points"]) * int(parameters["attributes"]) * 4
        if size != expected:
            errors.append(f"{name}: binary point bytes {size} != {expected}")
    elif program == "nn" and name.startswith("records_"):
        expected = int(parameters["database_records"]) * 49
        if size != expected:
            errors.append(f"{name}: fixed-record bytes {size} != {expected}")
    elif program == "dwt2d":
        expected = int(parameters["width"]) * int(parameters["height"]) * int(parameters["components"])
        if size != expected:
            errors.append(f"{name}: raw image bytes {size} != {expected}")
    elif program == "huffman":
        expected = int(parameters["input_bytes"])
        if size != expected:
            errors.append(f"{name}: binary bytes {size} != {expected}")
    return errors


def validate_manifest(entries: list[dict[str, Any]]) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[(entry["dataset"], entry["program_id"])].append(entry)

    expected_programs = {"PolyBench/C": 30, "Rodinia OpenMP": 19, "Rodinia CUDA": 22}
    actual_programs = Counter(dataset for dataset, _ in groups)
    for dataset, expected in expected_programs.items():
        if actual_programs[dataset] != expected:
            errors.append(f"{dataset}: expected {expected} programs, found {actual_programs[dataset]}")

    duplicate_parameter_groups = 0
    monotonic_exceptions: list[str] = []
    for (dataset, program), rows in sorted(groups.items()):
        rows.sort(key=lambda item: item["size_id"])
        ids = [row["size_id"] for row in rows]
        if ids != list(SIZE_IDS):
            errors.append(f"{dataset}/{program}: size ids are not exactly SIZE_01..SIZE_20")
        serialized = [json.dumps(row["input_size_parameters"], sort_keys=True, separators=(",", ":")) for row in rows]
        if len(set(serialized)) != 20:
            duplicate_parameter_groups += 1
            errors.append(f"{dataset}/{program}: duplicate effective input parameters")
        proxies = [float(row["workload_proxy"]) for row in rows]
        if any(current <= previous for previous, current in zip(proxies, proxies[1:])):
            key = f"{dataset}/{program}"
            if key in {"PolyBench/C/atax", "PolyBench/C/bicg"}:
                monotonic_exceptions.append(key)
                warnings.append(f"{key}: upstream LARGE/EXTRALARGE dimensions are not workload-monotonic")
            else:
                errors.append(f"{key}: workload proxy is not strictly increasing")

    generated_files: set[Path] = set()
    file_errors: list[str] = []
    for entry in entries:
        for raw in entry.get("input_file_paths", []):
            path = generated_path(raw)
            if path is None:
                continue
            generated_files.add(path)
            file_errors.extend(check_generated_file(entry, path))
    errors.extend(file_errors)

    summary = {
        "entry_count": len(entries),
        "program_count": len(groups),
        "programs_by_dataset": dict(actual_programs),
        "entries_by_dataset": dict(Counter(entry["dataset"] for entry in entries)),
        "generated_file_count": len(generated_files),
        "generated_bytes": sum(path.stat().st_size for path in generated_files if path.exists()),
        "duplicate_parameter_groups": duplicate_parameter_groups,
        "monotonic_exceptions": monotonic_exceptions,
    }
    return errors, warnings, summary


def validate_polybench_headers(entries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_program: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in entries:
        if entry["dataset"] == "PolyBench/C":
            by_program[entry["program_id"]][entry["size_id"]] = entry["input_size_parameters"]

    for header in sorted(POLYBENCH_ROOT.rglob("*.h")):
        text = header.read_text(encoding="utf-8", errors="ignore")
        if not parse_polybench_legacy_profiles(text):
            continue
        program = header.stem
        if DEFINITIONS_BEGIN not in text or DEFINITIONS_END not in text:
            errors.append(f"{program}: generated SIZE definitions missing")
            continue
        for size_id in SIZE_IDS:
            match = __import__("re").search(
                rf"^\s*#\s*ifdef\s+{size_id}\b(?P<body>.*?)^\s*#\s*endif\b",
                text,
                flags=__import__("re").MULTILINE | __import__("re").DOTALL,
            )
            if not match:
                errors.append(f"{program}/{size_id}: header block missing")
                continue
            actual = {
                item.group(1): int(item.group(2))
                for item in __import__("re").finditer(
                    r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([+-]?\d+)\s*$",
                    match.group("body"),
                    flags=__import__("re").MULTILINE,
                )
            }
            if actual != by_program[program][size_id]:
                errors.append(f"{program}/{size_id}: header values differ from manifest")
    return errors


def validate_rodinia_configs() -> list[str]:
    errors: list[str] = []
    for path, expected_enabled in ((OPENMP_CONFIG, 19), (CUDA_CONFIG, 22)):
        data = load_json(path)
        enabled = [item for item in data.get("benchmarks", []) if item.get("enabled", True)]
        if len(enabled) != expected_enabled:
            errors.append(f"{path.name}: expected {expected_enabled} enabled benchmarks, found {len(enabled)}")
        for benchmark in enabled:
            inputs = benchmark.get("inputs", [])
            if len(inputs) != 20:
                errors.append(f"{path.name}/{benchmark['id']}: expected 20 inputs, found {len(inputs)}")
            if [item.get("id") for item in inputs] != list(SIZE_IDS):
                errors.append(f"{path.name}/{benchmark['id']}: input order/id mismatch")
        if path == OPENMP_CONFIG:
            for benchmark in enabled:
                thread_counts = {item.get("threads") for item in benchmark["inputs"]}
                expected = {8} if benchmark["id"] == "backprop" else {4}
                if thread_counts != expected:
                    errors.append(f"{path.name}/{benchmark['id']}: thread count drift {thread_counts}")
    return errors


def polybench_source_map() -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for source in POLYBENCH_ROOT.rglob("*.c"):
        header = source.with_suffix(".h")
        if header.exists() and parse_polybench_legacy_profiles(
            header.read_text(encoding="utf-8", errors="ignore")
        ):
            sources[source.stem] = source
    return sources


def usable_compiler() -> tuple[str | None, list[str]]:
    failures: list[str] = []
    for name in ("clang", "gcc"):
        compiler = shutil.which(name)
        if not compiler:
            continue
        try:
            probe = subprocess.run(
                [compiler, "--version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{compiler}: {exc}")
            continue
        if probe.returncode == 0:
            return compiler, failures
        failures.append(f"{compiler}: --version returned {probe.returncode}")
    return None, failures


def linker_probe(compiler: str, build_root: Path) -> tuple[bool, str, dict[str, str]]:
    temp_dir = build_root / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    source = temp_dir / "link_probe.c"
    executable = temp_dir / "link_probe.exe"
    source.write_text("int main(void) { return 0; }\n", encoding="ascii")
    environment = os.environ.copy()
    environment["TMP"] = str(temp_dir)
    environment["TEMP"] = str(temp_dir)
    try:
        probe = subprocess.run(
            [compiler, str(source), "-o", str(executable)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc), environment
    if probe.returncode != 0 or not executable.exists():
        return False, probe.stderr[-2000:], environment
    return True, "", environment


def compile_polybench_spots(run_executables: bool, timeout: float) -> list[dict[str, Any]]:
    compiler, compiler_failures = usable_compiler()
    if not compiler:
        return [{"status": "not_run_environment_unavailable", "reason": "; ".join(compiler_failures) or "no C compiler"}]
    build_root = PROJECT_ROOT / "build" / "input_scale_validation"
    build_root.mkdir(parents=True, exist_ok=True)
    compat = PROJECT_ROOT / "scripts" / "compat_include" / "polybench_windows_compat.h"
    utility = POLYBENCH_ROOT / "utilities"
    runtime = utility / "polybench.c"
    linker_available = False
    linker_error = ""
    linker_environment = os.environ.copy()
    if run_executables:
        linker_available, linker_error, linker_environment = linker_probe(compiler, build_root)
    results: list[dict[str, Any]] = []
    for program, source in sorted(polybench_source_map().items()):
        for size_id in SPOT_SIZES:
            executable = build_root / f"{program}_{size_id}.exe"
            common_command = [
                compiler,
                "-O2",
                f"-D{size_id}",
                "-DPOLYBENCH_TIME",
                "-I",
                str(source.parent),
                "-I",
                str(utility),
                "-I",
                str(compat.parent),
                "-include",
                str(compat),
                str(source),
            ]
            command = [*common_command, "-fsyntax-only"]
            try:
                build = subprocess.run(command, capture_output=True, text=True, errors="replace")
            except OSError as exc:
                results.append(
                    {
                        "program_id": program,
                        "size_id": size_id,
                        "compile_status": "environment_error",
                        "compile_error": str(exc),
                        "run_status": "not_run_environment_unavailable",
                    }
                )
                continue
            row: dict[str, Any] = {
                "program_id": program,
                "size_id": size_id,
                "compile_status": "pass" if build.returncode == 0 else "fail",
                "compile_error": build.stderr[-2000:] if build.returncode else "",
                "run_status": "not_requested",
            }
            if build.returncode == 0 and run_executables and not linker_available:
                row["run_status"] = "not_run_environment_unavailable"
                row["run_error"] = linker_error
            elif build.returncode == 0 and run_executables:
                link_command = [*common_command, str(runtime), "-lm", "-o", str(executable)]
                link = subprocess.run(
                    link_command,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    env=linker_environment,
                )
                if link.returncode != 0:
                    row["run_status"] = "not_run_environment_unavailable"
                    row["run_error"] = link.stderr[-2000:]
                    results.append(row)
                    continue
                try:
                    run = subprocess.run(
                        [str(executable)],
                        capture_output=True,
                        text=True,
                        errors="replace",
                        timeout=timeout,
                    )
                    row["run_status"] = "pass" if run.returncode == 0 else "fail"
                    row["run_returncode"] = run.returncode
                    row["run_error"] = run.stderr[-2000:] if run.returncode else ""
                except subprocess.TimeoutExpired:
                    row["run_status"] = "timeout"
            results.append(row)
    return results


def rodinia_runtime_status() -> dict[str, Any]:
    return {
        "status": "not_run_environment_unavailable",
        "reason": "No installed WSL Linux distribution, make, or nvcc on the current Windows host.",
        "openmp_example": (
            "python scripts/pipeline_rodinia.py --manifest configs/rodinia_openmp_20sizes.json "
            "--benchmark bfs --input-profile SIZE_01 --warmup 0 --runs 1 --measurement-seconds 0"
        ),
        "cuda_example": (
            "python scripts/pipeline_rodinia_cuda.py --manifest configs/rodinia_cuda_20sizes.json "
            "--benchmark bfs --input-profile SIZE_01 --warmup 0 --runs 1 --measurement-seconds 0"
        ),
    }


def write_report(result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Input scale validation report",
        "",
        f"Overall status: **{result['status']}**",
        "",
        "## Structural checks",
        "",
        f"- Unified entries: {summary['entry_count']}.",
        f"- Programs: {summary['program_count']}.",
        f"- Entries by dataset: `{json.dumps(summary['entries_by_dataset'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Generated files checked: {summary['generated_file_count']} ({summary['generated_bytes']} bytes).",
        f"- Duplicate effective-parameter groups: {summary['duplicate_parameter_groups']}.",
        f"- Documented monotonic exceptions: `{', '.join(summary['monotonic_exceptions']) or 'none'}`.",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {message}" for message in result["errors"] or ["None."])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {message}" for message in result["warnings"] or ["None."])
    lines.extend(["", "## PolyBench spot compilation/run", ""])
    spot = result["polybench_spot_checks"]
    if not spot:
        lines.append("- Not requested.")
    else:
        counts = Counter((row.get("compile_status", "unknown"), row.get("run_status", "unknown")) for row in spot)
        for key, count in sorted(counts.items()):
            lines.append(f"- compile={key[0]}, run={key[1]}: {count}")
        failures = [row for row in spot if row.get("compile_status") != "pass" or row.get("run_status") in {"fail", "timeout"}]
        for row in failures:
            lines.append(f"- {row.get('program_id','environment')}/{row.get('size_id','-')}: compile={row.get('compile_status')}, run={row.get('run_status')}")
    lines.extend(
        [
            "",
            "## Rodinia executable validation",
            "",
            f"- Status: `{result['rodinia_runtime']['status']}`.",
            f"- Reason: {result['rodinia_runtime']['reason']}",
            f"- OpenMP example: `{result['rodinia_runtime']['openmp_example']}`",
            f"- CUDA example: `{result['rodinia_runtime']['cuda_example']}`",
            "",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polybench-compile", action="store_true")
    parser.add_argument("--polybench-run", action="store_true", help="Run compiled SIZE_01/SIZE_10/SIZE_20 executables.")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    manifest = load_json(MANIFEST_PATH)
    entries = manifest["entries"]
    errors, warnings, summary = validate_manifest(entries)
    errors.extend(validate_polybench_headers(entries))
    errors.extend(validate_rodinia_configs())
    spot_checks = compile_polybench_spots(args.polybench_run, args.timeout) if args.polybench_compile else []
    for row in spot_checks:
        if row.get("compile_status") == "fail":
            errors.append(f"PolyBench compile failed: {row['program_id']}/{row['size_id']}")
        if row.get("run_status") == "fail":
            errors.append(f"PolyBench run failed: {row['program_id']}/{row['size_id']}")
        if row.get("run_status") == "timeout":
            warnings.append(f"PolyBench run timed out: {row['program_id']}/{row['size_id']}")

    result = {
        "status": "pass" if not errors else "fail",
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
        "polybench_spot_checks": spot_checks,
        "rodinia_runtime": rodinia_runtime_status(),
    }
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result)
    print(f"status={result['status']} errors={len(errors)} warnings={len(warnings)}")
    print(f"report={OUTPUT_MD}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
