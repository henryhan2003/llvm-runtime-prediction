#!/usr/bin/env python3
"""Generate reproducible 20-level input configurations for LLVM_v01 benchmarks.

The script preserves the original Rodinia manifests and data files. Generated
Rodinia inputs are written below data/generated_20, while PolyBench headers are
extended in place with idempotent, marker-delimited SIZE_01..SIZE_20 blocks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import struct
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYBENCH_ROOT = PROJECT_ROOT / "datasets" / "PolyBenchC-4.2.1"
RODINIA_ROOT = PROJECT_ROOT / "datasets" / "rodinia_3.1"
GENERATED_DATA_ROOT = RODINIA_ROOT / "data" / "generated_20"
HUFFMAN_REFERENCE = RODINIA_ROOT / "data" / "huffman" / "test1024_H2.206587175259.in"
OPENMP_SOURCE_MANIFEST = PROJECT_ROOT / "configs" / "rodinia_openmp.json"
CUDA_SOURCE_MANIFEST = PROJECT_ROOT / "configs" / "rodinia_cuda.json"
OPENMP_OUTPUT_MANIFEST = PROJECT_ROOT / "configs" / "rodinia_openmp_20sizes.json"
CUDA_OUTPUT_MANIFEST = PROJECT_ROOT / "configs" / "rodinia_cuda_20sizes.json"
UNIFIED_MANIFEST = PROJECT_ROOT / "configs" / "input_scales_20.json"
REPORT_PATH = PROJECT_ROOT / "input_scale_expansion_report.md"
VALIDATION_JSON = PROJECT_ROOT / "input_scale_validation.json"

SIZE_IDS = tuple(f"SIZE_{index:02d}" for index in range(1, 21))
LEGACY_PROFILES = (
    "MINI_DATASET",
    "SMALL_DATASET",
    "MEDIUM_DATASET",
    "LARGE_DATASET",
    "EXTRALARGE_DATASET",
)
ANCHOR_POSITIONS = (0, 4, 9, 14, 19)
ANCHOR_BY_INDEX = dict(zip(ANCHOR_POSITIONS, LEGACY_PROFILES))
DEFAULT_SEED = 20261020

SELECTOR_BEGIN = "/* LLVM_V01_SIZE_SELECTOR_BEGIN */"
SELECTOR_END = "/* LLVM_V01_SIZE_SELECTOR_END */"
DEFINITIONS_BEGIN = "/* LLVM_V01_SIZE_DEFINITIONS_BEGIN */"
DEFINITIONS_END = "/* LLVM_V01_SIZE_DEFINITIONS_END */"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_interpolate(start: int, end: int, numerator: int, denominator: int) -> int:
    delta = end - start
    if delta >= 0:
        return start + (delta * numerator + denominator // 2) // denominator
    return start - ((-delta) * numerator + denominator // 2) // denominator


def piecewise_values(points: list[tuple[int, int]], *, multiple: int = 1) -> list[int]:
    """Return 20 positive integers through exact anchor points.

    Positive segments use geometric interpolation, which gives useful coverage
    across several orders of magnitude. Exact supplied anchors are never rounded.
    """

    if points[0][0] != 0 or points[-1][0] != 19:
        raise ValueError(f"piecewise points must span indices 0..19: {points}")
    result = [0] * 20
    for (left_i, left_v), (right_i, right_v) in zip(points, points[1:]):
        if left_v <= 0 or right_v <= 0 or right_i <= left_i:
            raise ValueError(f"invalid piecewise segment: {points}")
        span = right_i - left_i
        for index in range(left_i, right_i + 1):
            if index == left_i:
                value = left_v
            elif index == right_i:
                value = right_v
            else:
                ratio = (index - left_i) / span
                value = int(round(math.exp(math.log(left_v) + ratio * (math.log(right_v) - math.log(left_v)))))
                value = max(multiple, int(round(value / multiple)) * multiple)
            result[index] = value

    for index in range(1, 20):
        if result[index] <= result[index - 1]:
            result[index] = result[index - 1] + multiple

    for index, value in points:
        result[index] = value
    if any(value <= 0 for value in result):
        raise ValueError(f"non-positive generated sequence: {result}")
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise ValueError(f"non-increasing generated sequence: {result}")
    return result


def linear_values(points: list[tuple[int, int]], *, multiple: int = 1) -> list[int]:
    if points[0][0] != 0 or points[-1][0] != 19:
        raise ValueError(f"linear points must span indices 0..19: {points}")
    result = [0] * 20
    for (left_i, left_v), (right_i, right_v) in zip(points, points[1:]):
        span = right_i - left_i
        for index in range(left_i, right_i + 1):
            if index == left_i:
                value = left_v
            elif index == right_i:
                value = right_v
            else:
                value = rounded_interpolate(left_v, right_v, index - left_i, span)
                value = max(multiple, int(round(value / multiple)) * multiple)
            result[index] = value
    for index in range(1, 20):
        if result[index] <= result[index - 1]:
            result[index] = result[index - 1] + multiple
    for index, value in points:
        result[index] = value
    if any(current <= previous for previous, current in zip(result, result[1:])):
        raise ValueError(f"non-increasing generated sequence: {result}")
    return result


def linear_nondecreasing_values(points: list[tuple[int, int]]) -> list[int]:
    """Interpolate an integer control that may legitimately repeat across sizes."""

    if points[0][0] != 0 or points[-1][0] != 19:
        raise ValueError(f"linear points must span indices 0..19: {points}")
    result = [0] * 20
    for (left_i, left_v), (right_i, right_v) in zip(points, points[1:]):
        span = right_i - left_i
        for index in range(left_i, right_i + 1):
            result[index] = rounded_interpolate(left_v, right_v, index - left_i, span)
    if any(current < previous for previous, current in zip(result, result[1:])):
        raise ValueError(f"decreasing generated sequence: {result}")
    return result


def parse_polybench_legacy_profiles(header_text: str) -> dict[str, dict[str, int]]:
    profiles: dict[str, dict[str, int]] = {}
    for profile in LEGACY_PROFILES:
        match = re.search(
            rf"^\s*#\s*ifdef\s+{profile}\b(?P<body>.*?)^\s*#\s*endif\b",
            header_text,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not match:
            continue
        values: dict[str, int] = {}
        for define in re.finditer(
            r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+([+-]?\d+)[uUlL]*\s*$",
            match.group("body"),
            flags=re.MULTILINE,
        ):
            values[define.group(1)] = int(define.group(2))
        if values:
            profiles[profile] = values
    return profiles


def interpolate_polybench_profiles(
    profiles: dict[str, dict[str, int]],
) -> list[dict[str, int]]:
    missing = [profile for profile in LEGACY_PROFILES if profile not in profiles]
    if missing:
        raise ValueError(f"missing legacy profiles: {missing}")
    dimensions = list(profiles[LEGACY_PROFILES[0]])
    if any(set(profiles[name]) != set(dimensions) for name in LEGACY_PROFILES):
        raise ValueError("PolyBench legacy profiles do not share dimensions")

    values: list[dict[str, int]] = [dict() for _ in SIZE_IDS]
    for dimension in dimensions:
        anchors = [profiles[name][dimension] for name in LEGACY_PROFILES]
        for segment in range(4):
            left_index = ANCHOR_POSITIONS[segment]
            right_index = ANCHOR_POSITIONS[segment + 1]
            span = right_index - left_index
            for index in range(left_index, right_index + 1):
                values[index][dimension] = rounded_interpolate(
                    anchors[segment],
                    anchors[segment + 1],
                    index - left_index,
                    span,
                )
    if len({tuple(item.items()) for item in values}) != 20:
        raise ValueError(f"interpolation produced duplicate profiles: {values}")
    return values


def replace_marked_block(text: str, begin: str, end: str, replacement: str) -> str:
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), flags=re.DOTALL)
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text


def patch_polybench_header(path: Path, profiles: list[dict[str, int]]) -> None:
    text = path.read_text(encoding="utf-8")
    selector_lines = [SELECTOR_BEGIN]
    selector_lines.append("# if " + " || ".join(f"defined({size_id})" for size_id in SIZE_IDS))
    selector_lines.append("#  define POLYBENCH_20_SIZE_SELECTED")
    selector_lines.append("# endif")
    selector_lines.append(SELECTOR_END)
    selector_block = "\n".join(selector_lines)

    definition_lines = [DEFINITIONS_BEGIN, "/* Generated by scripts/expand_input_scales.py. */"]
    for size_id, values in zip(SIZE_IDS, profiles):
        definition_lines.append(f"#  ifdef {size_id}")
        for name, value in values.items():
            definition_lines.append(f"#   define {name} {value}")
        definition_lines.append("#  endif")
        definition_lines.append("")
    definition_lines.append(DEFINITIONS_END)
    definitions_block = "\n".join(definition_lines)

    if SELECTOR_BEGIN in text:
        text = replace_marked_block(text, SELECTOR_BEGIN, SELECTOR_END, selector_block)
    else:
        marker = "/* Default to LARGE_DATASET. */"
        if marker not in text:
            raise ValueError(f"default profile marker not found in {path}")
        text = text.replace(marker, selector_block + "\n\n" + marker, 1)

    default_pattern = re.compile(
        r"(#\s*if\s+!defined\(MINI_DATASET\).*?!defined\(EXTRALARGE_DATASET\))"
    )
    default_match = default_pattern.search(text)
    if not default_match:
        raise ValueError(f"default profile condition not found in {path}")
    condition = default_match.group(1)
    if "POLYBENCH_20_SIZE_SELECTED" not in condition:
        text = text[: default_match.start(1)] + condition + " && !defined(POLYBENCH_20_SIZE_SELECTED)" + text[default_match.end(1) :]

    if DEFINITIONS_BEGIN in text:
        text = replace_marked_block(text, DEFINITIONS_BEGIN, DEFINITIONS_END, definitions_block)
    else:
        closing = re.search(r"^\s*#\s*endif\s*/\*\s*!\(", text, flags=re.MULTILINE)
        if not closing:
            raise ValueError(f"dimension guard closing marker not found in {path}")
        text = text[: closing.start()] + definitions_block + "\n\n" + text[closing.start() :]

    path.write_text(text, encoding="utf-8")


def polybench_entries(*, apply_headers: bool) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, int]]]]:
    entries: list[dict[str, Any]] = []
    profile_map: dict[str, list[dict[str, int]]] = {}
    for header in sorted(POLYBENCH_ROOT.rglob("*.h")):
        text = header.read_text(encoding="utf-8", errors="ignore")
        legacy = parse_polybench_legacy_profiles(text)
        if not legacy:
            continue
        profiles = interpolate_polybench_profiles(legacy)
        program_id = header.stem
        profile_map[program_id] = profiles
        if apply_headers:
            patch_polybench_header(header, profiles)
        for index, (size_id, parameters) in enumerate(zip(SIZE_IDS, profiles)):
            legacy_name = ANCHOR_BY_INDEX.get(index)
            entries.append(
                {
                    "dataset": "PolyBench/C",
                    "execution_model": "cpu_serial",
                    "program_id": program_id,
                    "size_id": size_id,
                    "input_size_parameters": parameters,
                    "input_file_paths": [],
                    "generation_method": "piecewise_linear_compile_time_macros",
                    "compile_config": [f"-D{size_id}"],
                    "run_command": (
                        "python scripts/pipeline_new.py --source-dir datasets/PolyBenchC-4.2.1 "
                        f"--dataset PolyBenchC-4.2.1 --polybench-size {size_id}"
                    ),
                    "original_config": legacy_name is not None,
                    "original_reference": legacy_name or "",
                    "random_seed": None,
                    "workload_proxy": math.prod(parameters.values()),
                    "header_path": str(header.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                }
            )
    return entries, profile_map


def generated_relative(program: str, filename: str) -> str:
    return f"../../data/generated_20/{program}/{filename}"


def generated_absolute(program: str, filename: str) -> Path:
    return GENERATED_DATA_ROOT / program / filename


def input_spec(
    index: int,
    args: Iterable[Any],
    parameters: dict[str, Any],
    *,
    threads: int = 1,
    required_files: Iterable[str] = (),
    generation_method: str = "command_line",
    original_reference: str = "",
    seed: int | None = DEFAULT_SEED,
    workload_proxy: float,
) -> dict[str, Any]:
    exact_original = bool(original_reference) and not generation_method.startswith("generated_")
    return {
        "id": SIZE_IDS[index],
        "profile": SIZE_IDS[index],
        "args": [str(value) for value in args],
        "threads": threads,
        "parameters": parameters,
        "required_files": list(required_files),
        "generation_method": generation_method,
        "original_config": exact_original,
        "original_reference": original_reference if exact_original else "",
        "original_parameter_anchor": "" if exact_original else original_reference,
        "random_seed": seed,
        "workload_proxy": workload_proxy,
    }


def original_at(index: int, references: dict[int, str]) -> str:
    return references.get(index, "")


def common_scale_vectors() -> dict[str, Any]:
    return {
        "btree_records": piecewise_values([(0, 1024), (14, 1_000_000), (19, 2_000_000)]),
        "backprop_layers": piecewise_values([(0, 1024), (14, 65536), (19, 131072)], multiple=16),
        "bfs_nodes": piecewise_values([(0, 4096), (14, 1_000_000), (19, 2_000_000)]),
        "cfd_cells": piecewise_values([(0, 8192), (9, 97046), (14, 193000), (19, 262144)]),
        "heartwall_frames": piecewise_values([(0, 2), (9, 20), (19, 100)]),
        "hotspot_side": piecewise_values([(0, 64), (14, 1024), (19, 1536)], multiple=16),
        "hotspot3d_side": piecewise_values([(0, 64), (14, 512), (19, 640)], multiple=16),
        "hotspot3d_iterations": piecewise_values([(0, 10), (14, 100), (19, 140)]),
        "kmeans_points": piecewise_values([(0, 100), (14, 494020), (19, 600000)]),
        "lavamd_boxes": linear_values([(0, 2), (2, 4), (8, 10), (19, 21)]),
        "leukocyte_frames": linear_values([(0, 1), (4, 5), (19, 20)]),
        "myocyte_time": piecewise_values([(0, 10), (14, 100), (19, 200)]),
        "nn_records": piecewise_values([(0, 1024), (9, 42764), (19, 200000)]),
        "nw_dimension": piecewise_values([(0, 256), (14, 2048), (19, 4096)], multiple=16),
        "particle_side": linear_values([(0, 32), (6, 128), (14, 256), (19, 336)], multiple=16),
        "particle_frames": linear_nondecreasing_values([(0, 2), (14, 10), (19, 20)]),
        "particle_count": piecewise_values([(0, 100), (14, 1000), (19, 10000)]),
        "pathfinder_columns": piecewise_values([(0, 1000), (14, 100000), (19, 200000)]),
        "pathfinder_rows": piecewise_values([(0, 20), (14, 100), (19, 200)]),
        "srad1_rows": piecewise_values([(0, 128), (14, 502), (19, 768)]),
        "srad1_columns": piecewise_values([(0, 128), (14, 458), (19, 768)]),
        "srad1_iterations": piecewise_values([(0, 10), (14, 100), (19, 150)]),
        "srad2_side": piecewise_values([(0, 128), (4, 256), (14, 2048), (19, 3072)], multiple=16),
        "stream_points": piecewise_values([(0, 1000), (14, 65536), (19, 131072)]),
        "stream_dimensions": piecewise_values([(0, 16), (14, 256), (19, 320)], multiple=4),
        "dwt_side": piecewise_values([(0, 192), (14, 1024), (19, 1536)], multiple=16),
        "gaussian_side": piecewise_values([(0, 16), (14, 1024), (19, 1536)], multiple=16),
        "huffman_bytes": piecewise_values([(0, 65536), (14, 1048576), (19, 8388608)], multiple=1024),
        "lud_openmp": piecewise_values([(0, 256), (14, 2048), (19, 8000)], multiple=16),
        "lud_cuda": piecewise_values([(0, 64), (14, 2048), (19, 4096)], multiple=16),
    }


def build_openmp_inputs(program_id: str, vectors: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if program_id == "b+tree":
        for i, records in enumerate(vectors["btree_records"]):
            query_count = max(128, min(50000, records // 8))
            data = generated_relative("b+tree", f"keys_{SIZE_IDS[i]}.txt")
            commands = generated_relative("b+tree", f"commands_{SIZE_IDS[i]}.txt")
            params = {"records": records, "queries": query_count}
            result.append(input_spec(i, ["core", 4, "file", data, "command", commands], params, threads=4, required_files=[data, commands], generation_method="generated_btree_text", original_reference=original_at(i, {14: "mil.txt+command.txt"}), workload_proxy=records + query_count))
    elif program_id == "backprop":
        for i, width in enumerate(vectors["backprop_layers"]):
            result.append(input_spec(i, [width], {"input_layer_size": width}, threads=8, seed=None, workload_proxy=width, original_reference=original_at(i, {0: "layer-1024", 14: "layer-65536"})))
    elif program_id == "bfs":
        for i, nodes in enumerate(vectors["bfs_nodes"]):
            degree = 6
            path = generated_relative("bfs", f"graph_{SIZE_IDS[i]}.txt")
            result.append(input_spec(i, [4, path], {"nodes": nodes, "edges": nodes * degree, "average_degree": degree}, threads=4, required_files=[path], generation_method="generated_connected_graph", original_reference=original_at(i, {0: "graph4096.txt", 14: "graph1MW_6.txt"}), workload_proxy=nodes * (degree + 1)))
    elif program_id == "cfd":
        for i, cells in enumerate(vectors["cfd_cells"]):
            path = generated_relative("cfd", f"mesh_{SIZE_IDS[i]}.dat")
            result.append(input_spec(i, [path], {"mesh_cells": cells, "faces_per_cell": 4}, threads=4, required_files=[path], generation_method="generated_ring_cfd_mesh", workload_proxy=cells, original_reference=original_at(i, {9: "fvcorr.domn.097K", 14: "fvcorr.domn.193K"})))
    elif program_id == "heartwall":
        source = "../../data/heartwall/test.avi"
        for i, frames in enumerate(vectors["heartwall_frames"]):
            result.append(input_spec(i, [source, frames, 4], {"input_file": "test.avi", "frames": frames}, threads=4, required_files=[source], generation_method="reuse_video_vary_frames", seed=None, workload_proxy=frames, original_reference=original_at(i, {0: "video-2-frames", 9: "video-20-frames"})))
    elif program_id == "hotspot":
        for i, side in enumerate(vectors["hotspot_side"]):
            temp = generated_relative("hotspot", f"temp_{SIZE_IDS[i]}.txt")
            power = generated_relative("hotspot", f"power_{SIZE_IDS[i]}.txt")
            params = {"rows": side, "columns": side, "iterations": 2}
            result.append(input_spec(i, [side, side, 2, 4, temp, power, "{null_device}"], params, threads=4, required_files=[temp, power], generation_method="generated_thermal_fields", workload_proxy=side * side * 2, original_reference=original_at(i, {0: "grid-64", 14: "grid-1024"})))
    elif program_id == "hotspot3D":
        for i, (side, iterations) in enumerate(zip(vectors["hotspot3d_side"], vectors["hotspot3d_iterations"])):
            layers = 8
            power = generated_relative("hotspot3D", f"power_{SIZE_IDS[i]}.txt")
            temp = generated_relative("hotspot3D", f"temp_{SIZE_IDS[i]}.txt")
            params = {"grid_size": side, "layers": layers, "iterations": iterations}
            result.append(input_spec(i, [side, layers, iterations, power, temp, "{null_device}"], params, threads=4, required_files=[power, temp], generation_method="generated_3d_thermal_fields", workload_proxy=side * side * layers * iterations, original_reference=original_at(i, {0: "grid-64x8", 14: "grid-512x8"})))
    elif program_id == "kmeans":
        attributes = 16
        for i, points in enumerate(vectors["kmeans_points"]):
            path = generated_relative("kmeans", f"points_{SIZE_IDS[i]}.bin")
            params = {"points": points, "attributes": attributes, "clusters": 5}
            result.append(input_spec(i, ["-n", 4, "-b", "-k", 5, "-i", path], params, threads=4, required_files=[path], generation_method="generated_clustered_binary_points", workload_proxy=points * attributes, original_reference=original_at(i, {0: "points-100", 14: "kdd-cup"})))
    elif program_id == "lavaMD":
        for i, boxes in enumerate(vectors["lavamd_boxes"]):
            params = {"boxes1d": boxes, "boxes_total": boxes**3, "particles_per_box": 100}
            result.append(input_spec(i, ["-cores", 4, "-boxes1d", boxes], params, threads=4, seed=None, workload_proxy=boxes**3 * 100, original_reference=original_at(i, {2: "boxes-4", 8: "boxes-10"})))
    elif program_id == "leukocyte":
        source = "../../data/leukocyte/testfile.avi"
        for i, frames in enumerate(vectors["leukocyte_frames"]):
            result.append(input_spec(i, [frames, 4, source], {"input_file": "testfile.avi", "frames": frames}, threads=4, required_files=[source], generation_method="reuse_video_vary_frames", seed=None, workload_proxy=frames, original_reference=original_at(i, {0: "video-1-frame", 4: "video-5-frames"})))
    elif program_id == "lud":
        for i, side in enumerate(vectors["lud_openmp"]):
            result.append(input_spec(i, ["-n", 4, "-s", side], {"matrix_dimension": side}, threads=4, generation_method="program_internal_deterministic_matrix", seed=None, workload_proxy=side**3, original_reference=original_at(i, {0: "matrix-256", 19: "matrix-8000"})))
    elif program_id == "myocyte":
        for i, horizon in enumerate(vectors["myocyte_time"]):
            params = {"simulation_time_ms": horizon, "instances": 1, "mode": 0}
            result.append(input_spec(i, [horizon, 1, 0, 4], params, threads=4, seed=None, workload_proxy=horizon, original_reference=original_at(i, {0: "workload-10", 14: "workload-100"})))
    elif program_id == "nn":
        for i, records in enumerate(vectors["nn_records"]):
            filelist = generated_relative("nn", f"filelist_{SIZE_IDS[i]}.txt")
            params = {"database_records": records, "neighbors": 5, "latitude": 30, "longitude": 90}
            result.append(input_spec(i, [filelist, 5, 30, 90], params, threads=4, required_files=[filelist, generated_relative("nn", f"records_{SIZE_IDS[i]}.db")], generation_method="generated_fixed_record_database", workload_proxy=records, original_reference=original_at(i, {9: "cane4 databases"})))
    elif program_id == "nw":
        for i, dimension in enumerate(vectors["nw_dimension"]):
            result.append(input_spec(i, [dimension, 10, 4], {"sequence_dimension": dimension, "penalty": 10}, threads=4, seed=None, workload_proxy=dimension**2, original_reference=original_at(i, {0: "dimension-256", 14: "dimension-2048"})))
    elif program_id == "particlefilter":
        for i, (side, frames, particles) in enumerate(zip(vectors["particle_side"], vectors["particle_frames"], vectors["particle_count"])):
            params = {"width": side, "height": side, "frames": frames, "particles": particles}
            result.append(input_spec(i, ["-x", side, "-y", side, "-z", frames, "-np", particles], params, threads=4, workload_proxy=side * side * frames * particles))
    elif program_id == "pathfinder":
        for i, (columns, rows) in enumerate(zip(vectors["pathfinder_columns"], vectors["pathfinder_rows"])):
            params = {"columns": columns, "rows": rows}
            result.append(input_spec(i, [columns, rows], params, threads=4, seed=None, workload_proxy=columns * rows, original_reference=original_at(i, {14: "grid-100000x100"})))
    elif program_id == "srad_v1":
        for i, (rows, columns, iterations) in enumerate(zip(vectors["srad1_rows"], vectors["srad1_columns"], vectors["srad1_iterations"])):
            params = {"iterations": iterations, "lambda": 0.5, "rows": rows, "columns": columns}
            result.append(input_spec(i, [iterations, 0.5, rows, columns, 4], params, threads=4, workload_proxy=rows * columns * iterations, original_reference=original_at(i, {0: "image-128", 14: "image-502x458"})))
    elif program_id == "srad_v2":
        for i, side in enumerate(vectors["srad2_side"]):
            roi_end = min(127, side - 1)
            params = {"rows": side, "columns": side, "roi": f"0:{roi_end},0:{roi_end}", "iterations": 2, "lambda": 0.5}
            result.append(input_spec(i, [side, side, 0, roi_end, 0, roi_end, 2, 0.5, 4], params, threads=4, seed=None, workload_proxy=side * side * 2, original_reference=original_at(i, {0: "image-128", 14: "image-2048"})))
    elif program_id == "streamcluster":
        for i, (points, dimensions) in enumerate(zip(vectors["stream_points"], vectors["stream_dimensions"])):
            kmin = max(3, int(round(3 + 9 * i / 19)))
            kmax = max(kmin + 2, int(round(5 + 19 * i / 19)))
            chunk = min(points, 65536)
            center_size = max(100, int(round(100 + 1900 * i / 19)))
            params = {"kmin": kmin, "kmax": kmax, "dimensions": dimensions, "points": points, "chunk_size": chunk, "center_size": center_size}
            result.append(input_spec(i, [kmin, kmax, dimensions, points, chunk, center_size, "none", "{null_device}", 4], params, threads=4, seed=None, workload_proxy=points * dimensions, original_reference=original_at(i, {14: "synthetic-65536"})))
    else:
        raise KeyError(f"no OpenMP scale builder for {program_id}")
    return result


def build_cuda_inputs(program_id: str, vectors: dict[str, Any]) -> list[dict[str, Any]]:
    if program_id in {"b+tree", "backprop", "bfs", "cfd", "heartwall", "hotspot", "hotspot3D", "lavaMD", "leukocyte", "lud", "myocyte", "nn", "nw", "pathfinder", "srad_v1", "srad_v2", "streamcluster"}:
        pass
    result: list[dict[str, Any]] = []
    if program_id == "b+tree":
        for i, records in enumerate(vectors["btree_records"]):
            queries = max(128, min(50000, records // 8))
            data = generated_relative("b+tree", f"keys_{SIZE_IDS[i]}.txt")
            commands = generated_relative("b+tree", f"commands_{SIZE_IDS[i]}.txt")
            result.append(input_spec(i, ["file", data, "command", commands], {"records": records, "queries": queries, "work_group_size": 256}, required_files=[data, commands], generation_method="generated_btree_text", workload_proxy=records + queries, original_reference=original_at(i, {14: "mil.txt+command.txt"})))
    elif program_id == "backprop":
        for i, width in enumerate(vectors["backprop_layers"]):
            result.append(input_spec(i, [width], {"input_layer_size": width}, seed=None, workload_proxy=width, original_reference=original_at(i, {0: "layer-1024", 14: "layer-65536"})))
    elif program_id == "bfs":
        for i, nodes in enumerate(vectors["bfs_nodes"]):
            path = generated_relative("bfs", f"graph_{SIZE_IDS[i]}.txt")
            result.append(input_spec(i, [path], {"nodes": nodes, "edges": nodes * 6, "average_degree": 6}, required_files=[path], generation_method="generated_connected_graph", workload_proxy=nodes * 7, original_reference=original_at(i, {0: "graph4096.txt", 14: "graph1MW_6.txt"})))
    elif program_id == "cfd":
        for i, cells in enumerate(vectors["cfd_cells"]):
            path = generated_relative("cfd", f"mesh_{SIZE_IDS[i]}.dat")
            result.append(input_spec(i, [path], {"mesh_cells": cells, "faces_per_cell": 4}, required_files=[path], generation_method="generated_ring_cfd_mesh", workload_proxy=cells, original_reference=original_at(i, {9: "fvcorr.domn.097K", 14: "fvcorr.domn.193K"})))
    elif program_id == "dwt2d":
        for i, side in enumerate(vectors["dwt_side"]):
            filename = f"image_{SIZE_IDS[i]}.raw"
            required = generated_relative("dwt2d", filename)
            program_arg = f"../generated_20/dwt2d/{filename}"
            params = {"width": side, "height": side, "components": 3, "levels": 3, "wavelet": "5/3"}
            result.append(input_spec(i, ["-d", f"{side}x{side}", "-c", 3, "-f", "-5", "-l", 3, program_arg], params, required_files=[required], generation_method="generated_raw_rgb", workload_proxy=side * side * 3))
    elif program_id == "gaussian":
        for i, side in enumerate(vectors["gaussian_side"]):
            result.append(input_spec(i, ["-s", side, "-q"], {"matrix_size": side}, generation_method="program_internal_deterministic_matrix", seed=None, workload_proxy=side**3))
    elif program_id == "heartwall":
        source = "../../data/heartwall/test.avi"
        for i, frames in enumerate(vectors["heartwall_frames"]):
            result.append(input_spec(i, [source, frames], {"input_file": "test.avi", "frames": frames}, required_files=[source], generation_method="reuse_video_vary_frames", seed=None, workload_proxy=frames, original_reference=original_at(i, {0: "test-2-frames", 9: "test-20-frames"})))
    elif program_id == "hotspot":
        for i, side in enumerate(vectors["hotspot_side"]):
            temp = generated_relative("hotspot", f"temp_{SIZE_IDS[i]}.txt")
            power = generated_relative("hotspot", f"power_{SIZE_IDS[i]}.txt")
            params = {"grid_size": side, "pyramid_height": 2, "iterations": 2}
            result.append(input_spec(i, [side, 2, 2, temp, power, "output.out"], params, required_files=[temp, power], generation_method="generated_thermal_fields", workload_proxy=side * side * 2, original_reference=original_at(i, {0: "grid-64", 14: "grid-512/1024-range"})))
    elif program_id == "hotspot3D":
        for i, (side, iterations) in enumerate(zip(vectors["hotspot3d_side"], vectors["hotspot3d_iterations"])):
            layers = 8
            power = generated_relative("hotspot3D", f"power_{SIZE_IDS[i]}.txt")
            temp = generated_relative("hotspot3D", f"temp_{SIZE_IDS[i]}.txt")
            params = {"grid_size": side, "layers": layers, "iterations": iterations}
            result.append(input_spec(i, [side, layers, iterations, power, temp, "output.out"], params, required_files=[power, temp], generation_method="generated_3d_thermal_fields", workload_proxy=side * side * layers * iterations, original_reference=original_at(i, {0: "grid-64x8", 14: "grid-512x8"})))
    elif program_id == "huffman":
        for i, byte_count in enumerate(vectors["huffman_bytes"]):
            path = generated_relative("huffman", f"input_{SIZE_IDS[i]}.bin")
            result.append(input_spec(i, [path], {"input_bytes": byte_count, "block_threads": 256}, required_files=[path], generation_method="generated_repeated_reference_corpus", seed=None, workload_proxy=byte_count, original_reference=original_at(i, {14: "test1024_H2.206587175259.in"})))
    elif program_id == "lavaMD":
        for i, boxes in enumerate(vectors["lavamd_boxes"]):
            params = {"boxes1d": boxes, "boxes_total": boxes**3, "particles_per_box": 100}
            result.append(input_spec(i, ["-boxes1d", boxes], params, seed=None, workload_proxy=boxes**3 * 100, original_reference=original_at(i, {0: "boxes-2", 8: "boxes-10"})))
    elif program_id == "leukocyte":
        source = "../../../data/leukocyte/testfile.avi"
        for i, frames in enumerate(vectors["leukocyte_frames"]):
            result.append(input_spec(i, [source, frames], {"input_file": "testfile.avi", "frames": frames}, required_files=[source], generation_method="reuse_video_vary_frames", seed=None, workload_proxy=frames, original_reference=original_at(i, {4: "test-5-frames"})))
    elif program_id == "lud":
        for i, side in enumerate(vectors["lud_cuda"]):
            result.append(input_spec(i, ["-s", side], {"matrix_dimension": side}, generation_method="program_internal_deterministic_matrix", seed=None, workload_proxy=side**3))
    elif program_id == "myocyte":
        for i, horizon in enumerate(vectors["myocyte_time"]):
            result.append(input_spec(i, [horizon, 1, 0], {"simulation_time_ms": horizon, "instances": 1, "mode": 0}, seed=None, workload_proxy=horizon, original_reference=original_at(i, {0: "time-10-one-instance", 14: "time-100-one-instance"})))
    elif program_id == "nn":
        for i, records in enumerate(vectors["nn_records"]):
            filelist = generated_relative("nn", f"filelist_{SIZE_IDS[i]}.txt")
            params = {"database_records": records, "result_count": 5, "latitude": 30, "longitude": 90}
            result.append(input_spec(i, [filelist, "-r", 5, "-lat", 30, "-lng", 90], params, required_files=[filelist, generated_relative("nn", f"records_{SIZE_IDS[i]}.db")], generation_method="generated_fixed_record_database", workload_proxy=records, original_reference=original_at(i, {9: "cane4 databases"})))
    elif program_id == "nw":
        for i, dimension in enumerate(vectors["nw_dimension"]):
            result.append(input_spec(i, [dimension, 10], {"sequence_dimension": dimension, "penalty": 10, "work_group_size": 16}, seed=None, workload_proxy=dimension**2, original_reference=original_at(i, {0: "dimension-256", 14: "dimension-2048"})))
    elif program_id in {"particlefilter_naive", "particlefilter_float"}:
        for i, (side, frames, particles) in enumerate(zip(vectors["particle_side"], vectors["particle_frames"], vectors["particle_count"])):
            params = {"width": side, "height": side, "frames": frames, "particles": particles}
            result.append(input_spec(i, ["-x", side, "-y", side, "-z", frames, "-np", particles], params, workload_proxy=side * side * frames * particles, original_reference=original_at(i, {0: "video-32x32x2-p100"})))
    elif program_id == "pathfinder":
        for i, (columns, rows) in enumerate(zip(vectors["pathfinder_columns"], vectors["pathfinder_rows"])):
            params = {"columns": columns, "rows": rows, "pyramid_height": 20}
            result.append(input_spec(i, [columns, rows, 20], params, seed=None, workload_proxy=columns * rows, original_reference=original_at(i, {14: "grid-100000x100"})))
    elif program_id == "srad_v1":
        for i, (rows, columns, iterations) in enumerate(zip(vectors["srad1_rows"], vectors["srad1_columns"], vectors["srad1_iterations"])):
            params = {"iterations": iterations, "lambda": 0.5, "rows": rows, "columns": columns}
            result.append(input_spec(i, [iterations, 0.5, rows, columns], params, workload_proxy=rows * columns * iterations, original_reference=original_at(i, {14: "image-502x458-100"})))
    elif program_id == "srad_v2":
        for i, side in enumerate(vectors["srad2_side"]):
            roi_end = min(127, side - 1)
            params = {"rows": side, "columns": side, "roi": f"0:{roi_end},0:{roi_end}", "lambda": 0.5, "iterations": 2}
            result.append(input_spec(i, [side, side, 0, roi_end, 0, roi_end, 0.5, 2], params, seed=None, workload_proxy=side * side * 2, original_reference=original_at(i, {4: "matrix-256", 14: "matrix-2048"})))
    elif program_id == "streamcluster":
        for i, (points, dimensions) in enumerate(zip(vectors["stream_points"], vectors["stream_dimensions"])):
            kmin = max(3, int(round(3 + 9 * i / 19)))
            kmax = max(kmin + 2, int(round(5 + 19 * i / 19)))
            chunk = min(points, 65536)
            center_size = max(100, int(round(100 + 1900 * i / 19)))
            params = {"kmin": kmin, "kmax": kmax, "dimensions": dimensions, "points": points, "chunk_size": chunk, "center_size": center_size}
            result.append(input_spec(i, [kmin, kmax, dimensions, points, chunk, center_size, "none", "{null_device}", 1], params, seed=None, workload_proxy=points * dimensions, original_reference=original_at(i, {14: "synthetic-65536"})))
    else:
        raise KeyError(f"no CUDA scale builder for {program_id}")
    return result


def build_rodinia_manifest(source_path: Path, model: str, vectors: dict[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(load_json(source_path))
    manifest["schema_version"] = 2
    manifest["input_scale_scheme"] = "SIZE_01..SIZE_20"
    manifest["generated_by"] = "scripts/expand_input_scales.py"
    for benchmark in manifest["benchmarks"]:
        if benchmark.get("enabled", True) is False:
            continue
        if model == "openmp":
            benchmark["inputs"] = build_openmp_inputs(benchmark["id"], vectors)
        else:
            benchmark["inputs"] = build_cuda_inputs(benchmark["id"], vectors)
    return manifest


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        batch: list[str] = []
        for line in lines:
            batch.append(line)
            if len(batch) >= 8192:
                stream.write("".join(batch))
                batch.clear()
        if batch:
            stream.write("".join(batch))


def generate_btree(vectors: dict[str, Any]) -> None:
    for i, records in enumerate(vectors["btree_records"]):
        keys = generated_absolute("b+tree", f"keys_{SIZE_IDS[i]}.txt")
        commands = generated_absolute("b+tree", f"commands_{SIZE_IDS[i]}.txt")
        write_lines(keys, (f"{records}\n" if value == -1 else f"{value}\n" for value in range(-1, records)))
        query_count = max(128, min(50000, records // 8))
        commands.parent.mkdir(parents=True, exist_ok=True)
        commands.write_text(
            f"j {records // 2} {max(1, records // 4)}\nk {query_count}\n\n",
            encoding="ascii",
        )


def generate_bfs(vectors: dict[str, Any], seed: int) -> None:
    degree = 6
    for i, nodes in enumerate(vectors["bfs_nodes"]):
        path = generated_absolute("bfs", f"graph_{SIZE_IDS[i]}.txt")
        def lines() -> Iterable[str]:
            yield f"{nodes}\n"
            for node in range(nodes):
                yield f"{node * degree} {degree}\n"
            yield "0\n"
            yield f"{nodes * degree}\n"
            offsets = (1, -1, 2, -2, 17, -17)
            for node in range(nodes):
                for edge_index, offset in enumerate(offsets):
                    destination = (node + offset) % nodes
                    cost = 1 + ((node * 17 + edge_index * 13 + seed) % 10)
                    yield f"{destination} {cost}\n"
        write_lines(path, lines())


def generate_cfd(vectors: dict[str, Any]) -> None:
    normals = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
    for i, cells in enumerate(vectors["cfd_cells"]):
        path = generated_absolute("cfd", f"mesh_{SIZE_IDS[i]}.dat")
        def lines() -> Iterable[str]:
            yield f"{cells}\n"
            for cell in range(cells):
                previous = cells if cell == 0 else cell
                following = 1 if cell == cells - 1 else cell + 2
                neighbours = (previous, following, -1, -1)
                fields = ["1.0"]
                for neighbour, normal in zip(neighbours, normals):
                    fields.extend((str(neighbour), *(f"{value:.1f}" for value in normal)))
                yield " ".join(fields) + "\n"
        write_lines(path, lines())


def field_value(index: int, seed: int, base: float, spread: float) -> float:
    mixed = (index * 1103515245 + seed * 12345 + 1013904223) & 0xFFFFFFFF
    return base + spread * (mixed % 10000) / 10000.0


def generate_hotspot(vectors: dict[str, Any], seed: int) -> None:
    for i, side in enumerate(vectors["hotspot_side"]):
        count = side * side
        temp = generated_absolute("hotspot", f"temp_{SIZE_IDS[i]}.txt")
        power = generated_absolute("hotspot", f"power_{SIZE_IDS[i]}.txt")
        write_lines(temp, (f"{field_value(index, seed + i, 320.0, 25.0):.6f}\n" for index in range(count)))
        write_lines(power, (f"{field_value(index, seed + 100 + i, 0.1, 2.0):.6f}\n" for index in range(count)))


def generate_hotspot3d(vectors: dict[str, Any], seed: int) -> None:
    layers = 8
    for i, side in enumerate(vectors["hotspot3d_side"]):
        count = side * side * layers
        temp = generated_absolute("hotspot3D", f"temp_{SIZE_IDS[i]}.txt")
        power = generated_absolute("hotspot3D", f"power_{SIZE_IDS[i]}.txt")
        write_lines(temp, (f"{field_value(index, seed + 200 + i, 320.0, 25.0):.6f}\n" for index in range(count)))
        write_lines(power, (f"{field_value(index, seed + 300 + i, 0.1, 2.0):.6f}\n" for index in range(count)))


def generate_kmeans(vectors: dict[str, Any], seed: int) -> None:
    attributes = 16
    for i, points in enumerate(vectors["kmeans_points"]):
        path = generated_absolute("kmeans", f"points_{SIZE_IDS[i]}.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(struct.pack("<ii", points, attributes))
            batch: list[float] = []
            for point in range(points):
                cluster = point % 5
                for attribute in range(attributes):
                    jitter = ((point * 17 + attribute * 31 + seed + i) % 1000) / 1000.0
                    batch.append(cluster * 10.0 + attribute * 0.1 + jitter)
                if len(batch) >= 16384:
                    stream.write(struct.pack(f"<{len(batch)}f", *batch))
                    batch.clear()
            if batch:
                stream.write(struct.pack(f"<{len(batch)}f", *batch))


def make_nn_record(index: int, seed: int) -> bytes:
    record = bytearray(b" " * 48)
    prefix = f"{1950 + index % 75:04d} {1 + index % 12:2d} {1 + index % 28:2d} {index % 24:2d} {index % 40:2d} STORM"
    record[: min(27, len(prefix))] = prefix[:27].encode("ascii")
    latitude = -89.0 + ((index * 37 + seed) % 1780) / 10.0
    longitude = ((index * 53 + seed) % 3600) / 10.0
    record[28:33] = f"{latitude:5.1f}".encode("ascii")
    record[33:38] = f"{longitude:5.1f}".encode("ascii")
    record[39:43] = f"{index % 9999:4d}".encode("ascii")
    record[44:48] = f"{(index * 7) % 9999:4d}".encode("ascii")
    return bytes(record) + b"\n"


def generate_nn(vectors: dict[str, Any], seed: int) -> None:
    for i, records in enumerate(vectors["nn_records"]):
        database = generated_absolute("nn", f"records_{SIZE_IDS[i]}.db")
        filelist = generated_absolute("nn", f"filelist_{SIZE_IDS[i]}.txt")
        database.parent.mkdir(parents=True, exist_ok=True)
        with database.open("wb") as stream:
            for record_index in range(records):
                stream.write(make_nn_record(record_index, seed + i))
        filelist.write_text(
            f"../../data/generated_20/nn/{database.name}\n",
            encoding="ascii",
        )


def generate_dwt(vectors: dict[str, Any], seed: int) -> None:
    for i, side in enumerate(vectors["dwt_side"]):
        path = generated_absolute("dwt2d", f"image_{SIZE_IDS[i]}.raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        remaining = side * side * 3
        offset = 0
        with path.open("wb") as stream:
            while remaining:
                count = min(1024 * 1024, remaining)
                stream.write(bytes((((offset + index) * 37 + seed + i * 11) & 0xFF) for index in range(count)))
                offset += count
                remaining -= count


def generate_huffman(vectors: dict[str, Any]) -> None:
    reference = HUFFMAN_REFERENCE.read_bytes()
    if not reference:
        raise ValueError(f"empty Huffman reference corpus: {HUFFMAN_REFERENCE}")
    for i, byte_count in enumerate(vectors["huffman_bytes"]):
        path = generated_absolute("huffman", f"input_{SIZE_IDS[i]}.bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        remaining = byte_count
        offset = 0
        with path.open("wb") as stream:
            while remaining:
                count = min(len(reference) - offset, remaining)
                stream.write(reference[offset : offset + count])
                remaining -= count
                offset = (offset + count) % len(reference)


def materialize_generated_data(vectors: dict[str, Any], seed: int) -> None:
    generators = (
        generate_btree,
        lambda value: generate_bfs(value, seed),
        generate_cfd,
        lambda value: generate_hotspot(value, seed),
        lambda value: generate_hotspot3d(value, seed),
        lambda value: generate_kmeans(value, seed),
        lambda value: generate_nn(value, seed),
        lambda value: generate_dwt(value, seed),
        generate_huffman,
    )
    for generator in generators:
        generator(vectors)


def manifest_entries_from_rodinia(manifest: dict[str, Any], dataset: str, script: str, config_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    model = "openmp" if "OpenMP" in dataset else "cuda"
    for benchmark in manifest["benchmarks"]:
        if benchmark.get("enabled", True) is False:
            continue
        for item in benchmark["inputs"]:
            size_id = item["id"]
            entries.append(
                {
                    "dataset": dataset,
                    "execution_model": model,
                    "program_id": benchmark["id"],
                    "size_id": size_id,
                    "input_size_parameters": item["parameters"],
                    "input_file_paths": item.get("required_files", []),
                    "generation_method": item["generation_method"],
                    "compile_config": benchmark.get("build", []),
                    "run_command": (
                        f"python {script} --manifest {config_path.relative_to(PROJECT_ROOT).as_posix()} "
                        f"--benchmark {benchmark['id']} --input-profile {size_id}"
                    ),
                    "program_arguments": item["args"],
                    "original_config": item["original_config"],
                    "original_reference": item["original_reference"],
                    "original_parameter_anchor": item.get("original_parameter_anchor", ""),
                    "random_seed": item["random_seed"],
                    "workload_proxy": item["workload_proxy"],
                    "thread_count": item.get("threads") if model == "openmp" else None,
                }
            )
    return entries


def attach_generated_file_metadata(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        metadata: list[dict[str, Any]] = []
        for raw_path in entry.get("input_file_paths", []):
            normalized = raw_path.replace("\\", "/")
            marker = "data/generated_20/"
            if marker not in normalized:
                continue
            relative = normalized.split(marker, 1)[1]
            path = GENERATED_DATA_ROOT / relative
            if path.exists():
                metadata.append(
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        entry["generated_files"] = metadata


def render_implementation_report(entries: list[dict[str, Any]], materialized: bool) -> None:
    prefix = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# Input scale expansion report\n"
    prefix = prefix.split("\n## 4. Generated 20-level configurations", 1)[0].rstrip()
    lines = [prefix, "", "## 4. Generated 20-level configurations", ""]
    lines.append(f"- Unified manifest: `configs/input_scales_20.json`")
    lines.append(f"- OpenMP manifest: `configs/rodinia_openmp_20sizes.json`")
    lines.append(f"- CUDA manifest: `configs/rodinia_cuda_20sizes.json`")
    lines.append(f"- Rodinia generated data materialized in this run: `{str(materialized).lower()}`")
    lines.append(f"- Reproducibility seed: `{DEFAULT_SEED}`")
    lines.append("")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((entry["dataset"], entry["program_id"]), []).append(entry)
    for (dataset, program_id), rows in sorted(grouped.items()):
        lines.append(f"### {dataset} / {program_id}")
        lines.append("")
        lines.append("| size_id | parameters | origin | workload_proxy | input files |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for row in sorted(rows, key=lambda item: item["size_id"]):
            parameters = json.dumps(row["input_size_parameters"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if row["original_config"]:
                origin = row["original_reference"]
            elif row.get("original_parameter_anchor"):
                origin = f"new; parameter anchor={row['original_parameter_anchor']}"
            else:
                origin = "new"
            files = "<br>".join(row.get("input_file_paths", [])) or "-"
            lines.append(f"| {row['size_id']} | `{parameters}` | {origin} | {row['workload_proxy']} | {files} |")
        lines.append("")

    lines.extend(["## 5. Validation status", ""])
    if VALIDATION_JSON.exists():
        validation = load_json(VALIDATION_JSON)
        summary = validation.get("summary", {})
        spot_checks = validation.get("polybench_spot_checks", [])
        compile_passes = sum(row.get("compile_status") == "pass" for row in spot_checks)
        run_passes = sum(row.get("run_status") == "pass" for row in spot_checks)
        run_unavailable = sum(
            row.get("run_status") == "not_run_environment_unavailable"
            for row in spot_checks
        )
        lines.extend(
            [
                "The latest completed validation is recorded in `input_scale_validation.json` and `input_scale_validation_report.md`.",
                "",
                f"- Structural/data-format status: `{validation.get('status', 'unknown')}` with {len(validation.get('errors', []))} errors.",
                f"- Unified configurations: {summary.get('entry_count', 'unknown')} across {summary.get('program_count', 'unknown')} enabled programs.",
                f"- Generated Rodinia files: {summary.get('generated_file_count', 'unknown')} files, {summary.get('generated_bytes', 'unknown')} bytes.",
                f"- Effective-parameter duplicate groups: {summary.get('duplicate_parameter_groups', 'unknown')}.",
                f"- Known monotonic exceptions: `{', '.join(summary.get('monotonic_exceptions', [])) or 'none'}`.",
                f"- PolyBench spot compilation: {compile_passes}/{len(spot_checks)} passed; executable runs passed={run_passes}, environment-unavailable={run_unavailable}.",
                f"- Rodinia executable validation: `{validation.get('rodinia_runtime', {}).get('status', 'unknown')}`; see the validation report before formal collection.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Run `python scripts/validate_input_scales.py` for structural and data-format checks. Add `--polybench-compile` to compile the PolyBench small/middle/large selections on this Windows host. Rodinia executable checks require a Linux/OpenMP and CUDA host; commands are emitted by the validator and are not reported as passed until run there.",
                "",
            ]
        )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true", help="Generate Rodinia input files, including large files.")
    parser.add_argument(
        "--materialize-huffman-only",
        action="store_true",
        help="Regenerate only the 20 deterministic Huffman inputs.",
    )
    parser.add_argument("--no-polybench-header-update", action="store_true", help="Do not update PolyBench headers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.materialize and args.materialize_huffman_only:
        parser.error("use either --materialize or --materialize-huffman-only")

    vectors = common_scale_vectors()
    polybench_rows, _ = polybench_entries(apply_headers=not args.no_polybench_header_update)
    openmp_manifest = build_rodinia_manifest(OPENMP_SOURCE_MANIFEST, "openmp", vectors)
    cuda_manifest = build_rodinia_manifest(CUDA_SOURCE_MANIFEST, "cuda", vectors)
    write_json(OPENMP_OUTPUT_MANIFEST, openmp_manifest)
    write_json(CUDA_OUTPUT_MANIFEST, cuda_manifest)

    if args.materialize:
        materialize_generated_data(vectors, args.seed)
    elif args.materialize_huffman_only:
        generate_huffman(vectors)

    rows = [
        *polybench_rows,
        *manifest_entries_from_rodinia(openmp_manifest, "Rodinia OpenMP", "scripts/pipeline_rodinia.py", OPENMP_OUTPUT_MANIFEST),
        *manifest_entries_from_rodinia(cuda_manifest, "Rodinia CUDA", "scripts/pipeline_rodinia_cuda.py", CUDA_OUTPUT_MANIFEST),
    ]
    attach_generated_file_metadata(rows)
    write_json(
        UNIFIED_MANIFEST,
        {
            "schema_version": 1,
            "size_ids": list(SIZE_IDS),
            "random_seed": args.seed,
            "generated_by": "scripts/expand_input_scales.py",
            "entries": rows,
        },
    )
    render_implementation_report(rows, GENERATED_DATA_ROOT.exists())

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    print("Generated manifests:")
    for dataset, count in counts.items():
        print(f"  {dataset}: {count} configurations")
    print(f"  unified: {UNIFIED_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
