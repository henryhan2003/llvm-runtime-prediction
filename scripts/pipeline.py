from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "datasets" / "test"
AST_DIR = PROJECT_ROOT / "ast"
IR_DIR = PROJECT_ROOT / "llvm_ir"
CFG_DIR = PROJECT_ROOT / "cfg"
RESULTS_DIR = PROJECT_ROOT / "results"
BUILD_DIR = PROJECT_ROOT / "build" / "pipeline"
COMPAT_INCLUDE_DIR = PROJECT_ROOT / "scripts" / "compat_include"

SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx"}


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
    "input_size_n",
    "matrix_m",
    "matrix_n",
    "matrix_k",
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

    @property
    def relative_path(self) -> Path:
        return self.source_path.relative_to(self.source_root)

    @property
    def program_id(self) -> str:
        return self.relative_path.as_posix()

    @property
    def output_stem(self) -> Path:
        return self.relative_path.with_suffix("")

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
    try:
        result = run_command(cmd, timeout=10)
    except Exception as exc:
        return f"unavailable: {exc}"
    text = (result.stdout or result.stderr or "").strip()
    return text.splitlines()[0] if text else "unavailable"


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
    polybench_runtime = source_root / "utilities" / "polybench.c"
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
        dot_files = sorted(program_cfg_dir.glob(".*.dot")) + sorted(program_cfg_dir.glob("*.dot"))
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


def source_owned_ast_text(ast_text: str, source_path: Path) -> str:
    lines = ast_text.splitlines()
    marker = source_path.name
    starts = [index for index, line in enumerate(lines) if marker in line]
    if not starts:
        return ast_text
    return "\n".join(lines[min(starts) :])


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


def extract_ast_features(ast_path: Path, source_path: Path) -> dict[str, float | int]:
    raw_text = ast_path.read_text(encoding="utf-8", errors="ignore") if ast_path.exists() else ""
    text = source_owned_ast_text(raw_text, source_path)
    source_text = source_path.read_text(encoding="utf-8", errors="ignore")
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


def parse_dot_edges(text: str) -> tuple[set[str], set[tuple[str, str]]]:
    nodes = set(re.findall(r'Node(0x[0-9A-Fa-f]+|\d+)\s*\[', text))
    edges = set(re.findall(r'Node(0x[0-9A-Fa-f]+|\d+)\s*->\s*Node(0x[0-9A-Fa-f]+|\d+)', text))
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


def graph_features(nodes: set, edges: set[tuple]) -> dict[str, float | int]:
    node_count = len(nodes)
    edge_count = len(edges)
    out_degree = {node: 0 for node in nodes}
    in_degree = {node: 0 for node in nodes}
    for src, dst in edges:
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1
    branch_blocks = sum(1 for value in out_degree.values() if value > 1)
    merge_blocks = sum(1 for value in in_degree.values() if value > 1)
    backedges = 0
    for src, dst in edges:
        try:
            backedges += 1 if int(str(dst), 16 if str(dst).startswith("0x") else 10) <= int(str(src), 16 if str(src).startswith("0x") else 10) else 0
        except ValueError:
            pass
    depth = approximate_depth(nodes, edges)
    return {
        "basic_block_count": node_count,
        "cfg_edge_count": edge_count,
        "cyclomatic_complexity": max(edge_count - node_count + 2, 0) if node_count else 0,
        "branch_block_count": branch_blocks,
        "merge_block_count": merge_blocks,
        "avg_in_degree": statistics.mean(in_degree.values()) if in_degree else 0,
        "avg_out_degree": statistics.mean(out_degree.values()) if out_degree else 0,
        "max_out_degree": max(out_degree.values()) if out_degree else 0,
        "loop_backedge_count": backedges,
        "natural_loop_count": backedges,
        "max_cfg_loop_depth": 1 if backedges else 0,
        "scc_count": 0,
        "cfg_depth": depth,
        "unreachable_block_count": 0,
        "critical_edge_count": sum(1 for src, dst in edges if out_degree.get(src, 0) > 1 and in_degree.get(dst, 0) > 1),
    }


def approximate_depth(nodes: set, edges: set[tuple]) -> int:
    if not nodes:
        return 0
    starts = [node for node in nodes if not any(dst == node for _, dst in edges)]
    if not starts:
        starts = [next(iter(nodes))]
    adjacency: dict[object, list[object]] = {node: [] for node in nodes}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)
    best = 0
    queue = [(start, 1, {start}) for start in starts]
    while queue:
        node, depth, seen = queue.pop(0)
        best = max(best, depth)
        for dst in adjacency.get(node, []):
            if dst not in seen:
                queue.append((dst, depth + 1, seen | {dst}))
    return best


def extract_cfg_features(cfg_path: Path) -> dict[str, float | int]:
    rows = {field: 0 for field in CFG_FIELDS}
    if cfg_path.is_dir():
        functions = 0
        all_nodes: set[str] = set()
        all_edges: set[tuple[str, str]] = set()
        for dot_file in sorted(list(cfg_path.glob(".*.dot")) + list(cfg_path.glob("*.dot"))):
            text = dot_file.read_text(encoding="utf-8", errors="ignore")
            nodes, edges = parse_dot_edges(text)
            if nodes:
                functions += 1
                prefix = dot_file.name
                all_nodes.update(f"{prefix}:{node}" for node in nodes)
                all_edges.update((f"{prefix}:{src}", f"{prefix}:{dst}") for src, dst in edges)
        rows.update(graph_features(all_nodes, all_edges))
        rows["cfg_function_count"] = functions
        return rows

    text = cfg_path.read_text(encoding="utf-8", errors="ignore") if cfg_path.exists() else ""
    sections = re.split(r"\n(?=[A-Za-z_]\w*\s*\()", text)
    total_nodes: set[str] = set()
    total_edges: set[tuple[str, str]] = set()
    function_count = 0
    for index, section in enumerate(sections):
        nodes, edges = parse_clang_cfg_blocks(section)
        if not nodes:
            continue
        function_count += 1
        total_nodes.update(f"{index}:{node}" for node in nodes)
        total_edges.update((f"{index}:{src}", f"{index}:{dst}") for src, dst in edges)
    rows.update(graph_features(total_nodes, total_edges))
    rows["cfg_function_count"] = function_count
    return rows


def infer_input_scale(source_path: Path, ast_features: dict[str, float | int], ir_features: dict[str, float | int]) -> dict[str, int]:
    source = source_path.read_text(encoding="utf-8", errors="ignore")
    constants = [int(value) for value in re.findall(r"(?<![\w.])-?\b\d+\b(?!\.\d)", source) if abs(int(value)) < 100000000]
    positive = [value for value in constants if value > 0]
    max_value = max(positive) if positive else 0
    upper_names = dict((name, int(value)) for name, value in re.findall(r"#\s*define\s+([A-Z_][A-Z0-9_]*)\s+(\d+)", source))
    n_value = upper_names.get("N", upper_names.get("SIZE", max_value))
    return {
        "input_size_n": n_value,
        "matrix_m": upper_names.get("M", 0),
        "matrix_n": upper_names.get("N", n_value if "matrix" in source_path.stem.lower() else 0),
        "matrix_k": upper_names.get("K", 0),
        "graph_nodes": upper_names.get("NODES", 0),
        "graph_edges": upper_names.get("EDGES", 0),
        "image_width": upper_names.get("WIDTH", 0),
        "image_height": upper_names.get("HEIGHT", 0),
        "iterations": upper_names.get("ITERATIONS", upper_names.get("ITERS", 0)),
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


def run_program(exe_path: Path, warmup: int, runs: int, timeout: int) -> tuple[list[dict[str, object]], dict[str, float | int], bool, str]:
    records: list[dict[str, object]] = []
    measured: list[float] = []
    error = ""
    success = True
    for phase, count in [("warmup", warmup), ("measure", runs)]:
        for index in range(count):
            start = time.perf_counter()
            try:
                result = run_command([str(exe_path)], timeout=timeout)
                elapsed = time.perf_counter() - start
                ok = result.returncode == 0
                if phase == "measure" and ok:
                    measured.append(elapsed)
                if not ok:
                    success = False
                    error = (result.stderr or result.stdout or f"return code {result.returncode}").strip()
                records.append(
                    {
                        "phase": phase,
                        "run_index": index + 1,
                        "runtime_sec": elapsed,
                        "return_code": result.returncode,
                        "success": ok,
                    }
                )
            except subprocess.TimeoutExpired:
                success = False
                error = f"timeout after {timeout}s"
                records.append(
                    {
                        "phase": phase,
                        "run_index": index + 1,
                        "runtime_sec": timeout,
                        "return_code": "timeout",
                        "success": False,
                    }
                )
    return records, summarize(measured), success, error


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


def environment_manifest(args: argparse.Namespace) -> dict[str, object]:
    return {
        "env_id": args.env_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "processor": platform.processor(),
        "machine": platform.machine(),
        "c_compiler": args.c_compiler,
        "cxx_compiler": args.cxx_compiler,
        "run_c_compiler": args.run_c_compiler,
        "run_cxx_compiler": args.run_cxx_compiler,
        "clang_version": command_text([args.c_compiler, "--version"]),
        "gcc_version": command_text(["gcc", "--version"]),
        "opt_version": command_text(["opt", "--version"]) if shutil.which("opt") else "unavailable",
        "opt_level": args.opt_level,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "timeout_sec": args.timeout,
    }


def process_program(program: Program, args: argparse.Namespace) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    error_messages: list[str] = []
    extra_flags = args.extra_flag or []
    ast_path, ast_success, ast_error = generate_ast(program, args.c_compiler, args.cxx_compiler, extra_flags)
    if ast_error and not ast_success:
        error_messages.append(f"AST: {ast_error[:300]}")
    ast_features = extract_ast_features(ast_path, program.source_path)

    ir_path, ir_success, ir_error = generate_ir(program, args.c_compiler, args.cxx_compiler, args.opt_level, extra_flags)
    if ir_error and not ir_success:
        error_messages.append(f"IR: {ir_error[:300]}")
    ir_features = extract_ir_features(ir_path)

    cfg_path, cfg_success, cfg_error, cfg_kind = generate_cfg(program, ir_path, args.c_compiler, args.cxx_compiler, extra_flags)
    if cfg_error and not cfg_success:
        error_messages.append(f"CFG: {cfg_error[:300]}")
    cfg_features = extract_cfg_features(cfg_path)

    exe_path, build_success, build_error = compile_executable(program, args.run_c_compiler, args.run_cxx_compiler, args.opt_level, extra_flags)
    if build_error and not build_success:
        error_messages.append(f"BUILD: {build_error[:300]}")

    run_records: list[dict[str, object]] = []
    runtime_summary = summarize([])
    run_success = False
    run_error = ""
    if build_success and not args.no_run:
        run_records, runtime_summary, run_success, run_error = run_program(exe_path, args.warmup, args.runs, args.timeout)
        if run_error and not run_success:
            error_messages.append(f"RUN: {run_error[:300]}")

    input_scale = infer_input_scale(program.source_path, ast_features, ir_features)
    base = {
        "program_id": program.program_id,
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
    }
    static_row = {**base, **input_scale, **ast_features, **cfg_features, **ir_features}
    run_id_base = uuid.uuid5(uuid.NAMESPACE_URL, f"{args.env_id}:{program.program_id}:{time.time()}").hex[:12]
    run_rows = []
    for record in run_records:
        run_rows.append(
            {
                "run_id": f"{run_id_base}-{record['phase']}-{record['run_index']}",
                "program_id": program.program_id,
                "env_id": args.env_id,
                "input_id": args.input_id,
                "compile_config_id": args.compile_config_id,
                **record,
            }
        )
    summary_row = {
        "program_id": program.program_id,
        "env_id": args.env_id,
        "input_id": args.input_id,
        "compile_config_id": args.compile_config_id,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "timeout_sec": args.timeout,
        **runtime_summary,
        "run_success": run_success,
    }
    return static_row, run_rows, summary_row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AST/CFG/LLVM IR and collect runtime prediction features.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Directory containing source files.")
    parser.add_argument("--dataset", default="test", help="Dataset label written to CSV.")
    parser.add_argument("--task-type", default="mixed_test", help="Task type label written to CSV.")
    parser.add_argument("--parallel-model", default="serial", help="Parallel model label written to CSV.")
    parser.add_argument("--env-id", default="local_windows", help="Environment id written to CSV.")
    parser.add_argument("--input-id", default="default", help="Input id written to CSV.")
    parser.add_argument("--compile-config-id", default="release_O2", help="Compile config id written to CSV.")
    parser.add_argument("--opt-level", default="O2", choices=["O0", "O1", "O2", "O3"], help="Compiler optimization level.")
    parser.add_argument("--c-compiler", default="clang", help="C compiler command.")
    parser.add_argument("--cxx-compiler", default="clang++", help="C++ compiler command.")
    parser.add_argument("--run-c-compiler", default="gcc", help="C compiler used to build runnable executables.")
    parser.add_argument("--run-cxx-compiler", default="g++", help="C++ compiler used to build runnable executables.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs. Use 5 for formal collection.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs. Use 30 for formal collection.")
    parser.add_argument("--timeout", type=int, default=30, help="Per-run timeout in seconds.")
    parser.add_argument("--extra-flag", action="append", help="Extra compiler flag. Can be repeated.")
    parser.add_argument("--exclude", action="append", default=[], help="Glob pattern for source files to skip, relative to --source-dir. Can be repeated.")
    parser.add_argument("--no-run", action="store_true", help="Only generate representations and static features.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ensure_dirs()
    source_dir = Path(args.source_dir).resolve()
    programs = discover_sources(source_dir, SOURCE_EXTENSIONS, args.exclude)
    if not programs:
        print(f"No source files found in {source_dir}")
        return 1

    static_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for program in programs:
        print(f"[pipeline] processing {program.program_id}")
        static_row, program_run_rows, summary_row = process_program(program, args)
        static_rows.append(static_row)
        run_rows.extend(program_run_rows)
        summary_rows.append(summary_row)

    static_fields = [
        "program_id",
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
    ]
    combined_rows = []
    summary_by_program = {row["program_id"]: row for row in summary_rows}
    for row in static_rows:
        combined_rows.append({**row, **summary_by_program.get(row["program_id"], {})})

    static_fields = unique_fields(static_fields)
    summary_fields = unique_fields(summary_fields)
    run_fields = unique_fields(run_fields)
    results_fields = unique_fields(static_fields + [field for field in summary_fields if field != "program_id"])

    output_paths = [
        write_csv(RESULTS_DIR / "program_static_features.csv", static_rows, static_fields),
        write_csv(RESULTS_DIR / "run_records.csv", run_rows, run_fields),
        write_csv(RESULTS_DIR / "run_summary.csv", summary_rows, summary_fields),
        write_csv(RESULTS_DIR / "results.csv", combined_rows, results_fields),
    ]

    manifest = environment_manifest(args)
    output_paths.append(write_text(
        RESULTS_DIR / "environment_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    ))
    print(f"[pipeline] processed {len(programs)} programs")
    print("[pipeline] wrote:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
