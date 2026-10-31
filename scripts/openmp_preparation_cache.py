"""Verified per-benchmark OpenMP preparation reuse, with no runtime observations."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile


KIND = "rodinia_openmp_v1"
PAYLOAD_FIELDS = {"build_success", "build_command", "build_errors", "static_features", "static_errors", "executable_file"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(core, spec, args, openmp_root):
    root = Path(openmp_root).resolve()
    workdir = (root / spec.workdir).resolve()
    dependencies = {}
    roots = [workdir, root.parent / "common", core.pipeline_core().COMPAT_INCLUDE_DIR]
    roots.extend((workdir / item).resolve() for item in spec.include_dirs)
    for directory in dict.fromkeys(roots):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix.lower() in {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".inc", ".mk", ".config"}
                                   or path.name.lower().startswith("makefile")):
                dependencies[str(path.resolve())] = sha256(path)
    for source in [*spec.sources, *spec.ast_owned_sources]:
        path = (workdir / source).resolve()
        if not path.is_file():
            raise RuntimeError(f"Preparation dependency is missing: {path}")
        dependencies[str(path)] = sha256(path)
    tools = {}
    for command in sorted({args.c_compiler, args.cxx_compiler, "gcc", "g++", "make", "opt"}):
        resolved = shutil.which(command)
        if not resolved:
            raise RuntimeError(f"Preparation tool unavailable: {command}")
        path = Path(resolved).resolve()
        stat = path.stat()
        tools[command] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                          "version": core.pipeline_core().llvm_opt_version() if command == "opt" else core._compiler_version(command)}
    definition = asdict(spec)
    definition.pop("inputs")
    value = {"schema_version": 1, "cache_kind": KIND, "scope": "benchmark", "program_id": spec.benchmark_id,
             "benchmark": definition, "dependencies": dependencies, "tools": tools, "opt_level": args.opt_level,
             "execution_path": str((workdir / spec.executable).resolve()), "workdir": str(workdir),
             "platform": platform.platform(), "machine": platform.machine(), "build_policy": "make-force-on-cache-miss",
             "pipeline_sha256": sha256(core.__file__), "static_pipeline_sha256": sha256(core.pipeline_core().__file__), "cache_sha256": sha256(__file__),
             "compile_environment": {name: os.environ.get(name) for name in ["PATH", "CC", "CXX", "AR", "LD", "CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS",
                 "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "COMPILER_PATH", "CONDA_PREFIX", "MAKEFLAGS"]}}
    key = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return key, value


def read_verified(entry, key):
    entry = Path(entry).resolve()
    receipt = entry / "preparation.json"
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    actual_key = hashlib.sha256(json.dumps(saved["fingerprint"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    payload = saved["payload"]
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (saved.get("cache_kind") != KIND or saved["key"] != key or actual_key != key or set(payload) != PAYLOAD_FIELDS or not saved["files"]
            or payload_hash != saved.get("payload_sha256") or saved.get("contains_runtime_or_load_observations") is not False):
        raise RuntimeError("Invalid OpenMP preparation identity")
    for name, expected in saved["files"].items():
        path = (entry / name).resolve()
        if not path.is_relative_to(entry) or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"OpenMP preparation artifact missing/changed: {name}")
    binary = (entry / payload["executable_file"]).resolve()
    if not binary.is_relative_to(entry) or not binary.is_file() or payload["executable_file"] not in saved["files"]:
        raise RuntimeError("Invalid cached executable path")
    return payload.copy(), {"cache_kind": KIND, "cache_scope": "benchmark", "key": key,
            "manifest_file": str(receipt), "manifest_sha256": sha256(receipt), "program_id": saved["fingerprint"]["program_id"],
            "executable_sha256": sha256(binary), "execution_path": saved["fingerprint"]["execution_path"],
            "execution_verification": "sha256_before_each_invocation"}


def verify_execution_file(path, expected):
    if not Path(path).is_file() or sha256(path) != expected:
        raise RuntimeError("Actual OpenMP executable differs from the verified cached preparation; stop and inspect the build")


def get_or_prepare(core, spec, args, openmp_root):
    cache = Path(args.preparation_cache_dir).resolve()
    if cache.is_relative_to(Path(openmp_root).resolve()):
        raise ValueError("Preparation cache must be outside benchmark source tree")
    if args.clean or args.skip_build or not args.collect_static:
        raise ValueError("Preparation cache requires normal building and complete static collection; omit --clean/--skip-build/--no-static")
    if os.environ.get("MAKEFLAGS"):
        raise ValueError("Unset MAKEFLAGS before cached preparation so make cannot silently skip or alter the forced build")
    key, fingerprint = signature(core, spec, args, openmp_root)
    cache.mkdir(parents=True, exist_ok=True)
    entry = cache / key
    hit = entry.exists()
    if not hit:
        commands = []
        for command in spec.build_commands:
            if Path(command[0]).name not in {"make", "gmake"}:
                raise ValueError("OpenMP cache currently supports only declared make build commands")
            commands.append(command if "-B" in command or "--always-make" in command else (*command, "-B"))
        forced = replace(spec, build_commands=tuple(commands))
        success, build_command, errors = core.build_benchmark(forced, openmp_root, False, False, args.build_timeout)
        if not success:
            raise RuntimeError("OpenMP preparation build failed: " + "; ".join(errors))
        with tempfile.TemporaryDirectory(dir=cache, prefix="preparing-") as temporary:
            temporary = Path(temporary)
            static, static_errors = core.collect_static_features(spec, openmp_root, args.c_compiler, args.cxx_compiler, args.opt_level, temporary)
            allowed = {"static_status", "static_source_count", "static_source_success_count", *core.AST_FIELDS, *core.CFG_FIELDS, *core.IR_FIELDS}
            if set(static) != allowed or static["static_status"] != "success" or static_errors:
                raise RuntimeError("Incomplete or invalid static preparation; no entry published")
            after, _ = signature(core, spec, args, openmp_root)
            if after != key:
                raise RuntimeError("Build changed source/config dependencies; inspect generated sources before retrying")
            binary = (Path(openmp_root) / spec.workdir / spec.executable).resolve()
            (temporary / "build").mkdir()
            shutil.copy2(binary, temporary / "build" / binary.name)
            payload = {"build_success": True, "build_command": build_command, "build_errors": [], "static_features": static,
                       "static_errors": [], "executable_file": "build/" + binary.name}
            files = {p.relative_to(temporary).as_posix(): sha256(p) for p in temporary.rglob("*") if p.is_file()}
            record = {"schema_version": 1, "cache_kind": KIND, "key": key, "fingerprint": fingerprint, "payload": payload,
                      "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                      "files": files, "contains_runtime_or_load_observations": False}
            (temporary / "preparation.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
            if not entry.exists():
                temporary.rename(entry)
    payload, metadata = read_verified(entry, key)
    verify_execution_file(metadata["execution_path"], metadata["executable_sha256"])
    metadata["cache_hit"] = hit
    if not hasattr(args, "_preparation_provenance"):
        args._preparation_provenance = {}
    for item in spec.inputs:
        args._preparation_provenance[(spec.benchmark_id, item.input_id)] = {**metadata, "input_profile": item.input_id}
    print(f"[prepare] {'HIT' if hit else 'MISS'} {spec.benchmark_id} {key[:12]}", flush=True)
    return payload, metadata
