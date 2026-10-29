"""Experiment-local, verified preparation cache; never stores timings or load state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile


PATH_FIELDS = ("ast_path", "ir_path", "cfg_path", "exe_path")
PAYLOAD_FIELDS = {*PATH_FIELDS, "ast_success", "ir_success", "cfg_success", "build_success", "cfg_kind", "errors",
                  "ast_features", "ir_features", "cfg_features"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(core, program, args):
    if args.extra_flag:
        raise ValueError("Preparation cache supports default PolyBench flags only; omit the cache for custom flags")
    roots = [program.source_root, core.COMPAT_INCLUDE_DIR,
             core.PROJECT_ROOT / "datasets/PolyBenchC-4.2.1/utilities"]
    dependencies = {str(program.source_path.resolve()): sha256(program.source_path)}
    for root in roots:
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".c", ".h", ".inc", ".hpp", ".cc", ".cpp", ".cxx", ".hxx"}:
                    dependencies[str(path.resolve())] = sha256(path)
    tools = {}
    commands = {args.c_compiler, args.cxx_compiler, args.run_c_compiler, args.run_cxx_compiler, "opt"}
    for command in sorted(commands):
        executable = shutil.which(command)
        if not executable:
            raise RuntimeError(f"Cache toolchain executable unavailable: {command}")
        path = Path(executable).resolve()
        stat = path.stat()
        tools[command] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                          "version": core.llvm_opt_version() if command == "opt" else core.compiler_version(command)}
    value = {"schema_version": 1, "program_id": program.program_id, "input_profile": program.input_profile,
             "source_root": str(program.source_root.resolve()), "dependencies": dependencies,
             "tools": tools, "opt_level": args.opt_level,
             "flags": core.program_extra_flags(program, []), "platform": platform.platform(),
             "machine": platform.machine(), "pipeline_sha256": sha256(core.__file__), "cache_sha256": sha256(__file__),
             "compiler_environment": {k: os.environ.get(k) for k in ["PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
                 "LIBRARY_PATH", "LD_LIBRARY_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "CONDA_PREFIX", "SOURCE_DATE_EPOCH"]}}
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest(), value


def read_verified(entry, key):
    manifest_path = entry / "preparation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint_key = hashlib.sha256(json.dumps(manifest["fingerprint"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    if manifest["key"] != key or fingerprint_key != key or not manifest.get("files") or set(manifest["payload"]) != PAYLOAD_FIELDS:
        raise RuntimeError("Invalid preparation cache identity")
    for name, expected in manifest["files"].items():
        path = (entry / name).resolve()
        if not path.is_relative_to(entry.resolve()) or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Preparation cache artifact missing/changed: {name}")
    payload = manifest["payload"].copy()
    for field in PATH_FIELDS:
        relative = payload[field]
        path = (entry / relative).resolve()
        if not path.is_relative_to(entry.resolve()) or not path.exists():
            raise RuntimeError(f"Preparation cache path is invalid: {field}")
        payload[field] = path
    return payload, {"key": key, "manifest_file": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                     "artifact_count": len(manifest["files"]), "executable_sha256": sha256(payload["exe_path"]),
                     "program_id": manifest["fingerprint"]["program_id"], "input_profile": manifest["fingerprint"]["input_profile"]}


def get_or_prepare(core, program, args, producer):
    cache = getattr(args, "preparation_cache_dir", None)
    if cache is None:
        return producer()
    if not program.input_profile:
        raise ValueError("Preparation cache requires an explicit PolyBench input profile")
    root = Path(cache).resolve()
    if root.is_relative_to(program.source_root.resolve()):
        raise ValueError("Preparation cache must be outside the benchmark source tree")
    key, fingerprint = signature(core, program, args)
    root.mkdir(parents=True, exist_ok=True)
    entry = root / key
    hit = entry.exists()
    if not hit:
        previous = {name: getattr(core, name) for name in ["AST_DIR", "IR_DIR", "CFG_DIR", "BUILD_DIR"]}
        with tempfile.TemporaryDirectory(prefix="preparing-", dir=root) as temporary:
            temporary = Path(temporary)
            try:
                for name, subdir in [("AST_DIR", "ast"), ("IR_DIR", "ir"), ("CFG_DIR", "cfg"), ("BUILD_DIR", "build")]:
                    setattr(core, name, temporary / subdir)
                payload = producer()
            finally:
                for name, path in previous.items():
                    setattr(core, name, path)
            if set(payload) != PAYLOAD_FIELDS:
                raise ValueError("Only pre-execution preparation fields may be cached")
            if not all(payload[name] for name in ["ast_success", "ir_success", "cfg_success", "build_success"]):
                raise RuntimeError("Preparation failed; no failed/incomplete artifacts were cached: " + str(payload.get("errors", [])))
            for field in PATH_FIELDS:
                payload[field] = Path(payload[field]).relative_to(temporary).as_posix()
            files = {p.relative_to(temporary).as_posix(): sha256(p) for p in temporary.rglob("*") if p.is_file()}
            manifest = {"schema_version": 1, "key": key, "fingerprint": fingerprint, "payload": payload, "files": files,
                        "contains_runtime_or_load_observations": False}
            (temporary / "preparation.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
            # Publish once; never overwrite a completed or corrupt entry.
            if not entry.exists():
                temporary.rename(entry)
    payload, provenance = read_verified(entry, key)
    provenance["cache_hit"] = hit
    if not hasattr(args, "_preparation_provenance"):
        args._preparation_provenance = {}
    args._preparation_provenance[(program.program_id, program.input_profile)] = provenance
    print(f"[prepare] {'HIT' if hit else 'MISS'} {program.program_id}[{program.input_profile}] {key[:12]}", flush=True)
    return payload
