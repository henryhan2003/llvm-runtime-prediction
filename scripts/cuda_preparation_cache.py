"""Benchmark-scoped CUDA preparation; preserve host-only IR/source-counter scope."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time

try:
    from openmp_preparation_cache import sha256
except ImportError:
    from scripts.openmp_preparation_cache import sha256

KIND="rodinia_cuda_v1"
PAYLOAD_FIELDS={"build_success","build_command","build_errors","static_features","static_errors","executable_file"}


def verify_execution_file(path,expected):
    if not Path(path).is_file() or sha256(path)!=expected:
        raise RuntimeError("Actual CUDA executable differs from verified preparation; inspect the build without overwriting it")


def signature(module,spec,args,cuda_root,context,device):
    work=(Path(cuda_root)/spec.workdir).resolve()
    dependencies={}
    directories=[work,Path(cuda_root).parent/"common",module.SCRIPT_DIR/"compat_include",*[(work/p).resolve() for p in spec.include_dirs]]
    for directory in dict.fromkeys(directories):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and (path.suffix.lower() in {".c",".h",".cpp",".hpp",".cu",".cuh",".cc",".cxx",".inc",".mk",".config",".ptx"} or path.name.lower().startswith("makefile")):
                dependencies[str(path.resolve())]=sha256(path)
    for name in {*spec.sources,*spec.ast_owned_sources}:
        path=(work/name).resolve()
        if not path.is_file():
            raise RuntimeError(f"Missing CUDA preparation source: {path}")
        dependencies[str(path)]=sha256(path)
    tools={}
    for command in sorted({context["nvcc"],args.c_compiler,args.cxx_compiler,"gcc","g++","make","opt"}):
        resolved=shutil.which(command)
        if not resolved:
            raise RuntimeError("Preparation tool unavailable: "+command)
        path=Path(resolved).resolve()
        tools[command]={"path":str(path),"size":path.stat().st_size,"mtime_ns":path.stat().st_mtime_ns,
                        "version":module._full_version(command)}
    for name in ["cuda.h","cuda_runtime.h","cuda_runtime_api.h"]:
        path=Path(context["cuda_include_dir"])/name
        if not path.is_file():
            raise RuntimeError("CUDA toolkit header missing: "+str(path))
        dependencies[str(path.resolve())]=sha256(path)
    definition=asdict(spec)
    definition.pop("inputs")
    value={"schema_version":1,"cache_kind":KIND,"scope":"benchmark","program_id":spec.benchmark_id,
           "benchmark":definition,"dependencies":dependencies,"tools":tools,"context":context,"gpu_uuid":device.uuid,
           "workdir":str(work),"execution_path":str((work/spec.executable).resolve()),"opt_level":args.opt_level,
           "generic_ir_scope":"CUDA host-only translation units; not complete device kernel IR",
           "cuda_source_scope":"existing declared-source counters; unchanged extraction",
           "platform":platform.platform(),"machine":platform.machine(),"source_code_sha256":{
               str(p):sha256(p) for p in [Path(module.__file__),Path(module.base.__file__),Path(module.base.pipeline_core().__file__),Path(__file__)]},
           "compile_environment":{k:os.environ.get(k) for k in ["PATH","CC","CXX","CFLAGS","CXXFLAGS","CPPFLAGS","LDFLAGS","CPATH",
                "C_INCLUDE_PATH","CPLUS_INCLUDE_PATH","LIBRARY_PATH","LD_LIBRARY_PATH","LD_PRELOAD","CUDA_HOME","CUDA_PATH","CONDA_PREFIX","MAKEFLAGS"]}}
    key=hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return key,value


def read_verified(entry,key):
    entry=Path(entry).resolve()
    receipt=entry/"preparation.json"
    saved=json.loads(receipt.read_text(encoding="utf-8"))
    payload=saved["payload"]
    actual=hashlib.sha256(json.dumps(saved["fingerprint"],sort_keys=True,allow_nan=False).encode()).hexdigest()
    payload_hash=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if saved.get("cache_kind")!=KIND or saved["key"]!=key or actual!=key or set(payload)!=PAYLOAD_FIELDS or payload_hash!=saved["payload_sha256"] or saved["contains_runtime_or_load_observations"] is not False:
        raise RuntimeError("Invalid CUDA preparation identity/payload")
    for name,expected in saved["files"].items():
        path=(entry/name).resolve()
        if not path.is_relative_to(entry) or not path.is_file() or sha256(path)!=expected:
            raise RuntimeError("CUDA preparation artifact missing/changed: "+name)
    binary=(entry/payload["executable_file"]).resolve()
    if not binary.is_relative_to(entry) or payload["executable_file"] not in saved["files"]:
        raise RuntimeError("Invalid CUDA executable receipt")
    return payload.copy(),{"cache_kind":KIND,"cache_scope":"benchmark","key":key,"manifest_file":str(receipt),"manifest_sha256":sha256(receipt),
        "program_id":saved["fingerprint"]["program_id"],"gpu_uuid":saved["fingerprint"]["gpu_uuid"],"executable_sha256":sha256(binary),
        "execution_path":saved["fingerprint"]["execution_path"],"execution_verification":"sha256_before_each_invocation"}


def get_or_prepare(module,spec,args,cuda_root,context,device):
    folder=Path(args.preparation_cache_dir).resolve()
    if folder.is_relative_to(Path(cuda_root).resolve()) or args.clean or args.skip_build or not args.collect_static or os.environ.get("MAKEFLAGS"):
        raise ValueError("CUDA cache requires an external cache directory, normal full-static build, and unset MAKEFLAGS")
    key,fingerprint=signature(module,spec,args,cuda_root,context,device)
    folder.mkdir(parents=True,exist_ok=True)
    entry=folder/key
    hit=entry.exists()
    if not hit:
        commands=[]
        for command in spec.build_commands:
            if Path(command[0]).name in {"make","gmake"} and "-B" not in command and "--always-make" not in command:
                command=(*command,"-B")
            commands.append(command)
        success,command,errors=module.build_benchmark(replace(spec,build_commands=tuple(commands)),cuda_root,context,False,False,args.build_timeout)
        if not success:
            raise RuntimeError("CUDA preparation build failed: "+"; ".join(errors))
        with tempfile.TemporaryDirectory(dir=folder,prefix="preparing-") as temporary:
            temporary=Path(temporary)
            static,errors=module.collect_static_features(spec,cuda_root,args.c_compiler,args.cxx_compiler,args.opt_level,
                                                        Path(context["cuda_dir"]),context["cuda_arch"],temporary)
            allowed={"static_status","static_source_count","static_source_success_count","cuda_static_source_metrics_success",
                     *module.base.AST_FIELDS,*module.base.CFG_FIELDS,*module.base.IR_FIELDS,*module.CUDA_STATIC_FIELDS}
            if set(static)!=allowed or static["static_status"]!="success" or not static["cuda_static_source_metrics_success"] or errors:
                raise RuntimeError("CUDA static preparation incomplete; inspect extraction errors before collection: "+"; ".join(errors))
            if signature(module,spec,args,cuda_root,context,device)[0]!=key:
                raise RuntimeError("Build changed CUDA source/config dependencies; no stale cache entry published")
            binary=(Path(cuda_root)/spec.workdir/spec.executable).resolve()
            (temporary/"build").mkdir()
            shutil.copy2(binary,temporary/"build"/binary.name)
            payload={"build_success":True,"build_command":command,"build_errors":[],"static_features":static,"static_errors":[],"executable_file":"build/"+binary.name}
            receipt={"schema_version":1,"cache_kind":KIND,"key":key,"fingerprint":fingerprint,"payload":payload,
                "payload_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest(),
                "files":{p.relative_to(temporary).as_posix():sha256(p) for p in temporary.rglob("*") if p.is_file()},"contains_runtime_or_load_observations":False}
            (temporary/"preparation.json").write_text(json.dumps(receipt,indent=2,allow_nan=False),encoding="utf-8")
            for attempt in range(5):
                try:
                    temporary.rename(entry)
                    break
                except PermissionError:
                    if os.name!="nt" or attempt==4 or entry.exists():
                        raise
                    time.sleep(.1)
    payload,metadata=read_verified(entry,key)
    verify_execution_file(metadata["execution_path"],metadata["executable_sha256"])
    metadata["cache_hit"]=hit
    if not hasattr(args,"_preparation_provenance"):
        args._preparation_provenance={}
    for item in spec.inputs:
        args._preparation_provenance[(spec.benchmark_id,item.input_id)]={**metadata,"input_profile":item.input_id}
    print(f"[prepare] {'HIT' if hit else 'MISS'} CUDA {spec.benchmark_id} {key[:12]}",flush=True)
    return payload,metadata
