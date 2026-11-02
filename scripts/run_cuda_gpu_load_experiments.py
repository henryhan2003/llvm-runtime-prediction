"""Plan, preflight or explicitly collect CUDA under four GPU compute-load levels."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import sys
import time
import uuid
from types import SimpleNamespace

try:
    import gpu_resource_load as gpu
    import resource_load as host
    import run_load_experiments as common
    import check_load_experiment as checker
    import cpu_runtime_parameters as cpu_runtime
except ImportError:
    from scripts import gpu_resource_load as gpu, resource_load as host
    from scripts import run_load_experiments as common, check_load_experiment as checker
    from scripts import cpu_runtime_parameters as cpu_runtime

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT / "configs/rodinia_cuda_20sizes.json"


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["pilot","batch"],default="pilot")
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--permission-confirmed",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--pilot-validation",type=Path)
    p.add_argument("--gpu-index",type=int,default=0)
    p.add_argument("--cuda-arch",default="auto")
    p.add_argument("--nvcc",default="nvcc")
    p.add_argument("--nvidia-smi",default="nvidia-smi")
    p.add_argument("--temperature-policy",choices=["abort","record-only"],default="abort")
    p.add_argument("--timeout",type=float,default=300)
    p.add_argument("--trial-timeout",type=float,default=1800)
    p.add_argument("--max-measured-runs",type=int,default=20)
    p.add_argument("--prometheus-url",default="http://127.0.0.1:9090")
    p.add_argument("--prometheus-job",default="node")
    p.add_argument("--prometheus-instance",default="127.0.0.1:9100")
    p.add_argument("--prometheus-query-timeout",type=float,default=10)
    return p


def cuda_pipeline():
    try:
        import pipeline_rodinia_cuda as cuda
    except ImportError:
        from scripts import pipeline_rodinia_cuda as cuda
    return cuda


def preflight(args):
    gpu.require(platform.system()=="Linux","GPU preflight/execution requires Linux")
    cuda=cuda_pipeline()
    gpu.require(shutil.which(args.nvcc) and shutil.which(args.nvidia_smi),"Activate the CUDA-capable environment: nvcc/nvidia-smi unavailable")
    devices=cuda.query_gpu_devices(args.nvidia_smi)
    device=next((d for d in devices if d.physical_index==args.gpu_index),None)
    gpu.require(device is not None,"Selected physical GPU index is unavailable")
    arch=cuda.resolve_cuda_arch(args.cuda_arch,device)
    error=cuda.validate_nvcc_arch(args.nvcc,arch)
    gpu.require(not error,error)
    version=cuda._toolkit_version(cuda._full_version(args.nvcc))
    benchmarks=cuda.base.selected_benchmarks(cuda.base.load_manifest(MANIFEST),[],True)
    incompatible=cuda.incompatible_benchmarks_for_toolkit(benchmarks,version)
    gpu.require(not incompatible,"All 22 benchmarks must be supported; toolkit is incompatible with: "+", ".join(b.benchmark_id for b in incompatible)+". Use the previously validated CUDA environment; do not silently skip programs.")
    report=gpu.preflight(cuda,args,device)
    host_args=argparse.Namespace(**vars(args),prometheus_query_step=1.)
    report["host_monitoring"]=host.monitoring_preflight(host_args)
    cuda_root=ROOT/"datasets/rodinia_3.1/cuda"
    missing=[]
    for spec in benchmarks:
        for item in spec.inputs:
            if item.input_id.startswith("SIZE_"):
                missing.extend(cuda.base.validate_paths(spec,item,cuda_root,False))
    gpu.require(not missing,"Missing source/input files; materialize the existing 20-size inputs first: "+"; ".join(missing[:10]))
    report.update(cuda_arch=arch,nvcc_version=version,benchmark_count=len(benchmarks),
                  gpu_name=device.name,gpu_total_memory_bytes=device.total_memory_bytes,
                  gpu_compute_capability=device.compute_capability,gpu_driver_version=device.driver_version,
                  hostname=socket.gethostname(),disk_free_bytes=shutil.disk_usage(ROOT).free,
                  gpu_load_started=False,worker_compiled=False,
                  disk_headroom_warning="Keep several GiB free; cached CUDA AST artifacts may be large. No previous data is deleted.")
    return report


def create_manifest(args,root):
    definitions=json.loads(gpu.DEFINITIONS.read_text())
    benchmarks=[b for b in json.loads(MANIFEST.read_text())["benchmarks"] if b.get("enabled",True)]
    if args.stage=="pilot":
        benchmarks=[b for b in benchmarks if b["id"] in definitions["pilot_programs"]]
    sizes=definitions["pilot_sizes"] if args.stage=="pilot" else [f"SIZE_{i:02d}" for i in range(1,21)]
    trials=[]
    index=0
    for benchmark in benchmarks:
        for item in benchmark["inputs"]:
            if item["id"] not in sizes:
                continue
            profiles=definitions["profiles"]
            order=profiles[index%4:]+profiles[:index%4]
            if index%2:
                order=list(reversed(order))
            index+=1
            for profile in order:
                tid=f"gpu__{benchmark['id']}__{item['id']}__{profile['profile_id']}"
                directory=root/tid
                directory.mkdir(parents=True)
                config_path=directory/"load_config.json"
                gpu_path=directory/"gpu_load_config.json"
                host_config={"profile_id":profile["profile_id"],"load_type":"idle","permission_confirmed":args.permission_confirmed,
                    "temperature_policy":args.temperature_policy,"collection_protocol":"fast-v2","minimum_measured_runs":2,
                    "maximum_measured_runs":args.max_measured_runs,"stable_window_seconds":6,"min_raw_scrapes":3}
                gpu_config={**definitions["control"],**profile,"temperature_policy":args.temperature_policy}
                host.write_json(config_path,host_config)
                host.write_json(gpu_path,gpu_config)
                host.read_config(config_path)
                gpu.read_config(gpu_path)
                results=directory/"results"
                command=[sys.executable,str(ROOT/"scripts/pipeline_rodinia_cuda.py"),"--manifest",str(MANIFEST),"--benchmark",benchmark["id"],
                    "--input-profile",item["id"],"--collect-static","--warmup","1","--measurement-seconds","3","--min-measured-runs","2",
                    "--max-measured-runs",str(args.max_measured_runs),"--preparation-cache-dir",str(root/"preparation_cache"),
                    "--timeout",str(args.timeout),"--gpu-index",str(args.gpu_index),"--cuda-arch",args.cuda_arch,"--nvcc",args.nvcc,
                    "--nvidia-smi",args.nvidia_smi,"--gpu-metrics","nvidia-smi","--gpu-sample-interval","0.5",
                    "--prometheus-exporter","node","--prometheus-url",args.prometheus_url,"--prometheus-job",args.prometheus_job,
                    "--prometheus-instance",args.prometheus_instance,"--prometheus-context-seconds","6","--prometheus-query-step","1",
                    "--prometheus-query-delay","2","--prometheus-query-timeout",str(args.prometheus_query_timeout),
                    "--load-config",str(config_path),"--gpu-load-config",str(gpu_path),"--results-dir",str(results)]
                trials.append({"trial_id":tid,"program":benchmark["id"],"input_profile":item["id"],"profile_id":profile["profile_id"],
                    "load_family":"gpu","directory":str(directory),"results_dir":str(results),"config_path":str(config_path),
                    "config_sha256":gpu.digest(config_path),"gpu_config_path":str(gpu_path),"gpu_config_sha256":gpu.digest(gpu_path),
                    "command":command,"status":"PLANNED"})
    sources=["pipeline_rodinia_cuda.py","pipeline_rodinia.py","pipeline_new.py","cuda_preparation_cache.py","openmp_preparation_cache.py",
             "gpu_resource_load.py","gpu_resource_load_worker.cu","resource_load.py","resource_load_supervisor.py","check_load_experiment.py",
             "run_load_experiments.py","run_cuda_gpu_load_experiments.py","rodinia_process_probe.c","cpu_runtime_io_collector.c","cpu_runtime_parameters.py"]
    tracked=[ROOT/"scripts"/name for name in sources]+[MANIFEST,gpu.DEFINITIONS]
    return {"schema_version":1,"status":"PLANNED","stage":args.stage,"dataset":"cuda","collection_protocol":"fast-v2","load_families":["gpu"],
        "expected_configurations":len(trials),"expected_program_count":len(benchmarks),"expected_input_profiles":sizes,
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"execution_host":socket.gethostname() if args.execute else None,
        "gpu_uuid":None,"gpu_sharing_mode":"native_context_timeslicing_no_MPS","trials":trials,
        "process_resource_supervision":{"required":True,"probe_schema_version":4,"targets":list(cpu_runtime.RESOURCE_TARGETS),
                                        "scope":"target host process, not GPU VRAM, device memory traffic or whole-device telemetry"},
        "preparation_cache":{"enabled":True,"directory":str(root/"preparation_cache"),"scope":"benchmark","expected_unique_preparations":len(benchmarks)},
        "measurement_policy":{"warmup":1,"minimum_runs":2,"maximum_runs":args.max_measured_runs,"budget_seconds":3,"context_seconds":6,
                              "gpu_min_poll_observations_per_phase":3,"host_short_during_missing_permitted":True,"gpu_short_missing_permitted":False},
        "scripts_and_config_sha256":{str(p):gpu.digest(p) for p in tracked},
        "launcher_arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}


def verify_sources(manifest):
    for name,expected in manifest["scripts_and_config_sha256"].items():
        gpu.require(Path(name).is_file() and gpu.digest(name)==expected,"Source changed after experiment creation: "+name)


def verify_resume_preparations(manifest,report,completed):
    try:
        import cuda_preparation_cache as preparation
    except ImportError:
        from scripts import cuda_preparation_cache as preparation
    cuda=cuda_pipeline()
    observed={}
    for row in completed:
        observed.setdefault(row["program_id"],set()).add(row["preparation_cache_key"])
    seen=set()
    for trial in manifest["trials"]:
        if trial["status"]=="COMPLETED" or trial["program"] in seen:
            continue
        seen.add(trial["program"])
        args=cuda.parse_args(trial["command"][2:])
        nvcc=shutil.which(args.nvcc)
        gpu.require(nvcc is not None,"Original nvcc is no longer available")
        directory=args.cuda_dir or cuda.infer_cuda_dir(nvcc)
        context=cuda.token_context(nvcc,directory,args.cuda_arch)
        specs=cuda.base.selected_benchmarks(cuda.base.load_manifest(args.manifest),args.benchmark,args.all)
        gpu.require(len(specs)==1,"Each resumed CUDA trial must bind one benchmark")
        saved=dict(os.environ)
        try:
            cuda.configure_cuda_environment(context)
            os.environ["CUDA_VISIBLE_DEVICES"]=report["gpu_uuid"]
            key,_=preparation.signature(cuda,specs[0],args,args.rodinia_root.resolve()/"cuda",context,SimpleNamespace(uuid=report["gpu_uuid"]))
        finally:
            os.environ.clear();os.environ.update(saved)
        gpu.require(not observed.get(trial["program"]) or observed[trial["program"]]=={key},
                    "Original CUDA preparation environment changed; restore PATH/compiler/toolkit settings rather than rebuilding around old measurements")
        entry=args.preparation_cache_dir/key
        if entry.exists():
            _,receipt=preparation.read_verified(entry,key)
            preparation.verify_execution_file(receipt["execution_path"],receipt["executable_sha256"])


def batch_gate(args,report):
    if args.stage!="batch":
        return
    gpu.require(args.pilot_validation is not None and args.pilot_validation.is_file(),"Batch requires --pilot-validation from the 36-trial CUDA GPU-load pilot")
    pilot=json.loads(args.pilot_validation.read_text())
    gpu.require(pilot.get("status")=="PASS" and pilot.get("stage")=="pilot" and pilot.get("dataset")=="cuda" and
                pilot.get("gpu_four_levels_validated") and pilot.get("validated_configurations")==pilot.get("expected_configurations")==36 and
                pilot.get("preparation_cache_kinds")==["rodinia_cuda_v1"],"CUDA GPU-load pilot has not validated the full 36-trial protocol")
    gpu.require(pilot.get("process_resource_label_configurations")==36 and pilot.get("process_resource_targets")==cpu_runtime.RESOURCE_TARGETS,
                "Pilot did not validate all five host-process CPU/Memory/IO targets for all 36 configurations; run a new pilot")
    gpu.require(pilot.get("hostname")==socket.gethostname() and pilot.get("gpu_uuids")==[report["gpu_uuid"]],"Pilot GPU/host differs from batch")
    gpu.require(pilot.get("temperature_policies")==[args.temperature_policy],"Pilot/batch temperature policy differs")
    policy=pilot.get("measurement_policy",{})
    gpu.require(policy.get("maximum_runs")==args.max_measured_runs and policy.get("minimum_runs")==2 and
                policy.get("gpu_min_poll_observations_per_phase")==3 and not policy.get("gpu_short_missing_permitted",True),
                "Pilot/batch measurement and GPU coverage policies differ")
    gpu.require(bool(pilot.get("experiment_source_sha256")),"Pilot source fingerprints missing")
    for name,expected in pilot["experiment_source_sha256"].items():
        gpu.require(Path(name).is_file() and gpu.digest(name)==expected,"Pilot code/config changed; validate a new pilot before batch: "+name)


def run(args,root,manifest):
    path=root/"experiment_manifest.json"
    previous={}
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f"CUDA GPU-load experiment interrupted by {signum}")
    for sig in [signal.SIGHUP,signal.SIGTERM]:
        previous[sig]=signal.getsignal(sig)
        signal.signal(sig,interrupted)
    manifest["status"]="RUNNING"
    host.write_json(path,manifest)
    current=None
    try:
        for trial in manifest["trials"]:
            if trial["status"]=="COMPLETED":
                continue
            verify_sources(manifest)
            gpu.require(gpu.digest(trial["config_path"])==trial["config_sha256"] and
                        gpu.digest(trial["gpu_config_path"])==trial["gpu_config_sha256"],"Trial control configuration changed")
            gpu.require(shutil.disk_usage(root).free>=host.GIB,"Less than 1 GiB free; stop without deleting data")
            current=trial
            Path(trial["directory"]).mkdir(parents=True,exist_ok=True)
            trial.update(status="RUNNING",started_at=time.time())
            host.write_json(path,manifest)
            print("[run] "+trial["trial_id"],flush=True)
            code=common.run_trial(trial,args.trial_timeout)
            trial.update(exit_code=code,finished_at=time.time(),status="COMPLETED" if code==0 else "FAIL")
            host.write_json(path,manifest)
            gpu.require(code==0,"Trial failed; all prior results retained: "+trial["directory"]+"/pipeline.log")
            validated=checker.validate_trial(trial)
            gpu.require(validated.get("gpu_uuid")==manifest["gpu_uuid"],"Physical GPU identity changed during collection")
            current=None
        manifest["status"]="VALIDATING"
        host.write_json(path,manifest)
        report=checker.validate(path)
        manifest["status"]=report["status"]
        host.write_json(path,manifest)
        print(f"[{report['status']}] {root/'validation_report.json'}",flush=True)
        if report.get("sample_table"):
            print("[results] "+report["sample_table"],flush=True)
        return 0 if report["status"]=="PASS" else 1
    except BaseException as exc:
        if current is not None:
            current.update(status="FAIL",finished_at=time.time())
        manifest.update(status="FAIL",error=str(exc))
        host.write_json(path,manifest)
        raise
    finally:
        for sig,handler in previous.items():
            signal.signal(sig,handler)


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    if args.max_measured_runs<2 or args.timeout<=0 or args.trial_timeout<=0:
        p.error("Invalid measurement/timeout limits")
    if args.execute and (platform.system()!="Linux" or not args.permission_confirmed):
        p.error("Actual collection requires Linux and explicit --permission-confirmed")
    if args.resume and (not args.execute or args.output_dir is None):
        p.error("--resume requires --execute and the existing --output-dir")
    if args.preflight_only:
        if args.execute or args.resume:
            p.error("Read-only preflight cannot execute or resume")
        print(json.dumps(preflight(args),indent=2),flush=True)
        return 0
    root=(args.output_dir or ROOT/"results"/("cuda_gpu_load_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])).resolve()
    if args.resume:
        path=root/"experiment_manifest.json"
        manifest=json.loads(path.read_text())
        gpu.require(manifest["status"]=="FAIL" and manifest["dataset"]=="cuda" and manifest["load_families"]==["gpu"],"Resume requires a stopped CUDA GPU-load manifest")
        verify_sources(manifest)
        original=path.read_bytes()
        saved=manifest["launcher_arguments"]
        args=parser().parse_args([])
        for key,value in saved.items():
            setattr(args,key,Path(value) if key in {"output_dir","pilot_validation"} and value is not None else value)
        args.execute=args.permission_confirmed=args.resume=True
        report=preflight(args)
        gpu.require(report["gpu_uuid"]==manifest["gpu_uuid"] and socket.gethostname()==manifest["execution_host"],"Resume GPU/host changed")
        completed=[]
        for index,trial in enumerate(manifest["trials"],1):
            if trial["status"]=="COMPLETED":
                validated=checker.validate_trial(trial)
                gpu.require(validated.get("gpu_uuid")==manifest["gpu_uuid"],"Completed trial GPU identity differs from manifest")
                completed.append(validated)
            if index%100==0:
                print(f"[resume-check] {index}/{len(manifest['trials'])}",flush=True)
        verify_resume_preparations(manifest,report,completed)
        tag=datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6]
        (root/("manifest_before_resume_"+tag+".json")).write_bytes(original)
        for trial in manifest["trials"]:
            if trial["status"]=="COMPLETED":
                continue
            trial.setdefault("previous_attempts",[]).append({k:trial.get(k) for k in ["directory","results_dir","status","exit_code","started_at","finished_at"]})
            directory=root/"retries"/tag/trial["trial_id"]
            trial.update(directory=str(directory),results_dir=str(directory/"results"),status="PLANNED")
            trial["command"][trial["command"].index("--results-dir")+1]=trial["results_dir"]
        return run(args,root,manifest)
    gpu.require(not root.exists(),"Choose a fresh --output-dir; use --resume explicitly for a stopped experiment")
    report=preflight(args) if args.execute else None
    if report:
        args.cuda_arch=report["cuda_arch"]
        batch_gate(args,report)
    root.mkdir(parents=True)
    manifest=create_manifest(args,root)
    if report:
        manifest["gpu_uuid"]=report["gpu_uuid"]
        manifest["preflight"]=report
    host.write_json(root/"experiment_manifest.json",manifest)
    print(f"[plan] {len(manifest['trials'])} trials: {root/'experiment_manifest.json'}",flush=True)
    if not args.execute:
        print("[plan] No GPU load, benchmark, compiler, SSH connection or monitoring request started.")
        return 0
    return run(args,root,manifest)


if __name__=="__main__":
    raise SystemExit(main())
