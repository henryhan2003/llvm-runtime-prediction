"""Controlled GPU compute background; reuse host load guards and CUDA telemetry."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

try:
    import resource_load as host
except ImportError:
    from scripts import resource_load as host

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = ROOT / "configs/gpu_load_profiles.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_config(path):
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    definitions = json.loads(DEFINITIONS.read_text(encoding="utf-8"))
    allowed = {"profile_id", "target_gpu_pct", *definitions["control"]}
    require(set(supplied) == allowed, "Unknown/missing GPU load configuration fields")
    for key, value in supplied.items():
        if key in {"profile_id", "temperature_policy"}:
            continue
        require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value), "Nonfinite/non-numeric GPU control")
    profiles={p["profile_id"]:p["target_gpu_pct"] for p in definitions["profiles"]}
    require(supplied["profile_id"] in profiles and supplied["target_gpu_pct"] == profiles[supplied["profile_id"]], "GPU profile target mismatch")
    require(0 < supplied["tolerance_pct_points"] <= 10 and 0 <= supplied["max_idle_gpu_pct"] <= 15, "Unsafe GPU load tolerance")
    require(50 <= supplied["max_temperature_celsius"] <= 90 and supplied["temperature_policy"] in {"abort","record-only"}, "Invalid GPU temperature policy")
    require(.1 <= supplied["memory_reserve_fraction"] <= .5 and supplied["memory_reserve_bytes"] >= 256 * 1024**2, "Insufficient GPU memory reserve")
    for key in ["stable_window_seconds","stable_checks","min_phase_samples","startup_timeout_seconds","stabilization_timeout_seconds","max_duration_seconds","period_ms"]:
        require(supplied[key] > 0 and int(supplied[key])==supplied[key], "GPU controls must be positive integers")
    require(supplied["min_phase_samples"] >= 3 and supplied["stable_checks"] >= 3 and supplied["stable_window_seconds"] >= 6 and
            20 <= supplied["period_ms"] <= 100 and supplied["max_duration_seconds"] <= 3600, "Unsafe GPU sampling/duration policy")
    return supplied


def preflight(module, args, device):
    require(not any(os.environ.get(k) for k in ["CUDA_MPS_PIPE_DIRECTORY","CUDA_MPS_LOG_DIRECTORY","CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"]),
            "This protocol requires native non-MPS sharing; do not change an existing shared MPS service")
    for path in Path("/proc").glob("[0-9]*/comm"):
        try:
            if path.read_text().strip().startswith("nvidia-cuda-mps"):
                raise RuntimeError("An MPS service exists; this native-timeslicing protocol must not silently mix sharing modes")
        except (OSError,UnicodeError):
            pass
    result=module._run_text([args.nvidia_smi,f"--id={device.uuid}","--query-gpu=compute_mode","--format=csv,noheader,nounits"])
    require(result.returncode==0 and result.stdout.strip().lower()=="default", "Selected GPU must use DEFAULT compute mode; no system setting will be changed automatically")
    require(not module.query_compute_processes(args.nvidia_smi,device.uuid), "Selected GPU already has foreign compute jobs; wait or choose an authorized idle GPU")
    return {"gpu_uuid":device.uuid,"gpu_index":device.physical_index,"sharing_mode":"native_context_timeslicing_no_MPS",
            "compute_mode":"Default","foreign_compute_processes":0}


def build_worker(module, context):
    source=ROOT / "scripts/gpu_resource_load_worker.cu"
    version=module._full_version(context["nvcc"])
    identity={"source_sha256":digest(source),"nvcc":str(Path(context["nvcc"]).resolve()),"nvcc_version":version,
              "cuda_arch":context["cuda_arch"],"cuda_dir":context["cuda_dir"]}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=ROOT / "build/gpu_load_workers" / key
    binary=folder / "gpu_load_worker"
    receipt=folder / "worker.json"
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        require(saved["identity"]==identity and binary.is_file() and digest(binary)==saved["binary_sha256"], "GPU worker cache corrupted; inspect it instead of silently overwriting")
        return binary,saved
    require(not folder.exists(), "Incomplete GPU worker build directory exists; inspect before retrying")
    folder.mkdir(parents=True)
    command=[context["nvcc"],"-O2","-std=c++14","-arch="+context["cuda_arch"],str(source),"-o",str(binary)]
    result=module._run_text(command,timeout=180)
    require(result.returncode==0 and binary.is_file(), "GPU worker compilation failed: "+result.stderr[-3000:])
    saved={"identity":identity,"binary_sha256":digest(binary),"command":command}
    receipt.write_text(json.dumps(saved,indent=2),encoding="utf-8")
    return binary,saved


def descendant_of(pid, ancestor):
    seen=set()
    while pid > 1 and pid not in seen:
        if pid==ancestor:
            return True
        seen.add(pid)
        try:
            pid=int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")",1)[1].split()[1])
        except (OSError,ValueError,IndexError):
            return False
    return False


class Controller:
    def __init__(self, session, module, args, device, sampler, context):
        self.session,self.module,self.args,self.device,self.sampler=session,module,args,device,sampler
        self.config=read_config(args.gpu_load_config)
        self.binary,self.worker=build_worker(module,context)
        self.path=session.path.with_suffix(".duty")
        self.ready=session.path.with_suffix(".gpu-ready.json")
        self.process=None
        self.log=None
        self.started=None
        self.stopped=False
        self.duty=self.config["target_gpu_pct"]/100
        self.lock=threading.RLock()
        self.last_process_check=0.
        self.record={"config":self.config,"config_file":str(args.gpu_load_config.resolve()),"config_sha256":digest(args.gpu_load_config),
                     "gpu_uuid":device.uuid,"gpu_index":device.physical_index,"sharing_mode":"native_context_timeslicing_no_MPS",
                     "load_kind":"fixed_buffer_compute_duty","worker":self.worker,"calibration":[],
                     "target_percentages_are_not_observations":True,"control_parameters_are_model_inputs":False,
                     "after_phase_load_policy":"background remains active through after; cleanup outside target timing"}

    def samples(self,start,end):
        samples,errors=self.sampler.snapshot()
        require(not errors,"GPU sampler error: "+" | ".join(errors))
        selected=[s for s in samples if start <= s.timestamp <= end]
        require(all(math.isfinite(s.timestamp) and all(isinstance(v,(int,float)) and math.isfinite(v) for v in s.values.values()) for s in selected),
                "Nonfinite/invalid GPU observation")
        require(all(0<=s.values.get("gpu_utilization_pct",0)<=100 and s.values.get("gpu_memory_used_bytes",0)>=0 for s in selected),
                "GPU utilization/memory observation outside physical bounds")
        return selected

    def set_duty(self,value):
        require(0 <= value <= .85,"GPU worker duty hard cap exceeded")
        temporary=self.path.with_suffix(".duty.tmp")
        temporary.write_text(str(value),encoding="ascii")
        temporary.replace(self.path)
        self.duty=value

    def start(self):
        self.record["preflight"]=preflight(self.module,self.args,self.device)
        self.started=time.monotonic()
        self.record["load_started_at"]=time.time()
        if not self.config["target_gpu_pct"]:
            return
        self.set_duty(self.duty)
        require(digest(self.binary)==self.worker["binary_sha256"],"GPU worker binary changed")
        command=[sys.executable,str(ROOT / "scripts/resource_load_supervisor.py"),"--parent",str(os.getpid()),
                 "--timeout",str(self.config["max_duration_seconds"]),"--",str(self.binary),str(self.path),str(self.ready),
                 str(self.config["max_duration_seconds"]),str(self.config["period_ms"])]
        self.log=self.session.path.with_suffix(".gpu.log").open("wb")
        self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        self.record.update(command=command,owned_process_group=self.process.pid)
        deadline=time.monotonic()+self.config["startup_timeout_seconds"]
        while not self.ready.exists():
            self.session.check()
            require(time.monotonic()<deadline,"GPU worker startup timed out")
            time.sleep(.1)
        ready=json.loads(self.ready.read_text())
        require(descendant_of(int(ready["pid"]),self.process.pid),"GPU worker readiness PID is not owned")
        self.record["ready"]=ready

    def check(self):
        if self.started is None or self.stopped:
            return
        with self.lock:
            require(time.monotonic()-self.started<self.config["max_duration_seconds"],"GPU load duration cap reached")
            require(self.process is None or self.process.poll() is None,"GPU background worker exited early")
            samples=self.samples(time.time()-max(3.,self.args.gpu_sample_interval*4),time.time())
            require(bool(samples),"GPU samples are stale during load control")
            latest=samples[-1]
            temperature=latest.values.get("gpu_temperature_celsius")
            if self.config["temperature_policy"]=="abort":
                require(temperature is not None,"GPU temperature unavailable under abort policy")
                require(temperature<self.config["max_temperature_celsius"],"GPU temperature cap reached; owned load will stop")
            used=latest.values.get("gpu_memory_used_bytes")
            reserve=max(self.config["memory_reserve_bytes"],self.device.total_memory_bytes*self.config["memory_reserve_fraction"])
            require(used is not None and self.device.total_memory_bytes-used>=reserve,"GPU free-memory reserve reached")
            if time.monotonic()-self.last_process_check>=2:
                processes=self.module.query_compute_processes(self.args.nvidia_smi,self.device.uuid)
                foreign=[p for p in processes if Path("/proc",str(p["pid"])).exists() and not descendant_of(int(p["pid"]),os.getpid())]
                self.record["last_foreign_processes"]=foreign
                self.record.setdefault("process_checks",[]).append({"timestamp":time.time(),"reported_pids":[p["pid"] for p in processes],
                                                                   "foreign_pids":[p["pid"] for p in foreign]})
                require(not foreign,"Foreign GPU compute job appeared; controlled experiment stopped")
                self.last_process_check=time.monotonic()

    def stabilize(self):
        deadline=time.monotonic()+self.config["stabilization_timeout_seconds"]
        history=[]
        window=self.config["stable_window_seconds"]
        self.session.wait(window+1)
        while time.monotonic()<deadline:
            self.session.check()
            end=time.time()-1
            samples=self.samples(end-window,end)
            values=[s.values["gpu_utilization_pct"] for s in samples if "gpu_utilization_pct" in s.values]
            require(len(values)>=self.config["min_phase_samples"],"Too few GPU observations for calibration")
            observed=statistics.fmean(values)
            target=self.config["target_gpu_pct"]
            matched=abs(observed-target)<=self.config["tolerance_pct_points"] if target else observed<=self.config["max_idle_gpu_pct"]
            with self.lock:
                self.record["calibration"].append({"start":end-window,"end":end,"gpu_mean_pct":observed,"duty":self.duty,"matched":matched})
            history=history+[observed] if matched else []
            if len(history)>=self.config["stable_checks"] and max(history[-3:])-min(history[-3:])<=self.config["tolerance_pct_points"]:
                self.record.update(calibrated_duty=self.duty,duty_frozen_before_target=True,stable_at=time.time())
                return
            if not matched and target and observed>0:
                self.set_duty(min(.85,max(.01,self.duty*target/observed)))
                self.session.wait(window+1)
            else:
                self.session.wait(2)
        raise RuntimeError("GPU load could not reach a stable measured target; no target measurement accepted")

    def measurement_ready(self,start,end):
        return len(self.samples(start,end))>=self.config["min_phase_samples"]

    def stop(self):
        with self.lock:
            if self.stopped:
                return
            self.stopped=True
            if self.process is not None:
                for sig in [signal.SIGTERM,signal.SIGKILL]:
                    try:
                        os.killpg(self.process.pid,sig)
                    except ProcessLookupError:
                        pass
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        continue
                try:
                    os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
            residual=[]
            if self.process is not None:
                for path in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields=path.read_text().rsplit(")",1)[1].split()
                        if int(fields[2])==self.process.pid and fields[0]!="Z":
                            residual.append(int(path.parent.name))
                    except (OSError,ValueError,IndexError):
                        pass
            self.record.update(load_stopped_at=time.time(),residual_owned_group_pids=residual,
                               cleanup_success=(self.process is None or self.process.poll() is not None) and not residual)
            if self.log is not None:
                self.log.close()

    def snapshot(self,windows):
        with self.lock:
            record=deepcopy(self.record)
        record["phase_samples"]={p:[{"timestamp":s.timestamp,"values":s.values} for s in self.samples(*window)] for p,window in (windows or {}).items()}
        return record


def validate_record(record,row):
    gpu=record["gpu_load"]
    config=gpu["config"]
    require(gpu["cleanup_success"] is True and gpu["duty_frozen_before_target"] is True,"GPU load cleanup/calibration failed")
    require(gpu["gpu_uuid"]==row["gpu_uuid"] and gpu["sharing_mode"]=="native_context_timeslicing_no_MPS","GPU identity/sharing mode changed")
    require(not gpu.get("last_foreign_processes"),"Foreign GPU compute process recorded")
    require(gpu["stable_at"]<=record["phase_windows"]["before"][0] and gpu["load_stopped_at"]>=record["phase_windows"]["after"][1],"GPU load/phase order invalid")
    actual=float(row["gpu_utilization_pct_before_mean"])
    target=config["target_gpu_pct"]
    require(abs(actual-target)<=config["tolerance_pct_points"] if target else actual<=config["max_idle_gpu_pct"],"Actual BEFORE GPU utilization missed target")
    for phase in ["before","during","after"]:
        samples=gpu["phase_samples"][phase]
        start,end=record["phase_windows"][phase]
        require(len(samples)>=config["min_phase_samples"] and len({s["timestamp"] for s in samples})==len(samples) and
                all(start<=s["timestamp"]<=end and "gpu_utilization_pct" in s["values"] and "gpu_memory_used_bytes" in s["values"] for s in samples),"GPU phase lacks sufficient real observations")
        require(all(math.isfinite(s["timestamp"]) and 0<=s["values"]["gpu_utilization_pct"]<=100 and
                    math.isfinite(s["values"]["gpu_memory_used_bytes"]) and s["values"]["gpu_memory_used_bytes"]>=0 for s in samples),"Invalid GPU phase observation")
        require(str(row[f"gpu_{phase}_collection_success"]).lower() in {"true","1"} and int(row[f"gpu_{phase}_sample_count"])==len(samples),"GPU phase summary/sample count mismatch")
        for metric in ["gpu_utilization_pct","gpu_memory_used_bytes"]:
            require(math.isclose(float(row[f"{metric}_{phase}_mean"]),statistics.fmean(s["values"][metric] for s in samples),rel_tol=1e-8,abs_tol=1e-8),"GPU raw samples/summary mean differ")
        for edge,timestamp in zip(["start","end"],[start,end]):
            require(math.isclose(float(row[f"gpu_{phase}_window_{edge}_timestamp"]),timestamp,rel_tol=0,abs_tol=1e-6),"GPU/host phase window mismatch")
    return {"gpu_uuid":gpu["gpu_uuid"],"gpu_before_mean_pct":actual,"gpu_target_pct":target,"gpu_sharing_mode":gpu["sharing_mode"]}


class GpuSession(host.LoadSession):
    def __init__(self,args,core,identity,module,device,sampler,context):
        super().__init__(args,core,identity)
        require(self.config["load_type"]=="idle","GPU experiment must not add CPU/memory pressure")
        self.gpu=Controller(self,module,args,device,sampler,context)

    def __enter__(self):
        super().__enter__()
        try:
            self.gpu.start()
            self.gpu.stabilize()
            self.record["stable_at"]=time.time()
            self.save()
            return self
        except BaseException as exc:
            self.record.update(status="FAIL",error=str(exc))
            self.cleanup()
            self.save()
            raise

    def check(self):
        super().check()
        self.gpu.check()

    def stop_owned_process(self):
        self.gpu.stop()
        super().stop_owned_process()

    def cleanup(self):
        super().cleanup()
        self.record["cleanup_success"] &= self.gpu.record.get("cleanup_success",False)

    def measurement_ready(self,start,end):
        return self.gpu.measurement_ready(start,end)

    def save(self):
        self.record["gpu_load"]=self.gpu.snapshot(self.last_windows)
        if self.record.get("outputs"):
            try:
                validate_record(self.record,self.record["measured_summary"])
            except (RuntimeError,KeyError,ValueError,TypeError) as exc:
                self.record.update(status="FAIL",error=str(exc))
        super().save()


def session(args,core,identity,module,device,sampler,context):
    if getattr(args,"gpu_load_config",None) is None:
        return host.session(args,core,identity)
    require(args.load_config is not None and sampler is not None,"GPU load requires host load guard and existing live GPU sampler")
    value=GpuSession(args,core,identity,module,device,sampler,context)
    args._load_sessions.append(value)
    return value
