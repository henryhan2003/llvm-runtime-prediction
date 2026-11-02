from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import types
from unittest import mock

import test_resource_load as fixtures
import test_load_fast_protocol as fast
import cuda_preparation_cache as cache
import gpu_resource_load as gpu
import run_cuda_gpu_load_experiments as runner

cuda,base,host,checker=fixtures.cuda,fixtures.omp,fixtures.load,fixtures.checker


class Fixture(fixtures.Fixture):
    def gpu_config(self,level="GPU_LOW",**changes):
        definitions=json.loads(gpu.DEFINITIONS.read_text())
        profile=next(p for p in definitions["profiles"] if p["profile_id"]==level)
        path=self.root/"gpu.json"
        path.write_text(json.dumps({**definitions["control"],**profile,**changes}),encoding="utf-8")
        return path

    def gpu_record(self):
        config=gpu.read_config(self.gpu_config())
        windows={"before":(10.,16.),"during":(17.,23.),"after":(23.,29.)}
        record={"phase_windows":windows,"gpu_load":{"config":config,"gpu_uuid":"GPU-mock","sharing_mode":"native_context_timeslicing_no_MPS",
            "cleanup_success":True,"duty_frozen_before_target":True,"stable_at":9.,"load_stopped_at":30.,"phase_samples":{},"last_foreign_processes":[]}}
        row={"gpu_uuid":"GPU-mock"}
        for phase,(start,end) in windows.items():
            record["gpu_load"]["phase_samples"][phase]=[{"timestamp":start+i,"values":{"gpu_utilization_pct":25.,"gpu_memory_used_bytes":1024.}} for i in [1,2,3]]
            row.update({f"gpu_{phase}_collection_success":True,f"gpu_{phase}_sample_count":3,
                        f"gpu_utilization_pct_{phase}_mean":25.,f"gpu_memory_used_bytes_{phase}_mean":1024.,
                        f"gpu_{phase}_window_start_timestamp":start,f"gpu_{phase}_window_end_timestamp":end})
        return record,row


class ConfigAndSafetyTests(Fixture):
    def controller(self):
        c=gpu.Controller.__new__(gpu.Controller)
        c.config=gpu.read_config(self.gpu_config())
        c.device=cuda.GpuDevice(0,"GPU-mock","MOCK","8.6",4*host.GIB,"MOCK",None)
        c.args=types.SimpleNamespace(gpu_sample_interval=.5,nvidia_smi="MOCK_SMI")
        c.module=types.SimpleNamespace(query_compute_processes=mock.Mock(return_value=[]))
        c.process=mock.Mock(pid=765432)
        c.process.poll.return_value=None
        c.started=0
        c.stopped=False
        c.last_process_check=0
        c.lock=threading.RLock()
        c.record={"calibration":[]}
        c.log=None
        c.duty=.25
        c.sampler=types.SimpleNamespace(snapshot=lambda:([cuda.GpuSample(99,{"gpu_temperature_celsius":50,"gpu_memory_used_bytes":1024,"gpu_utilization_pct":25})],[]))
        return c

    def test_profile_semantics_and_unsafe_values_rejected(self):
        self.assertEqual(25,gpu.read_config(self.gpu_config())["target_gpu_pct"])
        for change in [{"target_gpu_pct":100},{"target_gpu_pct":float("nan")},{"tolerance_pct_points":50},{"temperature_policy":"off"},
                       {"memory_reserve_fraction":0},{"min_phase_samples":1},{"max_duration_seconds":99999}]:
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                gpu.read_config(self.gpu_config(**change))

    def test_preflight_rejects_mps_exclusive_mode_and_foreign_jobs(self):
        args=types.SimpleNamespace(nvidia_smi="MOCK")
        device=types.SimpleNamespace(uuid="GPU-mock",physical_index=0)
        module=types.SimpleNamespace(_run_text=mock.Mock(return_value=types.SimpleNamespace(returncode=0,stdout="Default\n")),query_compute_processes=mock.Mock(return_value=[]))
        with mock.patch.dict(gpu.os.environ,{"CUDA_MPS_PIPE_DIRECTORY":"/some/mps"}):
            with self.assertRaisesRegex(RuntimeError,"non-MPS"):
                gpu.preflight(module,args,device)
        with mock.patch.dict(gpu.os.environ,{},clear=True),mock.patch.object(gpu.Path,"glob",return_value=[]):
            self.assertEqual("Default",gpu.preflight(module,args,device)["compute_mode"])
            module._run_text.return_value.stdout="Exclusive_Process"
            with self.assertRaisesRegex(RuntimeError,"DEFAULT"):
                gpu.preflight(module,args,device)
            module._run_text.return_value.stdout="Default"
            module.query_compute_processes.return_value=[{"pid":999}]
            with self.assertRaisesRegex(RuntimeError,"foreign"):
                gpu.preflight(module,args,device)

    def test_temperature_memory_stale_worker_and_foreign_guards(self):
        c=self.controller()
        with mock.patch.object(gpu.time,"time",return_value=100),mock.patch.object(gpu.time,"monotonic",return_value=100):
            c.check()
            c.sampler.snapshot=lambda:([cuda.GpuSample(99,{"gpu_temperature_celsius":86,"gpu_memory_used_bytes":1024})],[])
            with self.assertRaisesRegex(RuntimeError,"temperature cap"):
                c.check()
            c.config["temperature_policy"]="record-only"
            c.check()
            c.sampler.snapshot=lambda:([cuda.GpuSample(99,{"gpu_temperature_celsius":50,"gpu_memory_used_bytes":4*host.GIB-1})],[])
            with self.assertRaisesRegex(RuntimeError,"memory reserve"):
                c.check()
            c.sampler.snapshot=lambda:([],[])
            with self.assertRaisesRegex(RuntimeError,"stale"):
                c.check()
            c.process.poll.return_value=1
            with self.assertRaisesRegex(RuntimeError,"exited early"):
                c.check()
            c.process.poll.return_value=None
            c.sampler.snapshot=lambda:([cuda.GpuSample(99,{"gpu_temperature_celsius":50,"gpu_memory_used_bytes":1024})],[])
            c.last_process_check=0
            c.module.query_compute_processes.return_value=[{"pid":999}]
            with mock.patch.object(gpu.Path,"exists",return_value=True),mock.patch.object(gpu,"descendant_of",return_value=False):
                with self.assertRaisesRegex(RuntimeError,"Foreign GPU"):
                    c.check()

    def test_stop_signals_only_owned_group_and_records_cleanup(self):
        c=self.controller()
        c.process.wait.return_value=0
        c.process.poll.return_value=0
        with mock.patch.object(gpu.os,"killpg",create=True) as kill,mock.patch.object(gpu.Path,"glob",return_value=[]), \
             mock.patch.object(gpu.signal,"SIGKILL",9,create=True):
            c.stop()
            c.stop()
        self.assertTrue(c.record["cleanup_success"])
        self.assertTrue(all(call.args[0]==765432 for call in kill.call_args_list))

    def test_calibration_uses_measured_utilization_and_freezes_duty(self):
        c=self.controller()
        clock=fixtures.FakeClock()
        c.session=types.SimpleNamespace(check=lambda:None,wait=clock.sleep)
        calls=[]
        def samples(start,end):
            calls.append(end)
            value=50 if len(calls)==1 else 25
            return [cuda.GpuSample(end-i,{"gpu_utilization_pct":value}) for i in [1,2,3]]
        c.samples=samples
        c.set_duty=lambda value:setattr(c,"duty",value)
        with mock.patch.object(gpu.time,"time",side_effect=clock.now),mock.patch.object(gpu.time,"monotonic",side_effect=clock.now):
            c.stabilize()
        self.assertAlmostEqual(.125,c.record["calibrated_duty"])
        self.assertTrue(c.record["duty_frozen_before_target"])
        self.assertEqual(50,c.record["calibration"][0]["gpu_mean_pct"])
        self.assertEqual(4,len(calls))

    def test_gpu_samples_are_not_faked_for_short_during_windows(self):
        record,row=self.gpu_record()
        self.assertEqual(25,gpu.validate_record(record,row)["gpu_before_mean_pct"])
        record["gpu_load"]["phase_samples"]["during"]=[]
        row["gpu_during_sample_count"]=0
        row["gpu_during_collection_success"]=False
        with self.assertRaisesRegex(RuntimeError,"real observations"):
            gpu.validate_record(record,row)

    def test_before_target_cleanup_scope_and_sample_means_are_validated(self):
        original,row=self.gpu_record()
        for change in ["before","cleanup","scope","mean","window"]:
            record=deepcopy(original);observed=deepcopy(row)
            if change=="before":observed["gpu_utilization_pct_before_mean"]=85
            if change=="cleanup":record["gpu_load"]["cleanup_success"]=False
            if change=="scope":observed["gpu_uuid"]="different_gpu"
            if change=="mean":observed["gpu_memory_used_bytes_during_mean"]=999
            if change=="window":observed["gpu_after_window_start_timestamp"]=99
            with self.subTest(change=change),self.assertRaises(RuntimeError):
                gpu.validate_record(record,observed)

    def test_failed_gpu_validation_cannot_be_overwritten_by_host_finalizer_pass(self):
        record,row=self.gpu_record()
        record["gpu_load"]["cleanup_success"]=False
        session=gpu.GpuSession.__new__(gpu.GpuSession)
        session.record={"status":"PASS","outputs":{"summary":"MOCK"},"measured_summary":row}
        session.path=self.root/"sidecar.json"
        session.last_windows=record["phase_windows"]
        session.gpu=types.SimpleNamespace(snapshot=lambda _:record["gpu_load"])
        session.save()
        saved=json.loads(session.path.read_text())
        self.assertEqual("FAIL",saved["status"])
        self.assertIn("cleanup",saved["error"])


class CacheTests(Fixture):
    def preparation(self):
        root=self.root/"cuda";work=root/"toy";work.mkdir(parents=True)
        (work/"toy.cu").write_text("__global__ void toy() {}\n")
        (work/"toy.cuh").write_text("#define VALUE 1\n")
        include=self.root/"toolkit/include";include.mkdir(parents=True)
        for name in ["cuda.h","cuda_runtime.h","cuda_runtime_api.h"]:(include/name).write_text("// MOCK header\n")
        spec=base.BenchmarkSpec("toy","toy","toy",(("{nvcc}","-O3","toy.cu","-o","toy"),),(),("toy.cu",),("toy.cu",),(".",),(),(),{},
             tuple(base.InputSpec(f"SIZE_{i:02d}",f"SIZE_{i:02d}",(str(i),),1,{"N":i},(),{}) for i in range(1,21)),True,"")
        args=cuda.parse_args(["--benchmark","toy","--collect-static","--preparation-cache-dir",str(self.root/"cache")])
        context={"nvcc":sys.executable,"cuda_arch":"sm_86","cuda_dir":str(include.parent),"cuda_include_dir":str(include),"cuda_lib_dir":str(include.parent/"lib64")}
        def build(spec,*unused):
            (work/"toy").write_bytes(b"MOCK CUDA executable; never run")
            return True,"MOCK nvcc -O3",[]
        def static(*args):
            (args[-1]/"host.ll").write_text("; MOCK host-only IR\n")
            row=cuda.empty_static("success");row.update(static_source_count=1,static_source_success_count=1,cuda_static_source_metrics_success=True)
            return row,[]
        module=types.SimpleNamespace(__file__=cuda.__file__,SCRIPT_DIR=cuda.SCRIPT_DIR,base=base,CUDA_STATIC_FIELDS=cuda.CUDA_STATIC_FIELDS,
            _full_version=lambda command:"MOCK tool",build_benchmark=mock.Mock(side_effect=build),collect_static_features=mock.Mock(side_effect=static))
        device=types.SimpleNamespace(uuid="GPU-mock")
        return module,spec,args,root,context,device

    def test_twenty_runtime_inputs_share_one_static_and_binary_preparation(self):
        module,spec,args,root,context,device=self.preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            outputs=[cache.get_or_prepare(module,spec,args,root,context,device) for _ in range(4)]
        module.build_benchmark.assert_called_once();module.collect_static_features.assert_called_once()
        self.assertFalse(outputs[0][1]["cache_hit"])
        self.assertTrue(outputs[-1][1]["cache_hit"])
        self.assertEqual(20,len(args._preparation_provenance))
        receipt=json.loads(Path(outputs[0][1]["manifest_file"]).read_text())
        self.assertIn("host-only",receipt["fingerprint"]["generic_ir_scope"])
        self.assertFalse(receipt["contains_runtime_or_load_observations"])

    def test_source_arch_toolkit_and_gpu_identity_are_in_cache_key_not_runtime_size(self):
        module,spec,args,root,context,device=self.preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            key=cache.signature(module,spec,args,root,context,device)[0]
            same=cache.signature(module,replace(spec,inputs=(replace(spec.inputs[0],args=("999",)),)),args,root,context,device)[0]
            arch=cache.signature(module,spec,args,root,{**context,"cuda_arch":"sm_89"},device)[0]
            other=cache.signature(module,spec,args,root,context,types.SimpleNamespace(uuid="GPU-other"))[0]
            (root/"toy/toy.cuh").write_text("#define VALUE 2\n")
            changed=cache.signature(module,spec,args,root,context,device)[0]
        self.assertEqual(key,same)
        self.assertEqual(4,len({key,arch,other,changed}))

    def test_changed_artifact_or_actual_binary_is_rejected_without_overwrite(self):
        module,spec,args,root,context,device=self.preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            _,meta=cache.get_or_prepare(module,spec,args,root,context,device)
            (root/"toy/toy").write_bytes(b"changed by user")
            with self.assertRaisesRegex(RuntimeError,"Actual CUDA executable"):
                cache.get_or_prepare(module,spec,args,root,context,device)
            (Path(meta["manifest_file"]).parent/"host.ll").write_text("corrupt")
            with self.assertRaisesRegex(RuntimeError,"missing/changed"):
                cache.get_or_prepare(module,spec,args,root,context,device)
        self.assertEqual(b"changed by user",(root/"toy/toy").read_bytes())


class RunnerTests(Fixture):
    def test_pilot36_and_batch1760_cpu_idle_gpu_sampling_and_no_pressure_in_plan(self):
        for stage,count,programs in [("pilot",36,3),("batch",1760,22)]:
            root=self.root/stage
            args=runner.parser().parse_args(["--stage",stage])
            with mock.patch.object(runner.common.subprocess,"Popen") as spawn:
                manifest=runner.create_manifest(args,root)
            spawn.assert_not_called()
            self.assertEqual(count,len(manifest["trials"]))
            self.assertEqual(programs,manifest["preparation_cache"]["expected_unique_preparations"])
            self.assertEqual(base.cpu_runtime.RESOURCE_TARGETS,manifest["process_resource_supervision"]["targets"])
            self.assertEqual(count,len({t["trial_id"] for t in manifest["trials"]}))
            for trial in manifest["trials"]:
                config=host.read_config(trial["config_path"])
                self.assertEqual("idle",config["load_type"])
                self.assertEqual(0,config["target_cpu_fraction"])
                self.assertFalse(config["permission_confirmed"])
                self.assertEqual(trial["profile_id"],gpu.read_config(trial["gpu_config_path"])["profile_id"])
                command=trial["command"]
                self.assertIn("--gpu-load-config",command)
                self.assertIn("--collect-static",command)
                self.assertNotIn("--allow-busy-gpu",command)
                self.assertEqual("0.5",command[command.index("--gpu-sample-interval")+1])

    def test_batch_gate_rejects_cpu_pilot_wrong_gpu_and_policy(self):
        args=runner.parser().parse_args(["--stage","batch","--pilot-validation",str(self.root/"pilot.json")])
        pilot={"status":"PASS","stage":"pilot","dataset":"openmp","gpu_four_levels_validated":True,"validated_configurations":36,
               "expected_configurations":36,"preparation_cache_kinds":["rodinia_cuda_v1"],"hostname":runner.socket.gethostname(),
               "gpu_uuids":["GPU-mock"],"temperature_policies":["abort"]}
        pilot["measurement_policy"]={"maximum_runs":20,"minimum_runs":2,"gpu_min_poll_observations_per_phase":3,"gpu_short_missing_permitted":False}
        pilot["experiment_source_sha256"]={str(gpu.DEFINITIONS):gpu.digest(gpu.DEFINITIONS)}
        host.write_json(args.pilot_validation,pilot)
        with self.assertRaisesRegex(RuntimeError,"36-trial"):
            runner.batch_gate(args,{"gpu_uuid":"GPU-mock"})
        pilot["dataset"]="cuda";host.write_json(args.pilot_validation,pilot)
        with self.assertRaisesRegex(RuntimeError,"five host-process"):
            runner.batch_gate(args,{"gpu_uuid":"GPU-mock"})
        pilot.update(process_resource_label_configurations=36,process_resource_targets=base.cpu_runtime.RESOURCE_TARGETS)
        host.write_json(args.pilot_validation,pilot)
        runner.batch_gate(args,{"gpu_uuid":"GPU-mock"})
        with self.assertRaisesRegex(RuntimeError,"GPU/host"):
            runner.batch_gate(args,{"gpu_uuid":"GPU-other"})
        args.temperature_policy="record-only"
        with self.assertRaisesRegex(RuntimeError,"temperature"):
            runner.batch_gate(args,{"gpu_uuid":"GPU-mock"})

    def test_gpu_coverage_extends_real_invocations_not_idle_padding(self):
        clock=fixtures.FakeClock()
        guard=fixtures.FakeGuard(clock)
        guard.measurement_ready=lambda start,end:end-start>=.005
        spec=base.BenchmarkSpec("toy","toy","toy",(),(),(),(),(),(),(),{},(),True,"")
        item=base.InputSpec("SIZE_01","SIZE_01",(),1,{},(),{})
        def native(*args):
            clock.sleep(.001)
            return base.ProcessResult([],0,.001,False,"",{})
        with mock.patch.object(base.time,"time",side_effect=clock.now),mock.patch.object(base.time,"sleep",side_effect=clock.sleep), \
             mock.patch.object(base,"run_process",side_effect=native):
            windows={}
            summary,runs,errors=base.measure_input("mock",spec,item,self.root,1,5,.003,2,300,windows,6,guard,20)
        self.assertFalse(errors)
        self.assertGreaterEqual(summary["measured_runs"],5)
        self.assertLessEqual(summary["measured_runs"],6)
        self.assertAlmostEqual(summary["measurement_seconds_actual"],windows["during"][1]-windows["during"][0])

    def test_cuda_parser_wont_allow_busy_gpu_bypass_for_owned_gpu_load(self):
        with self.assertRaises(SystemExit):
            cuda.parse_args(["--gpu-load-config","mock.json","--load-config","host.json","--allow-busy-gpu"])

    def test_nonexecuting_cli_does_not_query_or_compile(self):
        with mock.patch.object(runner,"preflight") as preflight,mock.patch.object(gpu,"build_worker") as compile, \
             mock.patch.object(runner.common.subprocess,"Popen") as spawn:
            self.assertEqual(0,runner.main(["--output-dir",str(self.root/"plan")]))
        preflight.assert_not_called();compile.assert_not_called();spawn.assert_not_called()

    def test_readonly_linux_preflight_checks_tools_inputs_and_gpu_without_compiling(self):
        args=runner.parser().parse_args([])
        device=cuda.GpuDevice(0,"GPU-mock","MOCK","8.6",4*host.GIB,"MOCK",None)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner.platform,"system",return_value="Linux"))
            stack.enter_context(mock.patch.object(runner,"cuda_pipeline",return_value=cuda))
            stack.enter_context(mock.patch.object(runner.shutil,"which",return_value=sys.executable))
            stack.enter_context(mock.patch.object(cuda,"query_gpu_devices",return_value=[device]))
            stack.enter_context(mock.patch.object(cuda,"validate_nvcc_arch",return_value=""))
            stack.enter_context(mock.patch.object(cuda,"_full_version",return_value="CUDA release 11.8, V11.8.89"))
            stack.enter_context(mock.patch.object(gpu,"preflight",return_value={"gpu_uuid":"GPU-mock"}))
            stack.enter_context(mock.patch.object(host,"monitoring_preflight",return_value={"scrape_interval_seconds":1}))
            stack.enter_context(mock.patch.object(cuda.base,"validate_paths",return_value=[]))
            compile=stack.enter_context(mock.patch.object(gpu,"build_worker"))
            spawn=stack.enter_context(mock.patch.object(runner.common.subprocess,"Popen"))
            report=runner.preflight(args)
        self.assertEqual(22,report["benchmark_count"])
        self.assertFalse(report["gpu_load_started"])
        self.assertFalse(report["worker_compiled"])
        compile.assert_not_called();spawn.assert_not_called()

    def test_failure_then_resume_skips_completed_and_preserves_failed_log(self):
        root=self.root/"execution"
        args=runner.parser().parse_args(["--execute","--permission-confirmed"])
        manifest=runner.create_manifest(args,root)
        manifest["trials"]=manifest["trials"][:3]
        manifest["gpu_uuid"]="GPU-mock"
        def native(trial,timeout):
            path=Path(trial["directory"])/"pipeline.log"
            path.write_text("MOCK attempt "+trial["trial_id"])
            return 1 if trial is manifest["trials"][1] else 0
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner.signal,"SIGHUP",1,create=True))
            stack.enter_context(mock.patch.object(runner.signal,"getsignal"))
            stack.enter_context(mock.patch.object(runner.signal,"signal"))
            stack.enter_context(mock.patch.object(runner.shutil,"disk_usage",return_value=types.SimpleNamespace(free=10*host.GIB)))
            stack.enter_context(mock.patch.object(checker,"validate_trial",return_value={"gpu_uuid":"GPU-mock"}))
            process=stack.enter_context(mock.patch.object(runner.common,"run_trial",side_effect=native))
            with self.assertRaisesRegex(RuntimeError,"Trial failed"):
                runner.run(args,root,manifest)
            self.assertEqual(2,process.call_count)
            path=root/"experiment_manifest.json"
            before=path.read_bytes()
            first=deepcopy(manifest["trials"][0])
            failed_log=Path(manifest["trials"][1]["directory"])/"pipeline.log"
            log_bytes=failed_log.read_bytes()
            stack.enter_context(mock.patch.object(runner.platform,"system",return_value="Linux"))
            stack.enter_context(mock.patch.object(runner,"preflight",return_value={"gpu_uuid":"GPU-mock"}))
            stack.enter_context(mock.patch.object(runner,"verify_resume_preparations"))
            process.side_effect=lambda *args:0
            process.reset_mock()
            validate=stack.enter_context(mock.patch.object(checker,"validate",return_value={"status":"PASS"}))
            self.assertEqual(0,runner.main(["--resume","--execute","--permission-confirmed","--output-dir",str(root)]))
            self.assertEqual(2,process.call_count)
            validate.assert_called_once()
        after=json.loads(path.read_text())
        self.assertEqual(first,after["trials"][0])
        self.assertEqual("PASS",after["status"])
        self.assertEqual(before,next(root.glob("manifest_before_resume_*.json")).read_bytes())
        self.assertEqual(log_bytes,failed_log.read_bytes())
        self.assertNotEqual(manifest["trials"][1]["results_dir"],after["trials"][1]["results_dir"])


class ValidatorTests(Fixture):
    rewrite=fast.FastValidationTests.rewrite

    def trial(self):
        trial,record,sidecar=fast.FastValidationTests.make_trial(self)
        config={**record["config"],"profile_id":"GPU_LOW","load_type":"idle","target_cpu_fraction":0,"maximum_measured_runs":20}
        record["config"]=config
        host.write_json(trial["config_path"],config)
        trial.update(config_sha256=gpu.digest(trial["config_path"]),profile_id="GPU_LOW",load_family="gpu")
        module,spec,args,root,context,device=CacheTests.preparation(self)
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            payload,_=cache.get_or_prepare(module,spec,args,root,context,device)
        record["preparation"]=args._preparation_provenance[("toy","SIZE_01")]
        gpu_record,gpu_row=self.gpu_record()
        trial.update(gpu_config_path=str(self.root/"gpu.json"),gpu_config_sha256=gpu.digest(self.root/"gpu.json"))
        record["gpu_load"]={**gpu_record["gpu_load"],"config_file":trial["gpu_config_path"],"config_sha256":trial["gpu_config_sha256"]}
        path=Path(record["outputs"]["summary_file"])
        row=checker.rows(path)[0]
        row.update(parallel_model="CUDA",host_cpu_usage_pct_before_mean=0,gpu_idle_preflight_passed=True,gpu_three_phase_collection_success=True,
                   **gpu_row,**payload["static_features"])
        details=checker.rows(record["outputs"]["runs_file"])
        for index,detail in enumerate(details):
            detail.update({**base.cpu_runtime.empty_metrics(),"process_io_collection_status":"ok",
                           "process_metrics_backend":"linux-process-probe-v4","process_cpu_user_sec":(index+1)*.1,
                           "process_cpu_system_sec":0.,"process_max_rss_bytes":(index+1)*4096,
                           **{field:index*10 for field in base.cpu_runtime.IO_NUMERIC_FIELDS}})
        row.update(base.cpu_runtime.summarize([{**detail,"success":True} for detail in details]))
        row.update(process_metrics_backend="linux-process-probe-v4",
                   **{field:sum(detail[field] for detail in details)/len(details) for field in base.cpu_runtime.CPU_RESOURCE_FIELDS})
        with Path(record["outputs"]["runs_file"]).open("w",encoding="utf-8",newline="") as stream:
            import csv
            writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
        with path.open("w",encoding="utf-8",newline="") as stream:
            import csv
            writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
        host.write_json(sidecar,record)
        return trial,record,sidecar

    def test_full_trial_validator_accepts_cuda_boolean_source_fields_and_gpu_evidence(self):
        trial,record,sidecar=self.trial()
        result=checker.validate_trial(trial)
        self.assertEqual("GPU-mock",result["gpu_uuid"])
        self.assertEqual("rodinia_cuda_v1",result["preparation_cache_kind"])
        self.assertEqual(25,result["gpu_before_mean_pct"])
        self.assertTrue(result["process_resource_labels_validated"])
        self.assertEqual(5,result["process_io_rchar_bytes"])
        record["gpu_load"]["phase_samples"]["during"]=[]
        host.write_json(sidecar,record)
        with self.assertRaisesRegex(RuntimeError,"real observations"):
            checker.validate_trial(trial)

    def test_gpu_quartet_full_report_exports_lossless_total_table(self):
        trial,record,_=self.trial()
        original_row=checker.rows(record["outputs"]["summary_file"])[0]
        rows=[];trials=[]
        definitions=json.loads(gpu.DEFINITIONS.read_text())
        for profile in definitions["profiles"]:
            t=deepcopy(trial);r=deepcopy(record);row=deepcopy(original_row)
            directory=self.root/profile["profile_id"]
            results=directory/"results";results.mkdir(parents=True)
            t.update(trial_id="gpu__toy__SIZE_01__"+profile["profile_id"],profile_id=profile["profile_id"],
                     config_path=str(directory/"host.json"),gpu_config_path=str(directory/"gpu.json"),directory=str(directory),results_dir=str(results))
            r["config"]["profile_id"]=profile["profile_id"]
            r["gpu_load"]["config"].update(profile)
            host.write_json(t["config_path"],r["config"]);host.write_json(t["gpu_config_path"],r["gpu_load"]["config"])
            t.update(config_sha256=gpu.digest(t["config_path"]),gpu_config_sha256=gpu.digest(t["gpu_config_path"]))
            r["gpu_load"].update(config_sha256=t["gpu_config_sha256"],config_file=t["gpu_config_path"])
            for phase in ["before","during","after"]:
                row[f"gpu_utilization_pct_{phase}_mean"]=profile["target_gpu_pct"]
                for sample in r["gpu_load"]["phase_samples"][phase]:sample["values"]["gpu_utilization_pct"]=profile["target_gpu_pct"]
            summary=results/"summary.csv";runs=results/"runs.csv"
            import csv
            with summary.open("w",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            runs.write_bytes(Path(record["outputs"]["runs_file"]).read_bytes())
            r["outputs"].update(summary_file=str(summary),runs_file=str(runs))
            host.write_json(results/"load_records/fixture.json",r)
            trials.append(t);rows.append(checker.rows(summary)[0])
        path=self.root/"manifest.json"
        host.write_json(path,{"dataset":"cuda","stage":"batch","status":"VALIDATING","gpu_uuid":"GPU-mock","load_families":["gpu"],
            "expected_configurations":4,"expected_program_count":1,"expected_input_profiles":["SIZE_01"],"execution_host":"MOCK",
            "created_at_utc":"MOCK","trials":trials,"preparation_cache":{"enabled":True,"expected_unique_preparations":1}})
        report=checker.validate(path)
        self.assertEqual("PASS",report["status"],report["errors"])
        self.assertTrue(report["gpu_four_levels_validated"])
        self.assertEqual(["GPU-mock"],report["gpu_uuids"])
        exported=checker.rows(report["sample_table"])
        self.assertEqual(4,len(exported))
        self.assertEqual(4,len({r["sample_id"] for r in exported}))
        for raw,export in zip(rows,exported):self.assertEqual(raw,{k:export[k] for k in raw})
        self.assertEqual(4,report["process_resource_label_configurations"])
        self.assertEqual(base.cpu_runtime.RESOURCE_TARGETS,report["process_resource_targets"])

    def test_validator_rejects_missing_io_column_despite_complete_gpu_monitoring(self):
        trial,record,_=self.trial()
        path=Path(record["outputs"]["summary_file"])
        rows=checker.rows(path)
        rows[0].pop("process_io_wchar_bytes")
        import csv
        with path.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,"process_io_wchar_bytes"):
            checker.validate_trial(trial)


class PipelineIntegrationTests(Fixture):
    def short_host_monitoring(self):
        record={"config":{"collection_protocol":"fast-v2","min_raw_scrapes":3},
                "monitoring_preflight":{"scrape_interval_seconds":1.},
                "phase_windows":{"during":(17.,20.)},
                "phase_quality":{"before":{"status":"PASS"},"during":{"status":"MISSING_SHORT_WINDOW"},
                                 "after":{"status":"PASS"}}}
        row={"prometheus_before_collection_success":True,"prometheus_after_collection_success":True,
             "prometheus_during_collection_success":False,"prometheus_three_phase_collection_success":False,
             "prometheus_three_phase_collection_error":"during: insufficient raw scrapes"}
        return record,row

    def test_host_monitoring_exception_requires_unique_recorded_short_window(self):
        record,row=self.short_host_monitoring()
        session=types.SimpleNamespace(identity={"program_id":"toy","input_id":"SIZE_01"},record=record)
        args=types.SimpleNamespace(load_config=Path("mock.json"),_load_sessions=[session])
        self.assertTrue(cuda.host_monitoring_accepted(args,"toy","SIZE_01",row))
        for change in ["strict","long","before","after","during","fabricated","identity","ambiguous","unrecorded","legacy"]:
            modified=deepcopy(args);observed=deepcopy(row)
            r=modified._load_sessions[0].record
            if change=="strict":r["config"]["collection_protocol"]="strict-v1"
            if change=="long":r["phase_windows"]["during"]=(17.,23.)
            if change in {"before","after","during"}:r["phase_quality"][change]["status"]="FAIL"
            if change=="fabricated":observed["host_cpu_usage_pct_during_mean"]=0
            if change=="identity":modified._load_sessions[0].identity["input_id"]="SIZE_02"
            if change=="ambiguous":modified._load_sessions*=2
            if change=="unrecorded":modified._load_sessions=[]
            if change=="legacy":modified.load_config=None
            with self.subTest(change=change):
                self.assertFalse(cuda.host_monitoring_accepted(modified,"toy","SIZE_01",observed))

    def run_cached_cuda_main(self,host_result=None,load_record=None,gpu_success=True,io_complete=True):
        root=self.root/"rodinia";work=root/"cuda/toy";work.mkdir(parents=True)
        (work/"toy.cu").write_text("// MOCK benchmark\n")
        inputs=tuple(base.InputSpec(f"SIZE_{i:02d}",f"SIZE_{i:02d}",(str(i),),1,{"N":i},(),{}) for i in [1,2])
        spec=base.BenchmarkSpec("toy","toy","toy",(),(),("toy.cu",),("toy.cu",),(),(),(),{},inputs,True,"")
        device=cuda.GpuDevice(0,"GPU-mock","MOCK","8.6",4*host.GIB,"MOCK",None)
        class Sampler:
            def __init__(self,*args):pass
            def start(self):pass
            def stop(self):pass
            def wait_for_first_sample(self,*args):return True
            def snapshot(self):return [cuda.GpuSample(t,{"gpu_utilization_pct":25,"gpu_memory_used_bytes":1024}) for t in [13,20,26]],[]
        def prepare(module,spec,args,*unused):
            args._preparation_provenance={("toy",i.input_id):{"MOCK_ONLY":True} for i in inputs}
            static=cuda.empty_static("success");static["cuda_static_source_metrics_success"]=True
            return {"build_success":True,"build_command":"MOCK build","build_errors":[],"static_features":static,"static_errors":[]},{"executable_sha256":"MOCK_SHA"}
        def measure(*args):
            self.assertEqual(20,args[12]);self.assertEqual("MOCK_SHA",args[13])
            self.assertIs(cache.verify_execution_file,args[14])
            args[9].update(before=(10,16),during=(17,23),after=(23,29))
            details=[]
            for index,phase in enumerate(["warmup","measure","measure"]):
                detail={**base.cpu_runtime.empty_metrics(),"program_id":"toy","input_id":args[2].input_id,
                        "run_id":args[2].input_id+str(index),"phase":phase,"success":True,
                        "process_metrics_backend":"linux-process-probe-v4","process_cpu_user_sec":.1,
                        "process_cpu_system_sec":0.,"process_max_rss_bytes":4096,"process_io_collection_status":"ok"}
                detail.update({field:(999 if phase=="warmup" else index*10) for field in base.cpu_runtime.IO_NUMERIC_FIELDS})
                if not io_complete and index==2:
                    detail.update(base.cpu_runtime.empty_metrics("permission_denied"))
                details.append(detail)
            summary={"run_success":True,"measured_runs":2,"runtime_sec_median":int(args[2].parameters["N"]),
                     "process_metrics_backend":"linux-process-probe-v4","process_cpu_user_sec":.1,
                     "process_cpu_system_sec":0.,"process_max_rss_bytes":4096,**base.cpu_runtime.summarize(details)}
            return summary,details,[]
        def load_session(args,core,identity,*unused):
            args._load_sessions.append(types.SimpleNamespace(identity=identity,record=deepcopy(load_record)))
            return nullcontext(None)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(cuda,"os",types.SimpleNamespace(name="posix",environ=dict(os.environ))))
            stack.enter_context(mock.patch.object(cuda.base,"load_manifest",return_value=[spec]))
            stack.enter_context(mock.patch.object(cuda.shutil,"which",return_value=sys.executable))
            stack.enter_context(mock.patch.object(cuda,"query_gpu_devices",return_value=[device]))
            stack.enter_context(mock.patch.object(cuda,"query_compute_processes",return_value=[]))
            stack.enter_context(mock.patch.object(cuda,"_full_version",return_value="CUDA compilation tools, release 11.8"))
            stack.enter_context(mock.patch.object(cuda,"validate_nvcc_arch",return_value=""))
            stack.enter_context(mock.patch.object(cuda,"infer_cuda_dir",return_value=self.root/"toolkit"))
            stack.enter_context(mock.patch.object(cuda,"configure_cuda_environment"))
            stack.enter_context(mock.patch.object(cuda.base,"ensure_process_probe",return_value=Path("MOCK")))
            stack.enter_context(mock.patch.object(cuda.base,"validate_paths",return_value=[]))
            stack.enter_context(mock.patch.object(cuda,"collect_environment",return_value={"gpu_uuid":"GPU-mock"}))
            stack.enter_context(mock.patch.object(cuda,"NvidiaSmiSampler",Sampler))
            prepared=stack.enter_context(mock.patch.object(cache,"get_or_prepare",side_effect=prepare))
            measured=stack.enter_context(mock.patch.object(cuda.base,"measure_input",side_effect=measure))
            rebuilt=stack.enter_context(mock.patch.object(cuda,"build_benchmark"))
            extracted=stack.enter_context(mock.patch.object(cuda,"collect_static_features"))
            argv=["--benchmark","toy","--input-profile","all","--rodinia-root",str(root),"--collect-static",
                  "--results-dir",str(self.root/"out"),"--preparation-cache-dir",str(self.root/"cache"),"--max-measured-runs","20",
                  "--prometheus-exporter","node" if host_result is not None else "off"]
            if host_result is not None:
                argv.extend(["--prometheus-job","node","--prometheus-instance","127.0.0.1:9100"])
                stack.enter_context(mock.patch.object(cuda.base.pipeline_core(),"collect_prometheus_three_phase_metrics",return_value=host_result))
                stack.enter_context(mock.patch.object(cuda.base.resource_load,"phase_guard",return_value=None))
                stack.enter_context(mock.patch.object(cuda.time,"sleep"))
            if load_record is not None:
                argv.extend(["--load-config",str(self.config(collection_protocol="fast-v2",minimum_measured_runs=2,maximum_measured_runs=20)),
                             "--min-measured-runs","2","--gpu-load-config",str(self.gpu_config())])
                stack.enter_context(mock.patch.object(cuda.gpu_resource_load,"session",side_effect=load_session))
                stack.enter_context(mock.patch.object(cuda.base.resource_load,"finalize"))
            if not gpu_success:
                stack.enter_context(mock.patch.object(cuda,"summarize_gpu_samples",return_value={"gpu_three_phase_collection_success":False,
                    "gpu_three_phase_collection_error":"during: no GPU observations"}))
            code=cuda.main(argv)
        prepared.assert_called_once();self.assertEqual(2,measured.call_count)
        rebuilt.assert_not_called();extracted.assert_not_called()
        environment=json.loads(next((self.root/"out").glob("*environment*.json")).read_text())
        self.assertFalse(any(k.startswith("_") for k in environment["arguments"]))
        self.assertEqual(base.cpu_runtime.schema_description(),environment["cpu_runtime_parameter_schema"])
        for row in checker.rows(next((self.root/"out").glob("*summary*.csv"))):
            self.assertTrue(set(base.cpu_runtime.SUMMARY_FIELDS).issubset(row))
            self.assertEqual("15.0" if io_complete else "",row["process_io_rchar_bytes"])
            self.assertEqual("ok" if io_complete else "partial",row["process_io_collection_status"])
        self.assertTrue(all("process_io_wchar_bytes" in r for r in checker.rows(next((self.root/"out").glob("*runs*.csv")))))
        return code

    def test_cache_cap_and_environment_serialization_through_cuda_main(self):
        self.assertEqual(0,self.run_cached_cuda_main())

    def test_cuda_main_accepts_recorded_fast_host_short_window(self):
        record,row=self.short_host_monitoring()
        self.assertEqual(0,self.run_cached_cuda_main(row,record))

    def test_cuda_main_still_rejects_missing_gpu_during_samples(self):
        record,row=self.short_host_monitoring()
        self.assertEqual(1,self.run_cached_cuda_main(row,record,gpu_success=False))

    def test_cuda_main_without_load_record_still_rejects_incomplete_host_monitoring(self):
        _,row=self.short_host_monitoring()
        self.assertEqual(1,self.run_cached_cuda_main(row))

    def test_missing_io_fails_controlled_cuda_even_when_execution_and_gpu_succeed(self):
        record,row=self.short_host_monitoring()
        self.assertEqual(1,self.run_cached_cuda_main(row,record,io_complete=False))

    def test_not_run_cuda_has_same_missing_io_semantics_as_cpu_pipelines(self):
        empty=cuda._not_run_metrics(6,.5)
        for field in base.cpu_runtime.IO_NUMERIC_FIELDS:
            self.assertEqual("",empty[field])
        self.assertEqual("not_run",empty["process_io_collection_status"])
        self.assertEqual(0,empty["process_io_sampled_runs"])


class PreservationAuditTests(Fixture):
    def test_only_explicit_cuda_io_summary_addition_is_allowed(self):
        import ast
        import check_load_aware_pipeline as audit
        expression=lambda code:ast.parse(code,mode="eval").body
        old=expression("['runtime', *base.PROCESS_FIELDS, 'end']")
        new=expression("['runtime', *base.PROCESS_FIELDS, *base.cpu_runtime.SUMMARY_FIELDS, 'end']")
        result=audit.csv_declaration_check("pipeline_rodinia_cuda.py","SUMMARY_FIELDS",old,new)
        self.assertFalse(result["unchanged"])
        self.assertTrue(result["original_fields_preserved"])
        self.assertEqual("base.cpu_runtime.SUMMARY_FIELDS",result["approved_extension"])
        for filename,label,candidate in [("pipeline_rodinia.py","SUMMARY_FIELDS",new),
                                         ("pipeline_rodinia_cuda.py","GPU_RESULT_FIELDS",new),
                                         ("pipeline_rodinia_cuda.py","SUMMARY_FIELDS",expression("['changed', *base.PROCESS_FIELDS, *base.cpu_runtime.SUMMARY_FIELDS, 'end']"))]:
            with self.subTest(filename=filename,label=label):
                self.assertFalse(audit.csv_declaration_check(filename,label,old,candidate)["original_fields_preserved"])
