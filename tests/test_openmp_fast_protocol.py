from __future__ import annotations

from dataclasses import replace
from contextlib import ExitStack, nullcontext
import csv
import hashlib
import json
from pathlib import Path
import sys
import types
from unittest import mock

import test_resource_load as fixtures
import test_load_fast_protocol as fast_fixtures
import openmp_preparation_cache as cache

omp, runner, load, checker = fixtures.omp, fixtures.runner, fixtures.load, fixtures.checker


class OpenMPCacheTests(fixtures.Fixture):
    def setup_preparation(self):
        root=self.root/"sources"
        work=root/"toy"
        work.mkdir(parents=True)
        (work/"toy.c").write_text("int main(void){return 0;}\n")
        (work/"toy.h").write_text("#define VALUE 1\n")
        (work/"Makefile").write_text("toy: toy.c\n\tcc toy.c -o toy\n")
        inputs=tuple(omp.InputSpec(f"SIZE_{i:02d}",f"SIZE_{i:02d}",(str(i),),4,{"N":i},(),{}) for i in range(1,21))
        spec=omp.BenchmarkSpec("toy","toy","toy",(("make","toy"),),(),("toy.c",),("toy.c",),(".",),("-fopenmp",),(),{},inputs,True,"")
        args=omp.parse_args(["--benchmark","bfs","--preparation-cache-dir",str(self.root/"cache")])
        static_core=types.SimpleNamespace(__file__=fixtures.core.__file__,COMPAT_INCLUDE_DIR=self.root/"compat",llvm_opt_version=lambda:"MOCK opt")
        core=types.SimpleNamespace(__file__=omp.__file__,pipeline_core=lambda:static_core,_compiler_version=lambda _:"MOCK compiler",
                                   AST_FIELDS=omp.AST_FIELDS,CFG_FIELDS=omp.CFG_FIELDS,IR_FIELDS=omp.IR_FIELDS)
        def build(forced,*unused):
            self.assertIn("-B",forced.build_commands[0])
            (work/"toy").write_bytes(b"MOCK binary; never executed")
            return True,"make toy -B",[]
        def static(*arguments):
            target=arguments[-1]
            (target/"ast").mkdir()
            (target/"ast/toy.ast").write_bytes(b"MOCK static AST")
            row=omp.empty_static("success")
            row["static_source_count"]=row["static_source_success_count"]=1
            return row,[]
        core.build_benchmark=mock.Mock(side_effect=build)
        core.collect_static_features=mock.Mock(side_effect=static)
        return core,spec,args,root

    def test_twenty_inputs_and_four_loads_share_one_preparation_but_bind_inputs_separately(self):
        core,spec,args,root=self.setup_preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            outputs=[cache.get_or_prepare(core,spec,args,root) for _ in range(4)]
        core.build_benchmark.assert_called_once()
        core.collect_static_features.assert_called_once()
        self.assertEqual(1,len(list(args.preparation_cache_dir.glob("*/preparation.json"))))
        self.assertFalse(outputs[0][1]["cache_hit"])
        self.assertTrue(all(meta["cache_hit"] for _,meta in outputs[1:]))
        self.assertEqual(20,len(args._preparation_provenance))
        self.assertEqual("SIZE_20",args._preparation_provenance[("toy","SIZE_20")]["input_profile"])
        saved=json.loads(Path(outputs[0][1]["manifest_file"]).read_text())
        self.assertFalse(saved["contains_runtime_or_load_observations"])
        self.assertNotIn("inputs",saved["fingerprint"]["benchmark"])

    def test_input_changes_do_not_change_compilation_key_but_sources_flags_and_makefile_do(self):
        core,spec,args,root=self.setup_preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            first,_=cache.signature(core,spec,args,root)
            changed,_=cache.signature(core,replace(spec,inputs=(replace(spec.inputs[0],args=("10000",),threads=8),)),args,root)
            self.assertEqual(first,changed)
            (root/"toy/toy.h").write_text("#define VALUE 2\n")
            header,_=cache.signature(core,spec,args,root)
            flag,_=cache.signature(core,replace(spec,static_flags=("-fopenmp","-DOTHER",)),args,root)
            (root/"toy/Makefile").write_text("toy: toy.c\n\tcc -O1 toy.c -o toy\n")
            makefile,_=cache.signature(core,spec,args,root)
        self.assertEqual(4,len({first,header,flag,makefile}))

    def test_cache_artifact_corruption_is_not_silently_rebuilt(self):
        core,spec,args,root=self.setup_preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            _,meta=cache.get_or_prepare(core,spec,args,root)
            (Path(meta["manifest_file"]).parent/"ast/toy.ast").write_text("corrupt")
            with self.assertRaisesRegex(RuntimeError,"missing/changed"):
                cache.get_or_prepare(core,spec,args,root)
        core.build_benchmark.assert_called_once()

    def test_actual_executable_change_aborts_without_overwriting_user_binary(self):
        core,spec,args,root=self.setup_preparation()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            cache.get_or_prepare(core,spec,args,root)
            (root/"toy/toy").write_bytes(b"user replacement")
            with self.assertRaisesRegex(RuntimeError,"Actual OpenMP executable"):
                cache.get_or_prepare(core,spec,args,root)
        self.assertEqual(b"user replacement",(root/"toy/toy").read_bytes())

    def test_cached_payload_corruption_and_make_dry_run_are_rejected(self):
        core,spec,args,root=self.setup_preparation()
        with mock.patch.dict(cache.os.environ,{"MAKEFLAGS":"--dry-run"}):
            with self.assertRaisesRegex(ValueError,"Unset MAKEFLAGS"):
                cache.get_or_prepare(core,spec,args,root)
        core.build_benchmark.assert_not_called()
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            _,meta=cache.get_or_prepare(core,spec,args,root)
            path=Path(meta["manifest_file"])
            record=json.loads(path.read_text())
            record["payload"]["static_features"][omp.AST_FIELDS[0]]=999
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError,"preparation identity"):
                cache.get_or_prepare(core,spec,args,root)

    def test_dynamic_values_cannot_enter_static_cache(self):
        core,spec,args,root=self.setup_preparation()
        core.collect_static_features=mock.Mock(return_value=({**omp.empty_static("success"),"runtime_sec_median":1},[]))
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable),self.assertRaisesRegex(RuntimeError,"static preparation"):
            cache.get_or_prepare(core,spec,args,root)
        self.assertFalse(list(args.preparation_cache_dir.glob("*/preparation.json")))

    def test_source_modified_by_build_is_not_published_under_stale_key(self):
        core,spec,args,root=self.setup_preparation()
        original=core.build_benchmark.side_effect
        def changed(*arguments):
            result=original(*arguments)
            (root/"toy/toy.h").write_text("#define VALUE 999\n")
            return result
        core.build_benchmark.side_effect=changed
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable),self.assertRaisesRegex(RuntimeError,"Build changed"):
            cache.get_or_prepare(core,spec,args,root)


class OpenMPFastMeasurementTests(fixtures.Fixture):
    def measure(self,durations,max_runs=10,expected=None):
        clock=fixtures.FakeClock()
        guard=fixtures.FakeGuard(clock)
        spec=omp.BenchmarkSpec("toy","toy","toy",(),(),(),(),(),(),(),{},(),True,"")
        item=omp.InputSpec("SIZE_01","SIZE_01",("123",),8,{},(),{})
        seen=[]
        def native(command,cwd,environment,*unused):
            duration=durations[min(len(seen),len(durations)-1)]
            seen.append(command)
            self.assertEqual(str(self.root/"toy/toy"),command[0])
            self.assertEqual("8",environment["OMP_NUM_THREADS"])
            self.assertEqual("FALSE",environment["OMP_DYNAMIC"])
            clock.sleep(duration)
            return omp.ProcessResult(command,0,duration,False,"",{})
        with mock.patch.object(omp.time,"time",side_effect=clock.now),mock.patch.object(omp.time,"sleep",side_effect=clock.sleep), \
             mock.patch.object(omp,"run_process",side_effect=native),mock.patch.object(cache,"verify_execution_file") as verify:
            windows={}
            summary,records,errors=omp.measure_input("mock",spec,item,self.root,1,5,3,2,300,windows,6,guard,max_runs,expected)
        return summary,records,errors,windows,guard,verify

    def test_short_inputs_stop_at_ten_without_idle_padding(self):
        summary,records,errors,windows,guard,_=self.measure([.001])
        self.assertFalse(errors)
        self.assertEqual(10,summary["measured_runs"])
        self.assertEqual(11,len(records))
        self.assertAlmostEqual(.01,windows["during"][1]-windows["during"][0])
        self.assertEqual("before",guard.events[0])
        self.assertEqual("warmup",guard.events[1])

    def test_long_inputs_stop_at_two_and_use_both_observations(self):
        summary,records,errors,windows,_,verify=self.measure([1,10,14],expected="MOCK")
        self.assertEqual(2,summary["measured_runs"])
        self.assertEqual(12,summary["runtime_sec_median"])
        self.assertEqual(24,windows["during"][1]-windows["during"][0])
        self.assertEqual(3,verify.call_count)

    def test_existing_uncapped_budget_mode_is_unchanged(self):
        summary,*_=self.measure([.25],max_runs=0)
        self.assertEqual(12,summary["measured_runs"])

    def test_bad_executable_is_rejected_before_target_spawn(self):
        spec=omp.BenchmarkSpec("toy","toy","toy",(),(),(),(),(),(),(),{},(),True,"")
        item=omp.InputSpec("SIZE_01","SIZE_01",(),4,{},(),{})
        with mock.patch.object(omp,"run_process") as native:
            with self.assertRaisesRegex(RuntimeError,"Actual OpenMP executable"):
                omp.measure_input("mock",spec,item,self.root,1,5,3,2,300,max_measured_runs=10,expected_executable_sha256="missing")
        native.assert_not_called()


class OpenMPFastLauncherTests(fixtures.Fixture):
    def test_full_plan_has_1520_trials_nineteen_preparations_and_unchanged_threads(self):
        args=runner.build_parser().parse_args(["--dataset","openmp","--stage","batch","--protocol","fast-v2"])
        with mock.patch.object(runner.subprocess,"Popen") as native:
            manifest=runner.create_manifest(args,self.root)
        native.assert_not_called()
        self.assertEqual(1520,len(manifest["trials"]))
        self.assertEqual(19,manifest["preparation_cache"]["expected_unique_preparations"])
        self.assertEqual("benchmark",manifest["preparation_cache"]["scope"])
        for trial in manifest["trials"]:
            command=trial["command"]
            self.assertEqual("10",command[command.index("--max-measured-runs")+1])
            self.assertIn("--preparation-cache-dir",command)
            self.assertNotIn("--allow-oversubscription",command)
            self.assertNotIn("--threads",command)

    def test_openmp_pilot_is_bfs_myocyte_three_sizes_four_loads(self):
        args=runner.build_parser().parse_args(["--dataset","openmp","--stage","pilot","--protocol","fast-v2"])
        manifest=runner.create_manifest(args,self.root)
        self.assertEqual(24,len(manifest["trials"]))
        self.assertEqual(2,manifest["preparation_cache"]["expected_unique_preparations"])
        self.assertEqual({"bfs","myocyte"},{t["program"] for t in manifest["trials"]})
        self.assertEqual({"SIZE_01","SIZE_10","SIZE_20"},{t["input_profile"] for t in manifest["trials"]})

    def test_polybench_pilot_cannot_unlock_new_openmp_integration(self):
        args=runner.build_parser().parse_args(["--dataset","openmp","--stage","batch","--protocol","fast-v2"])
        args.pilot_validation=self.root/"pilot.json"
        report={"status":"PASS","stage":"pilot","dataset":"polybench","hostname":runner.socket.gethostname(),
                "cpu_four_levels_validated":True,"collection_protocols":["fast-v2"],"temperature_policies":["abort"]}
        load.write_json(args.pilot_validation,report)
        with self.assertRaisesRegex(ValueError,"own successful OpenMP pilot"):
            runner.check_batch_gate(args)
        report["dataset"]="openmp"
        report.update(validated_configurations=24,expected_configurations=24,unique_preparations=2,preparation_cache_kinds=["rodinia_openmp_v1"])
        load.write_json(args.pilot_validation,report)
        runner.check_batch_gate(args)


class OpenMPFastValidationTests(fixtures.Fixture):
    rewrite=fast_fixtures.FastValidationTests.rewrite

    def make_trial(self):
        trial,record,sidecar=fast_fixtures.FastValidationTests.make_trial(self,short=True)
        core,spec,args,root=OpenMPCacheTests.setup_preparation(self)
        with mock.patch.object(cache.shutil,"which",return_value=sys.executable):
            payload,_=cache.get_or_prepare(core,spec,args,root)
        record["preparation"]=args._preparation_provenance[("toy","SIZE_01")]
        path=Path(record["outputs"]["summary_file"])
        row=checker.rows(path)[0]
        row.update(parallel_model="OpenMP",**payload["static_features"])
        row.update(process_cpu_user_sec_mean="0",process_max_rss_bytes_max="123456",process_io_wchar_bytes_mean="0")
        self.write_row(path,row)
        load.write_json(sidecar,record)
        return trial,record,sidecar

    @staticmethod
    def write_row(path,row):
        with path.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_validator_accepts_openmp_static_receipt_and_preserves_short_process_metrics(self):
        trial,record,_=self.make_trial()
        result=checker.validate_trial(trial)
        self.assertEqual("rodinia_openmp_v1",result["preparation_cache_kind"])
        self.assertEqual("benchmark",result["preparation_cache_scope"])
        self.assertEqual("MISSING_SHORT_WINDOW",result["during_monitoring_status"])
        original=Path(record["outputs"]["summary_file"]).read_bytes()
        manifest=self.root/"manifest.json"
        load.write_json(manifest,{"status":"VALIDATING","stage":"batch","dataset":"openmp",
            "created_at_utc":"MOCK_ONLY","execution_host":"MOCK_ONLY","trials":[trial],"load_families":["cpu"],
            "preparation_cache":{"enabled":True,"expected_unique_preparations":1}})
        first=checker.validate(manifest)
        self.assertEqual("PASS",first["status"],first["errors"])
        table=checker.rows(first["sample_table"])
        raw=checker.rows(record["outputs"]["summary_file"])[0]
        self.assertEqual(raw,{name:table[0][name] for name in raw})
        self.assertEqual("0",table[0]["process_io_wchar_bytes_mean"])
        self.assertEqual("",table[0]["host_cpu_usage_pct_during_mean"])
        self.assertEqual(1,first["unique_preparations"])
        self.assertEqual(1,first["sample_table_rows"])
        self.assertEqual(first["sample_table_sha256"],checker.validate(manifest)["sample_table_sha256"])
        self.assertEqual(original,Path(record["outputs"]["summary_file"]).read_bytes())

    def test_static_value_or_input_binding_tamper_is_rejected(self):
        trial,record,sidecar=self.make_trial()
        record["preparation"]["input_profile"]="SIZE_20"
        load.write_json(sidecar,record)
        with self.assertRaisesRegex(ValueError,"scope/input"):
            checker.validate_trial(trial)
        record["preparation"]["input_profile"]="SIZE_01"
        load.write_json(sidecar,record)
        path=Path(record["outputs"]["summary_file"])
        row=checker.rows(path)[0]
        row[omp.AST_FIELDS[0]]=999
        self.write_row(path,row)
        with self.assertRaisesRegex(ValueError,"Cached static value changed"):
            checker.validate_trial(trial)

    def test_aggregate_metadata_cannot_overwrite_observed_columns(self):
        trial,record,_=self.make_trial()
        path=Path(record["outputs"]["summary_file"])
        self.write_row(path,{"sample_id":"observed",**checker.rows(path)[0]})
        manifest={"created_at_utc":"MOCK_ONLY","execution_host":"MOCK_ONLY","trials":[trial]}
        with self.assertRaisesRegex(ValueError,"overwrite"):
            checker.write_openmp_samples(self.root,manifest,[{"trial_id":trial["trial_id"],"summary_file":str(path)}])
        self.assertFalse((self.root/"results.csv").exists())


class OpenMPMainIntegrationTests(fixtures.Fixture):
    def test_cached_preparation_still_runs_every_input_and_writes_serializable_environment(self):
        _,spec,_,source_root=OpenMPCacheTests.setup_preparation(self)
        spec=replace(spec,inputs=spec.inputs[:2])
        output=self.root/"output"
        argv=["--benchmark","toy","--input-profile","all","--rodinia-root",str(source_root.parent),
              "--results-dir",str(output),"--preparation-cache-dir",str(self.root/"cache"),"--max-measured-runs","10"]
        # The mocked platform only affects this module; pathlib must retain Windows semantics.
        source_root.rename(source_root.parent/"openmp")
        def validate(args,*unused):
            args._load_sessions=[object()]
        def prepare(module,spec,args,root):
            args._preparation_provenance={("toy",item.input_id):{"MOCK_ONLY":True} for item in spec.inputs}
            return {"build_success":True,"build_command":"MOCK make","build_errors":[],
                    "static_features":omp.empty_static("success"),"static_errors":[]},{"executable_sha256":"MOCK_SHA"}
        def measure(*arguments):
            item=arguments[2]
            self.assertEqual(10,arguments[12])
            self.assertEqual("MOCK_SHA",arguments[13])
            self.assertEqual(4,item.threads)
            return {"run_success":True,"runtime_sec_median":float(item.parameters["N"])},[],[]
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(omp,"os",types.SimpleNamespace(name="posix")))
            stack.enter_context(mock.patch.object(omp,"load_manifest",return_value=[spec]))
            stack.enter_context(mock.patch.object(omp,"ensure_process_probe",return_value=Path("MOCK_PROBE")))
            stack.enter_context(mock.patch.object(omp,"collect_environment",return_value={"cpu_logical_core_count":64}))
            stack.enter_context(mock.patch.object(omp,"validate_paths",return_value=[]))
            stack.enter_context(mock.patch.object(load,"validate_args",side_effect=validate))
            stack.enter_context(mock.patch.object(load,"session",side_effect=lambda *args:nullcontext(None)))
            stack.enter_context(mock.patch.object(load,"finalize"))
            prepared=stack.enter_context(mock.patch.object(cache,"get_or_prepare",side_effect=prepare))
            measured=stack.enter_context(mock.patch.object(omp,"measure_input",side_effect=measure))
            rebuild=stack.enter_context(mock.patch.object(omp,"build_benchmark"))
            extract=stack.enter_context(mock.patch.object(omp,"collect_static_features"))
            self.assertEqual(0,omp.main(argv))
        prepared.assert_called_once()
        self.assertEqual(2,measured.call_count)
        rebuild.assert_not_called()
        extract.assert_not_called()
        saved=json.loads(next(output.glob("*environment*.json")).read_text())
        self.assertFalse(any(name.startswith("_") for name in saved["arguments"]))
        rows=checker.rows(next(output.glob("*summary*.csv")))
        self.assertEqual(["1.0","2.0"],[row["runtime_sec_median"] for row in rows])
