from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "pipeline_rodinia_cuda.py"
MANIFEST_PATH = PROJECT_ROOT / "configs" / "rodinia_cuda.json"
RODINIA_ROOT = PROJECT_ROOT / "datasets" / "rodinia_3.1"


try:
    import prometheus_client  # noqa: F401
except ImportError:
    fake_prometheus = types.ModuleType("prometheus_client")

    class _Metric:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def labels(self, *args: object, **kwargs: object) -> "_Metric":
            return self

        def set(self, *args: object, **kwargs: object) -> None:
            pass

        def inc(self, *args: object, **kwargs: object) -> None:
            pass

        def observe(self, *args: object, **kwargs: object) -> None:
            pass

    fake_prometheus.Counter = _Metric
    fake_prometheus.Gauge = _Metric
    fake_prometheus.Histogram = _Metric
    fake_prometheus.start_http_server = lambda *args, **kwargs: None
    sys.modules["prometheus_client"] = fake_prometheus


sys.path.insert(0, str(SCRIPT_DIR))
MODULE_SPEC = importlib.util.spec_from_file_location("pipeline_rodinia_cuda", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
pipeline = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = pipeline
MODULE_SPEC.loader.exec_module(pipeline)


class CudaManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmarks = pipeline.base.load_manifest(MANIFEST_PATH)

    def test_manifest_has_expected_enabled_and_disabled_entries(self) -> None:
        identifiers = [item.benchmark_id for item in self.benchmarks]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertGreaterEqual(sum(item.enabled for item in self.benchmarks), 20)
        for benchmark_id in (
            "bfs",
            "cfd",
            "particlefilter_naive",
            "particlefilter_float",
            "srad_v2",
        ):
            self.assertTrue(
                next(item for item in self.benchmarks if item.benchmark_id == benchmark_id).enabled
            )
        for benchmark_id in ("hybridsort", "kmeans", "mummergpu"):
            benchmark = next(
                item for item in self.benchmarks if item.benchmark_id == benchmark_id
            )
            self.assertFalse(benchmark.enabled)
            self.assertTrue(benchmark.disabled_reason)

    def test_every_enabled_benchmark_has_smoke_and_official_inputs(self) -> None:
        for benchmark in self.benchmarks:
            if benchmark.enabled:
                self.assertEqual(
                    {"smoke", "official"},
                    {item.profile for item in benchmark.inputs},
                    benchmark.benchmark_id,
                )

    def test_all_declared_sources_and_required_inputs_exist(self) -> None:
        cuda_root = RODINIA_ROOT / "cuda"
        missing: list[str] = []
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            workdir = cuda_root / benchmark.workdir
            for source in benchmark.ast_owned_sources:
                if not (workdir / source).resolve().is_file():
                    missing.append(f"{benchmark.benchmark_id}: source {source}")
            for input_spec in benchmark.inputs:
                for required in (*benchmark.required_files, *input_spec.required_files):
                    if not (workdir / required).resolve().exists():
                        missing.append(
                            f"{benchmark.benchmark_id}[{input_spec.input_id}]: {required}"
                        )
        self.assertEqual([], missing)

    def test_enabled_builds_override_legacy_gpu_architecture(self) -> None:
        legacy_arch = pipeline.re.compile(r"(?:sm|compute)_(?:1\d|2\d|3[05])")
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            command = " ".join(
                token for build in benchmark.build_commands for token in build
            )
            self.assertIn("{cuda_arch}", command, benchmark.benchmark_id)
            self.assertIsNone(legacy_arch.search(command), benchmark.benchmark_id)

    def test_each_declared_source_is_owned_once_for_static_aggregation(self) -> None:
        cuda_root = RODINIA_ROOT / "cuda"
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            workdir = cuda_root / benchmark.workdir
            declared = {
                (workdir / source).resolve() for source in benchmark.ast_owned_sources
            }
            counts = {source: 0 for source in declared}
            for owned in pipeline.translation_unit_ownership(benchmark, cuda_root).values():
                for source in owned:
                    if source in counts:
                        counts[source] += 1
            unexpected = {str(path): count for path, count in counts.items() if count != 1}
            self.assertEqual({}, unexpected, benchmark.benchmark_id)

    def test_output_fields_are_unique(self) -> None:
        self.assertEqual(len(pipeline.SUMMARY_FIELDS), len(set(pipeline.SUMMARY_FIELDS)))
        self.assertEqual(len(pipeline.RUN_FIELDS), len(set(pipeline.RUN_FIELDS)))

    def test_enabled_sources_do_not_force_nonzero_cuda_device(self) -> None:
        cuda_root = RODINIA_ROOT / "cuda"
        violations: list[str] = []
        pattern = pipeline.re.compile(r"\bcudaSetDevice\s*\(\s*([1-9]\d*)\s*\)")
        for benchmark in self.benchmarks:
            if not benchmark.enabled:
                continue
            workdir = cuda_root / benchmark.workdir
            for source in benchmark.ast_owned_sources:
                path = (workdir / source).resolve()
                text = path.read_text(encoding="utf-8", errors="ignore")
                if pattern.search(text):
                    violations.append(f"{benchmark.benchmark_id}: {source}")
        self.assertEqual([], violations)


class GpuDeviceTests(unittest.TestCase):
    def test_formal_gpu_safety_defaults_are_enabled(self) -> None:
        args = pipeline.parse_args(["--benchmark", "bfs"])
        self.assertEqual("auto", args.cuda_arch)
        self.assertEqual("nvidia-smi", args.gpu_metrics)
        self.assertEqual(0.5, args.gpu_sample_interval)
        self.assertEqual(5.0, args.prometheus_context_seconds)
        self.assertFalse(args.allow_busy_gpu)
        self.assertFalse(args.collect_static)

    def test_query_gpu_devices_parses_units_and_identity(self) -> None:
        completed = pipeline.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="1, GPU-abc, NVIDIA RTX 4090, 8.9, 24564, 575.64, 450.00\n",
            stderr="",
        )
        with mock.patch.object(pipeline, "_run_text", return_value=completed):
            devices = pipeline.query_gpu_devices("nvidia-smi")
        self.assertEqual(1, len(devices))
        self.assertEqual(1, devices[0].physical_index)
        self.assertEqual("GPU-abc", devices[0].uuid)
        self.assertEqual("8.9", devices[0].compute_capability)
        self.assertEqual(24564 * 1024 * 1024, devices[0].total_memory_bytes)
        self.assertEqual(450.0, devices[0].power_limit_watts)

    def test_query_gpu_devices_falls_back_when_compute_cap_is_unsupported(self) -> None:
        unsupported = pipeline.subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="",
            stderr='Field "compute_cap" is not a valid field to query.\n',
        )
        fallback = pipeline.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0, GPU-abc, NVIDIA GeForce RTX 3090, 24576, 470.57.02, 350.00\n",
            stderr="",
        )
        with mock.patch.object(
            pipeline, "_run_text", side_effect=[unsupported, fallback]
        ) as run_text:
            devices = pipeline.query_gpu_devices("nvidia-smi")

        self.assertEqual(2, run_text.call_count)
        self.assertEqual(1, len(devices))
        self.assertEqual("", devices[0].compute_capability)
        self.assertEqual("470.57.02", devices[0].driver_version)
        self.assertEqual(350.0, devices[0].power_limit_watts)
        self.assertEqual(24576 * 1024 * 1024, devices[0].total_memory_bytes)

    def test_auto_arch_uses_compute_capability(self) -> None:
        device = pipeline.GpuDevice(0, "GPU-x", "test", "8.6", 0, "", None)
        self.assertEqual("sm_86", pipeline.resolve_cuda_arch("auto", device))
        self.assertEqual("sm_90", pipeline.resolve_cuda_arch("SM_90", device))
        with self.assertRaises(ValueError):
            pipeline.resolve_cuda_arch("compute_86", device)

    def test_explicit_arch_works_without_nvidia_smi_compute_capability(self) -> None:
        device = pipeline.GpuDevice(0, "GPU-x", "test", "", 0, "", None)
        self.assertEqual("sm_86", pipeline.resolve_cuda_arch("sm_86", device))
        with self.assertRaisesRegex(ValueError, "does not expose compute capability"):
            pipeline.resolve_cuda_arch("auto", device)

    def test_cuda12_texture_reference_benchmarks_are_detected(self) -> None:
        benchmarks = pipeline.base.load_manifest(MANIFEST_PATH)
        incompatible = pipeline.incompatible_benchmarks_for_toolkit(
            benchmarks, "12.4"
        )
        self.assertEqual(
            {"leukocyte"},
            {benchmark.benchmark_id for benchmark in incompatible},
        )
        self.assertEqual([], pipeline.incompatible_benchmarks_for_toolkit(benchmarks, "11.8"))

    def test_compute_process_filter_is_gpu_specific(self) -> None:
        completed = pipeline.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "GPU-other, 10, python, 100\n"
                "GPU-selected, 22, train.py, 2048\n"
            ),
            stderr="",
        )
        with mock.patch.object(pipeline, "_run_text", return_value=completed):
            processes = pipeline.query_compute_processes("nvidia-smi", "GPU-selected")
        self.assertEqual(1, len(processes))
        self.assertEqual("22", processes[0]["pid"])

    def test_conda_cuda_layout_uses_prefix_include_and_lib(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            (prefix / "include").mkdir()
            (prefix / "include" / "cuda_runtime.h").touch()
            (prefix / "lib").mkdir()
            (prefix / "lib" / "libcudart.so").touch()

            context = pipeline.token_context(
                str(prefix / "bin" / "nvcc"), prefix, "sm_86"
            )

        self.assertEqual(str(prefix / "include"), context["cuda_include_dir"])
        self.assertEqual(str(prefix / "lib"), context["cuda_lib_dir"])

    def test_conda_prefix_is_preferred_over_resolved_nvcc_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir).resolve()
            nvcc = prefix / "bin" / "nvcc"
            with mock.patch.dict(pipeline.os.environ, {"CONDA_PREFIX": str(prefix)}):
                self.assertEqual(prefix, pipeline.infer_cuda_dir(str(nvcc)))

    def test_cuda_environment_exposes_selected_toolkit_paths(self) -> None:
        context = {
            "cuda_dir": "/opt/cuda",
            "cuda_include_dir": "/opt/cuda/include",
            "cuda_lib_dir": "/opt/cuda/lib64",
        }
        with mock.patch.dict(
            pipeline.os.environ,
            {"CPATH": "/other/include", "LD_LIBRARY_PATH": "/other/lib"},
            clear=True,
        ):
            pipeline.configure_cuda_environment(context)
            self.assertEqual("/opt/cuda", pipeline.os.environ["CUDA_HOME"])
            self.assertEqual("/opt/cuda", pipeline.os.environ["CUDA_PATH"])
            self.assertEqual(
                ["/opt/cuda/include", "/other/include"],
                pipeline.os.environ["CPATH"].split(pipeline.os.pathsep),
            )
            self.assertEqual(
                ["/opt/cuda/lib64", "/other/lib"],
                pipeline.os.environ["LD_LIBRARY_PATH"].split(pipeline.os.pathsep),
            )


class GpuMetricTests(unittest.TestCase):
    @staticmethod
    def sample(timestamp: float, util: float, memory: float, power: float) -> object:
        return pipeline.GpuSample(
            timestamp,
            {
                "gpu_utilization_pct": util,
                "gpu_memory_io_utilization_pct": util / 2,
                "gpu_memory_used_bytes": memory,
                "gpu_power_draw_watts": power,
                "gpu_temperature_celsius": 50 + util / 10,
                "gpu_sm_clock_mhz": 1500,
                "gpu_memory_clock_mhz": 7000,
            },
        )

    def test_sample_parser_converts_mib_to_bytes(self) -> None:
        sample = pipeline._parse_gpu_sample_line(
            "75, 20, 512, 125.5, 65, 1800, 9000", 10.0
        )
        self.assertEqual(512 * 1024 * 1024, sample.values["gpu_memory_used_bytes"])
        self.assertEqual(125.5, sample.values["gpu_power_draw_watts"])

    def test_three_phase_summary_subtracts_background(self) -> None:
        windows = {
            "before": (0.0, 1.5),
            "during": (1.5, 2.5),
            "after": (2.5, 3.5),
        }
        samples = [
            self.sample(1.0, 10, 100, 40),
            self.sample(2.0, 70, 500, 160),
            self.sample(3.0, 14, 120, 44),
        ]
        result = pipeline.summarize_gpu_samples(samples, windows, 0.2)
        self.assertTrue(result["gpu_three_phase_collection_success"])
        self.assertEqual(12, result["gpu_background_utilization_pct_mean"])
        self.assertEqual(58, result["gpu_utilization_pct_background_adjusted_mean"])
        self.assertEqual(390, result["gpu_memory_used_bytes_background_adjusted_mean"])
        self.assertEqual(118, result["gpu_power_draw_watts_background_adjusted_mean"])
        self.assertEqual(118, result["gpu_energy_joules_background_adjusted_estimate"])
        self.assertEqual("", result["gpu_background_stability_warning"])

    def test_missing_phase_marks_collection_failed(self) -> None:
        windows = {
            "before": (0.0, 1.0),
            "during": (1.0, 2.0),
            "after": (2.0, 3.0),
        }
        result = pipeline.summarize_gpu_samples(
            [self.sample(1.5, 60, 400, 120)], windows, 0.2
        )
        self.assertFalse(result["gpu_three_phase_collection_success"])
        self.assertFalse(result["gpu_before_collection_success"])
        self.assertTrue(result["gpu_during_collection_success"])

    def test_sampler_error_marks_collection_failed_even_with_samples(self) -> None:
        windows = {
            "before": (0.0, 1.5),
            "during": (1.5, 2.5),
            "after": (2.5, 3.5),
        }
        samples = [
            self.sample(1.0, 10, 100, 40),
            self.sample(2.0, 70, 500, 160),
            self.sample(3.0, 14, 120, 44),
        ]
        result = pipeline.summarize_gpu_samples(
            samples, windows, 0.2, ["driver query failed once"]
        )
        self.assertFalse(result["gpu_three_phase_collection_success"])
        self.assertIn("driver query failed", result["gpu_three_phase_collection_error"])


class CudaStaticFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        pipeline.cxx_standard_library_flags.cache_clear()

    def tearDown(self) -> None:
        pipeline.cxx_standard_library_flags.cache_clear()

    def test_cxx_header_probe_uses_host_gcc_toolchain_when_needed(self) -> None:
        missing_headers = pipeline.subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="fatal error: 'cmath' file not found",
        )
        success = pipeline.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with (
            mock.patch.object(
                pipeline,
                "_probe_cxx_headers",
                side_effect=[missing_headers, success],
            ) as probe,
            mock.patch.object(pipeline.shutil, "which", return_value="/usr/bin/g++"),
            mock.patch.object(
                pipeline, "_gcc_toolchain_root", return_value=Path("/usr")
            ),
        ):
            flags, error = pipeline.cxx_standard_library_flags("clang++")

        self.assertEqual(("--gcc-toolchain=/usr",), flags)
        self.assertEqual("", error)
        probe.assert_has_calls(
            [
                mock.call("clang++"),
                mock.call("clang++", ("--gcc-toolchain=/usr",)),
            ]
        )

    def test_cxx_include_search_parser_keeps_existing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "c++"
            second = root / "target"
            first.mkdir()
            second.mkdir()
            output = (
                '#include "..." search starts here:\n'
                "#include <...> search starts here:\n"
                f" {first}\n"
                " ignoring nonexistent directory /missing\n"
                f" {second}\n"
                "End of search list.\n"
            )
            paths = pipeline._parse_cxx_include_search_paths(output)

        self.assertEqual((str(first), str(second)), paths)

    def test_cuda_source_feature_counts(self) -> None:
        source = r"""
        __constant__ int c[4];
        __global__ void kernel(float *x) {
            __shared__ float tile[32];
            atomicAdd(x, 1.0f);
            __syncthreads();
        }
        void run(float *device, float *host) {
            cudaMalloc((void **)&device, 128);
            cudaMallocManaged((void **)&device, 128);
            cudaMemcpy(device, host, 128, cudaMemcpyHostToDevice);
            kernel<<<1, 32>>>(device);
            cudaDeviceSynchronize();
            cudaFree(device);
        }
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.cu"
            path.write_text(source, encoding="utf-8")
            result = pipeline.extract_cuda_source_features([path])
        self.assertTrue(result["cuda_static_source_metrics_success"])
        self.assertEqual(1, result["cuda_kernel_definition_count"])
        self.assertEqual(1, result["cuda_kernel_launch_count"])
        self.assertEqual(1, result["cuda_memory_allocation_call_count"])
        self.assertEqual(1, result["cuda_managed_memory_allocation_call_count"])
        self.assertEqual(0, result["cuda_symbol_copy_call_count"])
        self.assertEqual(1, result["cuda_host_to_device_copy_count"])
        self.assertEqual(2, result["cuda_synchronization_call_count"])

    def test_command_token_expansion(self) -> None:
        context = {
            "cuda_arch": "sm_86",
            "nvcc": "/opt/cuda/bin/nvcc",
            "null_device": "/dev/null",
        }
        result = pipeline.expand_tokens(
            ["{nvcc}", "-arch={cuda_arch}", "{null_device}"], context
        )
        self.assertEqual(
            ["/opt/cuda/bin/nvcc", "-arch=sm_86", "/dev/null"], result
        )


if __name__ == "__main__":
    unittest.main()
