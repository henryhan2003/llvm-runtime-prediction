# CUDA collection under controlled GPU compute load

## Scope and current validation

This is a new GPU-load protocol, not the previous four-level CPU stress protocol.
The CUDA pipeline keeps its existing runtime definition, static extraction and
GPU/Prometheus field names. CPU and host memory receive no synthetic pressure.
The original CSV field declarations remain unchanged. Load controls, worker/PID
provenance, calibration and raw GPU poll observations are stored in load sidecars.
There is no model training or automatic deletion of earlier results.

The pilot covers lavaMD, hotspot3D and BFS at SIZE_01 / SIZE_10 / SIZE_20 and all
four GPU profiles: 36 samples. Full collection covers all 22 currently enabled
benchmarks, twenty sizes each and four GPU profiles: 1,760 samples. Existing
smoke/official inputs and disabled benchmarks are not silently substituted.

Local Windows regression: 162 tests ran, 161 passed and one Linux-native probe
test was skipped. Static extraction functions and original CSV declarations
passed preservation checks. Native nvcc compilation and real GPU calibration
have NOT been performed locally: this machine has no nvcc. A real Linux pilot
is mandatory before full collection. The two plan-only manifests are not real
experimental evidence and cannot unlock a batch.

## Background load definition

GPU_IDLE starts no additional GPU compute process. GPU_LOW, GPU_MEDIUM and
GPU_HIGH target approximately 25%, 50% and 75% GPU busy time before the target.
A small independent CUDA program launches bounded FMA kernels across the device
with a fixed per-device buffer/grid and a short duty cycle. Its register-compute
loop and host sleep cycle generate demand; they do not reserve an SM quota.
CUDA context/driver memory overhead exists and is measured by the existing
whole-device memory fields. This is not a separate GPU memory-capacity or
bandwidth experiment.

The worker starts with the nominal duty and is calibrated using the existing
nvidia-smi sampler: six-second windows, three consecutive accepted windows,
eight percentage-point target tolerance, and a maximum ten-percent natural
GPU_IDLE baseline. The worker duty is capped at 0.85 and frozen before the
target starts. These are control settings, never fabricated observations.
An unstable or missed target causes failure rather than relabeling.

The GPU utilization metric measures time with kernels active, not percentage
of available FLOPs/SMs consumed. LOW/MEDIUM/HIGH are background-demand conditions,
not guaranteed remaining-capacity fractions. During target execution total GPU
utilization includes both programs and need not stay at the before target.
The background worker continues with frozen settings through the after window.

The selected sharing mode is native CUDA context time slicing without MPS.
Preflight rejects MPS settings/services and non-DEFAULT compute mode instead of
changing shared-server settings. Existing GPU compute processes cause a stop.
During collection, periodic process checks allow only descendants of this
pipeline, including its supervised worker and target. Detected foreign compute
jobs cause a stop; no foreign PID is signaled. Polling is not proof that every
very brief external job can be detected.

NVIDIA references:
- https://docs.nvidia.com/deploy/nvidia-smi/index.html
- https://docs.nvidia.com/deploy/mps/architecture.html

## Timing, monitoring and safety

The sequence is: prepare once per benchmark; establish quiet host state;
start/calibrate GPU background; six-second before window; one warmup; formal
invocations; six-second after window with background still active; cleanup.
Preparation, calibration, context waits and cleanup are outside individual
target timings. Runtime remains the existing command/process end-to-end time,
not CUDA kernel-only duration.

Default formal measurement: at least two invocations, three-second accumulated
target runtime budget and at most twenty invocations. CUDA deliberately has a
different cap from the CPU fast-v2 cap of ten to accommodate GPU observation
coverage. After the time budget is reached, more real invocations may be needed
until the GPU phase has three poll observations, still subject to the cap.
No idle padding or neighboring-window sample substitution is used. If the cap
is reached without enough GPU observations, the trial fails with its evidence
retained. Validate a new pilot before changing the cap for a batch.

GPU sampling remains 0.5 seconds through the existing nvidia-smi sampler. Every
GPU phase must contain at least three actual poll observations with valid
utilization and used-memory values. The raw selected observations and timestamps
are retained in sidecars and reaggregated during validation. Poll observations
are not necessarily independent hardware counter updates. The existing node
exporter is expected to scrape at one second; fast-v2 permits explicitly missing
short HOST during windows, but this protocol does not permit missing GPU phases.
Process CPU/RSS/I/O counters remain invocation scoped and are not copied from
the background worker or whole-device telemetry.

GPU utilization, memory, power and energy fields still describe the physical
GPU containing both loads, not resource demand exclusively attributable to the
target PID. Existing background-subtracted power/energy fields are estimates,
not precise target-only accounting. CUDA AST/CFG/generic IR retains its existing
host-only extraction scope; CUDA source counters are not complete device IR.

Defaults retain CPU/host safety checks, an 85 C GPU abort policy, free GPU memory
reserve of at least 512 MiB and 15% of device memory, per-command/trial timeouts,
owned-process supervision and process-group cleanup checks. The GPU worker also
has its own wall-clock timeout. Hardware protections are not modified. Passing
collection does not certify thermal safety, absence of throttling or benchmark
mathematical output correctness. `--temperature-policy record-only` is an explicit
opt-in override, not the recommended default; pilot and batch must match it.

## Preparation reuse and disk space

CUDA preparation is cached per benchmark, GPU UUID, architecture and toolchain.
Source/header/Makefile dependencies, selected toolkit headers, code and relevant
environment are fingerprinted. Cached AST/CFG/IR/source counters and executable
are hash-verified. Actual execution stays at the original executable path and
working directory, with its hash checked before each invocation. Input args,
GPU block/thread settings, original compiler optimization flags and algorithms
are not changed. A fresh pilot has three preparations; a fresh full batch has
twenty-two, not 1,760 copies. Runtime/resource/monitoring results are never cached.

No guarantee is made that CUDA storage will equal the 145 MiB OpenMP archive:
CUDA/standard-library AST artifacts may be much larger. Keep several GiB free
when possible. The launcher stops before a new trial below one GiB free, without
deleting anything. Check the pilot cache size before the full run. It does not
reserve disk space during compilation, so monitor the shared filesystem too.

The root results.csv is generated only after complete validation: 36 or 1,760
rows, with stable sample IDs incorporating both host and GPU configuration
hashes. All original values are preserved and round-trip checked. Added labels
are experimental metadata, not default model inputs. Whole-GPU during/after
fields are not same-run pre-execution prediction inputs.

## Files to synchronize

Only these changes are needed; leave unrelated worktree changes alone:

```text
scripts/gpu_resource_load.py
scripts/gpu_resource_load_worker.cu
scripts/cuda_preparation_cache.py
scripts/run_cuda_gpu_load_experiments.py
scripts/pipeline_rodinia.py
scripts/pipeline_rodinia_cuda.py
scripts/check_load_experiment.py
scripts/check_load_aware_pipeline.py
configs/gpu_load_profiles.json
tests/test_cuda_gpu_load.py
notes/cuda_gpu_load_collection.md
```

## Local Git and no-pressure validation

```powershell
cd E:\Dareway\LLVM_v01
python scripts/check_load_aware_pipeline.py --output outputs/cuda_gpu_load_validation/full_regression.json
$files = @(
  'scripts/gpu_resource_load.py', 'scripts/gpu_resource_load_worker.cu',
  'scripts/cuda_preparation_cache.py', 'scripts/run_cuda_gpu_load_experiments.py',
  'scripts/pipeline_rodinia.py', 'scripts/pipeline_rodinia_cuda.py',
  'scripts/check_load_experiment.py', 'scripts/check_load_aware_pipeline.py',
  'configs/gpu_load_profiles.json', 'tests/test_cuda_gpu_load.py',
  'notes/cuda_gpu_load_collection.md'
)
git add -- $files
git commit --only -m "Add controlled CUDA GPU-load collection" -- $files
git push origin main
```

The regression suite uses mocks for GPU execution. Default standalone plan-only
generation is safe on Windows and does not require importing the CUDA pipeline,
querying a GPU, compiling the worker or connecting to the server.

## Server preflight and pilot

No sudo or stress-ng installation is required for this GPU protocol. Use the
existing CUDA-capable llvm-runtime environment, with nvcc, nvidia-smi, host
compilers, LLVM, prometheus-client and existing monitoring services available.
All twenty-size inputs must already be materialized. The current leukocyte
source needs the previously validated pre-CUDA-12 toolkit; preflight rejects
unsupported programs rather than silently omitting them from the 22-program set.

```bash
cd /data_disk/hanli/llvm-runtime-prediction
git pull --ff-only origin main
tmux new -s cuda20_gpu
```

Inside the new pane, activate the environment manually:

```bash
conda activate llvm-runtime
nvidia-smi
nvcc --version
df -h /data_disk/hanli
python scripts/run_cuda_gpu_load_experiments.py --gpu-index 0 --preflight-only
```

Choose an authorized idle GPU. If GPU 0 is occupied, choose another permitted
physical index and use it consistently for preflight, pilot and batch. If the
driver cannot report compute capability, add the verified `--cuda-arch sm_XX`
explicitly. Do not use --allow-busy-gpu to bypass controlled-load ownership checks.

```bash
CUDA_PILOT="/data_disk/hanli/cuda_gpu_pilot_$(date +%Y%m%d_%H%M%S)_$$"
set -o pipefail
python -u scripts/run_cuda_gpu_load_experiments.py \
  --stage pilot --gpu-index 0 --execute --permission-confirmed \
  --output-dir "$CUDA_PILOT" 2>&1 | tee "${CUDA_PILOT}.launcher.log"
RUN_EXIT=${PIPESTATUS[0]}
echo "pilot_exit=$RUN_EXIT"
echo "$CUDA_PILOT"
```

This is the first real worker compilation and calibration. If it fails, keep
the log/sidecars and diagnose; do not immediately start the full batch or widen
load tolerances. CPU/OpenMP pilot reports cannot unlock this GPU experiment.

## Full batch after pilot PASS

```bash
CUDA_ROOT="/data_disk/hanli/cuda_gpu_load20_$(date +%Y%m%d_%H%M%S)_$$"
python -u scripts/run_cuda_gpu_load_experiments.py \
  --stage batch --gpu-index 0 --pilot-validation "$CUDA_PILOT/validation_report.json" \
  --execute --permission-confirmed --output-dir "$CUDA_ROOT" \
  2>&1 | tee "${CUDA_ROOT}.launcher.log"
RUN_EXIT=${PIPESTATUS[0]}
echo "batch_exit=$RUN_EXIT"
echo "$CUDA_ROOT"
```

The gate checks the 36-trial pilot, all four GPU levels, same GPU UUID/host,
measurement policy, temperature policy and source fingerprints. Do not pull Git
changes, rebuild benchmarks or edit controls during a live experiment.
After PASS the combined table is `$CUDA_ROOT/results.csv`.

## Failure and explicit resume

```bash
python -u scripts/run_cuda_gpu_load_experiments.py \
  --resume --execute --permission-confirmed --output-dir "$CUDA_ROOT" \
  2>&1 | tee "${CUDA_ROOT}.resume_$(date +%Y%m%d_%H%M%S).log"
RUN_EXIT=${PIPESTATUS[0]}
echo "resume_exit=$RUN_EXIT"
```

Resume revalidates completed trials, code/control hashes, original device and
pending preparation environment before any new target starts. Only unfinished
trials are retried, in fresh retry directories; old failed logs/results remain.
The prior manifest is archived as manifest_before_resume_*.json. Do not append
duplicate PATH entries or switch toolkits on resume: restore the original pane
environment instead of bypassing fingerprint checks. The original calibration
and statistical parameters are restored from the manifest, not new CLI defaults.

Detach from tmux with Ctrl+B, release both keys, then D. Reattach with
`tmux attach -t cuda20_gpu`; shell variables remain in that pane. The server run
is manual; this local implementation did not start a GPU workload remotely.
