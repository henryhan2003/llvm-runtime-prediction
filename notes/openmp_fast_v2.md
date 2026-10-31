# OpenMP CPU-load collection with fast-v2

## Scope and status

OpenMP now supports the same bounded fast-v2 measurement policy as the completed
PolyBench CPU-load experiment. No original CSV, benchmark input, thread count,
compiler optimization option, or static feature extraction algorithm was changed.
CUDA fast-v2 remains unsupported.

Local validation: 141 tests ran, 140 passed and one Linux/POSIX native-probe test
was skipped on Windows. Static extraction functions and the existing CSV field
declarations passed the preservation audit. Pilot and full manifests were
generated and checked without executing a benchmark or starting a stressor.
These results are not a substitute for an actual Linux pilot.

Artifacts: `outputs/openmp_fast_v2_validation/unit_test_report.json`,
`outputs/openmp_fast_v2_validation/pilot_plan/experiment_manifest.json`, and
`outputs/openmp_fast_v2_validation/batch_plan/experiment_manifest.json`.

## Measurement policy

| Item | fast-v2 behavior |
| --- | --- |
| Warmup | One invocation, excluded from formal runtime statistics |
| Formal observations | At least two, at most ten successful invocations |
| Measurement budget | Three seconds; stop when the budget and minimum count are satisfied, or at the cap |
| Long tasks | Two observations if their elapsed measurement window already exceeds three seconds |
| Short tasks | Up to ten observations; do not pad the during window with idle time |
| Runtime label | Existing `runtime_sec_median` calculation, unchanged |
| Before / after context | Six seconds each |
| Stable-load window | Six seconds, using the existing load-controller checks |
| Monitoring | Existing node-exporter/Prometheus fields, one-second query step and two-second query delay |
| Minimum phase scrapes | Three distinct real scrapes; a one-second actual scrape interval is expected |
| CPU levels | Existing IDLE / LOW / MEDIUM / HIGH control profiles |
| Threads | Existing benchmark/input manifest values, normally four or eight; no override |

The before window is after load stabilization and before warmup/target startup.
The during window contains formal invocations, not warmup. The background load
remains active through the existing after window. Preparation, stabilization,
monitoring context waits and cleanup are outside individual target runtime.

Short during windows may be explicitly classified as `MISSING_SHORT_WINDOW` by
the existing fast-v2 acceptance rule. Corresponding host during metrics stay
empty, not zero. Before and after requirements are not relaxed. Process CPU,
RSS and process I/O metrics still come from the real target invocations and are
not replaced by cached values. This change does not add missing resource
measurements or promise nonzero storage I/O for cached, compute-bound programs.

## Preparation reuse and provenance

The current nineteen enabled OpenMP benchmarks select their twenty inputs with
runtime arguments/files. Their preparation can therefore be shared across input
sizes as well as load profiles. A full CPU batch has 1,520 trials but nineteen
unique preparations; a fresh cache should have nineteen misses and 1,501 hits.
The pilot has twenty-four trials, two unique preparations and twenty-two hits.
Actual misses/hits are validated from receipts, not assumed to have occurred.

On a cache miss, the original make commands are forced with `-B`, and the
existing AST/CFG/IR extraction runs once. Compiler flags are not changed. The
cache key records declared build/static configuration, source/header/Makefile
dependencies, shared includes, compiler identities, relevant environment and
pipeline code. Source changes during preparation are rejected rather than
published under an obsolete fingerprint. Nonempty inherited `MAKEFLAGS` is
rejected to avoid accidental dry runs or altered build behavior.

The executable is still invoked in its original benchmark directory, at its
original path. A cached executable copy is for verification/provenance, not for
relocating execution. SHA-256 is checked before each warmup/formal invocation,
outside the measured target interval. Cache artifacts, static payload and
receipts are verified; corrupted entries or changed actual executables cause an
explicit failure, not silent rebuilding/overwriting. Do not run another build,
edit code, or pull Git changes into this worktree during collection.

The per-trial sidecar binds the benchmark-scoped preparation to its actual input
ID, load configuration, phase windows, summary and run IDs. Environment JSON
records public command-line arguments only, excluding internal session objects
and tuple-keyed cache indexes. Their provenance remains in the sidecars.

## Output and storage

After validation passes, the OpenMP experiment root automatically contains
`results.csv`: twenty-four rows for the default pilot, 1,520 for the full CPU
batch. Original summary values are copied as strings without changing precision,
zeros or missing values, then round-trip checked. The exporter adds only
`sample_id`, `trial_id`, `load_profile_id` and `load_type`; these are identity or
experimental metadata, not an automatic prediction whitelist. `load_type` here
denotes the CPU/memory experiment family, including that family's IDLE control.
Sample IDs are deterministic within the saved experiment manifest. The root
`validation_report.json` stores the aggregate table path, row count and SHA-256.

Raw per-trial CSVs, load sidecars, environment and logs are retained for audit.
The preparation cache is shared instead of duplicated in every trial. With the
default limits there are at most 15,200 formal invocation records plus 1,520
warmup records in the full CPU batch. The experiment manifest and load records
remain useful for failure diagnosis and reproducibility.

Do not estimate total bytes as simply 19/30 of the PolyBench download: different
programs generate different artifact sizes. No exact 500 MB bound is promised.
Check free disk space before starting and keep several GiB of headroom when
possible. The existing launcher stops before the next trial if less than one
GiB remains; that is a last-resort guard, not a reservation for a large build.
No previous experiment or cache is deleted automatically. After success, the
small root `results.csv` and validation report can be downloaded separately;
keep the raw audit bundle until local results have been verified.

## Changed files

- `scripts/openmp_preparation_cache.py`: verified benchmark-scoped preparation.
- `scripts/pipeline_rodinia.py`: optional cap/cache integration and environment serialization.
- `scripts/run_load_experiments.py`: OpenMP fast-v2 plans, pilot and batch gate.
- `scripts/check_load_experiment.py`: cache/static checks and lossless total table.
- `scripts/check_load_aware_pipeline.py`: includes new regression tests.
- `tests/test_openmp_fast_protocol.py`: cache, timing, plan, provenance/export and integration tests.
- `notes/openmp_fast_v2.md`: this protocol and operating commands.

## Reproduce local no-pressure checks

```powershell
cd E:\Dareway\LLVM_v01
python scripts/check_load_aware_pipeline.py --output outputs/openmp_fast_v2_validation/unit_test_report.json
```

Plan-only generation is safe on Windows. Choose fresh output directory names:

```powershell
python scripts/run_load_experiments.py --dataset openmp --stage pilot --protocol fast-v2 --output-dir outputs/openmp_pilot_plan_new
python scripts/run_load_experiments.py --dataset openmp --stage batch --protocol fast-v2 --output-dir outputs/openmp_batch_plan_new
```

## Synchronize only this change

Run in local Windows PowerShell. Other dirty files should remain untouched.
If Git reports an active index lock, first check the owning Git/editor process;
do not blindly delete the lock while another operation is running.

```powershell
cd E:\Dareway\LLVM_v01
$files = @('scripts/openmp_preparation_cache.py', 'scripts/pipeline_rodinia.py', 'scripts/run_load_experiments.py', 'scripts/check_load_experiment.py', 'scripts/check_load_aware_pipeline.py', 'tests/test_openmp_fast_protocol.py', 'notes/openmp_fast_v2.md')
git add -- $files
git commit --only -m "Support verified OpenMP fast-v2 load collection" -- $files
git push origin main
```

## Server pilot, then full collection

Run manually on the Linux server. The conda activation is deliberately separate.
No additional sudo installation is required if the existing stress-ng and
compiler/monitoring setup is still available. Confirm shared-server permission
before using `--permission-confirmed`.

```bash
cd /data_disk/hanli/llvm-runtime-prediction
git pull --ff-only origin main
tmux new -s openmp20_load
```

Inside that tmux pane:

```bash
conda activate llvm-runtime
export PATH="$PATH:$HOME/.local/conda-envs/load-tools/bin"
df -h /data_disk/hanli
command -v stress-ng
python scripts/run_load_experiments.py --dataset openmp --protocol fast-v2 --preflight-only
set -o pipefail
OMP_PILOT="/data_disk/hanli/openmp_load_fast_v2_pilot_$(date +%Y%m%d_%H%M%S)_$$"
python -u scripts/run_load_experiments.py \
  --dataset openmp --stage pilot --load-types cpu --protocol fast-v2 \
  --temperature-policy record-only --execute --permission-confirmed \
  --output-dir "$OMP_PILOT" 2>&1 | tee "${OMP_PILOT}.launcher.log"
RUN_EXIT=${PIPESTATUS[0]}
echo "launcher_exit=$RUN_EXIT"
echo "$OMP_PILOT"
```

This pilot covers BFS and myocyte, SIZE_01 / SIZE_10 / SIZE_20, all four CPU
levels. Only after it prints PASS, in the same pane:

```bash
OMP_FULL="/data_disk/hanli/openmp_cpu_load20_fast_v2_$(date +%Y%m%d_%H%M%S)_$$"
python -u scripts/run_load_experiments.py \
  --dataset openmp --stage batch --load-types cpu --protocol fast-v2 \
  --temperature-policy record-only \
  --pilot-validation "$OMP_PILOT/validation_report.json" \
  --execute --permission-confirmed --output-dir "$OMP_FULL" \
  2>&1 | tee "${OMP_FULL}.launcher.log"
RUN_EXIT=${PIPESTATUS[0]}
echo "launcher_exit=$RUN_EXIT"
echo "$OMP_FULL"
```

The batch requires an actual successful OpenMP fast-v2 pilot with at least
twenty-four configurations on the same host and matching temperature policy.
The old PolyBench pilot cannot unlock this new integration. After switching
panes or reconnecting, recover `OMP_PILOT` from its printed path rather than
assuming shell variables survived.

`record-only` matches the previous explicitly selected PolyBench experiment
policy. It records temperatures but disables script temperature aborts; PASS
does not prove thermal safety or absence of throttling. Other existing safety,
capacity, permission, timeout and cleanup checks remain in place. For a
temperature-aborting run, use `--temperature-policy abort` in both pilot and
batch; do not mix pilot/batch policies.

Detach with Ctrl+B, release both keys, then D. No extra command is needed to keep
an active tmux workload running. Do not close the pane with `exit` while a
foreground collection command is still active.
