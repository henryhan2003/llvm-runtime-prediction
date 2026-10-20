# Input scale validation report

Overall status: **pass**

## Structural checks

- Unified entries: 1420.
- Programs: 71.
- Entries by dataset: `{"PolyBench/C": 600, "Rodinia CUDA": 440, "Rodinia OpenMP": 380}`.
- Generated files checked: 260 (1994043397 bytes).
- Duplicate effective-parameter groups: 0.
- Documented monotonic exceptions: `PolyBench/C/atax, PolyBench/C/bicg`.

## Errors

- None.

## Warnings

- PolyBench/C/atax: upstream LARGE/EXTRALARGE dimensions are not workload-monotonic
- PolyBench/C/bicg: upstream LARGE/EXTRALARGE dimensions are not workload-monotonic

## PolyBench spot compilation/run

- compile=pass, run=not_run_environment_unavailable: 90

## Rodinia executable validation

- Status: `not_run_environment_unavailable`.
- Reason: No installed WSL Linux distribution, make, or nvcc on the current Windows host.
- OpenMP example: `python scripts/pipeline_rodinia.py --manifest configs/rodinia_openmp_20sizes.json --benchmark bfs --input-profile SIZE_01 --warmup 0 --runs 1 --measurement-seconds 0`
- CUDA example: `python scripts/pipeline_rodinia_cuda.py --manifest configs/rodinia_cuda_20sizes.json --benchmark bfs --input-profile SIZE_01 --warmup 0 --runs 1 --measurement-seconds 0`
