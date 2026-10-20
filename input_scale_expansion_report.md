# LLVM_v01 input scale expansion report

## 1. Scope and audit basis

This report covers the benchmark set that the current, validated collection manifests actually expose:

- PolyBench/C 4.2.1: 30 CPU-serial programs.
- Rodinia 3.1 OpenMP: 19 enabled programs; `mummergpu` is already disabled.
- Rodinia 3.1 CUDA: 22 enabled programs; `hybridsort`, `kmeans`, and `mummergpu` are already disabled.

The audit used the benchmark source, companion headers, Makefiles, run files, data directories, and the current `configs/rodinia_openmp.json`, `configs/rodinia_cuda.json`, `scripts/pipeline_new.py`, `scripts/pipeline_rodinia.py`, and `scripts/pipeline_rodinia_cuda.py`. Existing result CSV files were not used as configuration sources and are not modified by this work.

## 2. Pre-modification audit

### 2.1 PolyBench/C

All 30 PolyBench programs select their problem size at compile time in the source-matched `.h` file. If no profile macro is passed, every header defaults to `LARGE_DATASET`. Changing a profile therefore requires recompilation, but never an external input file. `scripts/pipeline_new.py` currently recognizes exactly the five upstream profile macros and compiles each program with one selected macro.

The five existing definitions remain compatibility anchors. The planned 20-size interface places them at `SIZE_01`, `SIZE_05`, `SIZE_10`, `SIZE_15`, and `SIZE_20`; the other 15 profiles are integer, piecewise-linear interpolations of every declared dimension.

| program_id | Workload dimensions in MINI / SMALL / MEDIUM / LARGE / EXTRALARGE |
| --- | --- |
| correlation | `(M,N)=(28,32)/(80,100)/(240,260)/(1200,1400)/(2600,3000)` |
| covariance | same as correlation |
| gemm | `(NI,NJ,NK)=(20,25,30)/(60,70,80)/(200,220,240)/(1000,1100,1200)/(2000,2300,2600)` |
| gemver | `N=40/120/400/2000/4000` |
| gesummv | `N=30/90/250/1300/2800` |
| symm | `(M,N)=(20,30)/(60,80)/(200,240)/(1000,1200)/(2000,2600)` |
| syr2k | same as symm |
| syrk | same as symm |
| trmm | same as symm |
| 2mm | `(NI,NJ,NK,NL)=(16,18,22,24)/(40,50,70,80)/(180,190,210,220)/(800,900,1100,1200)/(1600,1800,2200,2400)` |
| 3mm | `(NI,NJ,NK,NL,NM)=(16,18,20,22,24)/(40,50,60,70,80)/(180,190,200,210,220)/(800,900,1000,1100,1200)/(1600,1800,2000,2200,2400)` |
| atax | `(M,N)=(38,42)/(116,124)/(390,410)/(1900,2100)/(1800,2200)` |
| bicg | same as atax |
| doitgen | `(NQ,NR,NP)=(8,10,12)/(20,25,30)/(40,50,60)/(140,150,160)/(220,250,270)` |
| mvt | `N=40/120/400/2000/4000` |
| cholesky | `N=40/120/400/2000/4000` |
| durbin | `N=40/120/400/2000/4000` |
| gramschmidt | `(M,N)=(20,30)/(60,80)/(200,240)/(1000,1200)/(2000,2600)` |
| lu | `N=40/120/400/2000/4000` |
| ludcmp | `N=40/120/400/2000/4000` |
| trisolv | `N=40/120/400/2000/4000` |
| deriche | `(W,H)=(64,64)/(192,128)/(720,480)/(4096,2160)/(7680,4320)` |
| floyd-warshall | `N=60/180/500/2800/5600` |
| nussinov | `N=60/180/500/2500/5500` |
| adi | `(TSTEPS,N)=(20,20)/(40,60)/(100,200)/(500,1000)/(1000,2000)` |
| fdtd-2d | `(TMAX,NX,NY)=(20,20,30)/(40,60,80)/(100,200,240)/(500,1000,1200)/(1000,2000,2600)` |
| heat-3d | `(TSTEPS,N)=(20,10)/(40,20)/(100,40)/(500,120)/(1000,200)` |
| jacobi-1d | `(TSTEPS,N)=(20,30)/(40,120)/(100,400)/(500,2000)/(1000,4000)` |
| jacobi-2d | `(TSTEPS,N)=(20,30)/(40,90)/(100,250)/(500,1300)/(1000,2800)` |
| seidel-2d | `(TSTEPS,N)=(20,40)/(40,120)/(100,400)/(500,2000)/(1000,4000)` |

Audit exception: upstream `atax.h` and `bicg.h` reduce `M` from 1900 to 1800 between LARGE and EXTRALARGE while increasing `N` from 2100 to 2200. The legacy macros must remain unchanged, so strict component-wise monotonicity is impossible for those two anchor transitions. The expanded profile records this upstream exception instead of silently altering the original definitions.

### 2.2 Rodinia OpenMP

OpenMP execution parameters are fixed at four threads in the expanded manifest, except `backprop`, whose source hard-codes eight threads. Thread count is not used to create an input-size level.

| program_id | Existing input configurations | True workload controls | Execution-only controls | Expansion mechanism | Rebuild needed |
| --- | --- | --- | --- | --- | --- |
| b+tree | same one-million-key file at 2 and 4 cores | record count and query count in two text files | `cores` | deterministic key and valid query-file generation | no |
| backprop | input layer 1024, 65536 | input layer width, multiple of 16 | hard-coded 8 threads | command-line values | no |
| bfs | 4096-node and about one-million-node graphs | node and edge counts in graph file | thread argument | seeded connected graph generation | no |
| cfd | 97K and 193K mesh cells | mesh-cell count and connectivity | compile-time OpenMP setting | deterministic valid mesh generation | no |
| heartwall | 2 and 20 frames of one AVI | processed frame count | thread argument | reuse AVI, vary legal frame count | no |
| hotspot | 64x64 and 1024x1024, both 2 iterations | rows, columns, iterations and matching temperature/power files | thread argument | deterministic field generation | no |
| hotspot3D | 64x64x8/10 iterations and 512x512x8/100 | grid side, layers, iterations and matching fields | `OMP_NUM_THREADS` | deterministic field generation | no |
| kmeans | 100-point and KDD files | object count and attribute count in file | `-n` threads | seeded clustered-point generation | no |
| lavaMD | `boxes1d` 4 and 10 | boxes per dimension; particles grow cubically | `-cores` | command-line values | no |
| leukocyte | 1 and 5 frames of one AVI | processed frame count | thread argument | reuse AVI, vary legal frame count | no |
| lud | internally generated matrices 256 and 8000 | matrix dimension | `-n` threads | existing `-s` command-line generator | no |
| myocyte | simulation horizon 10 and 100, one instance | simulation horizon and instance count | mode and threads | command-line values | no |
| nn | same four database files, only neighbor count changes | database-record count | neighbor count and target coordinate | deterministic fixed-format database generation | no |
| nw | dimensions 256 and 2048 | sequence dimension | penalty and threads | aligned command-line values | no |
| particlefilter | 64x64x5/1000 and 128x128x10/10000 | video dimensions, frames, particle count | none | command-line values | no |
| pathfinder | 1000x100 and 100000x100 | columns and rows | OpenMP environment | command-line values | no |
| srad_v1 | 128x128/10 iterations and 502x458/100 | rows, columns, iterations | lambda and threads | command-line values | no |
| srad_v2 | 128x128 and 2048x2048 | rows, columns, ROI, iterations | lambda and threads | command-line values with bounded ROI | no |
| streamcluster | 1024x16 and 65536x256 synthetic points | points, dimensions, cluster bounds and chunk size | worker count | existing synthetic-input mode | no |

Audit corrections for the new manifest:

- The current `myocyte` metadata labels `argv[1]` and `argv[2]` backwards; source inspection shows `argv[1]=xmax` and `argv[2]=workload/instances`.
- The current `nn` smoke/official pair does not change the database, so it is not a real input-scale pair.
- Several current smoke/official pairs also change thread count. The expanded manifest holds threads fixed.

### 2.3 Rodinia CUDA

CUDA block sizes, work-group macros, compiler optimization, selected GPU, and algorithmic blocking controls are held fixed. They are not input-size dimensions.

| program_id | Existing input configurations | True workload controls | Fixed execution controls | Expansion mechanism | Rebuild needed |
| --- | --- | --- | --- | --- | --- |
| b+tree | same one-million-key file twice | record and query counts | work group 256 | shared deterministic files | no |
| backprop | layer 1024, 65536 | input layer width | build flags | command-line values | no |
| bfs | 4096 and about one-million nodes | node and edge counts | kernel launch code | shared graph files | no |
| cfd | 97K, 193K, and 0.2M cells | mesh-cell count/connectivity | CUDA blocks | shared mesh files | no |
| dwt2d | 192x192 and 1024x1024 files | width, height, component count | wavelet 5/3 and three levels | deterministic raw RGB byte fields | no |
| gaussian | file matrices 16 and 1024 | matrix dimension | CUDA blocks | existing deterministic `-s` generator | no |
| heartwall | 2 and 20 frames | frame count | CUDA kernels | reuse AVI | no |
| hotspot | 64 and 512 square grids | grid side and iterations | pyramid height | shared deterministic fields | no |
| hotspot3D | 64x64x8/10 and 512x512x8/100 | grid side, layers, iterations | CUDA blocks | shared deterministic fields | no |
| huffman | same 1 MiB file twice | byte count | 256 block threads | seeded binary files aligned to 1024 bytes | no |
| lavaMD | `boxes1d` 2 and 10 | boxes per dimension | CUDA launch geometry | command-line values | no |
| leukocyte | 2 and 5 frames | frame count | CUDA kernels | reuse AVI | no |
| lud | file matrices 64 and 2048 | matrix dimension | CUDA block geometry | existing deterministic `-s` generator | no |
| myocyte | horizons 10 and 100, one instance | simulation horizon and instances | mode 0 | command-line values | no |
| nn | same four databases twice | database-record count | default 256 threads/block | shared deterministic database files | no |
| nw | dimensions 256 and 2048 | sequence dimension | penalty 10, work group 16 | aligned command-line values | no |
| particlefilter_naive | 32x32x2/100 and 128x128x10/1000 | dimensions, frames, particles | CUDA implementation | command-line values | no |
| particlefilter_float | same anchors as naive | dimensions, frames, particles | CUDA implementation | command-line values | no |
| pathfinder | 1000x20 and 100000x100 | columns and rows | pyramid height | command-line values | no |
| srad_v1 | fixed 502x458, 10 and 100 iterations | rows, columns, iterations | lambda | command-line values | no |
| srad_v2 | 256 and 2048 square matrices | rows, columns, ROI, iterations | lambda | command-line values with bounded ROI | no |
| streamcluster | 1000x16 and 65536x256 synthetic points | points, dimensions, cluster bounds, chunk size | CUDA implementation selector | existing synthetic mode | no |

The currently disabled programs are not valid candidates for input-only expansion: CUDA `hybridsort` and `kmeans` require ports away from removed CUDA texture-reference APIs, CUDA `mummergpu` is CUDA-1.x/NVIDIA-SDK-era code, and the OpenMP manifest's `mummergpu` entry points to the same non-OpenMP implementation. They remain disabled; no fake 20-level entries are created.

### 2.4 Validation environment

The current Windows host has native Clang 19 and MinGW GCC, but no `nvcc`, `make`, or installed WSL Linux distribution. Consequently:

- PolyBench native syntax/compile checks can be executed locally through the existing Windows-compatible pipeline support.
- Rodinia data-format, command, path, uniqueness, monotonicity, and deterministic-regeneration checks can be executed locally.
- GNU/OpenMP and CUDA build/run validation cannot truthfully be marked as executed on this host. The validation script emits runnable small/middle/large commands for a Linux/CUDA host and records these checks as `not_run_environment_unavailable`, not as passes.

## 3. Expansion design

All expanded configurations use `SIZE_01` through `SIZE_20`. Existing files and the two original Rodinia manifests remain untouched. Generated files live under `datasets/rodinia_3.1/data/generated_20/`, and deterministic generators record their seed and semantic parameters. New manifests use four OpenMP threads consistently (or the source-fixed count noted above) and preserve CUDA launch/build settings.

Implementation and post-generation validation results are appended below by the expansion tooling.

## 4. Generated 20-level configurations

- Unified manifest: `configs/input_scales_20.json`
- OpenMP manifest: `configs/rodinia_openmp_20sizes.json`
- CUDA manifest: `configs/rodinia_cuda_20sizes.json`
- Rodinia generated data materialized in this run: `true`
- Reproducibility seed: `20261020`

### PolyBench/C / 2mm

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"NI":16,"NJ":18,"NK":22,"NL":24}` | MINI_DATASET | 152064 | - |
| SIZE_02 | `{"NI":22,"NJ":26,"NK":34,"NL":38}` | new | 739024 | - |
| SIZE_03 | `{"NI":28,"NJ":34,"NK":46,"NL":52}` | new | 2277184 | - |
| SIZE_04 | `{"NI":34,"NJ":42,"NK":58,"NL":66}` | new | 5466384 | - |
| SIZE_05 | `{"NI":40,"NJ":50,"NK":70,"NL":80}` | SMALL_DATASET | 11200000 | - |
| SIZE_06 | `{"NI":68,"NJ":78,"NK":98,"NL":108}` | new | 56137536 | - |
| SIZE_07 | `{"NI":96,"NJ":106,"NK":126,"NL":136}` | new | 174375936 | - |
| SIZE_08 | `{"NI":124,"NJ":134,"NK":154,"NL":164}` | new | 419653696 | - |
| SIZE_09 | `{"NI":152,"NJ":162,"NK":182,"NL":192}` | new | 860461056 | - |
| SIZE_10 | `{"NI":180,"NJ":190,"NK":210,"NL":220}` | MEDIUM_DATASET | 1580040000 | - |
| SIZE_11 | `{"NI":304,"NJ":332,"NK":388,"NL":416}` | new | 16290586624 | - |
| SIZE_12 | `{"NI":428,"NJ":474,"NK":566,"NL":612}` | new | 70273237824 | - |
| SIZE_13 | `{"NI":552,"NJ":616,"NK":744,"NL":808}` | new | 204410916864 | - |
| SIZE_14 | `{"NI":676,"NJ":758,"NK":922,"NL":1004}` | new | 474329936704 | - |
| SIZE_15 | `{"NI":800,"NJ":900,"NK":1100,"NL":1200}` | LARGE_DATASET | 950400000000 | - |
| SIZE_16 | `{"NI":960,"NJ":1080,"NK":1320,"NL":1440}` | new | 1970749440000 | - |
| SIZE_17 | `{"NI":1120,"NJ":1260,"NK":1540,"NL":1680}` | new | 3651056640000 | - |
| SIZE_18 | `{"NI":1280,"NJ":1440,"NK":1760,"NL":1920}` | new | 6228541440000 | - |
| SIZE_19 | `{"NI":1440,"NJ":1620,"NK":1980,"NL":2160}` | new | 9976919040000 | - |
| SIZE_20 | `{"NI":1600,"NJ":1800,"NK":2200,"NL":2400}` | EXTRALARGE_DATASET | 15206400000000 | - |

### PolyBench/C / 3mm

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"NI":16,"NJ":18,"NK":20,"NL":22,"NM":24}` | MINI_DATASET | 3041280 | - |
| SIZE_02 | `{"NI":22,"NJ":26,"NK":30,"NL":34,"NM":38}` | new | 22170720 | - |
| SIZE_03 | `{"NI":28,"NJ":34,"NK":40,"NL":46,"NM":52}` | new | 91087360 | - |
| SIZE_04 | `{"NI":34,"NJ":42,"NK":50,"NL":58,"NM":66}` | new | 273319200 | - |
| SIZE_05 | `{"NI":40,"NJ":50,"NK":60,"NL":70,"NM":80}` | SMALL_DATASET | 672000000 | - |
| SIZE_06 | `{"NI":68,"NJ":78,"NK":88,"NL":98,"NM":108}` | new | 4940103168 | - |
| SIZE_07 | `{"NI":96,"NJ":106,"NK":116,"NL":126,"NM":136}` | new | 20227608576 | - |
| SIZE_08 | `{"NI":124,"NJ":134,"NK":144,"NL":154,"NM":164}` | new | 60430132224 | - |
| SIZE_09 | `{"NI":152,"NJ":162,"NK":172,"NL":182,"NM":192}` | new | 147999301632 | - |
| SIZE_10 | `{"NI":180,"NJ":190,"NK":200,"NL":210,"NM":220}` | MEDIUM_DATASET | 316008000000 | - |
| SIZE_11 | `{"NI":304,"NJ":332,"NK":360,"NL":388,"NM":416}` | new | 5864611184640 | - |
| SIZE_12 | `{"NI":428,"NJ":474,"NK":520,"NL":566,"NM":612}` | new | 36542083668480 | - |
| SIZE_13 | `{"NI":552,"NJ":616,"NK":680,"NL":744,"NM":808}` | new | 138999423467520 | - |
| SIZE_14 | `{"NI":676,"NJ":758,"NK":840,"NL":922,"NM":1004}` | new | 398437146831360 | - |
| SIZE_15 | `{"NI":800,"NJ":900,"NK":1000,"NL":1100,"NM":1200}` | LARGE_DATASET | 950400000000000 | - |
| SIZE_16 | `{"NI":960,"NJ":1080,"NK":1200,"NL":1320,"NM":1440}` | new | 2364899328000000 | - |
| SIZE_17 | `{"NI":1120,"NJ":1260,"NK":1400,"NL":1540,"NM":1680}` | new | 5111479296000000 | - |
| SIZE_18 | `{"NI":1280,"NJ":1440,"NK":1600,"NL":1760,"NM":1920}` | new | 9965666304000000 | - |
| SIZE_19 | `{"NI":1440,"NJ":1620,"NK":1800,"NL":1980,"NM":2160}` | new | 17958454272000000 | - |
| SIZE_20 | `{"NI":1600,"NJ":1800,"NK":2000,"NL":2200,"NM":2400}` | EXTRALARGE_DATASET | 30412800000000000 | - |

### PolyBench/C / adi

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":20,"TSTEPS":20}` | MINI_DATASET | 400 | - |
| SIZE_02 | `{"N":30,"TSTEPS":25}` | new | 750 | - |
| SIZE_03 | `{"N":40,"TSTEPS":30}` | new | 1200 | - |
| SIZE_04 | `{"N":50,"TSTEPS":35}` | new | 1750 | - |
| SIZE_05 | `{"N":60,"TSTEPS":40}` | SMALL_DATASET | 2400 | - |
| SIZE_06 | `{"N":88,"TSTEPS":52}` | new | 4576 | - |
| SIZE_07 | `{"N":116,"TSTEPS":64}` | new | 7424 | - |
| SIZE_08 | `{"N":144,"TSTEPS":76}` | new | 10944 | - |
| SIZE_09 | `{"N":172,"TSTEPS":88}` | new | 15136 | - |
| SIZE_10 | `{"N":200,"TSTEPS":100}` | MEDIUM_DATASET | 20000 | - |
| SIZE_11 | `{"N":360,"TSTEPS":180}` | new | 64800 | - |
| SIZE_12 | `{"N":520,"TSTEPS":260}` | new | 135200 | - |
| SIZE_13 | `{"N":680,"TSTEPS":340}` | new | 231200 | - |
| SIZE_14 | `{"N":840,"TSTEPS":420}` | new | 352800 | - |
| SIZE_15 | `{"N":1000,"TSTEPS":500}` | LARGE_DATASET | 500000 | - |
| SIZE_16 | `{"N":1200,"TSTEPS":600}` | new | 720000 | - |
| SIZE_17 | `{"N":1400,"TSTEPS":700}` | new | 980000 | - |
| SIZE_18 | `{"N":1600,"TSTEPS":800}` | new | 1280000 | - |
| SIZE_19 | `{"N":1800,"TSTEPS":900}` | new | 1620000 | - |
| SIZE_20 | `{"N":2000,"TSTEPS":1000}` | EXTRALARGE_DATASET | 2000000 | - |

### PolyBench/C / atax

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":38,"N":42}` | MINI_DATASET | 1596 | - |
| SIZE_02 | `{"M":58,"N":63}` | new | 3654 | - |
| SIZE_03 | `{"M":77,"N":83}` | new | 6391 | - |
| SIZE_04 | `{"M":97,"N":104}` | new | 10088 | - |
| SIZE_05 | `{"M":116,"N":124}` | SMALL_DATASET | 14384 | - |
| SIZE_06 | `{"M":171,"N":181}` | new | 30951 | - |
| SIZE_07 | `{"M":226,"N":238}` | new | 53788 | - |
| SIZE_08 | `{"M":280,"N":296}` | new | 82880 | - |
| SIZE_09 | `{"M":335,"N":353}` | new | 118255 | - |
| SIZE_10 | `{"M":390,"N":410}` | MEDIUM_DATASET | 159900 | - |
| SIZE_11 | `{"M":692,"N":748}` | new | 517616 | - |
| SIZE_12 | `{"M":994,"N":1086}` | new | 1079484 | - |
| SIZE_13 | `{"M":1296,"N":1424}` | new | 1845504 | - |
| SIZE_14 | `{"M":1598,"N":1762}` | new | 2815676 | - |
| SIZE_15 | `{"M":1900,"N":2100}` | LARGE_DATASET | 3990000 | - |
| SIZE_16 | `{"M":1880,"N":2120}` | new | 3985600 | - |
| SIZE_17 | `{"M":1860,"N":2140}` | new | 3980400 | - |
| SIZE_18 | `{"M":1840,"N":2160}` | new | 3974400 | - |
| SIZE_19 | `{"M":1820,"N":2180}` | new | 3967600 | - |
| SIZE_20 | `{"M":1800,"N":2200}` | EXTRALARGE_DATASET | 3960000 | - |

### PolyBench/C / bicg

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":38,"N":42}` | MINI_DATASET | 1596 | - |
| SIZE_02 | `{"M":58,"N":63}` | new | 3654 | - |
| SIZE_03 | `{"M":77,"N":83}` | new | 6391 | - |
| SIZE_04 | `{"M":97,"N":104}` | new | 10088 | - |
| SIZE_05 | `{"M":116,"N":124}` | SMALL_DATASET | 14384 | - |
| SIZE_06 | `{"M":171,"N":181}` | new | 30951 | - |
| SIZE_07 | `{"M":226,"N":238}` | new | 53788 | - |
| SIZE_08 | `{"M":280,"N":296}` | new | 82880 | - |
| SIZE_09 | `{"M":335,"N":353}` | new | 118255 | - |
| SIZE_10 | `{"M":390,"N":410}` | MEDIUM_DATASET | 159900 | - |
| SIZE_11 | `{"M":692,"N":748}` | new | 517616 | - |
| SIZE_12 | `{"M":994,"N":1086}` | new | 1079484 | - |
| SIZE_13 | `{"M":1296,"N":1424}` | new | 1845504 | - |
| SIZE_14 | `{"M":1598,"N":1762}` | new | 2815676 | - |
| SIZE_15 | `{"M":1900,"N":2100}` | LARGE_DATASET | 3990000 | - |
| SIZE_16 | `{"M":1880,"N":2120}` | new | 3985600 | - |
| SIZE_17 | `{"M":1860,"N":2140}` | new | 3980400 | - |
| SIZE_18 | `{"M":1840,"N":2160}` | new | 3974400 | - |
| SIZE_19 | `{"M":1820,"N":2180}` | new | 3967600 | - |
| SIZE_20 | `{"M":1800,"N":2200}` | EXTRALARGE_DATASET | 3960000 | - |

### PolyBench/C / cholesky

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / correlation

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":28,"N":32}` | MINI_DATASET | 896 | - |
| SIZE_02 | `{"M":41,"N":49}` | new | 2009 | - |
| SIZE_03 | `{"M":54,"N":66}` | new | 3564 | - |
| SIZE_04 | `{"M":67,"N":83}` | new | 5561 | - |
| SIZE_05 | `{"M":80,"N":100}` | SMALL_DATASET | 8000 | - |
| SIZE_06 | `{"M":112,"N":132}` | new | 14784 | - |
| SIZE_07 | `{"M":144,"N":164}` | new | 23616 | - |
| SIZE_08 | `{"M":176,"N":196}` | new | 34496 | - |
| SIZE_09 | `{"M":208,"N":228}` | new | 47424 | - |
| SIZE_10 | `{"M":240,"N":260}` | MEDIUM_DATASET | 62400 | - |
| SIZE_11 | `{"M":432,"N":488}` | new | 210816 | - |
| SIZE_12 | `{"M":624,"N":716}` | new | 446784 | - |
| SIZE_13 | `{"M":816,"N":944}` | new | 770304 | - |
| SIZE_14 | `{"M":1008,"N":1172}` | new | 1181376 | - |
| SIZE_15 | `{"M":1200,"N":1400}` | LARGE_DATASET | 1680000 | - |
| SIZE_16 | `{"M":1480,"N":1720}` | new | 2545600 | - |
| SIZE_17 | `{"M":1760,"N":2040}` | new | 3590400 | - |
| SIZE_18 | `{"M":2040,"N":2360}` | new | 4814400 | - |
| SIZE_19 | `{"M":2320,"N":2680}` | new | 6217600 | - |
| SIZE_20 | `{"M":2600,"N":3000}` | EXTRALARGE_DATASET | 7800000 | - |

### PolyBench/C / covariance

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":28,"N":32}` | MINI_DATASET | 896 | - |
| SIZE_02 | `{"M":41,"N":49}` | new | 2009 | - |
| SIZE_03 | `{"M":54,"N":66}` | new | 3564 | - |
| SIZE_04 | `{"M":67,"N":83}` | new | 5561 | - |
| SIZE_05 | `{"M":80,"N":100}` | SMALL_DATASET | 8000 | - |
| SIZE_06 | `{"M":112,"N":132}` | new | 14784 | - |
| SIZE_07 | `{"M":144,"N":164}` | new | 23616 | - |
| SIZE_08 | `{"M":176,"N":196}` | new | 34496 | - |
| SIZE_09 | `{"M":208,"N":228}` | new | 47424 | - |
| SIZE_10 | `{"M":240,"N":260}` | MEDIUM_DATASET | 62400 | - |
| SIZE_11 | `{"M":432,"N":488}` | new | 210816 | - |
| SIZE_12 | `{"M":624,"N":716}` | new | 446784 | - |
| SIZE_13 | `{"M":816,"N":944}` | new | 770304 | - |
| SIZE_14 | `{"M":1008,"N":1172}` | new | 1181376 | - |
| SIZE_15 | `{"M":1200,"N":1400}` | LARGE_DATASET | 1680000 | - |
| SIZE_16 | `{"M":1480,"N":1720}` | new | 2545600 | - |
| SIZE_17 | `{"M":1760,"N":2040}` | new | 3590400 | - |
| SIZE_18 | `{"M":2040,"N":2360}` | new | 4814400 | - |
| SIZE_19 | `{"M":2320,"N":2680}` | new | 6217600 | - |
| SIZE_20 | `{"M":2600,"N":3000}` | EXTRALARGE_DATASET | 7800000 | - |

### PolyBench/C / deriche

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"H":64,"W":64}` | MINI_DATASET | 4096 | - |
| SIZE_02 | `{"H":80,"W":96}` | new | 7680 | - |
| SIZE_03 | `{"H":96,"W":128}` | new | 12288 | - |
| SIZE_04 | `{"H":112,"W":160}` | new | 17920 | - |
| SIZE_05 | `{"H":128,"W":192}` | SMALL_DATASET | 24576 | - |
| SIZE_06 | `{"H":198,"W":298}` | new | 59004 | - |
| SIZE_07 | `{"H":269,"W":403}` | new | 108407 | - |
| SIZE_08 | `{"H":339,"W":509}` | new | 172551 | - |
| SIZE_09 | `{"H":410,"W":614}` | new | 251740 | - |
| SIZE_10 | `{"H":480,"W":720}` | MEDIUM_DATASET | 345600 | - |
| SIZE_11 | `{"H":816,"W":1395}` | new | 1138320 | - |
| SIZE_12 | `{"H":1152,"W":2070}` | new | 2384640 | - |
| SIZE_13 | `{"H":1488,"W":2746}` | new | 4086048 | - |
| SIZE_14 | `{"H":1824,"W":3421}` | new | 6239904 | - |
| SIZE_15 | `{"H":2160,"W":4096}` | LARGE_DATASET | 8847360 | - |
| SIZE_16 | `{"H":2592,"W":4813}` | new | 12475296 | - |
| SIZE_17 | `{"H":3024,"W":5530}` | new | 16722720 | - |
| SIZE_18 | `{"H":3456,"W":6246}` | new | 21586176 | - |
| SIZE_19 | `{"H":3888,"W":6963}` | new | 27072144 | - |
| SIZE_20 | `{"H":4320,"W":7680}` | EXTRALARGE_DATASET | 33177600 | - |

### PolyBench/C / doitgen

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"NP":12,"NQ":8,"NR":10}` | MINI_DATASET | 960 | - |
| SIZE_02 | `{"NP":17,"NQ":11,"NR":14}` | new | 2618 | - |
| SIZE_03 | `{"NP":21,"NQ":14,"NR":18}` | new | 5292 | - |
| SIZE_04 | `{"NP":26,"NQ":17,"NR":21}` | new | 9282 | - |
| SIZE_05 | `{"NP":30,"NQ":20,"NR":25}` | SMALL_DATASET | 15000 | - |
| SIZE_06 | `{"NP":36,"NQ":24,"NR":30}` | new | 25920 | - |
| SIZE_07 | `{"NP":42,"NQ":28,"NR":35}` | new | 41160 | - |
| SIZE_08 | `{"NP":48,"NQ":32,"NR":40}` | new | 61440 | - |
| SIZE_09 | `{"NP":54,"NQ":36,"NR":45}` | new | 87480 | - |
| SIZE_10 | `{"NP":60,"NQ":40,"NR":50}` | MEDIUM_DATASET | 120000 | - |
| SIZE_11 | `{"NP":80,"NQ":60,"NR":70}` | new | 336000 | - |
| SIZE_12 | `{"NP":100,"NQ":80,"NR":90}` | new | 720000 | - |
| SIZE_13 | `{"NP":120,"NQ":100,"NR":110}` | new | 1320000 | - |
| SIZE_14 | `{"NP":140,"NQ":120,"NR":130}` | new | 2184000 | - |
| SIZE_15 | `{"NP":160,"NQ":140,"NR":150}` | LARGE_DATASET | 3360000 | - |
| SIZE_16 | `{"NP":182,"NQ":156,"NR":170}` | new | 4826640 | - |
| SIZE_17 | `{"NP":204,"NQ":172,"NR":190}` | new | 6666720 | - |
| SIZE_18 | `{"NP":226,"NQ":188,"NR":210}` | new | 8922480 | - |
| SIZE_19 | `{"NP":248,"NQ":204,"NR":230}` | new | 11636160 | - |
| SIZE_20 | `{"NP":270,"NQ":220,"NR":250}` | EXTRALARGE_DATASET | 14850000 | - |

### PolyBench/C / durbin

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / fdtd-2d

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"NX":20,"NY":30,"TMAX":20}` | MINI_DATASET | 12000 | - |
| SIZE_02 | `{"NX":30,"NY":43,"TMAX":25}` | new | 32250 | - |
| SIZE_03 | `{"NX":40,"NY":55,"TMAX":30}` | new | 66000 | - |
| SIZE_04 | `{"NX":50,"NY":68,"TMAX":35}` | new | 119000 | - |
| SIZE_05 | `{"NX":60,"NY":80,"TMAX":40}` | SMALL_DATASET | 192000 | - |
| SIZE_06 | `{"NX":88,"NY":112,"TMAX":52}` | new | 512512 | - |
| SIZE_07 | `{"NX":116,"NY":144,"TMAX":64}` | new | 1069056 | - |
| SIZE_08 | `{"NX":144,"NY":176,"TMAX":76}` | new | 1926144 | - |
| SIZE_09 | `{"NX":172,"NY":208,"TMAX":88}` | new | 3148288 | - |
| SIZE_10 | `{"NX":200,"NY":240,"TMAX":100}` | MEDIUM_DATASET | 4800000 | - |
| SIZE_11 | `{"NX":360,"NY":432,"TMAX":180}` | new | 27993600 | - |
| SIZE_12 | `{"NX":520,"NY":624,"TMAX":260}` | new | 84364800 | - |
| SIZE_13 | `{"NX":680,"NY":816,"TMAX":340}` | new | 188659200 | - |
| SIZE_14 | `{"NX":840,"NY":1008,"TMAX":420}` | new | 355622400 | - |
| SIZE_15 | `{"NX":1000,"NY":1200,"TMAX":500}` | LARGE_DATASET | 600000000 | - |
| SIZE_16 | `{"NX":1200,"NY":1480,"TMAX":600}` | new | 1065600000 | - |
| SIZE_17 | `{"NX":1400,"NY":1760,"TMAX":700}` | new | 1724800000 | - |
| SIZE_18 | `{"NX":1600,"NY":2040,"TMAX":800}` | new | 2611200000 | - |
| SIZE_19 | `{"NX":1800,"NY":2320,"TMAX":900}` | new | 3758400000 | - |
| SIZE_20 | `{"NX":2000,"NY":2600,"TMAX":1000}` | EXTRALARGE_DATASET | 5200000000 | - |

### PolyBench/C / floyd-warshall

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":60}` | MINI_DATASET | 60 | - |
| SIZE_02 | `{"N":90}` | new | 90 | - |
| SIZE_03 | `{"N":120}` | new | 120 | - |
| SIZE_04 | `{"N":150}` | new | 150 | - |
| SIZE_05 | `{"N":180}` | SMALL_DATASET | 180 | - |
| SIZE_06 | `{"N":244}` | new | 244 | - |
| SIZE_07 | `{"N":308}` | new | 308 | - |
| SIZE_08 | `{"N":372}` | new | 372 | - |
| SIZE_09 | `{"N":436}` | new | 436 | - |
| SIZE_10 | `{"N":500}` | MEDIUM_DATASET | 500 | - |
| SIZE_11 | `{"N":960}` | new | 960 | - |
| SIZE_12 | `{"N":1420}` | new | 1420 | - |
| SIZE_13 | `{"N":1880}` | new | 1880 | - |
| SIZE_14 | `{"N":2340}` | new | 2340 | - |
| SIZE_15 | `{"N":2800}` | LARGE_DATASET | 2800 | - |
| SIZE_16 | `{"N":3360}` | new | 3360 | - |
| SIZE_17 | `{"N":3920}` | new | 3920 | - |
| SIZE_18 | `{"N":4480}` | new | 4480 | - |
| SIZE_19 | `{"N":5040}` | new | 5040 | - |
| SIZE_20 | `{"N":5600}` | EXTRALARGE_DATASET | 5600 | - |

### PolyBench/C / gemm

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"NI":20,"NJ":25,"NK":30}` | MINI_DATASET | 15000 | - |
| SIZE_02 | `{"NI":30,"NJ":36,"NK":43}` | new | 46440 | - |
| SIZE_03 | `{"NI":40,"NJ":48,"NK":55}` | new | 105600 | - |
| SIZE_04 | `{"NI":50,"NJ":59,"NK":68}` | new | 200600 | - |
| SIZE_05 | `{"NI":60,"NJ":70,"NK":80}` | SMALL_DATASET | 336000 | - |
| SIZE_06 | `{"NI":88,"NJ":100,"NK":112}` | new | 985600 | - |
| SIZE_07 | `{"NI":116,"NJ":130,"NK":144}` | new | 2171520 | - |
| SIZE_08 | `{"NI":144,"NJ":160,"NK":176}` | new | 4055040 | - |
| SIZE_09 | `{"NI":172,"NJ":190,"NK":208}` | new | 6797440 | - |
| SIZE_10 | `{"NI":200,"NJ":220,"NK":240}` | MEDIUM_DATASET | 10560000 | - |
| SIZE_11 | `{"NI":360,"NJ":396,"NK":432}` | new | 61585920 | - |
| SIZE_12 | `{"NI":520,"NJ":572,"NK":624}` | new | 185602560 | - |
| SIZE_13 | `{"NI":680,"NJ":748,"NK":816}` | new | 415050240 | - |
| SIZE_14 | `{"NI":840,"NJ":924,"NK":1008}` | new | 782369280 | - |
| SIZE_15 | `{"NI":1000,"NJ":1100,"NK":1200}` | LARGE_DATASET | 1320000000 | - |
| SIZE_16 | `{"NI":1200,"NJ":1340,"NK":1480}` | new | 2379840000 | - |
| SIZE_17 | `{"NI":1400,"NJ":1580,"NK":1760}` | new | 3893120000 | - |
| SIZE_18 | `{"NI":1600,"NJ":1820,"NK":2040}` | new | 5940480000 | - |
| SIZE_19 | `{"NI":1800,"NJ":2060,"NK":2320}` | new | 8602560000 | - |
| SIZE_20 | `{"NI":2000,"NJ":2300,"NK":2600}` | EXTRALARGE_DATASET | 11960000000 | - |

### PolyBench/C / gemver

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / gesummv

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":30}` | MINI_DATASET | 30 | - |
| SIZE_02 | `{"N":45}` | new | 45 | - |
| SIZE_03 | `{"N":60}` | new | 60 | - |
| SIZE_04 | `{"N":75}` | new | 75 | - |
| SIZE_05 | `{"N":90}` | SMALL_DATASET | 90 | - |
| SIZE_06 | `{"N":122}` | new | 122 | - |
| SIZE_07 | `{"N":154}` | new | 154 | - |
| SIZE_08 | `{"N":186}` | new | 186 | - |
| SIZE_09 | `{"N":218}` | new | 218 | - |
| SIZE_10 | `{"N":250}` | MEDIUM_DATASET | 250 | - |
| SIZE_11 | `{"N":460}` | new | 460 | - |
| SIZE_12 | `{"N":670}` | new | 670 | - |
| SIZE_13 | `{"N":880}` | new | 880 | - |
| SIZE_14 | `{"N":1090}` | new | 1090 | - |
| SIZE_15 | `{"N":1300}` | LARGE_DATASET | 1300 | - |
| SIZE_16 | `{"N":1600}` | new | 1600 | - |
| SIZE_17 | `{"N":1900}` | new | 1900 | - |
| SIZE_18 | `{"N":2200}` | new | 2200 | - |
| SIZE_19 | `{"N":2500}` | new | 2500 | - |
| SIZE_20 | `{"N":2800}` | EXTRALARGE_DATASET | 2800 | - |

### PolyBench/C / gramschmidt

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":20,"N":30}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"M":30,"N":43}` | new | 1290 | - |
| SIZE_03 | `{"M":40,"N":55}` | new | 2200 | - |
| SIZE_04 | `{"M":50,"N":68}` | new | 3400 | - |
| SIZE_05 | `{"M":60,"N":80}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"M":88,"N":112}` | new | 9856 | - |
| SIZE_07 | `{"M":116,"N":144}` | new | 16704 | - |
| SIZE_08 | `{"M":144,"N":176}` | new | 25344 | - |
| SIZE_09 | `{"M":172,"N":208}` | new | 35776 | - |
| SIZE_10 | `{"M":200,"N":240}` | MEDIUM_DATASET | 48000 | - |
| SIZE_11 | `{"M":360,"N":432}` | new | 155520 | - |
| SIZE_12 | `{"M":520,"N":624}` | new | 324480 | - |
| SIZE_13 | `{"M":680,"N":816}` | new | 554880 | - |
| SIZE_14 | `{"M":840,"N":1008}` | new | 846720 | - |
| SIZE_15 | `{"M":1000,"N":1200}` | LARGE_DATASET | 1200000 | - |
| SIZE_16 | `{"M":1200,"N":1480}` | new | 1776000 | - |
| SIZE_17 | `{"M":1400,"N":1760}` | new | 2464000 | - |
| SIZE_18 | `{"M":1600,"N":2040}` | new | 3264000 | - |
| SIZE_19 | `{"M":1800,"N":2320}` | new | 4176000 | - |
| SIZE_20 | `{"M":2000,"N":2600}` | EXTRALARGE_DATASET | 5200000 | - |

### PolyBench/C / heat-3d

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":10,"TSTEPS":20}` | MINI_DATASET | 200 | - |
| SIZE_02 | `{"N":13,"TSTEPS":25}` | new | 325 | - |
| SIZE_03 | `{"N":15,"TSTEPS":30}` | new | 450 | - |
| SIZE_04 | `{"N":18,"TSTEPS":35}` | new | 630 | - |
| SIZE_05 | `{"N":20,"TSTEPS":40}` | SMALL_DATASET | 800 | - |
| SIZE_06 | `{"N":24,"TSTEPS":52}` | new | 1248 | - |
| SIZE_07 | `{"N":28,"TSTEPS":64}` | new | 1792 | - |
| SIZE_08 | `{"N":32,"TSTEPS":76}` | new | 2432 | - |
| SIZE_09 | `{"N":36,"TSTEPS":88}` | new | 3168 | - |
| SIZE_10 | `{"N":40,"TSTEPS":100}` | MEDIUM_DATASET | 4000 | - |
| SIZE_11 | `{"N":56,"TSTEPS":180}` | new | 10080 | - |
| SIZE_12 | `{"N":72,"TSTEPS":260}` | new | 18720 | - |
| SIZE_13 | `{"N":88,"TSTEPS":340}` | new | 29920 | - |
| SIZE_14 | `{"N":104,"TSTEPS":420}` | new | 43680 | - |
| SIZE_15 | `{"N":120,"TSTEPS":500}` | LARGE_DATASET | 60000 | - |
| SIZE_16 | `{"N":136,"TSTEPS":600}` | new | 81600 | - |
| SIZE_17 | `{"N":152,"TSTEPS":700}` | new | 106400 | - |
| SIZE_18 | `{"N":168,"TSTEPS":800}` | new | 134400 | - |
| SIZE_19 | `{"N":184,"TSTEPS":900}` | new | 165600 | - |
| SIZE_20 | `{"N":200,"TSTEPS":1000}` | EXTRALARGE_DATASET | 200000 | - |

### PolyBench/C / jacobi-1d

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":30,"TSTEPS":20}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"N":53,"TSTEPS":25}` | new | 1325 | - |
| SIZE_03 | `{"N":75,"TSTEPS":30}` | new | 2250 | - |
| SIZE_04 | `{"N":98,"TSTEPS":35}` | new | 3430 | - |
| SIZE_05 | `{"N":120,"TSTEPS":40}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"N":176,"TSTEPS":52}` | new | 9152 | - |
| SIZE_07 | `{"N":232,"TSTEPS":64}` | new | 14848 | - |
| SIZE_08 | `{"N":288,"TSTEPS":76}` | new | 21888 | - |
| SIZE_09 | `{"N":344,"TSTEPS":88}` | new | 30272 | - |
| SIZE_10 | `{"N":400,"TSTEPS":100}` | MEDIUM_DATASET | 40000 | - |
| SIZE_11 | `{"N":720,"TSTEPS":180}` | new | 129600 | - |
| SIZE_12 | `{"N":1040,"TSTEPS":260}` | new | 270400 | - |
| SIZE_13 | `{"N":1360,"TSTEPS":340}` | new | 462400 | - |
| SIZE_14 | `{"N":1680,"TSTEPS":420}` | new | 705600 | - |
| SIZE_15 | `{"N":2000,"TSTEPS":500}` | LARGE_DATASET | 1000000 | - |
| SIZE_16 | `{"N":2400,"TSTEPS":600}` | new | 1440000 | - |
| SIZE_17 | `{"N":2800,"TSTEPS":700}` | new | 1960000 | - |
| SIZE_18 | `{"N":3200,"TSTEPS":800}` | new | 2560000 | - |
| SIZE_19 | `{"N":3600,"TSTEPS":900}` | new | 3240000 | - |
| SIZE_20 | `{"N":4000,"TSTEPS":1000}` | EXTRALARGE_DATASET | 4000000 | - |

### PolyBench/C / jacobi-2d

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":30,"TSTEPS":20}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"N":45,"TSTEPS":25}` | new | 1125 | - |
| SIZE_03 | `{"N":60,"TSTEPS":30}` | new | 1800 | - |
| SIZE_04 | `{"N":75,"TSTEPS":35}` | new | 2625 | - |
| SIZE_05 | `{"N":90,"TSTEPS":40}` | SMALL_DATASET | 3600 | - |
| SIZE_06 | `{"N":122,"TSTEPS":52}` | new | 6344 | - |
| SIZE_07 | `{"N":154,"TSTEPS":64}` | new | 9856 | - |
| SIZE_08 | `{"N":186,"TSTEPS":76}` | new | 14136 | - |
| SIZE_09 | `{"N":218,"TSTEPS":88}` | new | 19184 | - |
| SIZE_10 | `{"N":250,"TSTEPS":100}` | MEDIUM_DATASET | 25000 | - |
| SIZE_11 | `{"N":460,"TSTEPS":180}` | new | 82800 | - |
| SIZE_12 | `{"N":670,"TSTEPS":260}` | new | 174200 | - |
| SIZE_13 | `{"N":880,"TSTEPS":340}` | new | 299200 | - |
| SIZE_14 | `{"N":1090,"TSTEPS":420}` | new | 457800 | - |
| SIZE_15 | `{"N":1300,"TSTEPS":500}` | LARGE_DATASET | 650000 | - |
| SIZE_16 | `{"N":1600,"TSTEPS":600}` | new | 960000 | - |
| SIZE_17 | `{"N":1900,"TSTEPS":700}` | new | 1330000 | - |
| SIZE_18 | `{"N":2200,"TSTEPS":800}` | new | 1760000 | - |
| SIZE_19 | `{"N":2500,"TSTEPS":900}` | new | 2250000 | - |
| SIZE_20 | `{"N":2800,"TSTEPS":1000}` | EXTRALARGE_DATASET | 2800000 | - |

### PolyBench/C / lu

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / ludcmp

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / mvt

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / nussinov

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":60}` | MINI_DATASET | 60 | - |
| SIZE_02 | `{"N":90}` | new | 90 | - |
| SIZE_03 | `{"N":120}` | new | 120 | - |
| SIZE_04 | `{"N":150}` | new | 150 | - |
| SIZE_05 | `{"N":180}` | SMALL_DATASET | 180 | - |
| SIZE_06 | `{"N":244}` | new | 244 | - |
| SIZE_07 | `{"N":308}` | new | 308 | - |
| SIZE_08 | `{"N":372}` | new | 372 | - |
| SIZE_09 | `{"N":436}` | new | 436 | - |
| SIZE_10 | `{"N":500}` | MEDIUM_DATASET | 500 | - |
| SIZE_11 | `{"N":900}` | new | 900 | - |
| SIZE_12 | `{"N":1300}` | new | 1300 | - |
| SIZE_13 | `{"N":1700}` | new | 1700 | - |
| SIZE_14 | `{"N":2100}` | new | 2100 | - |
| SIZE_15 | `{"N":2500}` | LARGE_DATASET | 2500 | - |
| SIZE_16 | `{"N":3100}` | new | 3100 | - |
| SIZE_17 | `{"N":3700}` | new | 3700 | - |
| SIZE_18 | `{"N":4300}` | new | 4300 | - |
| SIZE_19 | `{"N":4900}` | new | 4900 | - |
| SIZE_20 | `{"N":5500}` | EXTRALARGE_DATASET | 5500 | - |

### PolyBench/C / seidel-2d

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40,"TSTEPS":20}` | MINI_DATASET | 800 | - |
| SIZE_02 | `{"N":60,"TSTEPS":25}` | new | 1500 | - |
| SIZE_03 | `{"N":80,"TSTEPS":30}` | new | 2400 | - |
| SIZE_04 | `{"N":100,"TSTEPS":35}` | new | 3500 | - |
| SIZE_05 | `{"N":120,"TSTEPS":40}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"N":176,"TSTEPS":52}` | new | 9152 | - |
| SIZE_07 | `{"N":232,"TSTEPS":64}` | new | 14848 | - |
| SIZE_08 | `{"N":288,"TSTEPS":76}` | new | 21888 | - |
| SIZE_09 | `{"N":344,"TSTEPS":88}` | new | 30272 | - |
| SIZE_10 | `{"N":400,"TSTEPS":100}` | MEDIUM_DATASET | 40000 | - |
| SIZE_11 | `{"N":720,"TSTEPS":180}` | new | 129600 | - |
| SIZE_12 | `{"N":1040,"TSTEPS":260}` | new | 270400 | - |
| SIZE_13 | `{"N":1360,"TSTEPS":340}` | new | 462400 | - |
| SIZE_14 | `{"N":1680,"TSTEPS":420}` | new | 705600 | - |
| SIZE_15 | `{"N":2000,"TSTEPS":500}` | LARGE_DATASET | 1000000 | - |
| SIZE_16 | `{"N":2400,"TSTEPS":600}` | new | 1440000 | - |
| SIZE_17 | `{"N":2800,"TSTEPS":700}` | new | 1960000 | - |
| SIZE_18 | `{"N":3200,"TSTEPS":800}` | new | 2560000 | - |
| SIZE_19 | `{"N":3600,"TSTEPS":900}` | new | 3240000 | - |
| SIZE_20 | `{"N":4000,"TSTEPS":1000}` | EXTRALARGE_DATASET | 4000000 | - |

### PolyBench/C / symm

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":20,"N":30}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"M":30,"N":43}` | new | 1290 | - |
| SIZE_03 | `{"M":40,"N":55}` | new | 2200 | - |
| SIZE_04 | `{"M":50,"N":68}` | new | 3400 | - |
| SIZE_05 | `{"M":60,"N":80}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"M":88,"N":112}` | new | 9856 | - |
| SIZE_07 | `{"M":116,"N":144}` | new | 16704 | - |
| SIZE_08 | `{"M":144,"N":176}` | new | 25344 | - |
| SIZE_09 | `{"M":172,"N":208}` | new | 35776 | - |
| SIZE_10 | `{"M":200,"N":240}` | MEDIUM_DATASET | 48000 | - |
| SIZE_11 | `{"M":360,"N":432}` | new | 155520 | - |
| SIZE_12 | `{"M":520,"N":624}` | new | 324480 | - |
| SIZE_13 | `{"M":680,"N":816}` | new | 554880 | - |
| SIZE_14 | `{"M":840,"N":1008}` | new | 846720 | - |
| SIZE_15 | `{"M":1000,"N":1200}` | LARGE_DATASET | 1200000 | - |
| SIZE_16 | `{"M":1200,"N":1480}` | new | 1776000 | - |
| SIZE_17 | `{"M":1400,"N":1760}` | new | 2464000 | - |
| SIZE_18 | `{"M":1600,"N":2040}` | new | 3264000 | - |
| SIZE_19 | `{"M":1800,"N":2320}` | new | 4176000 | - |
| SIZE_20 | `{"M":2000,"N":2600}` | EXTRALARGE_DATASET | 5200000 | - |

### PolyBench/C / syr2k

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":20,"N":30}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"M":30,"N":43}` | new | 1290 | - |
| SIZE_03 | `{"M":40,"N":55}` | new | 2200 | - |
| SIZE_04 | `{"M":50,"N":68}` | new | 3400 | - |
| SIZE_05 | `{"M":60,"N":80}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"M":88,"N":112}` | new | 9856 | - |
| SIZE_07 | `{"M":116,"N":144}` | new | 16704 | - |
| SIZE_08 | `{"M":144,"N":176}` | new | 25344 | - |
| SIZE_09 | `{"M":172,"N":208}` | new | 35776 | - |
| SIZE_10 | `{"M":200,"N":240}` | MEDIUM_DATASET | 48000 | - |
| SIZE_11 | `{"M":360,"N":432}` | new | 155520 | - |
| SIZE_12 | `{"M":520,"N":624}` | new | 324480 | - |
| SIZE_13 | `{"M":680,"N":816}` | new | 554880 | - |
| SIZE_14 | `{"M":840,"N":1008}` | new | 846720 | - |
| SIZE_15 | `{"M":1000,"N":1200}` | LARGE_DATASET | 1200000 | - |
| SIZE_16 | `{"M":1200,"N":1480}` | new | 1776000 | - |
| SIZE_17 | `{"M":1400,"N":1760}` | new | 2464000 | - |
| SIZE_18 | `{"M":1600,"N":2040}` | new | 3264000 | - |
| SIZE_19 | `{"M":1800,"N":2320}` | new | 4176000 | - |
| SIZE_20 | `{"M":2000,"N":2600}` | EXTRALARGE_DATASET | 5200000 | - |

### PolyBench/C / syrk

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":20,"N":30}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"M":30,"N":43}` | new | 1290 | - |
| SIZE_03 | `{"M":40,"N":55}` | new | 2200 | - |
| SIZE_04 | `{"M":50,"N":68}` | new | 3400 | - |
| SIZE_05 | `{"M":60,"N":80}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"M":88,"N":112}` | new | 9856 | - |
| SIZE_07 | `{"M":116,"N":144}` | new | 16704 | - |
| SIZE_08 | `{"M":144,"N":176}` | new | 25344 | - |
| SIZE_09 | `{"M":172,"N":208}` | new | 35776 | - |
| SIZE_10 | `{"M":200,"N":240}` | MEDIUM_DATASET | 48000 | - |
| SIZE_11 | `{"M":360,"N":432}` | new | 155520 | - |
| SIZE_12 | `{"M":520,"N":624}` | new | 324480 | - |
| SIZE_13 | `{"M":680,"N":816}` | new | 554880 | - |
| SIZE_14 | `{"M":840,"N":1008}` | new | 846720 | - |
| SIZE_15 | `{"M":1000,"N":1200}` | LARGE_DATASET | 1200000 | - |
| SIZE_16 | `{"M":1200,"N":1480}` | new | 1776000 | - |
| SIZE_17 | `{"M":1400,"N":1760}` | new | 2464000 | - |
| SIZE_18 | `{"M":1600,"N":2040}` | new | 3264000 | - |
| SIZE_19 | `{"M":1800,"N":2320}` | new | 4176000 | - |
| SIZE_20 | `{"M":2000,"N":2600}` | EXTRALARGE_DATASET | 5200000 | - |

### PolyBench/C / trisolv

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"N":40}` | MINI_DATASET | 40 | - |
| SIZE_02 | `{"N":60}` | new | 60 | - |
| SIZE_03 | `{"N":80}` | new | 80 | - |
| SIZE_04 | `{"N":100}` | new | 100 | - |
| SIZE_05 | `{"N":120}` | SMALL_DATASET | 120 | - |
| SIZE_06 | `{"N":176}` | new | 176 | - |
| SIZE_07 | `{"N":232}` | new | 232 | - |
| SIZE_08 | `{"N":288}` | new | 288 | - |
| SIZE_09 | `{"N":344}` | new | 344 | - |
| SIZE_10 | `{"N":400}` | MEDIUM_DATASET | 400 | - |
| SIZE_11 | `{"N":720}` | new | 720 | - |
| SIZE_12 | `{"N":1040}` | new | 1040 | - |
| SIZE_13 | `{"N":1360}` | new | 1360 | - |
| SIZE_14 | `{"N":1680}` | new | 1680 | - |
| SIZE_15 | `{"N":2000}` | LARGE_DATASET | 2000 | - |
| SIZE_16 | `{"N":2400}` | new | 2400 | - |
| SIZE_17 | `{"N":2800}` | new | 2800 | - |
| SIZE_18 | `{"N":3200}` | new | 3200 | - |
| SIZE_19 | `{"N":3600}` | new | 3600 | - |
| SIZE_20 | `{"N":4000}` | EXTRALARGE_DATASET | 4000 | - |

### PolyBench/C / trmm

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"M":20,"N":30}` | MINI_DATASET | 600 | - |
| SIZE_02 | `{"M":30,"N":43}` | new | 1290 | - |
| SIZE_03 | `{"M":40,"N":55}` | new | 2200 | - |
| SIZE_04 | `{"M":50,"N":68}` | new | 3400 | - |
| SIZE_05 | `{"M":60,"N":80}` | SMALL_DATASET | 4800 | - |
| SIZE_06 | `{"M":88,"N":112}` | new | 9856 | - |
| SIZE_07 | `{"M":116,"N":144}` | new | 16704 | - |
| SIZE_08 | `{"M":144,"N":176}` | new | 25344 | - |
| SIZE_09 | `{"M":172,"N":208}` | new | 35776 | - |
| SIZE_10 | `{"M":200,"N":240}` | MEDIUM_DATASET | 48000 | - |
| SIZE_11 | `{"M":360,"N":432}` | new | 155520 | - |
| SIZE_12 | `{"M":520,"N":624}` | new | 324480 | - |
| SIZE_13 | `{"M":680,"N":816}` | new | 554880 | - |
| SIZE_14 | `{"M":840,"N":1008}` | new | 846720 | - |
| SIZE_15 | `{"M":1000,"N":1200}` | LARGE_DATASET | 1200000 | - |
| SIZE_16 | `{"M":1200,"N":1480}` | new | 1776000 | - |
| SIZE_17 | `{"M":1400,"N":1760}` | new | 2464000 | - |
| SIZE_18 | `{"M":1600,"N":2040}` | new | 3264000 | - |
| SIZE_19 | `{"M":1800,"N":2320}` | new | 4176000 | - |
| SIZE_20 | `{"M":2000,"N":2600}` | EXTRALARGE_DATASET | 5200000 | - |

### Rodinia CUDA / b+tree

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"queries":128,"records":1024,"work_group_size":256}` | new | 1152 | ../../data/generated_20/b+tree/keys_SIZE_01.txt<br>../../data/generated_20/b+tree/commands_SIZE_01.txt |
| SIZE_02 | `{"queries":209,"records":1674,"work_group_size":256}` | new | 1883 | ../../data/generated_20/b+tree/keys_SIZE_02.txt<br>../../data/generated_20/b+tree/commands_SIZE_02.txt |
| SIZE_03 | `{"queries":342,"records":2738,"work_group_size":256}` | new | 3080 | ../../data/generated_20/b+tree/keys_SIZE_03.txt<br>../../data/generated_20/b+tree/commands_SIZE_03.txt |
| SIZE_04 | `{"queries":559,"records":4477,"work_group_size":256}` | new | 5036 | ../../data/generated_20/b+tree/keys_SIZE_04.txt<br>../../data/generated_20/b+tree/commands_SIZE_04.txt |
| SIZE_05 | `{"queries":915,"records":7320,"work_group_size":256}` | new | 8235 | ../../data/generated_20/b+tree/keys_SIZE_05.txt<br>../../data/generated_20/b+tree/commands_SIZE_05.txt |
| SIZE_06 | `{"queries":1496,"records":11969,"work_group_size":256}` | new | 13465 | ../../data/generated_20/b+tree/keys_SIZE_06.txt<br>../../data/generated_20/b+tree/commands_SIZE_06.txt |
| SIZE_07 | `{"queries":2446,"records":19570,"work_group_size":256}` | new | 22016 | ../../data/generated_20/b+tree/keys_SIZE_07.txt<br>../../data/generated_20/b+tree/commands_SIZE_07.txt |
| SIZE_08 | `{"queries":4000,"records":32000,"work_group_size":256}` | new | 36000 | ../../data/generated_20/b+tree/keys_SIZE_08.txt<br>../../data/generated_20/b+tree/commands_SIZE_08.txt |
| SIZE_09 | `{"queries":6540,"records":52324,"work_group_size":256}` | new | 58864 | ../../data/generated_20/b+tree/keys_SIZE_09.txt<br>../../data/generated_20/b+tree/commands_SIZE_09.txt |
| SIZE_10 | `{"queries":10694,"records":85556,"work_group_size":256}` | new | 96250 | ../../data/generated_20/b+tree/keys_SIZE_10.txt<br>../../data/generated_20/b+tree/commands_SIZE_10.txt |
| SIZE_11 | `{"queries":17486,"records":139894,"work_group_size":256}` | new | 157380 | ../../data/generated_20/b+tree/keys_SIZE_11.txt<br>../../data/generated_20/b+tree/commands_SIZE_11.txt |
| SIZE_12 | `{"queries":28593,"records":228744,"work_group_size":256}` | new | 257337 | ../../data/generated_20/b+tree/keys_SIZE_12.txt<br>../../data/generated_20/b+tree/commands_SIZE_12.txt |
| SIZE_13 | `{"queries":46753,"records":374024,"work_group_size":256}` | new | 420777 | ../../data/generated_20/b+tree/keys_SIZE_13.txt<br>../../data/generated_20/b+tree/commands_SIZE_13.txt |
| SIZE_14 | `{"queries":50000,"records":611575,"work_group_size":256}` | new | 661575 | ../../data/generated_20/b+tree/keys_SIZE_14.txt<br>../../data/generated_20/b+tree/commands_SIZE_14.txt |
| SIZE_15 | `{"queries":50000,"records":1000000,"work_group_size":256}` | new; parameter anchor=mil.txt+command.txt | 1050000 | ../../data/generated_20/b+tree/keys_SIZE_15.txt<br>../../data/generated_20/b+tree/commands_SIZE_15.txt |
| SIZE_16 | `{"queries":50000,"records":1148698,"work_group_size":256}` | new | 1198698 | ../../data/generated_20/b+tree/keys_SIZE_16.txt<br>../../data/generated_20/b+tree/commands_SIZE_16.txt |
| SIZE_17 | `{"queries":50000,"records":1319508,"work_group_size":256}` | new | 1369508 | ../../data/generated_20/b+tree/keys_SIZE_17.txt<br>../../data/generated_20/b+tree/commands_SIZE_17.txt |
| SIZE_18 | `{"queries":50000,"records":1515717,"work_group_size":256}` | new | 1565717 | ../../data/generated_20/b+tree/keys_SIZE_18.txt<br>../../data/generated_20/b+tree/commands_SIZE_18.txt |
| SIZE_19 | `{"queries":50000,"records":1741101,"work_group_size":256}` | new | 1791101 | ../../data/generated_20/b+tree/keys_SIZE_19.txt<br>../../data/generated_20/b+tree/commands_SIZE_19.txt |
| SIZE_20 | `{"queries":50000,"records":2000000,"work_group_size":256}` | new | 2050000 | ../../data/generated_20/b+tree/keys_SIZE_20.txt<br>../../data/generated_20/b+tree/commands_SIZE_20.txt |

### Rodinia CUDA / backprop

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"input_layer_size":1024}` | layer-1024 | 1024 | - |
| SIZE_02 | `{"input_layer_size":1376}` | new | 1376 | - |
| SIZE_03 | `{"input_layer_size":1856}` | new | 1856 | - |
| SIZE_04 | `{"input_layer_size":2496}` | new | 2496 | - |
| SIZE_05 | `{"input_layer_size":3360}` | new | 3360 | - |
| SIZE_06 | `{"input_layer_size":4528}` | new | 4528 | - |
| SIZE_07 | `{"input_layer_size":6080}` | new | 6080 | - |
| SIZE_08 | `{"input_layer_size":8192}` | new | 8192 | - |
| SIZE_09 | `{"input_layer_size":11024}` | new | 11024 | - |
| SIZE_10 | `{"input_layer_size":14832}` | new | 14832 | - |
| SIZE_11 | `{"input_layer_size":19968}` | new | 19968 | - |
| SIZE_12 | `{"input_layer_size":26880}` | new | 26880 | - |
| SIZE_13 | `{"input_layer_size":36176}` | new | 36176 | - |
| SIZE_14 | `{"input_layer_size":48688}` | new | 48688 | - |
| SIZE_15 | `{"input_layer_size":65536}` | layer-65536 | 65536 | - |
| SIZE_16 | `{"input_layer_size":75280}` | new | 75280 | - |
| SIZE_17 | `{"input_layer_size":86480}` | new | 86480 | - |
| SIZE_18 | `{"input_layer_size":99328}` | new | 99328 | - |
| SIZE_19 | `{"input_layer_size":114112}` | new | 114112 | - |
| SIZE_20 | `{"input_layer_size":131072}` | new | 131072 | - |

### Rodinia CUDA / bfs

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"average_degree":6,"edges":24576,"nodes":4096}` | new; parameter anchor=graph4096.txt | 28672 | ../../data/generated_20/bfs/graph_SIZE_01.txt |
| SIZE_02 | `{"average_degree":6,"edges":36396,"nodes":6066}` | new | 42462 | ../../data/generated_20/bfs/graph_SIZE_02.txt |
| SIZE_03 | `{"average_degree":6,"edges":53904,"nodes":8984}` | new | 62888 | ../../data/generated_20/bfs/graph_SIZE_03.txt |
| SIZE_04 | `{"average_degree":6,"edges":79824,"nodes":13304}` | new | 93128 | ../../data/generated_20/bfs/graph_SIZE_04.txt |
| SIZE_05 | `{"average_degree":6,"edges":118218,"nodes":19703}` | new | 137921 | ../../data/generated_20/bfs/graph_SIZE_05.txt |
| SIZE_06 | `{"average_degree":6,"edges":175080,"nodes":29180}` | new | 204260 | ../../data/generated_20/bfs/graph_SIZE_06.txt |
| SIZE_07 | `{"average_degree":6,"edges":259290,"nodes":43215}` | new | 302505 | ../../data/generated_20/bfs/graph_SIZE_07.txt |
| SIZE_08 | `{"average_degree":6,"edges":384000,"nodes":64000}` | new | 448000 | ../../data/generated_20/bfs/graph_SIZE_08.txt |
| SIZE_09 | `{"average_degree":6,"edges":568692,"nodes":94782}` | new | 663474 | ../../data/generated_20/bfs/graph_SIZE_09.txt |
| SIZE_10 | `{"average_degree":6,"edges":842214,"nodes":140369}` | new | 982583 | ../../data/generated_20/bfs/graph_SIZE_10.txt |
| SIZE_11 | `{"average_degree":6,"edges":1247292,"nodes":207882}` | new | 1455174 | ../../data/generated_20/bfs/graph_SIZE_11.txt |
| SIZE_12 | `{"average_degree":6,"edges":1847202,"nodes":307867}` | new | 2155069 | ../../data/generated_20/bfs/graph_SIZE_12.txt |
| SIZE_13 | `{"average_degree":6,"edges":2735646,"nodes":455941}` | new | 3191587 | ../../data/generated_20/bfs/graph_SIZE_13.txt |
| SIZE_14 | `{"average_degree":6,"edges":4051404,"nodes":675234}` | new | 4726638 | ../../data/generated_20/bfs/graph_SIZE_14.txt |
| SIZE_15 | `{"average_degree":6,"edges":6000000,"nodes":1000000}` | new; parameter anchor=graph1MW_6.txt | 7000000 | ../../data/generated_20/bfs/graph_SIZE_15.txt |
| SIZE_16 | `{"average_degree":6,"edges":6892188,"nodes":1148698}` | new | 8040886 | ../../data/generated_20/bfs/graph_SIZE_16.txt |
| SIZE_17 | `{"average_degree":6,"edges":7917048,"nodes":1319508}` | new | 9236556 | ../../data/generated_20/bfs/graph_SIZE_17.txt |
| SIZE_18 | `{"average_degree":6,"edges":9094302,"nodes":1515717}` | new | 10610019 | ../../data/generated_20/bfs/graph_SIZE_18.txt |
| SIZE_19 | `{"average_degree":6,"edges":10446606,"nodes":1741101}` | new | 12187707 | ../../data/generated_20/bfs/graph_SIZE_19.txt |
| SIZE_20 | `{"average_degree":6,"edges":12000000,"nodes":2000000}` | new | 14000000 | ../../data/generated_20/bfs/graph_SIZE_20.txt |

### Rodinia CUDA / cfd

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"faces_per_cell":4,"mesh_cells":8192}` | new | 8192 | ../../data/generated_20/cfd/mesh_SIZE_01.dat |
| SIZE_02 | `{"faces_per_cell":4,"mesh_cells":10781}` | new | 10781 | ../../data/generated_20/cfd/mesh_SIZE_02.dat |
| SIZE_03 | `{"faces_per_cell":4,"mesh_cells":14189}` | new | 14189 | ../../data/generated_20/cfd/mesh_SIZE_03.dat |
| SIZE_04 | `{"faces_per_cell":4,"mesh_cells":18675}` | new | 18675 | ../../data/generated_20/cfd/mesh_SIZE_04.dat |
| SIZE_05 | `{"faces_per_cell":4,"mesh_cells":24578}` | new | 24578 | ../../data/generated_20/cfd/mesh_SIZE_05.dat |
| SIZE_06 | `{"faces_per_cell":4,"mesh_cells":32347}` | new | 32347 | ../../data/generated_20/cfd/mesh_SIZE_06.dat |
| SIZE_07 | `{"faces_per_cell":4,"mesh_cells":42571}` | new | 42571 | ../../data/generated_20/cfd/mesh_SIZE_07.dat |
| SIZE_08 | `{"faces_per_cell":4,"mesh_cells":56028}` | new | 56028 | ../../data/generated_20/cfd/mesh_SIZE_08.dat |
| SIZE_09 | `{"faces_per_cell":4,"mesh_cells":73738}` | new | 73738 | ../../data/generated_20/cfd/mesh_SIZE_09.dat |
| SIZE_10 | `{"faces_per_cell":4,"mesh_cells":97046}` | new; parameter anchor=fvcorr.domn.097K | 97046 | ../../data/generated_20/cfd/mesh_SIZE_10.dat |
| SIZE_11 | `{"faces_per_cell":4,"mesh_cells":111351}` | new | 111351 | ../../data/generated_20/cfd/mesh_SIZE_11.dat |
| SIZE_12 | `{"faces_per_cell":4,"mesh_cells":127764}` | new | 127764 | ../../data/generated_20/cfd/mesh_SIZE_12.dat |
| SIZE_13 | `{"faces_per_cell":4,"mesh_cells":146597}` | new | 146597 | ../../data/generated_20/cfd/mesh_SIZE_13.dat |
| SIZE_14 | `{"faces_per_cell":4,"mesh_cells":168206}` | new | 168206 | ../../data/generated_20/cfd/mesh_SIZE_14.dat |
| SIZE_15 | `{"faces_per_cell":4,"mesh_cells":193000}` | new; parameter anchor=fvcorr.domn.193K | 193000 | ../../data/generated_20/cfd/mesh_SIZE_15.dat |
| SIZE_16 | `{"faces_per_cell":4,"mesh_cells":205189}` | new | 205189 | ../../data/generated_20/cfd/mesh_SIZE_16.dat |
| SIZE_17 | `{"faces_per_cell":4,"mesh_cells":218148}` | new | 218148 | ../../data/generated_20/cfd/mesh_SIZE_17.dat |
| SIZE_18 | `{"faces_per_cell":4,"mesh_cells":231925}` | new | 231925 | ../../data/generated_20/cfd/mesh_SIZE_18.dat |
| SIZE_19 | `{"faces_per_cell":4,"mesh_cells":246572}` | new | 246572 | ../../data/generated_20/cfd/mesh_SIZE_19.dat |
| SIZE_20 | `{"faces_per_cell":4,"mesh_cells":262144}` | new | 262144 | ../../data/generated_20/cfd/mesh_SIZE_20.dat |

### Rodinia CUDA / dwt2d

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"components":3,"height":192,"levels":3,"wavelet":"5/3","width":192}` | new | 110592 | ../../data/generated_20/dwt2d/image_SIZE_01.raw |
| SIZE_02 | `{"components":3,"height":224,"levels":3,"wavelet":"5/3","width":224}` | new | 150528 | ../../data/generated_20/dwt2d/image_SIZE_02.raw |
| SIZE_03 | `{"components":3,"height":240,"levels":3,"wavelet":"5/3","width":240}` | new | 172800 | ../../data/generated_20/dwt2d/image_SIZE_03.raw |
| SIZE_04 | `{"components":3,"height":272,"levels":3,"wavelet":"5/3","width":272}` | new | 221952 | ../../data/generated_20/dwt2d/image_SIZE_04.raw |
| SIZE_05 | `{"components":3,"height":304,"levels":3,"wavelet":"5/3","width":304}` | new | 277248 | ../../data/generated_20/dwt2d/image_SIZE_05.raw |
| SIZE_06 | `{"components":3,"height":352,"levels":3,"wavelet":"5/3","width":352}` | new | 371712 | ../../data/generated_20/dwt2d/image_SIZE_06.raw |
| SIZE_07 | `{"components":3,"height":400,"levels":3,"wavelet":"5/3","width":400}` | new | 480000 | ../../data/generated_20/dwt2d/image_SIZE_07.raw |
| SIZE_08 | `{"components":3,"height":448,"levels":3,"wavelet":"5/3","width":448}` | new | 602112 | ../../data/generated_20/dwt2d/image_SIZE_08.raw |
| SIZE_09 | `{"components":3,"height":496,"levels":3,"wavelet":"5/3","width":496}` | new | 738048 | ../../data/generated_20/dwt2d/image_SIZE_09.raw |
| SIZE_10 | `{"components":3,"height":560,"levels":3,"wavelet":"5/3","width":560}` | new | 940800 | ../../data/generated_20/dwt2d/image_SIZE_10.raw |
| SIZE_11 | `{"components":3,"height":640,"levels":3,"wavelet":"5/3","width":640}` | new | 1228800 | ../../data/generated_20/dwt2d/image_SIZE_11.raw |
| SIZE_12 | `{"components":3,"height":720,"levels":3,"wavelet":"5/3","width":720}` | new | 1555200 | ../../data/generated_20/dwt2d/image_SIZE_12.raw |
| SIZE_13 | `{"components":3,"height":800,"levels":3,"wavelet":"5/3","width":800}` | new | 1920000 | ../../data/generated_20/dwt2d/image_SIZE_13.raw |
| SIZE_14 | `{"components":3,"height":912,"levels":3,"wavelet":"5/3","width":912}` | new | 2495232 | ../../data/generated_20/dwt2d/image_SIZE_14.raw |
| SIZE_15 | `{"components":3,"height":1024,"levels":3,"wavelet":"5/3","width":1024}` | new | 3145728 | ../../data/generated_20/dwt2d/image_SIZE_15.raw |
| SIZE_16 | `{"components":3,"height":1104,"levels":3,"wavelet":"5/3","width":1104}` | new | 3656448 | ../../data/generated_20/dwt2d/image_SIZE_16.raw |
| SIZE_17 | `{"components":3,"height":1200,"levels":3,"wavelet":"5/3","width":1200}` | new | 4320000 | ../../data/generated_20/dwt2d/image_SIZE_17.raw |
| SIZE_18 | `{"components":3,"height":1312,"levels":3,"wavelet":"5/3","width":1312}` | new | 5164032 | ../../data/generated_20/dwt2d/image_SIZE_18.raw |
| SIZE_19 | `{"components":3,"height":1408,"levels":3,"wavelet":"5/3","width":1408}` | new | 5947392 | ../../data/generated_20/dwt2d/image_SIZE_19.raw |
| SIZE_20 | `{"components":3,"height":1536,"levels":3,"wavelet":"5/3","width":1536}` | new | 7077888 | ../../data/generated_20/dwt2d/image_SIZE_20.raw |

### Rodinia CUDA / gaussian

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"matrix_size":16}` | new | 4096 | - |
| SIZE_02 | `{"matrix_size":32}` | new | 32768 | - |
| SIZE_03 | `{"matrix_size":48}` | new | 110592 | - |
| SIZE_04 | `{"matrix_size":64}` | new | 262144 | - |
| SIZE_05 | `{"matrix_size":80}` | new | 512000 | - |
| SIZE_06 | `{"matrix_size":96}` | new | 884736 | - |
| SIZE_07 | `{"matrix_size":112}` | new | 1404928 | - |
| SIZE_08 | `{"matrix_size":128}` | new | 2097152 | - |
| SIZE_09 | `{"matrix_size":176}` | new | 5451776 | - |
| SIZE_10 | `{"matrix_size":224}` | new | 11239424 | - |
| SIZE_11 | `{"matrix_size":320}` | new | 32768000 | - |
| SIZE_12 | `{"matrix_size":416}` | new | 71991296 | - |
| SIZE_13 | `{"matrix_size":560}` | new | 175616000 | - |
| SIZE_14 | `{"matrix_size":768}` | new | 452984832 | - |
| SIZE_15 | `{"matrix_size":1024}` | new | 1073741824 | - |
| SIZE_16 | `{"matrix_size":1104}` | new | 1345572864 | - |
| SIZE_17 | `{"matrix_size":1200}` | new | 1728000000 | - |
| SIZE_18 | `{"matrix_size":1312}` | new | 2258403328 | - |
| SIZE_19 | `{"matrix_size":1408}` | new | 2791309312 | - |
| SIZE_20 | `{"matrix_size":1536}` | new | 3623878656 | - |

### Rodinia CUDA / heartwall

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":2,"input_file":"test.avi"}` | test-2-frames | 2 | ../../data/heartwall/test.avi |
| SIZE_02 | `{"frames":3,"input_file":"test.avi"}` | new | 3 | ../../data/heartwall/test.avi |
| SIZE_03 | `{"frames":4,"input_file":"test.avi"}` | new | 4 | ../../data/heartwall/test.avi |
| SIZE_04 | `{"frames":5,"input_file":"test.avi"}` | new | 5 | ../../data/heartwall/test.avi |
| SIZE_05 | `{"frames":6,"input_file":"test.avi"}` | new | 6 | ../../data/heartwall/test.avi |
| SIZE_06 | `{"frames":7,"input_file":"test.avi"}` | new | 7 | ../../data/heartwall/test.avi |
| SIZE_07 | `{"frames":9,"input_file":"test.avi"}` | new | 9 | ../../data/heartwall/test.avi |
| SIZE_08 | `{"frames":12,"input_file":"test.avi"}` | new | 12 | ../../data/heartwall/test.avi |
| SIZE_09 | `{"frames":15,"input_file":"test.avi"}` | new | 15 | ../../data/heartwall/test.avi |
| SIZE_10 | `{"frames":20,"input_file":"test.avi"}` | test-20-frames | 20 | ../../data/heartwall/test.avi |
| SIZE_11 | `{"frames":23,"input_file":"test.avi"}` | new | 23 | ../../data/heartwall/test.avi |
| SIZE_12 | `{"frames":28,"input_file":"test.avi"}` | new | 28 | ../../data/heartwall/test.avi |
| SIZE_13 | `{"frames":32,"input_file":"test.avi"}` | new | 32 | ../../data/heartwall/test.avi |
| SIZE_14 | `{"frames":38,"input_file":"test.avi"}` | new | 38 | ../../data/heartwall/test.avi |
| SIZE_15 | `{"frames":45,"input_file":"test.avi"}` | new | 45 | ../../data/heartwall/test.avi |
| SIZE_16 | `{"frames":53,"input_file":"test.avi"}` | new | 53 | ../../data/heartwall/test.avi |
| SIZE_17 | `{"frames":62,"input_file":"test.avi"}` | new | 62 | ../../data/heartwall/test.avi |
| SIZE_18 | `{"frames":72,"input_file":"test.avi"}` | new | 72 | ../../data/heartwall/test.avi |
| SIZE_19 | `{"frames":85,"input_file":"test.avi"}` | new | 85 | ../../data/heartwall/test.avi |
| SIZE_20 | `{"frames":100,"input_file":"test.avi"}` | new | 100 | ../../data/heartwall/test.avi |

### Rodinia CUDA / hotspot

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"grid_size":64,"iterations":2,"pyramid_height":2}` | new; parameter anchor=grid-64 | 8192 | ../../data/generated_20/hotspot/temp_SIZE_01.txt<br>../../data/generated_20/hotspot/power_SIZE_01.txt |
| SIZE_02 | `{"grid_size":80,"iterations":2,"pyramid_height":2}` | new | 12800 | ../../data/generated_20/hotspot/temp_SIZE_02.txt<br>../../data/generated_20/hotspot/power_SIZE_02.txt |
| SIZE_03 | `{"grid_size":96,"iterations":2,"pyramid_height":2}` | new | 18432 | ../../data/generated_20/hotspot/temp_SIZE_03.txt<br>../../data/generated_20/hotspot/power_SIZE_03.txt |
| SIZE_04 | `{"grid_size":112,"iterations":2,"pyramid_height":2}` | new | 25088 | ../../data/generated_20/hotspot/temp_SIZE_04.txt<br>../../data/generated_20/hotspot/power_SIZE_04.txt |
| SIZE_05 | `{"grid_size":144,"iterations":2,"pyramid_height":2}` | new | 41472 | ../../data/generated_20/hotspot/temp_SIZE_05.txt<br>../../data/generated_20/hotspot/power_SIZE_05.txt |
| SIZE_06 | `{"grid_size":176,"iterations":2,"pyramid_height":2}` | new | 61952 | ../../data/generated_20/hotspot/temp_SIZE_06.txt<br>../../data/generated_20/hotspot/power_SIZE_06.txt |
| SIZE_07 | `{"grid_size":208,"iterations":2,"pyramid_height":2}` | new | 86528 | ../../data/generated_20/hotspot/temp_SIZE_07.txt<br>../../data/generated_20/hotspot/power_SIZE_07.txt |
| SIZE_08 | `{"grid_size":256,"iterations":2,"pyramid_height":2}` | new | 131072 | ../../data/generated_20/hotspot/temp_SIZE_08.txt<br>../../data/generated_20/hotspot/power_SIZE_08.txt |
| SIZE_09 | `{"grid_size":320,"iterations":2,"pyramid_height":2}` | new | 204800 | ../../data/generated_20/hotspot/temp_SIZE_09.txt<br>../../data/generated_20/hotspot/power_SIZE_09.txt |
| SIZE_10 | `{"grid_size":384,"iterations":2,"pyramid_height":2}` | new | 294912 | ../../data/generated_20/hotspot/temp_SIZE_10.txt<br>../../data/generated_20/hotspot/power_SIZE_10.txt |
| SIZE_11 | `{"grid_size":464,"iterations":2,"pyramid_height":2}` | new | 430592 | ../../data/generated_20/hotspot/temp_SIZE_11.txt<br>../../data/generated_20/hotspot/power_SIZE_11.txt |
| SIZE_12 | `{"grid_size":560,"iterations":2,"pyramid_height":2}` | new | 627200 | ../../data/generated_20/hotspot/temp_SIZE_12.txt<br>../../data/generated_20/hotspot/power_SIZE_12.txt |
| SIZE_13 | `{"grid_size":688,"iterations":2,"pyramid_height":2}` | new | 946688 | ../../data/generated_20/hotspot/temp_SIZE_13.txt<br>../../data/generated_20/hotspot/power_SIZE_13.txt |
| SIZE_14 | `{"grid_size":832,"iterations":2,"pyramid_height":2}` | new | 1384448 | ../../data/generated_20/hotspot/temp_SIZE_14.txt<br>../../data/generated_20/hotspot/power_SIZE_14.txt |
| SIZE_15 | `{"grid_size":1024,"iterations":2,"pyramid_height":2}` | new; parameter anchor=grid-512/1024-range | 2097152 | ../../data/generated_20/hotspot/temp_SIZE_15.txt<br>../../data/generated_20/hotspot/power_SIZE_15.txt |
| SIZE_16 | `{"grid_size":1104,"iterations":2,"pyramid_height":2}` | new | 2437632 | ../../data/generated_20/hotspot/temp_SIZE_16.txt<br>../../data/generated_20/hotspot/power_SIZE_16.txt |
| SIZE_17 | `{"grid_size":1200,"iterations":2,"pyramid_height":2}` | new | 2880000 | ../../data/generated_20/hotspot/temp_SIZE_17.txt<br>../../data/generated_20/hotspot/power_SIZE_17.txt |
| SIZE_18 | `{"grid_size":1312,"iterations":2,"pyramid_height":2}` | new | 3442688 | ../../data/generated_20/hotspot/temp_SIZE_18.txt<br>../../data/generated_20/hotspot/power_SIZE_18.txt |
| SIZE_19 | `{"grid_size":1408,"iterations":2,"pyramid_height":2}` | new | 3964928 | ../../data/generated_20/hotspot/temp_SIZE_19.txt<br>../../data/generated_20/hotspot/power_SIZE_19.txt |
| SIZE_20 | `{"grid_size":1536,"iterations":2,"pyramid_height":2}` | new | 4718592 | ../../data/generated_20/hotspot/temp_SIZE_20.txt<br>../../data/generated_20/hotspot/power_SIZE_20.txt |

### Rodinia CUDA / hotspot3D

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"grid_size":64,"iterations":10,"layers":8}` | new; parameter anchor=grid-64x8 | 327680 | ../../data/generated_20/hotspot3D/power_SIZE_01.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_01.txt |
| SIZE_02 | `{"grid_size":80,"iterations":12,"layers":8}` | new | 614400 | ../../data/generated_20/hotspot3D/power_SIZE_02.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_02.txt |
| SIZE_03 | `{"grid_size":96,"iterations":14,"layers":8}` | new | 1032192 | ../../data/generated_20/hotspot3D/power_SIZE_03.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_03.txt |
| SIZE_04 | `{"grid_size":112,"iterations":16,"layers":8}` | new | 1605632 | ../../data/generated_20/hotspot3D/power_SIZE_04.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_04.txt |
| SIZE_05 | `{"grid_size":128,"iterations":19,"layers":8}` | new | 2490368 | ../../data/generated_20/hotspot3D/power_SIZE_05.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_05.txt |
| SIZE_06 | `{"grid_size":144,"iterations":23,"layers":8}` | new | 3815424 | ../../data/generated_20/hotspot3D/power_SIZE_06.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_06.txt |
| SIZE_07 | `{"grid_size":160,"iterations":27,"layers":8}` | new | 5529600 | ../../data/generated_20/hotspot3D/power_SIZE_07.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_07.txt |
| SIZE_08 | `{"grid_size":176,"iterations":32,"layers":8}` | new | 7929856 | ../../data/generated_20/hotspot3D/power_SIZE_08.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_08.txt |
| SIZE_09 | `{"grid_size":208,"iterations":37,"layers":8}` | new | 12806144 | ../../data/generated_20/hotspot3D/power_SIZE_09.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_09.txt |
| SIZE_10 | `{"grid_size":240,"iterations":44,"layers":8}` | new | 20275200 | ../../data/generated_20/hotspot3D/power_SIZE_10.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_10.txt |
| SIZE_11 | `{"grid_size":288,"iterations":52,"layers":8}` | new | 34504704 | ../../data/generated_20/hotspot3D/power_SIZE_11.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_11.txt |
| SIZE_12 | `{"grid_size":320,"iterations":61,"layers":8}` | new | 49971200 | ../../data/generated_20/hotspot3D/power_SIZE_12.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_12.txt |
| SIZE_13 | `{"grid_size":384,"iterations":72,"layers":8}` | new | 84934656 | ../../data/generated_20/hotspot3D/power_SIZE_13.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_13.txt |
| SIZE_14 | `{"grid_size":448,"iterations":85,"layers":8}` | new | 136478720 | ../../data/generated_20/hotspot3D/power_SIZE_14.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_14.txt |
| SIZE_15 | `{"grid_size":512,"iterations":100,"layers":8}` | new; parameter anchor=grid-512x8 | 209715200 | ../../data/generated_20/hotspot3D/power_SIZE_15.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_15.txt |
| SIZE_16 | `{"grid_size":528,"iterations":107,"layers":8}` | new | 238639104 | ../../data/generated_20/hotspot3D/power_SIZE_16.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_16.txt |
| SIZE_17 | `{"grid_size":560,"iterations":114,"layers":8}` | new | 286003200 | ../../data/generated_20/hotspot3D/power_SIZE_17.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_17.txt |
| SIZE_18 | `{"grid_size":592,"iterations":122,"layers":8}` | new | 342052864 | ../../data/generated_20/hotspot3D/power_SIZE_18.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_18.txt |
| SIZE_19 | `{"grid_size":608,"iterations":131,"layers":8}` | new | 387407872 | ../../data/generated_20/hotspot3D/power_SIZE_19.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_19.txt |
| SIZE_20 | `{"grid_size":640,"iterations":140,"layers":8}` | new | 458752000 | ../../data/generated_20/hotspot3D/power_SIZE_20.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_20.txt |

### Rodinia CUDA / huffman

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"block_threads":256,"input_bytes":65536}` | new | 65536 | ../../data/generated_20/huffman/input_SIZE_01.bin |
| SIZE_02 | `{"block_threads":256,"input_bytes":79872}` | new | 79872 | ../../data/generated_20/huffman/input_SIZE_02.bin |
| SIZE_03 | `{"block_threads":256,"input_bytes":97280}` | new | 97280 | ../../data/generated_20/huffman/input_SIZE_03.bin |
| SIZE_04 | `{"block_threads":256,"input_bytes":118784}` | new | 118784 | ../../data/generated_20/huffman/input_SIZE_04.bin |
| SIZE_05 | `{"block_threads":256,"input_bytes":144384}` | new | 144384 | ../../data/generated_20/huffman/input_SIZE_05.bin |
| SIZE_06 | `{"block_threads":256,"input_bytes":176128}` | new | 176128 | ../../data/generated_20/huffman/input_SIZE_06.bin |
| SIZE_07 | `{"block_threads":256,"input_bytes":215040}` | new | 215040 | ../../data/generated_20/huffman/input_SIZE_07.bin |
| SIZE_08 | `{"block_threads":256,"input_bytes":262144}` | new | 262144 | ../../data/generated_20/huffman/input_SIZE_08.bin |
| SIZE_09 | `{"block_threads":256,"input_bytes":319488}` | new | 319488 | ../../data/generated_20/huffman/input_SIZE_09.bin |
| SIZE_10 | `{"block_threads":256,"input_bytes":389120}` | new | 389120 | ../../data/generated_20/huffman/input_SIZE_10.bin |
| SIZE_11 | `{"block_threads":256,"input_bytes":475136}` | new | 475136 | ../../data/generated_20/huffman/input_SIZE_11.bin |
| SIZE_12 | `{"block_threads":256,"input_bytes":578560}` | new | 578560 | ../../data/generated_20/huffman/input_SIZE_12.bin |
| SIZE_13 | `{"block_threads":256,"input_bytes":705536}` | new | 705536 | ../../data/generated_20/huffman/input_SIZE_13.bin |
| SIZE_14 | `{"block_threads":256,"input_bytes":860160}` | new | 860160 | ../../data/generated_20/huffman/input_SIZE_14.bin |
| SIZE_15 | `{"block_threads":256,"input_bytes":1048576}` | new; parameter anchor=test1024_H2.206587175259.in | 1048576 | ../../data/generated_20/huffman/input_SIZE_15.bin |
| SIZE_16 | `{"block_threads":256,"input_bytes":1589248}` | new | 1589248 | ../../data/generated_20/huffman/input_SIZE_16.bin |
| SIZE_17 | `{"block_threads":256,"input_bytes":2409472}` | new | 2409472 | ../../data/generated_20/huffman/input_SIZE_17.bin |
| SIZE_18 | `{"block_threads":256,"input_bytes":3651584}` | new | 3651584 | ../../data/generated_20/huffman/input_SIZE_18.bin |
| SIZE_19 | `{"block_threads":256,"input_bytes":5534720}` | new | 5534720 | ../../data/generated_20/huffman/input_SIZE_19.bin |
| SIZE_20 | `{"block_threads":256,"input_bytes":8388608}` | new | 8388608 | ../../data/generated_20/huffman/input_SIZE_20.bin |

### Rodinia CUDA / lavaMD

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"boxes1d":2,"boxes_total":8,"particles_per_box":100}` | boxes-2 | 800 | - |
| SIZE_02 | `{"boxes1d":3,"boxes_total":27,"particles_per_box":100}` | new | 2700 | - |
| SIZE_03 | `{"boxes1d":4,"boxes_total":64,"particles_per_box":100}` | new | 6400 | - |
| SIZE_04 | `{"boxes1d":5,"boxes_total":125,"particles_per_box":100}` | new | 12500 | - |
| SIZE_05 | `{"boxes1d":6,"boxes_total":216,"particles_per_box":100}` | new | 21600 | - |
| SIZE_06 | `{"boxes1d":7,"boxes_total":343,"particles_per_box":100}` | new | 34300 | - |
| SIZE_07 | `{"boxes1d":8,"boxes_total":512,"particles_per_box":100}` | new | 51200 | - |
| SIZE_08 | `{"boxes1d":9,"boxes_total":729,"particles_per_box":100}` | new | 72900 | - |
| SIZE_09 | `{"boxes1d":10,"boxes_total":1000,"particles_per_box":100}` | boxes-10 | 100000 | - |
| SIZE_10 | `{"boxes1d":11,"boxes_total":1331,"particles_per_box":100}` | new | 133100 | - |
| SIZE_11 | `{"boxes1d":12,"boxes_total":1728,"particles_per_box":100}` | new | 172800 | - |
| SIZE_12 | `{"boxes1d":13,"boxes_total":2197,"particles_per_box":100}` | new | 219700 | - |
| SIZE_13 | `{"boxes1d":14,"boxes_total":2744,"particles_per_box":100}` | new | 274400 | - |
| SIZE_14 | `{"boxes1d":15,"boxes_total":3375,"particles_per_box":100}` | new | 337500 | - |
| SIZE_15 | `{"boxes1d":16,"boxes_total":4096,"particles_per_box":100}` | new | 409600 | - |
| SIZE_16 | `{"boxes1d":17,"boxes_total":4913,"particles_per_box":100}` | new | 491300 | - |
| SIZE_17 | `{"boxes1d":18,"boxes_total":5832,"particles_per_box":100}` | new | 583200 | - |
| SIZE_18 | `{"boxes1d":19,"boxes_total":6859,"particles_per_box":100}` | new | 685900 | - |
| SIZE_19 | `{"boxes1d":20,"boxes_total":8000,"particles_per_box":100}` | new | 800000 | - |
| SIZE_20 | `{"boxes1d":21,"boxes_total":9261,"particles_per_box":100}` | new | 926100 | - |

### Rodinia CUDA / leukocyte

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":1,"input_file":"testfile.avi"}` | new | 1 | ../../../data/leukocyte/testfile.avi |
| SIZE_02 | `{"frames":2,"input_file":"testfile.avi"}` | new | 2 | ../../../data/leukocyte/testfile.avi |
| SIZE_03 | `{"frames":3,"input_file":"testfile.avi"}` | new | 3 | ../../../data/leukocyte/testfile.avi |
| SIZE_04 | `{"frames":4,"input_file":"testfile.avi"}` | new | 4 | ../../../data/leukocyte/testfile.avi |
| SIZE_05 | `{"frames":5,"input_file":"testfile.avi"}` | test-5-frames | 5 | ../../../data/leukocyte/testfile.avi |
| SIZE_06 | `{"frames":6,"input_file":"testfile.avi"}` | new | 6 | ../../../data/leukocyte/testfile.avi |
| SIZE_07 | `{"frames":7,"input_file":"testfile.avi"}` | new | 7 | ../../../data/leukocyte/testfile.avi |
| SIZE_08 | `{"frames":8,"input_file":"testfile.avi"}` | new | 8 | ../../../data/leukocyte/testfile.avi |
| SIZE_09 | `{"frames":9,"input_file":"testfile.avi"}` | new | 9 | ../../../data/leukocyte/testfile.avi |
| SIZE_10 | `{"frames":10,"input_file":"testfile.avi"}` | new | 10 | ../../../data/leukocyte/testfile.avi |
| SIZE_11 | `{"frames":11,"input_file":"testfile.avi"}` | new | 11 | ../../../data/leukocyte/testfile.avi |
| SIZE_12 | `{"frames":12,"input_file":"testfile.avi"}` | new | 12 | ../../../data/leukocyte/testfile.avi |
| SIZE_13 | `{"frames":13,"input_file":"testfile.avi"}` | new | 13 | ../../../data/leukocyte/testfile.avi |
| SIZE_14 | `{"frames":14,"input_file":"testfile.avi"}` | new | 14 | ../../../data/leukocyte/testfile.avi |
| SIZE_15 | `{"frames":15,"input_file":"testfile.avi"}` | new | 15 | ../../../data/leukocyte/testfile.avi |
| SIZE_16 | `{"frames":16,"input_file":"testfile.avi"}` | new | 16 | ../../../data/leukocyte/testfile.avi |
| SIZE_17 | `{"frames":17,"input_file":"testfile.avi"}` | new | 17 | ../../../data/leukocyte/testfile.avi |
| SIZE_18 | `{"frames":18,"input_file":"testfile.avi"}` | new | 18 | ../../../data/leukocyte/testfile.avi |
| SIZE_19 | `{"frames":19,"input_file":"testfile.avi"}` | new | 19 | ../../../data/leukocyte/testfile.avi |
| SIZE_20 | `{"frames":20,"input_file":"testfile.avi"}` | new | 20 | ../../../data/leukocyte/testfile.avi |

### Rodinia CUDA / lud

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"matrix_dimension":64}` | new | 262144 | - |
| SIZE_02 | `{"matrix_dimension":80}` | new | 512000 | - |
| SIZE_03 | `{"matrix_dimension":112}` | new | 1404928 | - |
| SIZE_04 | `{"matrix_dimension":128}` | new | 2097152 | - |
| SIZE_05 | `{"matrix_dimension":176}` | new | 5451776 | - |
| SIZE_06 | `{"matrix_dimension":224}` | new | 11239424 | - |
| SIZE_07 | `{"matrix_dimension":288}` | new | 23887872 | - |
| SIZE_08 | `{"matrix_dimension":368}` | new | 49836032 | - |
| SIZE_09 | `{"matrix_dimension":464}` | new | 99897344 | - |
| SIZE_10 | `{"matrix_dimension":592}` | new | 207474688 | - |
| SIZE_11 | `{"matrix_dimension":768}` | new | 452984832 | - |
| SIZE_12 | `{"matrix_dimension":976}` | new | 929714176 | - |
| SIZE_13 | `{"matrix_dimension":1248}` | new | 1943764992 | - |
| SIZE_14 | `{"matrix_dimension":1600}` | new | 4096000000 | - |
| SIZE_15 | `{"matrix_dimension":2048}` | new | 8589934592 | - |
| SIZE_16 | `{"matrix_dimension":2352}` | new | 13011038208 | - |
| SIZE_17 | `{"matrix_dimension":2704}` | new | 19770609664 | - |
| SIZE_18 | `{"matrix_dimension":3104}` | new | 29906468864 | - |
| SIZE_19 | `{"matrix_dimension":3568}` | new | 45422866432 | - |
| SIZE_20 | `{"matrix_dimension":4096}` | new | 68719476736 | - |

### Rodinia CUDA / myocyte

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"instances":1,"mode":0,"simulation_time_ms":10}` | time-10-one-instance | 10 | - |
| SIZE_02 | `{"instances":1,"mode":0,"simulation_time_ms":12}` | new | 12 | - |
| SIZE_03 | `{"instances":1,"mode":0,"simulation_time_ms":14}` | new | 14 | - |
| SIZE_04 | `{"instances":1,"mode":0,"simulation_time_ms":16}` | new | 16 | - |
| SIZE_05 | `{"instances":1,"mode":0,"simulation_time_ms":19}` | new | 19 | - |
| SIZE_06 | `{"instances":1,"mode":0,"simulation_time_ms":23}` | new | 23 | - |
| SIZE_07 | `{"instances":1,"mode":0,"simulation_time_ms":27}` | new | 27 | - |
| SIZE_08 | `{"instances":1,"mode":0,"simulation_time_ms":32}` | new | 32 | - |
| SIZE_09 | `{"instances":1,"mode":0,"simulation_time_ms":37}` | new | 37 | - |
| SIZE_10 | `{"instances":1,"mode":0,"simulation_time_ms":44}` | new | 44 | - |
| SIZE_11 | `{"instances":1,"mode":0,"simulation_time_ms":52}` | new | 52 | - |
| SIZE_12 | `{"instances":1,"mode":0,"simulation_time_ms":61}` | new | 61 | - |
| SIZE_13 | `{"instances":1,"mode":0,"simulation_time_ms":72}` | new | 72 | - |
| SIZE_14 | `{"instances":1,"mode":0,"simulation_time_ms":85}` | new | 85 | - |
| SIZE_15 | `{"instances":1,"mode":0,"simulation_time_ms":100}` | time-100-one-instance | 100 | - |
| SIZE_16 | `{"instances":1,"mode":0,"simulation_time_ms":115}` | new | 115 | - |
| SIZE_17 | `{"instances":1,"mode":0,"simulation_time_ms":132}` | new | 132 | - |
| SIZE_18 | `{"instances":1,"mode":0,"simulation_time_ms":152}` | new | 152 | - |
| SIZE_19 | `{"instances":1,"mode":0,"simulation_time_ms":174}` | new | 174 | - |
| SIZE_20 | `{"instances":1,"mode":0,"simulation_time_ms":200}` | new | 200 | - |

### Rodinia CUDA / nn

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"database_records":1024,"latitude":30,"longitude":90,"result_count":5}` | new | 1024 | ../../data/generated_20/nn/filelist_SIZE_01.txt<br>../../data/generated_20/nn/records_SIZE_01.db |
| SIZE_02 | `{"database_records":1550,"latitude":30,"longitude":90,"result_count":5}` | new | 1550 | ../../data/generated_20/nn/filelist_SIZE_02.txt<br>../../data/generated_20/nn/records_SIZE_02.db |
| SIZE_03 | `{"database_records":2347,"latitude":30,"longitude":90,"result_count":5}` | new | 2347 | ../../data/generated_20/nn/filelist_SIZE_03.txt<br>../../data/generated_20/nn/records_SIZE_03.db |
| SIZE_04 | `{"database_records":3553,"latitude":30,"longitude":90,"result_count":5}` | new | 3553 | ../../data/generated_20/nn/filelist_SIZE_04.txt<br>../../data/generated_20/nn/records_SIZE_04.db |
| SIZE_05 | `{"database_records":5378,"latitude":30,"longitude":90,"result_count":5}` | new | 5378 | ../../data/generated_20/nn/filelist_SIZE_05.txt<br>../../data/generated_20/nn/records_SIZE_05.db |
| SIZE_06 | `{"database_records":8142,"latitude":30,"longitude":90,"result_count":5}` | new | 8142 | ../../data/generated_20/nn/filelist_SIZE_06.txt<br>../../data/generated_20/nn/records_SIZE_06.db |
| SIZE_07 | `{"database_records":12326,"latitude":30,"longitude":90,"result_count":5}` | new | 12326 | ../../data/generated_20/nn/filelist_SIZE_07.txt<br>../../data/generated_20/nn/records_SIZE_07.db |
| SIZE_08 | `{"database_records":18660,"latitude":30,"longitude":90,"result_count":5}` | new | 18660 | ../../data/generated_20/nn/filelist_SIZE_08.txt<br>../../data/generated_20/nn/records_SIZE_08.db |
| SIZE_09 | `{"database_records":28248,"latitude":30,"longitude":90,"result_count":5}` | new | 28248 | ../../data/generated_20/nn/filelist_SIZE_09.txt<br>../../data/generated_20/nn/records_SIZE_09.db |
| SIZE_10 | `{"database_records":42764,"latitude":30,"longitude":90,"result_count":5}` | new; parameter anchor=cane4 databases | 42764 | ../../data/generated_20/nn/filelist_SIZE_10.txt<br>../../data/generated_20/nn/records_SIZE_10.db |
| SIZE_11 | `{"database_records":49897,"latitude":30,"longitude":90,"result_count":5}` | new | 49897 | ../../data/generated_20/nn/filelist_SIZE_11.txt<br>../../data/generated_20/nn/records_SIZE_11.db |
| SIZE_12 | `{"database_records":58220,"latitude":30,"longitude":90,"result_count":5}` | new | 58220 | ../../data/generated_20/nn/filelist_SIZE_12.txt<br>../../data/generated_20/nn/records_SIZE_12.db |
| SIZE_13 | `{"database_records":67930,"latitude":30,"longitude":90,"result_count":5}` | new | 67930 | ../../data/generated_20/nn/filelist_SIZE_13.txt<br>../../data/generated_20/nn/records_SIZE_13.db |
| SIZE_14 | `{"database_records":79261,"latitude":30,"longitude":90,"result_count":5}` | new | 79261 | ../../data/generated_20/nn/filelist_SIZE_14.txt<br>../../data/generated_20/nn/records_SIZE_14.db |
| SIZE_15 | `{"database_records":92481,"latitude":30,"longitude":90,"result_count":5}` | new | 92481 | ../../data/generated_20/nn/filelist_SIZE_15.txt<br>../../data/generated_20/nn/records_SIZE_15.db |
| SIZE_16 | `{"database_records":107907,"latitude":30,"longitude":90,"result_count":5}` | new | 107907 | ../../data/generated_20/nn/filelist_SIZE_16.txt<br>../../data/generated_20/nn/records_SIZE_16.db |
| SIZE_17 | `{"database_records":125905,"latitude":30,"longitude":90,"result_count":5}` | new | 125905 | ../../data/generated_20/nn/filelist_SIZE_17.txt<br>../../data/generated_20/nn/records_SIZE_17.db |
| SIZE_18 | `{"database_records":146906,"latitude":30,"longitude":90,"result_count":5}` | new | 146906 | ../../data/generated_20/nn/filelist_SIZE_18.txt<br>../../data/generated_20/nn/records_SIZE_18.db |
| SIZE_19 | `{"database_records":171409,"latitude":30,"longitude":90,"result_count":5}` | new | 171409 | ../../data/generated_20/nn/filelist_SIZE_19.txt<br>../../data/generated_20/nn/records_SIZE_19.db |
| SIZE_20 | `{"database_records":200000,"latitude":30,"longitude":90,"result_count":5}` | new | 200000 | ../../data/generated_20/nn/filelist_SIZE_20.txt<br>../../data/generated_20/nn/records_SIZE_20.db |

### Rodinia CUDA / nw

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"penalty":10,"sequence_dimension":256,"work_group_size":16}` | dimension-256 | 65536 | - |
| SIZE_02 | `{"penalty":10,"sequence_dimension":304,"work_group_size":16}` | new | 92416 | - |
| SIZE_03 | `{"penalty":10,"sequence_dimension":352,"work_group_size":16}` | new | 123904 | - |
| SIZE_04 | `{"penalty":10,"sequence_dimension":400,"work_group_size":16}` | new | 160000 | - |
| SIZE_05 | `{"penalty":10,"sequence_dimension":464,"work_group_size":16}` | new | 215296 | - |
| SIZE_06 | `{"penalty":10,"sequence_dimension":544,"work_group_size":16}` | new | 295936 | - |
| SIZE_07 | `{"penalty":10,"sequence_dimension":624,"work_group_size":16}` | new | 389376 | - |
| SIZE_08 | `{"penalty":10,"sequence_dimension":720,"work_group_size":16}` | new | 518400 | - |
| SIZE_09 | `{"penalty":10,"sequence_dimension":832,"work_group_size":16}` | new | 692224 | - |
| SIZE_10 | `{"penalty":10,"sequence_dimension":976,"work_group_size":16}` | new | 952576 | - |
| SIZE_11 | `{"penalty":10,"sequence_dimension":1136,"work_group_size":16}` | new | 1290496 | - |
| SIZE_12 | `{"penalty":10,"sequence_dimension":1312,"work_group_size":16}` | new | 1721344 | - |
| SIZE_13 | `{"penalty":10,"sequence_dimension":1520,"work_group_size":16}` | new | 2310400 | - |
| SIZE_14 | `{"penalty":10,"sequence_dimension":1760,"work_group_size":16}` | new | 3097600 | - |
| SIZE_15 | `{"penalty":10,"sequence_dimension":2048,"work_group_size":16}` | dimension-2048 | 4194304 | - |
| SIZE_16 | `{"penalty":10,"sequence_dimension":2352,"work_group_size":16}` | new | 5531904 | - |
| SIZE_17 | `{"penalty":10,"sequence_dimension":2704,"work_group_size":16}` | new | 7311616 | - |
| SIZE_18 | `{"penalty":10,"sequence_dimension":3104,"work_group_size":16}` | new | 9634816 | - |
| SIZE_19 | `{"penalty":10,"sequence_dimension":3568,"work_group_size":16}` | new | 12730624 | - |
| SIZE_20 | `{"penalty":10,"sequence_dimension":4096,"work_group_size":16}` | new | 16777216 | - |

### Rodinia CUDA / particlefilter_float

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":2,"height":32,"particles":100,"width":32}` | video-32x32x2-p100 | 204800 | - |
| SIZE_02 | `{"frames":3,"height":48,"particles":118,"width":48}` | new | 815616 | - |
| SIZE_03 | `{"frames":3,"height":64,"particles":139,"width":64}` | new | 1708032 | - |
| SIZE_04 | `{"frames":4,"height":80,"particles":164,"width":80}` | new | 4198400 | - |
| SIZE_05 | `{"frames":4,"height":96,"particles":193,"width":96}` | new | 7114752 | - |
| SIZE_06 | `{"frames":5,"height":112,"particles":228,"width":112}` | new | 14300160 | - |
| SIZE_07 | `{"frames":5,"height":128,"particles":268,"width":128}` | new | 21954560 | - |
| SIZE_08 | `{"frames":6,"height":144,"particles":316,"width":144}` | new | 39315456 | - |
| SIZE_09 | `{"frames":7,"height":160,"particles":373,"width":160}` | new | 66841600 | - |
| SIZE_10 | `{"frames":7,"height":176,"particles":439,"width":176}` | new | 95189248 | - |
| SIZE_11 | `{"frames":8,"height":192,"particles":518,"width":192}` | new | 152764416 | - |
| SIZE_12 | `{"frames":8,"height":208,"particles":611,"width":208}` | new | 211474432 | - |
| SIZE_13 | `{"frames":9,"height":224,"particles":720,"width":224}` | new | 325140480 | - |
| SIZE_14 | `{"frames":9,"height":240,"particles":848,"width":240}` | new | 439603200 | - |
| SIZE_15 | `{"frames":10,"height":256,"particles":1000,"width":256}` | new | 655360000 | - |
| SIZE_16 | `{"frames":12,"height":272,"particles":1585,"width":272}` | new | 1407175680 | - |
| SIZE_17 | `{"frames":14,"height":288,"particles":2512,"width":288}` | new | 2916974592 | - |
| SIZE_18 | `{"frames":16,"height":304,"particles":3981,"width":304}` | new | 5886529536 | - |
| SIZE_19 | `{"frames":18,"height":320,"particles":6310,"width":320}` | new | 11630592000 | - |
| SIZE_20 | `{"frames":20,"height":336,"particles":10000,"width":336}` | new | 22579200000 | - |

### Rodinia CUDA / particlefilter_naive

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":2,"height":32,"particles":100,"width":32}` | video-32x32x2-p100 | 204800 | - |
| SIZE_02 | `{"frames":3,"height":48,"particles":118,"width":48}` | new | 815616 | - |
| SIZE_03 | `{"frames":3,"height":64,"particles":139,"width":64}` | new | 1708032 | - |
| SIZE_04 | `{"frames":4,"height":80,"particles":164,"width":80}` | new | 4198400 | - |
| SIZE_05 | `{"frames":4,"height":96,"particles":193,"width":96}` | new | 7114752 | - |
| SIZE_06 | `{"frames":5,"height":112,"particles":228,"width":112}` | new | 14300160 | - |
| SIZE_07 | `{"frames":5,"height":128,"particles":268,"width":128}` | new | 21954560 | - |
| SIZE_08 | `{"frames":6,"height":144,"particles":316,"width":144}` | new | 39315456 | - |
| SIZE_09 | `{"frames":7,"height":160,"particles":373,"width":160}` | new | 66841600 | - |
| SIZE_10 | `{"frames":7,"height":176,"particles":439,"width":176}` | new | 95189248 | - |
| SIZE_11 | `{"frames":8,"height":192,"particles":518,"width":192}` | new | 152764416 | - |
| SIZE_12 | `{"frames":8,"height":208,"particles":611,"width":208}` | new | 211474432 | - |
| SIZE_13 | `{"frames":9,"height":224,"particles":720,"width":224}` | new | 325140480 | - |
| SIZE_14 | `{"frames":9,"height":240,"particles":848,"width":240}` | new | 439603200 | - |
| SIZE_15 | `{"frames":10,"height":256,"particles":1000,"width":256}` | new | 655360000 | - |
| SIZE_16 | `{"frames":12,"height":272,"particles":1585,"width":272}` | new | 1407175680 | - |
| SIZE_17 | `{"frames":14,"height":288,"particles":2512,"width":288}` | new | 2916974592 | - |
| SIZE_18 | `{"frames":16,"height":304,"particles":3981,"width":304}` | new | 5886529536 | - |
| SIZE_19 | `{"frames":18,"height":320,"particles":6310,"width":320}` | new | 11630592000 | - |
| SIZE_20 | `{"frames":20,"height":336,"particles":10000,"width":336}` | new | 22579200000 | - |

### Rodinia CUDA / pathfinder

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":1000,"pyramid_height":20,"rows":20}` | new | 20000 | - |
| SIZE_02 | `{"columns":1389,"pyramid_height":20,"rows":22}` | new | 30558 | - |
| SIZE_03 | `{"columns":1931,"pyramid_height":20,"rows":25}` | new | 48275 | - |
| SIZE_04 | `{"columns":2683,"pyramid_height":20,"rows":28}` | new | 75124 | - |
| SIZE_05 | `{"columns":3728,"pyramid_height":20,"rows":32}` | new | 119296 | - |
| SIZE_06 | `{"columns":5179,"pyramid_height":20,"rows":36}` | new | 186444 | - |
| SIZE_07 | `{"columns":7197,"pyramid_height":20,"rows":40}` | new | 287880 | - |
| SIZE_08 | `{"columns":10000,"pyramid_height":20,"rows":45}` | new | 450000 | - |
| SIZE_09 | `{"columns":13895,"pyramid_height":20,"rows":50}` | new | 694750 | - |
| SIZE_10 | `{"columns":19307,"pyramid_height":20,"rows":56}` | new | 1081192 | - |
| SIZE_11 | `{"columns":26827,"pyramid_height":20,"rows":63}` | new | 1690101 | - |
| SIZE_12 | `{"columns":37276,"pyramid_height":20,"rows":71}` | new | 2646596 | - |
| SIZE_13 | `{"columns":51795,"pyramid_height":20,"rows":79}` | new | 4091805 | - |
| SIZE_14 | `{"columns":71969,"pyramid_height":20,"rows":89}` | new | 6405241 | - |
| SIZE_15 | `{"columns":100000,"pyramid_height":20,"rows":100}` | grid-100000x100 | 10000000 | - |
| SIZE_16 | `{"columns":114870,"pyramid_height":20,"rows":115}` | new | 13210050 | - |
| SIZE_17 | `{"columns":131951,"pyramid_height":20,"rows":132}` | new | 17417532 | - |
| SIZE_18 | `{"columns":151572,"pyramid_height":20,"rows":152}` | new | 23038944 | - |
| SIZE_19 | `{"columns":174110,"pyramid_height":20,"rows":174}` | new | 30295140 | - |
| SIZE_20 | `{"columns":200000,"pyramid_height":20,"rows":200}` | new | 40000000 | - |

### Rodinia CUDA / srad_v1

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":128,"iterations":10,"lambda":0.5,"rows":128}` | new | 163840 | - |
| SIZE_02 | `{"columns":140,"iterations":12,"lambda":0.5,"rows":141}` | new | 236880 | - |
| SIZE_03 | `{"columns":154,"iterations":14,"lambda":0.5,"rows":156}` | new | 336336 | - |
| SIZE_04 | `{"columns":168,"iterations":16,"lambda":0.5,"rows":172}` | new | 462336 | - |
| SIZE_05 | `{"columns":184,"iterations":19,"lambda":0.5,"rows":189}` | new | 660744 | - |
| SIZE_06 | `{"columns":202,"iterations":23,"lambda":0.5,"rows":209}` | new | 971014 | - |
| SIZE_07 | `{"columns":221,"iterations":27,"lambda":0.5,"rows":230}` | new | 1372410 | - |
| SIZE_08 | `{"columns":242,"iterations":32,"lambda":0.5,"rows":253}` | new | 1959232 | - |
| SIZE_09 | `{"columns":265,"iterations":37,"lambda":0.5,"rows":279}` | new | 2735595 | - |
| SIZE_10 | `{"columns":290,"iterations":44,"lambda":0.5,"rows":308}` | new | 3930080 | - |
| SIZE_11 | `{"columns":318,"iterations":52,"lambda":0.5,"rows":340}` | new | 5622240 | - |
| SIZE_12 | `{"columns":349,"iterations":61,"lambda":0.5,"rows":375}` | new | 7983375 | - |
| SIZE_13 | `{"columns":382,"iterations":72,"lambda":0.5,"rows":413}` | new | 11359152 | - |
| SIZE_14 | `{"columns":418,"iterations":85,"lambda":0.5,"rows":455}` | new | 16166150 | - |
| SIZE_15 | `{"columns":458,"iterations":100,"lambda":0.5,"rows":502}` | image-502x458-100 | 22991600 | - |
| SIZE_16 | `{"columns":508,"iterations":108,"lambda":0.5,"rows":547}` | new | 30010608 | - |
| SIZE_17 | `{"columns":563,"iterations":118,"lambda":0.5,"rows":595}` | new | 39528230 | - |
| SIZE_18 | `{"columns":625,"iterations":128,"lambda":0.5,"rows":648}` | new | 51840000 | - |
| SIZE_19 | `{"columns":693,"iterations":138,"lambda":0.5,"rows":705}` | new | 67421970 | - |
| SIZE_20 | `{"columns":768,"iterations":150,"lambda":0.5,"rows":768}` | new | 88473600 | - |

### Rodinia CUDA / srad_v2

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":128,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":128}` | new | 32768 | - |
| SIZE_02 | `{"columns":160,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":160}` | new | 51200 | - |
| SIZE_03 | `{"columns":176,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":176}` | new | 61952 | - |
| SIZE_04 | `{"columns":208,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":208}` | new | 86528 | - |
| SIZE_05 | `{"columns":256,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":256}` | matrix-256 | 131072 | - |
| SIZE_06 | `{"columns":320,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":320}` | new | 204800 | - |
| SIZE_07 | `{"columns":384,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":384}` | new | 294912 | - |
| SIZE_08 | `{"columns":480,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":480}` | new | 460800 | - |
| SIZE_09 | `{"columns":592,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":592}` | new | 700928 | - |
| SIZE_10 | `{"columns":720,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":720}` | new | 1036800 | - |
| SIZE_11 | `{"columns":896,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":896}` | new | 1605632 | - |
| SIZE_12 | `{"columns":1104,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":1104}` | new | 2437632 | - |
| SIZE_13 | `{"columns":1344,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":1344}` | new | 3612672 | - |
| SIZE_14 | `{"columns":1664,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":1664}` | new | 5537792 | - |
| SIZE_15 | `{"columns":2048,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2048}` | matrix-2048 | 8388608 | - |
| SIZE_16 | `{"columns":2224,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2224}` | new | 9892352 | - |
| SIZE_17 | `{"columns":2416,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2416}` | new | 11674112 | - |
| SIZE_18 | `{"columns":2608,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2608}` | new | 13603328 | - |
| SIZE_19 | `{"columns":2832,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2832}` | new | 16040448 | - |
| SIZE_20 | `{"columns":3072,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":3072}` | new | 18874368 | - |

### Rodinia CUDA / streamcluster

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"center_size":100,"chunk_size":1000,"dimensions":16,"kmax":5,"kmin":3,"points":1000}` | new | 16000 | - |
| SIZE_02 | `{"center_size":200,"chunk_size":1348,"dimensions":20,"kmax":6,"kmin":3,"points":1348}` | new | 26960 | - |
| SIZE_03 | `{"center_size":300,"chunk_size":1818,"dimensions":24,"kmax":7,"kmin":4,"points":1818}` | new | 43632 | - |
| SIZE_04 | `{"center_size":400,"chunk_size":2450,"dimensions":28,"kmax":8,"kmin":4,"points":2450}` | new | 68600 | - |
| SIZE_05 | `{"center_size":500,"chunk_size":3304,"dimensions":36,"kmax":9,"kmin":5,"points":3304}` | new | 118944 | - |
| SIZE_06 | `{"center_size":600,"chunk_size":4454,"dimensions":44,"kmax":10,"kmin":5,"points":4454}` | new | 195976 | - |
| SIZE_07 | `{"center_size":700,"chunk_size":6005,"dimensions":52,"kmax":11,"kmin":6,"points":6005}` | new | 312260 | - |
| SIZE_08 | `{"center_size":800,"chunk_size":8095,"dimensions":64,"kmax":12,"kmin":6,"points":8095}` | new | 518080 | - |
| SIZE_09 | `{"center_size":900,"chunk_size":10914,"dimensions":80,"kmax":13,"kmin":7,"points":10914}` | new | 873120 | - |
| SIZE_10 | `{"center_size":1000,"chunk_size":14714,"dimensions":96,"kmax":14,"kmin":7,"points":14714}` | new | 1412544 | - |
| SIZE_11 | `{"center_size":1100,"chunk_size":19837,"dimensions":116,"kmax":15,"kmin":8,"points":19837}` | new | 2301092 | - |
| SIZE_12 | `{"center_size":1200,"chunk_size":26744,"dimensions":140,"kmax":16,"kmin":8,"points":26744}` | new | 3744160 | - |
| SIZE_13 | `{"center_size":1300,"chunk_size":36056,"dimensions":172,"kmax":17,"kmin":9,"points":36056}` | new | 6201632 | - |
| SIZE_14 | `{"center_size":1400,"chunk_size":48611,"dimensions":208,"kmax":18,"kmin":9,"points":48611}` | new | 10111088 | - |
| SIZE_15 | `{"center_size":1500,"chunk_size":65536,"dimensions":256,"kmax":19,"kmin":10,"points":65536}` | synthetic-65536 | 16777216 | - |
| SIZE_16 | `{"center_size":1600,"chunk_size":65536,"dimensions":268,"kmax":20,"kmin":10,"points":75281}` | new | 20175308 | - |
| SIZE_17 | `{"center_size":1700,"chunk_size":65536,"dimensions":280,"kmax":21,"kmin":11,"points":86475}` | new | 24213000 | - |
| SIZE_18 | `{"center_size":1800,"chunk_size":65536,"dimensions":292,"kmax":22,"kmin":11,"points":99334}` | new | 29005528 | - |
| SIZE_19 | `{"center_size":1900,"chunk_size":65536,"dimensions":304,"kmax":23,"kmin":12,"points":114105}` | new | 34687920 | - |
| SIZE_20 | `{"center_size":2000,"chunk_size":65536,"dimensions":320,"kmax":24,"kmin":12,"points":131072}` | new | 41943040 | - |

### Rodinia OpenMP / b+tree

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"queries":128,"records":1024}` | new | 1152 | ../../data/generated_20/b+tree/keys_SIZE_01.txt<br>../../data/generated_20/b+tree/commands_SIZE_01.txt |
| SIZE_02 | `{"queries":209,"records":1674}` | new | 1883 | ../../data/generated_20/b+tree/keys_SIZE_02.txt<br>../../data/generated_20/b+tree/commands_SIZE_02.txt |
| SIZE_03 | `{"queries":342,"records":2738}` | new | 3080 | ../../data/generated_20/b+tree/keys_SIZE_03.txt<br>../../data/generated_20/b+tree/commands_SIZE_03.txt |
| SIZE_04 | `{"queries":559,"records":4477}` | new | 5036 | ../../data/generated_20/b+tree/keys_SIZE_04.txt<br>../../data/generated_20/b+tree/commands_SIZE_04.txt |
| SIZE_05 | `{"queries":915,"records":7320}` | new | 8235 | ../../data/generated_20/b+tree/keys_SIZE_05.txt<br>../../data/generated_20/b+tree/commands_SIZE_05.txt |
| SIZE_06 | `{"queries":1496,"records":11969}` | new | 13465 | ../../data/generated_20/b+tree/keys_SIZE_06.txt<br>../../data/generated_20/b+tree/commands_SIZE_06.txt |
| SIZE_07 | `{"queries":2446,"records":19570}` | new | 22016 | ../../data/generated_20/b+tree/keys_SIZE_07.txt<br>../../data/generated_20/b+tree/commands_SIZE_07.txt |
| SIZE_08 | `{"queries":4000,"records":32000}` | new | 36000 | ../../data/generated_20/b+tree/keys_SIZE_08.txt<br>../../data/generated_20/b+tree/commands_SIZE_08.txt |
| SIZE_09 | `{"queries":6540,"records":52324}` | new | 58864 | ../../data/generated_20/b+tree/keys_SIZE_09.txt<br>../../data/generated_20/b+tree/commands_SIZE_09.txt |
| SIZE_10 | `{"queries":10694,"records":85556}` | new | 96250 | ../../data/generated_20/b+tree/keys_SIZE_10.txt<br>../../data/generated_20/b+tree/commands_SIZE_10.txt |
| SIZE_11 | `{"queries":17486,"records":139894}` | new | 157380 | ../../data/generated_20/b+tree/keys_SIZE_11.txt<br>../../data/generated_20/b+tree/commands_SIZE_11.txt |
| SIZE_12 | `{"queries":28593,"records":228744}` | new | 257337 | ../../data/generated_20/b+tree/keys_SIZE_12.txt<br>../../data/generated_20/b+tree/commands_SIZE_12.txt |
| SIZE_13 | `{"queries":46753,"records":374024}` | new | 420777 | ../../data/generated_20/b+tree/keys_SIZE_13.txt<br>../../data/generated_20/b+tree/commands_SIZE_13.txt |
| SIZE_14 | `{"queries":50000,"records":611575}` | new | 661575 | ../../data/generated_20/b+tree/keys_SIZE_14.txt<br>../../data/generated_20/b+tree/commands_SIZE_14.txt |
| SIZE_15 | `{"queries":50000,"records":1000000}` | new; parameter anchor=mil.txt+command.txt | 1050000 | ../../data/generated_20/b+tree/keys_SIZE_15.txt<br>../../data/generated_20/b+tree/commands_SIZE_15.txt |
| SIZE_16 | `{"queries":50000,"records":1148698}` | new | 1198698 | ../../data/generated_20/b+tree/keys_SIZE_16.txt<br>../../data/generated_20/b+tree/commands_SIZE_16.txt |
| SIZE_17 | `{"queries":50000,"records":1319508}` | new | 1369508 | ../../data/generated_20/b+tree/keys_SIZE_17.txt<br>../../data/generated_20/b+tree/commands_SIZE_17.txt |
| SIZE_18 | `{"queries":50000,"records":1515717}` | new | 1565717 | ../../data/generated_20/b+tree/keys_SIZE_18.txt<br>../../data/generated_20/b+tree/commands_SIZE_18.txt |
| SIZE_19 | `{"queries":50000,"records":1741101}` | new | 1791101 | ../../data/generated_20/b+tree/keys_SIZE_19.txt<br>../../data/generated_20/b+tree/commands_SIZE_19.txt |
| SIZE_20 | `{"queries":50000,"records":2000000}` | new | 2050000 | ../../data/generated_20/b+tree/keys_SIZE_20.txt<br>../../data/generated_20/b+tree/commands_SIZE_20.txt |

### Rodinia OpenMP / backprop

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"input_layer_size":1024}` | layer-1024 | 1024 | - |
| SIZE_02 | `{"input_layer_size":1376}` | new | 1376 | - |
| SIZE_03 | `{"input_layer_size":1856}` | new | 1856 | - |
| SIZE_04 | `{"input_layer_size":2496}` | new | 2496 | - |
| SIZE_05 | `{"input_layer_size":3360}` | new | 3360 | - |
| SIZE_06 | `{"input_layer_size":4528}` | new | 4528 | - |
| SIZE_07 | `{"input_layer_size":6080}` | new | 6080 | - |
| SIZE_08 | `{"input_layer_size":8192}` | new | 8192 | - |
| SIZE_09 | `{"input_layer_size":11024}` | new | 11024 | - |
| SIZE_10 | `{"input_layer_size":14832}` | new | 14832 | - |
| SIZE_11 | `{"input_layer_size":19968}` | new | 19968 | - |
| SIZE_12 | `{"input_layer_size":26880}` | new | 26880 | - |
| SIZE_13 | `{"input_layer_size":36176}` | new | 36176 | - |
| SIZE_14 | `{"input_layer_size":48688}` | new | 48688 | - |
| SIZE_15 | `{"input_layer_size":65536}` | layer-65536 | 65536 | - |
| SIZE_16 | `{"input_layer_size":75280}` | new | 75280 | - |
| SIZE_17 | `{"input_layer_size":86480}` | new | 86480 | - |
| SIZE_18 | `{"input_layer_size":99328}` | new | 99328 | - |
| SIZE_19 | `{"input_layer_size":114112}` | new | 114112 | - |
| SIZE_20 | `{"input_layer_size":131072}` | new | 131072 | - |

### Rodinia OpenMP / bfs

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"average_degree":6,"edges":24576,"nodes":4096}` | new; parameter anchor=graph4096.txt | 28672 | ../../data/generated_20/bfs/graph_SIZE_01.txt |
| SIZE_02 | `{"average_degree":6,"edges":36396,"nodes":6066}` | new | 42462 | ../../data/generated_20/bfs/graph_SIZE_02.txt |
| SIZE_03 | `{"average_degree":6,"edges":53904,"nodes":8984}` | new | 62888 | ../../data/generated_20/bfs/graph_SIZE_03.txt |
| SIZE_04 | `{"average_degree":6,"edges":79824,"nodes":13304}` | new | 93128 | ../../data/generated_20/bfs/graph_SIZE_04.txt |
| SIZE_05 | `{"average_degree":6,"edges":118218,"nodes":19703}` | new | 137921 | ../../data/generated_20/bfs/graph_SIZE_05.txt |
| SIZE_06 | `{"average_degree":6,"edges":175080,"nodes":29180}` | new | 204260 | ../../data/generated_20/bfs/graph_SIZE_06.txt |
| SIZE_07 | `{"average_degree":6,"edges":259290,"nodes":43215}` | new | 302505 | ../../data/generated_20/bfs/graph_SIZE_07.txt |
| SIZE_08 | `{"average_degree":6,"edges":384000,"nodes":64000}` | new | 448000 | ../../data/generated_20/bfs/graph_SIZE_08.txt |
| SIZE_09 | `{"average_degree":6,"edges":568692,"nodes":94782}` | new | 663474 | ../../data/generated_20/bfs/graph_SIZE_09.txt |
| SIZE_10 | `{"average_degree":6,"edges":842214,"nodes":140369}` | new | 982583 | ../../data/generated_20/bfs/graph_SIZE_10.txt |
| SIZE_11 | `{"average_degree":6,"edges":1247292,"nodes":207882}` | new | 1455174 | ../../data/generated_20/bfs/graph_SIZE_11.txt |
| SIZE_12 | `{"average_degree":6,"edges":1847202,"nodes":307867}` | new | 2155069 | ../../data/generated_20/bfs/graph_SIZE_12.txt |
| SIZE_13 | `{"average_degree":6,"edges":2735646,"nodes":455941}` | new | 3191587 | ../../data/generated_20/bfs/graph_SIZE_13.txt |
| SIZE_14 | `{"average_degree":6,"edges":4051404,"nodes":675234}` | new | 4726638 | ../../data/generated_20/bfs/graph_SIZE_14.txt |
| SIZE_15 | `{"average_degree":6,"edges":6000000,"nodes":1000000}` | new; parameter anchor=graph1MW_6.txt | 7000000 | ../../data/generated_20/bfs/graph_SIZE_15.txt |
| SIZE_16 | `{"average_degree":6,"edges":6892188,"nodes":1148698}` | new | 8040886 | ../../data/generated_20/bfs/graph_SIZE_16.txt |
| SIZE_17 | `{"average_degree":6,"edges":7917048,"nodes":1319508}` | new | 9236556 | ../../data/generated_20/bfs/graph_SIZE_17.txt |
| SIZE_18 | `{"average_degree":6,"edges":9094302,"nodes":1515717}` | new | 10610019 | ../../data/generated_20/bfs/graph_SIZE_18.txt |
| SIZE_19 | `{"average_degree":6,"edges":10446606,"nodes":1741101}` | new | 12187707 | ../../data/generated_20/bfs/graph_SIZE_19.txt |
| SIZE_20 | `{"average_degree":6,"edges":12000000,"nodes":2000000}` | new | 14000000 | ../../data/generated_20/bfs/graph_SIZE_20.txt |

### Rodinia OpenMP / cfd

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"faces_per_cell":4,"mesh_cells":8192}` | new | 8192 | ../../data/generated_20/cfd/mesh_SIZE_01.dat |
| SIZE_02 | `{"faces_per_cell":4,"mesh_cells":10781}` | new | 10781 | ../../data/generated_20/cfd/mesh_SIZE_02.dat |
| SIZE_03 | `{"faces_per_cell":4,"mesh_cells":14189}` | new | 14189 | ../../data/generated_20/cfd/mesh_SIZE_03.dat |
| SIZE_04 | `{"faces_per_cell":4,"mesh_cells":18675}` | new | 18675 | ../../data/generated_20/cfd/mesh_SIZE_04.dat |
| SIZE_05 | `{"faces_per_cell":4,"mesh_cells":24578}` | new | 24578 | ../../data/generated_20/cfd/mesh_SIZE_05.dat |
| SIZE_06 | `{"faces_per_cell":4,"mesh_cells":32347}` | new | 32347 | ../../data/generated_20/cfd/mesh_SIZE_06.dat |
| SIZE_07 | `{"faces_per_cell":4,"mesh_cells":42571}` | new | 42571 | ../../data/generated_20/cfd/mesh_SIZE_07.dat |
| SIZE_08 | `{"faces_per_cell":4,"mesh_cells":56028}` | new | 56028 | ../../data/generated_20/cfd/mesh_SIZE_08.dat |
| SIZE_09 | `{"faces_per_cell":4,"mesh_cells":73738}` | new | 73738 | ../../data/generated_20/cfd/mesh_SIZE_09.dat |
| SIZE_10 | `{"faces_per_cell":4,"mesh_cells":97046}` | new; parameter anchor=fvcorr.domn.097K | 97046 | ../../data/generated_20/cfd/mesh_SIZE_10.dat |
| SIZE_11 | `{"faces_per_cell":4,"mesh_cells":111351}` | new | 111351 | ../../data/generated_20/cfd/mesh_SIZE_11.dat |
| SIZE_12 | `{"faces_per_cell":4,"mesh_cells":127764}` | new | 127764 | ../../data/generated_20/cfd/mesh_SIZE_12.dat |
| SIZE_13 | `{"faces_per_cell":4,"mesh_cells":146597}` | new | 146597 | ../../data/generated_20/cfd/mesh_SIZE_13.dat |
| SIZE_14 | `{"faces_per_cell":4,"mesh_cells":168206}` | new | 168206 | ../../data/generated_20/cfd/mesh_SIZE_14.dat |
| SIZE_15 | `{"faces_per_cell":4,"mesh_cells":193000}` | new; parameter anchor=fvcorr.domn.193K | 193000 | ../../data/generated_20/cfd/mesh_SIZE_15.dat |
| SIZE_16 | `{"faces_per_cell":4,"mesh_cells":205189}` | new | 205189 | ../../data/generated_20/cfd/mesh_SIZE_16.dat |
| SIZE_17 | `{"faces_per_cell":4,"mesh_cells":218148}` | new | 218148 | ../../data/generated_20/cfd/mesh_SIZE_17.dat |
| SIZE_18 | `{"faces_per_cell":4,"mesh_cells":231925}` | new | 231925 | ../../data/generated_20/cfd/mesh_SIZE_18.dat |
| SIZE_19 | `{"faces_per_cell":4,"mesh_cells":246572}` | new | 246572 | ../../data/generated_20/cfd/mesh_SIZE_19.dat |
| SIZE_20 | `{"faces_per_cell":4,"mesh_cells":262144}` | new | 262144 | ../../data/generated_20/cfd/mesh_SIZE_20.dat |

### Rodinia OpenMP / heartwall

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":2,"input_file":"test.avi"}` | video-2-frames | 2 | ../../data/heartwall/test.avi |
| SIZE_02 | `{"frames":3,"input_file":"test.avi"}` | new | 3 | ../../data/heartwall/test.avi |
| SIZE_03 | `{"frames":4,"input_file":"test.avi"}` | new | 4 | ../../data/heartwall/test.avi |
| SIZE_04 | `{"frames":5,"input_file":"test.avi"}` | new | 5 | ../../data/heartwall/test.avi |
| SIZE_05 | `{"frames":6,"input_file":"test.avi"}` | new | 6 | ../../data/heartwall/test.avi |
| SIZE_06 | `{"frames":7,"input_file":"test.avi"}` | new | 7 | ../../data/heartwall/test.avi |
| SIZE_07 | `{"frames":9,"input_file":"test.avi"}` | new | 9 | ../../data/heartwall/test.avi |
| SIZE_08 | `{"frames":12,"input_file":"test.avi"}` | new | 12 | ../../data/heartwall/test.avi |
| SIZE_09 | `{"frames":15,"input_file":"test.avi"}` | new | 15 | ../../data/heartwall/test.avi |
| SIZE_10 | `{"frames":20,"input_file":"test.avi"}` | video-20-frames | 20 | ../../data/heartwall/test.avi |
| SIZE_11 | `{"frames":23,"input_file":"test.avi"}` | new | 23 | ../../data/heartwall/test.avi |
| SIZE_12 | `{"frames":28,"input_file":"test.avi"}` | new | 28 | ../../data/heartwall/test.avi |
| SIZE_13 | `{"frames":32,"input_file":"test.avi"}` | new | 32 | ../../data/heartwall/test.avi |
| SIZE_14 | `{"frames":38,"input_file":"test.avi"}` | new | 38 | ../../data/heartwall/test.avi |
| SIZE_15 | `{"frames":45,"input_file":"test.avi"}` | new | 45 | ../../data/heartwall/test.avi |
| SIZE_16 | `{"frames":53,"input_file":"test.avi"}` | new | 53 | ../../data/heartwall/test.avi |
| SIZE_17 | `{"frames":62,"input_file":"test.avi"}` | new | 62 | ../../data/heartwall/test.avi |
| SIZE_18 | `{"frames":72,"input_file":"test.avi"}` | new | 72 | ../../data/heartwall/test.avi |
| SIZE_19 | `{"frames":85,"input_file":"test.avi"}` | new | 85 | ../../data/heartwall/test.avi |
| SIZE_20 | `{"frames":100,"input_file":"test.avi"}` | new | 100 | ../../data/heartwall/test.avi |

### Rodinia OpenMP / hotspot

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":64,"iterations":2,"rows":64}` | new; parameter anchor=grid-64 | 8192 | ../../data/generated_20/hotspot/temp_SIZE_01.txt<br>../../data/generated_20/hotspot/power_SIZE_01.txt |
| SIZE_02 | `{"columns":80,"iterations":2,"rows":80}` | new | 12800 | ../../data/generated_20/hotspot/temp_SIZE_02.txt<br>../../data/generated_20/hotspot/power_SIZE_02.txt |
| SIZE_03 | `{"columns":96,"iterations":2,"rows":96}` | new | 18432 | ../../data/generated_20/hotspot/temp_SIZE_03.txt<br>../../data/generated_20/hotspot/power_SIZE_03.txt |
| SIZE_04 | `{"columns":112,"iterations":2,"rows":112}` | new | 25088 | ../../data/generated_20/hotspot/temp_SIZE_04.txt<br>../../data/generated_20/hotspot/power_SIZE_04.txt |
| SIZE_05 | `{"columns":144,"iterations":2,"rows":144}` | new | 41472 | ../../data/generated_20/hotspot/temp_SIZE_05.txt<br>../../data/generated_20/hotspot/power_SIZE_05.txt |
| SIZE_06 | `{"columns":176,"iterations":2,"rows":176}` | new | 61952 | ../../data/generated_20/hotspot/temp_SIZE_06.txt<br>../../data/generated_20/hotspot/power_SIZE_06.txt |
| SIZE_07 | `{"columns":208,"iterations":2,"rows":208}` | new | 86528 | ../../data/generated_20/hotspot/temp_SIZE_07.txt<br>../../data/generated_20/hotspot/power_SIZE_07.txt |
| SIZE_08 | `{"columns":256,"iterations":2,"rows":256}` | new | 131072 | ../../data/generated_20/hotspot/temp_SIZE_08.txt<br>../../data/generated_20/hotspot/power_SIZE_08.txt |
| SIZE_09 | `{"columns":320,"iterations":2,"rows":320}` | new | 204800 | ../../data/generated_20/hotspot/temp_SIZE_09.txt<br>../../data/generated_20/hotspot/power_SIZE_09.txt |
| SIZE_10 | `{"columns":384,"iterations":2,"rows":384}` | new | 294912 | ../../data/generated_20/hotspot/temp_SIZE_10.txt<br>../../data/generated_20/hotspot/power_SIZE_10.txt |
| SIZE_11 | `{"columns":464,"iterations":2,"rows":464}` | new | 430592 | ../../data/generated_20/hotspot/temp_SIZE_11.txt<br>../../data/generated_20/hotspot/power_SIZE_11.txt |
| SIZE_12 | `{"columns":560,"iterations":2,"rows":560}` | new | 627200 | ../../data/generated_20/hotspot/temp_SIZE_12.txt<br>../../data/generated_20/hotspot/power_SIZE_12.txt |
| SIZE_13 | `{"columns":688,"iterations":2,"rows":688}` | new | 946688 | ../../data/generated_20/hotspot/temp_SIZE_13.txt<br>../../data/generated_20/hotspot/power_SIZE_13.txt |
| SIZE_14 | `{"columns":832,"iterations":2,"rows":832}` | new | 1384448 | ../../data/generated_20/hotspot/temp_SIZE_14.txt<br>../../data/generated_20/hotspot/power_SIZE_14.txt |
| SIZE_15 | `{"columns":1024,"iterations":2,"rows":1024}` | new; parameter anchor=grid-1024 | 2097152 | ../../data/generated_20/hotspot/temp_SIZE_15.txt<br>../../data/generated_20/hotspot/power_SIZE_15.txt |
| SIZE_16 | `{"columns":1104,"iterations":2,"rows":1104}` | new | 2437632 | ../../data/generated_20/hotspot/temp_SIZE_16.txt<br>../../data/generated_20/hotspot/power_SIZE_16.txt |
| SIZE_17 | `{"columns":1200,"iterations":2,"rows":1200}` | new | 2880000 | ../../data/generated_20/hotspot/temp_SIZE_17.txt<br>../../data/generated_20/hotspot/power_SIZE_17.txt |
| SIZE_18 | `{"columns":1312,"iterations":2,"rows":1312}` | new | 3442688 | ../../data/generated_20/hotspot/temp_SIZE_18.txt<br>../../data/generated_20/hotspot/power_SIZE_18.txt |
| SIZE_19 | `{"columns":1408,"iterations":2,"rows":1408}` | new | 3964928 | ../../data/generated_20/hotspot/temp_SIZE_19.txt<br>../../data/generated_20/hotspot/power_SIZE_19.txt |
| SIZE_20 | `{"columns":1536,"iterations":2,"rows":1536}` | new | 4718592 | ../../data/generated_20/hotspot/temp_SIZE_20.txt<br>../../data/generated_20/hotspot/power_SIZE_20.txt |

### Rodinia OpenMP / hotspot3D

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"grid_size":64,"iterations":10,"layers":8}` | new; parameter anchor=grid-64x8 | 327680 | ../../data/generated_20/hotspot3D/power_SIZE_01.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_01.txt |
| SIZE_02 | `{"grid_size":80,"iterations":12,"layers":8}` | new | 614400 | ../../data/generated_20/hotspot3D/power_SIZE_02.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_02.txt |
| SIZE_03 | `{"grid_size":96,"iterations":14,"layers":8}` | new | 1032192 | ../../data/generated_20/hotspot3D/power_SIZE_03.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_03.txt |
| SIZE_04 | `{"grid_size":112,"iterations":16,"layers":8}` | new | 1605632 | ../../data/generated_20/hotspot3D/power_SIZE_04.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_04.txt |
| SIZE_05 | `{"grid_size":128,"iterations":19,"layers":8}` | new | 2490368 | ../../data/generated_20/hotspot3D/power_SIZE_05.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_05.txt |
| SIZE_06 | `{"grid_size":144,"iterations":23,"layers":8}` | new | 3815424 | ../../data/generated_20/hotspot3D/power_SIZE_06.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_06.txt |
| SIZE_07 | `{"grid_size":160,"iterations":27,"layers":8}` | new | 5529600 | ../../data/generated_20/hotspot3D/power_SIZE_07.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_07.txt |
| SIZE_08 | `{"grid_size":176,"iterations":32,"layers":8}` | new | 7929856 | ../../data/generated_20/hotspot3D/power_SIZE_08.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_08.txt |
| SIZE_09 | `{"grid_size":208,"iterations":37,"layers":8}` | new | 12806144 | ../../data/generated_20/hotspot3D/power_SIZE_09.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_09.txt |
| SIZE_10 | `{"grid_size":240,"iterations":44,"layers":8}` | new | 20275200 | ../../data/generated_20/hotspot3D/power_SIZE_10.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_10.txt |
| SIZE_11 | `{"grid_size":288,"iterations":52,"layers":8}` | new | 34504704 | ../../data/generated_20/hotspot3D/power_SIZE_11.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_11.txt |
| SIZE_12 | `{"grid_size":320,"iterations":61,"layers":8}` | new | 49971200 | ../../data/generated_20/hotspot3D/power_SIZE_12.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_12.txt |
| SIZE_13 | `{"grid_size":384,"iterations":72,"layers":8}` | new | 84934656 | ../../data/generated_20/hotspot3D/power_SIZE_13.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_13.txt |
| SIZE_14 | `{"grid_size":448,"iterations":85,"layers":8}` | new | 136478720 | ../../data/generated_20/hotspot3D/power_SIZE_14.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_14.txt |
| SIZE_15 | `{"grid_size":512,"iterations":100,"layers":8}` | new; parameter anchor=grid-512x8 | 209715200 | ../../data/generated_20/hotspot3D/power_SIZE_15.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_15.txt |
| SIZE_16 | `{"grid_size":528,"iterations":107,"layers":8}` | new | 238639104 | ../../data/generated_20/hotspot3D/power_SIZE_16.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_16.txt |
| SIZE_17 | `{"grid_size":560,"iterations":114,"layers":8}` | new | 286003200 | ../../data/generated_20/hotspot3D/power_SIZE_17.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_17.txt |
| SIZE_18 | `{"grid_size":592,"iterations":122,"layers":8}` | new | 342052864 | ../../data/generated_20/hotspot3D/power_SIZE_18.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_18.txt |
| SIZE_19 | `{"grid_size":608,"iterations":131,"layers":8}` | new | 387407872 | ../../data/generated_20/hotspot3D/power_SIZE_19.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_19.txt |
| SIZE_20 | `{"grid_size":640,"iterations":140,"layers":8}` | new | 458752000 | ../../data/generated_20/hotspot3D/power_SIZE_20.txt<br>../../data/generated_20/hotspot3D/temp_SIZE_20.txt |

### Rodinia OpenMP / kmeans

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"attributes":16,"clusters":5,"points":100}` | new; parameter anchor=points-100 | 1600 | ../../data/generated_20/kmeans/points_SIZE_01.bin |
| SIZE_02 | `{"attributes":16,"clusters":5,"points":184}` | new | 2944 | ../../data/generated_20/kmeans/points_SIZE_02.bin |
| SIZE_03 | `{"attributes":16,"clusters":5,"points":337}` | new | 5392 | ../../data/generated_20/kmeans/points_SIZE_03.bin |
| SIZE_04 | `{"attributes":16,"clusters":5,"points":619}` | new | 9904 | ../../data/generated_20/kmeans/points_SIZE_04.bin |
| SIZE_05 | `{"attributes":16,"clusters":5,"points":1136}` | new | 18176 | ../../data/generated_20/kmeans/points_SIZE_05.bin |
| SIZE_06 | `{"attributes":16,"clusters":5,"points":2085}` | new | 33360 | ../../data/generated_20/kmeans/points_SIZE_06.bin |
| SIZE_07 | `{"attributes":16,"clusters":5,"points":3829}` | new | 61264 | ../../data/generated_20/kmeans/points_SIZE_07.bin |
| SIZE_08 | `{"attributes":16,"clusters":5,"points":7029}` | new | 112464 | ../../data/generated_20/kmeans/points_SIZE_08.bin |
| SIZE_09 | `{"attributes":16,"clusters":5,"points":12904}` | new | 206464 | ../../data/generated_20/kmeans/points_SIZE_09.bin |
| SIZE_10 | `{"attributes":16,"clusters":5,"points":23689}` | new | 379024 | ../../data/generated_20/kmeans/points_SIZE_10.bin |
| SIZE_11 | `{"attributes":16,"clusters":5,"points":43490}` | new | 695840 | ../../data/generated_20/kmeans/points_SIZE_11.bin |
| SIZE_12 | `{"attributes":16,"clusters":5,"points":79841}` | new | 1277456 | ../../data/generated_20/kmeans/points_SIZE_12.bin |
| SIZE_13 | `{"attributes":16,"clusters":5,"points":146577}` | new | 2345232 | ../../data/generated_20/kmeans/points_SIZE_13.bin |
| SIZE_14 | `{"attributes":16,"clusters":5,"points":269095}` | new | 4305520 | ../../data/generated_20/kmeans/points_SIZE_14.bin |
| SIZE_15 | `{"attributes":16,"clusters":5,"points":494020}` | new; parameter anchor=kdd-cup | 7904320 | ../../data/generated_20/kmeans/points_SIZE_15.bin |
| SIZE_16 | `{"attributes":16,"clusters":5,"points":513601}` | new | 8217616 | ../../data/generated_20/kmeans/points_SIZE_16.bin |
| SIZE_17 | `{"attributes":16,"clusters":5,"points":533958}` | new | 8543328 | ../../data/generated_20/kmeans/points_SIZE_17.bin |
| SIZE_18 | `{"attributes":16,"clusters":5,"points":555122}` | new | 8881952 | ../../data/generated_20/kmeans/points_SIZE_18.bin |
| SIZE_19 | `{"attributes":16,"clusters":5,"points":577125}` | new | 9234000 | ../../data/generated_20/kmeans/points_SIZE_19.bin |
| SIZE_20 | `{"attributes":16,"clusters":5,"points":600000}` | new | 9600000 | ../../data/generated_20/kmeans/points_SIZE_20.bin |

### Rodinia OpenMP / lavaMD

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"boxes1d":2,"boxes_total":8,"particles_per_box":100}` | new | 800 | - |
| SIZE_02 | `{"boxes1d":3,"boxes_total":27,"particles_per_box":100}` | new | 2700 | - |
| SIZE_03 | `{"boxes1d":4,"boxes_total":64,"particles_per_box":100}` | boxes-4 | 6400 | - |
| SIZE_04 | `{"boxes1d":5,"boxes_total":125,"particles_per_box":100}` | new | 12500 | - |
| SIZE_05 | `{"boxes1d":6,"boxes_total":216,"particles_per_box":100}` | new | 21600 | - |
| SIZE_06 | `{"boxes1d":7,"boxes_total":343,"particles_per_box":100}` | new | 34300 | - |
| SIZE_07 | `{"boxes1d":8,"boxes_total":512,"particles_per_box":100}` | new | 51200 | - |
| SIZE_08 | `{"boxes1d":9,"boxes_total":729,"particles_per_box":100}` | new | 72900 | - |
| SIZE_09 | `{"boxes1d":10,"boxes_total":1000,"particles_per_box":100}` | boxes-10 | 100000 | - |
| SIZE_10 | `{"boxes1d":11,"boxes_total":1331,"particles_per_box":100}` | new | 133100 | - |
| SIZE_11 | `{"boxes1d":12,"boxes_total":1728,"particles_per_box":100}` | new | 172800 | - |
| SIZE_12 | `{"boxes1d":13,"boxes_total":2197,"particles_per_box":100}` | new | 219700 | - |
| SIZE_13 | `{"boxes1d":14,"boxes_total":2744,"particles_per_box":100}` | new | 274400 | - |
| SIZE_14 | `{"boxes1d":15,"boxes_total":3375,"particles_per_box":100}` | new | 337500 | - |
| SIZE_15 | `{"boxes1d":16,"boxes_total":4096,"particles_per_box":100}` | new | 409600 | - |
| SIZE_16 | `{"boxes1d":17,"boxes_total":4913,"particles_per_box":100}` | new | 491300 | - |
| SIZE_17 | `{"boxes1d":18,"boxes_total":5832,"particles_per_box":100}` | new | 583200 | - |
| SIZE_18 | `{"boxes1d":19,"boxes_total":6859,"particles_per_box":100}` | new | 685900 | - |
| SIZE_19 | `{"boxes1d":20,"boxes_total":8000,"particles_per_box":100}` | new | 800000 | - |
| SIZE_20 | `{"boxes1d":21,"boxes_total":9261,"particles_per_box":100}` | new | 926100 | - |

### Rodinia OpenMP / leukocyte

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":1,"input_file":"testfile.avi"}` | video-1-frame | 1 | ../../data/leukocyte/testfile.avi |
| SIZE_02 | `{"frames":2,"input_file":"testfile.avi"}` | new | 2 | ../../data/leukocyte/testfile.avi |
| SIZE_03 | `{"frames":3,"input_file":"testfile.avi"}` | new | 3 | ../../data/leukocyte/testfile.avi |
| SIZE_04 | `{"frames":4,"input_file":"testfile.avi"}` | new | 4 | ../../data/leukocyte/testfile.avi |
| SIZE_05 | `{"frames":5,"input_file":"testfile.avi"}` | video-5-frames | 5 | ../../data/leukocyte/testfile.avi |
| SIZE_06 | `{"frames":6,"input_file":"testfile.avi"}` | new | 6 | ../../data/leukocyte/testfile.avi |
| SIZE_07 | `{"frames":7,"input_file":"testfile.avi"}` | new | 7 | ../../data/leukocyte/testfile.avi |
| SIZE_08 | `{"frames":8,"input_file":"testfile.avi"}` | new | 8 | ../../data/leukocyte/testfile.avi |
| SIZE_09 | `{"frames":9,"input_file":"testfile.avi"}` | new | 9 | ../../data/leukocyte/testfile.avi |
| SIZE_10 | `{"frames":10,"input_file":"testfile.avi"}` | new | 10 | ../../data/leukocyte/testfile.avi |
| SIZE_11 | `{"frames":11,"input_file":"testfile.avi"}` | new | 11 | ../../data/leukocyte/testfile.avi |
| SIZE_12 | `{"frames":12,"input_file":"testfile.avi"}` | new | 12 | ../../data/leukocyte/testfile.avi |
| SIZE_13 | `{"frames":13,"input_file":"testfile.avi"}` | new | 13 | ../../data/leukocyte/testfile.avi |
| SIZE_14 | `{"frames":14,"input_file":"testfile.avi"}` | new | 14 | ../../data/leukocyte/testfile.avi |
| SIZE_15 | `{"frames":15,"input_file":"testfile.avi"}` | new | 15 | ../../data/leukocyte/testfile.avi |
| SIZE_16 | `{"frames":16,"input_file":"testfile.avi"}` | new | 16 | ../../data/leukocyte/testfile.avi |
| SIZE_17 | `{"frames":17,"input_file":"testfile.avi"}` | new | 17 | ../../data/leukocyte/testfile.avi |
| SIZE_18 | `{"frames":18,"input_file":"testfile.avi"}` | new | 18 | ../../data/leukocyte/testfile.avi |
| SIZE_19 | `{"frames":19,"input_file":"testfile.avi"}` | new | 19 | ../../data/leukocyte/testfile.avi |
| SIZE_20 | `{"frames":20,"input_file":"testfile.avi"}` | new | 20 | ../../data/leukocyte/testfile.avi |

### Rodinia OpenMP / lud

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"matrix_dimension":256}` | matrix-256 | 16777216 | - |
| SIZE_02 | `{"matrix_dimension":304}` | new | 28094464 | - |
| SIZE_03 | `{"matrix_dimension":352}` | new | 43614208 | - |
| SIZE_04 | `{"matrix_dimension":400}` | new | 64000000 | - |
| SIZE_05 | `{"matrix_dimension":464}` | new | 99897344 | - |
| SIZE_06 | `{"matrix_dimension":544}` | new | 160989184 | - |
| SIZE_07 | `{"matrix_dimension":624}` | new | 242970624 | - |
| SIZE_08 | `{"matrix_dimension":720}` | new | 373248000 | - |
| SIZE_09 | `{"matrix_dimension":832}` | new | 575930368 | - |
| SIZE_10 | `{"matrix_dimension":976}` | new | 929714176 | - |
| SIZE_11 | `{"matrix_dimension":1136}` | new | 1466003456 | - |
| SIZE_12 | `{"matrix_dimension":1312}` | new | 2258403328 | - |
| SIZE_13 | `{"matrix_dimension":1520}` | new | 3511808000 | - |
| SIZE_14 | `{"matrix_dimension":1760}` | new | 5451776000 | - |
| SIZE_15 | `{"matrix_dimension":2048}` | new | 8589934592 | - |
| SIZE_16 | `{"matrix_dimension":2688}` | new | 19421724672 | - |
| SIZE_17 | `{"matrix_dimension":3536}` | new | 44211654656 | - |
| SIZE_18 | `{"matrix_dimension":4640}` | new | 99897344000 | - |
| SIZE_19 | `{"matrix_dimension":6096}` | new | 226534772736 | - |
| SIZE_20 | `{"matrix_dimension":8000}` | matrix-8000 | 512000000000 | - |

### Rodinia OpenMP / myocyte

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"instances":1,"mode":0,"simulation_time_ms":10}` | workload-10 | 10 | - |
| SIZE_02 | `{"instances":1,"mode":0,"simulation_time_ms":12}` | new | 12 | - |
| SIZE_03 | `{"instances":1,"mode":0,"simulation_time_ms":14}` | new | 14 | - |
| SIZE_04 | `{"instances":1,"mode":0,"simulation_time_ms":16}` | new | 16 | - |
| SIZE_05 | `{"instances":1,"mode":0,"simulation_time_ms":19}` | new | 19 | - |
| SIZE_06 | `{"instances":1,"mode":0,"simulation_time_ms":23}` | new | 23 | - |
| SIZE_07 | `{"instances":1,"mode":0,"simulation_time_ms":27}` | new | 27 | - |
| SIZE_08 | `{"instances":1,"mode":0,"simulation_time_ms":32}` | new | 32 | - |
| SIZE_09 | `{"instances":1,"mode":0,"simulation_time_ms":37}` | new | 37 | - |
| SIZE_10 | `{"instances":1,"mode":0,"simulation_time_ms":44}` | new | 44 | - |
| SIZE_11 | `{"instances":1,"mode":0,"simulation_time_ms":52}` | new | 52 | - |
| SIZE_12 | `{"instances":1,"mode":0,"simulation_time_ms":61}` | new | 61 | - |
| SIZE_13 | `{"instances":1,"mode":0,"simulation_time_ms":72}` | new | 72 | - |
| SIZE_14 | `{"instances":1,"mode":0,"simulation_time_ms":85}` | new | 85 | - |
| SIZE_15 | `{"instances":1,"mode":0,"simulation_time_ms":100}` | workload-100 | 100 | - |
| SIZE_16 | `{"instances":1,"mode":0,"simulation_time_ms":115}` | new | 115 | - |
| SIZE_17 | `{"instances":1,"mode":0,"simulation_time_ms":132}` | new | 132 | - |
| SIZE_18 | `{"instances":1,"mode":0,"simulation_time_ms":152}` | new | 152 | - |
| SIZE_19 | `{"instances":1,"mode":0,"simulation_time_ms":174}` | new | 174 | - |
| SIZE_20 | `{"instances":1,"mode":0,"simulation_time_ms":200}` | new | 200 | - |

### Rodinia OpenMP / nn

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"database_records":1024,"latitude":30,"longitude":90,"neighbors":5}` | new | 1024 | ../../data/generated_20/nn/filelist_SIZE_01.txt<br>../../data/generated_20/nn/records_SIZE_01.db |
| SIZE_02 | `{"database_records":1550,"latitude":30,"longitude":90,"neighbors":5}` | new | 1550 | ../../data/generated_20/nn/filelist_SIZE_02.txt<br>../../data/generated_20/nn/records_SIZE_02.db |
| SIZE_03 | `{"database_records":2347,"latitude":30,"longitude":90,"neighbors":5}` | new | 2347 | ../../data/generated_20/nn/filelist_SIZE_03.txt<br>../../data/generated_20/nn/records_SIZE_03.db |
| SIZE_04 | `{"database_records":3553,"latitude":30,"longitude":90,"neighbors":5}` | new | 3553 | ../../data/generated_20/nn/filelist_SIZE_04.txt<br>../../data/generated_20/nn/records_SIZE_04.db |
| SIZE_05 | `{"database_records":5378,"latitude":30,"longitude":90,"neighbors":5}` | new | 5378 | ../../data/generated_20/nn/filelist_SIZE_05.txt<br>../../data/generated_20/nn/records_SIZE_05.db |
| SIZE_06 | `{"database_records":8142,"latitude":30,"longitude":90,"neighbors":5}` | new | 8142 | ../../data/generated_20/nn/filelist_SIZE_06.txt<br>../../data/generated_20/nn/records_SIZE_06.db |
| SIZE_07 | `{"database_records":12326,"latitude":30,"longitude":90,"neighbors":5}` | new | 12326 | ../../data/generated_20/nn/filelist_SIZE_07.txt<br>../../data/generated_20/nn/records_SIZE_07.db |
| SIZE_08 | `{"database_records":18660,"latitude":30,"longitude":90,"neighbors":5}` | new | 18660 | ../../data/generated_20/nn/filelist_SIZE_08.txt<br>../../data/generated_20/nn/records_SIZE_08.db |
| SIZE_09 | `{"database_records":28248,"latitude":30,"longitude":90,"neighbors":5}` | new | 28248 | ../../data/generated_20/nn/filelist_SIZE_09.txt<br>../../data/generated_20/nn/records_SIZE_09.db |
| SIZE_10 | `{"database_records":42764,"latitude":30,"longitude":90,"neighbors":5}` | new; parameter anchor=cane4 databases | 42764 | ../../data/generated_20/nn/filelist_SIZE_10.txt<br>../../data/generated_20/nn/records_SIZE_10.db |
| SIZE_11 | `{"database_records":49897,"latitude":30,"longitude":90,"neighbors":5}` | new | 49897 | ../../data/generated_20/nn/filelist_SIZE_11.txt<br>../../data/generated_20/nn/records_SIZE_11.db |
| SIZE_12 | `{"database_records":58220,"latitude":30,"longitude":90,"neighbors":5}` | new | 58220 | ../../data/generated_20/nn/filelist_SIZE_12.txt<br>../../data/generated_20/nn/records_SIZE_12.db |
| SIZE_13 | `{"database_records":67930,"latitude":30,"longitude":90,"neighbors":5}` | new | 67930 | ../../data/generated_20/nn/filelist_SIZE_13.txt<br>../../data/generated_20/nn/records_SIZE_13.db |
| SIZE_14 | `{"database_records":79261,"latitude":30,"longitude":90,"neighbors":5}` | new | 79261 | ../../data/generated_20/nn/filelist_SIZE_14.txt<br>../../data/generated_20/nn/records_SIZE_14.db |
| SIZE_15 | `{"database_records":92481,"latitude":30,"longitude":90,"neighbors":5}` | new | 92481 | ../../data/generated_20/nn/filelist_SIZE_15.txt<br>../../data/generated_20/nn/records_SIZE_15.db |
| SIZE_16 | `{"database_records":107907,"latitude":30,"longitude":90,"neighbors":5}` | new | 107907 | ../../data/generated_20/nn/filelist_SIZE_16.txt<br>../../data/generated_20/nn/records_SIZE_16.db |
| SIZE_17 | `{"database_records":125905,"latitude":30,"longitude":90,"neighbors":5}` | new | 125905 | ../../data/generated_20/nn/filelist_SIZE_17.txt<br>../../data/generated_20/nn/records_SIZE_17.db |
| SIZE_18 | `{"database_records":146906,"latitude":30,"longitude":90,"neighbors":5}` | new | 146906 | ../../data/generated_20/nn/filelist_SIZE_18.txt<br>../../data/generated_20/nn/records_SIZE_18.db |
| SIZE_19 | `{"database_records":171409,"latitude":30,"longitude":90,"neighbors":5}` | new | 171409 | ../../data/generated_20/nn/filelist_SIZE_19.txt<br>../../data/generated_20/nn/records_SIZE_19.db |
| SIZE_20 | `{"database_records":200000,"latitude":30,"longitude":90,"neighbors":5}` | new | 200000 | ../../data/generated_20/nn/filelist_SIZE_20.txt<br>../../data/generated_20/nn/records_SIZE_20.db |

### Rodinia OpenMP / nw

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"penalty":10,"sequence_dimension":256}` | dimension-256 | 65536 | - |
| SIZE_02 | `{"penalty":10,"sequence_dimension":304}` | new | 92416 | - |
| SIZE_03 | `{"penalty":10,"sequence_dimension":352}` | new | 123904 | - |
| SIZE_04 | `{"penalty":10,"sequence_dimension":400}` | new | 160000 | - |
| SIZE_05 | `{"penalty":10,"sequence_dimension":464}` | new | 215296 | - |
| SIZE_06 | `{"penalty":10,"sequence_dimension":544}` | new | 295936 | - |
| SIZE_07 | `{"penalty":10,"sequence_dimension":624}` | new | 389376 | - |
| SIZE_08 | `{"penalty":10,"sequence_dimension":720}` | new | 518400 | - |
| SIZE_09 | `{"penalty":10,"sequence_dimension":832}` | new | 692224 | - |
| SIZE_10 | `{"penalty":10,"sequence_dimension":976}` | new | 952576 | - |
| SIZE_11 | `{"penalty":10,"sequence_dimension":1136}` | new | 1290496 | - |
| SIZE_12 | `{"penalty":10,"sequence_dimension":1312}` | new | 1721344 | - |
| SIZE_13 | `{"penalty":10,"sequence_dimension":1520}` | new | 2310400 | - |
| SIZE_14 | `{"penalty":10,"sequence_dimension":1760}` | new | 3097600 | - |
| SIZE_15 | `{"penalty":10,"sequence_dimension":2048}` | dimension-2048 | 4194304 | - |
| SIZE_16 | `{"penalty":10,"sequence_dimension":2352}` | new | 5531904 | - |
| SIZE_17 | `{"penalty":10,"sequence_dimension":2704}` | new | 7311616 | - |
| SIZE_18 | `{"penalty":10,"sequence_dimension":3104}` | new | 9634816 | - |
| SIZE_19 | `{"penalty":10,"sequence_dimension":3568}` | new | 12730624 | - |
| SIZE_20 | `{"penalty":10,"sequence_dimension":4096}` | new | 16777216 | - |

### Rodinia OpenMP / particlefilter

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"frames":2,"height":32,"particles":100,"width":32}` | new | 204800 | - |
| SIZE_02 | `{"frames":3,"height":48,"particles":118,"width":48}` | new | 815616 | - |
| SIZE_03 | `{"frames":3,"height":64,"particles":139,"width":64}` | new | 1708032 | - |
| SIZE_04 | `{"frames":4,"height":80,"particles":164,"width":80}` | new | 4198400 | - |
| SIZE_05 | `{"frames":4,"height":96,"particles":193,"width":96}` | new | 7114752 | - |
| SIZE_06 | `{"frames":5,"height":112,"particles":228,"width":112}` | new | 14300160 | - |
| SIZE_07 | `{"frames":5,"height":128,"particles":268,"width":128}` | new | 21954560 | - |
| SIZE_08 | `{"frames":6,"height":144,"particles":316,"width":144}` | new | 39315456 | - |
| SIZE_09 | `{"frames":7,"height":160,"particles":373,"width":160}` | new | 66841600 | - |
| SIZE_10 | `{"frames":7,"height":176,"particles":439,"width":176}` | new | 95189248 | - |
| SIZE_11 | `{"frames":8,"height":192,"particles":518,"width":192}` | new | 152764416 | - |
| SIZE_12 | `{"frames":8,"height":208,"particles":611,"width":208}` | new | 211474432 | - |
| SIZE_13 | `{"frames":9,"height":224,"particles":720,"width":224}` | new | 325140480 | - |
| SIZE_14 | `{"frames":9,"height":240,"particles":848,"width":240}` | new | 439603200 | - |
| SIZE_15 | `{"frames":10,"height":256,"particles":1000,"width":256}` | new | 655360000 | - |
| SIZE_16 | `{"frames":12,"height":272,"particles":1585,"width":272}` | new | 1407175680 | - |
| SIZE_17 | `{"frames":14,"height":288,"particles":2512,"width":288}` | new | 2916974592 | - |
| SIZE_18 | `{"frames":16,"height":304,"particles":3981,"width":304}` | new | 5886529536 | - |
| SIZE_19 | `{"frames":18,"height":320,"particles":6310,"width":320}` | new | 11630592000 | - |
| SIZE_20 | `{"frames":20,"height":336,"particles":10000,"width":336}` | new | 22579200000 | - |

### Rodinia OpenMP / pathfinder

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":1000,"rows":20}` | new | 20000 | - |
| SIZE_02 | `{"columns":1389,"rows":22}` | new | 30558 | - |
| SIZE_03 | `{"columns":1931,"rows":25}` | new | 48275 | - |
| SIZE_04 | `{"columns":2683,"rows":28}` | new | 75124 | - |
| SIZE_05 | `{"columns":3728,"rows":32}` | new | 119296 | - |
| SIZE_06 | `{"columns":5179,"rows":36}` | new | 186444 | - |
| SIZE_07 | `{"columns":7197,"rows":40}` | new | 287880 | - |
| SIZE_08 | `{"columns":10000,"rows":45}` | new | 450000 | - |
| SIZE_09 | `{"columns":13895,"rows":50}` | new | 694750 | - |
| SIZE_10 | `{"columns":19307,"rows":56}` | new | 1081192 | - |
| SIZE_11 | `{"columns":26827,"rows":63}` | new | 1690101 | - |
| SIZE_12 | `{"columns":37276,"rows":71}` | new | 2646596 | - |
| SIZE_13 | `{"columns":51795,"rows":79}` | new | 4091805 | - |
| SIZE_14 | `{"columns":71969,"rows":89}` | new | 6405241 | - |
| SIZE_15 | `{"columns":100000,"rows":100}` | grid-100000x100 | 10000000 | - |
| SIZE_16 | `{"columns":114870,"rows":115}` | new | 13210050 | - |
| SIZE_17 | `{"columns":131951,"rows":132}` | new | 17417532 | - |
| SIZE_18 | `{"columns":151572,"rows":152}` | new | 23038944 | - |
| SIZE_19 | `{"columns":174110,"rows":174}` | new | 30295140 | - |
| SIZE_20 | `{"columns":200000,"rows":200}` | new | 40000000 | - |

### Rodinia OpenMP / srad_v1

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":128,"iterations":10,"lambda":0.5,"rows":128}` | image-128 | 163840 | - |
| SIZE_02 | `{"columns":140,"iterations":12,"lambda":0.5,"rows":141}` | new | 236880 | - |
| SIZE_03 | `{"columns":154,"iterations":14,"lambda":0.5,"rows":156}` | new | 336336 | - |
| SIZE_04 | `{"columns":168,"iterations":16,"lambda":0.5,"rows":172}` | new | 462336 | - |
| SIZE_05 | `{"columns":184,"iterations":19,"lambda":0.5,"rows":189}` | new | 660744 | - |
| SIZE_06 | `{"columns":202,"iterations":23,"lambda":0.5,"rows":209}` | new | 971014 | - |
| SIZE_07 | `{"columns":221,"iterations":27,"lambda":0.5,"rows":230}` | new | 1372410 | - |
| SIZE_08 | `{"columns":242,"iterations":32,"lambda":0.5,"rows":253}` | new | 1959232 | - |
| SIZE_09 | `{"columns":265,"iterations":37,"lambda":0.5,"rows":279}` | new | 2735595 | - |
| SIZE_10 | `{"columns":290,"iterations":44,"lambda":0.5,"rows":308}` | new | 3930080 | - |
| SIZE_11 | `{"columns":318,"iterations":52,"lambda":0.5,"rows":340}` | new | 5622240 | - |
| SIZE_12 | `{"columns":349,"iterations":61,"lambda":0.5,"rows":375}` | new | 7983375 | - |
| SIZE_13 | `{"columns":382,"iterations":72,"lambda":0.5,"rows":413}` | new | 11359152 | - |
| SIZE_14 | `{"columns":418,"iterations":85,"lambda":0.5,"rows":455}` | new | 16166150 | - |
| SIZE_15 | `{"columns":458,"iterations":100,"lambda":0.5,"rows":502}` | image-502x458 | 22991600 | - |
| SIZE_16 | `{"columns":508,"iterations":108,"lambda":0.5,"rows":547}` | new | 30010608 | - |
| SIZE_17 | `{"columns":563,"iterations":118,"lambda":0.5,"rows":595}` | new | 39528230 | - |
| SIZE_18 | `{"columns":625,"iterations":128,"lambda":0.5,"rows":648}` | new | 51840000 | - |
| SIZE_19 | `{"columns":693,"iterations":138,"lambda":0.5,"rows":705}` | new | 67421970 | - |
| SIZE_20 | `{"columns":768,"iterations":150,"lambda":0.5,"rows":768}` | new | 88473600 | - |

### Rodinia OpenMP / srad_v2

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"columns":128,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":128}` | image-128 | 32768 | - |
| SIZE_02 | `{"columns":160,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":160}` | new | 51200 | - |
| SIZE_03 | `{"columns":176,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":176}` | new | 61952 | - |
| SIZE_04 | `{"columns":208,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":208}` | new | 86528 | - |
| SIZE_05 | `{"columns":256,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":256}` | new | 131072 | - |
| SIZE_06 | `{"columns":320,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":320}` | new | 204800 | - |
| SIZE_07 | `{"columns":384,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":384}` | new | 294912 | - |
| SIZE_08 | `{"columns":480,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":480}` | new | 460800 | - |
| SIZE_09 | `{"columns":592,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":592}` | new | 700928 | - |
| SIZE_10 | `{"columns":720,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":720}` | new | 1036800 | - |
| SIZE_11 | `{"columns":896,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":896}` | new | 1605632 | - |
| SIZE_12 | `{"columns":1104,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":1104}` | new | 2437632 | - |
| SIZE_13 | `{"columns":1344,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":1344}` | new | 3612672 | - |
| SIZE_14 | `{"columns":1664,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":1664}` | new | 5537792 | - |
| SIZE_15 | `{"columns":2048,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2048}` | image-2048 | 8388608 | - |
| SIZE_16 | `{"columns":2224,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2224}` | new | 9892352 | - |
| SIZE_17 | `{"columns":2416,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2416}` | new | 11674112 | - |
| SIZE_18 | `{"columns":2608,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2608}` | new | 13603328 | - |
| SIZE_19 | `{"columns":2832,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":2832}` | new | 16040448 | - |
| SIZE_20 | `{"columns":3072,"iterations":2,"lambda":0.5,"roi":"0:127,0:127","rows":3072}` | new | 18874368 | - |

### Rodinia OpenMP / streamcluster

| size_id | parameters | origin | workload_proxy | input files |
| --- | --- | --- | ---: | --- |
| SIZE_01 | `{"center_size":100,"chunk_size":1000,"dimensions":16,"kmax":5,"kmin":3,"points":1000}` | new | 16000 | - |
| SIZE_02 | `{"center_size":200,"chunk_size":1348,"dimensions":20,"kmax":6,"kmin":3,"points":1348}` | new | 26960 | - |
| SIZE_03 | `{"center_size":300,"chunk_size":1818,"dimensions":24,"kmax":7,"kmin":4,"points":1818}` | new | 43632 | - |
| SIZE_04 | `{"center_size":400,"chunk_size":2450,"dimensions":28,"kmax":8,"kmin":4,"points":2450}` | new | 68600 | - |
| SIZE_05 | `{"center_size":500,"chunk_size":3304,"dimensions":36,"kmax":9,"kmin":5,"points":3304}` | new | 118944 | - |
| SIZE_06 | `{"center_size":600,"chunk_size":4454,"dimensions":44,"kmax":10,"kmin":5,"points":4454}` | new | 195976 | - |
| SIZE_07 | `{"center_size":700,"chunk_size":6005,"dimensions":52,"kmax":11,"kmin":6,"points":6005}` | new | 312260 | - |
| SIZE_08 | `{"center_size":800,"chunk_size":8095,"dimensions":64,"kmax":12,"kmin":6,"points":8095}` | new | 518080 | - |
| SIZE_09 | `{"center_size":900,"chunk_size":10914,"dimensions":80,"kmax":13,"kmin":7,"points":10914}` | new | 873120 | - |
| SIZE_10 | `{"center_size":1000,"chunk_size":14714,"dimensions":96,"kmax":14,"kmin":7,"points":14714}` | new | 1412544 | - |
| SIZE_11 | `{"center_size":1100,"chunk_size":19837,"dimensions":116,"kmax":15,"kmin":8,"points":19837}` | new | 2301092 | - |
| SIZE_12 | `{"center_size":1200,"chunk_size":26744,"dimensions":140,"kmax":16,"kmin":8,"points":26744}` | new | 3744160 | - |
| SIZE_13 | `{"center_size":1300,"chunk_size":36056,"dimensions":172,"kmax":17,"kmin":9,"points":36056}` | new | 6201632 | - |
| SIZE_14 | `{"center_size":1400,"chunk_size":48611,"dimensions":208,"kmax":18,"kmin":9,"points":48611}` | new | 10111088 | - |
| SIZE_15 | `{"center_size":1500,"chunk_size":65536,"dimensions":256,"kmax":19,"kmin":10,"points":65536}` | synthetic-65536 | 16777216 | - |
| SIZE_16 | `{"center_size":1600,"chunk_size":65536,"dimensions":268,"kmax":20,"kmin":10,"points":75281}` | new | 20175308 | - |
| SIZE_17 | `{"center_size":1700,"chunk_size":65536,"dimensions":280,"kmax":21,"kmin":11,"points":86475}` | new | 24213000 | - |
| SIZE_18 | `{"center_size":1800,"chunk_size":65536,"dimensions":292,"kmax":22,"kmin":11,"points":99334}` | new | 29005528 | - |
| SIZE_19 | `{"center_size":1900,"chunk_size":65536,"dimensions":304,"kmax":23,"kmin":12,"points":114105}` | new | 34687920 | - |
| SIZE_20 | `{"center_size":2000,"chunk_size":65536,"dimensions":320,"kmax":24,"kmin":12,"points":131072}` | new | 41943040 | - |

## 5. Validation status

The latest completed validation is recorded in `input_scale_validation.json` and `input_scale_validation_report.md`.

- Structural/data-format status: `pass` with 0 errors.
- Unified configurations: 1420 across 71 enabled programs.
- Generated Rodinia files: 260 files, 1994043397 bytes.
- Effective-parameter duplicate groups: 0.
- Known monotonic exceptions: `PolyBench/C/atax, PolyBench/C/bicg`.
- PolyBench spot compilation: 90/90 passed; executable runs passed=0, environment-unavailable=90.
- Rodinia executable validation: `not_run_environment_unavailable`; see the validation report before formal collection.
