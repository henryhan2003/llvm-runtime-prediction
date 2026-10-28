# 资源负载感知 Pipeline 扩展报告

日期：2026-10-28。

## 1. 交付状态和本轮边界

已实现三条现有 Pipeline 的可选 CPU/内存后台负载功能、独立实验记录、无压力自检、试验计划和结果验证器。没有重新设计监控系统，没有新建 Prometheus 指标，没有增加主 CSV 列，没有训练预测模型。

按照本轮随后确认的要求“先完成代码与测试，实测由我在服务器执行”，本机只做代码检查、已有数据审计、模拟单元测试和计划生成。**真实 Linux 负载试运行尚未执行，状态为 PENDING_SERVER_PILOT，不能把模拟测试或已有无负载结果称为新负载实验结果。** CPU 负载能否达到设定水平、不同压力下真实时间差异，以及 Linux 实际进程回收效果，均等待服务器试运行验收。

## 2. 原有 Pipeline 与数据审计

审计文件为 `outputs/load_aware_pipeline_validation/existing_data_audit.json`，包含原文件 SHA-256、全部字段、阶段窗口和采样统计。三套数据分别审查，不混合版本。

| 数据 | 采用的已有结果 | 行数 | 程序数 | 所选文件列数 | 窗口顺序异常 |
| --- | --- | ---: | ---: | ---: | ---: |
| PolyBench/C | `polybench_full_20260812_181722/results_20260812_181722.csv` | 150 | 30 | 252 | 0 |
| Rodinia OpenMP | `outputs/rodinia_openmp_cpu_runtime_repaired_v3_v4_20261027/rodinia_openmp_summary_repaired_v3_v4.csv` | 380 | 19 | 261 | 0 |
| Rodinia CUDA | `cuda_full_download/rodinia_cuda_full_20260814_155908/rodinia_cuda_summary_20260814_155916.csv` | 45 | 22 | 352 | 0 |

OpenMP 采用已确认的 v3/v4 修复派生表，261 列中含修复与来源信息；这里的列数不是宣布三条 Pipeline 当前应输出相同数量的字段。本次修改没有往任何主结果表增加列。

三套已有表分别有 150、380、45 行报告 Prometheus 三阶段成功。历史 before/after 查询点数均为 6，窗口约 5 秒；during 查询点数分别为 11～832、4～122、11～43。**这些是原代码汇总的查询点数，不足以证明真实抓取次数或实际抓取间隔。** 本机没有访问服务器确认其当前配置；新控制器在实际执行时读取 `/api/v1/targets` 中的 `scrapeInterval`，并从 `timestamp(node_cpu_seconds_total)` 和 `timestamp(node_memory_MemAvailable_bytes)` 的返回值核对去重后的真实抓取时间。

### 2.1 原来的时间窗口

PolyBench 的 `run_program` 和 Rodinia 共用的 `measure_input` 原来均先完成预热，然后记录 before 窗口；during 包围本配置的一组重复测量；after 在测量组结束之后等待并记录。CUDA 共用 Rodinia 的测量函数，另外由已有 `NvidiaSmiSampler` 持续采样，并按相同阶段边界汇总。

因此三阶段可以关联到**同一程序、同一输入配置的一组重复测量**，但不能宣称是每一次毫秒级原生调用都有独立的三阶段监测。PolyBench 在一轮所有配置结束后查询已记录的历史时间窗；OpenMP/CUDA 通常在该配置结束并等待查询延迟后查询。后查询不意味着窗口在查询发生时采集。

旧 before 数据在正式测量之前，但已发生过预热，不是“目标程序从未启动过”的纯背景快照。负载模式下调整 before 的位置，详见第 4 节；未启用负载时保持旧时序。

### 2.2 原指标直接复用

| 含义 | 已有字段/来源 | 本次处理 |
| --- | --- | --- |
| 三阶段主机 CPU | `host_cpu_usage_pct_before_mean/max`、`during_mean/max`、`after_mean/max` | 原名、单位、CPU 差分计算原样复用 |
| 三阶段内存占用 | `host_memory_used_bytes_before_mean/max` 等 | 原名复用；Linux 原查询就是 MemTotal−MemAvailable |
| 总内存 | `memory_total_bytes` | 原样保留；可用内存可在确认同一主机后以总内存减已有占用还原，不另建同义 CSV 列 |
| 磁盘、网络 | `host_disk_read_bytes_*_delta`、`host_disk_write_bytes_*_delta`、`host_network_bytes_*_delta` | 原样复用 |
| 阶段边界与质量 | `prometheus_*_window_start/end_timestamp`、`*_sample_count`、`*_collection_success/error` | 原样复用；不足采样时利用已有失败字段和空值 |
| GPU 利用率/显存 | `gpu_utilization_pct_before/during/after_mean/max`、`gpu_memory_used_bytes_*` | 保持原有 nvidia-smi 采样，不另建 GPU Prometheus exporter |
| GPU 温度/频率/功耗 | `gpu_temperature_celsius_*`、`gpu_sm_clock_mhz_*`、`gpu_power_draw_watts_*` 等 | 原样复用 |
| GPU 关联 | `gpu_uuid`、`gpu_physical_index`、`gpu_*_window_*` | 原样复用；保留原有忙卡检查 |
| 时间标签 | `runtime_sec_median/mean/std` 及原有 CV、min/max 字段 | 不改变统计口径 |
| 进程消耗 | 现有 `process_*` 字段 | 仍由目标程序的原生 probe 采集，不把压力进程当作目标子进程统计 |
| 样本定位 | `program_id/input_id/env_id`、Rodinia 的 `session_id/collected_at`、运行明细 `run_id` | 直接复用，配合独立输出目录和外部 manifest 关联 |

GPU 审计结论：当前 CUDA 的 GPU 三阶段是 nvidia-smi，不是 Prometheus；主机 CPU/内存三阶段是 Prometheus。不能将其描述成已有统一 GPU Prometheus 监控系统。

## 3. 改动文件

修改：

- `scripts/pipeline_new.py`：可选负载入口、负载模式时序、真实采样质量回调、独立输出目录、异常时回收目标进程。
- `scripts/pipeline_rodinia.py`：同一控制器接入共享测量函数，供 OpenMP/CUDA 使用。
- `scripts/pipeline_rodinia_cuda.py`：接入共享控制器，保留已有 nvidia-smi 采样与 GPU 安全检查。

新增：

- `scripts/resource_load.py`：配置校验、Linux/节点检查、CPU 压力命令、容量预算、稳定判定、安全监护、拥有者限定的进程回收和 JSON 记录。
- `scripts/resource_load_memory_worker.py`：真实触碰匿名内存的容量型压力进程。
- `scripts/resource_load_supervisor.py`：监护 Pipeline 父进程的存活与压力期限，异常时终止自己的压力进程组。
- `scripts/run_load_experiments.py`：默认只生成计划；显式执行时调度小规模或经过试运行门禁的 20 档实验。
- `scripts/check_load_experiment.py`：结果、真实抓取时间、运行明细、CPU 实测目标、阶段与负载时序的核对，以及 IDLE 对照输出。
- `scripts/audit_load_aware_pipeline.py`：只读历史数据审计。
- `scripts/check_load_aware_pipeline.py`：无压力回归测试及静态提取/CSV 声明的语法树比对。
- `configs/resource_load_profiles.json`：CPU 四档、内存容量四档、安全阈值、默认试运行程序。
- `tests/test_resource_load.py`：28 项模拟及纯计算测试。

静态特征的算法、白名单、计数方式和编译参数没有修改。仅在负载模式下将表示文件写到本次结果目录内的 `artifacts/`；这是输出位置隔离，避免覆盖以前的 AST/CFG/IR。Rodinia 的运行二进制仍按原 Makefile 在其工作目录构建，既有 build 行为未重写。不要由静态 IR 的 `--opt-level O2` 推断全部 Rodinia 运行 Makefile 已统一 O2。

## 4. 采集时序

启用 `--load-config` 时：

1. 原样完成静态提取与编译，尚未启动压力，不把编译负载混入测量。
2. 检查 Linux、显式使用许可、本机唯一 node exporter、真实抓取间隔和本机内存容量；获取机器级试验互斥锁。
3. 等待无新增压力的背景恢复窗口，测量真实基线；主机已忙、可用内存不足或温度超限则拒绝启动压力。
4. 启动本次自有压力进程或保持 IDLE。等待覆盖足够真实抓取的稳定窗口，并要求三次连续检查通过。
5. **before 窗口：负载已经稳定，但目标程序尚未首次启动，包含预热也未开始。**
6. 原有预热次数，仍在后台负载持续存在的条件下执行；预热耗时不进入正式运行时间标签。
7. 原有重复测量循环；during 仍只包围正式重复测量组。PolyBench 原有墙钟预算与 Rodinia 原有成功调用耗时累计预算保持各自口径，不趁本任务统一二者。
8. after 窗口：正式测量结束，但后台压力继续存在。此窗口仍表达目标程序结束之后的背景状态，没有改成“释放压力之后”。
9. 回收本次压力进程，记录资源释放后的本地安全快照；释放阶段不属于已有 after 字段。
10. 沿用原查询接口汇总三个已记录的时间窗，并补做真实抓取充分性检查，关联原结果文件、行身份和 run_id。

压力启动、背景恢复、稳定等待、before/after 等待和回收均不进入 `runtime_sec_median` 等单次调用时间。温度安全监护是额外的低频安全线程，其观测只进入日志，不进入主 CSV。

未提供 `--load-config` 时，不启动控制器、不加压力、不增加恢复/稳定等待；原来的“预热→before→测量→after”路径保留。

## 5. 负载定义与安全限制

### 5.1 CPU

IDLE 表示不额外施压，不承诺主机实测 CPU 等于零。LOW/MEDIUM/HIGH 的 before 主机目标分别为 25%、50%、75%。压力使用当前可用逻辑处理器数量的 stress-ng CPU worker，保持原有目标程序线程数、CUDA 配置和亲和性不变。初始 worker duty 根据基线和 CPU 可用范围估计，最终接受与否只根据已有 Prometheus 实测指标。

默认容差为 8 个百分点，压力设定硬上限为 75%；三档不能用设定数字冒充实测值。CPU affinity 或 quota 不足以安全达到主机级目标时拒绝，不能把容器内部比例偷偷当全机比例。主机初始 CPU 使用率超过 15% 时拒绝添加压力。

**目标是预先存在的背景压力，不是把运行中“目标程序＋压力”的总 CPU 强行限制为 25/50/75%。** 目标程序运行后主机总 CPU 升高是正常现象，压力进程不因目标程序开始而退出，也不根据目标程序的 CPU 用量反向调小。

### 5.2 内存容量

初版有意使用驻留容量负载，而不是 stress-ng vm 的持续带宽压力。先按页真实写入整个匿名映射，再每秒对少量分散页进行稀疏访问，保持进程及映射存活；这不是“只保留地址空间而不触碰页面”。初始化触碰发生在稳定完成之前。稀疏访问仍有少量 CPU/内存流量，因此不能声称绝对零带宽干扰。

安全预算取以下三者的最小值：明确的最大允许分配（默认 8 GiB）、有效容量的 25%、当前可用内存减安全余量。安全余量取 2 GiB 和有效容量 20% 的较大值；同时考虑 cgroup 内存限制。LOW/MEDIUM/HIGH 分别占安全预算的 25%/50%/75%，**不是宣称主机总内存占用为这三个百分比**。预算、字节分配和实测主机内存占用均在 sidecar 中记录。

例如大型服务器预算被 8 GiB 上限限制时，三档新增占用约 2/4/6 GiB。这是保守默认值，可能不会使目标程序明显变慢。需要提高时必须在获得许可后显式设置 `--max-memory-gib`，并先重做内存试运行。容量占用不等于带宽竞争；若剩余内存充足，执行时间不变也可能是正常结果。

### 5.3 保护措施

- 默认只生成计划；实际运行必须同时给出 `--execute --permission-confirmed`，程序不代替人获取共享服务器使用许可。
- 互斥锁防止本工具在同机叠加多个实验；不能据此断言其他用户没有任务，仍需人工确认与实测基线检查。
- 监测可用内存、Swap 增量和可读取的温度；超过限制会失败并清理。没有温度/频率传感器时记录不可用，不填零，也不声称已验证热稳定。
- 默认每配置压力最长 1200 秒，配置硬上限 3600 秒；每次目标调用仍受原有 timeout 控制。完整 trial 另有总超时。
- SIGINT/SIGTERM/SIGHUP、Python 异常、提前退出均走清理；额外 supervisor 检测 Pipeline 父进程消失并收回本次进程组。只操作本次创建的组，不使用 `pkill stress-ng` 等按名称清理。
- 目标进程在用户中断时也会被回收，修补了原计时包装在 KeyboardInterrupt 下可能留下独立目标进程的路径。
- 记录清理后仍存在的自有 session 进程，残留时不判定成功。内核不可中断进程、整机重启等仍不能由用户态程序保证即时回收，必须如实查看日志。
- 调度器发现结果盘剩余不足 1 GiB 时中止后续 trial，不自动删除任何文件。

## 6. 指标角色和关联方式

主 CSV 新增字段数：**0**。没有 `pre_cpu_util` 等别名，没有将 `load_profile_id` 或目标比例塞入现有模型输入。

外部 `experiment_manifest.json` 保存 trial、数据集、程序、规模、负载家族、配置路径、完整 Pipeline 命令、源码/配置哈希和执行状态。每配置的 `results/load_records/<uuid>.json` 保存实际控制命令、PID/进程组、稳定与阶段时间、原有字段形式的实测值、真实抓取时间戳、清理结果、输出 CSV 路径、原行身份以及 run_id 范围/数量/摘要。

PolyBench 没有 session_id 列，运行明细已有唯一 run_id；使用“独立输出 CSV 路径＋program_id/input_id/env_id＋原阶段起止时间＋run_id 摘要”足以可靠关联。Rodinia 额外复用 session_id。**不为区分压力而篡改 env_id、program_id 或 input_id。** 这些程序/规模在不同负载下会重复出现，未来建模不能直接使用旧的仅按 program+input 去重代码；应先关联外部 trial 与运行前实测状态，再定义样本身份。

可作为未来运行前特征候选的是 `host_cpu_usage_pct_before_*`、`host_memory_used_bytes_before_*`，以及适用的 GPU before 状态；仍需单独进行语义审核。本轮不自动修改模型白名单。during/after、`prometheus_background_*` 的 before/after 平均、`*_background_adjusted_*`、实测目标进程资源和时间统计均不能直接用于同次任务的运行前预测。控制参数、质量状态、JSON 标识默认只是实验元数据。

## 7. 采样充分性

对负载模式读取真实 scrape interval，要求 before/after context 至少为 `(min_raw_scrapes+1)×interval`。默认试运行使用 12 秒 context、至少 5 个真实抓取。稳定窗口最少为 10 秒且随 interval 放大。小于需要的窗口直接报错，不能通过缩小 query step 伪造更多独立抓取。

`timestamp(raw_metric)` 返回值被去重，并要求时间戳确实落在对应窗口内且足够新。lookback 延用窗口之前的值不算本窗口观测。短 during 窗口不足时，复用既有 `prometheus_during_collection_success=False` 和 error 字段，数值留空，不用 0 或邻近阶段填充。试运行默认 5 秒测量预算并至少 3 次成功测量，目标是让“重复测量组”的 during 有足够抓取；不声称每次亚毫秒程序调用的动态状态可单独解析。

## 8. 本机验证结果与待验收项

`python scripts/check_load_aware_pipeline.py`：86 项测试，85 项通过，1 项 Linux/POSIX 原生 probe 测试在 Windows 跳过，0 失败。包含原有 58 项与新增 28 项。`unit_test_report.json` 还对 Git 基线中的静态 extract/generate/infer 函数与字段声明做语法树比较，结果 PASS。模拟 CPU 生命周期并没有启动真实 stress-ng。

已生成 24 个 CPU 试运行 trial 的计划：GEMM（较高算术密度）和 Jacobi-2D（邻域更新、访存与重用特征不同），各 MINI/MEDIUM/EXTRALARGE，乘四档 CPU。压力级别顺序按配置轮换/反转，避免永远先 IDLE 后 HIGH。可选择独立的 memory 家族，仍不把 CPU 和内存压力同时施加在一条 trial 上。

旧无新增负载数据中，GEMM 三档中位时间约为 0.000497、0.010271、6.975786 秒，Jacobi-2D 约为 0.000533、0.014751、14.673943 秒。这些仅用来选择试运行程序和超时，**不是本轮负载实验结果，不能据此给出压力导致的时间变化**。

待服务器验证：实际 CPU 四档、驻留内存变化、Linux 活进程清理、温度/频率可用性、目标时间差异。验证器会重算运行明细中位数、检查至少 3 次有效测量、核对真实采样时间/控制目标/输出哈希和 before→首次调用→during→after→释放顺序；失败不允许作为合格批量前置结果。

## 9. 服务器执行指令

代码先按项目既有 Git 流程同步到服务器。在 tmux 中手动激活已有 conda 环境，脚本内部没有 conda activate。

```bash
cd /data_disk/hanli/llvm-runtime-prediction
conda activate llvm-runtime
sudo apt-get update
sudo apt-get install -y stress-ng
python -m pip install -r requirements.txt
python scripts/check_load_aware_pipeline.py
python scripts/run_load_experiments.py --preflight-only
```

这里只新增需要 stress-ng；Python 新模块均使用标准库，不新增机器学习依赖。Prometheus、node_exporter、CUDA 的 nvidia-smi 沿用已有部署。安装命令由使用者在获得权限后执行。

只生成计划，不启动压力：

```bash
python scripts/run_load_experiments.py --dataset polybench --stage pilot
```

获准后运行 CPU 试验；输出目录必须不存在：

```bash
python scripts/run_load_experiments.py --dataset polybench --stage pilot \
  --load-types cpu --execute --permission-confirmed \
  --output-dir /data_disk/hanli/load_cpu_pilot_$(date +%Y%m%d_%H%M%S)
```

CPU 与内存分开施加、放在同一份试运行清单（48 条 trial）：

```bash
python scripts/run_load_experiments.py --dataset polybench --stage pilot \
  --load-types cpu memory --max-memory-gib 8 --execute --permission-confirmed \
  --output-dir /data_disk/hanli/load_cpu_memory_pilot_$(date +%Y%m%d_%H%M%S)
```

如果节点实际 interval 比当前 context 所能支持的更长，先按报错增加 `--context-seconds`，不要将 query step 当 scrape interval。出现超时、繁忙主机、温度或采样错误，查看对应 trial 的 `pipeline.log` 和 `results/load_records/`；本轮不自动重试失败样本以隐藏问题。

单独复核已完成试验：

```bash
python scripts/check_load_experiment.py /data_disk/hanli/实际试验目录/experiment_manifest.json
```

成功后自动生成 `validation_report.json` 和 `load_experiment_comparisons.csv`，后者复用原指标名并附分析用途的 IDLE 时间比。观察差异不等同于显著性或因果证明，单配置内的重复调用也不等于独立日期重复实验。

## 10. 三套数据的 20 档后续批量入口

原统一规模清单有 PolyBench 600、OpenMP 380、CUDA 440 条程序+规模配置。每加入四档 CPU 就分别成为 2400、1520、1760 条 trial。目录和计划功能已支持全部 20 档，但实际批量执行要求提供同机真实试运行 PASS 报告；CPU 批量要求四档 CPU 都验证，内存批量还要求内存四档验证。

```bash
python scripts/run_load_experiments.py --dataset openmp --stage batch \
  --load-types cpu --execute --permission-confirmed \
  --pilot-validation /data_disk/hanli/已通过试运行目录/validation_report.json \
  --output-dir /data_disk/hanli/load_openmp_20_$(date +%Y%m%d_%H%M%S)
```

将 dataset 改为 polybench 或 cuda 可使用相应现有 Pipeline；CUDA 可继续指定 `--cuda-arch`、`--gpu-index`，不会改变 CUDA block/thread 配置，也不会启用 GPU 后台计算压力。原输入文件必须事先生成，运行参数仍由已有 manifest 校验。

GPU 后续扩展的已有基础是独立 sampler、UUID/物理编号与三阶段汇总。若将来新增 GPU 压力后端，仍需明确显存/功率安全预算、共享设备许可、忙卡策略和专用 kernel；本轮没有实现或假装验证 GPU 压力生成。

## 依据

- Prometheus HTTP API：<https://prometheus.io/docs/prometheus/latest/querying/api/>。
- Prometheus 查询点与 lookback：<https://prometheus.io/docs/prometheus/latest/querying/basics/>。
- stress-ng 官方说明：<https://colinianking.github.io/stress-ng/>。

这些文档用于核实接口和控制参数，不是本项目已经完成实机负载实验的证据。
