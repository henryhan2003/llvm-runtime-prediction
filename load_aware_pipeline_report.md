# 资源负载感知 Pipeline 扩展报告

初稿日期：2026-10-28；最新修订：2026-10-29。

## 1. 交付状态和本轮边界

已实现三条现有 Pipeline 的可选 CPU/内存后台负载功能、独立实验记录、无压力自检、试验计划和结果验证器。没有重新设计监控系统，没有新建 Prometheus 指标，没有增加主 CSV 列，没有训练预测模型。

按照本轮随后确认的要求“先完成代码与测试，实测由我在服务器执行”，本机只做代码检查、已有数据审计、模拟单元测试和计划生成。原交付状态为 PENDING_SERVER_PILOT。用户随后完成的 strict-v1 CPU pilot 已通过 24/24 配置校验并下载复核，见第 8.3～8.4 节；最新 fast-v2 优化已完成本机测试、尚待服务器新协议试运行，见第 8.5 节。不能把模拟测试、旧无负载结果或安全中止的记录称为合格的新负载性能结果。

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
- `tests/test_resource_load.py`：30 项模拟及纯计算测试。

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
- 监测可用内存、Swap 增量和可读取的温度；默认温度策略为 `abort`，达到配置中的 85℃时失败并清理。显式指定 `--temperature-policy record-only` 后，仅取消脚本启动前和监测线程内的温度中止判断，温度及最高测点仍持续记录，其他保护不变；不修改 BIOS、内核或设备自身保护。没有温度/频率传感器时记录不可用，不填零，也不声称已验证热稳定。
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

`python scripts/check_load_aware_pipeline.py`：123 项测试，122 项通过，1 项 Linux/POSIX 原生 probe 测试在 Windows 跳过，0 失败。包含原有 58 项、负载扩展与修复的 40 项，以及 fast-v2 的 25 项。`unit_test_report.json` 还对 Git 基线中的静态 extract/generate/infer 函数与字段声明做 43 项语法树比较，结果 PASS。模拟 CPU 生命周期并没有启动真实 stress-ng。

已生成 24 个 CPU 试运行 trial 的计划：GEMM（较高算术密度）和 Jacobi-2D（邻域更新、访存与重用特征不同），各 MINI/MEDIUM/EXTRALARGE，乘四档 CPU。压力级别顺序按配置轮换/反转，避免永远先 IDLE 后 HIGH。可选择独立的 memory 家族，仍不把 CPU 和内存压力同时施加在一条 trial 上。

旧无新增负载数据中，GEMM 三档中位时间约为 0.000497、0.010271、6.975786 秒，Jacobi-2D 约为 0.000533、0.014751、14.673943 秒。这些仅用来选择试运行程序和超时，**不是本轮负载实验结果，不能据此给出压力导致的时间变化**。

待服务器验证：实际 CPU 四档、驻留内存变化、Linux 活进程清理、温度/频率可用性、目标时间差异。验证器会重算运行明细中位数、检查至少 3 次有效测量、核对真实采样时间/控制目标/输出哈希和 before→首次调用→during→after→释放顺序；失败不允许作为合格批量前置结果。

### 8.1 2026-10-28 服务器部分试运行补充

用户提供的目录为 `/data_disk/hanli/load_cpu_pilot_20261028_163411`，失败配置为 `cpu__gemm__EXTRALARGE_DATASET__CPU_HIGH`。稳定阶段主机 CPU 平均 75.1685%、最大 75.3100%，说明该阶段达到了预设背景压力。随后在正式测量过程中出现 `Safety abort: temperature cap reached`，日志中的最高温度读数为 85.5℃，超过预设 85℃；sidecar 的 `cleanup_success=True`，stress-ng 日志显示其 64 个 worker 随后被停止。

这支持“温度保护触发”的诊断，不支持将这次目标程序实验记为成功。stress-ng 的 `successful run` 只说明压力工具本身的退出统计，不是完整 trial 的验收结论。目前没有本机可读的完整服务器结果目录，不能汇总此前已完成组的真实时间差异，也不能保证它们已通过最终整体验证。

原温度汇总取所有可读 thermal/hwmon 传感器的最大值，不能直接叫 CPU 核心温度；85℃是本试验的保守配置，不是按 CPU 型号自动识别的厂商极限。旧记录未保存最高测点的身份，不能事后确定 85.5℃来自哪一个传感器。上一次诊断修补补充了 sidecar 的 `temperature_max_source`（设备名、测点标签、sysfs 路径）以及错误消息详情，不改变最高值算法、默认 85℃阈值、负载比例或主 CSV。测试验证最高温度可以来自非 CPU 设备，缺失温度仍为空而非零。该诊断改进与下面的可选策略均需另行同步到服务器；没有重启真实压力。

### 8.2 显式选择温度只记录模式

用户随后要求不因脚本温度阈值而终止。新增 `--temperature-policy abort|record-only`，默认 `abort` 保持旧行为；不静默修改共享配置的默认值。选择 `record-only` 时，配置中保留原来的阈值用于追溯，但不作为启动或运行中的拒绝条件，控制台及每条 trial 的 pipeline.log 显示明确警告。CPU 目标仍为 25/50/75%，内存余量、Swap、持续时间、监控充分性、权限、回收等条件不放宽。持续记录最高温度、对应测点及原有平均 CPU 频率，不能据此声称能准确检测所有形式的硬件降频。

温度策略进入每配置 JSON 并由配置哈希关联，不进入主结果 CSV，也不作为模型输入。分析比较表额外记录 `temperature_policy` 与 `temperature_peak_observed_celsius_ANALYSIS_ONLY`，验证报告记录策略集合和只记录配置数，并明确 `thermal_safety_and_absence_of_throttling_validated=False`。报告 PASS 只代表采集、计时、负载控制和清理检查通过，不证明温度安全或没有降频。后续 batch 要求使用与通过 pilot 相同的温度策略，避免默认沿用另一策略下的前置验收。

用户此次提供的是压力停止后的温度快照：k10temp 的 Tdie/Tctl 43.12℃、CCD 最高 48℃，两个 NVMe Composite 分别 61.85℃和 59.85℃。它不能定位此前 85.5℃测点，也不能证明持续运行安全。AMD 的 [3970X 官方规格](https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen-threadripper/ryzen-threadripper-3000-series/amd-ryzen-threadripper-3970x.html)列出 Tjmax 95℃；该数值不是推荐长期运行温度，不能套用到 NVMe，也不用于自动调高本工具阈值。测点含义参考 [Linux k10temp 文档](https://kernel.org/doc/html/v5.10/hwmon/k10temp.html)。

旧目录和旧 manifest 不重写、不续跑，也不与新策略结果自动拼接。旧 sidecar 缺少策略键时按历史 `abort` 解释；旧 manifest 的代码哈希发生变化仍应报告，不抹掉版本差异。本轮只做本机测试和计划验证，没有执行服务器压力或重新采集时间。

### 8.3 2026-10-29 最终校验的全局状态误判修复

用户提供的 `/data_disk/hanli/load_cpu_pilot_record_only_20261028_173747/validation_report.json` 显示，24/24 配置已通过逐项校验，`cpu_four_levels_validated=True`，温度策略均为 `record-only`，唯一错误为 `Plan-only manifest contains no real load experiment results`。这不是温度、缺样或时间标签错误；`memory_four_levels_validated=False` 也不是本次 CPU-only 试验的失败条件。

根因是启动器只更新每条 trial 的状态，遗漏了顶层 manifest 状态的更新：顶层保持 `PLANNED` 直到首次最终校验，验证器因此追加了计划模式错误；启动器随后把这一错误的最终结果 `FAIL` 写回顶层。修复仅在执行路径增加两次持久化：进入实际执行阶段标记 `RUNNING`，所有 trial 完成、独立校验之前标记 `VALIDATING`，之后仍按验证结果写 `PASS/FAIL`。纯计划模式保持 `PLANNED`；真实运行失败仍中止并保留 `FAIL`。不改动任何采集逻辑、验收阈值或源 CSV，不删除验证器的 plan-only 检查。

已有这批数据不需要重采。应在服务器更新代码前先备份首次失败报告，然后使用原验证器重新检查原 `experiment_manifest.json`。顶层已由原启动器保存为 `FAIL`，不会再触发 `PLANNED` 判断；所有配置、CSV 哈希、时间中位数、阶段、清理与 CPU 控制检查仍完整重做。重新校验只重写派生校验报告和分析比较表，不更改原 manifest、配置、sidecar 或 CSV，也不启动压力或目标程序。不要手动改为 PASS 或删改原源码哈希。如果更新了实验时跟踪的源码，版本差异仍必须如实报错，不能为了重验而替换历史哈希。

新增回归测试覆盖：模拟完整 24-trial 启动器生命周期和最终校验入口的全局状态；真实 trial 非零退出时保持失败且不进入最终校验；使用一致性完整的合成 fixture 复现旧状态误判，并验证独立重验可以恢复 PASS 且原始观测、sidecar、配置及 manifest 字节不变。测试不连接服务器，不产生真实负载；本机没有下载这次完整原始数据，真实重验结果仍待用户在服务器执行确认。

```bash
PILOT_ROOT="/data_disk/hanli/load_cpu_pilot_record_only_20261028_173747"
cp -n "$PILOT_ROOT/validation_report.json" "$PILOT_ROOT/validation_report.initial_fail.json"
python scripts/check_load_experiment.py "$PILOT_ROOT/experiment_manifest.json"
```

### 8.4 2026-10-29 下载后离线分析

已读取 `load_pilot_download/load_cpu_pilot_record_only_20261028_173747`。本次包含 24 个配置，合计 33,483 条运行记录，其中正式测量 33,459 条、预热 24 条，整体执行跨度约 30.29 分钟。这仍然是 24 个程序/规模/背景负载组合，不是 33,459 个独立输入配置。

独立分析脚本 `scripts/analyze_load_pilot_results.py` 不导入或运行任何 Pipeline。533 项离线检查全部通过，包括每配置原始配置哈希、结果 CSV 和明细 CSV 哈希、run_id、重新计算的中位数/均值/CV、阶段边界和真实抓取计数、CPU 控制目标及进程清理。相同程序和输入规模在四档压力下，静态字段（不含产物路径）一致，可执行文件 SHA-256 一致；记录的硬件、编译器和计时参数一致。分析前后全部 310 个源文件的 SHA-256 不变。下载目录没有历史服务器源码快照，因此不以当前本机 checkout 冒充历史源码哈希复核。原 manifest 中保留的 FAIL 是上一节的状态误判历史，服务器独立重验报告为 PASS，本次均不改写。

**负载控制：** IDLE/LOW/MEDIUM/HIGH 的运行前 CPU 均值分别为 0.12%、25.11%、49.73%、75.19%；同档跨六个程序/规模组合的范围分别为 0.10%～0.13%、25.00%～25.25%、49.69%～49.80%、75.16%～75.22%。这些是 Prometheus 实测，不是把配置值写回结果。before/during/after 的每阶段最少独立原始抓取数分别为 12、5、12。阶段仍对应一组重复调用，而不是对每次亚毫秒调用各自独立采样。

下表为相对于相同程序和规模 IDLE 中位时间的变化，负值表示变快，正值表示变慢。没有删除与预期方向不一致的观测。

| 程序 | 输入规模 | IDLE 中位时间/s | LOW 时间变化 | MEDIUM 时间变化 | HIGH 时间变化 |
| --- | --- | --- | --- | --- | --- |
| GEMM | MINI | 0.000664 | -37.8% | -42.6% | -40.0% |
| GEMM | MEDIUM | 0.010314 | -38.8% | -19.1% | -7.6% |
| GEMM | EXTRALARGE | 7.016530 | +21.7% | +39.8% | +53.7% |
| Jacobi-2D | MINI | 0.000703 | -38.1% | -45.3% | -39.9% |
| Jacobi-2D | MEDIUM | 0.014878 | -19.0% | -16.8% | -3.6% |
| Jacobi-2D | EXTRALARGE | 14.797108 | +6.6% | +19.4% | +29.7% |

**频率解释的证据与边界：** 四档运行前主机平均频率约为 2228、3735、3367、4048 MHz，运行中分别约为 2258、3573、3469、4036 MHz。这说明加压时系统频率状态也发生变化，与小任务升频后更快的解释相符，但不能证明因果关系。这里是约 1 Hz 读取各逻辑 CPU 的 `scaling_cur_freq` 后取未加权平均，并非目标任务运行核心的实际频率，也不是任务执行期间按活跃时间加权的频率；中档低于低档还提示不要将这些平均读数简单解释为随负载单调变化。Linux CPUFreq 文档明确说明该接口不一定反映瞬时真实硬件频率。没有采集目标核心位置、迁移、C-state 驻留时间或硬件性能计数器，不能把所有时间变化归因于频率。

当前计时还包含 fork/exec、动态加载及退出采集回调，不是只计算法 kernel。MINI 约 0.38～0.70 ms，启动和系统状态的影响不能忽略。EXTRALARGE 的进程 CPU 时间/墙钟时间中位比约 0.999～1.000，暗示变慢并非主要表现为未获调度而等待；但仍不能区分 SMT、缓存、访存和频率等机制。相同主机 CPU 百分比不等于相同的目标核心竞争强度，本次也只覆盖全逻辑 CPU 上的 `stress-ng sqrt` 计算压力，不代表所有后台负载类型。

**重复性：** MINI 各组有 2773～4300 次测量，组内 CV 为 11.6%～25.0%；MEDIUM 有 319～573 次，组内 CV 为 4.7%～27.5%。EXTRALARGE 每组只有 3 次测量，GEMM CV 为 1.15%～6.18%，Jacobi-2D 为 0.31%～0.77%。每个程序/规模/负载只有一轮独立建立的背景状态，组内大量重复不能代替跨轮重复，也不能据此给出显著性或可靠的跨轮方差。

**温度：** 本批最高观测 85.25℃，发生于 Jacobi-2D EXTRALARGE/HIGH 的 during 阶段，最高测点为 `k10temp/Tccd5`，对应一次不低于 85℃的安全采样。不是 NVMe 测点，也不等于整颗 CPU 的统一温度。在 `record-only` 策略下没有因此中断。当前约 1 Hz 的主机频率均值不能证明没有热降频，采集 PASS 也不证明散热安全。

**结论：** 负载施加、现有三阶段监控及结果关联已跑通，本批数据可保留为有效 pilot，不应为了获得单调变慢而删除小规模观测。正式扩充前，应明确研究自然调频系统中的运行时间，还是单独识别资源竞争效应；前者需要正确使用运行前状态并保留频率/放置条件的限制，后者需要经许可控制或分层分析这些因素。建议先少量交替复测独立负载轮次，再扩到三个数据集、20 档，避免将当前一次 pilot 当作已充分验证的通用负载模型。本轮没有修改 Pipeline、系统设置、原始 CSV、模型或 folds，没有启动真实负载。

交付：`outputs/load_cpu_pilot_analysis_20261028_173747_verified/pilot_analysis_report.md`、`trial_analysis.csv`、`analysis_audit.json`。新增离线分析的 5 项单元测试全部通过。复现时使用新的输出目录：

```powershell
python scripts/analyze_load_pilot_results.py --input-dir load_pilot_download/load_cpu_pilot_record_only_20261028_173747 --output-dir outputs/load_cpu_pilot_analysis_reproduced
python -m unittest tests.test_analyze_load_pilot_results -v
```

### 8.5 2026-10-29 fast-v2：复用准备工作与限制测量次数

本轮先优化当前计划中的 PolyBench/C CPU serial，不把通用缓存未经审查地套用到 Rodinia Makefile 或 CUDA host/device 编译。`--protocol strict-v1` 仍是启动器默认值；显式选择 `--protocol fast-v2` 才启用新策略，OpenMP/CUDA 选择 fast-v2 会明确拒绝。三套原始 CSV 和现有建模结果均未修改。

| 项目 | 原 strict-v1 | 新 fast-v2 默认 |
| --- | --- | --- |
| AST/CFG/IR 与可执行文件 | 每条负载 trial 重复准备 | 每个程序/输入配置准备一次，四档压力共享 |
| 正式测量 | 5 秒预算，至少 3 次，无次数上限 | 3 秒预算，至少 2 次，最多 10 次；达到预算且满足最少次数，或触及次数上限即停止 |
| 预热 | 1 次 | 保留 1 次，仍在背景负载下执行 |
| before / after | 各 12 秒 | 各 6 秒 |
| 稳定窗口 | 10 秒 | 6 秒，仍须连续 3 次检查通过 |
| 独立抓取最低数量 | 每阶段至少 5 个 | before/after 及可观测 during 至少 3 个 |
| 极短 during 的处理 | 不足采样即失败 | 仅允许明确分类的短窗口缺失，不填零，不用空等或额外执行凑采样 |

没有额外规定一个“长程序名单”。例如一次执行 10 秒时，正式测量完成 2 次即超过 3 秒预算并停止；一次执行 1 ms 时，最多执行 10 次，不会继续重复到几千次。中等耗时按相同预算与上限自然落在 2～10 次之间。`runtime_sec_median` 仍由成功测量调用的原始时间计算；只有 2 次时，中位数就是两个时间的算术平均，抗异常和置信度会下降，不能当成与旧版大量重复具有相同精度。预热、启动/稳定等待和清理继续不计入单次 runtime。

**准备缓存：** 新模块 `scripts/polybench_preparation_cache.py` 只接收准备阶段的 AST/CFG/IR 特征和产物路径，不允许保存 runtime、run_success、进程资源或监控观测。缓存键包括程序身份、规模宏、项目源码/头文件、PolyBench utilities、兼容头文件、优化等级、工具路径/版本/文件状态、相关环境变量与提取代码哈希。缓存局限于单次实验目录 `preparation_cache/`，使用固定的系统头文件/链接库环境；不支持携带任意自定义编译参数复用缓存。系统库更新不属于本轮受控实验，应停止并创建新实验批次，而不是继续沿用缓存。

首次准备在后台负载启动之前完成，成功后发布收据和产物哈希；失败或不完整产物不发布。命中时验证收据和全部产物哈希，损坏即明确失败，不静默重建。每条 trial 的 sidecar 保存缓存 key、收据路径/哈希、可执行文件哈希及命中情况。运行时间和动态资源每条 trial 都重新测量。AST/IR/CFG/可执行文件只保留一份共享副本，不复制到四个 trial 目录，也不把缓存参数加入主 CSV 或预测白名单。保存/下载结果时要保留整批目录中的 `preparation_cache/`，不能只复制单个 trial 文件夹。

对 30 程序×20 规模×4 CPU 压力，计划仍为 2400 条样本，但只有 600 份准备产物，预期 1800 次缓存命中。服务器最终验证还检查同一程序/输入在不同负载条件下使用相同缓存 key，避免中途改源码后把不同程序产物当作同一控制条件。

**短窗口的语义边界：** CPU 与内存的实际 scrape interval 仍从 Prometheus 读取，不能通过减小 query step 冒充更多独立抓取。默认按服务器现有 1 秒间隔设计；若间隔更长而 context 不足，仍拒绝运行，需显式增大窗口。只在 fast-v2、仅 during 阶段、真实窗口短于 `(min_raw_scrapes+1)×interval` 且确实不足独立抓取时，sidecar 可记录 `MISSING_SHORT_WINDOW`。对应原 CSV 的 during 数值保持空，`prometheus_during_collection_success=False`、`prometheus_three_phase_collection_success=False`。before/after、控制目标、进程资源采集和清理仍严格检查；网络请求失败、断连、长窗口缺样不属于这个例外，仍然失败。

新协议的 trial 可以在这一预先声明的缺失规则下通过，但不叫“三阶段全部采齐”。最终报告明确输出 `complete_three_phase_configurations` 与 `short_during_missing_configurations`，并记录 `collection_protocols`、缓存命中/准备份数。新旧协议不静默混合；fast-v2 全量执行必须先提供同机、同温度策略、同为 fast-v2 的通过报告，不能拿旧 strict-v1 pilot 绕过验收。

后续审计或建模不能照搬“三阶段全成功才保留样本”的旧筛选规则，否则会系统性剔除 fast-v2 的短任务。应区分运行成功、运行前状态可用和运行中辅助信息缺失，并按协议审查；本轮没有擅自修改任何模型训练脚本或特征白名单。

**耗时预期：** 1 秒抓取间隔且首次稳定就达标时，恢复等待约从 12 秒变为 8 秒，稳定过程的名义初始等待及检查约从 16 秒变为 12 秒，before/after 从合计 24 秒变为 12 秒。每配置名义固定等待约省 20 秒，2400 组对应约 13.3 小时；这不是实测承诺，还不含缓存省下的准备时间和减少重复调用的收益，亦不能用来扣掉真实超时、重新稳定或安全中止。整体运行仍可能是一天以上，必须以新协议 pilot 的实测重新估计，不承诺几个小时完成。

**验证与交付：** 123 项管线/负载测试中 122 项通过、1 项 Windows 下跳过，43 项原静态函数与 CSV 字段声明比较通过；另有 6 项离线分析测试通过。已生成不执行负载的 24-trial 新协议 pilot 计划及 2400-trial 全量计划，检查其准备份数分别为 6 和 600。旧 24 组数据重新离线核验仍通过，310 个原文件未变。新增测试覆盖缓存命中与失效、腐坏/失败不误用、不缓存运行观测、测量上下限、短窗口不填零不凑时长、真正监控故障不豁免、两次中位数、旧模式兼容、原全局状态流转修复。尚未在 Linux 实测 fast-v2，不启动任何服务器压力或模型训练。

主要文件：`scripts/pipeline_new.py`、`scripts/resource_load.py`、`scripts/run_load_experiments.py`、`scripts/check_load_experiment.py`、`scripts/check_load_aware_pipeline.py`、新缓存模块和 `tests/test_load_fast_protocol.py`。离线分析器及其测试也已适配共享产物路径和新协议的明确缺失状态。`configs/resource_load_profiles.json` 未改默认值，快速参数由显式新协议写入每批独立配置。

同步后在 tmux 中手动激活环境并设置 stress-ng 的 PATH，再运行新协议 pilot：

```bash
python scripts/check_load_aware_pipeline.py
python scripts/run_load_experiments.py --dataset polybench --stage pilot \
  --protocol fast-v2 --load-types cpu --temperature-policy record-only \
  --execute --permission-confirmed \
  --output-dir "/data_disk/hanli/load_cpu_pilot_fast_v2_$(date +%Y%m%d_%H%M%S)"
```

## 9. 服务器执行指令

代码先按项目既有 Git 流程同步到服务器。在 tmux 中手动激活已有 conda 环境，脚本内部没有 conda activate。

```bash
cd /data_disk/hanli/llvm-runtime-prediction
conda activate llvm-runtime
export PATH="$PATH:$HOME/.local/conda-envs/load-tools/bin"
command -v stress-ng
stress-ng --version
python -m pip install -r requirements.txt
python scripts/check_load_aware_pipeline.py
python scripts/run_load_experiments.py --preflight-only
```

这里只新增需要 stress-ng；本服务器已使用独立 Conda 前缀安装，上面的 PATH 必须在实际执行实验的 tmux pane 中设置，无需 sudo。Python 新模块均使用标准库，不新增机器学习依赖。Prometheus、node_exporter、CUDA 的 nvidia-smi 沿用已有部署。

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
