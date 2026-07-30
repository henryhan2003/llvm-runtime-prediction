# Prometheus 监控数据集代码运行时指标流程

## 1. 明确监控目标

先把“运行时各类指标”分成几类，避免后面采集链路混在一起：

- 程序级指标：运行时间、退出码、执行次数、数据集名称、benchmark 名称、编译配置、输入规模。
- 进程级指标：CPU 使用率、内存 RSS/峰值内存、线程数、上下文切换、缺页次数、I/O 读写量。
- 系统级指标：整机 CPU、内存、磁盘、网络、负载、温度、NUMA、文件系统空间。
- 硬件性能计数器：cycles、instructions、cache-misses、branch-misses、IPC 等。
- GPU 指标：GPU 利用率、显存占用、功耗、温度、kernel 活跃情况。

PolyBenchC 这类 benchmark 通常运行时间较短，所以不要只依赖 Prometheus 定时 scrape；应当把每次运行的结果作为一次 batch job 结果写入 Pushgateway 或 node_exporter textfile collector。

## 2. 设计整体采集架构

推荐采用不侵入源码的外部采集架构：

1. Benchmark runner 负责按清单执行数据集程序，例如 `datasets/PolyBenchC-4.2.1` 下的各个 kernel。
2. 程序运行期间，由系统 exporter 持续暴露机器和进程状态。
3. 每次运行结束后，runner 汇总本次运行结果。
4. runner 将一次性结果写入 Pushgateway，或者写成 `.prom` 文件交给 node_exporter textfile collector。
5. Prometheus 周期性抓取 node_exporter、process exporter、Pushgateway、GPU exporter 等目标。
6. Grafana 读取 Prometheus，按 benchmark、dataset、compiler、problem_size、run_id 等维度展示趋势和对比。

逻辑链路：

```text
PolyBenchC binary
  -> wrapper/runner
  -> time/perf/usr_time/nvidia-smi/process snapshot
  -> Pushgateway 或 textfile collector
  -> Prometheus
  -> Grafana
```

## 3. 准备 Prometheus 侧组件

按指标类型选择 exporter：

- node_exporter：采集主机 CPU、内存、磁盘、网络等基础指标。
- process-exporter：按进程名或命令行匹配 benchmark 进程，采集进程级 CPU、内存、线程数等。
- Pushgateway：保存短生命周期 batch job 的运行结果，例如单次 benchmark 的耗时、退出码、峰值内存。
- node_exporter textfile collector：适合 runner 在本地落盘 `.prom` 指标文件。
- dcgm-exporter 或 nvidia_gpu_exporter：如果监控 CUDA/GPU benchmark，用于采集 GPU 利用率、显存、功耗、温度。
- perf_exporter 或自定义 textfile：如果需要硬件计数器，用 `perf stat` 或 PAPI 输出后转换为 Prometheus 指标。

Prometheus scrape 目标至少包含：

- 主机 exporter。
- batch 结果入口：Pushgateway 或 textfile collector 所在的 node_exporter。
- 进程 exporter。
- GPU exporter，如果有 GPU 程序。

## 4. 规范 benchmark 元数据

每次运行都要固定一组标签，后续查询和画图才不会混乱：

- `dataset`：例如 `polybenchc`、`rodinia`。
- `benchmark`：例如 `gemm`、`2mm`、`jacobi-2d`。
- `category`：例如 `linear-algebra`、`stencils`、`datamining`。
- `compiler`：例如 `clang`、`gcc`。
- `opt_level`：例如 `O0`、`O2`、`O3`。
- `problem_size`：例如 `MINI`、`SMALL`、`STANDARD`、`LARGE`、`EXTRALARGE`。
- `run_group`：一次实验批次的稳定编号。
- `host`：运行机器。

避免把时间戳、随机 UUID、完整命令行、完整文件路径作为高基数标签。若需要保留这些信息，放到日志文件或实验记录表里。

## 5. 编译数据集程序

不修改源码，只通过现有编译参数打开需要的运行输出：

1. 确认 PolyBenchC 的 benchmark 源码路径和 `utilities/polybench.c`。
2. 按实验设计选择编译器、优化等级和 problem size。
3. 如果只需要程序自带耗时，使用 PolyBenchC 支持的计时宏。
4. 如果需要硬件计数器，不在源码中加采集逻辑，而是在运行时外层套 `perf stat`、PAPI 工具或系统 profiler。
5. 每个可执行文件输出到独立目录，目录名包含 dataset、benchmark、compiler、opt_level、problem_size。

编译阶段本身也应记录：

- 编译命令。
- 编译器版本。
- LLVM/Clang 版本。
- 目标平台。
- 是否开启 OpenMP/CUDA 等并行选项。

## 6. 设计 runner 执行流程

runner 不改 benchmark 源码，只负责外部调度和采集：

1. 读取 benchmark 清单。
2. 为本次实验生成 `run_group`。
3. 对每个 benchmark 做固定次数预热。
4. 对每个 benchmark 做 N 次正式运行。
5. 每次运行前记录环境快照。
6. 启动外部采集器或标记采集窗口。
7. 执行 benchmark。
8. 捕获退出码、stdout、stderr。
9. 收集 `/usr/bin/time -v`、`perf stat`、进程快照、GPU 采样等结果。
10. 将结果转换为 Prometheus 指标。
11. 推送到 Pushgateway，或原子写入 textfile collector 目录。
12. 保存原始日志，方便复核。

短程序建议每个 benchmark 重复运行多次，并记录：

- `benchmark_run_seconds`：单次耗时。
- `benchmark_run_exit_code`：退出码。
- `benchmark_run_success`：是否成功。
- `benchmark_run_repetitions_total`：重复次数。
- `benchmark_run_mean_seconds`：均值。
- `benchmark_run_median_seconds`：中位数。
- `benchmark_run_stddev_seconds`：标准差。

## 7. 采集程序级指标

程序级指标优先来自外层 runner 和 PolyBenchC 已有计时能力：

1. 使用 PolyBenchC 自带计时输出获取 kernel 执行时间。
2. 使用外层 wall-clock 计时获取端到端运行时间。
3. 对比 kernel time 和 wall time，区分初始化、I/O、运行时开销。
4. 保存 benchmark 的 stdout/stderr，避免只留下聚合指标。
5. 对异常退出单独记录 exit code 和错误摘要。

建议指标名：

- `benchmark_kernel_seconds`
- `benchmark_wall_seconds`
- `benchmark_exit_code`
- `benchmark_success`
- `benchmark_iterations`

## 8. 采集进程级指标

进程级指标适合用 process-exporter 和外部采样结合：

1. runner 启动 benchmark 后记录 PID。
2. process-exporter 按进程名或命令行匹配运行中的 benchmark。
3. Prometheus scrape 间隔设置得比 benchmark 运行时间更短。
4. 对很短的程序，使用 runner 在运行期间主动采样 `/proc/<pid>` 或平台等价接口。
5. 运行结束后把峰值、均值、最后值写入 Pushgateway/textfile。

重点指标：

- CPU user/system time。
- 最大 RSS。
- 虚拟内存。
- minor/major page faults。
- voluntary/involuntary context switches。
- read/write bytes。
- 线程数。

## 9. 采集系统级指标

系统级指标通过 node_exporter 长期采集：

1. 实验前确认 Prometheus 能 scrape 到 node_exporter。
2. 固定 scrape interval，例如 1s、5s 或 15s。
3. 每次实验记录开始和结束时间。
4. Grafana 查询时按时间窗口叠加 benchmark 运行区间。
5. 对多 benchmark 批量运行，使用 runner 额外写入阶段标记指标，标识当前运行的是哪个 benchmark。

重点关注：

- CPU 利用率和负载。
- 内存可用量、swap。
- 磁盘 I/O。
- thermal throttling 或温度。
- 频率变化。
- NUMA 相关指标，如果机器是多路 CPU。

## 10. 采集硬件性能计数器

硬件计数器建议用外部 `perf stat` 或 PAPI 工具包，不改源码：

1. 为每次 benchmark 外层套性能计数器采集。
2. 固定事件列表，例如 cycles、instructions、cache-references、cache-misses、branches、branch-misses。
3. 将采集结果解析为数值。
4. 计算派生指标，例如 IPC、cache miss rate、branch miss rate。
5. 将原始事件和派生值都写入 Prometheus。

建议指标名：

- `benchmark_perf_cycles_total`
- `benchmark_perf_instructions_total`
- `benchmark_perf_cache_misses_total`
- `benchmark_perf_branches_total`
- `benchmark_perf_branch_misses_total`
- `benchmark_perf_ipc`
- `benchmark_perf_cache_miss_ratio`

注意事项：

- perf 权限可能需要调整。
- 虚拟机、容器、WSL 环境中的硬件计数器可能不完整。
- 多进程或多线程 benchmark 要确认计数器是否覆盖子进程和线程。

## 11. 采集 GPU 指标

如果数据集里包含 CUDA、OpenCL 或其他 GPU 程序：

1. 启动 GPU exporter 长期采集 GPU 状态。
2. runner 在运行前后记录 GPU 型号、驱动、CUDA 版本。
3. 对短程序，用高频 `nvidia-smi` 或 profiler 做运行窗口内采样。
4. 将本次运行的 GPU 峰值、均值和能耗估算写入 batch 结果。

重点指标：

- GPU 利用率。
- 显存使用量。
- 显存带宽。
- SM 活跃度。
- 功耗。
- 温度。
- ECC 错误。

## 12. 写入 Prometheus 指标

有两条常用路线：

### 路线 A：Pushgateway

适合每次 benchmark 结束后推送一次结果：

1. runner 运行 benchmark。
2. runner 生成 Prometheus exposition format 指标。
3. runner 按 `job` 和稳定标签推送到 Pushgateway。
4. Prometheus scrape Pushgateway。
5. 实验结束后清理本次 job，避免旧结果一直留在 Pushgateway。

适用场景：

- benchmark 很短。
- 每次运行结果是离散样本。
- 不想在本机维护 textfile 目录。

### 路线 B：node_exporter textfile collector

适合本机实验结果落盘：

1. 启动 node_exporter 的 textfile collector。
2. runner 将本次结果写成临时 `.prom` 文件。
3. 写完后原子重命名为正式 `.prom` 文件。
4. Prometheus scrape node_exporter 时自动读取这些指标。

适用场景：

- benchmark 和 node_exporter 在同一台机器。
- 希望指标文件能直接留档。
- 不想引入 Pushgateway。

## 13. 配置 Grafana 面板

建议按实验问题建立 dashboard：

1. 总览面板：成功率、平均耗时、最近一次实验状态。
2. benchmark 对比：不同 benchmark 的 kernel time、wall time。
3. 编译配置对比：不同 compiler、opt_level 的速度差异。
4. 资源面板：CPU、内存、I/O、GPU 随时间变化。
5. 硬件计数器面板：IPC、cache miss rate、branch miss rate。
6. 稳定性面板：重复运行的方差、标准差、异常点。

查询维度优先使用：

- `dataset`
- `benchmark`
- `category`
- `compiler`
- `opt_level`
- `problem_size`
- `run_group`
- `host`

## 14. 控制实验变量

为了让 Prometheus 采到的数据有比较意义，实验前固定环境：

1. 固定 CPU governor。
2. 固定线程数和 OpenMP 环境变量。
3. 固定 NUMA 绑定策略。
4. 关闭无关后台任务。
5. 固定输入规模。
6. 每个 benchmark 至少重复运行多次。
7. 区分冷启动、热身运行和正式运行。
8. 记录机器负载、温度和频率。

PolyBenchC 本身适合做 kernel 性能对比，但如果程序运行时间太短，应放大 problem size 或增加重复次数，否则 Prometheus 的定时采样很容易错过运行窗口。

## 15. 推荐落地顺序

按下面顺序实施最稳：

1. 先搭 Prometheus、node_exporter、Grafana。
2. 手动运行一个 PolyBenchC benchmark，确认能看到主机指标。
3. 增加 runner，统一执行 benchmark 清单。
4. 用 PolyBenchC 自带计时或外层计时记录 `benchmark_kernel_seconds` 和 `benchmark_wall_seconds`。
5. 将单次运行结果写入 Pushgateway 或 textfile collector。
6. 增加 `/usr/bin/time -v` 或等价工具，补充峰值内存、缺页、上下文切换。
7. 增加 `perf stat`，补充硬件计数器。
8. 如果有 GPU benchmark，再增加 GPU exporter 和 GPU 采样。
9. 建 Grafana dashboard，对比不同 benchmark、输入规模、优化等级。
10. 固化实验规范，保存原始日志和 Prometheus 指标快照。

## 16. 最小可行方案

如果只想先跑通闭环：

1. 启动 Prometheus。
2. 启动 node_exporter。
3. 启动 Pushgateway 或开启 textfile collector。
4. 编译一个 PolyBenchC 程序。
5. 用 wrapper 重复运行该程序。
6. wrapper 记录运行时间、退出码、峰值内存。
7. wrapper 写出 Prometheus 指标。
8. Prometheus scrape 这些指标。
9. Grafana 画出不同 benchmark 的耗时和资源占用。

跑通后再逐步扩展到 perf、PAPI、GPU、批量实验和自动 dashboard。

