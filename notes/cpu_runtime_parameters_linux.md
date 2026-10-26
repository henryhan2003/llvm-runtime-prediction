# PolyBench/C 与 Rodinia OpenMP：CPU runtime parameters 采集

## 本次改动

两条入口保持不变：

- `scripts/pipeline_new.py`：PolyBench/C。
- `scripts/pipeline_rodinia.py`：Rodinia OpenMP。

不改变输入规模、编译优化选项、OpenMP 线程配置、Prometheus 三阶段采样设置，也不修改已有结果 CSV。新字段写入以后新生成的结果表及逐次运行表。

运行期指标是资源监督标签或质量分析数据，不得作为纯静态执行时间模型的直接实测输入。两阶段模型应先预测它们，再把预测值传给第二阶段。

## 论文参数与实际采集字段

| 论文参数 | 两条 pipeline 的统一字段 | 说明 |
| --- | --- | --- |
| uCPU | `process_cpu_user_sec` | 用户态 CPU 秒数；OpenMP 包括进程线程累计使用量，不是 CPU 使用率。 |
| sCPU | `process_cpu_system_sec` | 内核态 CPU 秒数，单独保留。 |
| Memory usage | `process_max_rss_bytes` | 每次调用峰值 RSS；汇总表是各次峰值的均值，单位字节。不是时间平均内存、内存带宽或任意进程树 RSS 之和。 |
| Read operations / blocks | `process_fs_inputs` | 保留 rusage 的 `ru_inblock` 文件系统输入计数，不重命名为字节或 read 系统调用次数。 |
| Write operations / blocks | `process_fs_outputs` | 保留 rusage 的 `ru_oublock` 文件系统输出计数，不擅自按块大小换算字节。 |
| File transfer | `task_file_transfer_bytes` | 当前没有任务级工作流文件传输记录，数值留空。 |
| Bandwidth | `task_network_bandwidth_bytes_per_sec` | 当前没有传输阶段及其持续时间记录，数值留空。这里指网络传输带宽，不是内存带宽。 |

文件传输和带宽对应的状态为：

```text
task_transfer_metrics_status=not_collected_no_workflow_transfer_instrumentation
```

这不是采到了零，也不是断言程序绝不联网。它只表示当前采集器没有论文所依赖的任务级工作流传输观测。不能用输入文件大小、整机网络字节、背景扣除后的网络字节，或 I/O 字节除以真实运行时间，冒充这些参数。安装一个网络 exporter 也不能自动补齐任务归属与传输时段。

## 新增的进程级 I/O 明细

| 字段 | 来源 | 单位与边界 |
| --- | --- | --- |
| `process_io_rchar_bytes` | `/proc/PID/io` 的 `rchar` | 成功 read 类调用返回的逻辑字节数；读取缓存也会计入，但不覆盖所有 mmap 访问。 |
| `process_io_wchar_bytes` | `wchar` | 成功 write 类调用写出的逻辑字节数；可能包括 stdout/stderr 等，并非只包括普通文件。 |
| `process_io_read_syscalls` | `syscr` | read 家族等相关调用次数，不是读入块数。 |
| `process_io_write_syscalls` | `syscw` | write 家族等相关调用次数，不是写出块数。 |
| `process_storage_read_bytes` | `read_bytes` | 实际从存储层取回的字节；缓存命中时可以为零。 |
| `process_storage_write_bytes` | `write_bytes` | Linux 记账的存储层写出字节，受回写和文件系统影响。 |
| `process_storage_cancelled_write_bytes` | `cancelled_write_bytes` | 已记账、后来因截断/删除等取消回写的字节；单独保留，不静默相减或截零。 |

上述指标由被启动命令的 PID 获得，不读 Python pipeline 父进程的 I/O，不从 `/proc/PID/net/dev` 读取所谓“进程网络”，也不从主机 Prometheus 值推断进程用量。`/proc/PID/io` 的 Linux 口径包括已被该进程等待回收的子进程；任意脱离/未等待子进程不在这里承诺的范围内。

程序输出重定向方式沿用各 pipeline 原来的设置。OpenMP 的 stdout 默认丢弃，PolyBench 的 stdout/stderr 仍按原方式处理，因此逻辑输出字节含义必须结合重定向配置阅读，不能把它们当作纯算法输出文件大小。运行包含初始化、读入、计算、清理等整个被启动命令的活动，不是只测 kernel。

这些量均在逐次运行表中保留原值；汇总表给出每次成功正式运行的算术均值，不按测量窗口加总。原有 `*_total` 等兼容字段仍按原含义保留，不与本次的单次均值混淆。

## 短进程与计时边界

共享 C 探针升级为 `linux-process-probe-v2`。Linux 下先通过 `waitid(..., WEXITED | WNOWAIT)` 等待目标命令退出但暂不回收，读取其最终 `/proc/PID/io`，再通过 `wait4` 获得 CPU/RSS 等并回收进程。

这不依赖 1 秒或几十毫秒的轮询，短程序也有机会取得最终计数；不人为延长任务、不增加停顿。进程退出后的 procfs 读取和结果文件写入不计入 `runtime_sec`。v1 的计时终点在 `wait4` 返回之后，v2 在退出通知之后，因此必须保留 backend 版本，不应静默混合新旧采集批次。

旧 v1 探针输出仍可解析，但新增 I/O 字段保持空，状态为 `not_collected_probe_v1`。探针会根据新源码重新编译；请不要只同步两个 Python 入口而漏掉共享 C 源码和新辅助模块。

## 完整性与缺失状态

- `process_io_collection_status=ok`：这一条运行的七项新增 I/O 计数完整，真实零值保留为 0。
- `permission_denied`：目标进程 procfs 读权限被系统策略限制。
- `unavailable_proc_io` / `invalid_proc_io` / `invalid_probe_io`：procfs 不可用、格式不完整或解析失败。
- `not_collected_probe_v1`：读到旧版探针，不能生成新增值。
- `not_collected_platform_or_probe_failure` 等：不支持的平台或探针失败，数值留空。
- 汇总 `partial`：成功正式运行中只有一部分 I/O 完整。为了不静默平均偏差子集，七项汇总均值全部留空，但逐次运行表中的有效计数仍保留。
- 汇总 `unavailable`：存在成功正式运行，但新增 I/O 没有完整样本。
- 汇总 `not_run`：没有成功的正式运行，不解释成资源消耗为零。

汇总另给出 `process_io_sampled_runs`、`process_io_missing_runs`，错误详情在 `process_io_collection_error`。warmup 和失败运行不进入本次新增 I/O 均值。

CPU/RSS 成功不代表扩展 I/O 也成功，`run_success=True` 也不替代 `process_io_collection_status`。全量采集前必须执行下面的自检。

## Linux 需要安装什么

这次新增采集不需要 psutil、sysstat、tcpdump、eBPF 或额外 Prometheus exporter，也不要求 sudo/root 运行 pipeline。

需要：

1. Linux，挂载正常的 `/proc`，允许当前用户读取自己启动的普通子进程的 `/proc/PID/io`。某些容器、内核配置或安全策略可能限制，实际以自检为准。
2. C 编译器 `cc`、`gcc`、`clang` 三者至少一个，用于自动构建探针。
3. 原项目 Python 环境及已有 `prometheus-client`。原来的 LLVM 静态提取和 benchmark 编译依赖照旧。

如果服务器原来已经能编译运行这两个数据集，一般不需要再安装。Ubuntu/Debian 缺少 C 编译器时，手动执行：

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

在已手动激活的项目 Python 环境中，确认原依赖：

```bash
python -m pip install -r requirements.txt
```

此前服务器数据盘接近满盘，安装或全量实验前仍应检查可用空间。不要为运行采集而直接放宽整个系统的 procfs/ptrace 安全设置；权限失败先看自检报告并与管理员确认。

## 同步到服务器后先自检

至少同步这些改动文件：

```text
scripts/pipeline_new.py
scripts/pipeline_rodinia.py
scripts/rodinia_process_probe.c
scripts/cpu_runtime_parameters.py
scripts/check_cpu_runtime_parameters.py
```

登录服务器、进入项目目录、手动激活原来的 conda 环境后执行：

```bash
python scripts/check_cpu_runtime_parameters.py \
  --output results/cpu_runtime_parameters_check.json
```

脚本分别经过 PolyBench 和 OpenMP 的真实运行接口启动一个短命子进程，写入并读回已知的 64 KiB 数据；检查 CPU、RSS、逻辑 I/O、存储 I/O 字段、空的传输字段和退出状态。测试文件放在临时目录，自动清理，不会运行 20 档 benchmark 或修改旧 CSV。

预期退出码为 0，报告 `status=PASS`，两条路径的 `process_io_collection_status=ok`。存储读字节为 0 可以是正常缓存命中，脚本不会强制清缓存来制造非零值。

若失败，停止全量采集，先查看 JSON 中的错误。它失败时返回非零退出码，不会表面显示 PASS。这个脚本不测试服务器 Prometheus 是否可用；主机三阶段采集仍应沿用原来的 Prometheus pilot 检查。

通过后原来的 pipeline 启动参数可继续使用，不需要新增开关。不要立刻把两套全量实验同时启动，以免互相干扰。

可选回归测试：

```bash
python -m unittest discover -s tests -p 'test_cpu_runtime_parameters.py' -v
```

## 验证边界

当前 Windows 工作区已验证 v1/v2 解析、两入口字段一致、真实零/缺失区分、按正式运行汇总以及 CSV 字段输出。真实 Linux `/proc`/waitid 编译与执行测试在 Windows 上跳过；上述服务器自检通过之前，不宣称 Linux 实机验收完成。

共享探针和 Rodinia 运行工具原来也被 CUDA 的 host 侧复用，因此共享工具兼容 v2；没有修改 CUDA kernel、GPU 采样、输入配置或运行入口。无论哪条路径使用这些字段，都只能按 host/process 范围解释，不能视为 GPU device 指标。

## 定义来源

- Linux rusage：https://man7.org/linux/man-pages/man2/getrusage.2.html
- Linux 进程 I/O：https://man7.org/linux/man-pages/man5/proc_pid_io.5.html
- waitid/WNOWAIT：https://man7.org/linux/man-pages/man2/waitpid.2.html
