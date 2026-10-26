# CPU runtime parameters：Git 同步与 Linux 20 档采集指令

顺序为：Windows 提交推送，服务器更新，进入 tmux 手动激活 conda，自检与单配置 pilot，最后逐套全量采集。不要同时运行两套，避免资源竞争。以下指令不会替你执行 Git 提交或启动实验。

本次参数沿用之前 OpenMP 快速采集：warmup=1、measurement-seconds=3、至少 3 次正式运行、O2，Prometheus 上下文各 5 秒、查询步长 1 秒。3 秒是预算，不是超时；慢程序至少运行 3 次后才结束，实际用时可能明显超过 3 秒。每次调用超时设置为 1800 秒，防止大档位因原来 30 秒限制失败；这不是要求每次都跑 1800 秒。

## 一、Windows PowerShell：提交和推送

只提交本次采集改动，不提交大数据、历史 CSV、建模结果。当前分支已核对为 main，远端为 origin。

```powershell
Set-Location 'E:\Dareway\LLVM_v01'
$Files = @(
    'scripts/pipeline_new.py'
    'scripts/pipeline_rodinia.py'
    'scripts/rodinia_process_probe.c'
    'scripts/cpu_runtime_parameters.py'
    'scripts/check_cpu_runtime_parameters.py'
    'tests/test_cpu_runtime_parameters.py'
    'tests/test_pipeline_rodinia.py'
    'notes/cpu_runtime_parameters_linux.md'
    'notes/cpu_runtime_parameters_git_collection.md'
)
git add -- $Files
git --no-pager diff --cached --name-only
git commit -m "Collect CPU runtime parameters and process IO for CPU pipelines"
git push origin main
```

每条成功后再执行下一条。暂存清单应只有上述文件；若出现别的已暂存文件，先停下核对。若再出现 index.lock，不要不检查就删除；若进入显示 END 的 Git 分页器，按 q 退出即可。

## 二、服务器：更新代码并进入 tmux

在 Windows 另一个终端登录：

```bash
ssh hanli@211.87.232.179
```

以下开始均在服务器 Linux 终端执行：

```bash
cd /data_disk/hanli/llvm-runtime-prediction
git status --short
git pull --ff-only origin main
tmux new -s cpu_runtime_v2
```

如果 git pull 提示本地修改冲突，先停止并保留修改，不要使用 reset --hard、强制 checkout 或删除配置。若 tmux 提示会话已存在，使用 `tmux attach -t cpu_runtime_v2`，不要另开一个重复实验。

进入 tmux 后，手动执行环境激活：

```bash
cd /data_disk/hanli/llvm-runtime-prediction
source /data_disk/hanli/miniforge3/etc/profile.d/conda.sh
conda activate llvm-runtime
```

后续所有采集指令都在这个 tmux 会话中执行。不要在外面的 SSH 激活环境后，假定已有 tmux 会话也自动继承了它。

## 三、磁盘、监控与新增 I/O 自检

```bash
df -h /data_disk
free -h
pgrep -u "$(id -u)" -af '[p]ipeline_(new|rodinia|rodinia_cuda)\.py' || true
curl -fsS --max-time 5 http://127.0.0.1:9090/-/ready
curl -fsSG --max-time 5 http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=up{job="node",instance="127.0.0.1:9100"}'
python scripts/check_cpu_runtime_parameters.py --output results/cpu_runtime_parameters_check.json
```

继续条件：

- 数据盘不能满，建议预留数 GB。此前数据盘已接近满盘，不要忽略这项。
- 没有自己之前启动的 CPU/GPU 采集进程仍在运行。此检查不能保证别人的任务没有占用服务器，仍应避免竞争。
- readiness 返回 Ready；node 的 up 查询结果非空，值为 1。拒绝连接时先恢复监控，不要带病采集。
- 新增自检报告 `status=PASS`，两条路径的 `process_io_collection_status=ok`。

这次一般不需要再装软件。已有 gcc/cc/clang 就能自动编译探针，Python 仍用原环境的 prometheus-client。不需要安装 psutil、sysstat、tcpdump 或新 exporter。缺少编译器时再安装 build-essential，详见 cpu_runtime_parameters_linux.md。

## 四、一次定义公共参数、PolyBench 启动包装和结果验收

仍在同一个 tmux shell 中，将下面完整代码块执行一次。断开 SSH 不会清除 tmux 内的这些变量和函数；若退出了 tmux shell，则需要重新定义。

```bash
set -o pipefail
RUN_TAG="$(date +%Y%m%d_%H%M%S)_$$"
RUN_ROOT="/data_disk/hanli/cpu_runtime_v2_${RUN_TAG}"
mkdir -p "$RUN_ROOT"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
COMMON=(--warmup 1 --measurement-seconds 3 --min-measured-runs 3 --timeout 1800 --opt-level O2 --c-compiler clang --cxx-compiler clang++ --prometheus-url http://127.0.0.1:9090 --prometheus-exporter node --prometheus-job node --prometheus-instance 127.0.0.1:9100 --prometheus-query-step 1 --prometheus-query-timeout 10 --prometheus-query-delay 2 --prometheus-context-seconds 5)
PB_SIZES=()
for i in $(seq -w 1 20); do
    PB_SIZES+=(--polybench-size "SIZE_${i}")
done
pb_collect() {
    PB_OUTPUT_DIR="$1" python -u -c 'import os; from pathlib import Path; from scripts import pipeline_new as p; p.RESULTS_DIR=Path(os.environ["PB_OUTPUT_DIR"]); raise SystemExit(p.main())' "${@:2}"
}
check_cpu_csv() {
    python - "$1" "$2" "$3" <<'PY'
import csv, math, sys
from collections import defaultdict
from pathlib import Path
path = Path(sys.argv[1])
expected_rows, expected_programs = int(sys.argv[2]), int(sys.argv[3])
with path.open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))
errors = []
identities = [(r['program_id'], r['input_id']) for r in rows]
if len(rows) != expected_rows or len(set(identities)) != expected_rows:
    errors.append(f'expected {expected_rows} unique rows, got {len(rows)} rows / {len(set(identities))} unique')
groups = defaultdict(set)
for row in rows:
    identity = (row['program_id'], row['input_id'])
    groups[row['program_id']].add(row['input_id'])
    for name in ['build_success', 'run_success', 'prometheus_three_phase_collection_success']:
        if row.get(name, '').lower() not in ('true', '1'):
            errors.append(f'{identity}: {name}={row.get(name)}')
    for name in ['ast_success', 'cfg_success', 'ir_success']:
        if name in row and row[name].lower() not in ('true', '1'):
            errors.append(f'{identity}: {name}={row[name]}')
    if 'static_status' in row and row['static_status'] != 'success':
        errors.append(f'{identity}: static_status={row["static_status"]}')
    if row.get('process_io_collection_status') != 'ok':
        errors.append(f'{identity}: I/O={row.get("process_io_collection_status")} {row.get("process_io_collection_error")}')
    nonnegative = ['process_cpu_user_sec', 'process_cpu_system_sec', 'process_fs_inputs', 'process_fs_outputs',
                   'process_io_rchar_bytes', 'process_io_wchar_bytes', 'process_io_read_syscalls',
                   'process_io_write_syscalls', 'process_storage_read_bytes', 'process_storage_write_bytes',
                   'process_storage_cancelled_write_bytes']
    for name in nonnegative + ['runtime_sec_median', 'process_max_rss_bytes']:
        try:
            value = float(row[name])
            valid = math.isfinite(value) and (value >= 0 if name in nonnegative else value > 0)
        except (ValueError, KeyError):
            valid = False
        if not valid:
            errors.append(f'{identity}: invalid {name}={row.get(name)}')
    if row.get('task_file_transfer_bytes', '') != '' or row.get('task_network_bandwidth_bytes_per_sec', '') != '':
        errors.append(f'{identity}: uninstrumented transfer values must stay blank')
if len(groups) != expected_programs:
    errors.append(f'expected {expected_programs} programs, got {len(groups)}')
if expected_rows > 1:
    sizes = {f'SIZE_{i:02d}' for i in range(1, 21)}
    for program, actual in groups.items():
        if actual != sizes:
            errors.append(f'{program}: does not have all twenty sizes')
print(f'{"FAIL" if errors else "PASS"}: {path}; rows={len(rows)}, programs={len(groups)}, errors={len(errors)}')
for error in errors[:30]:
    print(error)
sys.exit(1 if errors else 0)
PY
}
```

说明：PolyBench 目前没有 `--results-dir` 参数，pb_collect 只在当前 Python 进程中设置输出目录，然后调用原来的 main；不会改源码或覆盖项目 results 下的旧辅助表。输入参数、编译流程和采样逻辑不变。

## 五、OpenMP：先单配置 pilot，再 380 条全量

先执行 BFS 的 SIZE_01。它会重建 benchmark、采集静态表示、CPU 与 I/O、Prometheus 三阶段：

```bash
OMP_PILOT="$RUN_ROOT/openmp_pilot"
mkdir -p "$OMP_PILOT"
python -u scripts/pipeline_rodinia.py --manifest configs/rodinia_openmp_20sizes.json --benchmark bfs --input-profile SIZE_01 --results-dir "$OMP_PILOT" --build-timeout 1800 --collect-static "${COMMON[@]}" 2>&1 | tee "$OMP_PILOT/pipeline.log"
OMP_PILOT_STATUS=("${PIPESTATUS[@]}")
printf 'pipeline_exit=%s tee_exit=%s\n' "${OMP_PILOT_STATUS[0]}" "${OMP_PILOT_STATUS[1]}" | tee -a "$OMP_PILOT/pipeline.log"
check_cpu_csv "$(find "$OMP_PILOT" -maxdepth 1 -name 'rodinia_openmp_summary_*.csv' -print -quit)" 1 1
```

两个退出码都为 0 且验收 PASS 才继续。不能只看 pipeline_exit=0：原来 pipeline 的运行成功不代表所有监控字段完整。

380 条正式采集：

```bash
OMP_DIR="$RUN_ROOT/openmp_full"
mkdir -p "$OMP_DIR"
python -u scripts/pipeline_rodinia.py --manifest configs/rodinia_openmp_20sizes.json --all --input-profile all --results-dir "$OMP_DIR" --build-timeout 1800 --collect-static "${COMMON[@]}" 2>&1 | tee "$OMP_DIR/pipeline.log"
OMP_STATUS=("${PIPESTATUS[@]}")
printf 'pipeline_exit=%s tee_exit=%s\n' "${OMP_STATUS[0]}" "${OMP_STATUS[1]}" | tee -a "$OMP_DIR/pipeline.log"
check_cpu_csv "$(find "$OMP_DIR" -maxdepth 1 -name 'rodinia_openmp_summary_*.csv' -print -quit)" 380 19
```

PIPESTATUS 数组的保存行必须紧接 pipeline 命令；不要在两者中间插入 date、ls、echo 或其他命令。上面的验收检查字段/配置完整性，不是重新运行 benchmark，也不是模型训练。

## 六、PolyBench/C：先单配置 pilot，再 600 条全量

等 OpenMP 完成、没有采集进程运行后再进行。先执行 GEMM 的 SIZE_01：

```bash
PB_PILOT="$RUN_ROOT/polybench_pilot"
mkdir -p "$PB_PILOT"
pb_collect "$PB_PILOT" --source-dir datasets/PolyBenchC-4.2.1/linear-algebra/blas/gemm --dataset PolyBenchC-4.2.1 --parallel-model serial --polybench-size SIZE_01 --run-c-compiler gcc --run-cxx-compiler g++ "${COMMON[@]}" 2>&1 | tee "$PB_PILOT/pipeline.log"
PB_PILOT_STATUS=("${PIPESTATUS[@]}")
printf 'pipeline_exit=%s tee_exit=%s\n' "${PB_PILOT_STATUS[0]}" "${PB_PILOT_STATUS[1]}" | tee -a "$PB_PILOT/pipeline.log"
check_cpu_csv "$(find "$PB_PILOT" -maxdepth 1 -name 'results_*.csv' -print -quit)" 1 1
```

两个退出码都为 0 且验收 PASS 才继续。600 条正式采集，显式传入 20 个规模；不传 --polybench-size 时，原 pipeline 默认仍然是旧五档，不是 20 档：

```bash
PB_DIR="$RUN_ROOT/polybench_full"
mkdir -p "$PB_DIR"
pb_collect "$PB_DIR" --source-dir datasets/PolyBenchC-4.2.1 --dataset PolyBenchC-4.2.1 --parallel-model serial --run-c-compiler gcc --run-cxx-compiler g++ "${PB_SIZES[@]}" "${COMMON[@]}" 2>&1 | tee "$PB_DIR/pipeline.log"
PB_STATUS=("${PIPESTATUS[@]}")
printf 'pipeline_exit=%s tee_exit=%s\n' "${PB_STATUS[0]}" "${PB_STATUS[1]}" | tee -a "$PB_DIR/pipeline.log"
check_cpu_csv "$(find "$PB_DIR" -maxdepth 1 -name 'results_*.csv' -print -quit)" 600 30
```

如果想把预算改回 10 秒，应在启动任何 pilot/全量前修改第四步 COMMON 中的 `--measurement-seconds 3` 为 10，让本轮两套使用同一预算；不要中途改完后混合解释。

## 七、断开与重新连接

采集运行时按 Ctrl+B，松开，再按 D，返回普通 SSH shell。不要输入 exit 退出承载运行的 tmux shell，不要按 Ctrl+C，也不要同时再启动第二份 pipeline。

SSH 掉线后重新登录并恢复：

```bash
ssh hanli@211.87.232.179
tmux attach -t cpu_runtime_v2
```

tmux 能防 SSH 掉线，不能防服务器重启、磁盘满或进程被管理员终止。pipeline 仍在批次末尾写汇总，当前不提供断点续采。

## 八、结果位置

运行根目录由第四步输出，如 `/data_disk/hanli/cpu_runtime_v2_时间_PID/`。其中：

- `openmp_pilot/`、`polybench_pilot/`：pilot 结果，不与正式结果混合。
- `openmp_full/`：19×20=380 条，带时间戳的 summary、runs、environment。
- `polybench_full/`：30×20=600 条，带时间戳 results 汇总和独立的 run_records/run_summary/static/environment 文件。
- 每个目录都有 pipeline.log；全量记录应使用本轮 v2 backend，不混入旧批次。

文件传输量和网络带宽为空且带明确未采集状态是本轮预期，不是验收遗漏；它们没有任务级传输观测，不能强行变成零值。
