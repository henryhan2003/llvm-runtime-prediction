"""Opt-in Linux background load control; reuse the pipelines' host metric collector.

Control parameters, safety telemetry and provenance live only in JSON sidecars.
This module never defines CSV features or changes static feature extraction.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
import uuid

GIB = 1024 ** 3
DEFAULTS = {
    "schema_version": 1, "profile_id": "IDLE", "load_type": "idle",
    "target_cpu_fraction": 0.0, "memory_budget_fraction": 0.0,
    "permission_confirmed": False, "max_cpu_fraction": 0.75,
    "max_memory_bytes": 8 * GIB, "memory_reserve_fraction": 0.20,
    "memory_reserve_bytes": 2 * GIB, "max_duration_seconds": 1200,
    "stabilization_timeout_seconds": 120, "stable_window_seconds": 10,
    "min_raw_scrapes": 5, "cpu_tolerance_pct": 8.0,
    "max_baseline_cpu_pct": 15.0, "max_temperature_celsius": 85.0,
    "temperature_policy": "abort",
    "collection_protocol": "strict-v1", "minimum_measured_runs": 3, "maximum_measured_runs": 0,
    "cleanup_timeout_seconds": 10, "stress_ng": "stress-ng",
}


def add_arguments(parser):
    parser.add_argument("--load-config", type=Path, default=None,
                        help="Opt-in Linux resource-load JSON. Omit for unchanged legacy behavior.")
    parser.add_argument("--load-record-dir", type=Path, default=None,
                        help="JSON sidecars only; load controls are not CSV model inputs.")


def read_config(path):
    supplied = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown load configuration fields")
    config = {**DEFAULTS, **supplied}
    if config["schema_version"] != 1 or config["load_type"] not in {"idle", "cpu", "memory"}:
        raise ValueError("Expected load schema 1 and idle/cpu/memory; GPU load is not implemented")
    for key, default in DEFAULTS.items():
        value = config[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be nonempty text")
    if not (0 < config["max_cpu_fraction"] <= .75 and 0 <= config["target_cpu_fraction"] <= config["max_cpu_fraction"]):
        raise ValueError("CPU fraction must not exceed the configured cap or hard cap 0.75")
    if not (0 <= config["memory_budget_fraction"] <= .75 and .10 <= config["memory_reserve_fraction"] <= .8):
        raise ValueError("Unsafe memory fractions")
    for key in ["max_memory_bytes", "memory_reserve_bytes", "max_duration_seconds", "stabilization_timeout_seconds",
                "stable_window_seconds", "min_raw_scrapes", "cleanup_timeout_seconds"]:
        if config[key] <= 0 or int(config[key]) != config[key]:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_duration_seconds"] > 3600 or config["min_raw_scrapes"] < 3:
        raise ValueError("Hard duration cap is one hour; at least three real scrapes required")
    if not (0 < config["cpu_tolerance_pct"] <= 10 and 0 <= config["max_baseline_cpu_pct"] <= 20 and
            50 <= config["max_temperature_celsius"] <= 90):
        raise ValueError("Unsafe CPU/temperature gate")
    if config["temperature_policy"] not in {"abort", "record-only"}:
        raise ValueError("temperature_policy must be abort or record-only")
    if config["collection_protocol"] not in {"strict-v1", "fast-v2"}:
        raise ValueError("Unknown collection protocol")
    minimum, maximum = config["minimum_measured_runs"], config["maximum_measured_runs"]
    if int(minimum) != minimum or int(maximum) != maximum or maximum < 0 or minimum < (2 if config["collection_protocol"] == "fast-v2" else 3):
        raise ValueError("Invalid protocol measurement limits")
    if maximum and maximum < minimum:
        raise ValueError("Measurement cap is below the minimum count")
    if config["load_type"] == "cpu" and (config["target_cpu_fraction"] <= 0 or config["memory_budget_fraction"] != 0):
        raise ValueError("CPU load requires a positive CPU target and no memory stress")
    if config["load_type"] == "memory" and (config["memory_budget_fraction"] <= 0 or config["target_cpu_fraction"] != 0):
        raise ValueError("Memory load requires a memory budget fraction and no CPU stress")
    if config["load_type"] == "idle" and (config["memory_budget_fraction"] or config["target_cpu_fraction"]):
        raise ValueError("IDLE must not apply any synthetic load")
    return config


def validate_args(args, results_dir):
    if getattr(args, "load_config", None) is None:
        return
    config = read_config(args.load_config)
    if config["collection_protocol"] == "fast-v2":
        if getattr(args, "min_measured_runs", None) != config["minimum_measured_runs"] or getattr(args, "max_measured_runs", None) != config["maximum_measured_runs"]:
            raise ValueError("Pipeline repetition limits must match the fast-v2 load configuration")
    if args.prometheus_exporter not in {"node", "auto"} or not args.prometheus_job or not args.prometheus_instance:
        raise ValueError("Load experiments require node exporter and explicit --prometheus-job/--prometheus-instance")
    if getattr(args, "no_run", False) or getattr(args, "build_only", False):
        raise ValueError("Load experiments require target execution")
    if hasattr(args, "gpu_metrics") and args.gpu_metrics == "off":
        raise ValueError("CUDA load experiments must retain the existing nvidia-smi monitoring")
    directory = Path(results_dir).resolve()
    if directory.exists() and any(directory.glob("*.csv")):
        raise ValueError("Use a fresh results directory for load experiments; never overwrite earlier CSVs")
    args._load_results_dir = directory
    args._load_sessions = []


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def cgroup_directories(controller):
    roots = [(Path("/sys/fs/cgroup"), True), (Path("/sys/fs/cgroup") / controller, False)]
    candidates = list(roots)
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            for root, unified in roots:
                if (unified and controllers) or (not unified and controller not in controllers.split(",")):
                    continue
                candidate = (root / relative.lstrip("/")).resolve()
                boundary = root.resolve()
                if not candidate.is_relative_to(boundary):
                    continue
                while candidate != boundary:
                    candidates.append((candidate, unified))
                    candidate = candidate.parent
    except (OSError, ValueError):
        pass
    return list(dict.fromkeys(candidates))


def cpu_quota_cores():
    limits = []
    for directory, unified in cgroup_directories("cpu"):
        try:
            if unified:
                quota, period = (directory / "cpu.max").read_text().split()
            else:
                quota = (directory / "cpu.cfs_quota_us").read_text().strip()
                period = (directory / "cpu.cfs_period_us").read_text().strip()
            if quota != "max" and int(quota) > 0 and int(period) > 0:
                limits.append(int(quota) / int(period))
        except (OSError, ValueError):
            pass
    return min(limits) if limits else None


def memory_info():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapFree", "SwapTotal"}:
            values[name] = int(raw.split()[0]) * 1024
    # Respect cgroup limits in addition to the host's MemAvailable.
    for directory, unified in cgroup_directories("memory"):
        try:
            maximum = directory / ("memory.max" if unified else "memory.limit_in_bytes")
            current = directory / ("memory.current" if unified else "memory.usage_in_bytes")
            cap = maximum.read_text().strip()
            if cap != "max" and int(cap) < values["MemTotal"]:
                values["MemAvailable"] = min(values["MemAvailable"], max(0, int(cap) - int(current.read_text())))
                values["EffectiveTotal"] = min(values.get("EffectiveTotal", values["MemTotal"]), int(cap))
        except (OSError, ValueError):
            pass
    values.setdefault("EffectiveTotal", values["MemTotal"])
    return values


def safety_snapshot():
    memory = memory_info()
    temperatures = []
    for path in [*Path("/sys/class/thermal").glob("thermal_zone*/temp"), *Path("/sys/class/hwmon").glob("hwmon*/temp*_input")]:
        try:
            value = float(path.read_text()) / 1000
            if 0 < value < 150:
                temperatures.append((value, path))
        except (OSError, ValueError):
            pass
    hottest = max(temperatures, key=lambda item: item[0]) if temperatures else None
    temperature_source = None
    if hottest is not None:
        path = hottest[1]
        device_file = path.parent / ("type" if path.name == "temp" else "name")
        label_file = path.with_name(path.name.removesuffix("_input") + "_label")
        try:
            device = device_file.read_text().strip()
        except OSError:
            device = path.parent.name
        try:
            label = label_file.read_text().strip() if path.name != "temp" else device
        except OSError:
            label = path.name
        temperature_source = {"device": device, "label": label, "path": str(path)}
    frequencies = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            frequencies.append(float(path.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return {"timestamp": time.time(), "memory_available_bytes": memory["MemAvailable"],
            "temperature_max_celsius": hottest[0] if hottest else None,
            "temperature_max_source": temperature_source,
            "cpu_frequency_mean_mhz": statistics.mean(frequencies) if frequencies else None}


def allocation_bytes(config, memory):
    reserve = max(config["memory_reserve_bytes"], memory["EffectiveTotal"] * config["memory_reserve_fraction"])
    budget = min(config["max_memory_bytes"], memory["EffectiveTotal"] * .25, memory["MemAvailable"] - reserve)
    if budget <= 0:
        raise RuntimeError("Insufficient free memory after reserve")
    amount = int(budget * config["memory_budget_fraction"]) // 4096 * 4096
    if amount < 16 * 1024 ** 2:
        raise RuntimeError("Memory load would be smaller than 16 MiB")
    return amount, int(reserve), int(budget)


def duration_seconds(value):
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value)
    if not parts or "".join(a + b for a, b in parts) != value:
        raise ValueError(f"Unsupported Prometheus duration: {value}")
    return sum(float(a) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[b] for a, b in parts)


def api(args, endpoint, parameters=None):
    url = args.prometheus_url.rstrip("/") + endpoint
    if parameters:
        url += "?" + urlencode(parameters)
    with urlopen(url, timeout=args.prometheus_query_timeout) as response:
        result = json.load(response)
    if result.get("status") != "success":
        raise RuntimeError(result.get("error", "Prometheus API failure"))
    return result["data"]


def monitoring_preflight(args):
    local_names = {"localhost", "127.0.0.1", "::1", socket.gethostname(), socket.getfqdn()}
    local_names.update(address[4][0] for address in socket.getaddrinfo(socket.gethostname(), None))
    if urlparse(args.prometheus_url).hostname not in local_names or urlparse("http://" + args.prometheus_instance).hostname not in local_names:
        raise RuntimeError("Background load and monitored node must be local to the same Linux host")
    targets = api(args, "/api/v1/targets", {"state": "active"})["activeTargets"]
    selected = [t for t in targets if t.get("labels", {}).get("job") == args.prometheus_job and
                t.get("labels", {}).get("instance") == args.prometheus_instance]
    if len(selected) != 1 or selected[0].get("health") != "up":
        raise RuntimeError("Exactly one healthy matching node exporter target is required")
    interval = duration_seconds(selected[0]["scrapeInterval"])
    if interval <= 0:
        raise RuntimeError("Invalid scrape interval")
    selector = "{job=" + json.dumps(args.prometheus_job) + ",instance=" + json.dumps(args.prometheus_instance) + "}"
    observed_total = api(args, "/api/v1/query", {"query": "node_memory_MemTotal_bytes" + selector})["result"]
    if len(observed_total) != 1 or not math.isclose(float(observed_total[0]["value"][1]), memory_info()["MemTotal"], rel_tol=.02):
        raise RuntimeError("Exporter memory capacity does not match the local host")
    return {"scrape_interval_seconds": interval, "target": selected[0], "selector": selector}


def fresh_scrape_times(core, args, selector, start, end, interval):
    output = {}
    for name, query in [("cpu", "node_cpu_seconds_total"), ("memory", "node_memory_MemAvailable_bytes")]:
        series = core.prometheus_query_range(args.prometheus_url, "timestamp(" + query + selector + ")",
            start, end, min(args.prometheus_query_step, interval / 2), args.prometheus_query_timeout)
        per_series = [sorted({value for _, value in core.parsed_series_values(s) if start <= value <= end}) for s in series]
        if not per_series or any(not values for values in per_series):
            output[name] = []
        else:
            output[name] = min(per_series, key=len)
    return output


class InsufficientScrapes(RuntimeError):
    def __init__(self, times):
        super().__init__("Not enough distinct in-window CPU/memory scrapes; query points are not raw scrapes")
        self.times = times


def short_during_allowed(record):
    window = record["phase_windows"]["during"]
    config = record["config"]
    interval = record["monitoring_preflight"]["scrape_interval_seconds"]
    return config.get("collection_protocol", "strict-v1") == "fast-v2" and 0 <= window[1] - window[0] < (config["min_raw_scrapes"] + 1) * interval


def monitoring_accepted(record, row):
    truth = lambda value: str(value).lower() in {"true", "1"}
    quality = record["phase_quality"]
    if any(quality.get(p, {}).get("status") != "PASS" or not truth(row.get(f"prometheus_{p}_collection_success")) for p in ["before", "after"]):
        return False
    if quality.get("during", {}).get("status") == "PASS":
        return truth(row.get("prometheus_during_collection_success")) and truth(row.get("prometheus_three_phase_collection_success"))
    if quality.get("during", {}).get("status") != "MISSING_SHORT_WINDOW" or not short_during_allowed(record):
        return False
    numeric = [f"host_{name}_during_{kind}" for name, kind in [
        ("cpu_usage_pct", "mean"), ("cpu_usage_pct", "max"), ("memory_used_bytes", "mean"), ("memory_used_bytes", "max"),
        ("disk_read_bytes", "delta"), ("disk_write_bytes", "delta"), ("network_bytes", "delta")]]
    return (not truth(row.get("prometheus_during_collection_success")) and not truth(row.get("prometheus_three_phase_collection_success"))
            and all(row.get(k, "") == "" for k in numeric))


class LoadSession:
    def __init__(self, args, core, identity):
        self.args, self.core, self.identity = args, core, identity
        self.config = read_config(args.load_config)
        root = getattr(args, "load_record_dir", None) or (args._load_results_dir / "load_records")
        self.path = Path(root) / (uuid.uuid4().hex + ".json")
        self.record = {"schema_version": 1, "status": "STARTING", "config": self.config, "identity": identity,
            "hostname": socket.gethostname(), "pid": os.getpid(), "created_at": time.time(),
            "after_phase_load_policy": "background_load_still_active; cleanup is outside all three phases",
            "before_phase_policy": "after load stabilization and before the first warmup/measurement invocation",
            "control_parameters_are_model_inputs": False, "safety_samples": [], "phase_quality": {}}
        prepared = getattr(args, "_preparation_provenance", {}).get((identity.get("program_id"), identity.get("input_id")))
        if prepared:
            self.record["preparation"] = prepared
        self.record["measurement_policy"] = {"minimum_runs": self.config["minimum_measured_runs"],
            "maximum_runs": self.config["maximum_measured_runs"], "budget_seconds": getattr(args, "measurement_seconds", None)}
        self.process = None
        self.log = None
        self.error = ""
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None
        self.previous_handlers = {}
        self.last_windows = None
        self.lock_fd = None
        self.queries = core.prometheus_host_queries("node", args.prometheus_job, args.prometheus_instance)

    def save(self):
        write_json(self.path, self.record)

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        if self.started is not None and time.monotonic() - self.started >= self.config["max_duration_seconds"]:
            raise RuntimeError("Load session maximum duration exceeded")
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"Background load exited early ({self.process.returncode}); see {self.path}")

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        self.check()

    def enforce_temperature_limit(self, sample):
        if self.config["temperature_policy"] == "record-only":
            return
        temperature = sample["temperature_max_celsius"]
        if temperature is not None and temperature >= self.config["max_temperature_celsius"]:
            source = sample.get("temperature_max_source") or {}
            raise RuntimeError(f"Safety abort: temperature cap reached ({temperature:.2f} C >= "
                f"{self.config['max_temperature_celsius']:.2f} C; device={source.get('device', 'unknown')}, "
                f"label={source.get('label', 'unknown')}, path={source.get('path', 'unavailable')})")

    def watchdog(self):
        while not self.stop_event.wait(1):
            try:
                sample = safety_snapshot()
                if len(self.record["safety_samples"]) < 3600:
                    self.record["safety_samples"].append(sample)
                if sample["memory_available_bytes"] < self.reserve:
                    raise RuntimeError("Safety abort: free memory fell below reserve")
                self.enforce_temperature_limit(sample)
                memory = memory_info()
                initial = self.record["memory_before"]
                if memory["SwapTotal"] - memory["SwapFree"] > initial["SwapTotal"] - initial["SwapFree"] + 64 * 1024 ** 2:
                    raise RuntimeError("Safety abort: system swap usage increased during the trial")
                self.check()
            except Exception as exc:
                self.error = str(exc)
                self.stop_owned_process()
                break

    def observation(self, start, end):
        times = self.checked_scrapes(start, end)
        metrics = self.core.collect_prometheus_host_metrics(self.args.prometheus_url, start, end,
            min(self.args.prometheus_query_step, self.interval), self.args.prometheus_query_timeout, self.queries)
        if not metrics.get("prometheus_collection_success") or metrics.get("host_cpu_usage_pct_mean", "") == "":
            raise RuntimeError("Missing measured CPU/memory state")
        return metrics, times

    def checked_scrapes(self, start, end):
        times = fresh_scrape_times(self.core, self.args, self.monitor["selector"], start, end, self.interval)
        if any(len(values) < self.config["min_raw_scrapes"] for values in times.values()):
            raise InsufficientScrapes(times)
        if any(end - values[-1] > 2.5 * self.interval for values in times.values()):
            raise RuntimeError("Node exporter samples are stale")
        return times

    def __enter__(self):
        self.save()
        try:
            if platform.system() != "Linux":
                raise RuntimeError("Resource load experiments must execute on Linux")
            if not self.config["permission_confirmed"]:
                raise RuntimeError("Explicit server-use permission confirmation is required, including IDLE trials")
            if self.config["temperature_policy"] == "record-only":
                print("[load] WARNING: temperature is recorded only; script temperature abort is disabled. "
                      "Thermal safety and absence of throttling are NOT validated; system protections are unchanged.",
                      file=sys.stderr, flush=True)
            import fcntl
            self.lock_fd = os.open(str(Path(tempfile.gettempdir()) / "llvm-v01-resource-load.lock"),
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.monitor = monitoring_preflight(self.args)
            self.interval = self.monitor["scrape_interval_seconds"]
            self.window = max(self.config["stable_window_seconds"], (self.config["min_raw_scrapes"] + 2) * self.interval)
            if self.args.prometheus_context_seconds < (self.config["min_raw_scrapes"] + 1) * self.interval:
                raise RuntimeError("Increase --prometheus-context-seconds to at least (min_raw_scrapes+1)*scrape_interval")
            self.record["monitoring_preflight"] = self.monitor
            self.record["baseline_recovery_wait_started_at"] = time.time()
            time.sleep(self.window + 2 * self.interval)
            baseline, baseline_times = self.observation(time.time() - self.window, time.time())
            self.record["baseline_metrics"] = baseline
            self.record["baseline_scrape_timestamps"] = baseline_times
            self.record["observed_scrape_median_interval_seconds"] = {
                name: statistics.median(b - a for a, b in zip(times, times[1:]))
                for name, times in baseline_times.items() if len(times) >= 2}
            if float(baseline["host_cpu_usage_pct_mean"]) > self.config["max_baseline_cpu_pct"]:
                raise RuntimeError("Host is already busy; do not add load on top of unapproved work")
            memory = memory_info()
            self.reserve = int(max(self.config["memory_reserve_bytes"], memory["EffectiveTotal"] * self.config["memory_reserve_fraction"]))
            initial_safety = safety_snapshot()
            self.record["safety_samples"].append(initial_safety)
            if initial_safety["memory_available_bytes"] <= self.reserve:
                raise RuntimeError("Insufficient free memory for safe benchmark headroom")
            self.enforce_temperature_limit(initial_safety)
            self.started = time.monotonic()
            self.record["load_started_at"] = time.time()
            self.record["memory_before"] = memory
            command = []
            if self.config["load_type"] == "cpu":
                executable = shutil.which(self.config["stress_ng"])
                if not executable:
                    raise RuntimeError("Install stress-ng on Linux (sudo apt install stress-ng)")
                host_cpus = os.cpu_count() or 1
                allowed = sorted(os.sched_getaffinity(0))
                quota = cpu_quota_cores()
                if quota is not None and quota < len(allowed) - .01:
                    raise RuntimeError("CPU quota limits effective capacity; host-wide targets are not reliable")
                baseline_cpu = float(baseline["host_cpu_usage_pct_mean"])
                duty = 100 * (100 * self.config["target_cpu_fraction"] - baseline_cpu) / (100 - baseline_cpu) * host_cpus / len(allowed)
                if not 0 < duty <= 100 * self.config["max_cpu_fraction"]:
                    raise RuntimeError("Requested host CPU target cannot be met safely within this CPU affinity")
                command = [executable, "--cpu", str(len(allowed)), "--cpu-load", str(round(duty)),
                    "--cpu-method", "sqrt", "--timeout", f"{self.config['max_duration_seconds']}s", "--metrics-brief"]
                self.record["cpu_control"] = {"host_logical_processors": host_cpus, "affinity_cpus": allowed,
                    "worker_duty_pct": round(duty), "target_is_not_an_observation": True}
            elif self.config["load_type"] == "memory":
                amount, self.reserve, budget = allocation_bytes(self.config, memory)
                ready = self.path.with_suffix(".ready.json")
                command = [sys.executable, str(Path(__file__).with_name("resource_load_memory_worker.py")),
                    "--bytes", str(amount), "--timeout", str(self.config["max_duration_seconds"]), "--ready", str(ready)]
                self.record["memory_control"] = {"allocation_bytes": amount, "safe_budget_bytes": budget,
                    "reserve_bytes": self.reserve, "access_pattern": "touch every page once, then sparse periodic access; not a bandwidth stressor"}
                self.ready_path = ready
            self.record["command"] = command
            if command:
                self.log = self.path.with_suffix(".log").open("wb")
                supervised = [sys.executable, str(Path(__file__).with_name("resource_load_supervisor.py")),
                    "--parent", str(os.getpid()), "--timeout", str(self.config["max_duration_seconds"]), "--", *command]
                self.record["supervised_command"] = supervised
                self.process = subprocess.Popen(supervised, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
                self.record["owned_process_group"] = self.process.pid
            if threading.current_thread() is threading.main_thread():
                for signum in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
                    self.previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, self.interrupted)
            self.thread = threading.Thread(target=self.watchdog, daemon=True)
            self.thread.start()
            self.stabilize(baseline)
            self.record["stable_at"] = time.time()
            self.record["status"] = "STABLE"
            self.save()
            return self
        except BaseException:
            self.record["status"] = "FAIL"
            self.record["error"] = str(sys.exc_info()[1])
            self.cleanup()
            self.save()
            raise

    def interrupted(self, signum, frame):
        raise KeyboardInterrupt(f"Load experiment interrupted by signal {signum}")

    def stabilize(self, baseline):
        deadline = time.monotonic() + self.config["stabilization_timeout_seconds"]
        consecutive = 0
        self.wait(self.window + 2 * self.interval)
        while time.monotonic() < deadline:
            self.check()
            end = time.time() - self.interval
            try:
                metrics, times = self.observation(end - self.window, end)
                cpu = float(metrics["host_cpu_usage_pct_mean"])
                stable = float(metrics["host_cpu_usage_pct_max"]) - cpu <= self.config["cpu_tolerance_pct"]
                if self.config["load_type"] == "cpu":
                    stable &= abs(cpu - 100 * self.config["target_cpu_fraction"]) <= self.config["cpu_tolerance_pct"]
                else:
                    stable &= cpu <= self.config["max_baseline_cpu_pct"]
                if self.config["load_type"] == "memory":
                    if not self.ready_path.exists():
                        stable = False
                    else:
                        ready = json.loads(self.ready_path.read_text())
                        self.record["memory_worker_ready"] = ready
                        growth = float(metrics["host_memory_used_bytes_mean"]) - float(baseline["host_memory_used_bytes_mean"])
                        stable &= growth >= .7 * self.record["memory_control"]["allocation_bytes"]
                self.record["last_stability_metrics"] = metrics
                self.record["last_stability_scrapes"] = times
                consecutive = consecutive + 1 if stable else 0
                if consecutive >= 3:
                    return
            except (OSError, RuntimeError, ValueError) as exc:
                self.record["last_stability_error"] = str(exc)
                consecutive = 0
            self.wait(max(1.0, self.interval * 2))
        raise RuntimeError("Background load did not reach a stable measured target before timeout")

    def before_window(self, seconds):
        self.check()
        start = time.time()
        self.wait(seconds)
        return start, time.time()

    def before_invocation(self, phase, index):
        self.check()
        self.record.setdefault("target_first_invocation_at", time.time())

    def record_windows(self, windows):
        self.check()
        if windows["before"][0] < self.record["stable_at"] or not (
            windows["before"][1] <= windows["during"][0] <= windows["during"][1] <= windows["after"][0]):
            raise RuntimeError("Invalid load/target/phase ordering")
        self.last_windows = windows
        self.record["phase_windows"] = windows

    def validate_phase(self, phase, start, end):
        try:
            times = self.checked_scrapes(start, end)
            self.record["phase_quality"][phase] = {"status": "PASS", "distinct_raw_scrapes": {key: len(v) for key, v in times.items()},
                "scrape_timestamps": times}
        except Exception as exc:
            if isinstance(exc, InsufficientScrapes) and phase == "during" and short_during_allowed(self.record):
                self.record["phase_quality"][phase] = {"status": "MISSING_SHORT_WINDOW",
                    "error": "Short bounded measurement window: during metrics unavailable, not zero or padded",
                    "distinct_raw_scrapes": {key: len(v) for key, v in exc.times.items()}, "scrape_timestamps": exc.times}
            else:
                self.record["phase_quality"][phase] = {"status": "FAIL", "error": str(exc)}
            self.save()
            raise
        self.save()

    def stop_owned_process(self):
        process = self.process
        if process is None:
            return
        # Every worker is created in a new session; never signal a name or a foreign group.
        for signum in [signal.SIGTERM, signal.SIGKILL]:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=self.config["cleanup_timeout_seconds"])
            except subprocess.TimeoutExpired:
                continue
            if signum == signal.SIGTERM:
                # The leader may exit before a descendant. A final group signal handles survivors.
                continue
        try:
            process.wait(timeout=self.config["cleanup_timeout_seconds"])
        except subprocess.TimeoutExpired:
            self.error = "Owned background process could not be reaped"

    def cleanup(self):
        self.stop_event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        self.stop_owned_process()
        for signum, handler in self.previous_handlers.items():
            signal.signal(signum, handler)
        if self.log is not None:
            self.log.close()
        self.record["load_stopped_at"] = time.time()
        residual = []
        if self.process is not None and platform.system() == "Linux":
            for path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    fields = path.read_text().rsplit(")", 1)[1].split()
                    if int(fields[3]) == self.process.pid and fields[0] != "Z":
                        residual.append(int(path.parent.name))
                except (OSError, IndexError, ValueError):
                    pass
        self.record["residual_owned_group_pids"] = residual
        self.record["cleanup_success"] = (self.process is None or self.process.poll() is not None) and not residual
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if platform.system() == "Linux":
            try:
                self.record["post_cleanup_safety"] = safety_snapshot()
            except (OSError, ValueError) as exc:
                self.record["post_cleanup_safety_error"] = str(exc)

    def __exit__(self, kind, value, trace):
        self.record["status"] = "MEASUREMENT_COMPLETED" if kind is None and self.last_windows else "FAIL"
        if value is not None:
            self.record["error"] = str(value)
        self.cleanup()
        self.save()
        if kind is None and not self.record["cleanup_success"]:
            raise RuntimeError("Load process cleanup failed")
        return False


def session(args, core, identity):
    if getattr(args, "load_config", None) is None:
        return nullcontext(None)
    load = LoadSession(args, core, identity)
    args._load_sessions.append(load)
    return load


def phase_guard(args, program_id, input_id):
    if getattr(args, "load_config", None) is None:
        return None
    matches = [s for s in args._load_sessions if s.identity["program_id"] == program_id and s.identity["input_id"] == input_id]
    if len(matches) != 1:
        raise RuntimeError("Ambiguous load-to-configuration association")
    return matches[0].validate_phase


def finalize(args, summary_path, runs_path, summary_rows, run_rows):
    if getattr(args, "load_config", None) is not None and len(getattr(args, "_load_sessions", [])) != len(summary_rows):
        raise RuntimeError("Load experiment contains a configuration with no completed load session")
    output_hashes = {}
    if getattr(args, "_load_sessions", []):
        for label, path in [("summary_sha256", summary_path), ("runs_sha256", runs_path)]:
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            output_hashes[label] = digest.hexdigest()
    for load in getattr(args, "_load_sessions", []):
        key = load.identity
        matched = [r for r in summary_rows if r["program_id"] == key["program_id"] and r["input_id"] == key["input_id"]]
        if len(matched) != 1:
            raise RuntimeError("Load record cannot uniquely identify a summary row")
        row = matched[0]
        runs = [r for r in run_rows if r["program_id"] == key["program_id"] and r["input_id"] == key["input_id"]]
        ids = [r["run_id"] for r in runs]
        load.record["outputs"] = {"summary_file": str(Path(summary_path).resolve()), "runs_file": str(Path(runs_path).resolve()),
            **output_hashes,
            "summary_row_identity": {k: row[k] for k in ["program_id", "input_id", "env_id", "session_id"] if k in row},
            "run_id_count": len(ids), "first_run_id": ids[0] if ids else None, "last_run_id": ids[-1] if ids else None,
            "run_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        load.record["measured_summary"] = {k: v for k, v in row.items() if k.startswith(("host_", "prometheus_", "gpu_", "runtime_")) or k in {"run_success", "measured_runs"}}
        phases_ok = monitoring_accepted(load.record, row)
        load.record["monitoring_acceptance"] = ("COMPLETE_THREE_PHASE" if row.get("prometheus_three_phase_collection_success") is True
                                                else "SHORT_DURING_MISSING" if phases_ok else "FAIL")
        actual_cpu = row.get("host_cpu_usage_pct_before_mean", "")
        if load.config["load_type"] == "cpu":
            control_ok = isinstance(actual_cpu, (int, float)) and abs(actual_cpu - 100 * load.config["target_cpu_fraction"]) <= load.config["cpu_tolerance_pct"]
        else:
            control_ok = isinstance(actual_cpu, (int, float)) and actual_cpu <= load.config["max_baseline_cpu_pct"]
        if load.config["load_type"] == "memory":
            actual_memory = row.get("host_memory_used_bytes_before_mean", "")
            control_ok &= isinstance(actual_memory, (int, float)) and (actual_memory - load.record["baseline_metrics"]["host_memory_used_bytes_mean"] >=
                .7 * load.record["memory_control"]["allocation_bytes"])
        load.record["before_load_control_validated"] = bool(control_ok)
        measured_count = sum(r.get("phase") == "measure" and r.get("success") is True for r in runs)
        enough_runs = measured_count >= load.config["minimum_measured_runs"] and (not load.config["maximum_measured_runs"] or measured_count <= load.config["maximum_measured_runs"])
        gpu_ok = getattr(args, "gpu_metrics", "off") == "off" or row.get("gpu_three_phase_collection_success") is True
        load.record["status"] = "PASS" if (phases_ok and control_ok and enough_runs and gpu_ok and
            row.get("run_success") is True and load.record.get("cleanup_success")) else "FAIL"
        load.save()
    if any(load.record["status"] != "PASS" for load in getattr(args, "_load_sessions", [])):
        raise RuntimeError("Load/monitoring validation failed; results preserved but not accepted as a valid load experiment")
