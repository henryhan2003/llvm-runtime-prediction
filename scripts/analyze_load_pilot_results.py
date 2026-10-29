"""Read-only offline analysis of a downloaded load pilot; no pipeline imports or execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics

try:
    import polybench_preparation_cache as preparation_cache
except ImportError:
    from scripts import polybench_preparation_cache as preparation_cache


LEVELS = ["CPU_IDLE", "CPU_LOW", "CPU_MEDIUM", "CPU_HIGH"]
SIZES = ["MINI_DATASET", "MEDIUM_DATASET", "EXTRALARGE_DATASET"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def number(value):
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite observation")
    return value


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def describe(values):
    values = [v for v in values if v is not None]
    return {"count": len(values), "min": min(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "max": max(values) if values else None}


def observed_true(value):
    return str(value).lower() in {"true", "1"}


def local_artifact(root, trial, remote_path, manifest):
    remote = PurePosixPath(remote_path)
    try:
        relative = remote.relative_to(PurePosixPath(trial["results_dir"]))
        local = root / trial["trial_id"] / "results" / str(relative)
    except ValueError:
        cache_root = manifest.get("preparation_cache", {}).get("directory")
        if not cache_root:
            raise ValueError("Artifact is not inside a declared result/cache directory")
        relative = remote.relative_to(PurePosixPath(cache_root))
        local = root / "preparation_cache" / str(relative)
    local = local.resolve()
    if not local.is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes the downloaded source root")
    return local


def within(samples, window):
    return [s for s in samples if window[0] <= s["timestamp"] <= window[1]]


def telemetry(samples):
    temperatures = [s for s in samples if number(s.get("temperature_max_celsius")) is not None]
    peak = max(temperatures, key=lambda s: s["temperature_max_celsius"]) if temperatures else {}
    return {"samples": len(samples),
            "frequency_mhz": describe([number(s.get("cpu_frequency_mean_mhz")) for s in samples]),
            "peak_temperature_celsius": peak.get("temperature_max_celsius"),
            "peak_temperature_source": peak.get("temperature_max_source"),
            "observations_ge_85_celsius": sum(s["temperature_max_celsius"] >= 85 for s in temperatures)}


def table(headers, data):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(cell) for cell in row) + " |" for row in data]]


def fmt(value, decimals=3):
    return f"{value:.{decimals}f}" if value is not None else "not recorded"


def analyze(root):
    manifest = read_json(root / "experiment_manifest.json")
    server_report = read_json(root / "validation_report.json")
    comparisons = {r["trial_id"]: r for r in rows(root / "load_experiment_comparisons.csv")}
    checks, results, phase_rows = [], [], []
    all_ids = []
    static_by_input, executable_by_input, environments = {}, {}, []

    def check(trial, name, passed):
        checks.append({"trial_id": trial, "check": name, "passed": bool(passed)})

    check("all", "server_revalidation_passed", server_report["status"] == "PASS" and not server_report["errors"])
    check("all", "pilot_expected_24_trials", manifest["stage"] == "pilot" and len(manifest["trials"]) == 24)
    check("all", "comparison_rows_cover_trials", set(comparisons) == {t["trial_id"] for t in manifest["trials"]})
    for order, trial in enumerate(manifest["trials"], 1):
        trial_id = trial["trial_id"]
        if Path(trial_id).name != trial_id or trial_id in {".", ".."}:
            raise ValueError("Invalid trial directory")
        directory = root / trial_id
        config_file = directory / "load_config.json"
        supplied_config = read_json(config_file)
        sidecars = [p for p in (directory / "results/load_records").glob("*.json") if not p.name.endswith(".ready.json")]
        if len(sidecars) != 1:
            raise ValueError(f"{trial_id}: expected one sidecar")
        record = read_json(sidecars[0])
        config = record["config"]
        files = {kind: directory / "results" / PurePosixPath(record["outputs"][kind + "_file"]).name
                 for kind in ["summary", "runs"]}
        summary_rows = rows(files["summary"])
        if len(summary_rows) != 1:
            raise ValueError(f"{trial_id}: expected one summary row")
        summary = summary_rows[0]
        static = rows(directory / "results/program_static_features.csv")[0]
        static_values = {k: v for k, v in static.items() if k not in {"ast_path", "ir_path", "cfg_path", "exe_path"}}
        key = (trial["program"], trial["input_profile"])
        check(trial_id, "static_fields_identical_across_load_levels", static_by_input.setdefault(key, static_values) == static_values)
        check(trial_id, "static_extraction_and_build_success", all(observed_true(static[k]) for k in ["ast_success", "cfg_success", "ir_success", "build_success"]))
        executable_hash = digest(local_artifact(root, trial, static["exe_path"], manifest))
        check(trial_id, "same_executable_for_same_program_input", executable_by_input.setdefault(key, executable_hash) == executable_hash)
        if record.get("preparation"):
            metadata = record["preparation"]
            receipt = local_artifact(root, trial, metadata["manifest_file"], manifest)
            check(trial_id, "preparation_receipt_sha256", digest(receipt) == metadata["manifest_sha256"])
            _, verified = preparation_cache.read_verified(receipt.parent, metadata["key"])
            check(trial_id, "preparation_artifacts_and_identity", verified["executable_sha256"] == executable_hash and
                  verified["program_id"] == summary["program_id"] and verified["input_profile"] == summary["input_id"])
        environment = read_json(directory / "results/environment_manifest.json")
        environments.append({k: environment.get(k) for k in ["cpu_model", "cpu_physical_core_count", "cpu_logical_core_count",
                            "run_c_compiler_version", "run_cxx_compiler_version", "opt_level", "warmup_runs",
                            "measurement_seconds", "min_measured_runs", "timeout_sec", "prometheus_context_seconds",
                            "prometheus_query_step_sec", "prometheus_job", "prometheus_instance", "cpu_runtime_parameter_schema"]})
        runs = rows(files["runs"])
        measured = [r for r in runs if r["phase"] == "measure" and observed_true(r["success"])]
        values = [float(r["runtime_sec"]) for r in measured]
        ids = [r["run_id"] for r in runs]
        all_ids.extend(ids)
        check(trial_id, "trial_completed", trial["status"] == "COMPLETED" and trial["exit_code"] == 0)
        check(trial_id, "config_sha256", digest(config_file) == trial["config_sha256"])
        check(trial_id, "config_matches_sidecar", all(record["config"].get(k) == v for k, v in supplied_config.items()))
        for kind, path in files.items():
            check(trial_id, kind + "_sha256", digest(path) == record["outputs"][kind + "_sha256"])
        check(trial_id, "sidecar_and_cleanup_passed", record["status"] == "PASS" and record["cleanup_success"] and record["before_load_control_validated"])
        check(trial_id, "identity", summary["program_id"] == record["identity"]["program_id"] and
              summary["input_id"] == trial["input_profile"] == record["identity"]["input_id"] and
              PurePosixPath(summary["program_id"]).stem == trial["program"])
        check(trial_id, "run_ids", len(ids) == len(set(ids)) == record["outputs"]["run_id_count"] and
              hashlib.sha256(json.dumps(ids).encode()).hexdigest() == record["outputs"]["run_ids_sha256"])
        check(trial_id, "all_invocations_succeeded", all(observed_true(r["success"]) and int(r["return_code"]) == 0 for r in runs))
        check(trial_id, "process_and_io_metrics_success", all(observed_true(r["process_metrics_success"]) and
              r["process_io_collection_status"] == "ok" for r in runs))
        check(trial_id, "measured_count_and_positive_runtime", len(values) >= config.get("minimum_measured_runs", 3) and
              (not config.get("maximum_measured_runs", 0) or len(values) <= config["maximum_measured_runs"]) and
              len(values) == int(summary["measured_runs"]) and all(math.isfinite(v) and v > 0 for v in values))
        median = statistics.median(values)
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.
        cv = sd / mean
        check(trial_id, "recomputed_runtime_statistics", all(math.isclose(observed, float(summary[key]), rel_tol=1e-8, abs_tol=1e-10)
              for key, observed in [("runtime_sec_median", median), ("runtime_sec_mean", mean), ("runtime_cv", cv)]))
        check(trial_id, "comparison_runtime_matches_raw", math.isclose(median, float(comparisons[trial_id]["runtime_sec_median"]), rel_tol=1e-8))
        windows = record["phase_windows"]
        check(trial_id, "phase_order", record["stable_at"] <= windows["before"][0] <= windows["before"][1] <=
              record["target_first_invocation_at"] <= windows["during"][0] <= windows["during"][1] <=
              windows["after"][0] <= windows["after"][1] <= record["load_stopped_at"])
        cpu_before = float(summary["host_cpu_usage_pct_before_mean"])
        target = 100 * config["target_cpu_fraction"]
        check(trial_id, "measured_background_cpu_target", abs(cpu_before - target) <= config["cpu_tolerance_pct"]
              if config["load_type"] == "cpu" else cpu_before <= config["max_baseline_cpu_pct"])
        phases = {}
        for phase, window in windows.items():
            quality = record["phase_quality"][phase]
            short_missing = (config.get("collection_protocol", "strict-v1") == "fast-v2" and phase == "during" and
                             quality["status"] == "MISSING_SHORT_WINDOW" and
                             0 <= window[1] - window[0] < (config["min_raw_scrapes"] + 1) * record["monitoring_preflight"]["scrape_interval_seconds"])
            valid = (quality["status"] == "PASS" and observed_true(summary[f"prometheus_{phase}_collection_success"])) or short_missing
            if short_missing:
                valid &= not observed_true(summary[f"prometheus_{phase}_collection_success"]) and not observed_true(summary["prometheus_three_phase_collection_success"])
                valid &= all(summary.get(f"host_{metric}_during_{suffix}", "") == "" for metric, suffix in [
                    ("cpu_usage_pct", "mean"), ("cpu_usage_pct", "max"), ("memory_used_bytes", "mean"), ("memory_used_bytes", "max"),
                    ("disk_read_bytes", "delta"), ("disk_write_bytes", "delta"), ("network_bytes", "delta")])
            counts = []
            for metric, timestamps in quality["scrape_timestamps"].items():
                unique = {t for t in timestamps if math.isfinite(t) and window[0] <= t <= window[1]}
                counts.append(len(unique))
                valid &= len(unique) == quality["distinct_raw_scrapes"][metric] and (short_missing or len(unique) >= config["min_raw_scrapes"])
            if short_missing:
                valid &= min(counts) < config["min_raw_scrapes"]
            valid &= all(math.isclose(float(summary[f"prometheus_{phase}_window_{edge}_timestamp"]), t, rel_tol=0, abs_tol=1e-6)
                         for edge, t in zip(["start", "end"], window))
            check(trial_id, phase + "_scrapes_and_window", valid and bool(counts))
            phases[phase] = telemetry(within(record["safety_samples"], window))
            phases[phase]["min_distinct_raw_scrapes"] = min(counts) if counts else None
            phase_rows.append({"trial_id": trial_id, "program": trial["program"], "input_id": trial["input_profile"],
                               "profile_id": trial["profile_id"], "phase": phase, "cpu_mean_pct": float(summary[f"host_cpu_usage_pct_{phase}_mean"]),
                               "duration_sec": window[1] - window[0], **phases[phase]})
        cpu_ratios = [number(r.get("process_cpu_total_sec")) / float(r["runtime_sec"]) for r in measured
                      if number(r.get("process_cpu_total_sec")) is not None]
        chunk = len(values) // 3
        results.append({"trial_id": trial_id, "execution_order": order, "program": trial["program"], "program_id": summary["program_id"],
                        "input_id": summary["input_id"], "profile_id": trial["profile_id"], "target_cpu_pct": target,
                        "cpu_before_pct": cpu_before, "cpu_during_pct": float(summary["host_cpu_usage_pct_during_mean"]),
                        "cpu_after_pct": float(summary["host_cpu_usage_pct_after_mean"]),
                        "measured_runs": len(values), "warmup_runs": sum(r["phase"] == "warmup" for r in runs),
                        "runtime_sec_median": median, "runtime_sec_mean": mean, "runtime_cv_pct": cv * 100,
                        "runtime_p10_sec": percentile(values, .1), "runtime_p90_sec": percentile(values, .9),
                        "runtime_min_sec": min(values), "runtime_max_sec": max(values),
                        "last_to_first_third_median_ratio": statistics.median(values[-chunk:]) / statistics.median(values[:chunk]) if len(values) >= 6 else None,
                        "process_cpu_to_wall_median_ratio": statistics.median(cpu_ratios) if cpu_ratios else None,
                        "before_frequency_mhz": phases["before"]["frequency_mhz"]["mean"],
                        "during_frequency_mhz": phases["during"]["frequency_mhz"]["mean"],
                        "after_frequency_mhz": phases["after"]["frequency_mhz"]["mean"],
                        "peak_temperature_celsius": telemetry(record["safety_samples"])["peak_temperature_celsius"],
                        "peak_temperature_source": telemetry(record["safety_samples"])["peak_temperature_source"],
                        "during_peak_temperature_celsius": phases["during"]["peak_temperature_celsius"],
                        "observations_ge_85_celsius": telemetry(record["safety_samples"])["observations_ge_85_celsius"],
                        "duration_sec": trial["finished_at"] - trial["started_at"],
                        "summary_file": str(files["summary"].resolve()), "runs_file": str(files["runs"].resolve()),
                        "sidecar_file": str(sidecars[0].resolve()), "temperature_policy": config["temperature_policy"]})
        results[-1]["collection_protocol"] = config.get("collection_protocol", "strict-v1")
        results[-1]["during_monitoring_status"] = record["phase_quality"]["during"]["status"]
    check("all", "globally_unique_run_ids", len(all_ids) == len(set(all_ids)))
    check("all", "hardware_compiler_and_timing_config_unchanged", all(e == environments[0] for e in environments))
    for item in results:
        idle = next(r for r in results if r["program"] == item["program"] and r["input_id"] == item["input_id"] and r["profile_id"] == "CPU_IDLE")
        item["runtime_ratio_to_idle"] = item["runtime_sec_median"] / idle["runtime_sec_median"]
        item["runtime_change_vs_idle_pct"] = (item["runtime_ratio_to_idle"] - 1) * 100
        check(item["trial_id"], "comparison_ratio_matches_raw", math.isclose(item["runtime_ratio_to_idle"],
              float(comparisons[item["trial_id"]]["runtime_ratio_to_idle_ANALYSIS_ONLY"]), rel_tol=1e-8))
    summary = {"offline_data_checks": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks,
               "source_root": str(root), "original_manifest_status": manifest["status"], "server_validation_status": server_report["status"],
               "source_code_snapshot_available": False, "server_code_hashes_revalidated_offline": False,
               "trial_count": len(results), "run_record_count": len(all_ids), "measured_run_count": sum(r["measured_runs"] for r in results),
               "collection_protocols": sorted({r["collection_protocol"] for r in results}),
               "complete_three_phase_configurations": sum(r["during_monitoring_status"] == "PASS" for r in results),
               "short_during_missing_configurations": sum(r["during_monitoring_status"] == "MISSING_SHORT_WINDOW" for r in results),
               "warmup_count": sum(r["warmup_runs"] for r in results),
               "wall_elapsed_seconds": max(t["finished_at"] for t in manifest["trials"]) - min(t["started_at"] for t in manifest["trials"]),
               "independent_load_blocks_per_program_size_level": 1, "statistical_significance_claimed": False,
               "new_load_or_target_execution": False, "model_training": False,
               "shared_environment": environments[0],
               "load_levels": {level: {key: describe([r[key] for r in results if r["profile_id"] == level])
                                      for key in ["cpu_before_pct", "before_frequency_mhz", "during_frequency_mhz", "runtime_cv_pct"]} for level in LEVELS},
               "phase_telemetry": phase_rows,
               "missing_observations": {key: sum(r[key] is None for r in results) for key in
                                        ["before_frequency_mhz", "during_frequency_mhz", "after_frequency_mhz", "peak_temperature_celsius"]}}
    return summary, results


def report_lines(summary, results):
    lines = ["# CPU Load Pilot Analysis", "", f"Source: `{summary['source_root']}`.", "",
             f"Offline data checks: **{summary['offline_data_checks']}**. {summary['trial_count']} configurations; "
             f"{summary['measured_run_count']:,} measured invocations and {summary['warmup_count']} warmups. "
             f"Elapsed experiment span: {summary['wall_elapsed_seconds'] / 60:.2f} minutes.", "",
             "The preserved manifest says FAIL because of the previously diagnosed launcher state bug; the independent server report says PASS. "
             "This analysis does not rewrite either. Downloaded configuration and collected summary/run file hashes are checked against their recorded hashes. "
             "Static fields (excluding artifact paths) and executable hashes are identical across load levels for each program/input pair. "
             "Historical server source files were not archived in this download, so their code hashes are not revalidated against a different local checkout.", "",
             "## Measured Background CPU Load", ""]
    lines += table(["Level", "Before CPU mean (%)", "Across-trial range (%)", "Before host frequency (MHz)", "During host frequency (MHz)"],
                   [[level, fmt(v["cpu_before_pct"]["mean"], 2), f"{v['cpu_before_pct']['min']:.2f} to {v['cpu_before_pct']['max']:.2f}",
                     fmt(v["before_frequency_mhz"]["mean"], 0), fmt(v["during_frequency_mhz"]["mean"], 0)]
                    for level, v in summary["load_levels"].items()])
    lines += ["", "Host frequency is an unweighted mean across readable logical-CPU scaling_cur_freq files, sampled at about 1 Hz, "
              "not the actual frequency of the core running the target. It cannot certify the absence of thermal throttling.", "",
              "## Runtime By Program And Size", "", "Seconds are medians of successful measured invocations. Ratios use the IDLE median from the same program and size; values below 1 mean faster than IDLE.", ""]
    groups = sorted({(r["program"], r["input_id"]) for r in results}, key=lambda k: (k[0], SIZES.index(k[1])))
    data = []
    for program, size in groups:
        group = {r["profile_id"]: r for r in results if (r["program"], r["input_id"]) == (program, size)}
        data.append([program, size, *[fmt(group[level]["runtime_sec_median"], 6) for level in LEVELS],
                     *[fmt(group[level]["runtime_ratio_to_idle"], 3) for level in LEVELS[1:]]])
    lines += table(["Program", "Size", "IDLE (s)", "LOW (s)", "MEDIUM (s)", "HIGH (s)", "LOW/IDLE", "MEDIUM/IDLE", "HIGH/IDLE"], data)
    lines += ["", "## Repetition And Telemetry", ""]
    lines += table(["Program", "Size", "Level", "Measured n", "Within-block CV (%)", "CPU-time / wall-time", "Peak temperature (C)", "Peak sensor"],
                   [[r["program"], r["input_id"], r["profile_id"], r["measured_runs"], fmt(r["runtime_cv_pct"], 2),
                     fmt(r["process_cpu_to_wall_median_ratio"]), fmt(r["peak_temperature_celsius"], 2),
                     " / ".join(str((r["peak_temperature_source"] or {}).get(k, "unknown")) for k in ["device", "label"])] for r in results])
    minima = {phase: min(r["min_distinct_raw_scrapes"] for r in summary["phase_telemetry"] if r["phase"] == phase)
              for phase in ["before", "during", "after"]}
    lines += ["", f"Minimum distinct in-window raw scrapes by phase: `{minima}`. All phases are group-of-invocations windows, not individual short-invocation observations.",
              f"Complete three-phase configurations: {summary['complete_three_phase_configurations']}; explicitly unavailable short during windows: {summary['short_during_missing_configurations']}. "
              "fast-v2 accepts that documented missingness, without calling it complete monitoring or substituting zeros.", "",
              "## Interpretation And Limits", "",
              "1. Large-input slowdowns and short-input speedups coexist. Do not require a monotone slowdown versus whole-host CPU utilization or delete the faster-under-load observations.",
              "2. Higher host-average frequency under pressure supports a power-management explanation for short-input speedups, but does not prove causality. "
              "Target-core frequency, idle-state residency, placement/migration and scheduler/performance counters were not collected. "
              "The recorded timing includes process launch, loader and collection callback overhead, rather than kernel-only compute time.",
              "3. Each program/size/level has only ONE independently established load block. Thousands of invocations within a block are not thousands of independent load experiments. "
              "With very few measured invocations (fast-v2 permits two), no significance or reliable between-block variance is established. "
              "The median of two observations is their arithmetic mean.",
              "4. The pressure method is stress-ng sqrt across 64 logical CPUs, with duty-cycle levels. Equal host CPU percentages do not imply equal target-core, SMT, cache or memory-bandwidth contention; "
              "this CPU-compute pilot cannot validate a memory-capacity experiment or a different pressure mix.",
              "5. Before-state CPU observations are future prediction candidates after semantic review. During/after telemetry, target-process resource counters, "
              "trial labels and derived runtime ratios are analysis/supervision data, not same-run pre-execution model inputs.",
              "6. Preserve this pilot as valid collection evidence. Before treating a 20-size batch as a controlled resource-competition study, "
              "first choose whether to model the naturally adaptive system or deliberately control frequency/placement with administrator approval. "
              "Repeated interleaved load blocks are needed to estimate between-block variability. No new experiment, system setting or model is launched by this analysis.", "",
              "Mechanism reference: [Linux CPU performance scaling](https://docs.kernel.org/admin-guide/pm/cpufreq.html) explains utilization-dependent P-states "
              "and the limitations of scaling_cur_freq. [Linux CPU idle management](https://docs.kernel.org/admin-guide/pm/cpuidle.html) explains idle-state exit latency. "
              "These support plausible mechanisms, not a measured causal attribution for this pilot.", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if output == source or output.is_relative_to(source):
        parser.error("Analysis output must be outside the source directory")
    if output.exists():
        parser.error("Use a fresh analysis output directory")
    before = snapshot(source)
    summary, results = analyze(source)
    output.mkdir(parents=True)
    fields = list(results[0])
    with (output / "trial_analysis.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v for k, v in item.items()})
    after = snapshot(source)
    summary["source_files_sha256"] = before
    summary["source_files_unchanged"] = before == after
    summary["source_file_count"] = len(before)
    if before != after:
        summary["offline_data_checks"] = "FAIL"
    (output / "pilot_analysis_report.md").write_text("\n".join(report_lines(summary, results)), encoding="utf-8")
    (output / "analysis_audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_files_sha256", "checks", "phase_telemetry", "shared_environment"}}, indent=2))
    return 0 if summary["offline_data_checks"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
