"""Validate collected load trials, preserving raw CSVs and original field names."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics

try:
    import resource_load as load
except ImportError:
    from scripts import resource_load as load


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def true(value):
    return str(value).lower() in {"true", "1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_trial(trial):
    require(trial["status"] == "COMPLETED", "Trial was not completed")
    require(hashlib.sha256(Path(trial["config_path"]).read_bytes()).hexdigest() == trial["config_sha256"], "Load configuration changed")
    sidecars = [json.loads(p.read_text(encoding="utf-8")) for p in (Path(trial["results_dir"]) / "load_records").glob("*.json")
                if not p.name.endswith(".ready.json")]
    require(len(sidecars) == 1, "Expected exactly one load sidecar per isolated configuration")
    record = sidecars[0]
    require(record["config"] == load.read_config(trial["config_path"]), "Sidecar control configuration mismatch")
    require(record["status"] == "PASS" and record["cleanup_success"] and record["before_load_control_validated"], "Load/monitoring/cleanup quality failed")
    require(record["config"]["profile_id"] == trial["profile_id"], "Profile identity mismatch")
    source = rows(record["outputs"]["summary_file"])
    for kind in ["summary", "runs"]:
        if kind + "_sha256" in record["outputs"]:
            require(hashlib.sha256(Path(record["outputs"][kind + "_file"]).read_bytes()).hexdigest() == record["outputs"][kind + "_sha256"],
                    "Collected CSV hash changed")
    require(len(source) == 1, "Expected one configuration in trial summary")
    row = source[0]
    identity = record["identity"]
    require(row["program_id"] == identity["program_id"] and row["input_id"] == identity["input_id"], "Summary identity mismatch")
    require((row["program_id"] == trial["program"] or Path(row["program_id"]).stem == trial["program"]) and
            row["input_id"] == trial["input_profile"], "Manifest program/size differs from collected row")
    require(true(row["run_success"]) and true(row["prometheus_three_phase_collection_success"]), "Collection failed")
    details = rows(record["outputs"]["runs_file"])
    ids = [r["run_id"] for r in details]
    require(len(set(ids)) == len(ids) and len(ids) == record["outputs"]["run_id_count"], "Run IDs not unique/complete")
    require(hashlib.sha256(json.dumps(ids).encode()).hexdigest() == record["outputs"]["run_ids_sha256"], "Run IDs changed")
    measured = [r for r in details if r["phase"] == "measure" and true(r["success"])]
    require(len(measured) >= 3 and int(row["measured_runs"]) == len(measured), "At least three valid measurements required")
    runtimes = [float(r["runtime_sec"]) for r in measured]
    require(all(math.isfinite(t) and t > 0 for t in runtimes), "Invalid runtime")
    require(math.isclose(statistics.median(runtimes), float(row["runtime_sec_median"]), rel_tol=1e-8, abs_tol=1e-10), "Runtime median mismatch")
    for phase in ["before", "during", "after"]:
        quality = record["phase_quality"][phase]
        require(quality["status"] == "PASS" and min(quality["distinct_raw_scrapes"].values()) >= record["config"]["min_raw_scrapes"], "Insufficient raw scrapes")
        require(true(row[f"prometheus_{phase}_collection_success"]), "Phase monitoring failed")
        for edge, timestamp in zip(["start", "end"], record["phase_windows"][phase]):
            require(math.isclose(float(row[f"prometheus_{phase}_window_{edge}_timestamp"]), timestamp, abs_tol=1e-6, rel_tol=0), "Phase window association changed")
        for metric, timestamps in quality["scrape_timestamps"].items():
            start, end = record["phase_windows"][phase]
            unique = {t for t in timestamps if math.isfinite(t) and start <= t <= end}
            require(len(unique) == quality["distinct_raw_scrapes"][metric] and len(unique) >= record["config"]["min_raw_scrapes"],
                    "Raw scrape counts/timestamps disagree")
    windows = record["phase_windows"]
    require(record["stable_at"] <= windows["before"][0] <= windows["before"][1] <= record["target_first_invocation_at"] <= windows["during"][0], "Before phase was not before target startup")
    require(record["load_stopped_at"] >= windows["after"][1], "Load was released before after phase ended")
    cpu = float(row["host_cpu_usage_pct_before_mean"])
    if record["config"]["load_type"] == "cpu":
        require(abs(cpu - 100 * record["config"]["target_cpu_fraction"]) <= record["config"]["cpu_tolerance_pct"], "Measured CPU target missed")
    else:
        require(cpu <= record["config"]["max_baseline_cpu_pct"], "Unexpected CPU competition in capacity/idle experiment")
    if "gpu_three_phase_collection_success" in row:
        require(true(row["gpu_three_phase_collection_success"]), "CUDA GPU phases incomplete")
    result = {"trial_id": trial["trial_id"], "load_family": trial["load_family"], "profile_id": trial["profile_id"],
              "program_id": row["program_id"], "input_id": row["input_id"], "env_id": row["env_id"],
              "measured_runs": len(measured), "runtime_sec_median": float(row["runtime_sec_median"]),
              "runtime_sec_mean": float(row["runtime_sec_mean"]), "summary_file": record["outputs"]["summary_file"]}
    for field in row:
        if field.startswith(("host_cpu_usage_pct_", "host_memory_used_bytes_")) and field.endswith("_mean"):
            result[field] = float(row[field]) if row[field] else ""
    result["scrape_interval_seconds"] = record["monitoring_preflight"]["scrape_interval_seconds"]
    return result


def validate(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors, collected = [], []
    if manifest["status"] == "PLANNED":
        errors.append("Plan-only manifest contains no real load experiment results")
    for path, expected_hash in manifest.get("scripts_and_config_sha256", {}).items():
        if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            errors.append(f"Experiment source changed: {path}")
    for trial in manifest["trials"]:
        try:
            collected.append(validate_trial(trial))
        except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError) as exc:
            errors.append(f"{trial['trial_id']}: {exc}")
    keys = [(t["load_family"], t["program"], t["input_profile"], t["profile_id"]) for t in manifest["trials"]]
    require(len(set(keys)) == len(keys), "Duplicate experiment configurations")
    for item in collected:
        idle = next((r for r in collected if r["load_family"] == item["load_family"] and r["program_id"] == item["program_id"] and
                     r["input_id"] == item["input_id"] and r["profile_id"].endswith("_IDLE")), None)
        item["runtime_ratio_to_idle_ANALYSIS_ONLY"] = item["runtime_sec_median"] / idle["runtime_sec_median"] if idle else ""
    complete_families = {}
    for family in ["cpu", "memory"]:
        groups = {}
        for item in collected:
            if item["load_family"] == family:
                groups.setdefault((item["program_id"], item["input_id"]), set()).add(item["profile_id"])
        complete_families[family] = bool(groups) and all(profiles == {family.upper() + "_" + level for level in ["IDLE", "LOW", "MEDIUM", "HIGH"]} for profiles in groups.values())
    if manifest["stage"] == "pilot":
        expected_sizes = {"MINI_DATASET", "MEDIUM_DATASET", "EXTRALARGE_DATASET"}
        if len({r["program_id"] for r in collected}) < 2 or {r["input_id"] for r in collected} != expected_sizes:
            errors.append("Pilot requires at least two programs and all three legacy input profiles")
        for family in manifest["load_families"]:
            if not complete_families[family]:
                errors.append(f"Pilot requires all four {family} profiles per program/size")
    report = {"status": "PASS" if not errors and len(collected) == len(manifest["trials"]) else "FAIL",
        "stage": manifest["stage"], "dataset": manifest["dataset"], "hostname": manifest.get("execution_host"),
        "validated_configurations": len(collected), "expected_configurations": len(manifest["trials"]),
        "cpu_four_levels_validated": complete_families["cpu"], "memory_four_levels_validated": complete_families["memory"],
        "errors": errors, "raw_csv_modified": False,
        "comparison_fields_are_analysis_not_a_prediction_whitelist": True}
    root = manifest_path.parent
    if collected:
        fields = list(dict.fromkeys(key for item in collected for key in item))
        with (root / "load_experiment_comparisons.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(collected)
    load.write_json(root / "validation_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = validate(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
