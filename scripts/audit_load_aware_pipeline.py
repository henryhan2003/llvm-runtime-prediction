"""Read-only audit of existing three-phase data, before adding load trials."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "polybench_full_20260812_181722/results_20260812_181722.csv",
    ROOT / "outputs/rodinia_openmp_cpu_runtime_repaired_v3_v4_20261027/rodinia_openmp_summary_repaired_v3_v4.csv",
    ROOT / "cuda_full_download/rodinia_cuda_full_20260814_155908/rodinia_cuda_summary_20260814_155916.csv"]


def audit(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    phases = {}
    ordering_errors = 0
    for row in rows:
        try:
            windows = [(float(row[f"prometheus_{phase}_window_start_timestamp"]), float(row[f"prometheus_{phase}_window_end_timestamp"]))
                       for phase in ["before", "during", "after"]]
            if not (windows[0][0] <= windows[0][1] <= windows[1][0] <= windows[1][1] <= windows[2][0] <= windows[2][1]):
                ordering_errors += 1
        except (ValueError, KeyError):
            ordering_errors += 1
    for phase in ["before", "during", "after"]:
        counts = [float(r[f"prometheus_{phase}_sample_count"]) for r in rows if r.get(f"prometheus_{phase}_sample_count")]
        durations = [float(r[f"prometheus_{phase}_window_duration_seconds"]) for r in rows if r.get(f"prometheus_{phase}_window_duration_seconds")]
        phases[phase] = {"query_sample_count_min_max_NOT_raw_scrapes": [min(counts), max(counts)] if counts else None,
                        "window_duration_min_max_seconds": [min(durations), max(durations)] if durations else None}
    return {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
        "programs": len({r["program_id"] for r in rows}), "columns": len(columns), "fields": columns,
        "phase_statistics": phases, "window_ordering_errors": ordering_errors,
        "three_phase_success_rows": sum(r.get("prometheus_three_phase_collection_success", "").lower() == "true" for r in rows),
        "gpu_three_phase_fields_present": "gpu_three_phase_collection_success" in columns,
        "identity_fields": [field for field in ["sample_id", "session_id", "collected_at", "program_id", "input_id", "env_id", "compile_config_id"] if field in columns]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/load_aware_pipeline_validation/existing_data_audit.json")
    args = parser.parse_args()
    reports = [audit(path) for path in SOURCES]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"read_only": True, "sources": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ["source", "rows", "programs", "columns", "window_ordering_errors", "three_phase_success_rows"]} for row in reports], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
