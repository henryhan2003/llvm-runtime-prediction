"""Run no-pressure regression tests and audit unchanged static logic/CSV declarations."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source_preservation_audit():
    findings = []
    for name in ["pipeline_new.py", "pipeline_rodinia.py", "pipeline_rodinia_cuda.py"]:
        path = ROOT / "scripts" / name
        try:
            original = subprocess.check_output(["git", "show", "HEAD:scripts/" + name], cwd=ROOT, stderr=subprocess.DEVNULL).decode("utf-8")
        except (OSError, subprocess.CalledProcessError):
            return {"status": "NOT_CHECKED", "reason": "Git baseline unavailable; tests still run"}
        old, new = ast.parse(original), ast.parse(path.read_text(encoding="utf-8"))
        old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
        new_functions = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
        for label, node in old_functions.items():
            if label.startswith(("extract_", "generate_", "infer_input_")):
                unchanged = label in new_functions and ast.dump(node) == ast.dump(new_functions[label])
                findings.append({"file": name, "definition": label, "unchanged": unchanged, "kind": "static_logic"})
        for scope, before, after in [("module", old, new), ("main", old_functions.get("main"), new_functions.get("main"))]:
            if before is None or after is None:
                continue
            def declarations(tree):
                return {target.id: ast.dump(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        for target in node.targets if isinstance(target, ast.Name) and
                        (target.id.endswith("_FIELDS") or target.id in {"static_fields", "summary_fields", "run_fields", "results_fields"})}
            left, right = declarations(before), declarations(after)
            for label, definition in left.items():
                findings.append({"file": name, "definition": label, "unchanged": right.get(label) == definition, "kind": "csv_declaration"})
    return {"status": "PASS" if all(row["unchanged"] for row in findings) else "FAIL", "checks": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/load_aware_pipeline_validation/unit_test_report.json")
    args = parser.parse_args()
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for pattern in ["test_pipeline*.py", "test_resource_load.py", "test_load_fast_protocol.py", "test_openmp_fast_protocol.py"]:
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    preservation = source_preservation_audit()
    report = {"status": "PASS" if result.wasSuccessful() and preservation["status"] != "FAIL" else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "static_and_csv_preservation": preservation, "real_load_trials_executed": False,
        "mocked_safety_tests_are_not_real_load_validation": True, "model_training_executed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{report['status']}] {args.output}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
