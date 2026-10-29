from pathlib import Path
import tempfile
import unittest

from scripts import analyze_load_pilot_results as analysis


class AnalysisTests(unittest.TestCase):
    def test_remote_artifacts_map_to_downloaded_trial_or_shared_cache(self):
        root = Path(__file__).resolve().parents[1] / "tmp" / "mapping-fixture"
        trial = {"trial_id": "t1", "results_dir": "/server/run/t1/results"}
        manifest = {"preparation_cache": {"directory": "/server/run/preparation_cache"}}
        self.assertEqual((root / "t1/results/build/toy").resolve(), analysis.local_artifact(root, trial, "/server/run/t1/results/build/toy", manifest))
        self.assertEqual((root / "preparation_cache/key/build/toy").resolve(), analysis.local_artifact(root, trial, "/server/run/preparation_cache/key/build/toy", manifest))
        with self.assertRaises(ValueError):
            analysis.local_artifact(root, trial, "/unrelated/toy", manifest)

    def test_missing_values_remain_distinct_from_zero(self):
        self.assertIsNone(analysis.number(""))
        self.assertIsNone(analysis.number(None))
        self.assertEqual(0, analysis.number("0"))
        self.assertEqual({"count": 0, "min": None, "max": None, "median": None, "mean": None}, analysis.describe([None]))
        for value in ["nan", "inf", "-inf"]:
            with self.assertRaises(ValueError):
                analysis.number(value)

    def test_percentiles_use_linear_interpolation_without_dropping_outliers(self):
        self.assertEqual(2.5, analysis.percentile([1, 2, 3, 100], .5))
        self.assertEqual(100, analysis.percentile([1, 2, 3, 100], 1))
        self.assertEqual(2, analysis.percentile([2], .1))
        self.assertIsNone(analysis.percentile([], .9))

    def test_phase_telemetry_uses_only_in_window_samples_and_preserves_peak_sensor(self):
        samples = [{"timestamp": 9, "temperature_max_celsius": 99, "cpu_frequency_mean_mhz": 5000},
                   {"timestamp": 10, "temperature_max_celsius": 85.5, "temperature_max_source": {"device": "nvme", "label": "Composite"},
                    "cpu_frequency_mean_mhz": 3000},
                   {"timestamp": 11, "temperature_max_celsius": 70, "cpu_frequency_mean_mhz": 4000},
                   {"timestamp": 12, "temperature_max_celsius": 60, "cpu_frequency_mean_mhz": None}]
        result = analysis.telemetry(analysis.within(samples, (10, 11)))
        self.assertEqual(2, result["samples"])
        self.assertEqual(85.5, result["peak_temperature_celsius"])
        self.assertEqual("nvme", result["peak_temperature_source"]["device"])
        self.assertEqual(3500, result["frequency_mhz"]["mean"])
        self.assertEqual(1, result["observations_ge_85_celsius"])

    def test_empty_temperature_and_frequency_are_not_zero(self):
        result = analysis.telemetry([{"timestamp": 10, "temperature_max_celsius": None}])
        self.assertIsNone(result["peak_temperature_celsius"])
        self.assertIsNone(result["peak_temperature_source"])
        self.assertIsNone(result["frequency_mhz"]["mean"])

    def test_snapshot_hashes_bytes_and_detects_source_changes(self):
        parent = Path(__file__).resolve().parents[1] / "tmp"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            path = root / "fixture.csv"
            path.write_bytes(b"x\n1\n")
            initial = analysis.snapshot(root)
            self.assertEqual(initial, analysis.snapshot(root))
            path.write_bytes(b"x\n2\n")
            self.assertNotEqual(initial, analysis.snapshot(root))


if __name__ == "__main__":
    unittest.main()
