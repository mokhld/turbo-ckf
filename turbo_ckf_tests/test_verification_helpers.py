import json
import tempfile
import unittest
from pathlib import Path

from turbo_ckf_tests import verify_before_after


class VerificationHelpersTests(unittest.TestCase):
    def test_generate_report_shape(self):
        report = verify_before_after.generate_report(steps=10, repeats=2, warmup=0, parity_steps=8)

        self.assertEqual(report["schema_version"], 1)
        self.assertIn("environment", report)
        self.assertIn("config", report)
        self.assertIn("benchmarks", report)
        self.assertIn("parity", report)

        self.assertGreaterEqual(len(report["benchmarks"]), 3)
        for case in report["benchmarks"]:
            self.assertIn("name", case)
            self.assertIn("median_s", case)
            self.assertEqual(case["steps"], 10)
            self.assertEqual(case["repeats"], 2)
            self.assertEqual(len(case["samples_s"]), 2)

        parity = report["parity"]
        self.assertIn("enabled", parity)
        self.assertEqual(parity["steps"], 8)
        if parity["enabled"]:
            self.assertIn("max_abs_state_diff", parity)
            self.assertIn("rmse_state_diff", parity)

    def test_compare_reports_math(self):
        baseline = {
            "benchmarks": [
                {"name": "a", "median_s": 2.0},
                {"name": "b", "median_s": 4.0},
            ],
            "parity": {"enabled": True, "max_abs_state_diff": 1e-8},
        }
        current = {
            "benchmarks": [
                {"name": "a", "median_s": 1.0},
                {"name": "b", "median_s": 5.0},
            ],
            "parity": {"enabled": True, "max_abs_state_diff": 2e-8},
        }

        comparison = verify_before_after.compare_reports(baseline, current)
        self.assertEqual(comparison["shared_benchmark_cases"], ["a", "b"])
        self.assertAlmostEqual(comparison["benchmarks"]["a"]["speedup_vs_baseline"], 2.0)
        self.assertAlmostEqual(comparison["benchmarks"]["a"]["delta_pct"], -50.0)
        self.assertAlmostEqual(comparison["benchmarks"]["b"]["delta_pct"], 25.0)
        self.assertTrue(comparison["parity"]["available"])
        self.assertAlmostEqual(comparison["parity"]["ratio_current_to_baseline"], 2.0)

    def test_main_writes_comparison_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            baseline_path = tmp_path / "before.json"
            output_path = tmp_path / "after.json"

            baseline = verify_before_after.generate_report(steps=6, repeats=1, warmup=0, parity_steps=5)
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

            rc = verify_before_after.main(
                [
                    "--steps",
                    "6",
                    "--repeats",
                    "1",
                    "--warmup",
                    "0",
                    "--parity-steps",
                    "5",
                    "--compare-to",
                    str(baseline_path),
                    "--output",
                    str(output_path),
                ]
            )

            self.assertEqual(rc, 0)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("comparison", saved)


if __name__ == "__main__":
    unittest.main()
