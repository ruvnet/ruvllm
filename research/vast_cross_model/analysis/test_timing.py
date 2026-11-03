import copy
from pathlib import Path
import tempfile
import unittest

from timing import batch_timings, finite_seconds, report


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {"batch_size": 8, "panel_max_tokens": {"math": 8192}}
        self.cases = [{"id": str(n), "kind": "math", "panel": "math"} for n in range(10)]
        self.rows = [{**case, "batch_index": n // 8, "batch_seconds": 2.0 if n < 8 else 1.0,
                      "output_tokens": 5, "token_ids": [0] * 5, "max_tokens": 8192,
                      "finish_reason": "stop"} for n, case in enumerate(self.cases)]

    def test_repeated_batch_time_counted_once(self):
        result = batch_timings(self.rows, self.cases, self.protocol)
        self.assertEqual(result["inference_api_wall_seconds"], 3)
        self.assertEqual(result["output_tokens"], 50)
        self.assertAlmostEqual(result["output_tokens_per_inference_api_second"], 50 / 3)
        self.assertEqual(result["remaining_output_token_cap"], 0)

    def test_partial_snapshot_bound_uses_remaining_caps(self):
        result = batch_timings(self.rows[:8], self.cases, self.protocol)
        self.assertEqual(result["remaining_rows"], 2)
        self.assertEqual(result["remaining_output_token_cap"], 16384)
        self.assertEqual(result["conditional_cap_seconds_at_observed_rate"], 819.2)

    def test_rejects_mismatched_repeated_duration(self):
        self.rows[1]["batch_seconds"] = 2.1
        with self.assertRaisesRegex(ValueError, "durations differ"):
            batch_timings(self.rows, self.cases, self.protocol)

    def test_rejects_nonfinite_or_nonpositive_duration(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            rows = copy.deepcopy(self.rows)
            rows[0]["batch_seconds"] = value
            with self.assertRaisesRegex(ValueError, "Invalid batch duration"):
                batch_timings(rows, self.cases, self.protocol)

    def test_rejects_incomplete_batch_and_wrong_order(self):
        with self.assertRaisesRegex(ValueError, "inside a batch"):
            batch_timings(self.rows[:7], self.cases, self.protocol)
        self.rows[0], self.rows[1] = self.rows[1], self.rows[0]
        with self.assertRaisesRegex(ValueError, "frozen case prefix"):
            batch_timings(self.rows, self.cases, self.protocol)

    def test_rejects_missing_tokens_and_false_length_termination(self):
        self.rows[0]["token_ids"] = []
        with self.assertRaisesRegex(ValueError, "token count"):
            batch_timings(self.rows, self.cases, self.protocol)
        self.rows[0]["token_ids"] = [0] * 5
        self.rows[0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "token cap"):
            batch_timings(self.rows, self.cases, self.protocol)

    def test_receipt_durations_reject_bool_strings_and_nonfinite_values(self):
        for value in (True, "3", -1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "Invalid receipt duration"):
                finite_seconds(value, "receipt duration")
        self.assertEqual(finite_seconds(0, "receipt duration"), 0)

    def test_partial_report_rejects_missing_work_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Work directory does not exist"):
                report(Path(temporary) / "missing", allow_partial=True)


if __name__ == "__main__":
    unittest.main()
