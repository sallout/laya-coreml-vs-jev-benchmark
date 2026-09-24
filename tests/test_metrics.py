import unittest

from laya_jev_bench.metrics import percentile, summarize
from laya_jev_bench.providers import Decision
from laya_jev_bench.report import exact_mcnemar_log10_p_value, exact_mcnemar_p_value
from laya_jev_bench.runner import validate_decision


class MetricTests(unittest.TestCase):
    def test_percentile(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_summary_counts_errors_as_incorrect(self) -> None:
        records = [
            {
                "expected": "a",
                "choice": "a",
                "correct": True,
                "probabilities": {"a": 0.8, "b": 0.2},
                "schema_valid_strict": True,
                "argmax_agrees": True,
                "latency_ms": 5.0,
                "status": "ok",
                "group": "test",
            },
            {"expected": "b", "status": "error", "group": "test"},
        ]
        result = summarize(records, ("a", "b"))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["accuracy_answered"], 1.0)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["latency_ms"]["first"], 5.0)

    def test_probability_normalization(self) -> None:
        decision = Decision("a", {"a": 0.6, "b": 0.399}, 0.6, "model", {})
        probabilities, strict = validate_decision(decision, ("a", "b"))
        self.assertTrue(strict)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        self.assertLess(exact_mcnemar_p_value(10, 0), 0.01)
        self.assertAlmostEqual(exact_mcnemar_log10_p_value(10, 0), -2.709269960975831)
        self.assertLess(exact_mcnemar_log10_p_value(340, 8222), -300)


if __name__ == "__main__":
    unittest.main()
