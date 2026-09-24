import math
import statistics
from collections import defaultdict
from typing import Any


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def macro_f1(records: list[dict[str, Any]], labels: tuple[str, ...]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(
            record["expected"] == label and record.get("choice") == label for record in records
        )
        false_positive = sum(
            record["expected"] != label and record.get("choice") == label for record in records
        )
        false_negative = sum(
            record["expected"] == label and record.get("choice") != label for record in records
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return statistics.fmean(scores)


def calibration_error(records: list[dict[str, Any]], bins: int = 10) -> float | None:
    valid = [record for record in records if record.get("status") == "ok"]
    if not valid:
        return None
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        confidence = float(record["probabilities"][record["choice"]])
        index = min(int(confidence * bins), bins - 1)
        grouped[index].append(record)
    error = 0.0
    for bucket in grouped.values():
        accuracy = statistics.fmean(record["correct"] for record in bucket)
        confidence = statistics.fmean(
            record["probabilities"][record["choice"]] for record in bucket
        )
        error += len(bucket) / len(valid) * abs(accuracy - confidence)
    return error


def summarize(records: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    total = len(records)
    valid = [record for record in records if record.get("status") == "ok"]
    correct = sum(record.get("correct", False) for record in records)
    latencies = [float(record["latency_ms"]) for record in valid]
    steady_latencies = latencies[1:]
    observed_labels = tuple(
        label for label in labels if any(r["expected"] == label for r in records)
    )
    brier_values = []
    strict_valid = 0
    argmax_agreement = 0
    usage: dict[str, float] = defaultdict(float)
    for record in valid:
        probabilities = record["probabilities"]
        brier_values.append(
            sum(
                (float(probabilities[label]) - float(record["expected"] == label)) ** 2
                for label in labels
            )
        )
        strict_valid += bool(record.get("schema_valid_strict"))
        argmax_agreement += bool(record.get("argmax_agrees"))
        for key, value in record.get("usage", {}).items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                usage[key] += value
    result = {
        "examples": total,
        "answered": len(valid),
        "errors": total - len(valid),
        "coverage": len(valid) / total if total else None,
        "accuracy": correct / total if total else None,
        "accuracy_answered": correct / len(valid) if valid else None,
        "macro_f1": macro_f1(records, observed_labels) if observed_labels else None,
        "macro_f1_all_labels": macro_f1(records, labels) if records else None,
        "brier_answered": statistics.fmean(brier_values) if brier_values else None,
        "ece_10_answered": calibration_error(records),
        "schema_valid_strict": strict_valid / total if total else None,
        "choice_argmax_agreement": argmax_agreement / len(valid) if valid else None,
        "usage": dict(usage),
        "latency_ms": {
            "first": latencies[0] if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "steady_state_latency_ms": {
            "mean": statistics.fmean(steady_latencies) if steady_latencies else None,
            "p50": percentile(steady_latencies, 0.5),
            "p95": percentile(steady_latencies, 0.95),
        },
    }
    groups = sorted({record["group"] for record in records})
    if len(groups) > 1:
        result["groups"] = {
            group: summarize([record for record in records if record["group"] == group], labels)
            for group in groups
        }
        for group_result in result["groups"].values():
            group_result.pop("groups", None)
    return result
