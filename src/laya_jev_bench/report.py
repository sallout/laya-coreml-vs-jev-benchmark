import json
import math
from pathlib import Path
from typing import Any


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


def exact_mcnemar_log10_p_value(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 0.0
    boundary = min(left_only, right_only)
    logs = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2)
        for index in range(boundary + 1)
    ]
    maximum = max(logs)
    log_tail = maximum + math.log(sum(math.exp(value - maximum) for value in logs))
    return min(0.0, (math.log(2) + log_tail) / math.log(10))


def exact_mcnemar_p_value(left_only: int, right_only: int) -> float:
    return 10 ** exact_mcnemar_log10_p_value(left_only, right_only)


def compare(run_dir: Path) -> dict[str, Any]:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    laya = load_predictions(run_dir / "laya" / "predictions.jsonl")
    jev = load_predictions(run_dir / "jev" / "predictions.jsonl")
    ids = [item_id for item_id in protocol["selected_ids"] if item_id in laya and item_id in jev]
    pairs = [(laya[item_id], jev[item_id]) for item_id in ids]
    paired_answered = sum(
        left.get("status") == "ok" and right.get("status") == "ok" for left, right in pairs
    )
    laya_answered = sum(left.get("status") == "ok" for left, _ in pairs)
    jev_answered = sum(right.get("status") == "ok" for _, right in pairs)
    outcomes = [
        (
            left.get("status") == "ok" and bool(left.get("correct")),
            right.get("status") == "ok" and bool(right.get("correct")),
        )
        for left, right in pairs
    ]
    both_correct = sum(left and right for left, right in outcomes)
    laya_only = sum(left and not right for left, right in outcomes)
    jev_only = sum(not left and right for left, right in outcomes)
    both_wrong = sum(not left and not right for left, right in outcomes)
    result = {
        "dataset": protocol["dataset"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "selected_examples": len(protocol["selected_ids"]),
        "paired_examples": len(pairs),
        "paired_answered": paired_answered,
        "laya_coverage": laya_answered / len(pairs) if pairs else None,
        "jev_coverage": jev_answered / len(pairs) if pairs else None,
        "laya_accuracy": (both_correct + laya_only) / len(pairs) if pairs else None,
        "jev_accuracy": (both_correct + jev_only) / len(pairs) if pairs else None,
        "accuracy_difference_laya_minus_jev": (laya_only - jev_only) / len(pairs)
        if pairs
        else None,
        "both_correct": both_correct,
        "laya_only_correct": laya_only,
        "jev_only_correct": jev_only,
        "both_wrong": both_wrong,
        "mcnemar_exact_p_value": exact_mcnemar_p_value(laya_only, jev_only),
        "mcnemar_exact_log10_p_value": exact_mcnemar_log10_p_value(laya_only, jev_only),
    }
    (run_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    p_value = result["mcnemar_exact_p_value"]
    p_value_text = (
        f"{p_value:.6g}"
        if p_value > 0
        else f"< 1e-300 (log10 p = {result['mcnemar_exact_log10_p_value']:.1f})"
    )
    markdown = [
        f"# {protocol['dataset']} comparison",
        "",
        f"Protocol: `{protocol['protocol_fingerprint']}`",
        "",
        "| Metric | Laya | Jev |",
        "|---|---:|---:|",
        f"| Accuracy | {result['laya_accuracy']:.2%} | {result['jev_accuracy']:.2%} |"
        if pairs
        else "| Accuracy | n/a | n/a |",
        "",
        f"Paired examples: {len(pairs)}",
        "",
        f"Paired answered examples: {paired_answered}",
        "",
        f"Coverage: Laya {result['laya_coverage']:.2%}, Jev {result['jev_coverage']:.2%}"
        if pairs
        else "Coverage: n/a",
        "",
        f"Accuracy difference, Laya minus Jev: {result['accuracy_difference_laya_minus_jev']:.2%}"
        if pairs
        else "Accuracy difference, Laya minus Jev: n/a",
        "",
        f"Exact McNemar p-value: {p_value_text}",
        "",
    ]
    (run_dir / "comparison.md").write_text("\n".join(markdown), encoding="utf-8")
    return result
