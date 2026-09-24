import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from laya_jev_bench.catalog import SOURCES
from laya_jev_bench.datasets import Dataset, Example
from laya_jev_bench.metrics import summarize
from laya_jev_bench.providers import Decision, ProviderFatalError


def json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def protocol_for(dataset: Dataset, examples: tuple[Example, ...], seed: int) -> dict[str, Any]:
    payload = {
        "dataset": dataset.name,
        "dataset_fingerprint": dataset.fingerprint,
        "instruction": dataset.instruction,
        "criteria": dataset.criteria,
        "selected_ids": [example.id for example in examples],
        "seed": seed,
        "sources": [
            {"path": source.path, "url": source.url, "sha256": source.sha256}
            for source in SOURCES[dataset.name]
        ],
    }
    payload["protocol_fingerprint"] = hashlib.sha256(json_dump(payload).encode()).hexdigest()
    return payload


def validate_decision(decision: Decision, labels: tuple[str, ...]) -> tuple[dict[str, float], bool]:
    if decision.choice not in labels:
        raise ValueError(f"Unknown choice: {decision.choice}")
    if set(decision.probabilities) != set(labels):
        missing = set(labels) - set(decision.probabilities)
        extra = set(decision.probabilities) - set(labels)
        raise ValueError(
            f"Probability labels mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    probabilities = {label: float(decision.probabilities[label]) for label in labels}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("Probabilities must be finite values between zero and one")
    total = sum(probabilities.values())
    strict = round(abs(total - 1.0), 12) <= 0.001
    if abs(total - 1.0) > 0.02 or total <= 0:
        raise ValueError(f"Probabilities sum to {total}, outside the accepted tolerance")
    return {label: value / total for label, value in probabilities.items()}, strict


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_provider(
    provider: Any,
    dataset: Dataset,
    examples: tuple[Example, ...],
    protocol: dict[str, Any],
    output_dir: Path,
    resume: bool,
) -> Path:
    if len(dataset.labels) > provider.max_options:
        raise ValueError(
            f"{provider.name} supports {provider.max_options} options, but {dataset.name} "
            f"requires {len(dataset.labels)}"
        )
    provider_dir = output_dir / provider.name
    provider_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol["protocol_fingerprint"] != protocol["protocol_fingerprint"]:
            raise ValueError(f"Protocol mismatch in {output_dir}")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    predictions_path = provider_dir / "predictions.jsonl"
    if predictions_path.exists() and not resume:
        raise FileExistsError(f"Result file already exists: {predictions_path}")
    records = load_records(predictions_path) if resume else {}
    metadata = {
        **provider.metadata(),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "started_at": datetime.now(UTC).isoformat(),
        "environment": environment(),
    }
    (provider_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    question = {
        "type": "choice",
        "instructions": dataset.instruction,
        "criteria": dataset.criteria,
    }
    fatal_error = None
    with predictions_path.open("a", encoding="utf-8") as output:
        for index, example in enumerate(examples, 1):
            previous = records.get(example.id)
            if previous and previous.get("status") == "ok":
                continue
            if hasattr(provider, "pace"):
                provider.pace()
            started = time.perf_counter()
            try:
                decision = provider.decide(example.text, question)
                latency_ms = (time.perf_counter() - started) * 1000
                probabilities, strict = validate_decision(decision, dataset.labels)
                argmax = max(dataset.labels, key=probabilities.__getitem__)
                record = {
                    "id": example.id,
                    "group": example.group,
                    "expected": example.label,
                    "choice": decision.choice,
                    "correct": decision.choice == example.label,
                    "probabilities": probabilities,
                    "confidence": decision.confidence,
                    "argmax": argmax,
                    "argmax_agrees": argmax == decision.choice,
                    "schema_valid_strict": strict,
                    "latency_ms": round(latency_ms, 3),
                    "model": decision.model,
                    "upstream_provider": decision.provider,
                    "request_id": decision.request_id,
                    "usage": decision.usage,
                    "status": "ok",
                    "protocol_fingerprint": protocol["protocol_fingerprint"],
                }
            except Exception as error:
                latency_ms = (time.perf_counter() - started) * 1000
                record = {
                    "id": example.id,
                    "group": example.group,
                    "expected": example.label,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "latency_ms": round(latency_ms, 3),
                    "protocol_fingerprint": protocol["protocol_fingerprint"],
                }
                if isinstance(error, ProviderFatalError):
                    fatal_error = error
            output.write(json_dump(record) + "\n")
            output.flush()
            records[example.id] = record
            completed = sum(item.get("status") == "ok" for item in records.values())
            print(
                f"\r{dataset.name} {provider.name}: {index}/{len(examples)} answered={completed}",
                end="",
                flush=True,
            )
            if fatal_error:
                break
    print()
    ordered_records = [records[example.id] for example in examples if example.id in records]
    summary = {
        "dataset": dataset.name,
        "provider": provider.name,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "metrics": summarize(ordered_records, dataset.labels),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    summary_path = provider_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if fatal_error:
        raise fatal_error
    return summary_path


def new_run_directory(root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate
