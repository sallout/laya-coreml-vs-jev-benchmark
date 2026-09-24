# Laya vs Jev Intent Benchmark

A reproducible, paired comparison of Laya and Jev on Banking77, ArBanking77, and CLINC150. Every model receives the same state, instruction, option labels, option order, and test examples. Each dataset produces an independent protocol and report.

The completed full-run results are in [RESULTS.md](RESULTS.md).

## What this measures

| Dataset | Test examples | Options | Focus |
|---|---:|---:|---|
| Banking77 | 3,080 | 77 | Fine-grained English banking intents |
| ArBanking77 | 15,534 | 77 | MSA, Palestinian, Saudi, Moroccan, and Tunisian banking intents |
| CLINC150 | 5,500 | 151 | 150 assistant intents plus out-of-scope detection |

This is a zero-shot classification benchmark. Training and validation data are never passed to either model. It does not measure general knowledge, long-form reasoning, or text generation.

Laya's maintainers recommend keeping choice questions below roughly 20 options because all option labels share a fixed token budget. Banking77 and CLINC150 deliberately exceed that recommendation. They are valid stress tests of the requested 77-way and 151-way use case, but their results should not be presented as a general measure of every Laya capability.

## Reproducibility rules

- Dataset files come from the original repositories at fixed commit revisions.
- Every downloaded file is verified against a committed SHA-256 digest.
- The dataset-specific Laya checkpoints and Jev model version are pinned.
- The moving `jev-latest` alias is not used.
- Full official test splits are used unless `--limit` is supplied.
- Limited runs use deterministic stratified sampling across labels and dataset groups.
- The complete protocol receives a SHA-256 fingerprint.
- A comparison uses paired records under one protocol and counts provider errors as incorrect.
- API failures count against headline coverage and accuracy and are never replaced by invented answers.
- Requests are not retried automatically; interrupted runs use explicit append-only resume files.
- Jev latency includes network time. Laya latency is local inference time. They are reported, not treated as equivalent deployment conditions.

## Requirements

- macOS 15 or newer on Apple Silicon for Laya-CoreML
- Python 3.11 through 3.13
- An OpenRouter API key for Jev
- `uv`

## Install

```bash
uv sync --extra convert --extra dev
```

Create a key at [OpenRouter API Keys](https://openrouter.ai/settings/keys), add credits, and expose it only in the current terminal session:

```bash
read -s "OPENROUTER_API_KEY?OpenRouter API key: "
export OPENROUTER_API_KEY
echo
```

The key is read from the environment, sent only in the authorization header, and never written to local results, protocol files, or run metadata.

## Download and verify datasets

```bash
uv run laya-jev-bench prepare
```

The files are stored under `data/raw` and excluded from Git. Their upstream license files are downloaded alongside them.

## Prepare the Laya benchmark exports

The published general-purpose Laya-CoreML bundles have capacity for 32 choices. This benchmark requires 151, so build reproducible Core ML exports with a larger output signature:

```bash
uv run laya-jev-bench prepare-laya
```

The command prepares two pinned exports:

| Variant | Dataset | Hugging Face revision |
|---|---|---|
| `laya-typed-decisions` | Banking77 and CLINC150 | `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` |
| `laya-multilingual` | ArBanking77 | `052592a15d198d9ad47da779604259b10b47b7aa` |

The routing rule is fixed before test evaluation. `laya-typed-decisions` is the strongest published Laya checkpoint on Banking77 and has the 1024-token capacity required by the 151-choice CLINC150 protocol. No official CLINC150 comparison among Laya checkpoints is published, so this is a preregistered task-and-capacity choice rather than a claim that the checkpoint is empirically best on CLINC150. `laya-multilingual` is the checkpoint intended for non-English inputs. Conversion metadata, source hashes, dependency versions, and artifact hashes are stored in each export's `coreml_config.json`. Model directories are excluded from Git.

## Run a small paired smoke test

```bash
uv run laya-jev-bench run \
  --datasets banking77 arbanking77 clinc150 \
  --providers both \
  --compute-units cpu_gpu \
  --limit 100
```

`--limit` applies separately to every dataset. The default seed is `42`.

## Run the full benchmark

```bash
uv run laya-jev-bench run \
  --datasets all \
  --providers both \
  --compute-units cpu_gpu
```

The full suite sends 24,114 paid Jev requests through OpenRouter. Check current pricing, credits, and account limits before starting it.

## Run one provider at a time

```bash
benchmark_run=results/manual-banking77

uv run laya-jev-bench run \
  --datasets banking77 \
  --providers laya \
  --compute-units cpu_gpu \
  --run-dir "$benchmark_run"

uv run laya-jev-bench run \
  --datasets banking77 \
  --providers jev \
  --run-dir "$benchmark_run"

uv run laya-jev-bench report "$benchmark_run/banking77"
```

For a direct comparison, both providers must write into the same dataset run directory. The simplest and safest command is `--providers both`.

## Resume an interrupted run

```bash
uv run laya-jev-bench run \
  --datasets banking77 \
  --providers both \
  --run-dir results/20260923T120000Z \
  --resume
```

Completed answers are skipped. Failed or missing answers are attempted again. HTTP 401, 403, and 429 responses stop a Jev run immediately.

## Generate a comparison again

```bash
uv run laya-jev-bench report results/20260923T120000Z/banking77
```

## Outputs

Each dataset directory contains:

```text
protocol.json
laya/predictions.jsonl
laya/run.json
laya/summary.json
jev/predictions.jsonl
jev/run.json
jev/summary.json
comparison.json
comparison.md
```

`protocol.json` records the exact source files, checksums, selected IDs, instruction, criteria, seed, dataset fingerprint, and protocol fingerprint. Raw dataset text is not copied into result files.

The summaries include:

- Accuracy with failed requests counted as incorrect
- Accuracy among answered examples
- Coverage
- Macro F1
- Multiclass Brier score
- 10-bin expected calibration error
- Strict probability-schema validity
- Choice and probability-argmax agreement
- Aggregate provider-reported token usage
- Mean, P50, and P95 latency
- First-request and steady-state latency reported separately
- Per-dialect results for ArBanking77
- Separate in-scope and out-of-scope results for CLINC150

The paired report includes the accuracy difference and an exact McNemar test.

## Model configuration

The default Laya route uses `models/laya-typed-decisions-151` for the English Banking77 and CLINC150 datasets, and `models/laya-multilingual-151` for ArBanking77. Both use a 1024-token context and 151 option slots. The route is based only on dataset language and task shape and is fixed before test evaluation. The published 32-option general bundles and 96-token ANE bundles cannot run these one-question 77-way and 151-way protocols.

The default Jev route uses OpenRouter's Decisions API with the pinned model `typesafe/jev-1.13`. It is the native decision endpoint and returns the selected choice, per-option probabilities, confidence, actual model snapshot, upstream provider, usage, cost, and request ID.

The direct TypeSafe backend remains available:

```bash
read -s "TYPESAFE_API_KEY?TypeSafe API key: "
export TYPESAFE_API_KEY
echo

uv run laya-jev-bench run \
  --datasets banking77 \
  --providers both \
  --jev-backend typesafe
```

Override pinned versions only when intentionally creating a new benchmark protocol:

```bash
uv run laya-jev-bench run \
  --datasets banking77 \
  --providers both \
  --laya-model MODEL_OR_DIRECTORY \
  --laya-revision REVISION \
  --jev-model MODEL
```

Use `--compute-units cpu`, `cpu_gpu`, `cpu_ne`, or `all` to record a specific Core ML execution policy. Use `--offline` after the pinned Laya checkpoint has been cached.

## Tests

```bash
uv run ruff check .
uv run python -m unittest discover -s tests
```

## Licenses and citations

The benchmark code is MIT licensed. Dataset and model licenses remain their own. See [THIRD_PARTY.md](THIRD_PARTY.md) before publishing data or result artifacts.
# laya-coreml-vs-jev-benchmark
