# Full benchmark results

Run directory: `results/full-20260923`. The run began on 2026-09-23 and its transient Jev failures were resumed on 2026-09-24. All 24,114 test examples now have a valid answer from both providers.

| Dataset | Examples | Options | Laya accuracy | Jev accuracy | Jev lead |
|---|---:|---:|---:|---:|---:|
| Banking77 | 3,080 | 77 | 46.27% | 79.97% | 33.70 percentage points |
| ArBanking77 | 15,534 | 77 | 12.42% | 63.42% | 51.00 percentage points |
| CLINC150 | 5,500 | 151 | 48.85% | 89.40% | 40.55 percentage points |

Accuracy is the number of correct answers divided by the full official test split. Coverage is 100% for both providers on all three datasets. No training, fine-tuning, or validation examples were supplied during this run.

| Dataset | Laya macro F1 | Jev macro F1 | Laya checkpoint |
|---|---:|---:|---|
| Banking77 | 43.37% | 79.28% | `convaiinnovations/laya-typed-decisions` |
| ArBanking77 | 10.83% | 62.77% | `convaiinnovations/laya-multilingual` |
| CLINC150 | 51.55% | 89.54% | `convaiinnovations/laya-typed-decisions` |

Jev used OpenRouter's Decisions API. Every successful response reported the snapshot `typesafe/jev-1.13-20260917`. The Laya Core ML exports used the fixed revisions in [README.md](README.md#model-configuration) with `cpu_gpu` compute units. The checkpoint route was fixed before test evaluation.

## Subgroups

| ArBanking77 dialect | Examples | Laya accuracy | Jev accuracy |
|---|---:|---:|---:|
| Modern Standard Arabic | 3,574 | 16.59% | 71.52% |
| Palestinian | 3,807 | 11.53% | 61.68% |
| Saudi | 3,580 | 11.76% | 63.16% |
| Moroccan | 3,574 | 11.39% | 60.91% |
| Tunisian | 999 | 7.01% | 51.05% |

| CLINC150 group | Examples | Laya accuracy | Jev accuracy |
|---|---:|---:|---:|
| In scope | 4,500 | 54.84% | 90.31% |
| Out of scope | 1,000 | 21.90% | 85.30% |

## Protocol and interpretation

Each paired example used the same state, instruction, option labels, and option order for both providers. The dataset-specific protocol fingerprints are:

| Dataset | SHA-256 protocol fingerprint |
|---|---|
| Banking77 | `014f7db72ba2df532fddf712e4649572fb4b0ab0498138bef304f7651d8e8645` |
| ArBanking77 | `5f5abaa7817b82c93cca0a12f5185578d9683a1abd3d031a0b01544fc162545b` |
| CLINC150 | `5348dcad2697f07150af99933cfaa37080b397fad3c9019dede6a7fc2fe93e4f` |

Banking77 had one transient Jev failure and ArBanking77 had 61. They were resumed without rerunning successful answers. The final summaries and comparisons contain one successful answer per selected example; the append-only raw prediction logs retain both failed attempts and their later successful answers.

These results measure 77-way and 151-way choice questions. [Laya's maintainers recommend fewer than roughly 20 choices](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md#limits-stated-plainly) because option labels share a fixed token budget. This benchmark tests the specified high-choice use case and does not establish a general ranking across all Laya tasks. The `laya-typed-decisions` checkpoint was the preregistered English choice for CLINC150; there is no published CLINC150 comparison proving it is the best Laya checkpoint for that dataset.

We did not audit whether either model encountered these public datasets during its own training.

The benchmark code, dataset revisions, file hashes, run commands, and report definitions are documented in [README.md](README.md). Local raw datasets, model exports, and run logs are excluded from Git.

The [full benchmark table](media/full-benchmark-table-simple.png) includes all three datasets and the paired outcome counts. It can be regenerated from the comparison files with `uv run --with pillow scripts/render_full_benchmark_table.py`.
