# Third-party datasets and models

The benchmark downloads datasets at fixed upstream revisions and verifies every file with SHA-256. Dataset files are not included in this repository.

| Dataset | Upstream revision | License |
|---|---|---|
| [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) | `57ec275d8078af65b7731c2a98be812d844a6d6b` | CC BY 4.0 |
| [ArBanking77](https://github.com/SinaLab/ArBanking77) | `2e3a5639e254bc28828ed0af6d3059d64d3b97fc` | CC BY-SA 4.0 |
| [CLINC150/OOS](https://github.com/clinc/oos-eval) | `828f8093932c8fe6ca7936c3d2e52903b1c523de` | CC BY 3.0 |

The original license file for each dataset is downloaded into `data/raw/<dataset>/LICENSE`.

[Laya-CoreML](https://github.com/mizorewww/laya-coreml) is an Apache-2.0 project. The benchmark uses [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) at revision `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` for English datasets and [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) at revision `052592a15d198d9ad47da779604259b10b47b7aa` for ArBanking77.

Jev is accessed by default through OpenRouter's native Decisions API using `typesafe/jev-1.13`. The optional direct TypeSafe backend uses `jev-1.13.0`. Moving `latest` aliases are not used.

## Citations

- Casanueva et al. “Efficient Intent Detection with Dual Sentence Encoders.” 2020.
- Jarrar et al. “ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic.” 2023.
- Larson et al. “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.” 2019.
