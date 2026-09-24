import argparse
import json
import os
from pathlib import Path

from laya_jev_bench.catalog import (
    LAYA_DATASET_VARIANTS,
    LAYA_VARIANTS,
    OPENROUTER_DECISIONS_URL,
    OPENROUTER_JEV_MODEL,
    SOURCES,
    TYPESAFE_JEV_MODEL,
    TYPESAFE_SYSTEM_ONE_URL,
)
from laya_jev_bench.datasets import load_dataset, select_examples
from laya_jev_bench.download import prepare_datasets
from laya_jev_bench.providers import JevProvider, LayaProvider
from laya_jev_bench.report import compare
from laya_jev_bench.runner import new_run_directory, protocol_for, run_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="laya-jev-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--datasets", nargs="+", choices=[*SOURCES, "all"], default=["all"])
    prepare.add_argument("--data-dir", type=Path, default=Path("data"))

    prepare_laya = subparsers.add_parser("prepare-laya")
    prepare_laya.add_argument(
        "--variants",
        nargs="+",
        choices=[*LAYA_VARIANTS, "all"],
        default=["all"],
    )
    prepare_laya.add_argument("--models-dir", type=Path, default=Path("models"))
    prepare_laya.add_argument("--max-options", type=int, default=151)
    prepare_laya.add_argument("--precision", choices=["float16", "float32"], default="float16")

    run = subparsers.add_parser("run")
    run.add_argument("--datasets", nargs="+", choices=[*SOURCES, "all"], required=True)
    run.add_argument("--providers", nargs="+", choices=["laya", "jev", "both"], required=True)
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--output-dir", type=Path, default=Path("results"))
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-download", action="store_true")
    run.add_argument("--laya-model")
    run.add_argument("--laya-revision")
    run.add_argument("--compute-units", choices=["all", "cpu", "cpu_gpu", "cpu_ne"])
    run.add_argument("--offline", action="store_true")
    run.add_argument("--jev-backend", choices=["openrouter", "typesafe"], default="openrouter")
    run.add_argument("--jev-model")
    run.add_argument("--api-key-env")
    run.add_argument("--endpoint")
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--delay-ms", type=float, default=0.0)

    report = subparsers.add_parser("report")
    report.add_argument("run_dir", type=Path)
    return parser


def expand(values: list[str], available: list[str]) -> list[str]:
    if "all" in values or "both" in values:
        return available
    return list(dict.fromkeys(values))


def command_prepare(args: argparse.Namespace) -> None:
    names = expand(args.datasets, list(SOURCES))
    downloaded = prepare_datasets(names, args.data_dir)
    for name, paths in downloaded.items():
        print(f"{name}: {len(paths)} verified files")


def command_prepare_laya(args: argparse.Namespace) -> None:
    try:
        from laya_coreml.convert import convert
    except ImportError as error:
        raise RuntimeError(
            "Laya conversion support is not installed. Run: uv sync --extra convert"
        ) from error
    variants = expand(args.variants, list(LAYA_VARIANTS))
    for variant in variants:
        config = LAYA_VARIANTS[variant]
        output = args.models_dir / Path(config["model"]).name
        manifest_path = output / "coreml_config.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "source": config["source"],
                "revision": config["revision"],
                "precision": args.precision,
            }
            actual = {key: manifest.get(key) for key in expected}
            shape = manifest.get("shape", {})
            if actual != expected or shape.get("max_options") != args.max_options:
                raise RuntimeError(f"Existing model does not match requested export: {output}")
            print(f"{output}: verified")
            continue
        converted = convert(
            config["source"],
            output,
            max_length=config["max_length"],
            max_options=args.max_options,
            precision=args.precision,
            revision=config["revision"],
        )
        print(converted)


def laya_model_for_dataset(args: argparse.Namespace, dataset: str) -> str:
    if args.laya_model:
        return args.laya_model
    variant = LAYA_DATASET_VARIANTS[dataset]
    return LAYA_VARIANTS[variant]["model"]


def make_providers(args: argparse.Namespace, dataset: str) -> list[object]:
    names = expand(args.providers, ["laya", "jev"])
    jev_defaults = {
        "openrouter": (
            OPENROUTER_JEV_MODEL,
            "OPENROUTER_API_KEY",
            OPENROUTER_DECISIONS_URL,
        ),
        "typesafe": (TYPESAFE_JEV_MODEL, "TYPESAFE_API_KEY", TYPESAFE_SYSTEM_ONE_URL),
    }
    default_model, default_key_env, default_endpoint = jev_defaults[args.jev_backend]
    jev_model = args.jev_model or default_model
    api_key_env = args.api_key_env or default_key_env
    endpoint = args.endpoint or default_endpoint
    if "jev" in names and not os.environ.get(api_key_env):
        raise RuntimeError(f"Environment variable {api_key_env} is not set")
    providers = []
    for name in names:
        if name == "laya":
            providers.append(
                LayaProvider(
                    laya_model_for_dataset(args, dataset),
                    args.laya_revision or None,
                    args.compute_units,
                    args.offline,
                )
            )
        else:
            providers.append(
                JevProvider(
                    jev_model,
                    api_key_env,
                    endpoint,
                    args.jev_backend,
                    args.timeout,
                    args.delay_ms,
                )
            )
    return providers


def command_run(args: argparse.Namespace) -> None:
    names = expand(args.datasets, list(SOURCES))
    root = args.run_dir or new_run_directory(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        providers = make_providers(args, name)
        dataset = load_dataset(name, args.data_dir, download=not args.no_download)
        examples = select_examples(dataset, args.limit, args.seed)
        protocol = protocol_for(dataset, examples, args.seed)
        dataset_dir = root / name
        for provider in providers:
            summary_path = run_provider(
                provider, dataset, examples, protocol, dataset_dir, args.resume
            )
            print(summary_path)
        if {provider.name for provider in providers} == {"laya", "jev"}:
            result = compare(dataset_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    print(root)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "prepare-laya":
        command_prepare_laya(args)
    elif args.command == "run":
        command_run(args)
    else:
        result = compare(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
