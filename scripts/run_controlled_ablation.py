#!/usr/bin/env python3
"""Prepare, train, evaluate, and plot the controlled five-condition benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from sage_avo.config import load_config
from sage_avo.experiments.dataset import prepare_controlled_dataset

REPOSITORY = Path(__file__).resolve().parents[1]

LEARNED_VARIANTS = ("full", "no_gnn", "no_rgt", "no_physics")
ALL_VARIANTS = ("low_prior",) + LEARNED_VARIANTS


def _paths(config_path: Path, experiment_name: str | None) -> tuple[dict, Path, Path]:
    config = load_config(config_path)
    name = experiment_name or str(config["experiment"]["name"])
    experiment = REPOSITORY / str(config["experiment"]["output_root"]) / name
    return config, experiment, experiment / "dataset"


def prepare(args: argparse.Namespace) -> None:
    config, experiment, dataset = _paths(args.config, args.experiment_name)
    experiment.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, experiment / "config.yaml")
    manifest = prepare_controlled_dataset(config, dataset, smoke=args.smoke)
    print(json.dumps(manifest, indent=2))


def train(args: argparse.Namespace) -> None:
    from sage_avo.experiments.training import train_controlled_variant

    config, experiment, dataset = _paths(args.config, args.experiment_name)
    variants = LEARNED_VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        output = train_controlled_variant(
            repository=REPOSITORY,
            config_path=args.config,
            config=config,
            dataset_directory=dataset,
            experiment_directory=experiment,
            variant=variant,
            device_name=args.device,
        )
        print(f"Completed {variant}: {output}")


def predict(args: argparse.Namespace) -> None:
    from sage_avo.experiments.prediction import predict_controlled_variant

    config, experiment, dataset = _paths(args.config, args.experiment_name)
    variants = ALL_VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        output = predict_controlled_variant(
            repository=REPOSITORY,
            config_path=args.config,
            config=config,
            dataset_directory=dataset,
            experiment_directory=experiment,
            variant=variant,
            device_name=args.device,
        )
        print(f"Predicted {variant}: {output}")


def evaluate(args: argparse.Namespace) -> None:
    from sage_avo.evaluation.controlled import evaluate_controlled_ablation

    config, experiment, dataset = _paths(args.config, args.experiment_name)
    evaluation = config["evaluation"]
    summary, per_realization, paired, representative_id = evaluate_controlled_ablation(
        experiment_directory=experiment,
        dataset_directory=dataset,
        bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
        bootstrap_confidence=float(evaluation["bootstrap_confidence"]),
        seed=int(config["experiment"]["seed"]),
    )
    results = REPOSITORY / "results"
    summary.to_csv(results / "controlled_ablation_metrics.csv", index=False)
    per_realization.to_csv(results / "per_realization_metrics.csv", index=False)
    paired.to_csv(results / "paired_ablation_comparisons.csv", index=False)
    summary.to_csv(experiment / "controlled_ablation_metrics.csv", index=False)
    per_realization.to_csv(experiment / "per_realization_metrics.csv", index=False)
    paired.to_csv(experiment / "paired_ablation_comparisons.csv", index=False)
    representative = {
        "realization_id": representative_id,
        "selection_rule": evaluation["representative_rule"],
        "selection_metric": "full model Vp RMSE",
        "cherry_picked": False,
    }
    (experiment / "representative_realization.json").write_text(
        json.dumps(representative, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Representative realization: {representative_id}")


def figures(args: argparse.Namespace) -> None:
    from sage_avo.visualization.publication import generate_all_publication_figures

    config, experiment, dataset = _paths(args.config, args.experiment_name)
    representative_path = experiment / "representative_realization.json"
    if not representative_path.exists():
        raise FileNotFoundError("Run the evaluate command before generating figures")
    representative = json.loads(representative_path.read_text(encoding="utf-8"))
    generate_all_publication_figures(
        config=config,
        experiment_directory=experiment,
        dataset_directory=dataset,
        figures_directory=REPOSITORY / "figures",
        representative_id=int(representative["realization_id"]),
        device_name=args.device,
    )
    print("Generated four controlled publication figures in figures/")


def status(args: argparse.Namespace) -> None:
    config, experiment, dataset = _paths(args.config, args.experiment_name)
    from sage_avo.experiments.prediction import preferred_inference_checkpoint

    checks = {
        "dataset_manifest": dataset / "dataset_manifest.json",
        **{
            f"checkpoint_{variant}": preferred_inference_checkpoint(
                config, experiment / "runs" / variant
            )
            for variant in LEARNED_VARIANTS
        },
        **{
            f"predictions_{variant}": experiment / "predictions" / variant / "manifest.json"
            for variant in ALL_VARIANTS
        },
        "metrics": experiment / "controlled_ablation_metrics.csv",
        "representative": experiment / "representative_realization.json",
    }
    for label, path in checks.items():
        print(f"{'READY' if path.exists() else 'MISSING':7s} {label:28s} {path}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY / "configs" / "controlled_ablation_v1.yaml",
    )
    root.add_argument("--experiment-name", help="Override the configured output directory name")
    subparsers = root.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--smoke", action="store_true", help="Generate six tiny realizations for harness validation")
    prepare_parser.set_defaults(function=prepare)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--variant", choices=("all",) + LEARNED_VARIANTS, default="all")
    train_parser.add_argument("--device", help="Explicit Torch device, for example cuda:0")
    train_parser.set_defaults(function=train)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--variant", choices=("all",) + ALL_VARIANTS, default="all")
    predict_parser.add_argument("--device", help="Explicit Torch device")
    predict_parser.set_defaults(function=predict)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.set_defaults(function=evaluate)
    figure_parser = subparsers.add_parser("figures")
    figure_parser.add_argument("--device", help="Explicit Torch device")
    figure_parser.set_defaults(function=figures)
    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(function=status)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
