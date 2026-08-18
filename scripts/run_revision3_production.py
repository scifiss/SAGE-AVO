#!/usr/bin/env python3
"""Run guarded Revision-3 production stages using only versioned v003 paths.

This entry point is intentionally separate from the bounded validation driver.
It does not run any stage unless an explicit subcommand is supplied, and it
refuses to restart an existing training run without ``--resume``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sage_avo.config import load_config
from sage_avo.evaluation.controlled import evaluate_controlled_ablation
from sage_avo.experiments import build_stage03_dataset, generate_stage02_dataset
from sage_avo.experiments.manifest import load_frozen_source_reference, write_json
from sage_avo.experiments.prediction import predict_controlled_variant
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.models import LEARNED_VARIANTS


REPOSITORY = Path(__file__).resolve().parents[1]
ALL_VARIANTS = ("low_prior",) + LEARNED_VARIANTS


def _contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    paths_file = REPOSITORY / "configs" / "paths.yaml"
    if not paths_file.exists():
        raise FileNotFoundError("Missing local configuration: configs/paths.yaml")
    paths = load_config(paths_file)
    synthetic = load_config(REPOSITORY / "configs" / "synthetic_s01_v003.yaml")
    dataset = load_config(REPOSITORY / "configs" / "ml_dataset_s01_v003.yaml")
    training = load_config(REPOSITORY / "configs" / "sage_avo_s01_v003.yaml")
    private = Path(paths["private_artifact_root"])
    source_snapshot = load_frozen_source_reference(private)
    for config in (synthetic, dataset, training):
        config["source_snapshot"] = source_snapshot
    locations = {
        "stage02": private
        / "stage_artifacts"
        / "stage02"
        / str(synthetic["outputs"]["version"])
        / "realizations",
        "stage03": private
        / "stage_artifacts"
        / "stage03"
        / str(dataset["outputs"]["version"])
        / "dataset",
        "stage04": private
        / "stage_artifacts"
        / "stage04"
        / str(training["experiment"]["name"]),
    }
    return synthetic, dataset, training, locations


def _resolved_training_snapshot(training: dict[str, Any], experiment: Path) -> Path:
    destination = experiment / "configs" / "training_resolved.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, training)
    return destination


def stage02(args: argparse.Namespace) -> None:
    synthetic, _, _, locations = _contracts()
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=load_config(REPOSITORY / "configs" / "paths.yaml"),
        output_directory=locations["stage02"],
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(json.dumps(manifest, indent=2))


def stage03(args: argparse.Namespace) -> None:
    _, dataset, _, locations = _contracts()
    manifest_path = locations["stage03"] / "dataset_manifest.json"
    if manifest_path.exists() and not args.rebuild:
        raise FileExistsError(
            f"Stage-03 dataset already exists at {locations['stage03']}; "
            "--rebuild explicitly replaces that versioned target"
        )
    manifest = build_stage03_dataset(
        config=dataset,
        paths=load_config(REPOSITORY / "configs" / "paths.yaml"),
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    print(json.dumps(manifest["integrity"], indent=2))


def train(args: argparse.Namespace) -> None:
    _, _, training, locations = _contracts()
    run_directory = locations["stage04"] / "runs" / args.variant
    last = run_directory / "last.pt"
    if last.exists() and not args.resume:
        raise FileExistsError(
            f"Existing v003 run found at {run_directory}; use --resume to continue it"
        )
    if args.resume and not last.exists():
        raise FileNotFoundError(f"Cannot resume because {last} does not exist")
    config_path = _resolved_training_snapshot(training, locations["stage04"])
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=config_path,
        config=training,
        dataset_directory=locations["stage03"],
        experiment_directory=locations["stage04"],
        variant=args.variant,
        device_name=args.device,
        resume_from=last if args.resume else None,
    )
    print(output)


def predict(args: argparse.Namespace) -> None:
    _, _, training, locations = _contracts()
    config_path = _resolved_training_snapshot(training, locations["stage04"])
    variants = ALL_VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        output = predict_controlled_variant(
            repository=REPOSITORY,
            config_path=config_path,
            config=training,
            dataset_directory=locations["stage03"],
            experiment_directory=locations["stage04"],
            variant=variant,
            device_name=args.device,
        )
        print(f"Predicted {variant}: {output}")


def evaluate(_: argparse.Namespace) -> None:
    _, _, training, locations = _contracts()
    evaluation = training["evaluation"]
    summary, per_realization, paired, representative_id = evaluate_controlled_ablation(
        experiment_directory=locations["stage04"],
        dataset_directory=locations["stage03"],
        bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
        bootstrap_confidence=float(evaluation["bootstrap_confidence"]),
        seed=int(training["experiment"]["seed"]),
    )
    destination = locations["stage04"] / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "controlled_ablation_metrics.csv", index=False)
    per_realization.to_csv(destination / "per_realization_metrics.csv", index=False)
    paired.to_csv(destination / "paired_ablation_comparisons.csv", index=False)
    write_json(
        destination / "representative_realization.json",
        {
            "realization_id": representative_id,
            "selection_rule": evaluation["representative_rule"],
            "selection_metric": "full-model Vp RMSE",
            "cherry_picked": False,
        },
    )
    write_json(
        destination / "evaluation_manifest.json",
        {
            "schema_version": 3,
            "stage": "05_controlled_whole_realization_evaluation",
            "status": "complete",
            "source_snapshot": training["source_snapshot"],
            "test_data_used_for_checkpoint_selection": False,
            "variants": list(ALL_VARIANTS),
            "representative_realization_id": representative_id,
            "metric_definitions": evaluation,
        },
    )
    print(summary.to_string(index=False))
    print(f"Representative realization: {representative_id}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("stage02")
    generation.add_argument("--workers", type=int, default=1)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(function=stage02)
    dataset = commands.add_parser("stage03")
    dataset.add_argument("--rebuild", action="store_true")
    dataset.set_defaults(function=stage03)
    training = commands.add_parser("train")
    training.add_argument("--variant", choices=LEARNED_VARIANTS, required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--resume", action="store_true")
    training.set_defaults(function=train)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--variant", choices=("all",) + ALL_VARIANTS, default="all")
    prediction.add_argument("--device", default="cuda")
    prediction.set_defaults(function=predict)
    evaluation = commands.add_parser("evaluate")
    evaluation.set_defaults(function=evaluate)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
