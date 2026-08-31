#!/usr/bin/env python3
"""Run the frozen epoch-40 full-model versus low-prior baseline evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from sage_avo.config import load_config
from sage_avo.evaluation.controlled import evaluate_controlled_ablation
from sage_avo.experiments.prediction import predict_controlled_variant
from sage_avo.runtime import print_torch_runtime, select_torch_device


REPOSITORY = Path(__file__).resolve().parents[1]


def final_config() -> tuple[dict, dict]:
    """Resolve the public, inference-relevant v00332d configuration."""
    contract = load_config(REPOSITORY / "configs" / "final_training_v00332d.yaml")
    config = deepcopy(
        load_config(REPOSITORY / "configs" / str(contract["base_training_config"]))
    )
    config["dataset"]["directory"] = f"datasets/{contract['immutable_dataset']}"
    config["experiment"]["name"] = "sage_avo_s01_v00332d_final_stable_training"
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["physics_guided_sampling"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["guidance_scale"] = 0.0
    config["final_training_contract"] = contract
    return config, contract


def locations(args: argparse.Namespace) -> tuple[dict, dict, Path, Path, Path]:
    config, contract = final_config()
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    dataset = Path(args.dataset) if args.dataset else (
        private / "stage_artifacts" / "stage03" / contract["immutable_dataset"] / "dataset"
    )
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        private
        / "stage_artifacts"
        / "stage04"
        / "sage_avo_s01_v00332d_final_production"
        / "runs"
        / "full"
        / "best_whole_realization.pt"
    )
    output = Path(args.output) if args.output else (
        private / "stage_artifacts" / "stage05" / "v00332d_epoch40_baseline"
    )
    return config, contract, dataset, checkpoint, output


def status(args: argparse.Namespace) -> None:
    _, contract, dataset, checkpoint, output = locations(args)
    values = {
        "stage03": contract["immutable_dataset"],
        "dataset_ready": (dataset / "dataset_manifest.json").exists(),
        "checkpoint_ready": checkpoint.exists(),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "low_prior_predictions": (output / "predictions" / "low_prior" / "manifest.json").exists(),
        "full_predictions": (output / "predictions" / "full" / "manifest.json").exists(),
        "summary": (output / "baseline_summary.csv").exists(),
    }
    print(json.dumps(values, indent=2))


def predict(args: argparse.Namespace) -> None:
    config, _, dataset, checkpoint, output = locations(args)
    print_torch_runtime()
    device = select_torch_device(
        args.device, require_cuda=True, context="epoch-40 baseline inference"
    )
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Selected epoch-40 checkpoint is unavailable: {checkpoint}")
    for variant in ("low_prior", "full"):
        predict_controlled_variant(
            repository=REPOSITORY,
            config_path=REPOSITORY / "configs" / "final_training_v00332d.yaml",
            config=config,
            dataset_directory=dataset,
            experiment_directory=output,
            prediction_directory=output,
            checkpoint_path=checkpoint if variant == "full" else None,
            variant=variant,
            device_name=str(device),
            require_cuda=True,
            inference_batch_size=args.batch_size,
        )
    print(f"Baseline predictions written to {output / 'predictions'}")


def evaluate(args: argparse.Namespace) -> None:
    config, _, dataset, _, output = locations(args)
    summary, per_realization, paired, representative = evaluate_controlled_ablation(
        experiment_directory=output,
        dataset_directory=dataset,
        bootstrap_repetitions=int(config["evaluation"]["bootstrap_repetitions"]),
        bootstrap_confidence=float(config["evaluation"]["bootstrap_confidence"]),
        seed=int(config["experiment"]["seed"]),
        variants=("low_prior", "full"),
    )
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "baseline_summary.csv", index=False)
    per_realization.to_csv(output / "baseline_per_realization.csv", index=False)
    paired.to_csv(output / "baseline_paired_improvements.csv", index=False)
    (output / "representative_realization.json").write_text(
        json.dumps(
            {
                "realization_id": representative,
                "selection_rule": "median full-model Vp RMSE",
                "checkpoint": "best_whole_realization.pt (epoch 40)",
                "test_used_for_checkpoint_selection": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--dataset", type=Path)
    root.add_argument("--checkpoint", type=Path)
    root.add_argument("--output", type=Path)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("status").set_defaults(function=status)
    prediction = commands.add_parser("predict")
    prediction.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda); production baseline refuses CPU fallback",
    )
    prediction.add_argument(
        "--batch-size", type=int, default=2, help="Tiling batch size (default: 2 for 4-GB VRAM)"
    )
    prediction.set_defaults(function=predict)
    commands.add_parser("evaluate").set_defaults(function=evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
