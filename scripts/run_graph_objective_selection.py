#!/usr/bin/env python3
"""Run the bounded Revision-3.3.2b graph-objective selection experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from sage_avo.config import load_config
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.diagnostics.reporting import initialize_machine_outputs
from sage_avo.experiments.manifest import file_sha256
from sage_avo.experiments.training import train_controlled_variant

import run_revision332_production as revision332
import run_revision332a_production as revision332a


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = revision332.PRIVATE
DATASET = revision332.DATASET
CONTROL_RUN = revision332a.CLEAN_PRODUCTION_EXPERIMENT / "runs" / "full"
EXPERIMENT = (
    PRIVATE
    / "stage_artifacts"
    / "stage04"
    / "sage_avo_s01_v00332b_graph_objective_selection"
)
SELECTION_CONFIG = REPOSITORY / "configs" / "graph_objective_selection_v00332b.yaml"
SELECTION_ROOT = PRIVATE / "revision332b" / "graph_objective_selection"
CONTROL_DIAGNOSTICS = SELECTION_ROOT / "control_current_smoothness_diagnostics"
VARIANTS = ("no_aux_graph_loss", "truth_edge_matching", "edge_aware_contrast")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_state_sha256(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name, value in sorted(payload["model_state"].items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _resolved(variant: str) -> tuple[dict, dict]:
    if variant not in VARIANTS:
        raise ValueError(f"Expected one of {VARIANTS}, received {variant!r}")
    config, observability = revision332a._configuration()
    selection = load_config(SELECTION_CONFIG)
    definition = copy.deepcopy(selection["variants"][variant])
    config["training"]["graph_objective"] = {
        key: value
        for key, value in definition.items()
        if key
        in {
            "mode",
            "same_layer_rgt_quantile",
            "same_layer_weight_quantile",
            "low_truth_contrast_quantile",
            "high_truth_contrast_quantile",
        }
    }
    if variant == "edge_aware_contrast":
        config["training"]["graph_objective"].update(
            {
                "same_layer_rgt_quantile": 0.25,
                "same_layer_weight_quantile": 0.75,
                "low_truth_contrast_quantile": 0.25,
                "high_truth_contrast_quantile": 0.75,
            }
        )
    config["training"]["loss_weights"]["structure"] = float(
        definition["structure_coefficient"]
    )
    config["graph_objective_selection"] = {
        "revision": selection["revision"],
        "variant": variant,
        "label": definition["label"],
        "predeclared_config": str(SELECTION_CONFIG),
        "predeclared_config_sha256": file_sha256(SELECTION_CONFIG),
        "control_run": str(CONTROL_RUN),
        "all_non_graph_training_settings_sha256": _canonical_sha256(
            {
                "model": config["model"],
                "dataset": config["dataset"],
                "patches": config["patches"],
                "training": {
                    key: value
                    for key, value in config["training"].items()
                    if key not in {"graph_objective", "loss_weights"}
                },
                "non_graph_loss_weights": {
                    key: value
                    for key, value in config["training"]["loss_weights"].items()
                    if key != "structure"
                },
                "hardware": config["hardware"],
                "seed": config["experiment"]["seed"],
            }
        ),
    }
    return config, observability


def prepare(_: argparse.Namespace) -> None:
    base_config, observability = revision332a._configuration()
    verification = revision332._verify(observability)
    control_epoch1 = CONTROL_RUN / "diagnostic_checkpoints" / "epoch_0001.pt"
    control_epoch5 = CONTROL_RUN / "diagnostic_checkpoints" / "epoch_0005.pt"
    control_payload = torch.load(control_epoch5, map_location="cpu", weights_only=False)
    if control_payload["config"] != base_config:
        raise RuntimeError("Completed control checkpoint does not match the resolved base config")
    if int(control_payload["epoch"]) != 5:
        raise RuntimeError("Completed control is not the clean epoch-5 checkpoint")
    candidate_contracts = {}
    invariant_hashes = set()
    for variant in VARIANTS:
        config, _ = _resolved(variant)
        record = config["graph_objective_selection"]
        invariant_hashes.add(record["all_non_graph_training_settings_sha256"])
        candidate_contracts[variant] = {
            "mode": config["training"]["graph_objective"]["mode"],
            "structure_coefficient": config["training"]["loss_weights"]["structure"],
            "all_non_graph_training_settings_sha256": record[
                "all_non_graph_training_settings_sha256"
            ],
            "resolved_config_sha256": _canonical_sha256(config),
        }
    if len(invariant_hashes) != 1:
        raise RuntimeError("Candidate non-graph training contracts are not identical")
    report = {
        "schema_version": 1,
        "status": "PREDECLARED_BEFORE_CANDIDATE_TRAINING",
        "selection_config": str(SELECTION_CONFIG),
        "selection_config_sha256": file_sha256(SELECTION_CONFIG),
        "immutable_input_verification": verification,
        "control": {
            "run": str(CONTROL_RUN),
            "manifest_sha256": file_sha256(CONTROL_RUN / "manifest.json"),
            "training_log_sha256": file_sha256(CONTROL_RUN / "training_log.csv"),
            "epoch1_checkpoint_sha256": file_sha256(control_epoch1),
            "epoch5_checkpoint_sha256": file_sha256(control_epoch5),
            "epoch1_model_state_sha256": _model_state_sha256(control_epoch1),
            "epoch5_model_state_sha256": _model_state_sha256(control_epoch5),
            "resolved_config_exact_match": True,
            "source_snapshot_id": control_payload["config"]["source_snapshot"]["snapshot_id"],
            "dataset_manifest_sha256": observability["frozen_inputs"][
                "stage03_manifest_sha256"
            ],
            "seed": control_payload["config"]["experiment"]["seed"],
        },
        "candidate_contracts": candidate_contracts,
        "truth_contract": load_config(SELECTION_CONFIG)["truth_contract"],
        "acceptance": load_config(SELECTION_CONFIG)["acceptance"],
    }
    SELECTION_ROOT.mkdir(parents=True, exist_ok=True)
    destination = SELECTION_ROOT / "predeclared_contract.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract": str(destination), "sha256": file_sha256(destination)}, indent=2))


def train(arguments: argparse.Namespace) -> None:
    variant = arguments.variant
    config, observability = _resolved(variant)
    revision332._verify(observability)
    run_directory = EXPERIMENT / "runs" / variant
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite graph-selection run: {run_directory}")
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=revision332.CONFIG_PATH,
        config=config,
        dataset_directory=DATASET,
        experiment_directory=EXPERIMENT,
        variant="full",
        device_name=arguments.device,
        run_name=variant,
        stop_after_epoch=5,
    )
    print(output)


def _diagnose_checkpoint(run_directory: Path, checkpoint: Path, output: Path, device: str) -> None:
    initialize_machine_outputs(output)
    analyze_checkpoint(
        checkpoint_path=checkpoint,
        dataset_directory=DATASET,
        run_directory=run_directory,
        sample_manifest_path=revision332.SAMPLE_MANIFEST,
        output_directory=output,
        device=device,
        include_whole_realizations=True,
    )


def diagnose(arguments: argparse.Namespace) -> None:
    if arguments.variant == "current_smoothness":
        run_directory = CONTROL_RUN
        output = CONTROL_DIAGNOSTICS
    else:
        run_directory = EXPERIMENT / "runs" / arguments.variant
        output = run_directory / "selection_diagnostics"
    for epoch in (1, 5):
        checkpoint = run_directory / "diagnostic_checkpoints" / f"epoch_{epoch:04d}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        _diagnose_checkpoint(run_directory, checkpoint, output, arguments.device)
    print(json.dumps({"variant": arguments.variant, "output": str(output)}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.set_defaults(function=prepare)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--variant", required=True, choices=VARIANTS)
    train_parser.add_argument("--device", default="cuda")
    train_parser.set_defaults(function=train)
    diagnose_parser = commands.add_parser("diagnose")
    diagnose_parser.add_argument(
        "--variant", required=True, choices=("current_smoothness", *VARIANTS)
    )
    diagnose_parser.add_argument("--device", default="cuda")
    diagnose_parser.set_defaults(function=diagnose)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
