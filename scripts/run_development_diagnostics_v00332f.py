#!/usr/bin/env python3
"""Run a short frozen-patch SAGE-AVO mechanism diagnostic without touching v00332e."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from sage_avo.config import load_config
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.data.sampling import build_patch_sampling_weights
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.diagnostics.contracts import build_diagnostic_sample_manifest
from sage_avo.experiments.training import _sampling, train_controlled_variant


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332f.yaml"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve() -> tuple[dict, dict, Path, Path]:
    contract = load_config(CONTRACT_PATH)
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    config = deepcopy(load_config(REPOSITORY / "configs" / contract["base_training_config"]))
    budget = contract["fixed_patch_budget"]
    config["dataset"]["directory"] = f"datasets/{contract['immutable_dataset']}"
    config["experiment"]["name"] = str(contract["experiment_name"])
    config["training"]["epochs"] = int(budget["epochs"])
    config["training"]["batch_size"] = int(budget["batch_size"])
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"]["enabled"] = False
    config["training"]["contrastive_loss"]["weight"] = 0.0
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["guidance_scale"] = 0.0
    config["observability"] = load_config(
        REPOSITORY / "configs" / contract["observability_config"]
    )
    config["observability"]["revision"] = str(contract["revision"])
    config["observability"]["scientific_methodology_changed"] = False
    config["observability"]["diagnostics"]["deterministic_time"] = float(
        contract["diagnostic_probes"]["deterministic_time"]
    )
    config["observability"]["diagnostics"]["flow_integration_steps"] = int(
        contract["diagnostic_probes"]["flow_integration_steps"]
    )
    dataset = private / "stage_artifacts" / "stage03" / contract["immutable_dataset"] / "dataset"
    experiment = private / "stage_artifacts" / "stage04" / config["experiment"]["name"]
    return contract, config, dataset, experiment


def _selection(contract: dict, config: dict, dataset: Path) -> dict:
    """Freeze the exact patch positions before the first pilot optimizer step."""
    budget = contract["fixed_patch_budget"]
    train = IndexedRealizationPatches(dataset, "train")
    validation = IndexedRealizationPatches(dataset, "validation")
    patch_count = int(budget["train_batches_per_epoch"]) * int(budget["batch_size"])
    weights = build_patch_sampling_weights(train, _sampling(config))
    generator = torch.Generator().manual_seed(int(contract["seed"]) + 13)
    indices = torch.multinomial(weights, patch_count, replacement=True, generator=generator)
    validation_count = int(budget["validation_batches_per_epoch"]) * int(budget["batch_size"])
    if validation_count > len(validation):
        raise ValueError("The fixed validation budget exceeds the validation patch index")
    train_indices = [int(value) for value in indices.tolist()]
    validation_indices = list(range(validation_count))
    return {
        "selection_algorithm": "torch.multinomial(weighted_patch_sampling_weights)",
        "selection_seed": int(contract["seed"]) + 13,
        "train_patch_count": len(train_indices),
        "validation_patch_count": len(validation_indices),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_indices_sha256": canonical_hash(train_indices),
        "validation_indices_sha256": canonical_hash(validation_indices),
        "repeats_same_train_patches_each_epoch": True,
    }


def _paths(experiment: Path) -> tuple[Path, Path, Path]:
    return (
        experiment / "development_diagnostic_contract.json",
        experiment / "fixed_patch_selection.json",
        experiment / "fixed_graph_probe_samples.json",
    )


def prepare(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment = resolve()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    experiment.mkdir(parents=True, exist_ok=True)
    contract_path, selection_path, samples_path = _paths(experiment)
    if any(path.exists() for path in (contract_path, selection_path, samples_path)):
        raise FileExistsError("Diagnostic contract already exists; refuse to replace its frozen selection")
    selection = _selection(contract, config, dataset)
    sample_manifest = build_diagnostic_sample_manifest(
        dataset_directory=dataset,
        observability_config=config["observability"],
        destination=samples_path,
    )
    payload = {
        "status": "PREPARED_NOT_TRAINED",
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "resolved_config_sha256": canonical_hash(config),
        "dataset": str(dataset),
        "dataset_manifest": str(dataset / "dataset_manifest.json"),
        "experiment": str(experiment),
        "fixed_patch_selection": str(selection_path),
        "fixed_graph_probe_samples": str(samples_path),
        "fixed_graph_probe_sample_count": len(sample_manifest["patches"]),
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    contract_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _completed_epoch(run: Path) -> int:
    manifest = run / "manifest.json"
    if not manifest.exists():
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("last_completed_epoch", 0))


def status(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment = resolve()
    _, selection_path, samples_path = _paths(experiment)
    run = experiment / "runs" / "full"
    result = {
        "revision": contract["revision"],
        "diagnostic_only": True,
        "dataset_ready": (dataset / "dataset_manifest.json").exists(),
        "resolved_config_sha256": canonical_hash(config),
        "selection_ready": selection_path.exists(),
        "probe_samples_ready": samples_path.exists(),
        "run": str(run),
        "last_completed_epoch": _completed_epoch(run),
        "resumable": (run / "last.pt").exists(),
        "probe_output": str(run / "gradient_graph_probe"),
    }
    print(json.dumps(result, indent=2))


def train(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment = resolve()
    contract_path, selection_path, samples_path = _paths(experiment)
    if not all(path.exists() for path in (contract_path, selection_path, samples_path)):
        raise FileNotFoundError("Run the prepare command before training diagnostics")
    prepared = json.loads(contract_path.read_text(encoding="utf-8"))
    if prepared.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Diagnostic contract is stale; create a new revision rather than replacing it")
    if prepared.get("resolved_config_sha256") != canonical_hash(config):
        raise RuntimeError("Resolved diagnostic config is stale; create a new revision")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    run = experiment / "runs" / "full"
    completed = _completed_epoch(run)
    maximum = int(contract["fixed_patch_budget"]["epochs"])
    target = int(args.until_epoch or maximum)
    if not 1 <= target <= maximum:
        raise ValueError(f"--until-epoch must lie in [1, {maximum}]")
    if target <= completed:
        raise ValueError(f"Target epoch {target} is not after completed epoch {completed}")
    for epoch in range(completed + 1, target + 1):
        train_controlled_variant(
            repository=REPOSITORY,
            config_path=CONTRACT_PATH,
            config=config,
            dataset_directory=dataset,
            experiment_directory=experiment,
            variant="full",
            device_name=args.device,
            epochs_override=maximum,
            max_train_batches=int(contract["fixed_patch_budget"]["train_batches_per_epoch"]),
            max_validation_batches=int(contract["fixed_patch_budget"]["validation_batches_per_epoch"]),
            run_name="full",
            resume_from=(run / "last.pt") if (run / "last.pt").exists() else None,
            stop_after_epoch=epoch,
            fixed_train_indices=selection["train_indices"],
            fixed_validation_indices=selection["validation_indices"],
            finite_state_check_batches=(1, 16, 32),
            abort_on_nonfinite=True,
        )
        analyze_checkpoint(
            checkpoint_path=run / "last.pt",
            dataset_directory=dataset,
            run_directory=run,
            sample_manifest_path=samples_path,
            output_directory=run / "gradient_graph_probe",
            device=args.device,
            maximum_patches=int(contract["diagnostic_probes"]["probe_patch_count"]),
            flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
            include_whole_realizations=bool(
                contract["diagnostic_probes"]["include_whole_realizations"]
            ),
        )
        print(f"Completed diagnostic epoch {epoch} and its checkpoint-only probes")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(function=prepare)
    commands.add_parser("status").set_defaults(function=status)
    training = commands.add_parser("train")
    training.add_argument("--device", required=True, help="Explicit device such as cuda:0")
    training.add_argument("--until-epoch", type=int, help="Absolute diagnostic epoch to stop after")
    training.set_defaults(function=train)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
