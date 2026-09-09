#!/usr/bin/env python3
"""Prepare and run the v00332i top-2 relation-candidate diagnostic."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from sage_avo.config import load_config
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332i.yaml"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def resolve() -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    contract = load_config(CONTRACT_PATH)
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    config = deepcopy(load_config(REPOSITORY / "configs" / contract["base_training_config"]))
    budget = contract["fixed_patch_budget"]
    graph = contract["experimental_graph"]
    config["dataset"]["directory"] = f"datasets/{contract['immutable_dataset']}"
    config["experiment"]["name"] = str(contract["experiment_name"])
    config["experiment"]["seed"] = int(contract["seed"])
    config["training"]["epochs"] = int(budget["epochs"])
    config["training"]["batch_size"] = int(budget["batch_size"])
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"]["enabled"] = False
    config["training"]["contrastive_loss"]["weight"] = 0.0
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["guidance_scale"] = 0.0
    config["model"]["experimental_graph"] = {
        "mode": str(graph["mode"]),
        "normal_lateral_shift_samples": int(graph["normal_lateral_shift_samples"]),
        "relation_candidates": int(graph["relation_candidates"]),
    }
    config["capabilities"]["top2_candidate_relation_attention"] = {
        "implemented": True,
        "enabled": True,
        "diagnostic_only": True,
    }
    config["observability"] = load_config(
        REPOSITORY / "configs" / contract["observability_config"]
    )
    config["observability"]["revision"] = str(contract["revision"])
    config["observability"]["scientific_methodology_changed"] = True
    config["observability"]["diagnostics"]["deterministic_time"] = float(
        contract["diagnostic_probes"]["deterministic_time"]
    )
    config["observability"]["diagnostics"]["flow_integration_steps"] = int(
        contract["diagnostic_probes"]["flow_integration_steps"]
    )
    dataset = private / "stage_artifacts" / "stage03" / contract["immutable_dataset"] / "dataset"
    experiment = private / "stage_artifacts" / "stage04" / config["experiment"]["name"]
    source = private / "stage_artifacts" / "stage04" / contract["source_experiment"][
        "experiment_name"
    ]
    return contract, config, dataset, experiment, source


def _source_paths(source: Path) -> tuple[Path, Path, Path]:
    return (
        source / "fixed_patch_selection.json",
        source / "fixed_graph_probe_samples.json",
        source / "initial_model_state.pt",
    )


def _paths(experiment: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        experiment / "development_diagnostic_contract.json",
        experiment / "fixed_patch_selection.json",
        experiment / "fixed_graph_probe_samples.json",
        experiment / "initial_model_state.pt",
        experiment / "initialization_manifest.json",
    )


def _validate_source(
    contract: dict[str, Any], config: dict[str, Any], source: Path
) -> dict[str, Any]:
    expected = contract["source_experiment"]
    selection_path, probes_path, state_path = _source_paths(source)
    file_checks = (
        (selection_path, "expected_selection_file_sha256"),
        (probes_path, "expected_probe_file_sha256"),
        (state_path, "expected_initial_state_file_sha256"),
    )
    for path, field in file_checks:
        if not path.exists() or file_sha256(path) != expected[field]:
            raise RuntimeError(f"Frozen v00332h source mismatch: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for field, count_field, hash_field in (
        ("train_indices", "expected_train_patch_count", "expected_train_indices_sha256"),
        (
            "validation_indices",
            "expected_validation_patch_count",
            "expected_validation_indices_sha256",
        ),
    ):
        values = [int(value) for value in selection[field]]
        if len(values) != int(expected[count_field]) or canonical_hash(values) != expected[
            hash_field
        ]:
            raise RuntimeError(f"Frozen v00332h {field} mismatch")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state_sha256(state) != expected["expected_initial_model_state_sha256"]:
        raise RuntimeError("Frozen v00332h initial model-state hash mismatch")
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config))
    model.load_state_dict(state, strict=True)
    return {
        "source_experiment": str(source),
        "selection_file_sha256": file_sha256(selection_path),
        "probe_file_sha256": file_sha256(probes_path),
        "initial_state_file_sha256": file_sha256(state_path),
        "initial_model_state_sha256": state_sha256(model.state_dict()),
        "architecture_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def prepare(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment, source = resolve()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    verification = _validate_source(contract, config, source)
    experiment.mkdir(parents=True, exist_ok=True)
    outputs = _paths(experiment)
    if any(path.exists() for path in outputs):
        raise FileExistsError("v00332i contract already exists; refusing to replace frozen inputs")
    _, selection_path, probes_path, state_path, initialization_path = outputs
    source_selection, source_probes, source_state = _source_paths(source)
    shutil.copyfile(source_selection, selection_path)
    shutil.copyfile(source_probes, probes_path)
    shutil.copyfile(source_state, state_path)
    verification["copied_files_byte_identical"] = all(
        file_sha256(source_path) == file_sha256(destination)
        for source_path, destination in zip(
            (source_selection, source_probes, source_state),
            (selection_path, probes_path, state_path),
        )
    )
    initialization_path.write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "status": "PREPARED_NOT_TRAINED",
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "resolved_config_sha256": canonical_hash(config),
        "dataset": str(dataset),
        "experiment": str(experiment),
        "fixed_patch_selection": str(selection_path),
        "fixed_graph_probe_samples": str(probes_path),
        "initial_model_state": str(state_path),
        "initialization_manifest": str(initialization_path),
        "source_verification": verification,
    }
    outputs[0].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _completed_epoch(run: Path) -> int:
    manifest = run / "manifest.json"
    if not manifest.exists():
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("last_completed_epoch", 0))


def status(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment, _ = resolve()
    run = experiment / "runs" / "full"
    print(
        json.dumps(
            {
                "revision": contract["revision"],
                "diagnostic_only": True,
                "dataset_ready": (dataset / "dataset_manifest.json").exists(),
                "resolved_config_sha256": canonical_hash(config),
                "prepared_files_ready": all(path.exists() for path in _paths(experiment)),
                "run": str(run),
                "last_completed_epoch": _completed_epoch(run),
                "resumable": (run / "last.pt").exists(),
                "probe_output": str(run / "gradient_graph_probe"),
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment, _ = resolve()
    contract_path, selection_path, probes_path, state_path, initialization_path = _paths(
        experiment
    )
    if not all(path.exists() for path in _paths(experiment)):
        raise FileNotFoundError("Run the prepare command before v00332i training")
    prepared = json.loads(contract_path.read_text(encoding="utf-8"))
    if prepared["contract_sha256"] != canonical_hash(contract):
        raise RuntimeError("v00332i contract is stale; create a new revision")
    if prepared["resolved_config_sha256"] != canonical_hash(config):
        raise RuntimeError("v00332i resolved config is stale; create a new revision")
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if initialization["initial_state_file_sha256"] != file_sha256(state_path):
        raise RuntimeError("v00332i initial state file SHA-256 mismatch")
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
            max_validation_batches=int(
                contract["fixed_patch_budget"]["validation_batches_per_epoch"]
            ),
            run_name="full",
            resume_from=(run / "last.pt") if (run / "last.pt").exists() else None,
            stop_after_epoch=epoch,
            fixed_train_indices=selection["train_indices"],
            fixed_validation_indices=selection["validation_indices"],
            initial_model_state=state_path,
            finite_state_check_batches=(1, 16, 32),
            abort_on_nonfinite=True,
        )
        analyze_checkpoint(
            checkpoint_path=run / "last.pt",
            dataset_directory=dataset,
            run_directory=run,
            sample_manifest_path=probes_path,
            output_directory=run / "gradient_graph_probe",
            device=args.device,
            maximum_patches=int(contract["diagnostic_probes"]["probe_patch_count"]),
            flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
            include_whole_realizations=bool(
                contract["diagnostic_probes"]["include_whole_realizations"]
            ),
        )
        print(f"Completed v00332i epoch {epoch} and its checkpoint-only probes")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(function=prepare)
    commands.add_parser("status").set_defaults(function=status)
    training = commands.add_parser("train")
    training.add_argument("--device", required=True, help="Explicit device such as cuda")
    training.add_argument("--until-epoch", type=int, help="Absolute epoch to stop after")
    training.set_defaults(function=train)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
