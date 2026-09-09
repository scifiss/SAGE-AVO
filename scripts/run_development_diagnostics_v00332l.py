#!/usr/bin/env python3
"""Prepare and run the v00332l task-specific structural-attention diagnostic."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
from typing import Any

import torch
import torch.nn.functional as F

from run_development_diagnostics_v00332j import (
    _completed_epoch,
    canonical_hash,
    file_sha256,
    state_sha256,
)
from sage_avo.config import load_config, seed_everything
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.diagnostics.elastic_strength_sweep import run_elastic_strength_sweep
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332l.yaml"


def resolve() -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    contract = load_config(CONTRACT_PATH)
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    config = deepcopy(load_config(REPOSITORY / "configs" / contract["base_training_config"]))
    budget = contract["fixed_patch_budget"]
    graph = contract["experimental_graph"]
    prior = graph["task_specific_structural_attention"]
    config["dataset"]["directory"] = f"datasets/{contract['immutable_dataset']}"
    config["experiment"]["name"] = str(contract["experiment_name"])
    config["experiment"]["seed"] = int(contract["seed"])
    config["training"]["epochs"] = int(budget["epochs"])
    config["training"]["batch_size"] = int(budget["batch_size"])
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"].update(enabled=False, weight=0.0)
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"].update(enabled=False, guidance_scale=0.0)
    config["model"]["experimental_graph"] = {
        "mode": str(graph["mode"]),
        "normal_lateral_shift_samples": int(graph["normal_lateral_shift_samples"]),
        "relation_candidates": int(graph["relation_candidates"]),
        "structural_prior_initial_strengths": [
            [float(value) for value in prior["segmentation_initial_strengths"]["tangential"]],
            [float(value) for value in prior["segmentation_initial_strengths"]["normal"]],
        ],
    }
    if "elastic_structural_prior_initial_strengths" in graph:
        config["model"]["experimental_graph"]["elastic_structural_prior_initial_strengths"] = [
            [float(value) for value in relation]
            for relation in graph["elastic_structural_prior_initial_strengths"]
        ]
    config["capabilities"]["task_specific_structural_attention"] = {
        "implemented": True,
        "enabled": True,
        "diagnostic_only": True,
        "added_trainable_scalars": int(prior["added_trainable_scalar_count"]),
        "adds_auxiliary_loss": False,
    }
    config["observability"] = load_config(REPOSITORY / "configs" / contract["observability_config"])
    config["observability"]["revision"] = str(contract["revision"])
    config["observability"]["scientific_methodology_changed"] = True
    diagnostics = config["observability"]["diagnostics"]
    diagnostics["deterministic_time"] = float(contract["diagnostic_probes"]["deterministic_time"])
    diagnostics["flow_integration_steps"] = int(
        contract["diagnostic_probes"]["flow_integration_steps"]
    )
    dataset = private / "stage_artifacts" / "stage03" / contract["immutable_dataset"] / "dataset"
    experiment = private / "stage_artifacts" / "stage04" / config["experiment"]["name"]
    source = (
        private / "stage_artifacts" / "stage04" / contract["source_experiment"]["experiment_name"]
    )
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


def _validate_source(contract: dict[str, Any], source: Path) -> dict[str, Any]:
    expected = contract["source_experiment"]
    selection_path, probes_path, state_path = _source_paths(source)
    for path, field in (
        (selection_path, "expected_selection_file_sha256"),
        (probes_path, "expected_probe_file_sha256"),
        (state_path, "expected_initial_state_file_sha256"),
    ):
        if not path.exists() or file_sha256(path) != expected[field]:
            raise RuntimeError(f"Frozen source mismatch for {contract['revision']}: {path}")
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
        if (
            len(values) != int(expected[count_field])
            or canonical_hash(values) != expected[hash_field]
        ):
            raise RuntimeError(f"Frozen {field} mismatch for {contract['revision']}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if state_sha256(state) != expected["expected_initial_model_state_sha256"]:
        raise RuntimeError(f"Frozen initial model-state mismatch for {contract['revision']}")
    return {
        "source_experiment": str(source),
        "selection_file_sha256": file_sha256(selection_path),
        "probe_file_sha256": file_sha256(probes_path),
        "source_initial_state_file_sha256": file_sha256(state_path),
        "source_initial_model_state_sha256": state_sha256(state),
    }


def _prepared_initialization(
    contract: dict[str, Any], config: dict[str, Any], source_state_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source = torch.load(source_state_path, map_location="cpu", weights_only=True)
    seed_everything(int(contract["seed"]), deterministic_torch=True)
    model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config))
    compatibility = model.load_state_dict(source, strict=False)
    expected_missing = ["graph.elastic_attention_raw_strengths"]
    if compatibility.missing_keys != expected_missing or compatibility.unexpected_keys:
        raise RuntimeError(
            f"{contract['revision']} initialization differs from its source beyond "
            "the declared elastic "
            f"strengths: missing={compatibility.missing_keys}, "
            f"unexpected={compatibility.unexpected_keys}"
        )
    state = dict(model.state_dict())
    common_equal = all(torch.equal(state[name], value) for name, value in source.items())
    if not common_equal:
        raise RuntimeError("A common source tensor changed during initialization")
    strengths = F.softplus(state["graph.elastic_attention_raw_strengths"])
    configured_strengths = contract["experimental_graph"]["task_specific_structural_attention"][
        "elastic_initial_strengths"
    ]
    if isinstance(configured_strengths, dict):
        configured_strengths = [
            configured_strengths["tangential"],
            configured_strengths["normal"],
        ]
    expected_strengths = torch.as_tensor(configured_strengths, dtype=strengths.dtype)
    if not torch.equal(strengths, expected_strengths):
        raise RuntimeError(f"{contract['revision']} elastic strengths do not match the contract")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_count = int(contract["source_experiment"]["expected_architecture_parameter_count"])
    if parameter_count != expected_count + strengths.numel():
        raise RuntimeError("Parameter-capacity delta is not exactly four scalars")
    return state, {
        "source_model_state_sha256": state_sha256(source),
        "prepared_model_state_sha256": state_sha256(state),
        "common_tensor_count": len(source),
        "common_tensors_byte_identical": common_equal,
        "added_parameter_names": expected_missing,
        "added_trainable_scalar_count": int(strengths.numel()),
        "initial_positive_elastic_strengths": strengths.tolist(),
        "architecture_parameter_count": parameter_count,
    }


def prepare(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment, source = resolve()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    verification = _validate_source(contract, source)
    experiment.mkdir(parents=True, exist_ok=True)
    outputs = _paths(experiment)
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            f"{contract['revision']} already exists; refusing to replace frozen inputs"
        )
    contract_path, selection_path, probes_path, state_path, initialization_path = outputs
    source_selection, source_probes, source_state = _source_paths(source)
    state, initialization = _prepared_initialization(contract, config, source_state)
    shutil.copyfile(source_selection, selection_path)
    shutil.copyfile(source_probes, probes_path)
    torch.save(state, state_path)
    initialization.update(verification)
    initialization["prepared_state_file_sha256"] = file_sha256(state_path)
    initialization["selection_and_probes_byte_identical"] = file_sha256(
        source_selection
    ) == file_sha256(selection_path) and file_sha256(source_probes) == file_sha256(probes_path)
    initialization_path.write_text(json.dumps(initialization, indent=2) + "\n", encoding="utf-8")
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
        "initial_model_state_sha256": initialization["prepared_model_state_sha256"],
    }
    contract_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


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


def _release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment, _ = resolve()
    contract_path, selection_path, probes_path, state_path, initialization_path = _paths(experiment)
    if not all(path.exists() for path in _paths(experiment)):
        raise FileNotFoundError(f"Run prepare before {contract['revision']} training")
    prepared = json.loads(contract_path.read_text(encoding="utf-8"))
    if prepared["contract_sha256"] != canonical_hash(contract):
        raise RuntimeError(f"{contract['revision']} contract is stale; create a new revision")
    if prepared["resolved_config_sha256"] != canonical_hash(config):
        raise RuntimeError(
            f"{contract['revision']} resolved config is stale; create a new revision"
        )
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if initialization["prepared_state_file_sha256"] != file_sha256(state_path):
        raise RuntimeError(f"{contract['revision']} initial state file SHA-256 mismatch")
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
        _release_cuda_cache()
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
        _release_cuda_cache()
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
        _release_cuda_cache()
        print(f"Completed {contract['revision']} epoch {epoch} and its checkpoint-only probes")


def sweep(args: argparse.Namespace) -> None:
    contract, _, dataset, experiment, _ = resolve()
    run = experiment / "runs" / "full"
    checkpoint = run / "last.pt"
    if _completed_epoch(run) != int(contract["fixed_patch_budget"]["epochs"]):
        raise RuntimeError(f"Complete {contract['revision']} before its read-only sweep")
    report = run_elastic_strength_sweep(
        checkpoint_path=checkpoint,
        dataset_directory=dataset,
        run_directory=run,
        fixed_selection_path=experiment / "fixed_patch_selection.json",
        probe_manifest_path=experiment / "fixed_graph_probe_samples.json",
        output_directory=run / "elastic_strength_sweep",
        strengths=args.strengths,
        device_name=args.device,
        flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
    )
    print(json.dumps(report, indent=2))


def component_sweep(args: argparse.Namespace) -> None:
    contract, _, dataset, experiment, _ = resolve()
    run = experiment / "runs" / "full"
    checkpoint = run / "last.pt"
    if _completed_epoch(run) != int(contract["fixed_patch_budget"]["epochs"]):
        raise RuntimeError(f"Complete {contract['revision']} before its read-only sweep")
    strength = float(args.strength)
    zero = 0.0
    settings = [
        ("zero", ((zero, zero), (zero, zero))),
        ("tangential_rgt_only", ((strength, zero), (zero, zero))),
        ("tangential_avo_only", ((zero, strength), (zero, zero))),
        ("tangential_both", ((strength, strength), (zero, zero))),
        ("normal_rgt_only", ((zero, zero), (strength, zero))),
        ("normal_avo_only", ((zero, zero), (zero, strength))),
        ("normal_both", ((zero, zero), (strength, strength))),
        ("all_components", ((strength, strength), (strength, strength))),
    ]
    report = run_elastic_strength_sweep(
        checkpoint_path=checkpoint,
        dataset_directory=dataset,
        run_directory=run,
        fixed_selection_path=experiment / "fixed_patch_selection.json",
        probe_manifest_path=experiment / "fixed_graph_probe_samples.json",
        output_directory=run / "elastic_strength_component_sweep",
        strengths=(),
        strength_matrices=settings,
        device_name=args.device,
        flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
    )
    print(json.dumps(report, indent=2))


def normal_avo_sweep(args: argparse.Namespace) -> None:
    contract, _, dataset, experiment, _ = resolve()
    run = experiment / "runs" / "full"
    checkpoint = run / "last.pt"
    if _completed_epoch(run) != int(contract["fixed_patch_budget"]["epochs"]):
        raise RuntimeError(f"Complete {contract['revision']} before its read-only sweep")
    settings = [
        (
            f"normal_avo_{strength:g}",
            ((0.0, 0.0), (0.0, float(strength))),
        )
        for strength in args.strengths
    ]
    report = run_elastic_strength_sweep(
        checkpoint_path=checkpoint,
        dataset_directory=dataset,
        run_directory=run,
        fixed_selection_path=experiment / "fixed_patch_selection.json",
        probe_manifest_path=experiment / "fixed_graph_probe_samples.json",
        output_directory=run / "normal_avo_strength_sweep",
        strengths=(),
        strength_matrices=settings,
        device_name=args.device,
        flow_steps=int(contract["diagnostic_probes"]["flow_integration_steps"]),
    )
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare").set_defaults(function=prepare)
    commands.add_parser("status").set_defaults(function=status)
    training = commands.add_parser("train")
    training.add_argument("--device", required=True, help="Explicit device such as cuda")
    training.add_argument("--until-epoch", type=int, help="Absolute epoch to stop after")
    training.set_defaults(function=train)
    strength_sweep = commands.add_parser(
        "sweep", help="Read-only counterfactual sweep of the frozen final checkpoint"
    )
    strength_sweep.add_argument("--device", required=True, help="Explicit device such as cuda")
    strength_sweep.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5],
    )
    strength_sweep.set_defaults(function=sweep)
    component = commands.add_parser(
        "component-sweep",
        help="Read-only tangent/normal and RGT/AVO elastic-prior ablation",
    )
    component.add_argument("--device", required=True, help="Explicit device such as cuda")
    component.add_argument("--strength", type=float, default=0.25)
    component.set_defaults(function=component_sweep)
    normal_avo = commands.add_parser(
        "normal-avo-sweep",
        help="Read-only magnitude sweep of the isolated normal AVO-contrast prior",
    )
    normal_avo.add_argument("--device", required=True, help="Explicit device such as cuda")
    normal_avo.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5],
    )
    normal_avo.set_defaults(function=normal_avo_sweep)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
