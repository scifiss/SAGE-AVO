#!/usr/bin/env python3
"""Prepare and run the v00332h dual-relation RGT diagnostic experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from sage_avo.config import load_config, seed_everything
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332h.yaml"


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


def resolve() -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, Path]:
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
    }
    config["capabilities"]["dual_relation_rgt_graph"] = {
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
    source = private / "stage_artifacts" / "stage04" / contract["selection_source"]["experiment_name"]
    return (
        contract,
        config,
        dataset,
        experiment,
        source / "fixed_patch_selection.json",
        source / "fixed_graph_probe_samples.json",
    )


def _validate_sources(
    contract: dict[str, Any], selection_path: Path, probes_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = contract["selection_source"]
    if file_sha256(selection_path) != expected["expected_selection_file_sha256"]:
        raise RuntimeError("Frozen v00332g selection file SHA-256 mismatch")
    if file_sha256(probes_path) != expected["expected_probe_file_sha256"]:
        raise RuntimeError("Frozen v00332g graph-probe file SHA-256 mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    probes = json.loads(probes_path.read_text(encoding="utf-8"))
    checks = (
        ("train_indices", "expected_train_patch_count", "expected_train_indices_sha256"),
        (
            "validation_indices",
            "expected_validation_patch_count",
            "expected_validation_indices_sha256",
        ),
    )
    for field, count_field, hash_field in checks:
        values = [int(value) for value in selection[field]]
        if len(values) != int(expected[count_field]):
            raise RuntimeError(f"Frozen v00332g {field} count mismatch")
        if canonical_hash(values) != expected[hash_field]:
            raise RuntimeError(f"Frozen v00332g {field} SHA-256 mismatch")
    return selection, probes


def _prepared_initialization(
    config: dict[str, Any],
    contract: dict[str, Any],
    normalization: dict[str, list[float]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Copy the legacy graph initialization into both relation branches."""
    seed_everything(int(contract["seed"]), deterministic_torch=True)
    relational_kwargs = sage_avo_model_kwargs(config)
    legacy_kwargs = dict(relational_kwargs)
    legacy_kwargs["graph_mode_override"] = "rgt"
    legacy = build_sage_avo_variant("full", **legacy_kwargs)
    legacy.set_norm_stats(normalization)

    seed_everything(int(contract["seed"]), deterministic_torch=True)
    relational = build_sage_avo_variant("full", **relational_kwargs)
    relational.set_norm_stats(normalization)
    legacy_state = legacy.state_dict()
    state = relational.state_dict()
    transfers: list[dict[str, str]] = []

    for destination in list(state):
        source = destination if destination in legacy_state else None
        if destination.startswith("graph.tangential_layers."):
            source = destination.replace("graph.tangential_layers.", "graph.layers.")
        elif destination.startswith("graph.normal_layers."):
            source = destination.replace("graph.normal_layers.", "graph.layers.")
        elif destination.startswith("graph.tangential_normalizations."):
            source = destination.replace(
                "graph.tangential_normalizations.", "graph.normalizations."
            )
        elif destination.startswith("graph.normal_normalizations."):
            source = destination.replace("graph.normal_normalizations.", "graph.normalizations.")
        if source not in legacy_state:
            continue
        if state[destination].shape == legacy_state[source].shape:
            state[destination] = legacy_state[source].clone()
            transfers.append({"source": source, "destination": destination})
        elif destination.endswith("lin_edge.weight") and legacy_state[source].shape[1] == 1:
            expanded = torch.zeros_like(state[destination])
            expanded[:, :1] = legacy_state[source]
            state[destination] = expanded
            transfers.append(
                {"source": source, "destination": destination, "rule": "channel_0_copy_rest_zero"}
            )

    state["graph.relation_gate.weight"] = torch.zeros_like(
        state["graph.relation_gate.weight"]
    )
    state["graph.relation_gate.bias"] = torch.tensor(
        contract["experimental_graph"]["initial_gate_bias"],
        dtype=state["graph.relation_gate.bias"].dtype,
    )
    relational.load_state_dict(state, strict=True)
    metadata = {
        "seed": int(contract["seed"]),
        "legacy_state_sha256": state_sha256(legacy_state),
        "relational_state_sha256": state_sha256(relational.state_dict()),
        "transferred_tensor_count": len(transfers),
        "transfers": transfers,
        "edge_attribute_initialization": "legacy edge weight copied to AVO-affinity channel; new channels zero",
        "gate_initialization": {
            "weight": "zeros",
            "bias": contract["experimental_graph"]["initial_gate_bias"],
        },
    }
    return dict(relational.state_dict()), metadata


def _paths(experiment: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        experiment / "development_diagnostic_contract.json",
        experiment / "fixed_patch_selection.json",
        experiment / "fixed_graph_probe_samples.json",
        experiment / "initial_model_state.pt",
        experiment / "initialization_manifest.json",
    )


def prepare(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment, source_selection, source_probes = resolve()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    selection, probes = _validate_sources(contract, source_selection, source_probes)
    experiment.mkdir(parents=True, exist_ok=True)
    contract_path, selection_path, probes_path, state_path, initialization_path = _paths(experiment)
    outputs = (contract_path, selection_path, probes_path, state_path, initialization_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("v00332h contract already exists; refusing to replace frozen inputs")

    normalization = json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
    state, initialization = _prepared_initialization(config, contract, normalization)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    probes_path.write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
    torch.save(state, state_path)
    initialization["state_file_sha256"] = file_sha256(state_path)
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
        "initial_model_state_sha256": initialization["relational_state_sha256"],
    }
    contract_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def _completed_epoch(run: Path) -> int:
    manifest = run / "manifest.json"
    if not manifest.exists():
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("last_completed_epoch", 0))


def status(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment, _, _ = resolve()
    paths = _paths(experiment)
    run = experiment / "runs" / "full"
    print(
        json.dumps(
            {
                "revision": contract["revision"],
                "diagnostic_only": True,
                "dataset_ready": (dataset / "dataset_manifest.json").exists(),
                "resolved_config_sha256": canonical_hash(config),
                "prepared_files_ready": all(path.exists() for path in paths),
                "run": str(run),
                "last_completed_epoch": _completed_epoch(run),
                "resumable": (run / "last.pt").exists(),
                "probe_output": str(run / "gradient_graph_probe"),
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment, _, _ = resolve()
    contract_path, selection_path, probes_path, state_path, initialization_path = _paths(experiment)
    if not all(path.exists() for path in _paths(experiment)):
        raise FileNotFoundError("Run the prepare command before v00332h training")
    prepared = json.loads(contract_path.read_text(encoding="utf-8"))
    if prepared["contract_sha256"] != canonical_hash(contract):
        raise RuntimeError("v00332h contract is stale; create a new revision")
    if prepared["resolved_config_sha256"] != canonical_hash(config):
        raise RuntimeError("v00332h resolved config is stale; create a new revision")
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if initialization["state_file_sha256"] != file_sha256(state_path):
        raise RuntimeError("v00332h initial model-state file SHA-256 mismatch")
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
        print(f"Completed v00332h epoch {epoch} and its checkpoint-only probes")


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
