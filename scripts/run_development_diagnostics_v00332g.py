#!/usr/bin/env python3
"""Run the corrected stratified validation diagnostic without touching v00332e/f."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from sage_avo.config import load_config
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.diagnostics.checkpoint_analysis import analyze_checkpoint
from sage_avo.diagnostics.contracts import build_diagnostic_sample_manifest
from sage_avo.experiments.training import train_controlled_variant


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "development_diagnostics_v00332g.yaml"


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


def resolve() -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
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
    source_selection = (
        private
        / "stage_artifacts"
        / "stage04"
        / contract["training_selection_source"]["experiment_name"]
        / "fixed_patch_selection.json"
    )
    return contract, config, dataset, experiment, source_selection


def _class1_valid_pixels(dataset: IndexedRealizationPatches, index: int) -> int:
    fields = dataset.sampling_fields(index)
    mask = fields["mask"][0] > 0.5
    return int(((fields["segmentation"] == 1) & mask).sum())


def _stratified_validation_selection(
    contract: dict[str, Any], dataset: IndexedRealizationPatches
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select validation-only native physics and multiscale class-1 patches."""
    strata = contract["validation_strata"]
    frame = dataset.index
    categories = tuple(str(value) for value in strata["candidate_categories"])
    native_shape = tuple(int(value) for value in strata["native_scale"])
    scales = tuple(tuple(int(value) for value in value) for value in strata["non_native_scales"])
    minimum_class1 = int(strata["minimum_class1_valid_pixels_per_patch"])
    class1_cache: dict[int, int] = {}
    selected: list[int] = []
    rows: list[dict[str, Any]] = []

    def class1_count(index: int) -> int:
        if index not in class1_cache:
            class1_cache[index] = _class1_valid_pixels(dataset, index)
        return class1_cache[index]

    def choose(
        *,
        role: str,
        count: int,
        category: str | None = None,
        shape: tuple[int, int] | None = None,
        physics_eligible: bool | None = None,
        require_class1: bool = False,
    ) -> None:
        candidates = frame
        if category is not None:
            candidates = candidates[candidates["candidate_category"] == category]
        if shape is not None:
            candidates = candidates[
                (candidates["raw_height"] == shape[0]) & (candidates["raw_width"] == shape[1])
            ]
        if physics_eligible is not None:
            candidates = candidates[candidates["physics_eligible"] == int(physics_eligible)]
        candidates = candidates.sort_values(["realization_id", "top", "left"])
        chosen = 0
        for index, record in candidates.iterrows():
            patch_index = int(index)
            if patch_index in selected:
                continue
            pixels = class1_count(patch_index) if require_class1 else None
            if require_class1 and pixels < minimum_class1:
                continue
            selected.append(patch_index)
            rows.append(
                {
                    "index": patch_index,
                    "role": role,
                    "realization_id": int(record["realization_id"]),
                    "top": int(record["top"]),
                    "left": int(record["left"]),
                    "raw_shape": [int(record["raw_height"]), int(record["raw_width"])],
                    "physics_eligible": bool(int(record["physics_eligible"])),
                    "candidate_category": str(record["candidate_category"]),
                    "class1_valid_pixels": class1_count(patch_index),
                }
            )
            chosen += 1
            if chosen == count:
                return
        raise RuntimeError(
            f"Unable to select {count} distinct validation patches for {role}; selected {chosen}"
        )

    for category in categories:
        choose(
            role=f"native_physics_{category}",
            count=int(strata["native_physics_per_category"]),
            category=category,
            shape=native_shape,
            physics_eligible=True,
        )
    for shape in scales:
        for category in categories:
            choose(
                role=f"multiscale_{shape[0]}x{shape[1]}_{category}",
                count=int(strata["non_native_per_category_per_scale"]),
                category=category,
                shape=shape,
                physics_eligible=False,
            )
    choose(
        role="extra_native_class1",
        count=int(strata["extra_native_class1_patches"]),
        shape=native_shape,
        physics_eligible=True,
        require_class1=True,
    )
    for shape in scales:
        choose(
            role=f"extra_multiscale_{shape[0]}x{shape[1]}_class1",
            count=int(strata["extra_non_native_class1_patches_per_scale"]),
            shape=shape,
            physics_eligible=False,
            require_class1=True,
        )
    expected = int(contract["fixed_patch_budget"]["validation_batches_per_epoch"]) * int(
        contract["fixed_patch_budget"]["batch_size"]
    )
    if len(selected) != expected:
        raise RuntimeError(f"Validation selection count {len(selected)} does not match budget {expected}")
    return selected, rows


def _selection(
    contract: dict[str, Any], dataset: Path, source_selection_path: Path
) -> dict[str, Any]:
    if not source_selection_path.exists():
        raise FileNotFoundError(f"Missing frozen v00332f training selection: {source_selection_path}")
    source = json.loads(source_selection_path.read_text(encoding="utf-8"))
    expected = contract["training_selection_source"]
    train_indices = [int(value) for value in source["train_indices"]]
    if len(train_indices) != int(expected["expected_train_patch_count"]):
        raise RuntimeError("v00332f training patch count does not match the declared v00332g contract")
    if canonical_hash(train_indices) != str(expected["expected_train_indices_sha256"]):
        raise RuntimeError("v00332f training patch selection hash does not match the declared v00332g contract")
    validation = IndexedRealizationPatches(dataset, "validation")
    validation_indices, validation_records = _stratified_validation_selection(contract, validation)
    return {
        "train_selection_source": str(source_selection_path),
        "train_selection_source_sha256": file_sha256(source_selection_path),
        "train_patch_count": len(train_indices),
        "validation_patch_count": len(validation_indices),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "train_indices_sha256": canonical_hash(train_indices),
        "validation_indices_sha256": canonical_hash(validation_indices),
        "validation_records": validation_records,
        "validation_physics_eligible_count": sum(
            int(record["physics_eligible"]) for record in validation_records
        ),
        "validation_class1_valid_pixels": sum(
            int(record["class1_valid_pixels"]) for record in validation_records
        ),
        "repeats_same_train_patches_each_epoch": True,
    }


def _paths(experiment: Path) -> tuple[Path, Path, Path]:
    return (
        experiment / "development_diagnostic_contract.json",
        experiment / "fixed_patch_selection.json",
        experiment / "fixed_graph_probe_samples.json",
    )


def prepare(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment, source_selection = resolve()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    experiment.mkdir(parents=True, exist_ok=True)
    contract_path, selection_path, samples_path = _paths(experiment)
    if any(path.exists() for path in (contract_path, selection_path, samples_path)):
        raise FileExistsError("Diagnostic contract already exists; refuse to replace its frozen selection")
    selection = _selection(contract, dataset, source_selection)
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
        "validation_physics_eligible_count": selection["validation_physics_eligible_count"],
        "validation_class1_valid_pixels": selection["validation_class1_valid_pixels"],
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
    contract, config, dataset, experiment, _ = resolve()
    _, selection_path, samples_path = _paths(experiment)
    run = experiment / "runs" / "full"
    print(
        json.dumps(
            {
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
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment, _ = resolve()
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
