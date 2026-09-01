#!/usr/bin/env python3
"""Prepare and launch predeclared matched v00332e architecture controls."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from sage_avo.config import load_config
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.models.variants import LEARNED_VARIANTS


REPOSITORY = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY / "configs" / "matched_training_v00332e.yaml"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve() -> tuple[dict, dict, Path, Path]:
    contract = load_config(CONTRACT_PATH)
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    config = deepcopy(load_config(REPOSITORY / "configs" / contract["base_training_config"]))
    config["dataset"]["directory"] = f"datasets/{contract['immutable_dataset']}"
    config["experiment"]["name"] = "sage_avo_s01_v00332e_matched_controls"
    config["training"]["epochs"] = int(contract["shared_contract"]["epochs"])
    config["training"]["batch_size"] = int(contract["shared_contract"]["batch_size"])
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"]["enabled"] = False
    config["training"]["contrastive_loss"]["weight"] = 0.0
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["enabled"] = False
    config["training"]["physics_guided_sampling"]["guidance_scale"] = 0.0
    dataset = private / "stage_artifacts" / "stage03" / contract["immutable_dataset"] / "dataset"
    experiment = private / "stage_artifacts" / "stage04" / config["experiment"]["name"]
    return contract, config, dataset, experiment


def status(_: argparse.Namespace) -> None:
    contract, config, dataset, experiment = resolve()
    result = {
        "contract": str(CONTRACT_PATH),
        "contract_status": contract["status"],
        "resolved_config_sha256": canonical_hash(config),
        "dataset_ready": (dataset / "dataset_manifest.json").exists(),
        "experiment": str(experiment),
        "runs": {
            variant: {
                "directory_exists": (experiment / "runs" / variant).exists(),
                "last_completed_epoch": _last_completed_epoch(
                    experiment / "runs" / variant
                ),
                "resumable": (experiment / "runs" / variant / "last.pt").exists(),
                "checkpoint_exists": (
                    experiment / "runs" / variant / "best_whole_realization.pt"
                ).exists(),
            }
            for variant in LEARNED_VARIANTS
        },
    }
    print(json.dumps(result, indent=2))


def _last_completed_epoch(run: Path) -> int:
    manifest = run / "manifest.json"
    if not manifest.exists():
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("last_completed_epoch", 0))


def _archive_incomplete_run(run: Path) -> Path:
    """Preserve a run that crashed before its first restart checkpoint."""
    if (run / "last.pt").exists() or _last_completed_epoch(run) > 0:
        raise RuntimeError("Only a run without a completed epoch may be archived as incomplete")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run.with_name(f"{run.name}_interrupted_before_epoch1_{timestamp}")
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    run.rename(destination)
    return destination


def prepare(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment = resolve()
    if not (dataset / "dataset_manifest.json").exists():
        raise FileNotFoundError(f"Immutable dataset is unavailable: {dataset}")
    experiment.mkdir(parents=True, exist_ok=True)
    destination = experiment / "matched_training_contract.json"
    if destination.exists() and not args.refresh:
        raise FileExistsError(f"Refusing to overwrite existing contract: {destination}")
    if args.refresh and any((experiment / "runs" / variant).exists() for variant in LEARNED_VARIANTS):
        raise RuntimeError("Refusing to refresh a contract after any matched run has started")
    payload = {
        "status": "PREPARED_NOT_TRAINED",
        "contract": contract,
        "contract_sha256": canonical_hash(contract),
        "resolved_config_sha256": canonical_hash(config),
        "dataset": str(dataset),
        "dataset_manifest": str(dataset / "dataset_manifest.json"),
        "experiment": str(experiment),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def train(args: argparse.Namespace) -> None:
    contract, config, dataset, experiment = resolve()
    prepared = experiment / "matched_training_contract.json"
    if not prepared.exists():
        raise FileNotFoundError("Run the prepare command before training")
    prepared_payload = json.loads(prepared.read_text(encoding="utf-8"))
    if prepared_payload.get("contract_sha256") != canonical_hash(contract):
        raise RuntimeError("Prepared contract hash is stale; review and run prepare --refresh")
    if prepared_payload.get("resolved_config_sha256") != canonical_hash(config):
        raise RuntimeError("Prepared resolved-config hash is stale; review and run prepare --refresh")
    run = experiment / "runs" / args.variant
    resume_from = run / "last.pt"
    if run.exists() and not resume_from.exists():
        if not args.archive_incomplete:
            raise RuntimeError(
                f"Run exists without a restart checkpoint: {run}. "
                "Inspect it, then pass --archive-incomplete to preserve and restart it."
            )
        archived = _archive_incomplete_run(run)
        print(f"Archived incomplete run: {archived}")
    resume_from = run / "last.pt"
    completed = _last_completed_epoch(run)
    target_epoch = int(args.until_epoch or contract["shared_contract"]["epochs"])
    maximum_epoch = int(contract["shared_contract"]["epochs"])
    if not 1 <= target_epoch <= maximum_epoch:
        raise ValueError(f"--until-epoch must lie in [1, {maximum_epoch}]")
    if target_epoch <= completed:
        raise ValueError(
            f"Target epoch {target_epoch} is not after completed epoch {completed}"
        )
    expected = contract["allowed_variant_differences"][args.variant]
    configured_physics = float(config["model"]["variants"][args.variant]["physics_weight"])
    if configured_physics != float(expected["physics_weight"]):
        raise RuntimeError("Variant physics weight violates the predeclared contract")
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=CONTRACT_PATH,
        config=config,
        dataset_directory=dataset,
        experiment_directory=experiment,
        variant=args.variant,
        device_name=args.device,
        resume_from=resume_from if resume_from.exists() else None,
        stop_after_epoch=target_epoch,
    )
    print(f"Completed matched {args.variant}: {output}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("status").set_defaults(function=status)
    preparation = commands.add_parser("prepare")
    preparation.add_argument(
        "--refresh",
        action="store_true",
        help="Replace a stale prepared contract only when no matched run has started",
    )
    preparation.set_defaults(function=prepare)
    training = commands.add_parser("train")
    training.add_argument("--variant", choices=LEARNED_VARIANTS, required=True)
    training.add_argument("--device", required=True, help="Explicit device such as cuda:0")
    training.add_argument(
        "--until-epoch",
        type=int,
        help="Stop cleanly after this absolute epoch; rerun with a larger value to resume",
    )
    training.add_argument(
        "--archive-incomplete",
        action="store_true",
        help="Preserve an existing zero-checkpoint crash directory before restarting",
    )
    training.set_defaults(function=train)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
