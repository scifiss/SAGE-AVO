#!/usr/bin/env python3
"""Run guarded Revision-3.3 production after an approved source freeze.

The driver refuses to expand the bounded workflow unless the immutable private
pointer contains exactly ``GO_SCENARIO_CO2`` or ``GO_CORE_ELASTIC``.  The
current approved route determines whether the CO2 module is enabled.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from sage_avo.config import load_config
from sage_avo.experiments import build_stage03_dataset, generate_stage02_dataset
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.experiments.training import train_controlled_variant


REPOSITORY = Path(__file__).resolve().parents[1]
PRODUCTION_VERSION = "v0033_production100_dry_frame_supported"
DATASET_VERSION = "ds_v0033_production100_dry_frame_supported"
EXPERIMENT_NAME = "sage_avo_s01_v0033_dry_frame_supported_production"
CALIBRATION_ID = "v0033_58a5fe39a11c4fe66431"


def _contracts() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]
]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    pointer_path = private / "source_freezes" / "revision33_approved_production.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            "Revision-3.3 production is not approved: the immutable "
            f"GO_SCENARIO_CO2/GO_CORE_ELASTIC pointer is absent at {pointer_path}"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    status = pointer.get("status")
    if status not in {"GO_SCENARIO_CO2", "GO_CORE_ELASTIC"}:
        raise RuntimeError("Revision-3.3 pointer does not contain exactly one approved route")
    source_snapshot = pointer["source_snapshot"]
    archive = Path(source_snapshot["archive_path"])
    if file_sha256(archive) != source_snapshot["archive_sha256"]:
        raise RuntimeError("Approved Revision-3.3 source archive hash mismatch")

    synthetic = deepcopy(load_config(REPOSITORY / "configs" / "synthetic_s01_v0032.yaml"))
    synthetic["stage"].update(
        {
            "name": "field_conditioned_synthetic_avo_v0033_dry_frame_supported",
            "geology_realization_count": 100,
            "observation_variants_per_geology": 1,
            "realization_count": 100,
            "realization_id_offset": 3_400_000,
        }
    )
    synthetic["outputs"].update(
        {
            "version": PRODUCTION_VERSION,
            "directory": f"synthetic/{PRODUCTION_VERSION}/realizations",
        }
    )
    if status == "GO_SCENARIO_CO2":
        synthetic["fluid_substitution"].update(
            {
                "enabled": True,
                "mode": "calibrated_differential_gassmann",
                "calibration_id": CALIBRATION_ID,
                "calibration_artifact": "derived/fluid_models_v0033/calibrated_dry_frame_scenario_ensemble.npz",
                "fluid_property_validation_artifact": "derived/fluid_models_v0033/fluid_property_validation.json",
            }
        )
    else:
        synthetic["fluid_substitution"].update(
            {"enabled": False, "mode": "disabled_core_elastic"}
        )

    dataset = deepcopy(load_config(REPOSITORY / "configs" / "ml_dataset_s01_v0032.yaml"))
    dataset["inputs"].update(
        {
            "synthetic_version": PRODUCTION_VERSION,
            "realization_directory": f"synthetic/{PRODUCTION_VERSION}/realizations",
            "expected_realization_count": 100,
        }
    )
    dataset["split"]["fractions"] = [0.70, 0.20, 0.10]
    dataset["outputs"].update(
        {"version": DATASET_VERSION, "directory": f"datasets/{DATASET_VERSION}"}
    )
    training = deepcopy(load_config(REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml"))
    training["experiment"].update(
        {"name": EXPERIMENT_NAME, "output_root": f"results/experiments/{EXPERIMENT_NAME}"}
    )
    training["dataset"]["directory"] = dataset["outputs"]["directory"]
    for config in (synthetic, dataset, training):
        config["source_snapshot"] = source_snapshot
        config["revision33_route"] = status
        config["approval_pointer_sha256"] = file_sha256(pointer_path)
    locations = {
        "stage02": private / "stage_artifacts" / "stage02" / PRODUCTION_VERSION / "realizations",
        "stage03": private / "stage_artifacts" / "stage03" / DATASET_VERSION / "dataset",
        "stage04": private / "stage_artifacts" / "stage04" / EXPERIMENT_NAME,
    }
    return paths, synthetic, dataset, training, locations


def stage02(args: argparse.Namespace) -> None:
    paths, synthetic, _, _, locations = _contracts()
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=locations["stage02"],
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(json.dumps(manifest, indent=2))


def stage03(_: argparse.Namespace) -> None:
    paths, _, dataset, _, locations = _contracts()
    manifest = build_stage03_dataset(
        config=dataset,
        paths=paths,
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    print(json.dumps(manifest["integrity"], indent=2))


def train(args: argparse.Namespace) -> None:
    _, _, _, training, locations = _contracts()
    config_path = locations["stage04"] / "configs" / "training_resolved.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, training)
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=config_path,
        config=training,
        dataset_directory=locations["stage03"],
        experiment_directory=locations["stage04"],
        variant=args.variant,
        device_name=args.device,
    )
    print(output)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("stage02")
    generation.add_argument("--workers", type=int, default=1)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(function=stage02)
    commands.add_parser("stage03").set_defaults(function=stage03)
    training = commands.add_parser("train")
    training.add_argument(
        "--variant", choices=("full", "no_gnn", "no_rgt", "no_physics"), required=True
    )
    training.add_argument("--device", default="cuda")
    training.set_defaults(function=train)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
