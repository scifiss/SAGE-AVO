#!/usr/bin/env python3
"""Run the approved Revision-3.3.1 support-aware Stage-02 production only."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from sage_avo.config import load_config
from sage_avo.experiments import generate_stage02_dataset
from sage_avo.experiments.manifest import file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
PRODUCTION_VERSION = "v00331_production100_support_aware"
CALIBRATION_ID = "v0033_58a5fe39a11c4fe66431"


def _contracts() -> tuple[dict[str, Any], dict[str, Any], Path]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    pointer_path = private / "source_freezes" / "revision331_approved_production.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            "Revision-3.3.1 production is not approved: the immutable pointer "
            f"is absent at {pointer_path}"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("status") != "GO_SCENARIO_CO2":
        raise RuntimeError("Revision-3.3.1 pointer is not GO_SCENARIO_CO2")
    if pointer.get("calibration_id") != CALIBRATION_ID:
        raise RuntimeError("Revision-3.3.1 pointer has the wrong calibration ID")
    snapshot = pointer["source_snapshot"]
    archive = Path(snapshot["archive_path"])
    if file_sha256(archive) != snapshot["archive_sha256"]:
        raise RuntimeError("Approved Revision-3.3.1 source archive hash mismatch")

    synthetic = deepcopy(
        load_config(REPOSITORY / "configs" / "synthetic_s01_v0032.yaml")
    )
    support_path = REPOSITORY / "configs" / "revision331_support_acceptance.yaml"
    support = load_config(support_path)
    if file_sha256(support_path) != pointer["support_contract_sha256"]:
        raise RuntimeError("Support acceptance contract changed after source freeze")
    synthetic["stage"].update(
        {
            "name": "field_conditioned_synthetic_avo_v00331_support_aware",
            "geology_realization_count": 100,
            "observation_variants_per_geology": 1,
            "realization_count": 100,
            "realization_id_offset": 3_400_000,
            "member_master_seeds": list(range(3_400_000, 3_400_100)),
        }
    )
    synthetic["fluid_substitution"].update(
        {
            "enabled": True,
            "mode": "calibrated_differential_gassmann",
            "calibration_id": CALIBRATION_ID,
            "calibration_artifact": (
                "derived/fluid_models_v0033/"
                "calibrated_dry_frame_scenario_ensemble.npz"
            ),
            "fluid_property_validation_artifact": (
                "derived/fluid_models_v0033/fluid_property_validation.json"
            ),
        }
    )
    synthetic["support_aware_acceptance"] = support
    synthetic["support_aware_acceptance_source"] = {
        "path": "configs/revision331_support_acceptance.yaml",
        "sha256": file_sha256(support_path),
    }
    synthetic["outputs"].update(
        {
            "version": PRODUCTION_VERSION,
            "directory": f"synthetic/{PRODUCTION_VERSION}/realizations",
        }
    )
    synthetic["source_snapshot"] = snapshot
    synthetic["approval_pointer_sha256"] = file_sha256(pointer_path)
    synthetic["revision331_route"] = "GO_SCENARIO_CO2"
    destination = (
        private
        / "stage_artifacts"
        / "stage02"
        / PRODUCTION_VERSION
        / "realizations"
    )
    return paths, synthetic, destination


def stage02(args: argparse.Namespace) -> None:
    paths, synthetic, destination = _contracts()
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=destination,
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "generated_realizations": manifest["generated_realizations"],
                "output_version": manifest["output_version"],
                "generation_config_sha256": manifest[
                    "generation_config_sha256"
                ],
                "support_aware_acceptance": manifest[
                    "support_aware_acceptance"
                ],
                "realization_timing_seconds": manifest[
                    "realization_timing_seconds"
                ],
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("stage02")
    generation.add_argument("--workers", type=int, default=1)
    generation.add_argument("--resume", action="store_true")
    generation.set_defaults(function=stage02)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
