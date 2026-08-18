#!/usr/bin/env python3
"""Write the local figure index for the bounded Revision-3 validation."""

from __future__ import annotations

import csv
from pathlib import Path

from sage_avo.config import load_config


REPOSITORY = Path(__file__).resolve().parents[1]
VALIDATION_ID = "v003_validation8_stage01v003"

FIGURES = {
    "figures/stage02/stage02_representative_realization.png": (
        "02_synthetic_avo_generation.ipynb",
        "representative-realization QC cell",
        "One deterministic field-conditioned geology, fluid state, and exact-PP AVO response.",
        True,
        True,
    ),
    "figures/stage02/v003_fluid_substitution_qc.png": (
        "scripts/run_revision3_validation.py",
        "_fluid_qc",
        "Local RF brine state, inverse-Gassmann CO2 state, saturation, and elastic deltas.",
        True,
        True,
    ),
    "figures/stage03/stage03_split_patch_contract.png": (
        "03_ml_dataset_construction.ipynb",
        "split-and-patch QC cell",
        "Leakage-safe group split and representative AVO/prior/target patch contract.",
        True,
        True,
    ),
    "figures/stage03/v003_diverse_patch_sampling_qc.png": (
        "scripts/run_revision3_validation.py",
        "_patch_qc",
        "Diverse candidate categories, depth coverage, and deterministic patch coordinates.",
        True,
        True,
    ),
    "figures/stage04/stage04_avo_feature_contract.png": (
        "04_sage_avo_training.ipynb",
        "AVO feature-contract cell",
        "Near/mid/far inputs and compact P/G features used by the graph branch.",
        True,
        True,
    ),
    "figures/stage05/stage05_field_input_contract.png": (
        "05_evaluation_and_field_application.ipynb",
        "field-input contract cell",
        "Configured field AVO, RGT, and common 2-Hz prior before field-domain calibration.",
        True,
        True,
    ),
}


def main() -> None:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    root = Path(paths["private_artifact_root"]) / "revision3" / VALIDATION_ID
    destination = root / "reports" / "revision3_figure_index.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative, metadata in FIGURES.items():
        path = root / relative
        if not path.exists():
            continue
        notebook, source, message, private_data, verify = metadata
        rows.append(
            {
                "filename": relative,
                "source_notebook": notebook,
                "source_cell_or_function": source,
                "scientific_message": message,
                "uses_field_or_private_data": private_data,
                "public_redistribution_needs_verification": verify,
                "bytes": path.stat().st_size,
            }
        )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(rows)} figures: {destination}")


if __name__ == "__main__":
    main()
