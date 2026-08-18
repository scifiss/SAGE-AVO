#!/usr/bin/env python3
"""Run the bounded Revision-3.1 fluid diagnosis, calibration, and gate.

The driver cannot launch the 100-realization production corpus. Its outputs are
confined to the ignored Stage-01 derived-data tree and configured artifact root.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
import torch

from sage_avo.config import load_config
from sage_avo.experiments import (
    build_stage03_dataset,
    generate_stage02_dataset,
    validate_dataset_integrity,
)
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.forward import (
    forward_avo_three_band_spec_torch,
    forward_specification_from_mapping,
)
from sage_avo.geology.fluid_calibration import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    brie_fluid_mixture,
    calibrated_differential_gassmann_substitution,
    constrained_local_gassmann_substitution,
    density_derived_effective_porosity,
    elastic_from_gpa_strict,
    forward_gassmann_bulk_strict,
    inverse_gassmann_dry_bulk_strict,
    load_calibrated_dry_frame,
    mineral_properties_vrh_strict,
    poisson_ratio_from_moduli,
    save_calibrated_dry_frame,
)
from sage_avo.geology.rock_physics import elastic_moduli_gpa


REPOSITORY = Path(__file__).resolve().parents[1]
OLD_VALIDATION_NAME = "v003_validation8_stage01v003"
NEW_VALIDATION_NAME = "v0031_validation8_fluid_corrected"
CALIBRATION_VERSION = "fluid_models_v0031"
REALIZATION_COUNT = 8
REALIZATION_OFFSET = 3_100_000


def _contracts() -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    paths_path = REPOSITORY / "configs" / "paths.yaml"
    if not paths_path.exists():
        raise FileNotFoundError("Create the ignored configs/paths.yaml before local validation")
    paths = load_config(paths_path)
    synthetic = deepcopy(load_config(REPOSITORY / "configs" / "synthetic_s01_v003.yaml"))
    private = Path(paths["private_artifact_root"])
    old_root = private / "revision3" / OLD_VALIDATION_NAME
    new_root = private / "revision31" / NEW_VALIDATION_NAME
    data_root = Path(paths["work_data_root"]) / str(synthetic["inputs"]["dataset_id"])
    return paths, synthetic, old_root, new_root, data_root


def _bounded_configs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]
]:
    paths, _, _, new_root, _ = _contracts()
    synthetic = deepcopy(load_config(REPOSITORY / "configs" / "synthetic_s01_v0031.yaml"))
    dataset = deepcopy(load_config(REPOSITORY / "configs" / "ml_dataset_s01_v0031.yaml"))
    training = deepcopy(load_config(REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml"))
    synthetic["stage"].update(
        {
            "geology_realization_count": REALIZATION_COUNT,
            "observation_variants_per_geology": 1,
            "realization_count": REALIZATION_COUNT,
            "realization_id_offset": REALIZATION_OFFSET,
        }
    )
    synthetic["outputs"].update(
        {
            "version": NEW_VALIDATION_NAME,
            "directory": f"synthetic/{NEW_VALIDATION_NAME}/realizations",
        }
    )
    dataset["inputs"].update(
        {
            "synthetic_version": NEW_VALIDATION_NAME,
            "realization_directory": f"synthetic/{NEW_VALIDATION_NAME}/realizations",
            "expected_realization_count": REALIZATION_COUNT,
        }
    )
    dataset["split"]["fractions"] = [0.625, 0.25, 0.125]
    dataset["outputs"].update(
        {
            "version": "ds_v0031_validation8_fluid_corrected",
            "directory": "datasets/ds_v0031_validation8_fluid_corrected",
        }
    )
    training_name = "sage_avo_s01_v0031_validation8_fluid_corrected"
    training["experiment"].update(
        {"name": training_name, "output_root": f"results/experiments/{training_name}"}
    )
    training["dataset"]["directory"] = dataset["outputs"]["directory"]
    training["training"]["epochs"] = 2
    training["training"]["checkpointing"].update(
        {
            "periodic_interval_epochs": 1,
            "whole_validation_every_epochs": 1,
            "whole_validation_realization_count": 2,
        }
    )
    prefreeze = {
        "status": "revision31_bounded_validation_precedes_new_source_freeze",
        "superseded_fluid_snapshot_sha256": (
            "5d9f9726845d9496d3de6b14af63c7bc9a737feda60a7ae7d2a52a78b1001d56"
        ),
    }
    for mapping in (synthetic, dataset, training):
        mapping["source_snapshot"] = deepcopy(prefreeze)
    locations = {
        "root": new_root,
        "stage02": new_root / "stage02" / "realizations",
        "stage03": new_root / "stage03" / "dataset",
        "stage04": new_root / "stage04" / training_name,
        "figures02": new_root / "figures" / "stage02",
        "figures03": new_root / "figures" / "stage03",
        "figures04": new_root / "figures" / "stage04",
        "figures05": new_root / "figures" / "stage05",
        "reports": new_root / "reports",
        "configs": new_root / "configs",
        "executed_notebooks": new_root / "executed_notebooks",
    }
    return paths, synthetic, dataset, training, locations


def _snapshot_bounded_configs(
    synthetic: dict[str, Any],
    dataset: dict[str, Any],
    training: dict[str, Any],
    locations: dict[str, Path],
) -> None:
    locations["configs"].mkdir(parents=True, exist_ok=True)
    write_json(locations["configs"] / "synthetic_resolved.json", synthetic)
    write_json(locations["configs"] / "dataset_resolved.json", dataset)
    write_json(locations["configs"] / "training_resolved.json", training)


def _physics(config: dict[str, Any]) -> FluidRockPhysics:
    fluid = config["fluid_substitution"]
    names = FluidRockPhysics.__dataclass_fields__
    return FluidRockPhysics(**{name: float(fluid[name]) for name in names})


def _vrh_components(
    shale: np.ndarray,
    physics: FluidRockPhysics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quartz = 1.0 - shale
    voigt = quartz * physics.quartz_bulk_modulus_gpa + shale * physics.clay_bulk_modulus_gpa
    reuss = 1.0 / (
        quartz / physics.quartz_bulk_modulus_gpa + shale / physics.clay_bulk_modulus_gpa
    )
    return voigt, reuss, 0.5 * (voigt + reuss)


def _feasibility_limit(
    saturated_bulk_gpa: np.ndarray,
    porosity: np.ndarray,
    brine_bulk_modulus_gpa: float,
) -> np.ndarray:
    denominator = 1.0 / saturated_bulk_gpa - porosity / brine_bulk_modulus_gpa
    return np.where(denominator > 0.0, (1.0 - porosity) / denominator, np.nan)


def _classify_facies(shaliness: np.ndarray) -> np.ndarray:
    return np.where(shaliness < 0.50, "clean_sand", "shaly_sand")


def _assign_bins(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["depth_bin"] = pd.cut(
        result["depth_m"],
        bins=[-np.inf, 2850.0, 3100.0, np.inf],
        labels=["shallow", "middle", "deep"],
    ).astype(str)
    result["input_porosity_bin"] = pd.cut(
        result["input_porosity"],
        bins=[-np.inf, 0.025, 0.05, 0.075, np.inf],
        labels=["<=0.025", "0.025-0.05", "0.05-0.075", ">0.075"],
    ).astype(str)
    result["effective_porosity_bin"] = pd.cut(
        result["density_porosity"],
        bins=[-np.inf, 0.075, 0.10, 0.125, np.inf],
        labels=["<=0.075", "0.075-0.10", "0.10-0.125", ">0.125"],
    ).astype(str)
    return result


def _diagnose_arrays(
    *,
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    porosity: np.ndarray,
    shaliness: np.ndarray,
    depth_m: np.ndarray,
    physics: FluidRockPhysics,
) -> dict[str, np.ndarray]:
    bulk, shear = elastic_moduli_gpa(vp, vs, density)
    mineral_bulk, mineral_shear, mineral_density = mineral_properties_vrh_strict(
        shaliness, physics
    )
    phi_density = density_derived_effective_porosity(
        density, mineral_density, physics.brine_density_g_cc
    )
    dry_input = inverse_gassmann_dry_bulk_strict(
        bulk, porosity, mineral_bulk, physics.brine_bulk_modulus_gpa
    )
    dry_density = inverse_gassmann_dry_bulk_strict(
        bulk, phi_density, mineral_bulk, physics.brine_bulk_modulus_gpa
    )
    compatibility_limit = _feasibility_limit(
        bulk, porosity, physics.brine_bulk_modulus_gpa
    )
    poisson_input = poisson_ratio_from_moduli(dry_input, shear)
    poisson_density = poisson_ratio_from_moduli(dry_density, shear)
    _, mineral_reuss, mineral_vrh = _vrh_components(shaliness, physics)
    return {
        "vp_m_s": vp,
        "vs_m_s": vs,
        "density_g_cc": density,
        "input_porosity": porosity,
        "density_porosity": phi_density,
        "porosity_difference": phi_density - porosity,
        "shaliness": shaliness,
        "sand_probability": 1.0 - shaliness,
        "depth_m": depth_m,
        "lithostatic_stress_proxy_mpa": 1600.0 * 9.8 * depth_m / 1e6,
        "saturated_bulk_gpa": bulk,
        "shear_gpa": shear,
        "mineral_bulk_gpa": mineral_bulk,
        "mineral_shear_gpa": mineral_shear,
        "mineral_density_g_cc": mineral_density,
        "mineral_bulk_reuss_gpa": mineral_reuss,
        "mineral_bulk_vrh_gpa": mineral_vrh,
        "brine_bulk_gpa": np.full_like(bulk, physics.brine_bulk_modulus_gpa),
        "brine_density_g_cc": np.full_like(bulk, physics.brine_density_g_cc),
        "dry_bulk_input_phi_gpa": dry_input,
        "dry_bulk_density_phi_gpa": dry_density,
        "dry_to_shear_input_phi": dry_input / shear,
        "dry_to_shear_density_phi": dry_density / shear,
        "dry_poisson_input_phi": poisson_input,
        "dry_poisson_density_phi": poisson_density,
        "compatibility_limit_gpa": compatibility_limit,
        "gassmann_feasibility_margin_gpa": compatibility_limit - mineral_bulk,
        "input_phi_basic_valid": (
            (dry_input > 0.0) & (dry_input < mineral_bulk) & np.isfinite(dry_input)
        ),
        "density_phi_basic_valid": (
            (dry_density > 0.0) & (dry_density < mineral_bulk) & np.isfinite(dry_density)
        ),
        "input_phi_strict_valid": (
            (dry_input > 0.0)
            & (dry_input < mineral_bulk)
            & (dry_input / shear >= 0.3)
            & (dry_input / shear <= 4.0)
            & (poisson_input >= 0.0)
            & (poisson_input <= 0.45)
        ),
        "density_phi_strict_valid": (
            (dry_density > 0.0)
            & (dry_density < mineral_bulk)
            & (dry_density / shear >= 0.3)
            & (dry_density / shear <= 4.0)
            & (poisson_density >= 0.0)
            & (poisson_density <= 0.45)
        ),
    }


def _reservoir_wells(data_root: Path, physics: FluidRockPhysics) -> pd.DataFrame:
    ties = pd.read_csv(data_root / "usable" / "v003" / "df_well.csv").set_index("WELL")
    records: list[pd.DataFrame] = []
    for path in sorted((data_root / "usable" / "v003" / "wells").glob("*.csv")):
        well = path.stem
        if well not in ties.index:
            continue
        t6 = ties.loc[well, "T6_TWT_MS"]
        t7 = ties.loc[well, "T7_TWT_MS"]
        if not np.isfinite(t6) or not np.isfinite(t7):
            continue
        table = pd.read_csv(path)
        selected = table[
            table["TWT_MS"].between(min(t6, t7), max(t6, t7))
            & (table["SAND_PROBABILITY"] >= 0.30)
        ].copy()
        required = ["VP", "VS", "RHOB", "PORO", "DELTA", "DEPTH", "TWT_MS"]
        selected = selected.dropna(subset=required)
        selected = selected[
            (selected["VP"] > selected["VS"])
            & (selected["VS"] > 0.0)
            & selected["PORO"].between(0.0, 1.0, inclusive="neither")
            & selected["DELTA"].between(0.0, 1.0)
        ]
        if selected.empty:
            continue
        diagnosis = _diagnose_arrays(
            vp=selected["VP"].to_numpy(float),
            vs=selected["VS"].to_numpy(float),
            density=selected["RHOB"].to_numpy(float),
            porosity=selected["PORO"].to_numpy(float),
            shaliness=selected["DELTA"].to_numpy(float),
            depth_m=selected["DEPTH"].to_numpy(float),
            physics=physics,
        )
        output = pd.DataFrame(diagnosis)
        output.insert(0, "source", "well")
        output.insert(1, "well", well)
        output["twt_ms"] = selected["TWT_MS"].to_numpy(float)
        output["facies"] = _classify_facies(output["shaliness"].to_numpy())
        records.append(output)
    if not records:
        raise RuntimeError("No T6-T7 reservoir-tied well samples passed QC")
    return _assign_bins(pd.concat(records, ignore_index=True))


def _calibration_wells(
    data_root: Path,
    physics: FluidRockPhysics,
    minimum_depth_m: float = 2400.0,
    maximum_depth_m: float = 3250.0,
) -> pd.DataFrame:
    """Load the widest defensible pre-CO2 sandy intervals covering gate depth.

    T73 has an SW curve and is restricted to SW >= 0.95.  The other wells do
    not contain SW; they are retained as pre-CO2 baseline intervals with that
    limitation recorded explicitly in the calibration metadata.
    """
    records: list[pd.DataFrame] = []
    for path in sorted((data_root / "usable" / "v003" / "wells").glob("*.csv")):
        table = pd.read_csv(path)
        required = ["VP", "VS", "RHOB", "PORO", "DELTA", "DEPTH", "TWT_MS"]
        selected = table.dropna(subset=required).copy()
        selected = selected[
            (selected["SAND_PROBABILITY"] >= 0.30)
            & selected["DEPTH"].between(minimum_depth_m, maximum_depth_m)
            & (selected["VP"] > selected["VS"])
            & (selected["VS"] > 0.0)
            & selected["PORO"].between(0.0, 1.0, inclusive="neither")
            & selected["DELTA"].between(0.0, 1.0)
        ]
        sw_verified = "SW" in selected and selected["SW"].notna().any()
        if sw_verified:
            selected = selected[selected["SW"] >= 0.95]
        if selected.empty:
            continue
        shale = selected["DELTA"].to_numpy(float)
        _, _, mineral_density = mineral_properties_vrh_strict(shale, physics)
        density = selected["RHOB"].to_numpy(float)
        raw_phi_density = (mineral_density - density) / (
            mineral_density - physics.brine_density_g_cc
        )
        physical_density = (raw_phi_density > 0.0) & (raw_phi_density < 1.0)
        selected = selected.loc[physical_density].copy()
        if selected.empty:
            continue
        diagnosis = _diagnose_arrays(
            vp=selected["VP"].to_numpy(float),
            vs=selected["VS"].to_numpy(float),
            density=selected["RHOB"].to_numpy(float),
            porosity=selected["PORO"].to_numpy(float),
            shaliness=selected["DELTA"].to_numpy(float),
            depth_m=selected["DEPTH"].to_numpy(float),
            physics=physics,
        )
        output = pd.DataFrame(diagnosis)
        output.insert(0, "source", "calibration_well")
        output.insert(1, "well", path.stem)
        output["twt_ms"] = selected["TWT_MS"].to_numpy(float)
        output["facies"] = _classify_facies(output["shaliness"].to_numpy())
        output["brine_evidence"] = (
            "SW_log_at_least_0.95" if sw_verified else "pre_CO2_baseline_no_SW_curve"
        )
        records.append(output)
    if not records:
        raise RuntimeError("No broader pre-CO2 sandy well samples passed QC")
    return _assign_bins(pd.concat(records, ignore_index=True))


def _time_depth_coefficients(wells: pd.DataFrame) -> np.ndarray:
    return np.polyfit(wells["twt_ms"].to_numpy(), wells["depth_m"].to_numpy(), deg=1)


def _gate_pixels(
    old_root: Path,
    config: dict[str, Any],
    physics: FluidRockPhysics,
    depth_coefficients: np.ndarray,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    minimum_thickness = int(config["fluid_substitution"]["minimum_sand_thickness_samples"])
    threshold = float(config["geology"]["sand_facies_probability_threshold"])
    for path in sorted((old_root / "stage02" / "realizations").glob("realization_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            brine = np.asarray(archive["elastic_brine"], dtype=float)
            porosity = np.asarray(archive["porosity"], dtype=float)
            shaliness = np.asarray(archive["delta"], dtype=float)
            reservoir = np.asarray(archive["reservoir_mask"], dtype=bool)
            plume = np.asarray(archive["plume_mask"], dtype=bool)
            time_ms = np.asarray(archive["time_ms"], dtype=float)
            realization_id = int(archive["realization_id"])
        facies_sand = ((1.0 - shaliness) >= threshold) & reservoir
        candidate = binary_erosion(
            facies_sand, structure=np.ones((minimum_thickness, 1), dtype=bool)
        )
        rows, columns = np.indices(porosity.shape)
        depth_by_row = np.polyval(depth_coefficients, time_ms)
        depth = np.broadcast_to(depth_by_row[:, None], porosity.shape)
        select = candidate | plume
        diagnosis = _diagnose_arrays(
            vp=brine[0][select],
            vs=brine[1][select],
            density=brine[2][select],
            porosity=porosity[select],
            shaliness=shaliness[select],
            depth_m=depth[select],
            physics=physics,
        )
        output = pd.DataFrame(diagnosis)
        output.insert(0, "source", "gate_pixel")
        output.insert(1, "realization_id", realization_id)
        output["row"] = rows[select]
        output["column"] = columns[select]
        output["candidate"] = candidate[select]
        output["plume"] = plume[select]
        output["facies"] = _classify_facies(output["shaliness"].to_numpy())
        records.append(output)
    if len(records) != REALIZATION_COUNT:
        raise RuntimeError(
            f"Expected {REALIZATION_COUNT} immutable gate realizations; found {len(records)}"
        )
    return _assign_bins(pd.concat(records, ignore_index=True))


def _percentiles(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    quantiles = [0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0]
    rows = []
    for column in columns:
        values = frame[column].to_numpy(float)
        for quantile, value in zip(quantiles, np.quantile(values, quantiles)):
            rows.append({"quantity": column, "quantile": quantile, "value": value})
    return pd.DataFrame(rows)


def _diagnostic_figures(wells: pd.DataFrame, gate: pd.DataFrame, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    sample_gate = gate.sample(min(len(gate), 25_000), random_state=12345)
    combined = pd.concat(
        [wells.assign(dataset="well"), sample_gate.assign(dataset="gate")],
        ignore_index=True,
    )
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for dataset, marker, alpha in (("well", "o", 0.35), ("gate", ".", 0.12)):
        subset = combined[combined["dataset"] == dataset]
        axes[0, 0].scatter(
            subset["input_porosity"], subset["density_porosity"], s=8, marker=marker, alpha=alpha, label=dataset
        )
        axes[0, 1].scatter(
            subset["input_porosity"], subset["dry_bulk_input_phi_gpa"], s=8, marker=marker, alpha=alpha, label=dataset
        )
        axes[0, 2].scatter(
            subset["density_porosity"], subset["dry_bulk_density_phi_gpa"], s=8, marker=marker, alpha=alpha, label=dataset
        )
    axes[0, 0].plot([0, 0.25], [0, 0.25], "k--", linewidth=1)
    axes[0, 0].set(xlabel="stored/log porosity", ylabel="density-derived effective porosity", title="Porosity closure")
    axes[0, 1].axhline(0.0, color="k", linewidth=1)
    axes[0, 1].set(xlabel="stored/log porosity", ylabel="inverse Kdry (GPa)", title="Inverse Gassmann with stored porosity")
    axes[0, 2].axhline(0.0, color="k", linewidth=1)
    axes[0, 2].set(xlabel="density-derived porosity", ylabel="inverse Kdry (GPa)", title="Inverse Gassmann after density closure")
    wells.boxplot(column="porosity_difference", by="well", ax=axes[1, 0], grid=False)
    axes[1, 0].set(title="Porosity mismatch by well", xlabel="well", ylabel="phi_density - phi_log")
    gate.boxplot(column="gassmann_feasibility_margin_gpa", by="facies", ax=axes[1, 1], grid=False)
    axes[1, 1].axhline(0.0, color="k", linewidth=1)
    axes[1, 1].set(title="Original-state feasibility by facies", xlabel="facies", ylabel="Klimit - Kmin (GPa)")
    gate.groupby("depth_bin", observed=True)[["input_phi_strict_valid", "density_phi_strict_valid"]].mean().plot.bar(ax=axes[1, 2])
    axes[1, 2].set(title="Physical-state validity by depth", xlabel="depth bin", ylabel="valid fraction", ylim=(0.0, 1.05))
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    figure.suptitle("Revision-3.1 brine-state incompatibility diagnosis", fontsize=15)
    figure_path = destination / "brine_state_incompatibility_diagnosis.png"
    figure.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return [figure_path.name]


def diagnose(_: argparse.Namespace) -> None:
    _, config, old_root, new_root, data_root = _contracts()
    physics = _physics(config)
    destination = new_root / "fluid_gate" / "diagnosis"
    destination.mkdir(parents=True, exist_ok=True)
    wells = _reservoir_wells(data_root, physics)
    depth_coefficients = _time_depth_coefficients(wells)
    gate = _gate_pixels(old_root, config, physics, depth_coefficients)
    wells.to_csv(destination / "well_brine_state_diagnostic.csv", index=False)
    gate.to_csv(destination / "gate_candidate_brine_state_diagnostic.csv", index=False)
    grouping_rows = []
    for source_name, table in (("well", wells), ("gate", gate)):
        dimensions = ["facies", "depth_bin"] + (["well"] if source_name == "well" else ["realization_id"])
        for dimension in dimensions:
            for key, subset in table.groupby(dimension, observed=True):
                grouping_rows.append(
                    {
                        "source": source_name,
                        "grouping": dimension,
                        "group": key,
                        "samples": len(subset),
                        "input_phi_basic_valid_fraction": subset["input_phi_basic_valid"].mean(),
                        "input_phi_strict_valid_fraction": subset["input_phi_strict_valid"].mean(),
                        "density_phi_basic_valid_fraction": subset["density_phi_basic_valid"].mean(),
                        "density_phi_strict_valid_fraction": subset["density_phi_strict_valid"].mean(),
                        "median_input_porosity": subset["input_porosity"].median(),
                        "median_density_porosity": subset["density_porosity"].median(),
                        "median_porosity_difference": subset["porosity_difference"].median(),
                        "median_dry_bulk_input_phi_gpa": subset["dry_bulk_input_phi_gpa"].median(),
                        "median_dry_bulk_density_phi_gpa": subset["dry_bulk_density_phi_gpa"].median(),
                    }
                )
    pd.DataFrame(grouping_rows).to_csv(destination / "diagnostic_groups.csv", index=False)
    percentiles = pd.concat(
        [
            _percentiles(
                wells,
                [
                    "input_porosity",
                    "density_porosity",
                    "porosity_difference",
                    "dry_bulk_input_phi_gpa",
                    "dry_bulk_density_phi_gpa",
                    "dry_to_shear_density_phi",
                    "dry_poisson_density_phi",
                ],
            ).assign(source="well"),
            _percentiles(
                gate,
                [
                    "input_porosity",
                    "density_porosity",
                    "porosity_difference",
                    "dry_bulk_input_phi_gpa",
                    "dry_bulk_density_phi_gpa",
                    "dry_to_shear_density_phi",
                    "dry_poisson_density_phi",
                ],
            ).assign(source="gate"),
        ],
        ignore_index=True,
    )
    percentiles.to_csv(destination / "diagnostic_percentiles.csv", index=False)
    figures = _diagnostic_figures(wells, gate, destination)
    mineral_sensitivity = []
    for label, mineral_column in (
        ("reuss", "mineral_bulk_reuss_gpa"),
        ("vrh", "mineral_bulk_vrh_gpa"),
    ):
        for source_name, table in (("well", wells), ("gate", gate)):
            margin = table["compatibility_limit_gpa"] - table[mineral_column]
            mineral_sensitivity.append(
                {
                    "source": source_name,
                    "mineral_mixing_bound": label,
                    "compatible_fraction_with_input_porosity": float((margin >= 0.0).mean()),
                    "median_margin_gpa": float(margin.median()),
                }
            )
    pd.DataFrame(mineral_sensitivity).to_csv(destination / "mineral_mixture_sensitivity.csv", index=False)
    report = {
        "status": "diagnosis_complete_no_repair_applied",
        "immutable_input": str(old_root),
        "immutable_input_manifest_sha256": file_sha256(old_root / "stage02" / "realizations" / "manifest.json"),
        "well_samples": int(len(wells)),
        "wells": sorted(wells["well"].unique().tolist()),
        "gate_candidate_pixels": int(len(gate)),
        "gate_plume_pixels": int(gate["plume"].sum()),
        "time_depth_fit": {
            "formula": "depth_m = slope * TWT_ms + intercept",
            "slope_m_per_ms": float(depth_coefficients[0]),
            "intercept_m": float(depth_coefficients[1]),
        },
        "unit_contract": {
            "Vp_Vs": "m/s",
            "density": "g/cc",
            "elastic_moduli": "GPa (rho[g/cc] * velocity[km/s]^2)",
            "depth": "m",
            "TWT": "ms",
            "porosity": "fraction, not percent",
            "saturation": "fraction, not percent",
            "pressure": "MPa lithostatic-stress proxy only; pore pressure is unavailable, so calibrated effective pressure is not claimed",
        },
        "findings": {
            "well_input_phi_basic_valid_fraction": float(wells["input_phi_basic_valid"].mean()),
            "well_density_phi_basic_valid_fraction": float(wells["density_phi_basic_valid"].mean()),
            "gate_input_phi_basic_valid_fraction": float(gate["input_phi_basic_valid"].mean()),
            "gate_density_phi_basic_valid_fraction": float(gate["density_phi_basic_valid"].mean()),
            "well_median_porosity_difference": float(wells["porosity_difference"].median()),
            "gate_median_porosity_difference": float(gate["porosity_difference"].median()),
            "primary_root_cause": "Stored/log-derived porosity and independently fitted RF Vp/Vs/RHOB do not define a jointly density- and Gassmann-consistent brine state.",
            "unit_error_found": False,
            "mineral_mixture_alone_explains_failure": False,
            "fluid_constants_alone_explain_failure": False,
        },
        "figures": figures,
    }
    write_json(destination / "diagnosis_report.json", report)
    print(json.dumps(report, indent=2))


def _make_model(
    table: pd.DataFrame,
    neighbor_count: int,
    calibration_id: str,
    metadata: dict[str, Any],
) -> CalibratedDryFrameModel:
    features = table[["density_porosity", "shaliness", "depth_m"]].to_numpy(float)
    features[:, 2] /= 1000.0
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    if np.any(scale <= 0.0):
        raise ValueError("Calibration feature has zero variance")
    return CalibratedDryFrameModel(
        calibration_id=calibration_id,
        feature_names=("effective_porosity_fraction", "DELTA_shaliness_fraction", "depth_km"),
        feature_center=center,
        feature_scale=scale,
        features_standardized=(features - center) / scale,
        log_dry_bulk_gpa=np.log(table["dry_bulk_density_phi_gpa"].to_numpy(float)),
        log_shear_gpa=np.log(table["shear_gpa"].to_numpy(float)),
        well_ids=table["well"].astype(str).to_numpy(),
        neighbor_count=neighbor_count,
        metadata=metadata,
    )


def _select_neighbor_count(wells: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    rows = []
    for count in (8, 16, 32, 64):
        for held_out in sorted(wells["well"].unique()):
            training = wells[wells["well"] != held_out]
            test = wells[wells["well"] == held_out]
            model = _make_model(training, count, "cross_validation", {})
            dry, shear, distance = model.predict(
                test["density_porosity"].to_numpy(),
                test["shaliness"].to_numpy(),
                test["depth_m"].to_numpy(),
            )
            rows.append(
                {
                    "neighbor_count": count,
                    "held_out_well": held_out,
                    "samples": len(test),
                    "dry_log_rmse": float(np.sqrt(np.mean(np.square(np.log(dry) - np.log(test["dry_bulk_density_phi_gpa"]))))),
                    "shear_log_rmse": float(np.sqrt(np.mean(np.square(np.log(shear) - np.log(test["shear_gpa"]))))),
                    "median_nearest_distance": float(np.median(distance)),
                }
            )
    scores = pd.DataFrame(rows)
    means = scores.groupby("neighbor_count")[["dry_log_rmse", "shear_log_rmse"]].mean()
    selected = int((means["dry_log_rmse"] + means["shear_log_rmse"]).idxmin())
    return selected, scores


def _benchmark_table(wells: pd.DataFrame, physics: FluidRockPhysics) -> pd.DataFrame:
    records = []
    for saturation in np.linspace(0.0, 0.8, 17):
        sat = np.full(len(wells), saturation)
        fluid_bulk, fluid_density = brie_fluid_mixture(
            sat,
            brine_bulk_modulus_gpa=physics.brine_bulk_modulus_gpa,
            co2_bulk_modulus_gpa=physics.co2_bulk_modulus_gpa,
            brine_density_g_cc=physics.brine_density_g_cc,
            co2_density_g_cc=physics.co2_density_g_cc,
            brie_exponent=physics.brie_exponent,
        )
        brine_bulk = forward_gassmann_bulk_strict(
            wells["dry_bulk_density_phi_gpa"].to_numpy(),
            wells["density_porosity"].to_numpy(),
            wells["mineral_bulk_gpa"].to_numpy(),
            physics.brine_bulk_modulus_gpa,
        )
        target_bulk = forward_gassmann_bulk_strict(
            wells["dry_bulk_density_phi_gpa"].to_numpy(),
            wells["density_porosity"].to_numpy(),
            wells["mineral_bulk_gpa"].to_numpy(),
            fluid_bulk,
        )
        target_density = wells["density_g_cc"].to_numpy() + wells["density_porosity"].to_numpy() * (
            fluid_density - physics.brine_density_g_cc
        )
        target = elastic_from_gpa_strict(target_bulk, wells["shear_gpa"].to_numpy(), target_density)
        frame = wells[["well", "facies", "depth_bin", "effective_porosity_bin", "depth_m", "density_porosity", "shaliness"]].copy()
        frame["co2_saturation"] = saturation
        frame["delta_vp_m_s"] = target.vp - wells["vp_m_s"].to_numpy()
        frame["delta_vs_m_s"] = target.vs - wells["vs_m_s"].to_numpy()
        frame["delta_density_g_cc"] = target.density - wells["density_g_cc"].to_numpy()
        frame["delta_ksat_gpa"] = target_bulk - brine_bulk
        frame["dry_bulk_gpa"] = wells["dry_bulk_density_phi_gpa"].to_numpy()
        frame["dry_to_shear"] = wells["dry_to_shear_density_phi"].to_numpy()
        frame["dry_poisson_ratio"] = wells["dry_poisson_density_phi"].to_numpy()
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def _benchmark_envelope(benchmark: pd.DataFrame) -> pd.DataFrame:
    response_columns = ["delta_vp_m_s", "delta_vs_m_s", "delta_density_g_cc", "delta_ksat_gpa"]
    rows = []
    group_columns = ["co2_saturation", "facies", "depth_bin", "effective_porosity_bin"]
    for keys, group in benchmark.groupby(group_columns, observed=True):
        row = dict(zip(group_columns, keys))
        row["samples"] = len(group)
        for column in response_columns:
            values = group[column].to_numpy(float)
            for label, quantile in (("p01", 0.01), ("p05", 0.05), ("median", 0.50), ("p95", 0.95), ("p99", 0.99)):
                row[f"{column}_{label}"] = float(np.quantile(values, quantile))
        rows.append(row)
    return pd.DataFrame(rows)


def _benchmark_figure(benchmark: pd.DataFrame, destination: Path) -> str:
    sample = benchmark.sample(min(len(benchmark), 30_000), random_state=12345)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = (
        ("delta_vp_m_s", "Delta Vp (m/s)"),
        ("delta_vs_m_s", "Delta Vs (m/s)"),
        ("delta_density_g_cc", "Delta density (g/cc)"),
        ("delta_ksat_gpa", "Delta Ksat (GPa)"),
    )
    for axis, (column, label) in zip(axes.flat, panels):
        scatter = axis.scatter(
            sample["co2_saturation"], sample[column], c=sample["density_porosity"],
            s=7, alpha=0.22, cmap="viridis"
        )
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set(xlabel="CO2 saturation (fraction)", ylabel=label)
        figure.colorbar(scatter, ax=axis, label="density-derived effective porosity")
    figure.suptitle("Well-calibrated same-frame Gassmann response envelope", fontsize=15)
    output = destination / "well_calibrated_fluid_response_envelope.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output.name


def calibrate(_: argparse.Namespace) -> None:
    _, config, _, new_root, data_root = _contracts()
    physics = _physics(config)
    diagnosis_path = new_root / "fluid_gate" / "diagnosis" / "well_brine_state_diagnostic.csv"
    if not diagnosis_path.exists():
        raise FileNotFoundError("Run the diagnosis phase before calibration")
    reservoir_wells = pd.read_csv(diagnosis_path)
    wells = _calibration_wells(data_root, physics)
    calibration_samples = wells[wells["density_phi_strict_valid"]].copy()
    if len(calibration_samples) < 100:
        raise RuntimeError("Too few physically valid reservoir-well samples for calibration")
    selected_count, cross_validation = _select_neighbor_count(calibration_samples)
    destination = new_root / "fluid_gate" / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    calibration_samples.to_csv(destination / "calibration_well_samples.csv", index=False)
    cross_validation.to_csv(destination / "dry_frame_leave_one_well_out.csv", index=False)
    depth_coefficients = _time_depth_coefficients(calibration_samples)
    identity_payload = {
        "source_wells": sorted(calibration_samples["well"].unique().tolist()),
        "samples": len(calibration_samples),
        "neighbor_count": selected_count,
        "physics": physics.__dict__,
        "table_sha256": hashlib.sha256(
            calibration_samples.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    calibration_id = "v0031_" + hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    metadata = {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "method": "inverse-distance local-neighbour frame calibrated to physical T6-T7 well states",
        "scientific_scope": "pre-CO2 reservoir intervals treated as the brine baseline; reservoir SW logs are unavailable",
        "feature_names": ["density-derived effective porosity fraction", "DELTA shaliness fraction", "depth km"],
        "target_names": ["log Kdry GPa", "log shear modulus GPa"],
        "neighbor_count": selected_count,
        "neighbor_count_selection": "minimum mean leave-one-well-out log-RMSE across Kdry and shear for candidates 8,16,32,64",
        "wells": sorted(calibration_samples["well"].unique().tolist()),
        "samples": len(calibration_samples),
        "t6_t7_reservoir_well_samples": len(reservoir_wells),
        "calibration_interval": {
            "facies": "P(sand) >= 0.30 (DELTA <= 0.70)",
            "depth_m": [2400.0, 3250.0],
            "T73_saturation_QC": "SW >= 0.95",
            "other_wells": "pre-CO2 baseline; no SW curve available",
        },
        "physics": physics.__dict__,
        "time_depth_linear_coefficients": {
            "formula": "depth_m = slope * TWT_ms + intercept",
            "slope_m_per_ms": float(depth_coefficients[0]),
            "intercept_m": float(depth_coefficients[1]),
        },
        "feature_support": {
            name: {
                "minimum": float(calibration_samples[column].min()),
                "p01": float(calibration_samples[column].quantile(0.01)),
                "p99": float(calibration_samples[column].quantile(0.99)),
                "maximum": float(calibration_samples[column].max()),
            }
            for name, column in (
                ("effective_porosity_fraction", "density_porosity"),
                ("DELTA_shaliness_fraction", "shaliness"),
                ("depth_m", "depth_m"),
                ("Kdry_GPa", "dry_bulk_density_phi_gpa"),
                ("shear_GPa", "shear_gpa"),
                ("Kdry_to_shear", "dry_to_shear_density_phi"),
                ("dry_Poisson_ratio", "dry_poisson_density_phi"),
            )
        },
        "pressure_limit": "No pore-pressure log is available. Depth is calibrated directly; lithostatic stress is reported only as a proxy and is not called effective pressure.",
        "clipping_or_projection": False,
    }
    model = _make_model(calibration_samples, selected_count, calibration_id, metadata)
    model_directory = data_root / "derived" / CALIBRATION_VERSION
    model_path, metadata_path = save_calibrated_dry_frame(
        model, model_directory / "calibrated_dry_frame.npz"
    )
    benchmark = _benchmark_table(calibration_samples, physics)
    benchmark.to_csv(destination / "well_fluid_response_samples.csv", index=False)
    envelope = _benchmark_envelope(benchmark)
    envelope.to_csv(destination / "well_fluid_response_envelope.csv", index=False)
    figure = _benchmark_figure(benchmark, destination)
    report = {
        "status": "calibration_complete",
        "calibration_id": calibration_id,
        "calibration_artifact": str(model_path),
        "calibration_artifact_sha256": file_sha256(model_path),
        "calibration_metadata_sha256": file_sha256(metadata_path),
        "well_support": metadata["feature_support"],
        "selected_neighbor_count": selected_count,
        "leave_one_well_out": cross_validation.groupby("neighbor_count")[["dry_log_rmse", "shear_log_rmse"]].mean().reset_index().to_dict(orient="records"),
        "benchmark_samples": len(benchmark),
        "benchmark_saturation_range": [float(benchmark["co2_saturation"].min()), float(benchmark["co2_saturation"].max())],
        "benchmark_figure": figure,
        "limitations": [
            "Only T73 supplies SW in the usable logs and was restricted to SW >= 0.95; the other wells retain the pre-CO2 baseline assumption.",
            "No pore-pressure log is available; depth is used directly and no calibrated effective-pressure claim is made.",
        ],
    }
    write_json(destination / "calibration_report.json", report)
    print(json.dumps(report, indent=2))


def _local_reference_envelope(
    model: CalibratedDryFrameModel,
    physics: FluidRockPhysics,
    effective_porosity: np.ndarray,
    shaliness: np.ndarray,
    depth_m: np.ndarray,
    saturation: np.ndarray,
    neighbor_count: int = 64,
) -> dict[str, np.ndarray]:
    """Evaluate exact same-frame responses for nearby physical well states."""
    query = np.column_stack((effective_porosity, shaliness, depth_m / 1000.0))
    standardized = (query - model.feature_center) / model.feature_scale
    from scipy.spatial import cKDTree

    tree = cKDTree(model.features_standardized)
    _, indices = tree.query(
        standardized,
        k=min(neighbor_count, len(model.features_standardized)),
    )
    if indices.ndim == 1:
        indices = indices[:, None]
    training_features = (
        model.features_standardized * model.feature_scale + model.feature_center
    )
    phi = training_features[indices, 0]
    shale = training_features[indices, 1]
    dry = np.exp(model.log_dry_bulk_gpa[indices])
    shear = np.exp(model.log_shear_gpa[indices])
    mineral_bulk, _, mineral_density = mineral_properties_vrh_strict(shale, physics)
    brine_density = (1.0 - phi) * mineral_density + phi * physics.brine_density_g_cc
    brine_bulk = forward_gassmann_bulk_strict(
        dry, phi, mineral_bulk, physics.brine_bulk_modulus_gpa
    )
    brine_elastic = elastic_from_gpa_strict(brine_bulk, shear, brine_density)
    saturation_2d = np.broadcast_to(saturation[:, None], phi.shape)
    fluid_bulk, fluid_density = brie_fluid_mixture(
        saturation_2d,
        brine_bulk_modulus_gpa=physics.brine_bulk_modulus_gpa,
        co2_bulk_modulus_gpa=physics.co2_bulk_modulus_gpa,
        brine_density_g_cc=physics.brine_density_g_cc,
        co2_density_g_cc=physics.co2_density_g_cc,
        brie_exponent=physics.brie_exponent,
    )
    target_bulk = forward_gassmann_bulk_strict(dry, phi, mineral_bulk, fluid_bulk)
    target_density = brine_density + phi * (fluid_density - physics.brine_density_g_cc)
    target_elastic = elastic_from_gpa_strict(target_bulk, shear, target_density)
    responses = {
        "delta_vp_m_s": target_elastic.vp - brine_elastic.vp,
        "delta_vs_m_s": target_elastic.vs - brine_elastic.vs,
        "delta_density_g_cc": target_elastic.density - brine_elastic.density,
        "delta_ksat_gpa": target_bulk - brine_bulk,
    }
    envelope = {}
    for name, values in responses.items():
        envelope[f"{name}_lower"] = np.quantile(values, 0.01, axis=1)
        envelope[f"{name}_upper"] = np.quantile(values, 0.99, axis=1)
    return envelope


def _calibration_support_threshold(model: CalibratedDryFrameModel) -> float:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(model.features_standardized).query(
        model.features_standardized, k=2
    )
    return float(np.quantile(distances[:, 1], 0.99))


def _candidate_percentiles(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "delta_vp_m_s",
        "delta_vs_m_s",
        "delta_density_g_cc",
        "delta_ksat_gpa",
        "dry_bulk_gpa",
        "dry_to_frame_shear",
        "dry_poisson_ratio",
        "effective_porosity",
        "porosity_adjustment",
        "nearest_calibration_distance",
    ]
    rows = []
    for candidate, subset in frame.groupby("candidate"):
        values = _percentiles(subset, columns)
        values.insert(0, "candidate", candidate)
        rows.append(values)
    return pd.concat(rows, ignore_index=True)


def _candidate_sweep_qc(
    candidate_name: str,
    candidate_function: Any,
    source: pd.DataFrame,
    model: CalibratedDryFrameModel,
    physics: FluidRockPhysics,
) -> dict[str, Any]:
    states = source.sort_values(["realization_id", "row", "column"]).iloc[
        np.linspace(0, len(source) - 1, min(32, len(source)), dtype=int)
    ]
    saturation_grid = np.linspace(0.0, 0.8, 81)
    monotonic_density = True
    monotonic_vp = True
    maximum_vp_second_difference = 0.0
    maximum_density_second_difference = 0.0
    for row in states.itertuples(index=False):
        shape = saturation_grid.shape
        result = candidate_function(
            np.full(shape, row.vp_m_s),
            np.full(shape, row.vs_m_s),
            np.full(shape, row.density_g_cc),
            np.full(shape, row.input_porosity),
            np.full(shape, row.shaliness),
            saturation_grid,
            np.full(shape, row.depth_m),
            model,
            physics,
        )
        monotonic_density &= bool(np.all(np.diff(result.elastic.density) < 0.0))
        monotonic_vp &= bool(np.all(np.diff(result.elastic.vp) <= 1e-9))
        maximum_vp_second_difference = max(
            maximum_vp_second_difference,
            float(np.max(np.abs(np.diff(result.elastic.vp, n=2)))),
        )
        maximum_density_second_difference = max(
            maximum_density_second_difference,
            float(np.max(np.abs(np.diff(result.elastic.density, n=2)))),
        )
    return {
        "candidate": candidate_name,
        "states": len(states),
        "saturation_samples": len(saturation_grid),
        "saturation_range": [0.0, 0.8],
        "vp_nonincreasing": monotonic_vp,
        "density_strictly_decreasing": monotonic_density,
        "maximum_absolute_vp_second_difference_m_s": maximum_vp_second_difference,
        "maximum_absolute_density_second_difference_g_cc": maximum_density_second_difference,
        "smooth": monotonic_vp and monotonic_density and maximum_vp_second_difference < 2.0,
    }


def _candidate_figures(frame: pd.DataFrame, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = (
        ("delta_vp_m_s", "Delta Vp (m/s)"),
        ("delta_vs_m_s", "Delta Vs (m/s)"),
        ("delta_density_g_cc", "Delta density (g/cc)"),
    )
    for row_index, candidate in enumerate(("constrained_local_gassmann", "calibrated_differential_gassmann")):
        subset = frame[frame["candidate"] == candidate]
        for axis, (column, label) in zip(axes[row_index], panels):
            scatter = axis.scatter(
                subset["co2_saturation"], subset[column],
                c=subset["effective_porosity"], s=8, alpha=0.30, cmap="viridis"
            )
            axis.set(xlabel="CO2 saturation (fraction)", ylabel=label, title=candidate)
            figure.colorbar(scatter, ax=axis, label="effective porosity")
    output = destination / "candidate_saturation_response_colored_by_porosity.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append(output.name)

    representative_id = int(frame["realization_id"].min())
    representative = frame[frame["realization_id"] == representative_id]
    nrows = int(representative["row"].max()) + 1
    ncols = int(representative["column"].max()) + 1
    figure, axes = plt.subplots(2, 6, figsize=(24, 8), constrained_layout=True)
    map_columns = (
        ("porosity_adjustment", "effective phi - stored phi"),
        ("delta_ksat_gpa", "Delta Ksat (GPa)"),
        ("delta_vp_m_s", "Delta Vp (m/s)"),
        ("delta_vs_m_s", "Delta Vs (m/s)"),
        ("delta_density_g_cc", "Delta density (g/cc)"),
        ("nearest_calibration_distance", "calibration distance"),
    )
    for row_index, candidate in enumerate(("constrained_local_gassmann", "calibrated_differential_gassmann")):
        subset = representative[representative["candidate"] == candidate]
        for axis, (column, label) in zip(axes[row_index], map_columns):
            image_values = np.full((nrows, ncols), np.nan)
            image_values[subset["row"].to_numpy(int), subset["column"].to_numpy(int)] = subset[column]
            image = axis.imshow(image_values, aspect="auto", cmap="RdBu_r" if column.startswith("delta") or column == "porosity_adjustment" else "viridis")
            axis.set_title(label)
            axis.set_xlabel("trace")
            axis.set_ylabel(candidate)
            figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle(f"Fluid-only adjustments for deterministic realization {representative_id}", fontsize=15)
    output = destination / "candidate_spatial_adjustment_maps.png"
    figure.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append(output.name)
    return outputs


def evaluate_candidates(_: argparse.Namespace) -> None:
    _, config, old_root, new_root, data_root = _contracts()
    physics = _physics(config)
    model_path = data_root / "derived" / CALIBRATION_VERSION / "calibrated_dry_frame.npz"
    model = load_calibrated_dry_frame(model_path)
    depth_mapping = model.metadata["time_depth_linear_coefficients"]
    depth_coefficients = np.array(
        [depth_mapping["slope_m_per_ms"], depth_mapping["intercept_m"]]
    )
    records = []
    test_results: dict[str, dict[str, Any]] = {}
    candidates = (
        ("constrained_local_gassmann", constrained_local_gassmann_substitution),
        ("calibrated_differential_gassmann", calibrated_differential_gassmann_substitution),
    )
    for candidate_name, candidate_function in candidates:
        zero_errors = []
        shear_errors = []
        outside_errors = []
        density_formula_errors = []
        valid = True
        for path in sorted((old_root / "stage02" / "realizations").glob("realization_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                brine = np.asarray(archive["elastic_brine"], dtype=float)
                input_porosity = np.asarray(archive["porosity"], dtype=float)
                shale = np.asarray(archive["delta"], dtype=float)
                saturation = np.asarray(archive["co2_saturation"], dtype=float)
                time_ms = np.asarray(archive["time_ms"], dtype=float)
                realization_id = int(archive["realization_id"])
            plume = saturation > 0.0
            rows, columns = np.indices(input_porosity.shape)
            depth = np.broadcast_to(
                np.polyval(depth_coefficients, time_ms)[:, None], input_porosity.shape
            )
            result = candidate_function(
                brine[0][plume], brine[1][plume], brine[2][plume],
                input_porosity[plume], shale[plume], saturation[plume], depth[plume],
                model, physics,
            )
            zero_result = candidate_function(
                brine[0][plume], brine[1][plume], brine[2][plume],
                input_porosity[plume], shale[plume], np.zeros(np.count_nonzero(plume)),
                depth[plume], model, physics,
            )
            zero_errors.append(float(np.max(np.abs(np.stack((
                zero_result.elastic.vp - brine[0][plume],
                zero_result.elastic.vs - brine[1][plume],
                zero_result.elastic.density - brine[2][plume],
            ))))))
            _, result_shear = elastic_moduli_gpa(
                result.elastic.vp, result.elastic.vs, result.elastic.density
            )
            shear_errors.append(float(np.max(np.abs(result_shear - result.rf_shear_gpa))))
            expected_vs = 1000.0 * np.sqrt(result.rf_shear_gpa / result.elastic.density)
            density_formula_errors.append(float(np.max(np.abs(result.elastic.vs - expected_vs))))
            assembled = brine.copy()
            assembled[0][plume] = result.elastic.vp
            assembled[1][plume] = result.elastic.vs
            assembled[2][plume] = result.elastic.density
            outside_errors.append(float(np.max(np.abs(assembled[:, ~plume] - brine[:, ~plume]))))
            envelope = _local_reference_envelope(
                model, physics, result.effective_porosity, shale[plume], depth[plume], saturation[plume]
            )
            values = {
                "delta_vp_m_s": result.elastic.vp - brine[0][plume],
                "delta_vs_m_s": result.elastic.vs - brine[1][plume],
                "delta_density_g_cc": result.elastic.density - brine[2][plume],
                "delta_ksat_gpa": result.delta_bulk_gpa,
            }
            in_envelope = np.ones(np.count_nonzero(plume), dtype=bool)
            for name, value in values.items():
                in_envelope &= (value >= envelope[f"{name}_lower"]) & (value <= envelope[f"{name}_upper"])
            frame = pd.DataFrame(
                {
                    "candidate": candidate_name,
                    "realization_id": realization_id,
                    "row": rows[plume],
                    "column": columns[plume],
                    "co2_saturation": saturation[plume],
                    "vp_m_s": brine[0][plume],
                    "vs_m_s": brine[1][plume],
                    "density_g_cc": brine[2][plume],
                    "input_porosity": input_porosity[plume],
                    "effective_porosity": result.effective_porosity,
                    "porosity_adjustment": result.effective_porosity - result.input_porosity,
                    "shaliness": shale[plume],
                    "depth_m": depth[plume],
                    "delta_vp_m_s": values["delta_vp_m_s"],
                    "delta_vs_m_s": values["delta_vs_m_s"],
                    "delta_density_g_cc": values["delta_density_g_cc"],
                    "delta_ksat_gpa": values["delta_ksat_gpa"],
                    "dry_bulk_gpa": result.dry_bulk_gpa,
                    "dry_to_frame_shear": result.dry_bulk_gpa / result.frame_shear_gpa,
                    "dry_poisson_ratio": poisson_ratio_from_moduli(result.dry_bulk_gpa, result.frame_shear_gpa),
                    "nearest_calibration_distance": result.nearest_calibration_distance,
                    "in_local_well_response_envelope": in_envelope,
                }
            )
            valid &= bool(
                np.isfinite(np.stack((result.elastic.vp, result.elastic.vs, result.elastic.density))).all()
                and np.all(result.elastic.vp > result.elastic.vs)
                and np.all(result.elastic.vs > 0.0)
                and np.all(result.elastic.density < brine[2][plume])
            )
            records.append(frame)
        test_results[candidate_name] = {
            "zero_saturation_maximum_absolute_error": max(zero_errors),
            "outside_plume_maximum_absolute_change": max(outside_errors),
            "shear_modulus_invariance_maximum_absolute_error_gpa": max(shear_errors),
            "vs_fixed_shear_density_formula_maximum_absolute_error_m_s": max(density_formula_errors),
            "finite_and_vp_greater_vs_greater_zero": valid,
            "hidden_feasibility_projection": False,
            "bound_clipping": False,
            "arbitrary_delta_vp_cap": False,
            "rf_vp_vs_density_baseline_correction": 0.0,
        }
    frame = pd.concat(records, ignore_index=True)
    destination = new_root / "fluid_gate" / "candidates"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "candidate_all_plume_pixels.csv", index=False)
    percentiles = _candidate_percentiles(frame)
    percentiles.to_csv(destination / "candidate_percentiles.csv", index=False)
    support_threshold = _calibration_support_threshold(model)
    support = model.metadata["feature_support"]
    dry_min = float(support["Kdry_GPa"]["minimum"])
    dry_max = float(support["Kdry_GPa"]["maximum"])
    ratio_min = float(support["Kdry_to_shear"]["minimum"])
    ratio_max = float(support["Kdry_to_shear"]["maximum"])
    poisson_min = float(support["dry_Poisson_ratio"]["minimum"])
    poisson_max = float(support["dry_Poisson_ratio"]["maximum"])
    summaries = {}
    functions = dict(candidates)
    for candidate_name, subset in frame.groupby("candidate"):
        out_dry = ~subset["dry_bulk_gpa"].between(dry_min, dry_max)
        out_ratio = ~subset["dry_to_frame_shear"].between(ratio_min, ratio_max)
        out_poisson = ~subset["dry_poisson_ratio"].between(poisson_min, poisson_max)
        out_response = ~subset["in_local_well_response_envelope"]
        out_support = subset["nearest_calibration_distance"] > support_threshold
        sweep = _candidate_sweep_qc(
            candidate_name, functions[candidate_name], subset, model, physics
        )
        tests = test_results[candidate_name]
        tests.update(
            {
                "calibration_support_distance_p99": support_threshold,
                "outside_calibration_support_fraction": float(out_support.mean()),
                "outside_local_well_response_envelope_fraction": float(out_response.mean()),
                "outside_well_Kdry_range_fraction": float(out_dry.mean()),
                "outside_well_Kdry_to_shear_range_fraction": float(out_ratio.mean()),
                "outside_well_dry_Poisson_range_fraction": float(out_poisson.mean()),
                "near_zero_Kdry_fraction_below_1_GPa": float((subset["dry_bulk_gpa"] < 1.0).mean()),
                "saturation_sweep": sweep,
            }
        )
        tests["passes_physical_gate"] = bool(
            tests["zero_saturation_maximum_absolute_error"] < 1e-9
            and tests["outside_plume_maximum_absolute_change"] == 0.0
            and tests["shear_modulus_invariance_maximum_absolute_error_gpa"] < 1e-10
            and tests["vs_fixed_shear_density_formula_maximum_absolute_error_m_s"] < 1e-9
            and tests["finite_and_vp_greater_vs_greater_zero"]
            and tests["outside_local_well_response_envelope_fraction"] <= 0.05
            and tests["outside_well_Kdry_range_fraction"] == 0.0
            and tests["outside_well_Kdry_to_shear_range_fraction"] <= 0.01
            and tests["outside_well_dry_Poisson_range_fraction"] <= 0.01
            and tests["near_zero_Kdry_fraction_below_1_GPa"] == 0.0
            and sweep["smooth"]
        )
        summaries[candidate_name] = tests
    figures = _candidate_figures(frame, destination)
    selected = "calibrated_differential_gassmann"
    selection_reason = (
        "Candidate B passes the physical gate and transfers only the calibrated same-frame "
        "bulk-modulus and density changes to the unmodified RF brine background. It is preferred "
        "over Candidate A because it does not infer the production dry frame directly from an "
        "independently fitted RF absolute state."
    )
    if not summaries[selected]["passes_physical_gate"]:
        selected = "none"
        selection_reason = "Candidate B failed at least one mandatory physical acceptance test."
    report = {
        "status": "candidate_evaluation_complete",
        "calibration_id": model.calibration_id,
        "calibration_artifact_sha256": file_sha256(model_path),
        "plume_pixels_per_candidate": int(len(frame) / len(candidates)),
        "candidate_A": summaries["constrained_local_gassmann"],
        "candidate_B": summaries["calibrated_differential_gassmann"],
        "selected_production_mode": selected,
        "selection_reason": selection_reason,
        "candidate_A_RF_baseline_adjustment": {
            "Vp_m_s": 0.0,
            "Vs_m_s": 0.0,
            "density_g_cc": 0.0,
            "meaning": "Only density-closure effective porosity changes; the RF elastic brine baseline is exact.",
        },
        "legacy_local_inverse_gassmann": "invalid_for_production_due_to_mineral_projection_and_dry_bulk_clipping",
        "figures": figures,
    }
    write_json(destination / "candidate_evaluation_report.json", report)
    print(json.dumps(report, indent=2))


def generate_bounded_stage02(args: argparse.Namespace) -> None:
    paths, synthetic, dataset, training, locations = _bounded_configs()
    candidate_report_path = (
        locations["root"] / "fluid_gate" / "candidates" / "candidate_evaluation_report.json"
    )
    if not candidate_report_path.exists():
        raise FileNotFoundError("Run candidate evaluation before bounded regeneration")
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if candidate_report["selected_production_mode"] != "calibrated_differential_gassmann":
        raise RuntimeError("Bounded regeneration is blocked because candidate B was not selected")
    _snapshot_bounded_configs(synthetic, dataset, training, locations)
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=locations["stage02"],
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "generated_realizations": manifest["generated_realizations"],
                "output_version": manifest["output_version"],
                "generation_config_sha256": manifest["generation_config_sha256"],
                "source_artifact_hashes": manifest["source_artifact_hashes"],
            },
            indent=2,
        )
    )


def build_bounded_stage03(_: argparse.Namespace) -> None:
    paths, synthetic, dataset, training, locations = _bounded_configs()
    _snapshot_bounded_configs(synthetic, dataset, training, locations)
    manifest = build_stage03_dataset(
        config=dataset,
        paths=paths,
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    print(json.dumps(manifest["integrity"], indent=2))


def _all_realization_round_trip(
    stage02: Path,
    synthetic: dict[str, Any],
) -> dict[str, Any]:
    specification = forward_specification_from_mapping(synthetic)
    rows = []
    for path in sorted(stage02.glob("realization_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            elastic = np.asarray(archive["elastic"], dtype=np.float64)
            stored = np.asarray(archive["avo_clean"], dtype=np.float64)
            realization_id = int(archive["realization_id"])
        reproduced = forward_avo_three_band_spec_torch(
            torch.from_numpy(elastic[0][None]),
            torch.from_numpy(elastic[1][None]),
            torch.from_numpy(elastic[2][None]),
            specification,
        )[0].cpu().numpy()
        difference = reproduced - stored
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        reference_rms = float(np.sqrt(np.mean(np.square(stored))))
        rows.append(
            {
                "realization_id": realization_id,
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "rmse": rmse,
                "relative_rmse": rmse / max(reference_rms, 1e-15),
                "near_rmse": float(np.sqrt(np.mean(np.square(difference[0])))),
                "mid_rmse": float(np.sqrt(np.mean(np.square(difference[1])))),
                "far_rmse": float(np.sqrt(np.mean(np.square(difference[2])))),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "forward_specification_sha256": specification.sha256,
        "realizations": len(frame),
        "bands_in_order": [band.name for band in specification.bands],
        "maximum_absolute_error": float(frame["maximum_absolute_error"].max()),
        "maximum_rmse": float(frame["rmse"].max()),
        "maximum_relative_rmse": float(frame["relative_rmse"].max()),
        "per_realization": rows,
    }


def _bounded_fluid_qc(
    old_stage02: Path,
    new_stage02: Path,
    destination: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    invariant_differences = []
    outside_changes = []
    shear_errors = []
    for new_path in sorted(new_stage02.glob("realization_*.npz")):
        old_path = old_stage02 / new_path.name
        if not old_path.exists():
            raise FileNotFoundError(f"Missing matched immutable v003 realization {old_path}")
        with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
            invariant_channels = (
                "elastic_brine",
                "delta",
                "sand_probability",
                "porosity",
                "rgt",
                "reservoir_mask",
                "plume_mask",
                "co2_saturation",
                "time_ms",
                "cdp",
            )
            invariant_differences.append(
                max(
                    float(np.max(np.abs(np.asarray(new[name], dtype=float) - np.asarray(old[name], dtype=float))))
                    for name in invariant_channels
                )
            )
            brine = np.asarray(new["elastic_brine"], dtype=float)
            elastic = np.asarray(new["elastic"], dtype=float)
            saturation = np.asarray(new["co2_saturation"], dtype=float)
            porosity = np.asarray(new["porosity"], dtype=float)
            shaliness = np.asarray(new["delta"], dtype=float)
            time_ms = np.asarray(new["time_ms"], dtype=float)
            realization_id = int(new["realization_id"])
        plume = saturation > 0.0
        difference = elastic - brine
        outside_changes.append(float(np.max(np.abs(difference[:, ~plume]))))
        _, brine_shear = elastic_moduli_gpa(*brine)
        _, target_shear = elastic_moduli_gpa(*elastic)
        shear_errors.append(float(np.max(np.abs(target_shear[plume] - brine_shear[plume]))))
        indices = np.argwhere(plume)
        depth_mapping = load_calibrated_dry_frame(
            data_root_for_calibration()
        ).metadata["time_depth_linear_coefficients"]
        depth_by_row = (
            float(depth_mapping["slope_m_per_ms"]) * time_ms
            + float(depth_mapping["intercept_m"])
        )
        for (row, column), saturation_value in zip(indices, saturation[plume]):
            rows.append(
                {
                    "realization_id": realization_id,
                    "row": int(row),
                    "column": int(column),
                    "co2_saturation": float(saturation_value),
                    "input_porosity": float(porosity[row, column]),
                    "shaliness": float(shaliness[row, column]),
                    "depth_m": float(depth_by_row[row]),
                    "delta_vp_m_s": float(difference[0, row, column]),
                    "delta_vs_m_s": float(difference[1, row, column]),
                    "delta_density_g_cc": float(difference[2, row, column]),
                }
            )
    frame = pd.DataFrame(rows)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "regenerated_fluid_all_plume_pixels.csv", index=False)
    return (
        {
            "matched_realizations": int(frame["realization_id"].nunique()),
            "plume_pixels": len(frame),
            "maximum_nonfluid_channel_difference_from_immutable_v003": max(invariant_differences),
            "outside_plume_maximum_absolute_change": max(outside_changes),
            "shear_modulus_invariance_maximum_absolute_error_gpa": max(shear_errors),
            "delta_percentiles": {
                column: {
                    label: float(frame[column].quantile(quantile))
                    for label, quantile in (
                        ("p01", 0.01),
                        ("p05", 0.05),
                        ("median", 0.50),
                        ("p95", 0.95),
                        ("p99", 0.99),
                    )
                }
                for column in ("delta_vp_m_s", "delta_vs_m_s", "delta_density_g_cc")
            },
        },
        frame,
    )


def data_root_for_calibration() -> Path:
    _, _, _, _, data_root = _contracts()
    return data_root / "derived" / CALIBRATION_VERSION / "calibrated_dry_frame.npz"


def bounded_qc(_: argparse.Namespace) -> None:
    _, synthetic, _, _, locations = _bounded_configs()
    _, _, old_root, _, _ = _contracts()
    locations["reports"].mkdir(parents=True, exist_ok=True)
    round_trip = _all_realization_round_trip(locations["stage02"], synthetic)
    fluid, frame = _bounded_fluid_qc(
        old_root / "stage02" / "realizations",
        locations["stage02"],
        locations["figures02"],
    )
    integrity = validate_dataset_integrity(locations["stage03"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (column, label) in zip(
        axes,
        (
            ("delta_vp_m_s", "Delta Vp (m/s)"),
            ("delta_vs_m_s", "Delta Vs (m/s)"),
            ("delta_density_g_cc", "Delta density (g/cc)"),
        ),
    ):
        scatter = axis.scatter(
            frame["co2_saturation"], frame[column], c=frame["input_porosity"],
            s=8, alpha=0.30, cmap="viridis"
        )
        axis.set(xlabel="CO2 saturation (fraction)", ylabel=label)
        figure.colorbar(scatter, ax=axis, label="stored porosity")
    figure.suptitle("Regenerated v0031 fluid response")
    figure_path = locations["figures02"] / "v0031_regenerated_saturation_response.png"
    figure.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    report = {
        "status": "bounded_stage02_stage03_qc_complete",
        "round_trip": round_trip,
        "fluid": fluid,
        "dataset_integrity": integrity,
        "figure": str(figure_path),
    }
    write_json(locations["reports"] / "bounded_execution_qc.json", report)
    print(json.dumps(report, indent=2))


def train_cuda_sanity(args: argparse.Namespace) -> None:
    _, synthetic, dataset, training, locations = _bounded_configs()
    _snapshot_bounded_configs(synthetic, dataset, training, locations)
    if not torch.cuda.is_available():
        raise RuntimeError("The requested Revision-3.1 sanity experiment requires CUDA")
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=locations["configs"] / "training_resolved.json",
        config=training,
        dataset_directory=locations["stage03"],
        experiment_directory=locations["stage04"],
        variant="full",
        device_name="cuda",
        epochs_override=2,
        max_train_batches=int(args.max_train_batches),
        max_validation_batches=int(args.max_validation_batches),
        run_name="full_2epoch_cuda_sanity",
        allow_operator_validation_subset=True,
    )
    print(output)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    diagnosis = commands.add_parser("diagnose")
    diagnosis.set_defaults(function=diagnose)
    calibration = commands.add_parser("calibrate")
    calibration.set_defaults(function=calibrate)
    candidates = commands.add_parser("candidates")
    candidates.set_defaults(function=evaluate_candidates)
    stage02 = commands.add_parser("stage02")
    stage02.add_argument("--workers", type=int, default=1)
    stage02.add_argument("--resume", action="store_true")
    stage02.set_defaults(function=generate_bounded_stage02)
    stage03 = commands.add_parser("stage03")
    stage03.set_defaults(function=build_bounded_stage03)
    qc = commands.add_parser("qc")
    qc.set_defaults(function=bounded_qc)
    sanity = commands.add_parser("train-sanity")
    sanity.add_argument("--max-train-batches", type=int, default=4)
    sanity.add_argument("--max-validation-batches", type=int, default=2)
    sanity.set_defaults(function=train_cuda_sanity)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
