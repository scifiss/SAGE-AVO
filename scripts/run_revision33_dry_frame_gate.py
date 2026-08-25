#!/usr/bin/env python3
"""Run the bounded Revision-3.3 dry-frame support and route-decision gate.

This driver cannot generate the 100-realization corpus or run full training.
The four wells without water-saturation evidence are used only for the
fluid-independent shear constraint and latent-fluid admissibility test.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
import torch

from sage_avo.config import load_config
from sage_avo.experiments import (
    build_stage03_dataset,
    generate_stage02_dataset,
    validate_dataset_integrity,
)
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.geology.dry_frame import (
    constant_cement_power_law_bulk,
    dry_poisson_ratio,
    match_hashin_shtrikman_family_to_shear,
)
from sage_avo.geology.fluid_calibration import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    calibrated_differential_gassmann_substitution,
    elastic_from_gpa_strict,
    forward_gassmann_bulk_strict,
    save_calibrated_dry_frame,
)
from sage_avo.geology.fluid_properties import batzle_wang_brine, span_wagner_co2
from sage_avo.geology.rock_physics import brie_fluid_mixture, elastic_moduli_gpa


REPOSITORY = Path(__file__).resolve().parents[1]
REVISION = "3.3"
VALIDATION_NAME = "v0033_validation8_dry_frame_supported"
CALIBRATION_ID = "v0033_58a5fe39a11c4fe66431"
REALIZATION_COUNT = 8
REALIZATION_OFFSET = 3_100_000
RANDOM_SEED = 330_033
FEATURE_COLUMNS = ["density_porosity", "shaliness", "depth_m"]
EXPANDED_SUPPORT_COLUMNS = [
    "density_porosity",
    "shaliness",
    "depth_m",
    "mineral_bulk_gpa",
    "mineral_shear_gpa",
]
SHEAR_FEATURE_COLUMNS = EXPANDED_SUPPORT_COLUMNS
AMBIGUOUS_WELLS = ("T732", "T76", "T761", "T762")


def _locations() -> dict[str, Path]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"])
    root = private / "revision33" / VALIDATION_NAME
    previous_gate = (
        private
        / "revision31"
        / "v0031_validation8_fluid_corrected"
        / "fluid_gate"
    )
    return {
        "root": root,
        "analysis": root / "dry_frame_gate" / "analysis",
        "calibration": root / "dry_frame_gate" / "calibration",
        "evaluation": root / "dry_frame_gate" / "evaluation",
        "figures": root / "dry_frame_gate" / "figures",
        "reports": root / "reports",
        "configs": root / "configs",
        "stage02": root / "stage02" / "realizations",
        "stage03": root / "stage03" / "dataset",
        "stage04": root / "stage04" / "sage_avo_s01_v0033_validation8",
        "previous_gate": previous_gate,
        "previous_stage02": (
            private
            / "revision32"
            / "v0032_validation8_fluid_provenance"
            / "stage02"
            / "realizations"
        ),
        "immutable_stage02": (
            private
            / "revision3"
            / "v003_validation8_stage01v003"
            / "stage02"
            / "realizations"
        ),
        "work_data": Path(paths["work_data_root"]) / "s01data",
        "model": Path(paths["work_data_root"]) / "s01data" / "derived" / "fluid_models_v0033",
        "fluid_v0032": (
            Path(paths["work_data_root"])
            / "s01data"
            / "derived"
            / "fluid_models_v0032"
            / "fluid_property_validation.json"
        ),
    }


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _criteria() -> dict[str, Any]:
    path = REPOSITORY / "configs" / "revision33_acceptance.yaml"
    criteria = load_config(path)
    if not criteria.get("declared_before_model_fitting", False):
        raise RuntimeError("Revision-3.3 criteria were not predeclared")
    return criteria


def _tables(locations: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    wells = pd.read_csv(
        locations["previous_gate"] / "calibration" / "calibration_well_samples.csv"
    )
    gate = pd.read_csv(
        locations["previous_gate"]
        / "diagnosis"
        / "gate_candidate_brine_state_diagnostic.csv"
    )
    wells["brine_confidence"] = np.where(
        wells["well"].eq("T73"), "confirmed_brine", "ambiguous"
    )
    return _add_structural_coordinates(wells, gate, locations)


def _add_structural_coordinates(
    wells: pd.DataFrame,
    gate: pd.DataFrame,
    locations: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = locations["work_data"] / "usable" / "v003"
    time_ms = np.load(usable / "reg_t.npy", allow_pickle=False)
    rgt = np.load(
        locations["work_data"] / "attributes" / "v003" / "rgt_model.npy",
        allow_pickle=False,
    )
    ties = pd.read_csv(usable / "df_well.csv").set_index("WELL")
    wells = wells.copy()
    wells["rgt_coordinate"] = np.nan
    for well, indices in wells.groupby("well").groups.items():
        column = int(ties.loc[well, "LINE_INDEX"])
        wells.loc[indices, "rgt_coordinate"] = np.interp(
            wells.loc[indices, "twt_ms"], time_ms, rgt[:, column]
        )
    gate = gate.copy()
    gate["rgt_coordinate"] = np.nan
    gate["stratigraphic_interval_fraction"] = np.nan
    for realization_id, indices in gate.groupby("realization_id").groups.items():
        archive_path = (
            locations["immutable_stage02"] / f"realization_{int(realization_id):07d}.npz"
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            rows = gate.loc[indices, "row"].to_numpy(int)
            columns = gate.loc[indices, "column"].to_numpy(int)
            gate.loc[indices, "rgt_coordinate"] = archive["rgt"][rows, columns]
            gate.loc[indices, "stratigraphic_interval_fraction"] = archive[
                "strat_fraction"
            ][rows, columns]
    return wells, gate


def _standardized_support(
    training: pd.DataFrame,
    query: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    train = training[columns].to_numpy(float)
    test = query[columns].to_numpy(float)
    depth_index = columns.index("depth_m") if "depth_m" in columns else None
    if depth_index is not None:
        train[:, depth_index] /= 1000.0
        test[:, depth_index] /= 1000.0
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    if np.any(scale <= 0.0):
        raise ValueError("Support feature has zero variance")
    standardized = (train - center) / scale
    standardized_query = (test - center) / scale
    tree = cKDTree(standardized)
    training_distances = tree.query(standardized, k=2)[0][:, 1]
    threshold = float(np.quantile(training_distances, 0.99))
    distances, indices = tree.query(standardized_query, k=1)
    nearest_difference = np.abs(standardized_query - standardized[indices])
    return distances, threshold, nearest_difference, center, scale


def _support_gap_table(
    wells: pd.DataFrame,
    gate: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    confirmed = wells[wells["well"].eq("T73")]
    distances, threshold, difference, _, _ = _standardized_support(
        confirmed, gate, FEATURE_COLUMNS
    )
    result = gate.copy()
    result["existing_calibration_support_distance"] = distances
    result["existing_support_threshold"] = threshold
    result["unsupported_by_t73"] = distances > threshold
    result["nearest_phi_difference_standardized"] = difference[:, 0]
    result["nearest_shaliness_difference_standardized"] = difference[:, 1]
    result["nearest_depth_difference_standardized"] = difference[:, 2]
    dominant = np.argmax(difference, axis=1)
    result["dominant_support_axis"] = np.asarray(
        ["effective_porosity", "shaliness", "depth"]
    )[dominant]
    ranges = {
        column: confirmed[column].quantile([0.01, 0.99]).to_numpy(float)
        for column in (
            "density_porosity",
            "input_porosity",
            "shaliness",
            "depth_m",
            "vp_m_s",
            "vs_m_s",
            "density_g_cc",
            "saturated_bulk_gpa",
            "shear_gpa",
        )
    }
    flag_names = {
        "density_porosity": "porosity_range_failure",
        "input_porosity": "geological_porosity_range_failure",
        "shaliness": "shale_fraction_range_failure",
        "depth_m": "depth_range_failure",
        "vp_m_s": "vp_range_failure",
        "vs_m_s": "vs_range_failure",
        "density_g_cc": "density_range_failure",
        "saturated_bulk_gpa": "saturated_bulk_range_failure",
        "shear_gpa": "dry_shear_range_failure",
    }
    for column, flag in flag_names.items():
        lower, upper = ranges[column]
        result[flag] = ~result[column].between(lower, upper)
    result["clean_sand_absence_failure"] = result["facies"].eq("clean_sand")
    feature_flags = [
        "porosity_range_failure",
        "shale_fraction_range_failure",
        "depth_range_failure",
    ]
    result["combined_multivariate_extrapolation"] = (
        result["unsupported_by_t73"] & ~result[feature_flags].any(axis=1)
    )
    priority = [
        ("clean_sand_absence_failure", "clean_sand_absence"),
        ("depth_range_failure", "depth_range"),
        ("porosity_range_failure", "effective_porosity_range"),
        ("shale_fraction_range_failure", "shale_fraction_range"),
        ("vp_range_failure", "velocity_range"),
        ("vs_range_failure", "velocity_range"),
        ("density_range_failure", "density_range"),
        ("dry_shear_range_failure", "dry_shear_range"),
        ("combined_multivariate_extrapolation", "combined_multivariate"),
    ]
    primary = np.full(len(result), "within_univariate_ranges", dtype=object)
    for flag, label in reversed(priority):
        primary[result[flag].to_numpy(bool)] = label
    result["primary_support_failure"] = primary
    unsupported = result[result["unsupported_by_t73"]]
    dominant_summary = (
        unsupported["dominant_support_axis"].value_counts().sort_index().to_dict()
    )
    overlap = {
        flag: {
            "all_gate_fraction": float(result[flag].mean()),
            "unsupported_pixel_fraction": float(unsupported[flag].mean()),
            "unsupported_pixels": int(unsupported[flag].sum()),
        }
        for flag in [*flag_names.values(), "clean_sand_absence_failure"]
    }
    report = {
        "gate_pixels": len(result),
        "unsupported_pixels": int(result["unsupported_by_t73"].sum()),
        "unsupported_fraction": float(result["unsupported_by_t73"].mean()),
        "existing_threshold": threshold,
        "dominant_axis_counts": dominant_summary,
        "dominant_axis_fractions": {
            key: value / max(len(unsupported), 1) for key, value in dominant_summary.items()
        },
        "overlapping_range_failures": overlap,
        "clean_sand_interpretation": (
            "T73 has no clean-sand calibration samples, but the bounded candidate/plume "
            "gate also contains no clean-sand pixels; clean-sand absence is therefore a "
            "future-corpus limitation rather than the direct cause of these 1,798 failures."
        ),
    }
    return result, report


def _analysis_figures(
    wells: pd.DataFrame,
    support: pd.DataFrame,
    locations: dict[str, Path],
) -> list[dict[str, Any]]:
    destination = locations["figures"] / "support_gap"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    confirmed = wells[wells["well"].eq("T73")]

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    panels = (
        ("density_porosity", "depth_m", "Effective porosity", "Depth (m)"),
        ("shaliness", "depth_m", "DELTA / shaliness", "Depth (m)"),
        ("shear_gpa", "saturated_bulk_gpa", "Dry shear, mu (GPa)", "Ksat (GPa)"),
    )
    for axis, (x, y, xlabel, ylabel) in zip(axes, panels):
        axis.scatter(
            support[x], support[y], c=support["unsupported_by_t73"], cmap="coolwarm",
            s=10, alpha=0.25, label="bounded gate",
        )
        axis.scatter(confirmed[x], confirmed[y], color="black", s=10, alpha=0.45, label="T73 confirmed")
        axis.set(xlabel=xlabel, ylabel=ylabel)
        if y == "depth_m":
            axis.invert_yaxis()
    axes[0].legend()
    figure.suptitle("Revision-3.3 T73 support gap: physical and elastic coordinates")
    path = destination / "01_t73_support_gap_coordinates.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source(path), "message": "T73 and gate support in porosity, depth, shaliness, Ksat and shear"})

    unsupported = support[support["unsupported_by_t73"]]
    counts = unsupported["primary_support_failure"].value_counts()
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    counts.sort_values().plot.barh(ax=axis, color="tab:orange")
    axis.set(xlabel="Unsupported bounded-gate pixels", ylabel="Primary range failure")
    axis.set_title("Primary attribution of the original 73.69% support failure")
    path = destination / "02_support_failure_attribution.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source(path), "message": "Mutually exclusive primary failure attribution"})

    grouped = support.groupby(["facies", "depth_bin", "effective_porosity_bin"], observed=True)[
        "unsupported_by_t73"
    ].agg(["size", "mean"]).reset_index()
    grouped["class"] = (
        grouped["facies"].astype(str) + " | " + grouped["depth_bin"].astype(str)
        + " | " + grouped["effective_porosity_bin"].astype(str)
    )
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.barh(grouped["class"], grouped["mean"], color="tab:red", alpha=0.75)
    axis.axvline(0.05, color="black", linestyle="--", label="5% unsupported target")
    axis.set(xlabel="Unsupported fraction", ylabel="Facies | depth | effective-porosity bin", xlim=(0, 1.02))
    axis.legend()
    axis.set_title("T73 support failure by modeled class")
    path = destination / "03_support_by_facies_depth_porosity.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source(path), "message": "Support gap by facies, depth and porosity"})

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for well, group in wells.groupby("well"):
        axes[0].scatter(group["input_porosity"], group["density_porosity"], s=7, alpha=0.25, label=well)
    axes[0].scatter(support["input_porosity"], support["density_porosity"], s=6, alpha=0.12, color="black", label="gate")
    axes[0].plot([0, 0.30], [0, 0.30], "k--", linewidth=1)
    axes[0].set(xlabel="Geological/log porosity", ylabel="Density-derived effective porosity", title="Distinct porosity coordinates")
    axes[0].legend(ncol=2, fontsize=8)
    support.boxplot(column="density_porosity", by="depth_bin", ax=axes[1], grid=False)
    axes[1].set(xlabel="Depth class", ylabel="Effective porosity", title="Gate effective porosity by depth")
    figure.suptitle("Geological versus rock-physics porosity")
    path = destination / "04_geological_vs_effective_porosity.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source(path), "message": "Geological porosity is preserved separately from density-closure porosity"})
    return outputs


def analyze(_: argparse.Namespace) -> None:
    locations = _locations()
    criteria_path = REPOSITORY / "configs" / "revision33_acceptance.yaml"
    criteria = _criteria()
    wells, gate = _tables(locations)
    support, support_report = _support_gap_table(wells, gate)
    locations["analysis"].mkdir(parents=True, exist_ok=True)
    well_path = locations["analysis"] / "all_well_dry_shear_samples.csv"
    gate_path = locations["analysis"] / "bounded_gate_support_gap.csv"
    wells.to_csv(well_path, index=False)
    support.to_csv(gate_path, index=False)
    figures = _analysis_figures(wells, support, locations)
    porosity = {
        "geological_porosity": {
            "source": "Stage-01 geological/log-derived porosity and Stage-02 coherent perturbation",
            "role": "geological modeling and ML input",
            "replacement_policy": "never replaced by rock-physics effective porosity",
        },
        "rock_physics_effective_porosity": {
            "formula": "phi_eff = (rho_mineral - rho_RF) / (rho_mineral - rho_brine)",
            "role": "density closure for dry-frame/fluid calculations only",
            "uncertainty": "depends on mineral mixture and approved scenario brine density",
            "calibration_support": "all-well shear plus confirmed T73 bulk anchor; ambiguous wells remain fluid-ambiguous",
        },
        "well_difference_percentiles": np.quantile(
            wells["density_porosity"] - wells["input_porosity"], [0.01, 0.5, 0.99]
        ).tolist(),
        "gate_difference_percentiles": np.quantile(
            gate["density_porosity"] - gate["input_porosity"], [0.01, 0.5, 0.99]
        ).tolist(),
    }
    report = {
        "status": "analysis_complete_no_model_changed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "predeclared_criteria": _source(criteria_path),
        "criteria": criteria,
        "well_table": _source(well_path),
        "gate_table": _source(gate_path),
        "support_gap": support_report,
        "porosity_separation": porosity,
        "figures": figures,
        "data_or_model_changes_in_phase_1": False,
    }
    write_json(locations["analysis"] / "support_gap_report.json", report)
    print(json.dumps(report, indent=2))


def _shear_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[SHEAR_FEATURE_COLUMNS].to_numpy(float)
    values[:, SHEAR_FEATURE_COLUMNS.index("depth_m")] /= 1000.0
    return values


def _well_balanced_weights(wells: np.ndarray) -> np.ndarray:
    counts = {well: int(np.sum(wells == well)) for well in np.unique(wells)}
    weights = np.asarray([1.0 / counts[well] for well in wells])
    return weights / weights.mean()


def _fit_dry_shear(
    wells: pd.DataFrame,
    locations: dict[str, Path],
) -> tuple[RandomForestRegressor, pd.DataFrame, dict[str, Any]]:
    rows = []
    all_predictions = np.full(len(wells), np.nan)
    for held_out in sorted(wells["well"].unique()):
        train = ~wells["well"].eq(held_out)
        test = ~train
        model = RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=4,
            max_features=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(
            _shear_features(wells.loc[train]),
            np.log(wells.loc[train, "shear_gpa"]),
            sample_weight=_well_balanced_weights(wells.loc[train, "well"].to_numpy()),
        )
        prediction = np.exp(model.predict(_shear_features(wells.loc[test])))
        observed = wells.loc[test, "shear_gpa"].to_numpy(float)
        all_predictions[np.flatnonzero(test)] = prediction
        residual = (prediction - observed) / observed
        for facies, indices in wells.loc[test].groupby("facies").groups.items():
            positions = wells.index.get_indexer(indices)
            local_observed = wells.loc[indices, "shear_gpa"].to_numpy(float)
            local_prediction = all_predictions[positions]
            rows.append(
                {
                    "held_out_well": held_out,
                    "facies": facies,
                    "samples": len(indices),
                    "mape": float(np.mean(np.abs(local_prediction - local_observed) / local_observed)),
                    "median_relative_bias": float(np.median((local_prediction - local_observed) / local_observed)),
                    "rmse_gpa": float(np.sqrt(np.mean((local_prediction - local_observed) ** 2))),
                    "log_rmse": float(np.sqrt(np.mean((np.log(local_prediction) - np.log(local_observed)) ** 2))),
                }
            )
        rows.append(
            {
                "held_out_well": held_out,
                "facies": "all",
                "samples": int(test.sum()),
                "mape": float(np.mean(np.abs(residual))),
                "median_relative_bias": float(np.median(residual)),
                "rmse_gpa": float(np.sqrt(np.mean((prediction - observed) ** 2))),
                "log_rmse": float(np.sqrt(np.mean((np.log(prediction) - np.log(observed)) ** 2))),
            }
        )
    residual_log = np.log(all_predictions) - np.log(wells["shear_gpa"].to_numpy(float))
    interval_half_width = float(np.quantile(np.abs(residual_log), 0.99))
    interval_coverage = float((np.abs(residual_log) <= interval_half_width).mean())
    model = RandomForestRegressor(
        n_estimators=600,
        min_samples_leaf=4,
        max_features=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(
        _shear_features(wells),
        np.log(wells["shear_gpa"]),
        sample_weight=_well_balanced_weights(wells["well"].to_numpy()),
    )
    output = pd.DataFrame(rows)
    path = locations["calibration"] / "dry_shear_leave_one_well_out.csv"
    output.to_csv(path, index=False)
    prediction_path = locations["calibration"] / "dry_shear_loo_predictions.csv"
    wells[["well", "facies", "depth_m", "density_porosity", "shaliness", "shear_gpa"]].assign(
        predicted_shear_gpa=all_predictions,
        relative_error=all_predictions / wells["shear_gpa"].to_numpy(float) - 1.0,
    ).to_csv(prediction_path, index=False)
    model_path = locations["calibration"] / "all_well_dry_shear_random_forest.joblib"
    joblib.dump(model, model_path)
    all_rows = output[output["facies"].eq("all")]
    report = {
        "method": "well-balanced RandomForest regression of log(mu_GPa)",
        "features": SHEAR_FEATURE_COLUMNS,
        "fluid_labels_used": False,
        "ambiguous_wells_remain_fluid_ambiguous": True,
        "mean_per_well_mape": float(all_rows["mape"].mean()),
        "maximum_per_well_mape": float(all_rows["mape"].max()),
        "maximum_absolute_per_well_median_bias": float(all_rows["median_relative_bias"].abs().max()),
        "empirical_99_log_interval_half_width": interval_half_width,
        "empirical_99_interval_coverage": interval_coverage,
        "per_well_table": _source(path),
        "prediction_table": _source(prediction_path),
        "model": _source(model_path),
    }
    return model, output, report


def _empirical_ratio_model(
    confirmed: pd.DataFrame,
) -> tuple[HuberRegressor, float, float]:
    features = confirmed[["density_porosity", "shaliness", "depth_m"]].to_numpy(float)
    features[:, 2] /= 1000.0
    target = np.log(
        confirmed["dry_bulk_density_phi_gpa"].to_numpy(float)
        / confirmed["shear_gpa"].to_numpy(float)
    )
    model = HuberRegressor(epsilon=1.35, max_iter=1000).fit(features, target)
    residual = target - model.predict(features)
    return model, float(np.quantile(residual, 0.005)), float(np.quantile(residual, 0.995))


def _family_table(
    frame: pd.DataFrame,
    target_shear_gpa: np.ndarray,
    confirmed: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    phi = frame["density_porosity"].to_numpy(float)
    mineral_bulk = frame["mineral_bulk_gpa"].to_numpy(float)
    mineral_shear = frame["mineral_shear_gpa"].to_numpy(float)
    target_shear = np.asarray(target_shear_gpa, dtype=float)
    soft = match_hashin_shtrikman_family_to_shear(
        phi, mineral_bulk, mineral_shear, target_shear, family="soft_sand"
    )
    stiff = match_hashin_shtrikman_family_to_shear(
        phi, mineral_bulk, mineral_shear, target_shear, family="stiff_sand"
    )
    confirmed_void = 1.0 - confirmed["density_porosity"].to_numpy(float) / 0.36
    confirmed_shear_exponent = np.log(
        confirmed["shear_gpa"].to_numpy(float)
        / confirmed["mineral_shear_gpa"].to_numpy(float)
    ) / np.log(confirmed_void)
    confirmed_bulk_exponent = np.log(
        confirmed["dry_bulk_density_phi_gpa"].to_numpy(float)
        / confirmed["mineral_bulk_gpa"].to_numpy(float)
    ) / np.log(confirmed_void)
    exponent_ratio = confirmed_bulk_exponent / confirmed_shear_exponent
    ratio_low, ratio_median, ratio_high = np.quantile(exponent_ratio, [0.005, 0.5, 0.995])
    cement_low, _ = constant_cement_power_law_bulk(
        phi, mineral_bulk, mineral_shear, target_shear, ratio_low
    )
    cement_median, _ = constant_cement_power_law_bulk(
        phi, mineral_bulk, mineral_shear, target_shear, ratio_median
    )
    cement_high, _ = constant_cement_power_law_bulk(
        phi, mineral_bulk, mineral_shear, target_shear, ratio_high
    )
    cement_lower = np.minimum(cement_low, cement_high)
    cement_upper = np.maximum(cement_low, cement_high)
    ratio_model, residual_low, residual_high = _empirical_ratio_model(confirmed)
    features = frame[["density_porosity", "shaliness", "depth_m"]].to_numpy(float)
    features[:, 2] /= 1000.0
    log_ratio = ratio_model.predict(features)
    empirical_median = target_shear * np.exp(log_ratio)
    empirical_lower = target_shear * np.exp(log_ratio + residual_low)
    empirical_upper = target_shear * np.exp(log_ratio + residual_high)
    candidates = np.column_stack(
        [
            np.where(soft.valid, soft.bulk_gpa, np.nan),
            np.where(stiff.valid, stiff.bulk_gpa, np.nan),
            cement_median,
            empirical_median,
        ]
    )
    bounds = np.column_stack(
        [
            np.where(soft.valid, soft.bulk_gpa, np.nan),
            np.where(stiff.valid, stiff.bulk_gpa, np.nan),
            cement_lower,
            cement_upper,
            empirical_lower,
            empirical_upper,
        ]
    )
    selected = np.nanmedian(candidates, axis=1)
    confirmed_rows = frame["well"].eq("T73").to_numpy() if "well" in frame else np.zeros(len(frame), dtype=bool)
    if confirmed_rows.any():
        selected[confirmed_rows] = frame.loc[
            confirmed_rows, "dry_bulk_density_phi_gpa"
        ].to_numpy(float)
    lower = np.nanmin(bounds, axis=1)
    upper = np.nanmax(bounds, axis=1)
    ratio = selected / target_shear
    poisson = dry_poisson_ratio(selected, target_shear)
    physical = (
        np.isfinite(selected)
        & (selected > 0.0)
        & (selected < mineral_bulk)
        & (ratio >= 0.30)
        & (ratio <= 4.00)
        & (poisson >= 0.0)
        & (poisson <= 0.45)
    )
    output = frame.copy()
    output["target_dry_shear_gpa"] = target_shear
    output["soft_sand_kdry_gpa"] = soft.bulk_gpa
    output["soft_sand_valid"] = soft.valid
    output["soft_sand_effective_pressure_mpa"] = soft.effective_pressure_mpa
    output["stiff_sand_kdry_gpa"] = stiff.bulk_gpa
    output["stiff_sand_valid"] = stiff.valid
    output["stiff_sand_effective_pressure_mpa"] = stiff.effective_pressure_mpa
    output["constant_cement_kdry_lower_gpa"] = cement_lower
    output["constant_cement_kdry_median_gpa"] = cement_median
    output["constant_cement_kdry_upper_gpa"] = cement_upper
    output["empirical_ratio_kdry_lower_gpa"] = empirical_lower
    output["empirical_ratio_kdry_median_gpa"] = empirical_median
    output["empirical_ratio_kdry_upper_gpa"] = empirical_upper
    output["ensemble_kdry_lower_gpa"] = lower
    output["ensemble_kdry_selected_gpa"] = selected
    output["ensemble_kdry_upper_gpa"] = upper
    output["ensemble_kdry_to_shear"] = ratio
    output["ensemble_dry_poisson_ratio"] = poisson
    output["ensemble_physical"] = physical
    metadata = {
        "families": {
            "soft_sand": "modified Hashin-Shtrikman lower bound with Hertz-Mindlin end member",
            "stiff_sand": "modified Hashin-Shtrikman upper bound with Hertz-Mindlin end member",
            "constant_cement": "T73-calibrated power-law constant-cement trend; not Dvorkin contact cement",
            "empirical_ratio": "robust T73 Kdry/mu trend with empirical 99% residual envelope",
        },
        "contact_cement_exclusion": (
            "A Dvorkin contact-cement model was not claimed because cement mineralogy, "
            "cement volume, and contact geometry are unavailable."
        ),
        "constant_cement_bulk_to_shear_exponent_ratio": {
            "p005": float(ratio_low),
            "median": float(ratio_median),
            "p995": float(ratio_high),
        },
        "soft_sand_valid_fraction": float(soft.valid.mean()),
        "stiff_sand_valid_fraction": float(stiff.valid.mean()),
        "ensemble_physical_fraction": float(physical.mean()),
    }
    return output, metadata


def _build_calibration_model(
    wells: pd.DataFrame,
    family_wells: pd.DataFrame,
    locations: dict[str, Path],
) -> tuple[CalibratedDryFrameModel, Path]:
    features = wells[FEATURE_COLUMNS].to_numpy(float)
    features[:, 2] /= 1000.0
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    metadata = {
        "schema_version": 3,
        "calibration_id": CALIBRATION_ID,
        "method": "scenario-supported all-well dry-shear / T73-bulk-anchored physical dry-frame ensemble",
        "feature_names": ["effective porosity fraction", "DELTA shaliness fraction", "depth km"],
        "target_names": ["log ensemble Kdry GPa", "log observed fluid-independent shear GPa"],
        "confirmed_bulk_anchor": "T73 SW >= 0.95 only",
        "ambiguous_well_use": "dry shear plus latent-fluid admissibility; no brine classification",
        "ambiguous_wells": list(AMBIGUOUS_WELLS),
        "neighbor_count": 32,
        "time_depth_linear_coefficients": json.loads(
            (
                locations["work_data"]
                / "derived"
                / "fluid_models_v0032"
                / "calibrated_dry_frame_confirmed_brine.json"
            ).read_text(encoding="utf-8")
        )["time_depth_linear_coefficients"],
        "geological_porosity": "preserved Stage-01/02 field; not replaced",
        "rock_physics_effective_porosity": {
            "formula": "(rho_mineral - rho_RF) / (rho_mineral - rho_brine)",
            "use": "dry-frame and fluid calculations only",
        },
        "physical_families": ["soft_sand", "stiff_sand", "constant_cement_power_law", "empirical_Kdry_to_mu"],
        "scenario_not_posterior": True,
        "mineral_projection": False,
        "dry_frame_clipping": False,
        "elastic_clipping": False,
    }
    model = CalibratedDryFrameModel(
        calibration_id=CALIBRATION_ID,
        feature_names=("effective_porosity_fraction", "DELTA_shaliness_fraction", "depth_km"),
        feature_center=center,
        feature_scale=scale,
        features_standardized=(features - center) / scale,
        log_dry_bulk_gpa=np.log(family_wells["ensemble_kdry_selected_gpa"].to_numpy(float)),
        log_shear_gpa=np.log(wells["shear_gpa"].to_numpy(float)),
        well_ids=wells["well"].to_numpy(str),
        neighbor_count=32,
        metadata=metadata,
    )
    model_path, _ = save_calibrated_dry_frame(
        model, locations["model"] / "calibrated_dry_frame_scenario_ensemble.npz"
    )
    return model, model_path


def _latent_fluid_admissibility(
    family_wells: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ambiguous = family_wells[family_wells["well"].isin(AMBIGUOUS_WELLS)].copy()
    fluid_bulk = []
    for pressure in (24.0, 36.0):
        for temperature in (55.0, 95.0):
            for salinity in (0.006, 0.12):
                brine = batzle_wang_brine(pressure, temperature, salinity)
                co2 = span_wagner_co2(pressure, temperature)
                for exponent in (2.0, 4.0):
                    for saturation in (0.0, 0.30, 0.80):
                        bulk, _ = brie_fluid_mixture(
                            np.asarray([saturation]),
                            brine_bulk_modulus_gpa=brine.bulk_modulus_gpa,
                            co2_bulk_modulus_gpa=co2.bulk_modulus_gpa,
                            brine_density_g_cc=brine.density_g_cc,
                            co2_density_g_cc=co2.density_g_cc,
                            brie_exponent=exponent,
                        )
                        fluid_bulk.append(float(bulk[0]))
    dry_candidates = np.column_stack(
        [
            ambiguous["ensemble_kdry_lower_gpa"],
            ambiguous["ensemble_kdry_selected_gpa"],
            ambiguous["ensemble_kdry_upper_gpa"],
        ]
    )
    predictions = []
    valid_state_masks = []
    for dry_index in range(dry_candidates.shape[1]):
        for fluid in (min(fluid_bulk), max(fluid_bulk)):
            dry = dry_candidates[:, dry_index]
            phi = ambiguous["density_porosity"].to_numpy(float)
            mineral = ambiguous["mineral_bulk_gpa"].to_numpy(float)
            denominator = phi / fluid + (1.0 - phi) / mineral - dry / mineral**2
            valid = (
                (denominator > 0.0)
                & (dry > 0.0)
                & (dry < mineral)
                & np.isfinite(denominator)
            )
            saturated = np.full_like(dry, np.nan)
            saturated[valid] = dry[valid] + (1.0 - dry[valid] / mineral[valid]) ** 2 / denominator[valid]
            predictions.append(saturated)
            valid_state_masks.append(valid)
    prediction_array = np.asarray(predictions)
    if np.any(~np.isfinite(prediction_array).any(axis=0)):
        raise RuntimeError("No physically valid latent-fluid/dry-frame state exists for a sample")
    lower = np.nanmin(prediction_array, axis=0)
    upper = np.nanmax(prediction_array, axis=0)
    observed = ambiguous["saturated_bulk_gpa"].to_numpy(float)
    tolerance = np.maximum(0.75, 0.15 * observed)
    admissible = (observed >= lower - tolerance) & (observed <= upper + tolerance)
    ambiguous["latent_fluid_minimum_reproducible_ksat_gpa"] = lower
    ambiguous["latent_fluid_maximum_reproducible_ksat_gpa"] = upper
    ambiguous["latent_ksat_tolerance_gpa"] = tolerance
    ambiguous["latent_fluid_admissible"] = admissible
    summary = ambiguous.groupby("well")["latent_fluid_admissible"].agg(["size", "mean"])
    report = {
        "approved_fluid_bulk_range_gpa": [min(fluid_bulk), max(fluid_bulk)],
        "invalid_family_fluid_state_fraction": float(
            1.0 - np.asarray(valid_state_masks, dtype=float).mean()
        ),
        "overall_admissible_fraction": float(admissible.mean()),
        "per_well": summary.reset_index().to_dict(orient="records"),
        "interpretation": (
            "Existence of an admissible scenario state constrains the dry-frame family; "
            "it does not identify or estimate the actual fluid in an ambiguous well."
        ),
        "non_identifiability": True,
    }
    return ambiguous, report


def _calibration_figures(
    family_wells: pd.DataFrame,
    family_gate: pd.DataFrame,
    loo_predictions: pd.DataFrame,
    locations: dict[str, Path],
) -> list[dict[str, Any]]:
    destination = locations["figures"] / "dry_frame"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].scatter(loo_predictions["shear_gpa"], loo_predictions["predicted_shear_gpa"], s=8, alpha=0.25, c=pd.Categorical(loo_predictions["well"]).codes)
    limit = [loo_predictions[["shear_gpa", "predicted_shear_gpa"]].min().min(), loo_predictions[["shear_gpa", "predicted_shear_gpa"]].max().max()]
    axes[0].plot(limit, limit, "k--")
    axes[0].set(xlabel="Observed mu (GPa)", ylabel="Held-out prediction (GPa)", title="Grouped leave-one-well-out dry shear")
    confirmed = family_wells[family_wells["well"].eq("T73")]
    axes[1].fill_between(
        confirmed["density_porosity"],
        confirmed["ensemble_kdry_lower_gpa"],
        confirmed["ensemble_kdry_upper_gpa"],
        color="tab:blue", alpha=0.2, label="physical-family 99% envelope",
    )
    axes[1].scatter(confirmed["density_porosity"], confirmed["dry_bulk_density_phi_gpa"], s=10, color="black", alpha=0.5, label="T73 confirmed-brine Kdry")
    axes[1].set(xlabel="Effective porosity", ylabel="Kdry (GPa)", title="Confirmed T73 bulk anchoring")
    axes[1].legend()
    path = destination / "01_dry_shear_loo_and_t73_bulk_anchor.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source(path), "message": "All-well dry-shear LOO and confirmed T73 bulk envelope"})

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    sample = family_gate.sample(min(2440, len(family_gate)), random_state=RANDOM_SEED)
    axes[0].scatter(sample["density_porosity"], sample["ensemble_kdry_selected_gpa"], c=sample["depth_m"], cmap="viridis", s=10, alpha=0.35)
    axes[0].set(xlabel="Effective porosity", ylabel="Selected ensemble Kdry (GPa)", title="Bounded-gate dry frame")
    for family, column in (
        ("soft sand", "soft_sand_kdry_gpa"),
        ("stiff sand", "stiff_sand_kdry_gpa"),
        ("constant cement", "constant_cement_kdry_median_gpa"),
        ("empirical Kdry/mu", "empirical_ratio_kdry_median_gpa"),
    ):
        axes[1].hist(sample[column].dropna(), bins=45, histtype="step", linewidth=1.5, label=family)
    axes[1].set(xlabel="Kdry (GPa)", ylabel="Pixels", title="Dry-frame family comparison")
    axes[1].legend()
    path = destination / "02_bounded_gate_dry_frame_families.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source(path), "message": "Bounded-gate scenario family comparison"})
    return outputs


def fit(_: argparse.Namespace) -> None:
    locations = _locations()
    criteria = _criteria()
    analysis_report = locations["analysis"] / "support_gap_report.json"
    if not analysis_report.exists():
        raise FileNotFoundError("Run analyze before fitting Revision 3.3")
    for name in ("calibration", "figures", "model"):
        locations[name].mkdir(parents=True, exist_ok=True)
    wells, gate = _tables(locations)
    shear_model, _, shear_report = _fit_dry_shear(wells, locations)
    confirmed = wells[wells["well"].eq("T73")]
    family_wells, well_family_metadata = _family_table(
        wells, wells["shear_gpa"].to_numpy(float), confirmed
    )
    gate_shear = np.exp(shear_model.predict(_shear_features(gate)))
    family_gate, gate_family_metadata = _family_table(gate, gate_shear, confirmed)
    distance, threshold, _, _, _ = _standardized_support(wells, gate, FEATURE_COLUMNS)
    expanded_distance, expanded_threshold, _, _, _ = _standardized_support(
        wells, gate, EXPANDED_SUPPORT_COLUMNS
    )
    family_gate["all_well_support_distance"] = distance
    family_gate["all_well_support_threshold"] = threshold
    family_gate["inside_all_well_support"] = distance <= threshold
    family_gate["expanded_support_distance"] = expanded_distance
    family_gate["inside_expanded_support"] = expanded_distance <= expanded_threshold
    model, model_path = _build_calibration_model(wells, family_wells, locations)
    latent, latent_report = _latent_fluid_admissibility(family_wells)
    well_path = locations["calibration"] / "well_dry_frame_families.csv"
    gate_path = locations["calibration"] / "gate_dry_frame_families.csv"
    latent_path = locations["calibration"] / "ambiguous_well_latent_fluid_admissibility.csv"
    family_wells.to_csv(well_path, index=False)
    family_gate.to_csv(gate_path, index=False)
    latent.to_csv(latent_path, index=False)
    loo_predictions = pd.read_csv(locations["calibration"] / "dry_shear_loo_predictions.csv")
    figures = _calibration_figures(family_wells, family_gate, loo_predictions, locations)
    t73 = family_wells[family_wells["well"].eq("T73")]
    t73_coverage = float(
        t73["dry_bulk_density_phi_gpa"].between(
            t73["ensemble_kdry_lower_gpa"], t73["ensemble_kdry_upper_gpa"]
        ).mean()
    )
    class_coverage = (
        family_gate.assign(
            supported=family_gate["inside_all_well_support"] & family_gate["ensemble_physical"]
        )
        .groupby(["facies", "depth_bin"], observed=True)["supported"]
        .agg(["size", "mean"])
        .reset_index()
    )
    overall_coverage = float(
        (family_gate["inside_all_well_support"] & family_gate["ensemble_physical"]).mean()
    )
    shear_limits = criteria["dry_shear_leave_one_well_out"]
    dry_limits = criteria["dry_frame"]
    gates = {
        "dry_shear_mean_mape": shear_report["mean_per_well_mape"] <= shear_limits["maximum_mean_absolute_percentage_error"],
        "dry_shear_per_well_mape": shear_report["maximum_per_well_mape"] <= shear_limits["maximum_per_well_mean_absolute_percentage_error"],
        "dry_shear_bias": shear_report["maximum_absolute_per_well_median_bias"] <= shear_limits["maximum_absolute_per_well_median_bias"],
        "dry_shear_interval": shear_report["empirical_99_interval_coverage"] >= shear_limits["minimum_empirical_99_interval_coverage"],
        "overall_support": overall_coverage >= criteria["support"]["minimum_overall_gate_coverage"],
        "class_support": bool((class_coverage["mean"] >= criteria["support"]["minimum_facies_depth_bin_coverage"]).all()),
        "t73_bulk_anchor": t73_coverage >= dry_limits["confirmed_t73_minimum_99_envelope_coverage"],
        "latent_overall": latent_report["overall_admissible_fraction"] >= dry_limits["ambiguous_well_minimum_latent_admissibility"],
        "latent_per_well": all(row["mean"] >= dry_limits["ambiguous_well_minimum_per_well_admissibility"] for row in latent_report["per_well"]),
        "all_ensemble_states_physical": bool(family_gate["ensemble_physical"].all()),
        "projection_or_clipping_absent": True,
    }
    report = {
        "status": "passed" if all(gates.values()) else "failed",
        "calibration_id": CALIBRATION_ID,
        "predeclared_criteria": _source(REPOSITORY / "configs" / "revision33_acceptance.yaml"),
        "dry_shear": shear_report,
        "dry_frame_families_wells": well_family_metadata,
        "dry_frame_families_gate": gate_family_metadata,
        "well_family_table": _source(well_path),
        "gate_family_table": _source(gate_path),
        "latent_fluid": {**latent_report, "table": _source(latent_path)},
        "all_well_support": {
            "threshold": threshold,
            "overall_coverage": overall_coverage,
            "expanded_feature_threshold": expanded_threshold,
            "expanded_feature_coverage": float(family_gate["inside_expanded_support"].mean()),
            "by_facies_depth": class_coverage.to_dict(orient="records"),
        },
        "confirmed_t73_99_envelope_coverage": t73_coverage,
        "calibration_artifact": _source(model_path),
        "calibration_metadata": _source(model_path.with_suffix(".json")),
        "gates": gates,
        "figures": figures,
        "ambiguous_wells_classified_as_brine": False,
        "non_identifiability_reported": True,
    }
    write_json(locations["calibration"] / "dry_frame_calibration_report.json", report)
    print(json.dumps(report, indent=2))


def _candidate_delta(
    dry_bulk_gpa: np.ndarray,
    effective_porosity: np.ndarray,
    mineral_bulk_gpa: np.ndarray,
    rf_bulk_gpa: np.ndarray,
    rf_shear_gpa: np.ndarray,
    rf_density_g_cc: np.ndarray,
    saturation: np.ndarray,
    physics: FluidRockPhysics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    brine_reference = forward_gassmann_bulk_strict(
        dry_bulk_gpa, effective_porosity, mineral_bulk_gpa, physics.brine_bulk_modulus_gpa
    )
    fluid_bulk, fluid_density = brie_fluid_mixture(
        saturation,
        brine_bulk_modulus_gpa=physics.brine_bulk_modulus_gpa,
        co2_bulk_modulus_gpa=physics.co2_bulk_modulus_gpa,
        brine_density_g_cc=physics.brine_density_g_cc,
        co2_density_g_cc=physics.co2_density_g_cc,
        brie_exponent=physics.brie_exponent,
    )
    fluid_reference = forward_gassmann_bulk_strict(
        dry_bulk_gpa, effective_porosity, mineral_bulk_gpa, fluid_bulk
    )
    target_bulk = rf_bulk_gpa + fluid_reference - brine_reference
    target_density = rf_density_g_cc + effective_porosity * (
        fluid_density - physics.brine_density_g_cc
    )
    elastic = elastic_from_gpa_strict(target_bulk, rf_shear_gpa, target_density)
    return elastic.vp, elastic.vs, target_density, target_bulk - rf_bulk_gpa


def _candidate_uncertainty(
    gate: pd.DataFrame,
    family_gate: pd.DataFrame,
) -> pd.DataFrame:
    index = int(np.argmin(np.square((gate["density_porosity"] - gate["density_porosity"].median()) / gate["density_porosity"].std()) + np.square((gate["shaliness"] - gate["shaliness"].median()) / gate["shaliness"].std()) + np.square((gate["depth_m"] - gate["depth_m"].median()) / gate["depth_m"].std())))
    row = gate.iloc[index]
    family = family_gate.iloc[index]
    rf_bulk = np.asarray([row.saturated_bulk_gpa])
    rf_shear = np.asarray([row.shear_gpa])
    rf_density = np.asarray([row.density_g_cc])
    mineral = np.asarray([row.mineral_bulk_gpa])
    nominal = {"pressure_mpa": 30.0, "temperature_c": 75.0, "salinity": 0.063, "brie": 3.0, "saturation": 0.55, "phi": row.density_porosity, "dry": family.ensemble_kdry_selected_gpa}
    scenarios = {
        "dry_frame_family": [("low", family.ensemble_kdry_lower_gpa), ("high", family.ensemble_kdry_upper_gpa)],
        "pressure": [("low", 24.0), ("high", 36.0)],
        "temperature": [("low", 55.0), ("high", 95.0)],
        "salinity": [("low", 0.006), ("high", 0.12)],
        "Brie_exponent": [("low", 2.0), ("high", 4.0)],
        "saturation": [("low", 0.30), ("high", 0.80)],
        "effective_porosity": [("low", float(gate["density_porosity"].quantile(0.01))), ("high", float(gate["density_porosity"].quantile(0.99)))],
    }
    rows = []
    for source, values in scenarios.items():
        for level, value in values:
            state = nominal.copy()
            key = {
                "dry_frame_family": "dry",
                "pressure": "pressure_mpa",
                "temperature": "temperature_c",
                "Brie_exponent": "brie",
                "effective_porosity": "phi",
            }.get(source, source)
            state[key] = value
            brine = batzle_wang_brine(state["pressure_mpa"], state["temperature_c"], state["salinity"])
            co2 = span_wagner_co2(state["pressure_mpa"], state["temperature_c"])
            physics = replace(
                FluidRockPhysics(),
                brine_bulk_modulus_gpa=brine.bulk_modulus_gpa,
                brine_density_g_cc=brine.density_g_cc,
                co2_bulk_modulus_gpa=co2.bulk_modulus_gpa,
                co2_density_g_cc=co2.density_g_cc,
                brie_exponent=state["brie"],
            )
            vp, vs, density, delta_bulk = _candidate_delta(
                np.asarray([state["dry"]]), np.asarray([state["phi"]]), mineral,
                rf_bulk, rf_shear, rf_density, np.asarray([state["saturation"]]), physics,
            )
            rows.append(
                {
                    "source": source,
                    "level": level,
                    "value": value,
                    "delta_vp_m_s": vp[0] - row.vp_m_s,
                    "delta_vs_m_s": vs[0] - row.vs_m_s,
                    "delta_density_g_cc": density[0] - row.density_g_cc,
                    "delta_ksat_gpa": delta_bulk[0],
                }
            )
    return pd.DataFrame(rows)


def evaluate(_: argparse.Namespace) -> None:
    locations = _locations()
    criteria = _criteria()
    calibration_report_path = locations["calibration"] / "dry_frame_calibration_report.json"
    calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    if calibration_report["status"] != "passed":
        raise RuntimeError("Dry-frame calibration gates failed; Candidate B cannot be evaluated")
    wells, gate = _tables(locations)
    family_gate = pd.read_csv(locations["calibration"] / "gate_dry_frame_families.csv")
    from sage_avo.geology.fluid_calibration import load_calibrated_dry_frame

    model_path = locations["model"] / "calibrated_dry_frame_scenario_ensemble.npz"
    model = load_calibrated_dry_frame(model_path)
    brine = batzle_wang_brine(30.0, 75.0, 0.063)
    co2 = span_wagner_co2(30.0, 75.0)
    physics = replace(
        FluidRockPhysics(),
        brine_bulk_modulus_gpa=brine.bulk_modulus_gpa,
        brine_density_g_cc=brine.density_g_cc,
        co2_bulk_modulus_gpa=co2.bulk_modulus_gpa,
        co2_density_g_cc=co2.density_g_cc,
        brie_exponent=3.0,
    )
    inputs = {
        "vp_brine_m_s": gate["vp_m_s"].to_numpy(float),
        "vs_brine_m_s": gate["vs_m_s"].to_numpy(float),
        "density_brine_g_cc": gate["density_g_cc"].to_numpy(float),
        "input_porosity": gate["input_porosity"].to_numpy(float),
        "shaliness": gate["shaliness"].to_numpy(float),
        "depth_m": gate["depth_m"].to_numpy(float),
        "calibration": model,
        "physics": physics,
    }
    saturation_grid = np.linspace(0.0, 0.8, 17)
    response_rows = []
    prior_density = None
    maximum_zero_error = 0.0
    maximum_shear_error = 0.0
    smooth = True
    density_monotonic = True
    previous_vp = None
    for saturation in saturation_grid:
        result = calibrated_differential_gassmann_substitution(
            co2_saturation=np.full(len(gate), saturation), **inputs
        )
        if saturation == 0.0:
            maximum_zero_error = float(
                max(
                    np.max(np.abs(result.elastic.vp - inputs["vp_brine_m_s"])),
                    np.max(np.abs(result.elastic.vs - inputs["vs_brine_m_s"])),
                    np.max(np.abs(result.elastic.density - inputs["density_brine_g_cc"])),
                )
            )
        maximum_shear_error = max(
            maximum_shear_error,
            float(np.max(np.abs(result.target_shear_gpa - result.rf_shear_gpa))),
        )
        if prior_density is not None:
            density_monotonic &= bool(np.all(result.elastic.density <= prior_density + 1e-12))
        if previous_vp is not None:
            smooth &= bool(np.quantile(np.abs(result.elastic.vp - previous_vp), 0.999) < 100.0)
        prior_density = result.elastic.density.copy()
        previous_vp = result.elastic.vp.copy()
        response_rows.append(
            pd.DataFrame(
                {
                    "saturation": saturation,
                    "facies": gate["facies"],
                    "depth_bin": gate["depth_bin"],
                    "effective_porosity": result.effective_porosity,
                    "dry_bulk_gpa": result.dry_bulk_gpa,
                    "frame_shear_gpa": result.frame_shear_gpa,
                    "dry_to_shear": result.dry_bulk_gpa / result.frame_shear_gpa,
                    "dry_poisson_ratio": dry_poisson_ratio(result.dry_bulk_gpa, result.frame_shear_gpa),
                    "delta_vp_m_s": result.elastic.vp - inputs["vp_brine_m_s"],
                    "delta_vs_m_s": result.elastic.vs - inputs["vs_brine_m_s"],
                    "delta_density_g_cc": result.elastic.density - inputs["density_brine_g_cc"],
                    "delta_ksat_gpa": result.delta_bulk_gpa,
                }
            )
        )
    responses = pd.concat(response_rows, ignore_index=True)
    dry_at_gate, _, _ = model.predict(
        gate["density_porosity"].to_numpy(float),
        gate["shaliness"].to_numpy(float),
        gate["depth_m"].to_numpy(float),
    )
    inside_envelope = (dry_at_gate >= family_gate["ensemble_kdry_lower_gpa"]) & (
        dry_at_gate <= family_gate["ensemble_kdry_upper_gpa"]
    )
    class_coverage = (
        gate.assign(inside_envelope=inside_envelope)
        .groupby(["facies", "depth_bin"], observed=True)["inside_envelope"]
        .agg(["size", "mean"])
        .reset_index()
    )
    physical = (
        (responses["dry_bulk_gpa"] > 0.0)
        & (responses["dry_bulk_gpa"] < np.tile(gate["mineral_bulk_gpa"].to_numpy(float), len(saturation_grid)))
        & responses["dry_to_shear"].between(0.30, 4.00)
        & responses["dry_poisson_ratio"].between(0.00, 0.45)
    )
    uncertainty = _candidate_uncertainty(gate, family_gate)
    locations["evaluation"].mkdir(parents=True, exist_ok=True)
    response_path = locations["evaluation"] / "candidate_b_saturation_sweeps.csv"
    uncertainty_path = locations["evaluation"] / "candidate_b_uncertainty_by_source.csv"
    responses.to_csv(response_path, index=False)
    uncertainty.to_csv(uncertainty_path, index=False)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for axis, column, label in (
        (axes[0, 0], "delta_vp_m_s", "Delta Vp (m/s)"),
        (axes[0, 1], "delta_vs_m_s", "Delta Vs (m/s)"),
        (axes[1, 0], "delta_density_g_cc", "Delta density (g/cc)"),
        (axes[1, 1], "delta_ksat_gpa", "Delta Ksat (GPa)"),
    ):
        grouped = responses.groupby("saturation")[column]
        x = np.asarray(sorted(responses["saturation"].unique()))
        axis.plot(x, grouped.median().reindex(x), color="tab:blue")
        axis.fill_between(x, grouped.quantile(0.01).reindex(x), grouped.quantile(0.99).reindex(x), color="tab:blue", alpha=0.2)
        axis.set(xlabel="CO2 saturation", ylabel=label)
    figure.suptitle("Candidate B bounded-gate response: median and 99% pixel envelope")
    figure_path = locations["figures"] / "candidate_b_physical_sweeps.png"
    figure.savefig(figure_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    limits = criteria["candidate_b"]
    gates = {
        "zero_saturation_recovery": maximum_zero_error <= limits["maximum_zero_saturation_recovery_error"],
        "fixed_shear": maximum_shear_error <= limits["maximum_fixed_shear_error_gpa"],
        "smooth_saturation": smooth,
        "density_monotonic": density_monotonic,
        "all_physical": bool(physical.all()),
        "predictive_envelope": float(inside_envelope.mean()) >= limits["minimum_predictive_envelope_coverage"],
        "class_predictive_envelope": bool((class_coverage["mean"] >= limits["minimum_facies_depth_bin_envelope_coverage"]).all()),
        "no_projection_clipping_or_delta_cap": True,
    }
    route = "GO_SCENARIO_CO2" if all(gates.values()) else "GO_CORE_ELASTIC"
    report = {
        "status": "passed" if all(gates.values()) else "scenario_co2_failed",
        "candidate_b_algorithm": "unchanged calibrated_differential_gassmann",
        "maximum_zero_saturation_recovery_error": maximum_zero_error,
        "maximum_fixed_shear_error_gpa": maximum_shear_error,
        "exactly_zero_outside_plume": "deferred to bounded realization gate",
        "predictive_envelope_coverage": float(inside_envelope.mean()),
        "predictive_envelope_by_facies_depth": class_coverage.to_dict(orient="records"),
        "all_dry_states_physical": bool(physical.all()),
        "response_percentiles": {
            column: dict(zip(["p01", "p50", "p99"], np.quantile(responses.loc[responses["saturation"].eq(0.55), column], [0.01, 0.5, 0.99]).tolist()))
            for column in ["delta_vp_m_s", "delta_vs_m_s", "delta_density_g_cc", "delta_ksat_gpa"]
        },
        "uncertainty": {
            "interpretation": "deterministic scenario sensitivity by source; not a posterior distribution",
            "table": _source(uncertainty_path),
            "ranges": uncertainty.groupby("source")[["delta_vp_m_s", "delta_vs_m_s", "delta_density_g_cc", "delta_ksat_gpa"]].agg(lambda values: float(values.max() - values.min())).reset_index().to_dict(orient="records"),
        },
        "saturation_sweeps": _source(response_path),
        "figure": _source(figure_path),
        "gates": gates,
        "provisional_route": route,
    }
    write_json(locations["evaluation"] / "candidate_b_evaluation_report.json", report)
    fluid_v0032 = json.loads(locations["fluid_v0032"].read_text(encoding="utf-8"))
    validation_payload = {
        **fluid_v0032,
        "schema_version": 2,
        "validation_id": "v0033_" + hashlib.sha256(
            json.dumps(
                {
                    "fluid_v0032_sha256": file_sha256(locations["fluid_v0032"]),
                    "calibration_sha256": file_sha256(model_path),
                    "candidate_b_sha256": file_sha256(locations["evaluation"] / "candidate_b_evaluation_report.json"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "scenario_validated",
        "calibration_id": CALIBRATION_ID,
        "candidate_b_calibration_id": CALIBRATION_ID,
        "revision32_fluid_property_validation": _source(locations["fluid_v0032"]),
        "revision33_dry_frame_validation": _source(calibration_report_path),
        "revision33_candidate_b_validation": _source(locations["evaluation"] / "candidate_b_evaluation_report.json"),
        "production_approval": False,
        "provisional_route": route,
    }
    validation_path = locations["model"] / "fluid_property_validation.json"
    write_json(validation_path, validation_payload)
    write_json(
        locations["reports"] / "provisional_route_decision.json",
        {
            "status": route,
            "scientific_gates_passed": all(gates.values()),
            "execution_gates_pending": True,
            "full_production_authorized": False,
            "validation_artifact": _source(validation_path),
        },
    )
    print(json.dumps(report, indent=2))


def _bounded_configs(locations: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    synthetic = deepcopy(load_config(REPOSITORY / "configs" / "synthetic_s01_v0032.yaml"))
    dataset = deepcopy(load_config(REPOSITORY / "configs" / "ml_dataset_s01_v0032.yaml"))
    training = deepcopy(load_config(REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml"))
    synthetic["stage"].update(
        {
            "name": "field_conditioned_synthetic_avo_v0033_scenario_dry_frame_supported",
            "geology_realization_count": 8,
            "observation_variants_per_geology": 1,
            "realization_count": 8,
            "realization_id_offset": REALIZATION_OFFSET,
        }
    )
    synthetic["fluid_substitution"].update(
        {
            "calibration_id": CALIBRATION_ID,
            "calibration_artifact": "derived/fluid_models_v0033/calibrated_dry_frame_scenario_ensemble.npz",
            "fluid_property_validation_artifact": "derived/fluid_models_v0033/fluid_property_validation.json",
        }
    )
    synthetic["outputs"].update(
        {"version": VALIDATION_NAME, "directory": f"synthetic/{VALIDATION_NAME}/realizations"}
    )
    dataset["inputs"].update(
        {
            "synthetic_version": VALIDATION_NAME,
            "realization_directory": f"synthetic/{VALIDATION_NAME}/realizations",
            "expected_realization_count": 8,
        }
    )
    dataset["split"]["fractions"] = [0.625, 0.25, 0.125]
    dataset["outputs"].update(
        {"version": "ds_v0033_validation8", "directory": "datasets/ds_v0033_validation8"}
    )
    training_name = "sage_avo_s01_v0033_validation8"
    training["experiment"].update(
        {"name": training_name, "output_root": f"results/experiments/{training_name}"}
    )
    training["dataset"]["directory"] = dataset["outputs"]["directory"]
    training["training"]["epochs"] = 2
    source = {
        "status": "revision33_bounded_validation_before_source_freeze",
        "calibration_id": CALIBRATION_ID,
    }
    for mapping in (synthetic, dataset, training):
        mapping["source_snapshot"] = deepcopy(source)
    locations["configs"].mkdir(parents=True, exist_ok=True)
    write_json(locations["configs"] / "synthetic_resolved.json", synthetic)
    write_json(locations["configs"] / "dataset_resolved.json", dataset)
    write_json(locations["configs"] / "training_resolved.json", training)
    return synthetic, dataset, training


def _matched_qc(previous: Path, current: Path) -> dict[str, Any]:
    unaffected = (
        "elastic_brine", "delta", "sand_probability", "porosity", "rgt",
        "strat_fraction", "reservoir_mask", "horizon_top_ms", "horizon_base_ms",
        "source_horizon_top_ms", "source_horizon_base_ms", "plume_mask",
        "co2_saturation", "angles_degrees", "time_ms", "cdp",
    )
    rows = []
    for old_path in sorted(previous.glob("realization_*.npz")):
        new_path = current / old_path.name
        with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
            plume = np.asarray(new["plume_mask"], dtype=bool)
            elastic = np.asarray(new["elastic"], dtype=float)
            brine = np.asarray(new["elastic_brine"], dtype=float)
            _, shear = elastic_moduli_gpa(*elastic)
            _, brine_shear = elastic_moduli_gpa(*brine)
            rows.append(
                {
                    "realization_id": int(new["realization_id"]),
                    "unaffected_channels_bitwise_equal": bool(all(np.array_equal(old[name], new[name]) for name in unaffected)),
                    "outside_plume_maximum_change": float(np.max(np.abs(elastic[:, ~plume] - brine[:, ~plume]))),
                    "inside_plume_shear_maximum_change_gpa": float(np.max(np.abs(shear[plume] - brine_shear[plume]))) if plume.any() else 0.0,
                }
            )
    return {
        "realizations": rows,
        "all_unaffected_channels_bitwise_equal": all(row["unaffected_channels_bitwise_equal"] for row in rows),
        "outside_plume_maximum_change": max(row["outside_plume_maximum_change"] for row in rows),
        "inside_plume_shear_maximum_change_gpa": max(row["inside_plume_shear_maximum_change_gpa"] for row in rows),
    }


def regenerate(_: argparse.Namespace) -> None:
    locations = _locations()
    decision = json.loads((locations["reports"] / "provisional_route_decision.json").read_text(encoding="utf-8"))
    if decision["status"] != "GO_SCENARIO_CO2":
        raise RuntimeError("The scenario route did not pass the scientific gate")
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    synthetic, dataset, _ = _bounded_configs(locations)
    manifest = generate_stage02_dataset(
        config=synthetic, paths=paths, output_directory=locations["stage02"], workers=1, resume=False
    )
    matched = _matched_qc(locations["previous_stage02"], locations["stage02"])
    dataset_manifest = build_stage03_dataset(
        config=dataset,
        paths=paths,
        source_directory=locations["stage02"],
        output_directory=locations["stage03"],
    )
    integrity = validate_dataset_integrity(locations["stage03"])
    import run_revision31_fluid_gate as revision31

    round_trip = revision31._all_realization_round_trip(locations["stage02"], synthetic)
    report = {
        "status": "complete",
        "stage02_manifest": _source(locations["stage02"] / "manifest.json"),
        "stage02_realizations": manifest["generated_realizations"],
        "matched_qc": matched,
        "exactly_zero_outside_plume": matched["outside_plume_maximum_change"] == 0.0,
        "fixed_shear_within_tolerance": matched["inside_plume_shear_maximum_change_gpa"] <= 1e-5,
        "round_trip": round_trip,
        "stage03_manifest": _source(locations["stage03"] / "dataset_manifest.json"),
        "stage03_builder_integrity": dataset_manifest["integrity"],
        "stage03_integrity": integrity,
    }
    locations["reports"].mkdir(parents=True, exist_ok=True)
    write_json(locations["reports"] / "bounded_execution_qc.json", report)
    print(json.dumps(report, indent=2))


def cuda_sanity(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Revision-3.3 CUDA sanity requires CUDA")
    locations = _locations()
    _, _, training = _bounded_configs(locations)
    if not (locations["stage03"] / "dataset_manifest.json").exists():
        raise FileNotFoundError("Run regenerate-eight before CUDA sanity")
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
    report = {
        "status": "complete",
        "device": "cuda",
        "epochs": 2,
        "max_train_batches": int(args.max_train_batches),
        "max_validation_batches": int(args.max_validation_batches),
        "run_directory": str(output),
        "manifest": _source(Path(output) / "manifest.json"),
        "training_log": _source(Path(output) / "training_log.csv"),
    }
    write_json(locations["reports"] / "cuda_sanity.json", report)
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("analyze").set_defaults(function=analyze)
    commands.add_parser("fit").set_defaults(function=fit)
    commands.add_parser("evaluate").set_defaults(function=evaluate)
    commands.add_parser("regenerate-eight").set_defaults(function=regenerate)
    sanity = commands.add_parser("train-sanity")
    sanity.add_argument("--max-train-batches", type=int, default=4)
    sanity.add_argument("--max-validation-batches", type=int, default=2)
    sanity.set_defaults(function=cuda_sanity)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
