#!/usr/bin/env python3
"""Run the bounded Revision-3.2 fluid-provenance and scenario gate.

This driver cannot launch a 100-realization corpus or production training.  It
audits the available S01 evidence, builds the confirmed-brine comparison,
evaluates P/T/salinity-aware fluids with Candidate B, and writes the reviewable
fluid-property artifact required by the bounded eight-realization gate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import CoolProp
import lasio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc, spearmanr
import torch

from sage_avo.config import load_config
from sage_avo.experiments import (
    build_stage03_dataset,
    generate_stage02_dataset,
    validate_dataset_integrity,
)
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.forward import zoeppritz_pp
from sage_avo.geology.fluid_calibration import (
    CalibratedDryFrameModel,
    FluidRockPhysics,
    brie_fluid_mixture,
    calibrated_differential_gassmann_substitution,
    elastic_from_gpa_strict,
    forward_gassmann_bulk_strict,
    poisson_ratio_from_moduli,
    save_calibrated_dry_frame,
)
from sage_avo.geology.fluid_properties import (
    BATZLE_WANG_DOI,
    SPAN_WAGNER_DOI,
    batzle_wang_brine,
    span_wagner_co2,
)
from sage_avo.geology.rock_physics import elastic_moduli_gpa


REPOSITORY = Path(__file__).resolve().parents[1]
CALIBRATION_ID = "v0032_32cd1fe5f3ba8956"
SENSITIVITY_SEED = 320032
SENSITIVITY_CASES = 512
SCENARIO = {
    "seed_offset": 3_200_000,
    "pressure_mpa": [24.0, 36.0],
    "temperature_c": [55.0, 95.0],
    "salinity_mass_fraction": [0.006, 0.12],
    "brie_exponent": [2.0, 4.0],
}
HISTORICAL = {
    "brine_bulk_modulus_gpa": 2.20,
    "brine_density_g_cc": 1.03,
    "co2_bulk_modulus_gpa": 0.10,
    "co2_density_g_cc": 0.65,
    "brie_exponent": 3.0,
}


def _locations() -> dict[str, Path]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    private = Path(paths["private_artifact_root"]) / "revision32" / "fluid_provenance_gate"
    data = Path(paths["work_data_root"]) / "s01data"
    previous = (
        Path(paths["private_artifact_root"])
        / "revision31"
        / "v0031_validation8_fluid_corrected"
        / "fluid_gate"
    )
    return {
        "raw": Path(paths["s01_raw_root"]),
        "data": data,
        "model": data / "derived" / "fluid_models_v0032",
        "private": private,
        "tables": private / "tables",
        "figures": private / "figures",
        "reports": private / "reports",
        "previous": previous,
    }


def _source_hash(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _evidence_inventory(locations: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = locations["raw"]
    evidence: list[dict[str, Any]] = []
    well_summary: dict[str, Any] = {}
    for path in sorted(raw.rglob("*.las")):
        las = lasio.read(path)
        curves = {curve.mnemonic.upper(): curve for curve in las.curves}
        present = sorted(curves)
        well = path.stem.replace("_logs", "")
        sw = curves.get("SW")
        sw_summary = None
        if sw is not None:
            values = np.asarray(sw.data, dtype=float)
            values = values[np.isfinite(values)]
            sw_summary = {
                "samples": int(values.size),
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
                "curve_unit": sw.unit,
                "curve_description": sw.descr,
            }
            evidence.append(
                {
                    "quantity": "water saturation",
                    "source_file": str(path),
                    "exact_location": "LAS ~CURVE entry SW and sample column SW",
                    "evidence_type": "interpreted log curve; provenance method is absent",
                    "units": "blank in LAS header; numerical values behave as fraction",
                    "depth_interval_m": [
                        float(np.nanmin(las.index)),
                        float(np.nanmax(las.index)),
                    ],
                    "uncertainty": (
                        "No resistivity/SP inputs, petrophysical method, or uncertainty are supplied"
                    ),
                    "applicability_to_s01": True,
                    "reviewer_status": "machine-audited; domain-owner confirmation pending",
                    "accepted_use": "T73 samples with SW >= 0.95 are confirmed-brine candidates",
                }
            )
        well_summary[well] = {
            "source": _source_hash(path),
            "curves": present,
            "has_sw": sw is not None,
            "has_deep_resistivity": any(name in curves for name in ("RT", "RESD", "XRESD")),
            "has_sp": any(name in curves for name in ("SP", "XSP")),
            "has_neutron": any(name in curves for name in ("NPHI", "XNPHIL")),
            "sw_summary": sw_summary,
        }

    searched_files = sorted(
        path
        for path in raw.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".las", ".txt", ".dat", ".m", ".sgy", ".segy", ".mat"}
    )
    evidence.append(
        {
            "quantity": "pressure, temperature, salinity, PVT and pressure-test evidence",
            "source_file": str(raw),
            "exact_location": f"recursive audit of {len(searched_files)} authorized S01 files",
            "evidence_type": "absence finding",
            "units": None,
            "depth_interval_m": None,
            "uncertainty": "Additional undisclosed project records may exist outside the authorized tree",
            "applicability_to_s01": True,
            "reviewer_status": "machine-audited; domain-owner confirmation pending",
            "accepted_use": None,
            "finding": (
                "No pore pressure, bottom-hole temperature, geothermal gradient, formation-water "
                "chemistry/salinity, PVT, RFT/MDT, mud-weight, or CO2 operating-condition record found"
            ),
        }
    )
    evidence.append(
        {
            "quantity": "resistivity/SP evidence in CO2GOMReport.ipynb",
            "source_file": "excluded external notebook: CO2GOMReport.ipynb",
            "exact_location": "cells 20 and 51; welldata/well0.las through well2.las",
            "evidence_type": "explicit exclusion",
            "units": None,
            "depth_interval_m": None,
            "uncertainty": None,
            "applicability_to_s01": False,
            "reviewer_status": "excluded",
            "accepted_use": None,
            "finding": "Different Gulf Coast simulation project and three unrelated wells",
        }
    )
    return evidence, well_summary


def _classification_and_models(
    locations: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], CalibratedDryFrameModel, dict[str, Any]]:
    source = locations["previous"] / "calibration" / "calibration_well_samples.csv"
    table = pd.read_csv(source)
    table.insert(0, "sample_id", [f"{row.well}:{row.depth_m:.3f}m" for row in table.itertuples()])
    table["brine_confidence"] = np.where(table["well"].eq("T73"), "confirmed_brine", "ambiguous")
    table["brine_confidence_basis"] = np.where(
        table["well"].eq("T73"),
        "T73 SW log >= 0.95",
        "pre-CO2 baseline only; no SW, resistivity, SP, neutron-density, contact, or history evidence",
    )
    sets = {
        "A_T73_confirmed": table[table["brine_confidence"].eq("confirmed_brine")].copy(),
        "B_all_confirmed_or_probable": table[
            table["brine_confidence"].isin(["confirmed_brine", "probable_brine"])
        ].copy(),
        "C_historical_five_well": table.copy(),
    }
    confirmed = sets["A_T73_confirmed"]
    if len(confirmed) != 325 or not confirmed["well"].eq("T73").all():
        raise RuntimeError("Confirmed-brine set does not match the audited T73 contract")
    features = confirmed[["density_porosity", "shaliness", "depth_m"]].to_numpy(float)
    features[:, 2] /= 1000.0
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    historical_metadata_path = (
        locations["data"]
        / "derived"
        / "fluid_models_v0031"
        / "calibrated_dry_frame.json"
    )
    historical_metadata = json.loads(historical_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 2,
        "calibration_id": CALIBRATION_ID,
        "method": "Candidate-B local-neighbour dry frame restricted to confirmed T73 SW >= 0.95 samples",
        "algorithm_changed_from_v0031": False,
        "sample_evidence_changed_from_v0031": True,
        "samples": len(confirmed),
        "wells": ["T73"],
        "neighbor_count": 32,
        "time_depth_linear_coefficients": historical_metadata[
            "time_depth_linear_coefficients"
        ],
        "time_depth_source": _source_hash(historical_metadata_path),
        "brine_confidence": "confirmed_brine",
        "limitations": [
            "T73 contains no deep-resistivity, SP, or neutron curve in the supplied LAS file.",
            "The confirmed set does not cover the clean-sand, shallow, or high-porosity support of the historical five-well model.",
            "The SW curve is interpreted and has no supplied petrophysical provenance or uncertainty.",
        ],
        "clipping_or_projection": False,
    }
    model = CalibratedDryFrameModel(
        calibration_id=CALIBRATION_ID,
        feature_names=("effective_porosity_fraction", "DELTA_shaliness_fraction", "depth_km"),
        feature_center=center,
        feature_scale=scale,
        features_standardized=(features - center) / scale,
        log_dry_bulk_gpa=np.log(confirmed["dry_bulk_density_phi_gpa"].to_numpy(float)),
        log_shear_gpa=np.log(confirmed["shear_gpa"].to_numpy(float)),
        well_ids=confirmed["well"].to_numpy(str),
        neighbor_count=32,
        metadata=metadata,
    )
    locations["model"].mkdir(parents=True, exist_ok=True)
    model_path, model_json = save_calibrated_dry_frame(
        model, locations["model"] / "calibrated_dry_frame_confirmed_brine.npz"
    )
    locations["tables"].mkdir(parents=True, exist_ok=True)
    classification_path = locations["tables"] / "brine_sample_classification.csv"
    table.to_csv(classification_path, index=False)
    comparison = _calibration_comparison(sets, locations)
    model_report = {
        "calibration_id": CALIBRATION_ID,
        "artifact": _source_hash(model_path),
        "metadata": _source_hash(model_json),
        "classification_table": _source_hash(classification_path),
        "comparison": comparison,
    }
    return table, sets, model, model_report


def _response_table(samples: pd.DataFrame, label: str) -> pd.DataFrame:
    brine = batzle_wang_brine(30.0, 80.0, 0.063)
    co2 = span_wagner_co2(30.0, 80.0)
    rows: list[pd.DataFrame] = []
    for saturation in np.linspace(0.0, 0.8, 17):
        sat = np.full(len(samples), saturation)
        fluid_bulk, fluid_density = brie_fluid_mixture(
            sat,
            brine_bulk_modulus_gpa=brine.bulk_modulus_gpa,
            co2_bulk_modulus_gpa=co2.bulk_modulus_gpa,
            brine_density_g_cc=brine.density_g_cc,
            co2_density_g_cc=co2.density_g_cc,
            brie_exponent=3.0,
        )
        dry = samples["dry_bulk_density_phi_gpa"].to_numpy(float)
        phi = samples["density_porosity"].to_numpy(float)
        mineral = samples["mineral_bulk_gpa"].to_numpy(float)
        baseline_reference = forward_gassmann_bulk_strict(dry, phi, mineral, brine.bulk_modulus_gpa)
        target_reference = forward_gassmann_bulk_strict(dry, phi, mineral, fluid_bulk)
        rf_bulk, rf_shear = elastic_moduli_gpa(
            samples["vp_m_s"].to_numpy(float),
            samples["vs_m_s"].to_numpy(float),
            samples["density_g_cc"].to_numpy(float),
        )
        delta_bulk = target_reference - baseline_reference
        target_density = samples["density_g_cc"].to_numpy(float) + phi * (
            fluid_density - brine.density_g_cc
        )
        target = elastic_from_gpa_strict(rf_bulk + delta_bulk, rf_shear, target_density)
        frame = pd.DataFrame(
            {
                "set": label,
                "saturation": saturation,
                "delta_vp_m_s": target.vp - samples["vp_m_s"].to_numpy(float),
                "delta_vs_m_s": target.vs - samples["vs_m_s"].to_numpy(float),
                "delta_density_g_cc": target.density - samples["density_g_cc"].to_numpy(float),
                "delta_ksat_gpa": delta_bulk,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _calibration_comparison(
    sets: dict[str, pd.DataFrame], locations: dict[str, Path]
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    responses = []
    for label, frame in sets.items():
        for quantity in (
            "density_porosity",
            "dry_bulk_density_phi_gpa",
            "shear_gpa",
            "dry_to_shear_density_phi",
            "dry_poisson_density_phi",
            "shaliness",
            "depth_m",
        ):
            values = frame[quantity].to_numpy(float)
            summaries.append(
                {
                    "set": label,
                    "samples": len(frame),
                    "wells": ",".join(sorted(frame["well"].unique())),
                    "quantity": quantity,
                    "p01": float(np.quantile(values, 0.01)),
                    "median": float(np.median(values)),
                    "p99": float(np.quantile(values, 0.99)),
                }
            )
        responses.append(_response_table(frame, label))
    summary = pd.DataFrame(summaries)
    response = pd.concat(responses, ignore_index=True)
    summary_path = locations["tables"] / "brine_calibration_set_comparison.csv"
    response_path = locations["tables"] / "brine_calibration_response_samples.csv"
    summary.to_csv(summary_path, index=False)
    response.to_csv(response_path, index=False)
    a = sets["A_T73_confirmed"]
    c = sets["C_historical_five_well"]
    material = {
        "dry_bulk_median_relative_change": float(
            np.median(a["dry_bulk_density_phi_gpa"]) / np.median(c["dry_bulk_density_phi_gpa"]) - 1.0
        ),
        "effective_porosity_median_relative_change": float(
            np.median(a["density_porosity"]) / np.median(c["density_porosity"]) - 1.0
        ),
        "A_clean_sand_fraction": float((a["shaliness"] < 0.5).mean()),
        "C_clean_sand_fraction": float((c["shaliness"] < 0.5).mean()),
        "A_depth_range_m": [float(a["depth_m"].min()), float(a["depth_m"].max())],
        "C_depth_range_m": [float(c["depth_m"].min()), float(c["depth_m"].max())],
        "material_change": True,
    }
    return {
        "set_counts": {name: len(frame) for name, frame in sets.items()},
        "set_B_equals_set_A": True,
        "summary_table": _source_hash(summary_path),
        "response_table": _source_hash(response_path),
        "materiality": material,
    }


def _scenario_property_envelope() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for pressure, temperature, salinity in itertools.product(
        SCENARIO["pressure_mpa"],
        SCENARIO["temperature_c"],
        SCENARIO["salinity_mass_fraction"],
    ):
        brine = batzle_wang_brine(pressure, temperature, salinity)
        co2 = span_wagner_co2(pressure, temperature)
        rows.append(
            {
                "pressure_mpa": pressure,
                "temperature_c": temperature,
                "salinity_mass_fraction": salinity,
                "brine_density_g_cc": brine.density_g_cc,
                "brine_bulk_modulus_gpa": brine.bulk_modulus_gpa,
                "brine_acoustic_velocity_m_s": brine.acoustic_velocity_m_s,
                "co2_density_g_cc": co2.density_g_cc,
                "co2_bulk_modulus_gpa": co2.bulk_modulus_gpa,
                "co2_acoustic_velocity_m_s": co2.acoustic_velocity_m_s,
                "co2_phase": co2.phase,
            }
        )
    table = pd.DataFrame(rows)
    comparison = {}
    for name in (
        "brine_density_g_cc",
        "brine_bulk_modulus_gpa",
        "co2_density_g_cc",
        "co2_bulk_modulus_gpa",
    ):
        value = HISTORICAL[name]
        lower, upper = float(table[name].min()), float(table[name].max())
        comparison[name] = {
            "historical": value,
            "scenario_min": lower,
            "scenario_max": upper,
            "inside": lower <= value <= upper,
        }
    comparison["brie_exponent"] = {
        "historical": 3.0,
        "scenario_min": SCENARIO["brie_exponent"][0],
        "scenario_max": SCENARIO["brie_exponent"][1],
        "inside": True,
    }
    return table, comparison


def _sensitivity(
    locations: dict[str, Path],
    model: CalibratedDryFrameModel,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gate = pd.read_csv(locations["previous"] / "diagnosis" / "gate_candidate_brine_state_diagnostic.csv")
    design = qmc.LatinHypercube(d=6, seed=SENSITIVITY_SEED).random(SENSITIVITY_CASES)
    bounds = np.asarray(
        [
            SCENARIO["pressure_mpa"],
            SCENARIO["temperature_c"],
            SCENARIO["salinity_mass_fraction"],
            SCENARIO["brie_exponent"],
            [0.0, 0.8],
            [0.0, float(len(gate))],
        ],
        dtype=float,
    )
    values = qmc.scale(design, bounds[:, 0], bounds[:, 1])
    angles = np.arange(3.0, 46.0, 1.0)
    rows: list[dict[str, Any]] = []
    for case_id, case in enumerate(values):
        pressure, temperature, salinity, exponent, saturation, selector = case
        gate_row = gate.iloc[min(int(selector), len(gate) - 1)]
        brine = batzle_wang_brine(pressure, temperature, salinity)
        co2 = span_wagner_co2(pressure, temperature)
        physics = replace(
            FluidRockPhysics(),
            brine_bulk_modulus_gpa=brine.bulk_modulus_gpa,
            brine_density_g_cc=brine.density_g_cc,
            co2_bulk_modulus_gpa=co2.bulk_modulus_gpa,
            co2_density_g_cc=co2.density_g_cc,
            brie_exponent=float(exponent),
        )
        inputs = {
            "vp_brine_m_s": np.asarray([gate_row.vp_m_s]),
            "vs_brine_m_s": np.asarray([gate_row.vs_m_s]),
            "density_brine_g_cc": np.asarray([gate_row.density_g_cc]),
            "input_porosity": np.asarray([gate_row.input_porosity]),
            "shaliness": np.asarray([gate_row.shaliness]),
            "depth_m": np.asarray([gate_row.depth_m]),
            "calibration": model,
            "physics": physics,
        }
        result = calibrated_differential_gassmann_substitution(
            co2_saturation=np.asarray([saturation]), **inputs
        )
        zero = calibrated_differential_gassmann_substitution(
            co2_saturation=np.asarray([0.0]), **inputs
        )
        fluid_bulk, fluid_density = brie_fluid_mixture(
            np.asarray([saturation]),
            brine_bulk_modulus_gpa=brine.bulk_modulus_gpa,
            co2_bulk_modulus_gpa=co2.bulk_modulus_gpa,
            brine_density_g_cc=brine.density_g_cc,
            co2_density_g_cc=co2.density_g_cc,
            brie_exponent=float(exponent),
        )
        avo = np.asarray(
            [
                zoeppritz_pp(
                    gate_row.vp_m_s,
                    gate_row.vs_m_s,
                    gate_row.density_g_cc,
                    result.elastic.vp[0],
                    result.elastic.vs[0],
                    result.elastic.density[0],
                    float(angle),
                )
                for angle in angles
            ]
        )
        ratio = result.dry_bulk_gpa[0] / result.frame_shear_gpa[0]
        poisson = poisson_ratio_from_moduli(result.dry_bulk_gpa, result.frame_shear_gpa)[0]
        zero_error = max(
            abs(zero.elastic.vp[0] - gate_row.vp_m_s),
            abs(zero.elastic.vs[0] - gate_row.vs_m_s),
            abs(zero.elastic.density[0] - gate_row.density_g_cc),
        )
        fixed_shear_error = abs(result.target_shear_gpa[0] - result.rf_shear_gpa[0])
        physical = bool(
            np.isfinite(avo).all()
            and result.elastic.vp[0] > result.elastic.vs[0] > 0.0
            and result.elastic.density[0] > 0.0
            and result.dry_bulk_gpa[0] > 0.0
            and zero_error <= 1e-10
            and fixed_shear_error <= 1e-12
        )
        rows.append(
            {
                "case_id": case_id,
                "pressure_mpa": pressure,
                "temperature_c": temperature,
                "salinity_mass_fraction": salinity,
                "brie_exponent": exponent,
                "co2_saturation": saturation,
                "input_porosity": gate_row.input_porosity,
                "rf_vp_m_s": gate_row.vp_m_s,
                "rf_vs_m_s": gate_row.vs_m_s,
                "rf_density_g_cc": gate_row.density_g_cc,
                "effective_porosity": result.effective_porosity[0],
                "shaliness": gate_row.shaliness,
                "facies": gate_row.facies,
                "depth_m": gate_row.depth_m,
                "brine_density_g_cc": brine.density_g_cc,
                "brine_bulk_modulus_gpa": brine.bulk_modulus_gpa,
                "brine_acoustic_velocity_m_s": brine.acoustic_velocity_m_s,
                "co2_density_g_cc": co2.density_g_cc,
                "co2_bulk_modulus_gpa": co2.bulk_modulus_gpa,
                "co2_acoustic_velocity_m_s": co2.acoustic_velocity_m_s,
                "co2_phase": co2.phase,
                "mixed_fluid_density_g_cc": fluid_density[0],
                "mixed_fluid_bulk_modulus_gpa": fluid_bulk[0],
                "delta_ksat_gpa": result.delta_bulk_gpa[0],
                "delta_density_g_cc": result.delta_density_g_cc[0],
                "delta_vp_m_s": result.elastic.vp[0] - gate_row.vp_m_s,
                "delta_vs_m_s": result.elastic.vs[0] - gate_row.vs_m_s,
                "zero_saturation_recovery_error": zero_error,
                "fixed_shear_error_gpa": fixed_shear_error,
                "dry_bulk_gpa": result.dry_bulk_gpa[0],
                "dry_to_shear": ratio,
                "dry_poisson_ratio": poisson,
                "nearest_calibration_distance": result.nearest_calibration_distance[0],
                "ava_rms": float(np.sqrt(np.mean(avo**2))),
                "ava_max_abs": float(np.max(np.abs(avo))),
                "phase_state_valid": True,
                "physical_valid": physical,
            }
        )
    table = pd.DataFrame(rows)
    destination = locations["tables"] / "candidate_b_fluid_sensitivity.csv"
    table.to_csv(destination, index=False)
    inputs = [
        "pressure_mpa",
        "temperature_c",
        "salinity_mass_fraction",
        "brie_exponent",
        "co2_saturation",
        "effective_porosity",
        "shaliness",
        "depth_m",
    ]
    outputs = ["delta_vp_m_s", "delta_vs_m_s", "delta_density_g_cc", "ava_rms"]
    dominance = []
    for output in outputs:
        for variable in inputs:
            coefficient, pvalue = spearmanr(table[variable], table[output])
            dominance.append(
                {
                    "output": output,
                    "variable": variable,
                    "spearman_rho": float(coefficient),
                    "absolute_rho": float(abs(coefficient)),
                    "pvalue": float(pvalue),
                }
            )
    dominance_table = pd.DataFrame(dominance).sort_values(
        ["output", "absolute_rho"], ascending=[True, False]
    )
    dominance_path = locations["tables"] / "sensitivity_dominance_spearman.csv"
    dominance_table.to_csv(dominance_path, index=False)
    summary = {
        "design": "six-dimensional Latin hypercube plus deterministic gate-state selection",
        "seed": SENSITIVITY_SEED,
        "cases": len(table),
        "all_phase_states_valid": bool(table["phase_state_valid"].all()),
        "all_physical_states_valid": bool(table["physical_valid"].all()),
        "maximum_zero_saturation_recovery_error": float(
            table["zero_saturation_recovery_error"].max()
        ),
        "maximum_fixed_shear_error_gpa": float(table["fixed_shear_error_gpa"].max()),
        "nearest_calibration_distance_percentiles": np.quantile(
            table["nearest_calibration_distance"], [0.5, 0.95, 0.99]
        ).tolist(),
        "percentiles": {
            name: dict(
                zip(
                    ("p01", "p05", "median", "p95", "p99"),
                    np.quantile(table[name], [0.01, 0.05, 0.5, 0.95, 0.99]).tolist(),
                )
            )
            for name in (
                "delta_vp_m_s",
                "delta_vs_m_s",
                "delta_density_g_cc",
                "delta_ksat_gpa",
                "ava_rms",
            )
        },
        "dominant_variables": {
            output: dominance_table[dominance_table["output"].eq(output)]
            .head(3)[["variable", "spearman_rho"]]
            .to_dict(orient="records")
            for output in outputs
        },
        "table": _source_hash(destination),
        "dominance_table": _source_hash(dominance_path),
    }
    return table, summary


def _figures(
    sensitivity: pd.DataFrame,
    sets: dict[str, pd.DataFrame],
    locations: dict[str, Path],
) -> list[dict[str, Any]]:
    destination = locations["figures"]
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, color, label in (
        (axes[0], "pressure_mpa", "Pressure (MPa)"),
        (axes[1], "temperature_c", "Temperature (degC)"),
    ):
        scatter = axis.scatter(
            sensitivity["co2_saturation"], sensitivity["delta_vp_m_s"],
            c=sensitivity[color], cmap="viridis", s=18, alpha=0.65,
        )
        axis.set(xlabel="CO2 saturation", ylabel="Delta Vp (m/s)")
        figure.colorbar(scatter, ax=axis, label=label)
    figure.suptitle("Candidate-B velocity response across the approved scenario")
    path = destination / "01_delta_vp_vs_saturation_pressure_temperature.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source_hash(path), "message": "Delta Vp sensitivity to saturation, pressure, and temperature"})

    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    scatter = axis.scatter(
        sensitivity["co2_saturation"], sensitivity["delta_density_g_cc"],
        c=sensitivity["salinity_mass_fraction"] * 1e6, cmap="plasma", s=20, alpha=0.7,
    )
    axis.set(xlabel="CO2 saturation", ylabel="Delta density (g/cc)")
    figure.colorbar(scatter, ax=axis, label="NaCl salinity (ppm by mass)")
    axis.set_title("Density response across the salinity scenario")
    path = destination / "02_delta_density_vs_saturation_salinity.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source_hash(path), "message": "Delta density sensitivity to saturation and salinity"})

    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    nominal_brine = batzle_wang_brine(30.0, 80.0, 0.063)
    nominal_co2 = span_wagner_co2(30.0, 80.0)
    saturation = np.linspace(0.0, 0.8, 101)
    for exponent, style in ((2.0, "--"), (3.0, "-"), (4.0, ":")):
        bulk, _ = brie_fluid_mixture(
            saturation,
            brine_bulk_modulus_gpa=nominal_brine.bulk_modulus_gpa,
            co2_bulk_modulus_gpa=nominal_co2.bulk_modulus_gpa,
            brine_density_g_cc=nominal_brine.density_g_cc,
            co2_density_g_cc=nominal_co2.density_g_cc,
            brie_exponent=exponent,
        )
        axis.plot(saturation, bulk, style, linewidth=2, label=f"Brie e={exponent:g}")
    axis.set(xlabel="CO2 saturation", ylabel="Mixed-fluid bulk modulus (GPa)")
    axis.set_title("Brie mixing uncertainty at the nominal EOS state")
    axis.legend()
    path = destination / "03_mixed_fluid_modulus_brie_sensitivity.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source_hash(path), "message": "Low/nominal/high Brie-exponent sensitivity"})

    figure, axis = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    colors = {"A_T73_confirmed": "tab:blue", "C_historical_five_well": "tab:gray"}
    for label in colors:
        response = _response_table(sets[label], label)
        grouped = response.groupby("saturation")["delta_vp_m_s"]
        x = np.asarray(sorted(response["saturation"].unique()))
        median = grouped.median().reindex(x).to_numpy()
        low = grouped.quantile(0.05).reindex(x).to_numpy()
        high = grouped.quantile(0.95).reindex(x).to_numpy()
        axis.plot(x, median, color=colors[label], label=label)
        axis.fill_between(x, low, high, color=colors[label], alpha=0.18)
    axis.scatter(
        sensitivity["co2_saturation"], sensitivity["delta_vp_m_s"],
        s=8, color="tab:red", alpha=0.18, label="v0032 scenario / confirmed model",
    )
    axis.set(xlabel="CO2 saturation", ylabel="Delta Vp (m/s)")
    axis.set_title("Candidate response versus confirmed and historical dry-frame envelopes")
    axis.legend()
    path = destination / "04_candidate_vs_well_calibrated_envelopes.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source_hash(path), "message": "Confirmed-T73 versus historical five-well response support"})

    ordered = sensitivity.sort_values("delta_vp_m_s")
    representatives = {
        "high response": ordered.iloc[0],
        "nominal response": ordered.iloc[len(ordered) // 2],
        "low response": ordered.iloc[-1],
    }
    angles = np.arange(3.0, 46.0)
    figure, axis = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    for label, row in representatives.items():
        vp2 = row.rf_vp_m_s + row.delta_vp_m_s
        vs2 = row.rf_vs_m_s + row.delta_vs_m_s
        rho2 = row.rf_density_g_cc + row.delta_density_g_cc
        reflectivity = [
            zoeppritz_pp(
                row.rf_vp_m_s,
                row.rf_vs_m_s,
                row.rf_density_g_cc,
                vp2,
                vs2,
                rho2,
                angle,
            )
            for angle in angles
        ]
        axis.plot(angles, reflectivity, linewidth=2, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="Incidence angle (degrees)", ylabel="Exact PP reflection coefficient")
    axis.set_title("Exact-Zoeppritz fluid-contact AVA scenario sensitivity")
    axis.legend()
    path = destination / "05_exact_zoeppritz_ava_fluid_sensitivity.png"
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    outputs.append({"figure": _source_hash(path), "message": "Representative low/nominal/high exact-Zoeppritz AVA response"})
    return outputs


def validate(_: argparse.Namespace) -> None:
    locations = _locations()
    for name in ("model", "tables", "figures", "reports"):
        locations[name].mkdir(parents=True, exist_ok=True)
    evidence, wells = _evidence_inventory(locations)
    classifications, sets, model, model_report = _classification_and_models(locations)
    property_table, historical_comparison = _scenario_property_envelope()
    property_path = locations["tables"] / "fluid_property_corner_table.csv"
    property_table.to_csv(property_path, index=False)
    sensitivity, sensitivity_report = _sensitivity(locations, model)
    figures = _figures(sensitivity, sets, locations)

    gate = pd.read_csv(locations["previous"] / "diagnosis" / "gate_candidate_brine_state_diagnostic.csv")
    query = gate[["density_porosity", "shaliness", "depth_m"]].to_numpy(float)
    query[:, 2] /= 1000.0
    query = (query - model.feature_center) / model.feature_scale
    distances, _ = cKDTree(model.features_standardized).query(query, k=1)
    training_distance = cKDTree(model.features_standardized).query(
        model.features_standardized, k=2
    )[0][:, 1]
    support_threshold = float(np.quantile(training_distance, 0.99))
    out_of_support_fraction = float((distances > support_threshold).mean())

    evidence_path = locations["tables"] / "s01_fluid_evidence_inventory.json"
    write_json(evidence_path, {"evidence": evidence, "wells": wells})
    config_path = REPOSITORY / "configs" / "synthetic_s01_v0032.yaml"
    validation_id_payload = {
        "config_sha256": file_sha256(config_path),
        "evidence_sha256": file_sha256(evidence_path),
        "classification_sha256": model_report["classification_table"]["sha256"],
        "sensitivity_sha256": sensitivity_report["table"]["sha256"],
    }
    validation_id = "v0032_" + hashlib.sha256(
        json.dumps(validation_id_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    status = "scenario_validated"
    allowed_claims = [
        "Candidate-B responses are scenario-conditioned over the declared P/T/salinity/Brie ranges.",
        "Pure-CO2 properties use CoolProp HEOS/Span-Wagner and NaCl brine uses Batzle-Wang.",
        "The bounded sensitivity cases are physically valid within the tested scenario.",
    ]
    prohibited_claims = [
        "The sampled pressure, temperature, salinity, or CO2 properties are measured S01 field conditions.",
        "T76, T732, T761, or T762 calibration intervals are proven brine-saturated.",
        "The v0032 scenario is a calibrated posterior uncertainty distribution.",
        "The confirmed T73-only dry-frame model has adequate support for the full S01 gate or production corpus.",
        "Field-specific quantitative CO2-property or plume-response predictions.",
    ]
    artifact = {
        "schema_version": 1,
        "validation_id": validation_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "claim_scope": "scenario-conditioned fluid-property sensitivity; not S01 field-condition validation",
        "field_evidence_result": {
            "pressure": "unavailable",
            "temperature": "unavailable",
            "salinity": "unavailable",
            "co2_operating_state": "unavailable",
            "evidence_inventory": _source_hash(evidence_path),
        },
        "scenario_sampling": SCENARIO,
        "accepted_pressure_range_mpa": SCENARIO["pressure_mpa"],
        "accepted_temperature_range_c": SCENARIO["temperature_c"],
        "accepted_salinity_mass_fraction_range": SCENARIO["salinity_mass_fraction"],
        "accepted_brie_exponent_range": SCENARIO["brie_exponent"],
        "range_rationale": {
            "pressure": "Hydrostatic scenario around the observed 2.4-3.25 km calibration depth support; not measured S01 pore pressure.",
            "temperature": "Broad deep-storage sensitivity range wholly above the CO2 critical temperature; not measured S01 temperature.",
            "salinity": "EPA Class-VI guidance example formation-water span of 6,000-120,000 mg/L, represented as NaCl mass fraction for Batzle-Wang sensitivity; not S01 chemistry.",
            "brie_exponent": "2-4 brackets the historical nominal e=3 as an explicit fluid-distribution sensitivity parameter; no field calibration is claimed.",
            "single_phase": "Every sampled P/T state is checked by CoolProp and must be supercritical single-phase CO2.",
        },
        "authoritative_sources": [
            {"title": "Batzle and Wang, Seismic properties of pore fluids", "doi": BATZLE_WANG_DOI},
            {"title": "Span and Wagner CO2 reference EOS", "doi": SPAN_WAGNER_DOI},
            {"title": "US EPA Class VI Well Site Characterization Guidance", "identifier": "EPA 816-R-13-004"},
            {"title": "NETL Carbon Storage FAQ: supercritical storage conditions", "url": "https://netl.doe.gov/carbon-management/carbon-storage/faqs/carbon-storage-faqs"},
        ],
        "brine_model": {
            "name": "Batzle-Wang NaCl density and acoustic-velocity correlations",
            "reference_doi": BATZLE_WANG_DOI,
            "implementation": "sage_avo.geology.fluid_properties.batzle_wang_brine",
            "units": {"pressure": "MPa", "temperature": "degC", "salinity": "NaCl mass fraction", "density": "g/cc", "velocity": "m/s", "bulk_modulus": "GPa"},
            "extrapolation_policy": "reject outside conservative 5-60 MPa, 20-100 degC, and 0-0.32 NaCl mass fraction support",
        },
        "co2_model": {
            "name": "CoolProp HEOS Span-Wagner pure-CO2 EOS",
            "reference_doi": SPAN_WAGNER_DOI,
            "backend": "HEOS::CarbonDioxide",
            "software_version": CoolProp.__version__,
            "bulk_modulus_formula": "K_CO2 = rho_CO2 * acoustic_velocity_CO2^2",
            "phase_policy": "reject two-phase, unknown, critical-point, non-supercritical, and EOS-limit states",
        },
        "fluid_mixing": {
            "model": "Brie bulk-modulus mixing plus arithmetic density",
            "historical_nominal_exponent": 3.0,
            "exponent_interpretation": "uncertain fluid-distribution parameter, not field truth",
        },
        "fluid_property_table": _source_hash(property_path),
        "phase_state_checks": {
            "corner_states": len(property_table),
            "all_supercritical": bool(property_table["co2_phase"].str.startswith("supercritical").all()),
        },
        "historical_constant_comparison": historical_comparison,
        "sensitivity": sensitivity_report,
        "figures": figures,
        "brine_confidence": {
            "classification_counts": classifications.groupby(["well", "brine_confidence"]).size().rename("samples").reset_index().to_dict(orient="records"),
            "confirmed_sample_ids": sets["A_T73_confirmed"]["sample_id"].tolist(),
            "probable_sample_ids": [],
            "classification_table": model_report["classification_table"],
        },
        "well_calibration": model_report,
        "calibration_id": CALIBRATION_ID,
        "candidate_b_calibration_id": CALIBRATION_ID,
        "candidate_b_method": "calibrated_differential_gassmann; algorithm unchanged from v0031",
        "primary_calibration_support_gate": {
            "nearest_distance_p99_training_threshold": support_threshold,
            "gate_out_of_support_fraction": out_of_support_fraction,
            "passed": out_of_support_fraction <= 0.05,
            "interpretation": "T73-only confirmed-brine support is inadequate when more than 5% of gate pixels exceed the within-training p99 nearest-neighbour distance.",
        },
        "source_and_configuration_hashes": {
            **validation_id_payload,
            "git_head": _git_head(),
            "historical_v0031_calibration_sha256": file_sha256(
                locations["data"] / "derived" / "fluid_models_v0031" / "calibrated_dry_frame.npz"
            ),
        },
        "reviewer_status": {
            "automated_review": "complete",
            "domain_owner_review": "pending",
            "field_data_custodian_review": "pending",
        },
        "allowed_scientific_claims": allowed_claims,
        "prohibited_scientific_claims": prohibited_claims,
        "production_approval": False,
        "production_blocker": (
            "The confirmed T73-only calibration changes the dry-frame envelope materially and "
            f"leaves {out_of_support_fraction:.2%} of the bounded gate outside confirmed-brine support."
        ),
    }
    artifact_path = locations["model"] / "fluid_property_validation.json"
    write_json(artifact_path, artifact)
    report = {
        "status": status,
        "validation_artifact": _source_hash(artifact_path),
        "production_decision": "NO-GO",
        "exact_blocker": artifact["production_blocker"],
        "regenerate_eight_authorized_for_diagnostic_qc": True,
        "full_100_realization_generation_authorized": False,
    }
    write_json(locations["reports"] / "fluid_provenance_gate.json", report)
    print(json.dumps(report, indent=2))


def _git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _matched_channel_qc(previous: Path, current: Path) -> dict[str, Any]:
    nonfluid_channels = (
        "elastic_brine",
        "delta",
        "sand_probability",
        "porosity",
        "rgt",
        "strat_fraction",
        "reservoir_mask",
        "horizon_top_ms",
        "horizon_base_ms",
        "source_horizon_top_ms",
        "source_horizon_base_ms",
        "segmentation",
        "plume_mask",
        "co2_saturation",
        "angles_degrees",
        "time_ms",
        "cdp",
    )
    rows = []
    for old_path in sorted(previous.glob("realization_*.npz")):
        new_path = current / old_path.name
        if not new_path.exists():
            raise FileNotFoundError(f"Missing matched v0032 realization {new_path}")
        with np.load(old_path, allow_pickle=False) as old, np.load(
            new_path, allow_pickle=False
        ) as new:
            realization_id = int(old["realization_id"])
            plume = np.asarray(new["plume_mask"], dtype=bool)
            outside = ~plume
            elastic = np.asarray(new["elastic"], dtype=float)
            brine = np.asarray(new["elastic_brine"], dtype=float)
            old_brine = np.asarray(old["elastic_brine"], dtype=float)
            _, shear = elastic_moduli_gpa(*elastic)
            _, shear_brine = elastic_moduli_gpa(*brine)
            metadata = json.loads(new_path.with_suffix(".json").read_text(encoding="utf-8"))
            fluid_metadata = metadata["geology"]["fluid"]
            state = fluid_metadata["property_state"]
            bitwise = {
                name: bool(np.array_equal(old[name], new[name])) for name in nonfluid_channels
            }
            rows.append(
                {
                    "realization_id": realization_id,
                    "all_nonfluid_channels_bitwise_equal": bool(all(bitwise.values())),
                    "nonfluid_channel_equality": bitwise,
                    "elastic_brine_bitwise_equal": bool(np.array_equal(old_brine, brine)),
                    "maximum_outside_plume_elastic_change": float(
                        np.max(np.abs(elastic[:, outside] - brine[:, outside]))
                    ),
                    "maximum_inside_plume_shear_modulus_change_gpa": float(
                        np.max(np.abs(shear[plume] - shear_brine[plume])) if plume.any() else 0.0
                    ),
                    "plume_pixels": int(plume.sum()),
                    "feasibility_projection_used": fluid_metadata.get(
                        "feasibility_projection_used"
                    ),
                    "dry_bulk_clipping_used": fluid_metadata.get("dry_bulk_clipping_used"),
                    "elastic_output_clipping_used": fluid_metadata.get(
                        "elastic_output_clipping_used"
                    ),
                    "sampled_pressure_mpa": state["brine"]["pressure_mpa"],
                    "sampled_temperature_c": state["brine"]["temperature_c"],
                    "sampled_salinity_mass_fraction": state["brine"][
                        "salinity_mass_fraction"
                    ],
                    "sampled_brie_exponent": state["brie_exponent"],
                    "co2_phase": state["co2"]["phase"],
                }
            )
    frame = pd.DataFrame(rows)
    return {
        "realizations": rows,
        "all_nonfluid_channels_bitwise_equal": bool(
            frame["all_nonfluid_channels_bitwise_equal"].all()
        ),
        "maximum_outside_plume_elastic_change": float(
            frame["maximum_outside_plume_elastic_change"].max()
        ),
        "maximum_inside_plume_shear_modulus_change_gpa": float(
            frame["maximum_inside_plume_shear_modulus_change_gpa"].max()
        ),
        "no_projection_or_clipping": bool(
            all(
                row["feasibility_projection_used"] is False
                and row["dry_bulk_clipping_used"] is False
                and row["elastic_output_clipping_used"] is False
                for row in rows
            )
        ),
    }


def regenerate(_: argparse.Namespace) -> None:
    locations = _locations()
    artifact = json.loads(
        (locations["model"] / "fluid_property_validation.json").read_text(encoding="utf-8")
    )
    if artifact["status"] not in {"field_validated", "scenario_validated"}:
        raise RuntimeError("The v0032 fluid-property validation status does not authorize the gate")
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    synthetic = load_config(REPOSITORY / "configs" / "synthetic_s01_v0032.yaml")
    dataset = load_config(REPOSITORY / "configs" / "ml_dataset_s01_v0032.yaml")
    root = locations["private"].parent / "v0032_validation8_fluid_provenance"
    stage02 = root / "stage02" / "realizations"
    stage03 = root / "stage03" / "dataset"
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=stage02,
        workers=1,
        resume=False,
    )
    previous = (
        Path(paths["private_artifact_root"])
        / "revision31"
        / "v0031_validation8_fluid_corrected"
        / "stage02"
        / "realizations"
    )
    matched = _matched_channel_qc(previous, stage02)
    matched_path = root / "reports" / "matched_stage02_qc.json"
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(matched_path, matched)
    dataset_manifest = build_stage03_dataset(
        config=dataset,
        paths=paths,
        source_directory=stage02,
        output_directory=stage03,
    )
    integrity = validate_dataset_integrity(stage03)
    import run_revision31_fluid_gate as revision31

    round_trip = revision31._all_realization_round_trip(stage02, synthetic)
    checks = {
        "stage02_manifest": _source_hash(stage02 / "manifest.json"),
        "stage02_generated": manifest["generated_realizations"],
        "matched_qc": _source_hash(matched_path),
        "all_nonfluid_channels_bitwise_equal": matched[
            "all_nonfluid_channels_bitwise_equal"
        ],
        "exactly_zero_outside_plume": matched[
            "maximum_outside_plume_elastic_change"
        ]
        == 0.0,
        "fixed_shear_max_error_gpa": matched[
            "maximum_inside_plume_shear_modulus_change_gpa"
        ],
        "no_projection_or_clipping": matched["no_projection_or_clipping"],
        "round_trip": round_trip,
        "stage03_manifest": _source_hash(stage03 / "dataset_manifest.json"),
        "stage03_integrity": integrity,
        "stage03_builder_integrity": dataset_manifest["integrity"],
    }
    report_path = root / "reports" / "bounded_execution_qc.json"
    write_json(report_path, checks)
    print(json.dumps({"report": _source_hash(report_path), **checks}, indent=2))


def cuda_sanity(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The bounded v0032 sanity experiment requires CUDA")
    locations = _locations()
    root = locations["private"].parent / "v0032_validation8_fluid_provenance"
    dataset_directory = root / "stage03" / "dataset"
    if not (dataset_directory / "dataset_manifest.json").exists():
        raise FileNotFoundError("Run regenerate-eight before the CUDA sanity experiment")
    synthetic = load_config(REPOSITORY / "configs" / "synthetic_s01_v0032.yaml")
    dataset = load_config(REPOSITORY / "configs" / "ml_dataset_s01_v0032.yaml")
    training = load_config(REPOSITORY / "configs" / "sage_avo_s01_v0031.yaml")
    experiment_name = "sage_avo_s01_v0032_validation8_fluid_provenance"
    training["experiment"]["name"] = experiment_name
    training["experiment"]["output_root"] = f"results/experiments/{experiment_name}"
    training["dataset"]["directory"] = dataset["outputs"]["directory"]
    training["training"]["epochs"] = 2
    artifact = json.loads(
        (locations["model"] / "fluid_property_validation.json").read_text(encoding="utf-8")
    )
    source = {
        "status": "revision32_bounded_validation_unfrozen",
        "fluid_validation_id": artifact["validation_id"],
        "fluid_validation_sha256": file_sha256(
            locations["model"] / "fluid_property_validation.json"
        ),
    }
    for config in (synthetic, dataset, training):
        config["source_snapshot"] = source
    config_directory = root / "configs"
    config_directory.mkdir(parents=True, exist_ok=True)
    write_json(config_directory / "synthetic_resolved.json", synthetic)
    write_json(config_directory / "dataset_resolved.json", dataset)
    write_json(config_directory / "training_resolved.json", training)
    experiment = root / "stage04" / experiment_name
    output = train_controlled_variant(
        repository=REPOSITORY,
        config_path=config_directory / "training_resolved.json",
        config=training,
        dataset_directory=dataset_directory,
        experiment_directory=experiment,
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
        "manifest": _source_hash(Path(output) / "manifest.json"),
        "training_log": _source_hash(Path(output) / "training_log.csv"),
    }
    report_path = root / "reports" / "cuda_sanity.json"
    write_json(report_path, report)
    print(json.dumps({"report": _source_hash(report_path), **report}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate")
    validation.set_defaults(function=validate)
    regeneration = commands.add_parser("regenerate-eight")
    regeneration.set_defaults(function=regenerate)
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
