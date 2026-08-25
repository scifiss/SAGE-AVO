#!/usr/bin/env python3
"""Audit and validate deterministic support-aware Stage-02 generation.

This gate never edits the failed Revision-3.3 corpus.  It first evaluates
complete geological/fluid candidates, rejects an unsupported candidate as a
whole, and advances to a deterministic retry seed.  Only accepted candidates
are passed to the unchanged exact Stage-02 forward operator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sage_avo.config import load_config
from sage_avo.experiments import generate_stage02_dataset
from sage_avo.experiments.manifest import file_sha256, write_json
from sage_avo.experiments.synthetic_generation import load_stage01_background
from sage_avo.geology import (
    FluidRockPhysics,
    deterministic_candidate_seed,
    evaluate_candidate_support,
    load_calibrated_dry_frame,
    make_field_conditioned_realization,
    sample_fluid_scenario,
    support_contract_from_mapping,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FAILED_VERSION = "v0033_production100_dry_frame_supported"
VALIDATION_VERSION = "v00331_support_aware_validation12"
CALIBRATION_ID = "v0033_58a5fe39a11c4fe66431"


def _source(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path)}


def _locations(paths: dict[str, Any]) -> dict[str, Path]:
    private = Path(paths["private_artifact_root"])
    root = private / "revision331" / "support_aware_generation_gate"
    return {
        "root": root,
        "audit": root / "audit",
        "stress": root / "stress",
        "bounded": root / "bounded",
        "reports": root / "reports",
        "figures": root / "figures",
        "stage02": private
        / "stage_artifacts"
        / "stage02"
        / VALIDATION_VERSION
        / "realizations",
        "failed_stage02": private
        / "stage_artifacts"
        / "stage02"
        / FAILED_VERSION
        / "realizations",
    }


def _contracts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
]:
    paths = load_config(REPOSITORY / "configs" / "paths.yaml")
    support_path = REPOSITORY / "configs" / "revision331_support_acceptance.yaml"
    support = load_config(support_path)
    synthetic = deepcopy(
        load_config(REPOSITORY / "configs" / "synthetic_s01_v0032.yaml")
    )
    synthetic["stage"].update(
        {
            "name": "field_conditioned_synthetic_avo_v00331_support_aware",
            "geology_realization_count": 100,
            "observation_variants_per_geology": 1,
            "realization_count": 100,
            "realization_id_offset": 3_410_000,
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
    synthetic["support_aware_acceptance"] = deepcopy(support)
    synthetic["support_aware_acceptance_source"] = {
        "path": "configs/revision331_support_acceptance.yaml",
        "sha256": file_sha256(support_path),
    }
    synthetic["outputs"].update(
        {
            "version": VALIDATION_VERSION,
            "directory": f"synthetic/{VALIDATION_VERSION}/realizations",
        }
    )
    synthetic["source_snapshot"] = {
        "status": "revision331_bounded_validation_before_source_freeze",
        "supersedes_for_full_corpus_generation": (
            "7dceaac7300313cd5390f55e7620baa5cdc563f1f5b9b2f2f3b3614118290507"
        ),
        "scientific_calibration_preserved": CALIBRATION_ID,
    }
    return paths, synthetic, support, _locations(paths)


def _load_dependencies(
    paths: dict[str, Any], synthetic: dict[str, Any]
) -> dict[str, Any]:
    arrays, reservoir_model, hashes = load_stage01_background(
        paths["work_data_root"],
        synthetic["inputs"]["dataset_id"],
        synthetic["inputs"]["structure_version"],
    )
    work_data = Path(paths["work_data_root"]) / synthetic["inputs"]["dataset_id"]
    fluid = synthetic["fluid_substitution"]
    calibration_path = work_data / fluid["calibration_artifact"]
    validation_path = work_data / fluid["fluid_property_validation_artifact"]
    calibration = load_calibrated_dry_frame(calibration_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    mapping = calibration.metadata["time_depth_linear_coefficients"]
    depth_by_row = (
        float(mapping["slope_m_per_ms"])
        * np.asarray(arrays["time_ms"], dtype=float)
        + float(mapping["intercept_m"])
    )
    depth_m = np.broadcast_to(depth_by_row[:, None], arrays["porosity"].shape)
    return {
        "arrays": arrays,
        "reservoir_model": reservoir_model,
        "source_hashes": hashes,
        "calibration": calibration,
        "validation": validation,
        "depth_m": depth_m,
        "calibration_path": calibration_path,
        "validation_path": validation_path,
    }


def _fluid_state(
    master_seed: int,
    synthetic: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], FluidRockPhysics]:
    fluid_config = deepcopy(synthetic["fluid_substitution"])
    metadata = sample_fluid_scenario(
        master_seed, validation["scenario_sampling"]
    )
    brine = metadata["brine"]
    co2 = metadata["co2"]
    fluid_config.update(
        {
            "brine_bulk_modulus_gpa": brine["bulk_modulus_gpa"],
            "brine_density_g_cc": brine["density_g_cc"],
            "co2_bulk_modulus_gpa": co2["bulk_modulus_gpa"],
            "co2_density_g_cc": co2["density_g_cc"],
            "brie_exponent": metadata["brie_exponent"],
        }
    )
    metadata.update(
        {
            "validation_id": validation["validation_id"],
            "validation_status": validation["status"],
            "claim_scope": validation["claim_scope"],
        }
    )
    physics = FluidRockPhysics(
        quartz_bulk_modulus_gpa=float(fluid_config["quartz_bulk_modulus_gpa"]),
        clay_bulk_modulus_gpa=float(fluid_config["clay_bulk_modulus_gpa"]),
        quartz_shear_modulus_gpa=float(fluid_config["quartz_shear_modulus_gpa"]),
        clay_shear_modulus_gpa=float(fluid_config["clay_shear_modulus_gpa"]),
        quartz_density_g_cc=float(fluid_config["quartz_density_g_cc"]),
        clay_density_g_cc=float(fluid_config["clay_density_g_cc"]),
        brine_bulk_modulus_gpa=float(fluid_config["brine_bulk_modulus_gpa"]),
        co2_bulk_modulus_gpa=float(fluid_config["co2_bulk_modulus_gpa"]),
        brine_density_g_cc=float(fluid_config["brine_density_g_cc"]),
        co2_density_g_cc=float(fluid_config["co2_density_g_cc"]),
        brie_exponent=float(fluid_config["brie_exponent"]),
    )
    return fluid_config, metadata, physics


def _candidate_hash(candidate: Any) -> str:
    digest = hashlib.sha256()
    for values in (
        candidate.elastic,
        candidate.elastic_brine,
        candidate.delta,
        candidate.porosity,
        candidate.rgt,
        candidate.plume_mask,
        candidate.co2_saturation,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _candidate(
    master_seed: int,
    attempt_index: int,
    synthetic: dict[str, Any],
    dependencies: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    support = synthetic["support_aware_acceptance"]
    candidate_seed = deterministic_candidate_seed(
        master_seed,
        attempt_index,
        namespace=str(support["retry_seed_namespace"]),
    )
    fluid_config, property_metadata, physics = _fluid_state(
        master_seed, synthetic, dependencies["validation"]
    )
    started = time.perf_counter()
    candidate = make_field_conditioned_realization(
        sand_probability_base=dependencies["arrays"]["sand_probability"],
        porosity_base=dependencies["arrays"]["porosity"],
        rgt_base=dependencies["arrays"]["rgt"],
        strat_fraction_base=dependencies["arrays"]["strat_fraction"],
        reservoir_mask_base=dependencies["arrays"]["reservoir_mask"],
        elastic_background_base=dependencies["arrays"]["elastic_background"],
        elastic_blend_weight_base=dependencies["arrays"]["elastic_blend_weight"],
        reservoir_model=dependencies["reservoir_model"],
        seed=candidate_seed,
        geology_config=synthetic["geology"],
        fluid_config=fluid_config,
        fluid_calibration=dependencies["calibration"],
        fluid_property_metadata=property_metadata,
        depth_m=dependencies["depth_m"],
    )
    report = evaluate_candidate_support(
        elastic=candidate.elastic,
        elastic_brine=candidate.elastic_brine,
        shaliness=candidate.delta,
        plume_mask=candidate.plume_mask,
        co2_saturation=candidate.co2_saturation,
        time_ms=dependencies["arrays"]["time_ms"],
        fluid_metadata=candidate.metadata["fluid"],
        calibration=dependencies["calibration"],
        physics=physics,
        contract=support_contract_from_mapping(
            support, dependencies["calibration"]
        ),
    )
    deformation = candidate.metadata["deformation"]
    faults = deformation["faults"]
    statistics = report.statistics
    summary = {
        "member_master_seed": int(master_seed),
        "attempt_index": int(attempt_index),
        "candidate_seed": int(candidate_seed),
        "accepted": bool(report.accepted),
        "rejection_reasons": ";".join(report.rejection_reasons),
        "elapsed_seconds": time.perf_counter() - started,
        "candidate_hash": _candidate_hash(candidate),
        "plume_depth_minimum": statistics["depth_m"]["minimum"],
        "plume_depth_median": statistics["depth_m"]["median"],
        "plume_depth_maximum": statistics["depth_m"]["maximum"],
        "plume_depth_span": (
            statistics["depth_m"]["maximum"]
            - statistics["depth_m"]["minimum"]
        ),
        "depth_outside_pixels": statistics["depth_m"]["outside_domain_pixels"],
        "overall_support_coverage": statistics["overall_support_coverage"],
        "plume_effective_porosity_median": statistics["effective_porosity"]["median"],
        "plume_shaliness_median": statistics["shaliness"]["median"],
        "plume_saturation_median": statistics["saturation"]["median"],
        "fold_amplitude_total": abs(deformation["fold_amplitude_1_samples"])
        + abs(deformation["fold_amplitude_2_samples"]),
        "maximum_absolute_fault_throw": max(
            (abs(float(fault["throw_samples"])) for fault in faults), default=0.0
        ),
        "fault_count": int(deformation["fault_count"]),
        "maximum_absolute_plume_vertical_displacement": float(
            np.max(
                np.abs(candidate.vertical_displacement[candidate.plume_mask])
            )
        ),
    }
    return candidate, summary


def _distribution_gate(
    initial: pd.DataFrame,
    accepted: pd.DataFrame,
    support: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    settings = support["stress_gate"]
    rows = []
    for metric in settings["required_metrics"]:
        before = initial[metric].to_numpy(dtype=float)
        after = accepted[metric].to_numpy(dtype=float)
        before_width = float(np.quantile(before, 0.95) - np.quantile(before, 0.05))
        after_width = float(np.quantile(after, 0.95) - np.quantile(after, 0.05))
        width_ratio = after_width / max(before_width, 1e-12)
        median_shift = abs(float(np.median(after) - np.median(before))) / max(
            float(np.std(before)), 1e-12
        )
        passed = (
            width_ratio >= float(settings["minimum_p05_p95_width_ratio"])
            and median_shift
            <= float(settings["maximum_absolute_standardized_median_shift"])
        )
        rows.append(
            {
                "metric": metric,
                "initial_p05_p95_width": before_width,
                "accepted_p05_p95_width": after_width,
                "width_ratio": width_ratio,
                "standardized_median_shift": median_shift,
                "passed": bool(passed),
            }
        )
    return rows, all(row["passed"] for row in rows)


def _stress_figures(
    initial: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    gate_rows: list[dict[str, Any]],
    locations: dict[str, Path],
) -> list[dict[str, object]]:
    locations["figures"].mkdir(parents=True, exist_ok=True)
    metrics = [row["metric"] for row in gate_rows]
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics):
        axis.hist(initial[metric], bins=18, alpha=0.55, label="unfiltered attempt 0")
        axis.hist(accepted[metric], bins=18, alpha=0.55, label="accepted final")
        axis.set_title(metric.replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    distribution_path = locations["figures"] / "support_filter_diversity_distributions.png"
    figure.savefig(distribution_path, dpi=220)
    plt.close(figure)

    reason_counts = Counter(
        reason
        for value in rejected["rejection_reasons"].tolist()
        for reason in str(value).split(";")
        if reason
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(list(reason_counts), list(reason_counts.values()))
    axes[0].set_title("Whole-candidate rejection reasons")
    axes[0].set_xlabel("candidate count")
    axes[1].hist(accepted["attempt_index"], bins=np.arange(-0.5, 6.5, 1.0))
    axes[1].set_title("Retries per accepted realization")
    axes[1].set_xlabel("accepted attempt index")
    axes[1].set_ylabel("realizations")
    retry_path = locations["figures"] / "support_rejections_and_retries.png"
    figure.savefig(retry_path, dpi=220)
    plt.close(figure)
    return [_source(distribution_path), _source(retry_path)]


def stress(_: argparse.Namespace) -> None:
    paths, synthetic, support, locations = _contracts()
    dependencies = _load_dependencies(paths, synthetic)
    settings = support["stress_gate"]
    seeds = range(
        int(settings["master_seed_start"]),
        int(settings["master_seed_start"]) + int(settings["candidate_count"]),
    )
    rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, master_seed in enumerate(seeds, start=1):
        for attempt_index in range(int(support["maximum_attempts"])):
            try:
                _, row = _candidate(
                    master_seed, attempt_index, synthetic, dependencies
                )
            except ValueError as error:
                row = {
                    "member_master_seed": master_seed,
                    "attempt_index": attempt_index,
                    "candidate_seed": deterministic_candidate_seed(
                        master_seed,
                        attempt_index,
                        namespace=str(support["retry_seed_namespace"]),
                    ),
                    "accepted": False,
                    "rejection_reasons": "candidate_physical_generation_error",
                    "error": str(error),
                }
            rows.append(row)
            if row["accepted"]:
                accepted_rows.append(row)
                break
        else:
            raise RuntimeError(
                f"Master seed {master_seed} exhausted deterministic retries"
            )
        if index % 10 == 0:
            print(f"stress candidates accepted: {index}/{settings['candidate_count']}", flush=True)
    frame = pd.DataFrame(rows)
    initial = frame.loc[frame["attempt_index"].eq(0)].copy()
    accepted = pd.DataFrame(accepted_rows)
    rejected = frame.loc[~frame["accepted"]].copy()
    initial_acceptance = float(initial["accepted"].mean())
    final_acceptance = len(accepted) / int(settings["candidate_count"])
    diversity, diversity_passed = _distribution_gate(initial, accepted, support)
    gates = {
        "initial_acceptance_rate": initial_acceptance
        >= float(settings["minimum_initial_acceptance_rate"]),
        "final_acceptance_rate": final_acceptance
        >= float(settings["minimum_final_acceptance_rate"]),
        "diversity_retained": diversity_passed,
        "known_deep_tail_rejected": not bool(
            initial.set_index("member_master_seed").loc[3400046, "accepted"]
        ),
        "known_shallow_tail_rejected": not bool(
            initial.set_index("member_master_seed").loc[3400055, "accepted"]
        ),
    }
    locations["stress"].mkdir(parents=True, exist_ok=True)
    frame.to_csv(locations["stress"] / "all_candidate_attempts.csv", index=False)
    initial.to_csv(locations["stress"] / "unfiltered_attempt0.csv", index=False)
    accepted.to_csv(locations["stress"] / "accepted_final.csv", index=False)
    rejected.to_csv(locations["stress"] / "rejected_candidates.csv", index=False)
    figures = _stress_figures(initial, accepted, rejected, diversity, locations)
    reasons = Counter(
        reason
        for value in rejected["rejection_reasons"].tolist()
        for reason in str(value).split(";")
        if reason
    )
    report = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "candidate_count": int(settings["candidate_count"]),
        "candidate_attempts": len(frame),
        "initial_acceptance_rate": initial_acceptance,
        "initial_rejection_rate": 1.0 - initial_acceptance,
        "final_acceptance_rate": final_acceptance,
        "candidate_attempt_acceptance_rate": len(accepted) / len(frame),
        "candidate_attempt_rejection_rate": len(rejected) / len(frame),
        "rejection_reasons": dict(reasons),
        "retries_per_accepted_realization": {
            str(int(key)): int(value)
            for key, value in accepted["attempt_index"].value_counts().sort_index().items()
        },
        "diversity": diversity,
        "gates": gates,
        "wall_time_seconds": time.perf_counter() - started,
        "tables": {
            name: _source(locations["stress"] / name)
            for name in (
                "all_candidate_attempts.csv",
                "unfiltered_attempt0.csv",
                "accepted_final.csv",
                "rejected_candidates.csv",
            )
        },
        "figures": figures,
        "support_contract": _source(
            REPOSITORY / "configs" / "revision331_support_acceptance.yaml"
        ),
        "failed_corpus_preserved": _source(
            locations["failed_stage02"] / "manifest.json"
        ),
    }
    locations["reports"].mkdir(parents=True, exist_ok=True)
    write_json(locations["reports"] / "stress_gate.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Support-aware stress/diversity gate failed")


def _archive_candidate_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in (
            "elastic",
            "elastic_brine",
            "delta",
            "porosity",
            "rgt",
            "plume_mask",
            "co2_saturation",
        ):
            array = np.ascontiguousarray(archive[name])
            digest.update(str(array.dtype).encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    return digest.hexdigest()


def bounded(args: argparse.Namespace) -> None:
    paths, synthetic, support, locations = _contracts()
    stress_report_path = locations["reports"] / "stress_gate.json"
    if not stress_report_path.exists():
        raise FileNotFoundError("Run the stress gate before bounded generation")
    stress_report = json.loads(stress_report_path.read_text(encoding="utf-8"))
    if stress_report.get("status") != "passed":
        raise RuntimeError("Stress/diversity gate did not pass")
    settings = support["bounded_gate"]
    count = int(settings["final_realization_count"])
    synthetic["stage"].update(
        {
            "geology_realization_count": count,
            "realization_count": count,
            "realization_id_offset": int(settings["realization_id_offset"]),
            "member_master_seeds": [
                int(value) for value in settings["member_master_seeds"]
            ],
        }
    )
    manifest = generate_stage02_dataset(
        config=synthetic,
        paths=paths,
        output_directory=locations["stage02"],
        workers=int(args.workers),
        resume=bool(args.resume),
    )
    records = manifest["records"]
    accepted = [record["support_aware_acceptance"] for record in records]
    retry_members = [
        record for record in records if record["support_aware_acceptance"]["attempt_index"] > 0
    ]
    dependencies = _load_dependencies(paths, synthetic)
    reproducibility = []
    for record in retry_members:
        support_record = record["support_aware_acceptance"]
        candidate, summary = _candidate(
            int(support_record["member_master_seed"]),
            int(support_record["attempt_index"]),
            synthetic,
            dependencies,
        )
        path = locations["stage02"] / record["file"]
        reproducibility.append(
            {
                "realization_id": int(record["realization_id"]),
                "member_master_seed": int(support_record["member_master_seed"]),
                "attempt_index": int(support_record["attempt_index"]),
                "candidate_seed": int(support_record["candidate_seed"]),
                "regenerated_candidate_seed": int(summary["candidate_seed"]),
                "candidate_hash": _candidate_hash(candidate),
                "archive_candidate_hash": _archive_candidate_hash(path),
                "bitwise_reproducible": _candidate_hash(candidate)
                == _archive_candidate_hash(path),
            }
        )
    import run_revision31_fluid_gate as revision31

    round_trip = revision31._all_realization_round_trip(
        locations["stage02"], synthetic
    )
    prior_failures = {
        int(record["support_aware_acceptance"]["member_master_seed"]): record[
            "support_aware_acceptance"
        ]["rejection_history"]
        for record in records
    }
    class_coverages = [
        float(row["coverage"])
        for item in accepted
        for row in item["accepted_support"]["statistics"]["classes"]
    ]
    gates = {
        "realization_count": len(records) == count,
        "all_accepted_plume_pixels_inside_depth_domain": all(
            item["accepted_support"]["statistics"]["depth_m"][
                "outside_domain_pixels"
            ]
            == 0
            for item in accepted
        ),
        "all_accepted_effective_porosity_inside_domain": all(
            item["accepted_support"]["statistics"]["effective_porosity"][
                "outside_domain_pixels"
            ]
            == 0
            for item in accepted
        ),
        "overall_support_at_least_95_percent": all(
            item["accepted_support"]["statistics"]["overall_support_coverage"]
            >= float(support["minimum_overall_coverage"])
            for item in accepted
        ),
        "all_represented_classes_at_least_90_percent": min(class_coverages)
        >= float(support["minimum_class_coverage"]),
        "candidate_b_physical": all(
            item["accepted_support"]["statistics"][
                "all_dry_frame_states_physical"
            ]
            and item["accepted_support"]["statistics"][
                "all_elastic_states_physical"
            ]
            for item in accepted
        ),
        "exact_zero_outside_plume": all(
            item["accepted_support"]["statistics"]["maximum_outside_plume_change"]
            == 0.0
            for item in accepted
        ),
        "fixed_shear": all(
            item["accepted_support"]["statistics"][
                "maximum_fixed_shear_error_gpa"
            ]
            <= float(support["maximum_fixed_shear_error_gpa"])
            for item in accepted
        ),
        "known_depth_and_multivariate_stress_cases_retried": all(
            len(prior_failures[seed]) > 0
            for seed in (3400046, 3400055, 3400097, 3400009)
        ),
        "deterministic_retry_reproduction": bool(reproducibility)
        and all(row["bitwise_reproducible"] for row in reproducibility),
        "round_trip_canonical_bands": round_trip["bands_in_order"]
        == ["near", "mid", "far"],
        "round_trip_precision": round_trip["maximum_relative_rmse"] <= 1e-6,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "stage02_manifest": _source(locations["stage02"] / "manifest.json"),
        "rejected_candidate_manifest": _source(
            locations["stage02"] / "rejected_candidates.json"
        ),
        "accepted_realizations": len(records),
        "rejected_candidate_attempts": sum(
            item["rejected_candidate_count"] for item in accepted
        ),
        "minimum_overall_support": min(
            item["accepted_support"]["statistics"]["overall_support_coverage"]
            for item in accepted
        ),
        "minimum_class_support": min(class_coverages),
        "accepted_depth_range_m": [
            min(
                item["accepted_support"]["statistics"]["depth_m"]["minimum"]
                for item in accepted
            ),
            max(
                item["accepted_support"]["statistics"]["depth_m"]["maximum"]
                for item in accepted
            ),
        ],
        "retry_reproducibility": reproducibility,
        "round_trip": round_trip,
        "gates": gates,
        "no_stage03_built": True,
        "no_training_run": True,
    }
    locations["reports"].mkdir(parents=True, exist_ok=True)
    write_json(locations["reports"] / "bounded_support_gate.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Bounded support-aware gate failed")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("stress").set_defaults(function=stress)
    bounded_command = commands.add_parser("bounded")
    bounded_command.add_argument("--workers", type=int, default=1)
    bounded_command.add_argument("--resume", action="store_true")
    bounded_command.set_defaults(function=bounded)
    return root


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
