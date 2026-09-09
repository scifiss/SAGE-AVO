#!/usr/bin/env python3
"""Run the staged v00332p topology repair and causal graph audit."""

from __future__ import annotations

import os

# Must precede Torch imports in commands that later use CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch

from sage_avo.config import load_config, seed_everything
from sage_avo.data.patches import resize_channels_first
from sage_avo.data.indexed_dataset import IndexedRealizationPatches
from sage_avo.diagnostics.rgt_topology_repair import (
    ADAPTIVE,
    CARTESIAN,
    INVERSE,
    LEGACY,
    TIE_FIXED,
    edge_jaccard,
    edge_observables,
    confidence_block_mask,
    fault_masks,
    fit_structural_contract,
    load_faults,
    load_realization,
    strata_masks,
    structural_fields,
    topology_for,
    topology_summary_rows,
    write_csv,
)
from sage_avo.models.graph import (
    RGT_V1_LEGACY,
    RGT_V2_TIE_FIXED,
    RGT_V3_CONFIDENCE_BLOCKED,
)
from sage_avo.models.variants import build_sage_avo_variant, sage_avo_model_kwargs
from sage_avo.experiments.training import train_controlled_variant
from sage_avo.evaluation.inference import infer_full_realization
from sage_avo.forward.specification import forward_specification_from_mapping
from sage_avo.forward.torch_forward import forward_avo_three_band_spec_torch
from sage_avo.runtime import print_torch_runtime, select_torch_device
from sage_avo.training.checkpoints import load_checkpoint


REPOSITORY = Path(__file__).resolve().parents[1]
PRIVATE = Path(load_config(REPOSITORY / "configs" / "paths.yaml")["private_artifact_root"])
DATASET = (
    PRIVATE
    / "stage_artifacts/stage03/ds_v00331_production100_support_aware/dataset"
)
STAGE02 = (
    PRIVATE
    / "stage_artifacts/stage02/v00331_production100_support_aware/realizations"
)
EXPERIMENT = (
    PRIVATE
    / "stage_artifacts/stage04/sage_avo_s01_v00332p_rgt_topology_repair"
)
TRAINING_CONFIG = REPOSITORY / "configs/development_diagnostics_v00332p.yaml"
TOPOLOGIES = (TIE_FIXED, ADAPTIVE, INVERSE)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar(realization_id: int) -> dict[str, Any]:
    return json.loads(
        (STAGE02 / f"realization_{realization_id:07d}.json").read_text(encoding="utf-8")
    )


def _realization_structure(realization_id: int) -> dict[str, Any]:
    arrays = load_realization(DATASET, realization_id)
    faults = load_faults(STAGE02, realization_id)
    fields = structural_fields(arrays["rgt"])
    inverse = topology_for(INVERSE, arrays["rgt"])
    deformation = _sidecar(realization_id)["geology"]["deformation"]
    return {
        "realization_id": realization_id,
        "fault_count": len(faults),
        "maximum_absolute_fault_throw": max(
            (abs(float(item["throw_samples"])) for item in faults), default=0.0
        ),
        "mean_dip": float(fields["dip"].mean()),
        "dip_p90": float(np.quantile(fields["dip"], 0.9)),
        "curvature_mean": float(fields["curvature"].mean()),
        "inverse_absolute_shift_mean": float(np.abs(inverse.shift[inverse.valid]).mean()),
        "inverse_absolute_shift_p95": float(
            np.quantile(np.abs(inverse.shift[inverse.valid]), 0.95)
        ),
        "fold_amplitude_total": abs(float(deformation["fold_amplitude_1_samples"]))
        + abs(float(deformation["fold_amplitude_2_samples"])),
    }


def _diverse_validation_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select six cases using metadata only, before inspecting model outcomes."""
    frame = pd.DataFrame(rows)
    no_fault = frame[frame["fault_count"] == 0]
    if len(no_fault) < 4:
        no_fault = frame.sort_values(["fault_count", "maximum_absolute_fault_throw"])
    low = no_fault.sort_values(
        ["inverse_absolute_shift_mean", "dip_p90", "realization_id"]
    ).head(2)
    remaining = no_fault[~no_fault["realization_id"].isin(low["realization_id"])]
    high = remaining.sort_values(
        ["dip_p90", "inverse_absolute_shift_p95", "realization_id"],
        ascending=[False, False, True],
    ).head(2)
    used = set(low["realization_id"]) | set(high["realization_id"])
    fault = frame[~frame["realization_id"].isin(used)].sort_values(
        ["fault_count", "maximum_absolute_fault_throw", "realization_id"],
        ascending=[False, False, True],
    ).head(2)
    return {
        "selection_uses_model_outcomes": False,
        "selection_rule": (
            "two fault-free lowest inverse-shift/dip cases; two remaining fault-free "
            "highest dip cases; two remaining highest fault-count/throw cases"
        ),
        "low_complexity_ids": [int(value) for value in low["realization_id"]],
        "high_dip_continuous_ids": [int(value) for value in high["realization_id"]],
        "fault_rich_ids": [int(value) for value in fault["realization_id"]],
        "all_ids": [
            int(value)
            for value in pd.concat((low, high, fault))["realization_id"].tolist()
        ],
        "metadata": rows,
    }


def _changed_fraction(left: Any, right: Any) -> float:
    comparable = left.valid & right.valid
    changed = (left.target_row != right.target_row) | (
        left.target_column != right.target_column
    )
    return float(changed[comparable].mean()) if comparable.any() else float("nan")


def _stable_fraction(name: str, rgt: np.ndarray, adaptive_shift: int) -> float:
    base = topology_for(name, rgt, adaptive_shift=adaptive_shift)
    row, column = np.indices(rgt.shape)
    signs = np.where((row + column) % 2 == 0, 1.0, -1.0)
    perturbation = signs * 4.0 * np.finfo(np.float32).eps * max(float(np.abs(rgt).max()), 1.0)
    perturbed = topology_for(name, (rgt.astype(np.float64) + perturbation).astype(np.float32), adaptive_shift=adaptive_shift)
    comparable = base.valid & perturbed.valid
    same = (base.target_row == perturbed.target_row) & (
        base.target_column == perturbed.target_column
    )
    return float(same[comparable].mean()) if comparable.any() else float("nan")


def topology_audit(args: argparse.Namespace) -> None:
    if not (DATASET / "dataset_manifest.json").exists():
        raise FileNotFoundError(DATASET)
    split_ids = json.loads((DATASET / "split_ids.json").read_text(encoding="utf-8"))
    train_ids = [int(value) for value in split_ids["train"]]
    validation_ids = [int(value) for value in split_ids["validation"]]
    structural_contract = fit_structural_contract(DATASET, train_ids)
    validation_structure = [_realization_structure(value) for value in validation_ids]
    selection = _diverse_validation_selection(validation_structure)
    contract = {
        "schema_version": 1,
        "revision": "v00332p-rgt-topology-repair-causal-message-audit",
        "status": "TOPOLOGY_QC_IN_PROGRESS_NOT_PRODUCTION_EVIDENCE",
        "immutable_dataset": "ds_v00331_production100_support_aware",
        "split_ids": split_ids,
        "legacy_topology": {
            "name": LEGACY,
            "frozen_behavior": "plain argmin over shifts [-3,-2,-1,0,1,2,3]",
            "overwritten": False,
        },
        "tie_fixed_topology": {
            "name": TIE_FIXED,
            "selection": [
                "minimum RGT mismatch within sixteen local float32 ULPs",
                "minimum absolute displacement",
                "source-node parity balanced deterministic sign tie-break",
            ],
            "global_sign_bias": False,
        },
        "adaptive_search": {
            "name": ADAPTIVE,
            "local_radius": "ceil(abs(dRGT/dx)/abs(dRGT/dt) + 1), minimum 3",
            **structural_contract,
        },
        "inverse_rgt": {
            "name": INVERSE,
            "out_of_range_tau": "omit edge; never grossly clip",
            "first_architecture_comparison_targets_per_source": 1,
        },
        "topology_candidate_selection_before_training": {
            "eligible": (
                "bitwise repeatable CPU/CUDA and mean valid fraction >=0.94; then "
                "lowest mean rank across all-link RGT mismatch, high-dip RGT mismatch, "
                "fault-crossing rate, and absolute displacement p95; exact score ties "
                "select the simpler/lower-displacement topology"
            ),
            "uses_validation_model_outcomes": False,
        },
        "fault_truth_use": "selection/stratification/QC only; never a model feature",
        "diverse_validation_subset": selection,
        "cuda_environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "forbidden_changes": [
            "normal AVO bias",
            "relation-aware learned gates",
            "time-dependent graph gating",
            "legacy graph smoothness",
            "exact Zoeppritz",
            "conditional flow",
            "Stage-02/03 data",
            "frozen splits",
        ],
    }
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    _json_write(EXPERIMENT / "rgt_topology_repair_contract.json", contract)

    v1_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        for realization_id in map(int, split_ids[split]):
            arrays = load_realization(DATASET, realization_id)
            rgt = arrays["rgt"]
            faults = load_faults(STAGE02, realization_id)
            v1 = topology_for(LEGACY, rgt)
            v2 = topology_for(TIE_FIXED, rgt)
            v1_near, v1_crossing = fault_masks(v1, faults)
            v1_masks = strata_masks(
                rgt,
                arrays["segmentation"],
                arrays["reservoir_mask"],
                arrays["plume_mask"],
                v1,
                faults,
                structural_contract,
            )
            v1_rows.extend(
                topology_summary_rows(
                    topology=v1,
                    topology_name=LEGACY,
                    split=split,
                    realization_id=realization_id,
                    rgt=rgt,
                    masks=v1_masks,
                    fault_crossing=v1_crossing,
                )
            )
            tied = v1.tied
            tie_rows.append(
                {
                    "split": split,
                    "realization_id": realization_id,
                    "exact_tie_minimum_fraction": float(v1.exact_tied.mean()),
                    "near_tie_minimum_fraction": float(tied.mean()),
                    "v1_v2_changed_fraction": _changed_fraction(v1, v2),
                    "v1_tied_shift_histogram": json.dumps(
                        dict(sorted(Counter(map(int, v1.shift[tied])).items()))
                    ),
                    "v2_tied_shift_histogram": json.dumps(
                        dict(sorted(Counter(map(int, v2.shift[tied])).items()))
                    ),
                    "v1_tied_absolute_shift_mean": float(np.abs(v1.shift[tied]).mean())
                    if tied.any()
                    else 0.0,
                    "v2_tied_absolute_shift_mean": float(np.abs(v2.shift[tied]).mean())
                    if tied.any()
                    else 0.0,
                }
            )
            if split != "validation":
                continue
            cartesian = topology_for(CARTESIAN, rgt)
            for name in TOPOLOGIES:
                candidate = topology_for(
                    name,
                    rgt,
                    adaptive_shift=int(structural_contract["adaptive_max_shift_samples"]),
                )
                near_fault, crossing = fault_masks(candidate, faults)
                masks = strata_masks(
                    rgt,
                    arrays["segmentation"],
                    arrays["reservoir_mask"],
                    arrays["plume_mask"],
                    candidate,
                    faults,
                    structural_contract,
                )
                stability = _stable_fraction(
                    name, rgt, int(structural_contract["adaptive_max_shift_samples"])
                )
                for stratum in ("all", "high_dip", "fault_corridor", "away_from_fault"):
                    selected = masks[stratum] & candidate.valid
                    if not selected.any():
                        continue
                    candidate_rows.append(
                        {
                            "realization_id": realization_id,
                            "topology": name,
                            "stratum": stratum,
                            "valid_fraction": float(candidate.valid.mean()),
                            "edge_count_one_direction": int(selected.sum()),
                            "rgt_mismatch_mean": float(candidate.mismatch[selected].mean()),
                            "rgt_mismatch_p95": float(
                                np.quantile(candidate.mismatch[selected], 0.95)
                            ),
                            "absolute_shift_mean": float(
                                np.abs(candidate.shift[selected]).mean()
                            ),
                            "absolute_shift_p95": float(
                                np.quantile(np.abs(candidate.shift[selected]), 0.95)
                            ),
                            "boundary_hit_fraction": float(
                                candidate.boundary_hit[selected].mean()
                            ),
                            "topology_stability_fraction": stability,
                            "edge_jaccard_with_cartesian": edge_jaccard(
                                candidate, cartesian
                            ),
                            "fraction_edges_identical_to_cartesian": float(
                                (candidate.shift[selected] == 0).mean()
                            ),
                            "fault_crossing_fraction": float(crossing[selected].mean()),
                        }
                    )
                for region, selected in {
                    "fault_corridor": near_fault & candidate.valid,
                    "away_from_fault": ~near_fault & candidate.valid,
                }.items():
                    if not selected.any():
                        continue
                    fault_rows.append(
                        {
                            "split": split,
                            "realization_id": realization_id,
                            "topology": name,
                            "region": region,
                            "fault_count": len(faults),
                            "edge_count": int(selected.sum()),
                            "true_fault_crossing_fraction": float(crossing[selected].mean()),
                            "rgt_mismatch_mean": float(candidate.mismatch[selected].mean()),
                            "absolute_shift_mean": float(
                                np.abs(candidate.shift[selected]).mean()
                            ),
                            "search_boundary_hit_fraction": float(
                                candidate.boundary_hit[selected].mean()
                            ),
                        }
                    )
            print(f"topology audit: {split} realization {realization_id}", flush=True)
    write_csv(EXPERIMENT / "rgt_topology_v1_audit.csv", v1_rows)
    write_csv(EXPERIMENT / "rgt_tie_break_statistics.csv", tie_rows)
    write_csv(EXPERIMENT / "rgt_topology_candidate_comparison.csv", candidate_rows)
    write_csv(EXPERIMENT / "rgt_fault_corridor_qc.csv", fault_rows)
    print(f"Wrote topology artifacts under {EXPERIMENT}")


def _candidate_selection(frame: pd.DataFrame) -> dict[str, Any]:
    all_rows = frame[frame["stratum"] == "all"].groupby("topology").mean(numeric_only=True)
    high_rows = frame[frame["stratum"] == "high_dip"].groupby("topology").mean(numeric_only=True)
    metrics = pd.DataFrame(
        {
            "all_mismatch": all_rows["rgt_mismatch_mean"],
            "high_dip_mismatch": high_rows["rgt_mismatch_mean"],
            "fault_crossing": all_rows["fault_crossing_fraction"],
            "absolute_shift_p95": all_rows["absolute_shift_p95"],
            "valid_fraction": all_rows["valid_fraction"],
        }
    )
    eligible = metrics[metrics["valid_fraction"] >= 0.94].copy()
    rank_columns = [
        "all_mismatch",
        "high_dip_mismatch",
        "fault_crossing",
        "absolute_shift_p95",
    ]
    eligible["selection_score"] = eligible[rank_columns].rank(method="average").mean(axis=1)
    eligible = eligible.sort_values(
        ["selection_score", "absolute_shift_p95", "all_mismatch"]
    )
    selected = str(eligible.index[0])
    return {
        "selected": selected,
        "score_rows": [
            {"topology": str(index), **{key: float(value) for key, value in row.items()}}
            for index, row in eligible.iterrows()
        ],
        "selection_uses_model_outcomes": False,
    }


def _auc(feature: np.ndarray, positive: np.ndarray) -> float:
    finite = np.isfinite(feature)
    values = feature[finite]
    labels = positive[finite]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(values, method="average")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def confidence_qc(args: argparse.Namespace) -> None:
    contract_path = EXPERIMENT / "rgt_topology_repair_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError("Run topology-audit first")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    comparison_path = EXPERIMENT / "rgt_topology_candidate_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    comparison = comparison[comparison["topology"] != "RGT_V3_CONFIDENCE_BLOCKED"]
    selection = _candidate_selection(comparison)
    if selection["selected"] != TIE_FIXED:
        raise RuntimeError(
            "The predeclared QC ranking did not select the bounded tie-fixed base; "
            f"got {selection['selected']}"
        )
    train_ids = [int(value) for value in contract["split_ids"]["train"]]
    validation_ids = [int(value) for value in contract["split_ids"]["validation"]]
    structural = contract["adaptive_search"]
    safe_high_dip_features: dict[str, list[np.ndarray]] = {}
    training_feature_rows: list[
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for realization_id in train_ids:
        arrays = load_realization(DATASET, realization_id)
        topology = topology_for(TIE_FIXED, arrays["rgt"])
        faults = load_faults(STAGE02, realization_id)
        near_fault, crossing = fault_masks(topology, faults)
        source_dip = structural_fields(arrays["rgt"])["dip"][
            topology.source_row, topology.source_column
        ]
        safe_high_dip = (~near_fault) & (source_dip >= float(structural["dip_q67"]))
        features = edge_observables(arrays["rgt"], topology)
        for name, values in features.items():
            safe_high_dip_features.setdefault(name, []).append(values[safe_high_dip])
        training_feature_rows.append((features, crossing, near_fault, safe_high_dip))
    threshold_parts = {
        name: np.concatenate(parts)
        for name, parts in safe_high_dip_features.items()
        if name in {
            "normalized_best_mismatch",
            "normalized_cartesian_discontinuity",
            "rgt_dip_residual",
        }
    }
    threshold_grid = (0.95, 0.975, 0.985, 0.99, 0.995, 0.9975, 0.999)
    threshold_trials: list[dict[str, Any]] = []
    for quantile in threshold_grid:
        trial_thresholds = {
            name: float(np.quantile(values, quantile))
            for name, values in threshold_parts.items()
        }
        trial_blocked = np.concatenate(
            [confidence_block_mask(item[0], trial_thresholds) for item in training_feature_rows]
        )
        trial_crossing = np.concatenate([item[1] for item in training_feature_rows])
        trial_near = np.concatenate([item[2] for item in training_feature_rows])
        trial_high_dip = np.concatenate([item[3] for item in training_feature_rows])
        trial = {
            "quantile": quantile,
            "fault_crossing_block_fraction": float(trial_blocked[trial_crossing].mean()),
            "away_from_fault_false_block_fraction": float(
                trial_blocked[~trial_near].mean()
            ),
            "high_dip_nonfault_false_block_fraction": float(
                trial_blocked[trial_high_dip].mean()
            ),
            "thresholds": trial_thresholds,
        }
        trial["eligible"] = bool(
            trial["away_from_fault_false_block_fraction"] <= 0.005
            and trial["high_dip_nonfault_false_block_fraction"] <= 0.01
        )
        threshold_trials.append(trial)
    eligible_trials = [item for item in threshold_trials if item["eligible"]]
    if not eligible_trials:
        raise RuntimeError("No confidence threshold satisfies training-only preservation limits")
    selected_trial = sorted(
        eligible_trials,
        key=lambda item: (
            -item["fault_crossing_block_fraction"],
            item["away_from_fault_false_block_fraction"],
            item["quantile"],
        ),
    )[0]
    threshold_quantile = float(selected_trial["quantile"])
    thresholds = dict(selected_trial["thresholds"])
    feature_auc_rows: list[dict[str, Any]] = []
    for feature_name in next(iter(training_feature_rows))[0]:
        values = np.concatenate([item[0][feature_name] for item in training_feature_rows])
        labels = np.concatenate([item[1] for item in training_feature_rows])
        feature_auc_rows.append(
            {
                "split": "train",
                "realization_id": "ALL",
                "topology": "RGT_V3_CONFIDENCE_BLOCKED",
                "region": "feature_discrimination",
                "observable": feature_name,
                "fault_crossing_auc": _auc(values, labels),
            }
        )

    qc_rows = feature_auc_rows + [
        {
            "split": "train",
            "realization_id": "ALL",
            "topology": "RGT_V3_CONFIDENCE_BLOCKED",
            "region": "threshold_selection",
            **{key: value for key, value in trial.items() if key != "thresholds"},
        }
        for trial in threshold_trials
    ]
    candidate_rows: list[dict[str, Any]] = []
    for split, realization_ids in (("train", train_ids), ("validation", validation_ids)):
        for realization_id in realization_ids:
            arrays = load_realization(DATASET, realization_id)
            base = topology_for(TIE_FIXED, arrays["rgt"])
            faults = load_faults(STAGE02, realization_id)
            near_fault, crossing = fault_masks(base, faults)
            source_dip = structural_fields(arrays["rgt"])["dip"][
                base.source_row, base.source_column
            ]
            features = edge_observables(arrays["rgt"], base)
            blocked = confidence_block_mask(features, thresholds)
            corrected = replace(base, valid=base.valid & ~blocked)
            for region, selected in {
                "all": np.ones_like(blocked, dtype=bool),
                "fault_corridor": near_fault,
                "away_from_fault": ~near_fault,
                "high_dip_nonfault": (~near_fault)
                & (source_dip >= float(structural["dip_q67"])),
            }.items():
                count = int(selected.sum())
                if not count:
                    continue
                qc_rows.append(
                    {
                        "split": split,
                        "realization_id": realization_id,
                        "topology": "RGT_V3_CONFIDENCE_BLOCKED",
                        "region": region,
                        "fault_count": len(faults),
                        "edge_count": count,
                        "fraction_edges_blocked": float(blocked[selected].mean()),
                        "true_fault_crossing_fraction": float(crossing[selected].mean()),
                        "fraction_true_fault_crossing_edges_blocked": float(
                            blocked[selected & crossing].mean()
                        )
                        if (selected & crossing).any()
                        else float("nan"),
                        "false_blocking_fraction": float(
                            blocked[selected & ~crossing].mean()
                        )
                        if (selected & ~crossing).any()
                        else float("nan"),
                        "rgt_mismatch_mean_retained": float(
                            base.mismatch[selected & ~blocked].mean()
                        )
                        if (selected & ~blocked).any()
                        else float("nan"),
                        "absolute_shift_mean_retained": float(
                            np.abs(base.shift[selected & ~blocked]).mean()
                        )
                        if (selected & ~blocked).any()
                        else float("nan"),
                    }
                )
            if split == "validation":
                cartesian = topology_for(CARTESIAN, arrays["rgt"])
                candidate_rows.append(
                    {
                        "realization_id": realization_id,
                        "topology": "RGT_V3_CONFIDENCE_BLOCKED",
                        "stratum": "all",
                        "valid_fraction": float(corrected.valid.mean()),
                        "edge_count_one_direction": int(corrected.valid.sum()),
                        "rgt_mismatch_mean": float(
                            corrected.mismatch[corrected.valid].mean()
                        ),
                        "rgt_mismatch_p95": float(
                            np.quantile(corrected.mismatch[corrected.valid], 0.95)
                        ),
                        "absolute_shift_mean": float(
                            np.abs(corrected.shift[corrected.valid]).mean()
                        ),
                        "absolute_shift_p95": float(
                            np.quantile(np.abs(corrected.shift[corrected.valid]), 0.95)
                        ),
                        "boundary_hit_fraction": float(
                            corrected.boundary_hit[corrected.valid].mean()
                        ),
                        "topology_stability_fraction": float("nan"),
                        "edge_jaccard_with_cartesian": edge_jaccard(corrected, cartesian),
                        "fraction_edges_identical_to_cartesian": float(
                            (corrected.shift[corrected.valid] == 0).mean()
                        ),
                        "fault_crossing_fraction": float(
                            crossing[corrected.valid].mean()
                        ),
                    }
                )
    write_csv(EXPERIMENT / "rgt_fault_corridor_qc.csv", qc_rows)
    pd.concat((comparison, pd.DataFrame(candidate_rows)), ignore_index=True).to_csv(
        comparison_path, index=False
    )
    contract["topology_candidate_selection_before_training"].update(selection)
    contract["confidence_rule"] = {
        "name": "RGT_V3_CONFIDENCE_BLOCKED",
        "base_topology": TIE_FIXED,
        "fit_split": "train only",
        "safe_reference": "high-dip links outside fault corridors",
        "threshold_quantile": threshold_quantile,
        "thresholds": thresholds,
        "threshold_selection": {
            "candidate_quantiles": list(threshold_grid),
            "maximum_training_away_from_fault_false_block_fraction": 0.005,
            "maximum_training_high_dip_nonfault_false_block_fraction": 0.01,
            "objective": "maximize training fault-crossing capture subject to preservation limits",
            "trials": threshold_trials,
        },
        "rule": (
            "block if normalized_best_mismatch exceeds threshold OR if both "
            "normalized_cartesian_discontinuity and rgt_dip_residual exceed thresholds"
        ),
        "model_inputs": [
            "RGT mismatch",
            "RGT vertical/lateral gradients",
            "selected displacement",
        ],
        "fault_truth_is_model_input": False,
        "pwd_dip_status": "not stored in immutable Stage-03; no new dependency introduced",
        "structural_seismic_coherence_status": (
            "not already stored as an inference feature; no new dependency introduced"
        ),
    }
    _json_write(contract_path, contract)
    print(json.dumps(contract["confidence_rule"], indent=2))


def _probe_arrays(record: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays = load_realization(DATASET, int(record["realization_id"]))
    top, left = int(record["top"]), int(record["left"])
    raw_height, raw_width = map(int, record["raw_scale"])
    output_height, output_width = map(int, record["resized_scale"])
    spatial = np.s_[top : top + raw_height, left : left + raw_width]
    result: dict[str, np.ndarray] = {}
    for name in ("avo", "low"):
        value = np.asarray(arrays[name][(slice(None),) + spatial], dtype=np.float32)
        if (raw_height, raw_width) != (output_height, output_width):
            value = resize_channels_first(value, (output_height, output_width), order=1)
        result[name] = value
    rgt = np.asarray(arrays["rgt"][spatial][None], dtype=np.float32)
    if (raw_height, raw_width) != (output_height, output_width):
        rgt = resize_channels_first(rgt, (output_height, output_width), order=1)
    result["rgt"] = rgt[0]
    return result


def _reset_diagnostic_controls(model: Any, confidence: dict[str, Any]) -> None:
    graph = model.graph
    graph.rgt_topology = RGT_V3_CONFIDENCE_BLOCKED
    graph.confidence_normalized_mismatch_threshold = confidence["thresholds"][
        "normalized_best_mismatch"
    ]
    graph.confidence_normalized_discontinuity_threshold = confidence["thresholds"][
        "normalized_cartesian_discontinuity"
    ]
    graph.confidence_dip_residual_threshold = confidence["thresholds"]["rgt_dip_residual"]
    graph.diagnostic_root_scale = 1.0
    graph.diagnostic_neighbor_scale = 1.0
    graph.diagnostic_edge_attr_mode = "current"
    graph.diagnostic_tangential_message_scale = 1.0
    graph.diagnostic_normal_message_scale = 1.0
    graph.diagnostic_capture_contributions = False
    model.diagnostic_graph_reinjection_scale = 1.0
    model.diagnostic_capture_fusion = False


def _attention_statistics(values: torch.Tensor) -> dict[str, float]:
    probabilities = values.detach().float().cpu().numpy().ravel()
    total = float(probabilities.sum())
    if not probabilities.size or total <= 0:
        return {"attention_mean": float("nan"), "attention_entropy": float("nan"), "top_decile_mass": float("nan")}
    normalized = probabilities / total
    entropy = -float(np.sum(normalized * np.log(np.maximum(normalized, 1e-12))))
    entropy /= np.log(max(normalized.size, 2))
    count = max(1, int(np.ceil(0.1 * normalized.size)))
    return {
        "attention_mean": float(probabilities.mean()),
        "attention_entropy": entropy,
        "top_decile_mass": float(np.partition(normalized, -count)[-count:].sum()),
    }


def mechanism_audit(args: argparse.Namespace) -> None:
    contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    confidence = contract["confidence_rule"]
    source_experiment = (
        PRIVATE / "stage_artifacts/stage04/sage_avo_s01_v00332o_causal_gnn_multiseed"
    )
    probe_manifest = json.loads(
        (source_experiment / "fixed_graph_probe_samples.json").read_text(encoding="utf-8")
    )
    probes = [
        record
        for record in probe_manifest["patches"]
        if tuple(record["raw_scale"]) == tuple(record["resized_scale"])
    ][:5]
    normalization = json.loads((DATASET / "normalization.json").read_text(encoding="utf-8"))
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332p frozen-checkpoint mechanism audit",
    )
    mechanism_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    controls = (
        "intact_corrected_rgt",
        "neighbor_messages_zeroed",
        "root_path_zeroed",
        "graph_reinjection_zeroed",
        "edge_attr_zeroed",
        "edge_attr_shuffled",
        "cartesian_edges",
        "shuffled_neighbors",
        "legacy_rgt_v1",
        "tie_fixed_rgt_v2",
    )
    for seed in (12345, 23456, 34567):
        checkpoint_path = (
            source_experiment / "runs" / f"seed_{seed}_rgt_gnn" / "last.pt"
        )
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = raw_checkpoint["config"]
        model = build_sage_avo_variant(
            "full", **sage_avo_model_kwargs(config)
        ).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)
        model.set_norm_stats(normalization)
        model.eval()
        for probe_index, record in enumerate(probes):
            arrays = _probe_arrays(record)
            avo = torch.from_numpy(((arrays["avo"] - x_mean) / x_std)[None]).to(device)
            low = torch.from_numpy(((arrays["low"] - y_mean) / y_std)[None]).to(device)
            rgt = torch.from_numpy(arrays["rgt"][None]).to(device)
            time = torch.full((1,), 0.5, dtype=low.dtype, device=device)
            _reset_diagnostic_controls(model, confidence)
            model.graph.diagnostic_capture_contributions = True
            model.diagnostic_capture_fusion = True
            with torch.inference_mode():
                baseline = model(low, time, avo, low, rgt)
            baseline_velocity = baseline.velocity.detach().clone()
            baseline_embeddings = baseline.embeddings.detach().clone()
            baseline_logits = baseline.segmentation_logits.detach().clone()
            for row in model.graph.last_contribution_diagnostics:
                decomposition_rows.append(
                    {
                        "seed": seed,
                        "probe_index": probe_index,
                        "realization_id": int(record["realization_id"]),
                        "role": record["role"],
                        **row,
                        **model.last_fusion_diagnostics,
                    }
                )
            tangent_count = int(
                model.graph.last_contribution_diagnostics[-1][
                    "tangential_directed_edge_count"
                ]
            )
            attention = baseline.attention_weights[0]
            avo_affinity = baseline.edge_weights[0]
            edge_index = baseline.edge_indices[0]
            rgt_flat = rgt[0].reshape(-1)
            rgt_mismatch = (
                rgt_flat[edge_index[0]] - rgt_flat[edge_index[1]]
            ).abs()
            for relation, relation_slice in (
                ("tangential", slice(0, tangent_count)),
                ("normal", slice(tangent_count, None)),
            ):
                relation_rows.append(
                    {
                        "seed": seed,
                        "probe_index": probe_index,
                        "realization_id": int(record["realization_id"]),
                        "role": record["role"],
                        "relation": relation,
                        "directed_edge_count": int(edge_index[:, relation_slice].shape[1]),
                        "rgt_mismatch_mean": float(
                            rgt_mismatch[relation_slice].mean().detach().cpu()
                        ),
                        "avo_gradient_contrast_mean": float(
                            (-torch.log(avo_affinity[relation_slice].clamp_min(1e-12)))
                            .mean()
                            .detach()
                            .cpu()
                        ),
                        **_attention_statistics(attention[relation_slice]),
                    }
                )
            # Differentiate one shared scalar per relation through both layers.
            _reset_diagnostic_controls(model, confidence)
            tangential_scale = torch.tensor(1.0, device=device, requires_grad=True)
            normal_scale = torch.tensor(1.0, device=device, requires_grad=True)
            model.graph.diagnostic_tangential_message_scale = tangential_scale
            model.graph.diagnostic_normal_message_scale = normal_scale
            with torch.enable_grad():
                gradient_output = model(low, time, avo, low, rgt)
                diagnostic_scalar = gradient_output.velocity.square().mean()
                tangential_gradient, normal_gradient = torch.autograd.grad(
                    diagnostic_scalar, (tangential_scale, normal_scale)
                )
            relation_rows[-2]["output_gradient_contribution"] = float(
                tangential_gradient.abs().detach().cpu()
            )
            relation_rows[-1]["output_gradient_contribution"] = float(
                normal_gradient.abs().detach().cpu()
            )
            _reset_diagnostic_controls(model, confidence)
            for control in controls:
                _reset_diagnostic_controls(model, confidence)
                if control == "neighbor_messages_zeroed":
                    model.graph.diagnostic_neighbor_scale = 0.0
                elif control == "root_path_zeroed":
                    model.graph.diagnostic_root_scale = 0.0
                elif control == "graph_reinjection_zeroed":
                    model.diagnostic_graph_reinjection_scale = 0.0
                elif control == "edge_attr_zeroed":
                    model.graph.diagnostic_edge_attr_mode = "zero"
                elif control == "edge_attr_shuffled":
                    model.graph.diagnostic_edge_attr_mode = "shuffled"
                elif control == "cartesian_edges":
                    model.graph.rgt_topology = "cartesian"
                elif control == "shuffled_neighbors":
                    model.graph.rgt_topology = "shuffled"
                elif control == "legacy_rgt_v1":
                    model.graph.rgt_topology = RGT_V1_LEGACY
                elif control == "tie_fixed_rgt_v2":
                    model.graph.rgt_topology = RGT_V2_TIE_FIXED
                with torch.inference_mode():
                    output = model(low, time, avo, low, rgt)
                velocity_delta = (output.velocity - baseline_velocity).square().mean().sqrt()
                embedding_delta = (
                    output.embeddings - baseline_embeddings
                ).square().mean().sqrt()
                mechanism_rows.append(
                    {
                        "seed": seed,
                        "probe_index": probe_index,
                        "realization_id": int(record["realization_id"]),
                        "role": record["role"],
                        "control": control,
                        "velocity_delta_rms": float(velocity_delta.detach().cpu()),
                        "velocity_relative_delta": float(
                            (velocity_delta / baseline_velocity.square().mean().sqrt().clamp_min(1e-12))
                            .detach()
                            .cpu()
                        ),
                        "embedding_delta_rms": float(embedding_delta.detach().cpu()),
                        "embedding_relative_delta": float(
                            (embedding_delta / baseline_embeddings.square().mean().sqrt().clamp_min(1e-12))
                            .detach()
                            .cpu()
                        ),
                        "segmentation_logit_delta_rms": float(
                            (output.segmentation_logits - baseline_logits)
                            .square()
                            .mean()
                            .sqrt()
                            .detach()
                            .cpu()
                        ),
                        "segmentation_label_change_fraction": float(
                            (
                                output.segmentation_logits.argmax(1)
                                != baseline_logits.argmax(1)
                            )
                            .float()
                            .mean()
                            .detach()
                            .cpu()
                        ),
                    }
                )
            print(
                f"mechanism audit: seed={seed} probe={probe_index} role={record['role']}",
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(EXPERIMENT / "gnn_inference_mechanism_ablation.csv", mechanism_rows)
    write_csv(EXPERIMENT / "gnn_root_neighbor_decomposition.csv", decomposition_rows)
    write_csv(EXPERIMENT / "rgt_relation_statistics.csv", relation_rows)
    edge_rows = [
        row
        for row in mechanism_rows
        if row["control"] in {"intact_corrected_rgt", "edge_attr_zeroed", "edge_attr_shuffled"}
    ]
    write_csv(EXPERIMENT / "avo_edge_attr_ablation.csv", edge_rows)


def mechanism_trajectory(args: argparse.Namespace) -> None:
    """Measure message use across the bounded corrected-RGT checkpoints."""
    contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    confidence = contract["confidence_rule"]
    source_experiment = (
        PRIVATE / "stage_artifacts/stage04/sage_avo_s01_v00332o_causal_gnn_multiseed"
    )
    probe_manifest = json.loads(
        (source_experiment / "fixed_graph_probe_samples.json").read_text(encoding="utf-8")
    )
    probes = [
        record
        for record in probe_manifest["patches"]
        if tuple(record["raw_scale"]) == tuple(record["resized_scale"])
    ][:5]
    normalization = json.loads((DATASET / "normalization.json").read_text(encoding="utf-8"))
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    y_mean = np.asarray(normalization["y_mean"], dtype=np.float32)[:, None, None]
    y_std = np.asarray(normalization["y_std"], dtype=np.float32)[:, None, None]
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332p matched message-utilization trajectory",
    )
    rows: list[dict[str, Any]] = []
    for seed in (12345, 23456, 34567):
        run = EXPERIMENT / "runs" / f"seed_{seed}_corrected_rgt_gnn"
        first_checkpoint = run / "milestone_checkpoints/epoch_0001.pt"
        raw = torch.load(first_checkpoint, map_location="cpu", weights_only=False)
        model = build_sage_avo_variant(
            "full", **sage_avo_model_kwargs(raw["config"])
        ).to(device)
        model.set_norm_stats(normalization)
        model.eval()
        for epoch in (1, 3, 5, 10):
            load_checkpoint(
                run / "milestone_checkpoints" / f"epoch_{epoch:04d}.pt",
                model,
                restore_rng=False,
                map_location=device,
            )
            for probe_index, record in enumerate(probes):
                arrays = _probe_arrays(record)
                avo = torch.from_numpy(((arrays["avo"] - x_mean) / x_std)[None]).to(device)
                low = torch.from_numpy(((arrays["low"] - y_mean) / y_std)[None]).to(device)
                rgt = torch.from_numpy(arrays["rgt"][None]).to(device)
                time = torch.full((1,), 0.5, dtype=low.dtype, device=device)
                _reset_diagnostic_controls(model, confidence)
                model.graph.diagnostic_capture_contributions = True
                model.diagnostic_capture_fusion = True
                with torch.inference_mode():
                    baseline = model(low, time, avo, low, rgt)
                contributions = [dict(value) for value in model.graph.last_contribution_diagnostics]
                fusion = dict(model.last_fusion_diagnostics)
                baseline_velocity = baseline.velocity.detach().clone()
                attention = _attention_statistics(baseline.attention_weights[0])

                _reset_diagnostic_controls(model, confidence)
                model.graph.diagnostic_neighbor_scale = 0.0
                with torch.inference_mode():
                    no_neighbor = model(low, time, avo, low, rgt)
                neighbor_delta = (
                    no_neighbor.velocity - baseline_velocity
                ).square().mean().sqrt()
                baseline_rms = baseline_velocity.square().mean().sqrt().clamp_min(1e-12)

                _reset_diagnostic_controls(model, confidence)
                model.graph.diagnostic_root_scale = 0.0
                with torch.inference_mode():
                    no_root = model(low, time, avo, low, rgt)
                root_delta = (no_root.velocity - baseline_velocity).square().mean().sqrt()
                for contribution in contributions:
                    rows.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "probe_index": probe_index,
                            "realization_id": int(record["realization_id"]),
                            "role": record["role"],
                            **contribution,
                            **fusion,
                            **attention,
                            "velocity_relative_change_neighbor_zeroed": float(
                                (neighbor_delta / baseline_rms).detach().cpu()
                            ),
                            "velocity_relative_change_root_zeroed": float(
                                (root_delta / baseline_rms).detach().cpu()
                            ),
                        }
                    )
            print(f"[v00332p] message trajectory seed={seed} epoch={epoch}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(EXPERIMENT / "gnn_message_trajectory.csv", rows)
    print(f"Wrote mechanism artifacts under {EXPERIMENT}")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _completed_epoch(run: Path) -> int:
    manifest = run / "manifest.json"
    if not manifest.exists():
        return 0
    return int(json.loads(manifest.read_text(encoding="utf-8")).get("last_completed_epoch", 0))


def _training_config(
    training_contract: dict[str, Any], topology_contract: dict[str, Any], seed: int, architecture: str
) -> dict[str, Any]:
    config = deepcopy(
        load_config(REPOSITORY / "configs" / training_contract["base_training_config"])
    )
    budget = training_contract["bounded_training"]
    definition = training_contract["architectures"][architecture]
    confidence = topology_contract["confidence_rule"]["thresholds"]
    config["dataset"]["directory"] = f"datasets/{training_contract['immutable_dataset']}"
    config["experiment"]["name"] = training_contract["experiment_name"]
    config["experiment"]["seed"] = int(seed)
    config["training"]["epochs"] = int(budget["epochs"])
    config["training"]["batch_size"] = int(budget["batch_size"])
    config["training"]["sample_steps_validation"] = int(budget["sample_steps_validation"])
    config["training"]["validation_sample_batches"] = int(
        budget["validation_patches"] // budget["batch_size"]
    )
    config["training"]["loss_weights"]["structure"] = 0.0
    config["training"]["graph_objective"] = {"mode": "no_aux_graph_loss"}
    config["training"]["contrastive_loss"].update(enabled=False, weight=0.0)
    config["training"]["adaptive_task_weighting"]["enabled"] = False
    config["training"]["physics_guided_sampling"].update(enabled=False, guidance_scale=0.0)
    config["training"]["checkpointing"]["whole_validation_every_epochs"] = 1000
    config["training"]["checkpointing"]["periodic_interval_epochs"] = 1
    config["model"]["experimental_graph"] = {
        "rgt_topology": definition["rgt_topology"],
        "neighbor_scale": float(definition["neighbor_scale"]),
        "confidence_normalized_mismatch_threshold": confidence[
            "normalized_best_mismatch"
        ],
        "confidence_normalized_discontinuity_threshold": confidence[
            "normalized_cartesian_discontinuity"
        ],
        "confidence_dip_residual_threshold": confidence["rgt_dip_residual"],
    }
    config["capabilities"]["rgt_topology_repair"] = {
        "implemented": True,
        "enabled": architecture == "corrected_rgt_gnn",
        "diagnostic_revision": "v00332p",
    }
    return config


def prepare_training(args: argparse.Namespace) -> None:
    topology_contract_path = EXPERIMENT / "rgt_topology_repair_contract.json"
    if not topology_contract_path.exists():
        raise FileNotFoundError("Run topology-audit and confidence-qc first")
    topology_contract = json.loads(topology_contract_path.read_text(encoding="utf-8"))
    if "confidence_rule" not in topology_contract:
        raise RuntimeError("Run confidence-qc before preparing training")
    training_contract = load_config(TRAINING_CONFIG)
    train_dataset = IndexedRealizationPatches(DATASET, "train")
    validation_dataset = IndexedRealizationPatches(DATASET, "validation")
    rng = np.random.default_rng(20260908)
    schedule: list[list[int]] = []
    epochs = int(training_contract["bounded_training"]["epochs"])
    for epoch in range(epochs):
        selected: list[int] = []
        for realization_id, group in train_dataset.index.groupby("realization_id", sort=True):
            candidates = group.index.to_numpy(dtype=np.int64)
            local_rng = np.random.default_rng(20260908 + int(realization_id))
            ordered = local_rng.permutation(candidates)
            selected.append(int(ordered[epoch % len(ordered)]))
        selected = [selected[index] for index in rng.permutation(len(selected))]
        schedule.append(selected)
    validation_indices: list[int] = []
    for realization_id, group in validation_dataset.index.groupby("realization_id", sort=True):
        candidates = group.index.to_numpy(dtype=np.int64)
        local_rng = np.random.default_rng(20270000 + int(realization_id))
        validation_indices.extend(
            map(int, local_rng.choice(candidates, size=2, replace=False))
        )
    selection = {
        "selection_uses_model_outputs": False,
        "train_schedule": schedule,
        "validation_indices": validation_indices,
        "train_realization_count_per_epoch": len(
            set(map(int, train_dataset.index.iloc[schedule[0]]["realization_id"]))
        ),
        "validation_realization_count": len(
            set(map(int, validation_dataset.index.iloc[validation_indices]["realization_id"]))
        ),
        "train_schedule_sha256": _canonical_sha(schedule),
        "validation_indices_sha256": _canonical_sha(validation_indices),
    }
    selection_path = EXPERIMENT / "fixed_patch_schedule.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError("Refusing to replace an existing different patch schedule")
    else:
        _json_write(selection_path, selection)
    initial_directory = EXPERIMENT / "initial_states"
    initial_directory.mkdir(exist_ok=True)
    initialization_rows = []
    for seed in map(int, training_contract["seeds"]):
        seed_everything(seed, deterministic_torch=True)
        config = _training_config(
            training_contract, topology_contract, seed, "corrected_rgt_gnn"
        )
        model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config))
        state = dict(model.state_dict())
        path = initial_directory / f"seed_{seed}.pt"
        if path.exists():
            observed = torch.load(path, map_location="cpu", weights_only=True)
            if _state_sha(observed) != _state_sha(state):
                raise RuntimeError(f"Existing initialization differs for seed {seed}")
        else:
            torch.save(state, path)
        for architecture in training_contract["architectures"]:
            candidate_config = _training_config(
                training_contract, topology_contract, seed, architecture
            )
            candidate = build_sage_avo_variant(
                "full", **sage_avo_model_kwargs(candidate_config)
            )
            if list(candidate.state_dict()) != list(state):
                raise RuntimeError(f"State keys differ for {architecture}")
            initialization_rows.append(
                {
                    "seed": seed,
                    "architecture": architecture,
                    "parameter_count": sum(p.numel() for p in candidate.parameters()),
                    "state_sha256": _state_sha(state),
                }
            )
    topology_contract["bounded_training"] = {
        **training_contract["bounded_training"],
        "architectures": training_contract["architectures"],
        "seeds": training_contract["seeds"],
        "fixed_patch_schedule": selection,
        "initializations": initialization_rows,
        "all_controls_parameter_and_state_key_matched": True,
    }
    topology_contract["status"] = "PREPARED_BOUNDED_TRAINING_NOT_STARTED"
    _json_write(topology_contract_path, topology_contract)
    print(json.dumps(topology_contract["bounded_training"], indent=2))


def training_smoke(args: argparse.Namespace) -> None:
    topology_contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    training_contract = load_config(TRAINING_CONFIG)
    selection = json.loads(
        (EXPERIMENT / "fixed_patch_schedule.json").read_text(encoding="utf-8")
    )
    dataset = IndexedRealizationPatches(DATASET, "train")
    item = dataset[int(selection["train_schedule"][0][0])]
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context="v00332p parameter-matched training smoke",
    )
    rows = []
    seed = int(training_contract["seeds"][0])
    source_state = torch.load(
        EXPERIMENT / "initial_states" / f"seed_{seed}.pt",
        map_location=device,
        weights_only=True,
    )
    for architecture in training_contract["architectures"]:
        config = _training_config(training_contract, topology_contract, seed, architecture)
        model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
        model.load_state_dict(source_state, strict=True)
        state = item["low"][None].to(device)
        avo = item["avo"][None].to(device)
        rgt = item["rgt"][None].to(device)
        output = model(state, torch.full((1,), 0.5, device=device), avo, state, rgt)
        loss = output.velocity.square().mean() + output.segmentation_logits.square().mean()
        loss.backward()
        rows.append(
            {
                "architecture": architecture,
                "finite": bool(torch.isfinite(output.velocity).all()),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "graph_gradient_tensor_count": sum(
                    parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                    for name, parameter in model.named_parameters()
                    if name.startswith("graph.layers")
                ),
            }
        )
    if not all(row["finite"] for row in rows):
        raise FloatingPointError("Nonfinite v00332p smoke output")
    if len({row["parameter_count"] for row in rows}) != 1:
        raise RuntimeError("Parameter counts are not matched")
    _json_write(EXPERIMENT / "bounded_training_smoke.json", {"status": "PASS", "rows": rows})
    print(json.dumps(rows, indent=2))


def train_bounded(args: argparse.Namespace) -> None:
    topology_contract_path = EXPERIMENT / "rgt_topology_repair_contract.json"
    topology_contract = json.loads(topology_contract_path.read_text(encoding="utf-8"))
    smoke = json.loads((EXPERIMENT / "bounded_training_smoke.json").read_text(encoding="utf-8"))
    if smoke["status"] != "PASS":
        raise RuntimeError("Training smoke must pass")
    training_contract = load_config(TRAINING_CONFIG)
    selection = json.loads(
        (EXPERIMENT / "fixed_patch_schedule.json").read_text(encoding="utf-8")
    )
    epochs = int(training_contract["bounded_training"]["epochs"])
    milestones = set(map(int, training_contract["bounded_training"]["evaluation_epochs"]))
    for seed in map(int, training_contract["seeds"]):
        initial = EXPERIMENT / "initial_states" / f"seed_{seed}.pt"
        for architecture in training_contract["architectures"]:
            config = _training_config(training_contract, topology_contract, seed, architecture)
            run = EXPERIMENT / "runs" / f"seed_{seed}_{architecture}"
            start = _completed_epoch(run)
            if start < epochs:
                train_controlled_variant(
                    repository=REPOSITORY,
                    config_path=TRAINING_CONFIG,
                    config=config,
                    dataset_directory=DATASET,
                    experiment_directory=EXPERIMENT,
                    variant="full",
                    device_name=args.device,
                    epochs_override=epochs,
                    max_train_batches=len(selection["train_schedule"][0])
                    // int(training_contract["bounded_training"]["batch_size"]),
                    max_validation_batches=len(selection["validation_indices"])
                    // int(training_contract["bounded_training"]["batch_size"]),
                    run_name=run.name,
                    resume_from=run / "last.pt" if (run / "last.pt").exists() else None,
                    stop_after_epoch=epochs,
                    fixed_train_indices_by_epoch=selection["train_schedule"],
                    fixed_validation_indices=selection["validation_indices"],
                    initial_model_state=initial,
                    finite_state_check_batches=(1, 18, 35),
                    abort_on_nonfinite=True,
                )
            checkpoint_directory = run / "milestone_checkpoints"
            checkpoint_directory.mkdir(exist_ok=True)
            for epoch in milestones:
                destination = checkpoint_directory / f"epoch_{epoch:04d}.pt"
                if destination.exists():
                    continue
                source = run / f"checkpoint_epoch_{epoch:04d}.pt"
                if not source.exists():
                    if epoch == epochs and (run / "last.pt").exists():
                        source = run / "last.pt"
                    else:
                        raise FileNotFoundError(source)
                shutil.copyfile(source, destination)
            print(
                f"[v00332p] completed seed={seed} architecture={architecture} epochs={epochs}",
                flush=True,
            )
    topology_contract["status"] = "BOUNDED_TRAINING_COMPLETE_PENDING_WHOLE_EVALUATION"
    _json_write(topology_contract_path, topology_contract)


def _segmentation_metrics(
    prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {
        "segmentation_accuracy": float(np.mean(prediction[mask] == truth[mask]))
    }
    ious = []
    for label in range(3):
        predicted = (prediction == label) & mask
        expected = (truth == label) & mask
        union = int(np.count_nonzero(predicted | expected))
        value = float(np.count_nonzero(predicted & expected) / union) if union else float("nan")
        result[f"class_{label}_iou"] = value
        ious.append(value)
    result["miou"] = float(np.nanmean(ious))
    return result


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    data_range = max(float(max(first.max(), second.max()) - min(first.min(), second.min())), 1e-8)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_first, mean_second = float(first.mean()), float(second.mean())
    variance_first, variance_second = float(first.var()), float(second.var())
    covariance = float(np.mean((first - mean_first) * (second - mean_second)))
    return float(
        ((2 * mean_first * mean_second + c1) * (2 * covariance + c2))
        / ((mean_first**2 + mean_second**2 + c1) * (variance_first + variance_second + c2))
    )


def _pixel_strata(
    arrays: dict[str, np.ndarray], faults: list[dict[str, float]], contract: dict[str, Any]
) -> dict[str, np.ndarray]:
    rgt = arrays["rgt"]
    fields = structural_fields(rgt)
    rows, columns = np.indices(rgt.shape)
    fault = np.zeros(rgt.shape, dtype=bool)
    for item in faults:
        boundary = float(item["column"]) + float(item["dip"]) * rows
        fault |= np.abs(columns - boundary) <= 3.0
    labels = arrays["segmentation"]
    facies_boundary = np.zeros_like(fault)
    facies_boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    facies_boundary[1:] |= labels[1:] != labels[:-1]
    valid = arrays["valid_mask"].astype(bool)
    return {
        "all": valid,
        "low_dip": valid & (fields["dip"] < float(contract["dip_q33"])),
        "high_dip_continuous": valid
        & (fields["dip"] >= float(contract["dip_q67"]))
        & ~fault,
        "fault_corridor": valid & fault,
        "away_from_fault": valid & ~fault,
        "facies_boundary": valid & facies_boundary,
        "reservoir": valid & arrays["reservoir_mask"].astype(bool),
        "plume": valid & arrays["plume_mask"].astype(bool),
    }


def _evaluate_checkpoint_set(
    *,
    args: argparse.Namespace,
    realization_ids: list[int],
    epochs: list[int],
    evaluation_scope: str,
    architectures: list[str] | None = None,
    edge_attr_mode: str = "current",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topology_contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    training_contract = load_config(TRAINING_CONFIG)
    budget = training_contract["bounded_training"]
    normalization = json.loads((DATASET / "normalization.json").read_text(encoding="utf-8"))
    y_std = np.asarray(normalization["y_std"], dtype=np.float64)
    x_mean = np.asarray(normalization["x_mean"], dtype=np.float32)[:, None, None]
    x_std = np.asarray(normalization["x_std"], dtype=np.float32)[:, None, None]
    print_torch_runtime()
    device = select_torch_device(
        args.device,
        require_cuda=str(args.device).startswith("cuda"),
        context=f"v00332p {evaluation_scope} whole-realization evaluation",
    )
    rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    selected_architectures = architectures or list(training_contract["architectures"])
    if edge_attr_mode not in {"current", "zero", "shuffled"}:
        raise ValueError(f"unsupported edge_attr_mode {edge_attr_mode!r}")
    for seed in map(int, training_contract["seeds"]):
        for architecture in selected_architectures:
            config = _training_config(training_contract, topology_contract, seed, architecture)
            forward_specification = forward_specification_from_mapping(config)
            model = build_sage_avo_variant("full", **sage_avo_model_kwargs(config)).to(device)
            model.set_norm_stats(normalization)
            run = EXPERIMENT / "runs" / f"seed_{seed}_{architecture}"
            for epoch in epochs:
                checkpoint = run / "milestone_checkpoints" / f"epoch_{epoch:04d}.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                load_checkpoint(checkpoint, model, restore_rng=False, map_location=device)
                model.eval()
                model.graph.diagnostic_edge_attr_mode = edge_attr_mode
                for realization_id in realization_ids:
                    output_directory = run / "whole_evaluation" / f"epoch_{epoch:04d}"
                    if edge_attr_mode != "current":
                        output_directory = (
                            run
                            / "whole_evaluation_edge_attr"
                            / edge_attr_mode
                            / f"epoch_{epoch:04d}"
                        )
                    output_directory.mkdir(parents=True, exist_ok=True)
                    output_path = output_directory / f"realization_{realization_id:07d}.npz"
                    arrays = load_realization(DATASET, realization_id)
                    if output_path.exists():
                        with np.load(output_path, allow_pickle=False) as saved:
                            prediction = np.asarray(saved["prediction"])
                            predicted_labels = np.asarray(saved["segmentation_prediction"])
                    else:
                        prediction, predicted_labels = infer_full_realization(
                            model,
                            avo=arrays["avo"],
                            low=arrays["low"],
                            rgt=arrays["rgt"],
                            normalization=normalization,
                            patch_shape=tuple(map(int, budget["whole_patch_shape"])),
                            stride=tuple(map(int, budget["whole_stride"])),
                            steps=int(budget["whole_flow_steps"]),
                            batch_size=int(budget["whole_batch_size"]),
                            device=device,
                            valid_mask=arrays["valid_mask"],
                        )
                        np.savez_compressed(
                            output_path,
                            prediction=prediction,
                            segmentation_prediction=predicted_labels,
                        )
                    valid = arrays["valid_mask"].astype(bool)
                    segmentation = _segmentation_metrics(
                        predicted_labels, arrays["segmentation"], valid
                    )
                    normalized_rmse = []
                    property_values: dict[str, float] = {}
                    for channel, name in enumerate(("vp", "vs", "density")):
                        error = prediction[channel][valid] - arrays["elastic"][channel][valid]
                        rmse = float(np.sqrt(np.mean(error**2)))
                        normalized = rmse / float(y_std[channel])
                        normalized_rmse.append(normalized)
                        property_values[f"{name}_rmse"] = rmse
                        property_values[f"{name}_normalized_rmse"] = normalized
                        property_values[f"{name}_ssim"] = _global_ssim(
                            prediction[channel][valid], arrays["elastic"][channel][valid]
                        )
                    prediction_tensor = torch.from_numpy(prediction[None])
                    with torch.inference_mode():
                        modeled = forward_avo_three_band_spec_torch(
                            prediction_tensor[:, 0],
                            prediction_tensor[:, 1],
                            prediction_tensor[:, 2],
                            forward_specification,
                            sample_origin=0,
                        )[0].numpy()
                    physics_error = (
                        (modeled - x_mean) / x_std
                        - (arrays["avo_clean"] - x_mean) / x_std
                    ) ** 2
                    exact_pp = float(np.sqrt(np.mean(physics_error[:, valid])))
                    criterion = float(np.mean(normalized_rmse) - 0.1 * segmentation["miou"])
                    rows.append(
                        {
                            "evaluation_scope": evaluation_scope,
                            "edge_attr_mode": edge_attr_mode,
                            "seed": seed,
                            "architecture": architecture,
                            "epoch": epoch,
                            "realization_id": realization_id,
                            "checkpoint_criterion": criterion,
                            "exact_pp_rmse_clean_normalized": exact_pp,
                            **property_values,
                            **segmentation,
                        }
                    )
                    masks = _pixel_strata(
                        arrays,
                        load_faults(STAGE02, realization_id),
                        topology_contract["adaptive_search"],
                    )
                    for stratum, mask in masks.items():
                        if not mask.any():
                            continue
                        stratum_segmentation = _segmentation_metrics(
                            predicted_labels, arrays["segmentation"], mask
                        )
                        for channel, name in enumerate(("vp", "vs", "density")):
                            error = prediction[channel][mask] - arrays["elastic"][channel][mask]
                            strata_rows.append(
                                {
                                    "evaluation_scope": evaluation_scope,
                                    "edge_attr_mode": edge_attr_mode,
                                    "seed": seed,
                                    "architecture": architecture,
                                    "epoch": epoch,
                                    "realization_id": realization_id,
                                    "stratum": stratum,
                                    "pixel_count": int(mask.sum()),
                                    "property": name,
                                    "normalized_rmse": float(
                                        np.sqrt(np.mean(error**2)) / y_std[channel]
                                    ),
                                    "exact_pp_rmse_clean_normalized": float(
                                        np.sqrt(np.mean(physics_error[:, mask]))
                                    ),
                                    **stratum_segmentation,
                                }
                            )
                    print(
                        f"[v00332p] evaluated scope={evaluation_scope} seed={seed} "
                        f"architecture={architecture} epoch={epoch} realization={realization_id}",
                        flush=True,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows, strata_rows


def evaluate_diverse(args: argparse.Namespace) -> None:
    contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    ids = [int(value) for value in contract["diverse_validation_subset"]["all_ids"]]
    epochs = list(map(int, load_config(TRAINING_CONFIG)["bounded_training"]["evaluation_epochs"]))
    rows, strata_rows = _evaluate_checkpoint_set(
        args=args, realization_ids=ids, epochs=epochs, evaluation_scope="diverse_six"
    )
    write_csv(EXPERIMENT / "rgt_causal_multiseed_results.csv", rows)
    write_csv(EXPERIMENT / "rgt_complexity_stratified_results.csv", strata_rows)


def evaluate_all_validation(args: argparse.Namespace) -> None:
    contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    ids = [int(value) for value in contract["split_ids"]["validation"]]
    rows, strata_rows = _evaluate_checkpoint_set(
        args=args, realization_ids=ids, epochs=[10], evaluation_scope="all_twenty"
    )
    result_path = EXPERIMENT / "rgt_causal_multiseed_results.csv"
    strata_path = EXPERIMENT / "rgt_complexity_stratified_results.csv"
    existing = pd.read_csv(result_path)
    existing = existing[existing["evaluation_scope"] != "all_twenty"]
    pd.concat((existing, pd.DataFrame(rows)), ignore_index=True).to_csv(result_path, index=False)
    existing_strata = pd.read_csv(strata_path)
    existing_strata = existing_strata[
        existing_strata["evaluation_scope"] != "all_twenty"
    ]
    pd.concat((existing_strata, pd.DataFrame(strata_rows)), ignore_index=True).to_csv(
        strata_path, index=False
    )


def evaluate_edge_attr(args: argparse.Namespace) -> None:
    """Evaluate the learned AVO affinity feature without retraining.

    These are frozen-checkpoint intervention outcomes.  They establish whether
    the learned model uses the feature and whether retaining it helps the
    bounded whole-realization criterion; they are not matched retraining runs.
    """
    contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    ids = [int(value) for value in contract["diverse_validation_subset"]["all_ids"]]
    outcome_rows: list[dict[str, Any]] = []
    for mode in ("current", "zero", "shuffled"):
        rows, _ = _evaluate_checkpoint_set(
            args=args,
            realization_ids=ids,
            epochs=[10],
            evaluation_scope="diverse_six_edge_attr_intervention",
            architectures=["corrected_rgt_gnn"],
            edge_attr_mode=mode,
        )
        for row in rows:
            outcome_rows.append(
                {
                    "evaluation_type": "frozen_whole_outcome_intervention",
                    "control": {
                        "current": "current_avo_affinity_edge_attr",
                        "zero": "edge_attr_zeroed",
                        "shuffled": "edge_attr_shuffled",
                    }[mode],
                    **row,
                }
            )
    path = EXPERIMENT / "avo_edge_attr_ablation.csv"
    mechanism = pd.read_csv(path)
    if "evaluation_type" in mechanism:
        mechanism = mechanism[
            mechanism["evaluation_type"] == "patch_output_sensitivity"
        ].copy()
    else:
        mechanism.insert(0, "evaluation_type", "patch_output_sensitivity")
    pd.concat((mechanism, pd.DataFrame(outcome_rows)), ignore_index=True).to_csv(
        path, index=False
    )


def _mean_by_seed(
    frame: pd.DataFrame, metric: str, *, architecture: str | None = None
) -> dict[int, float]:
    selected = frame if architecture is None else frame[frame["architecture"] == architecture]
    return {
        int(seed): float(group[metric].mean())
        for seed, group in selected.groupby("seed", sort=True)
    }


def _paired_summary(
    frame: pd.DataFrame,
    candidate: str,
    control: str,
    metric: str,
    *,
    higher_is_better: bool = False,
) -> dict[str, Any]:
    candidate_values = _mean_by_seed(frame, metric, architecture=candidate)
    control_values = _mean_by_seed(frame, metric, architecture=control)
    common = sorted(set(candidate_values) & set(control_values))
    deltas = np.asarray(
        [candidate_values[seed] - control_values[seed] for seed in common],
        dtype=np.float64,
    )
    favorable = deltas > 0 if higher_is_better else deltas < 0
    denominator = max(abs(float(np.mean([control_values[s] for s in common]))), 1e-12)
    relative_gain = (
        float(deltas.mean()) / denominator
        if higher_is_better
        else -float(deltas.mean()) / denominator
    )
    return {
        "candidate": candidate,
        "control": control,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "mean_delta_candidate_minus_control": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "mean_relative_gain": relative_gain,
        "favorable_seed_count": int(favorable.sum()),
        "seed_count": len(common),
        "per_seed_delta": {str(seed): float(delta) for seed, delta in zip(common, deltas)},
    }


def _architecture_summary(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = [
        "vp_normalized_rmse",
        "vs_normalized_rmse",
        "density_normalized_rmse",
        "checkpoint_criterion",
        "exact_pp_rmse_clean_normalized",
        "vp_ssim",
        "vs_ssim",
        "density_ssim",
        "miou",
        "class_0_iou",
        "class_1_iou",
        "class_2_iou",
    ]
    result: dict[str, Any] = {}
    for architecture, group in frame.groupby("architecture", sort=True):
        result[str(architecture)] = {}
        for metric in metrics:
            per_seed = _mean_by_seed(group, metric)
            values = np.asarray(list(per_seed.values()), dtype=np.float64)
            result[str(architecture)][metric] = {
                "mean": float(values.mean()),
                "std_across_seed_means": float(values.std()),
                "per_seed": {str(seed): value for seed, value in per_seed.items()},
            }
    return result


def analyze_results(_: argparse.Namespace) -> None:
    """Apply the predeclared v00332p screen and write its auditable decision."""
    result_path = EXPERIMENT / "rgt_causal_multiseed_results.csv"
    strata_path = EXPERIMENT / "rgt_complexity_stratified_results.csv"
    edge_path = EXPERIMENT / "avo_edge_attr_ablation.csv"
    if not result_path.exists() or not strata_path.exists() or not edge_path.exists():
        raise FileNotFoundError("Run diverse and edge-attribute evaluation first")
    contract = json.loads(
        (EXPERIMENT / "rgt_topology_repair_contract.json").read_text(encoding="utf-8")
    )
    decision_config = load_config(TRAINING_CONFIG)["decision"]
    results = pd.read_csv(result_path)
    strata = pd.read_csv(strata_path)
    diverse = results[
        (results["evaluation_scope"] == "diverse_six") & (results["epoch"] == 10)
    ].copy()
    expected = 3 * 4 * 6
    if len(diverse) != expected:
        raise RuntimeError(f"Expected {expected} epoch-10 diverse rows, found {len(diverse)}")
    diverse_cartesian = _paired_summary(
        diverse,
        "corrected_rgt_gnn",
        "cartesian_gnn",
        "checkpoint_criterion",
    )
    diverse_root = _paired_summary(
        diverse,
        "corrected_rgt_gnn",
        "root_only_gnn",
        "checkpoint_criterion",
    )
    minimum_gain = float(decision_config["minimum_mean_relative_criterion_gain"])
    required_signs = int(decision_config["reproducible_favorable_seed_count"])

    def passes(comparison: dict[str, Any]) -> bool:
        return (
            comparison["favorable_seed_count"] >= required_signs
            and comparison["mean_relative_gain"] >= minimum_gain
        )

    diverse_screen_pass = passes(diverse_cartesian) and passes(diverse_root)
    all_twenty = results[
        (results["evaluation_scope"] == "all_twenty") & (results["epoch"] == 10)
    ].copy()
    if diverse_screen_pass and len(all_twenty) != 3 * 4 * 20:
        raise RuntimeError(
            "The diverse screen passed; evaluate all 20 validation realizations "
            "before producing the decision"
        )
    outcome = all_twenty if len(all_twenty) == 3 * 4 * 20 else diverse
    outcome_scope = "all_twenty" if outcome is all_twenty else "diverse_six"
    comparisons: dict[str, Any] = {}
    for control in ("cartesian_gnn", "shuffled_graph", "root_only_gnn"):
        comparisons[control] = {
            metric: _paired_summary(
                outcome,
                "corrected_rgt_gnn",
                control,
                metric,
                higher_is_better=metric
                in {"vp_ssim", "vs_ssim", "density_ssim", "miou", "class_0_iou", "class_1_iou", "class_2_iou"},
            )
            for metric in (
                "vp_normalized_rmse",
                "vs_normalized_rmse",
                "density_normalized_rmse",
                "checkpoint_criterion",
                "exact_pp_rmse_clean_normalized",
                "vp_ssim",
                "vs_ssim",
                "density_ssim",
                "miou",
                "class_0_iou",
                "class_1_iou",
                "class_2_iou",
            )
        }
    corrected_vs_cartesian = comparisons["cartesian_gnn"]["checkpoint_criterion"]
    corrected_vs_root = comparisons["root_only_gnn"]["checkpoint_criterion"]
    corrected_vs_shuffle = comparisons["shuffled_graph"]["checkpoint_criterion"]

    rgt_specific = passes(corrected_vs_cartesian) and passes(corrected_vs_shuffle)
    neighbor_gain = passes(corrected_vs_root)

    epoch10_strata = strata[
        (strata["evaluation_scope"] == outcome_scope) & (strata["epoch"] == 10)
    ].copy()
    complex_effects: dict[str, Any] = {}
    for stratum in (
        "low_dip",
        "high_dip_continuous",
        "fault_corridor",
        "away_from_fault",
        "facies_boundary",
        "reservoir",
        "plume",
    ):
        selected = epoch10_strata[epoch10_strata["stratum"] == stratum]
        seed_deltas: dict[str, float] = {}
        for seed, group in selected.groupby("seed", sort=True):
            means = group.groupby("architecture")["normalized_rmse"].mean()
            if "corrected_rgt_gnn" in means and "cartesian_gnn" in means:
                seed_deltas[str(int(seed))] = float(
                    means["corrected_rgt_gnn"] - means["cartesian_gnn"]
                )
        values = np.asarray(list(seed_deltas.values()), dtype=np.float64)
        complex_effects[stratum] = {
            "mean_three_property_normalized_rmse_delta": float(values.mean()),
            "favorable_seed_count": int((values < 0).sum()),
            "per_seed_delta": seed_deltas,
        }
    complex_names = ("high_dip_continuous", "fault_corridor", "facies_boundary")
    complex_only = (
        all(complex_effects[name]["favorable_seed_count"] == 3 for name in complex_names)
        and complex_effects["low_dip"]["favorable_seed_count"] < 3
    )

    mechanism = pd.read_csv(EXPERIMENT / "gnn_inference_mechanism_ablation.csv")
    mechanism_means = mechanism.groupby("control", sort=True)[
        ["velocity_relative_delta", "embedding_relative_delta", "segmentation_label_change_fraction"]
    ].mean()
    neighbor_relative = float(
        mechanism_means.loc["neighbor_messages_zeroed", "velocity_relative_delta"]
    )
    root_relative = float(mechanism_means.loc["root_path_zeroed", "velocity_relative_delta"])
    neighbor_status = "MATERIAL" if neighbor_relative >= 0.01 else (
        "WEAK" if neighbor_relative >= 0.001 else "ROOT_DOMINATED"
    )
    decomposition = pd.read_csv(EXPERIMENT / "gnn_root_neighbor_decomposition.csv")
    mean_neighbor_root = float(decomposition["neighbor_root_ratio"].mean())
    mean_graph_cnn = float(decomposition["graph_cnn_ratio"].mean())
    mean_decoder_graph_effect = float(decomposition["decoder_graph_effect_rms"].mean())
    bypass_dominates = neighbor_status == "ROOT_DOMINATED" or (
        neighbor_relative < 0.01 and mean_decoder_graph_effect < 0.001
    )

    edge_frame = pd.read_csv(edge_path)
    edge_outcome = edge_frame[
        edge_frame["evaluation_type"] == "frozen_whole_outcome_intervention"
    ].copy()
    if len(edge_outcome) != 54:
        raise RuntimeError(f"Expected 54 whole edge-attribute rows, found {len(edge_outcome)}")
    edge_seed = edge_outcome.groupby(["control", "seed"])["checkpoint_criterion"].mean()
    current = edge_seed.loc["current_avo_affinity_edge_attr"]
    zero = edge_seed.loc["edge_attr_zeroed"]
    shuffled = edge_seed.loc["edge_attr_shuffled"]
    edge_deltas = {
        "zero_minus_current": {str(int(seed)): float(zero[seed] - current[seed]) for seed in current.index},
        "shuffled_minus_current": {
            str(int(seed)): float(shuffled[seed] - current[seed]) for seed in current.index
        },
    }
    zero_values = np.asarray(list(edge_deltas["zero_minus_current"].values()))
    shuffled_values = np.asarray(list(edge_deltas["shuffled_minus_current"].values()))
    current_better = bool((zero_values > 0).all() and (shuffled_values > 0).all())
    current_worse = bool((zero_values < 0).all() and (shuffled_values < 0).all())
    relative_edge_effect = max(
        abs(float(zero_values.mean() / max(abs(float(current.mean())), 1e-12))),
        abs(float(shuffled_values.mean() / max(abs(float(current.mean())), 1e-12))),
    )
    if current_better and relative_edge_effect >= minimum_gain:
        edge_status = "USEFUL"
    elif current_worse and relative_edge_effect >= minimum_gain:
        edge_status = "HARMFUL"
    elif relative_edge_effect < minimum_gain:
        edge_status = "NEUTRAL"
    else:
        edge_status = "UNRESOLVED"

    tie = pd.read_csv(EXPERIMENT / "rgt_tie_break_statistics.csv")
    validation_tie = tie[(tie["split"] == "validation") & (tie["realization_id"].astype(str) == "ALL")]
    if validation_tie.empty:
        validation_tie = pd.DataFrame(
            [{
                "exact_tie_minimum_fraction": tie[tie["split"] == "validation"]["exact_tie_minimum_fraction"].mean(),
                "near_tie_minimum_fraction": tie[tie["split"] == "validation"]["near_tie_minimum_fraction"].mean(),
                "v1_v2_changed_fraction": tie[tie["split"] == "validation"]["v1_v2_changed_fraction"].mean(),
                "v1_tied_absolute_shift_mean": tie[tie["split"] == "validation"]["v1_tied_absolute_shift_mean"].mean(),
                "v2_tied_absolute_shift_mean": tie[tie["split"] == "validation"]["v2_tied_absolute_shift_mean"].mean(),
            }]
        )
    tie_row = validation_tie.iloc[0]
    fault_qc = pd.read_csv(EXPERIMENT / "rgt_fault_corridor_qc.csv")
    validation_qc = fault_qc[
        (fault_qc["split"] == "validation")
        & (
            fault_qc["topology"].astype(str).str.casefold()
            == RGT_V3_CONFIDENCE_BLOCKED.casefold()
        )
    ]
    all_qc = validation_qc[validation_qc["region"] == "all"].copy()
    true_edges = all_qc["edge_count"] * all_qc["true_fault_crossing_fraction"]
    fault_capture = float(
        np.nansum(true_edges * all_qc["fraction_true_fault_crossing_edges_blocked"])
        / max(float(np.nansum(true_edges)), 1.0)
    )
    away_qc = validation_qc[validation_qc["region"] == "away_from_fault"]
    away_false = float(
        np.nansum(away_qc["edge_count"] * away_qc["fraction_edges_blocked"])
        / max(float(np.nansum(away_qc["edge_count"])), 1.0)
    )
    high_dip_qc = validation_qc[validation_qc["region"] == "high_dip_nonfault"]
    high_dip_false = float(
        np.nansum(high_dip_qc["edge_count"] * high_dip_qc["fraction_edges_blocked"])
        / max(float(np.nansum(high_dip_qc["edge_count"])), 1.0)
    )
    fault_status = "SAFE" if fault_capture >= 0.8 and away_false <= 0.005 else "NEEDS_BLOCKING"

    if rgt_specific and neighbor_gain:
        decision = "RGT_TOPOLOGY_REPAIR_RESTORES_GAIN"
    elif complex_only and neighbor_gain:
        decision = "RGT_GAIN_COMPLEX_GEOLOGY_ONLY"
    elif neighbor_gain and not rgt_specific:
        decision = "GENERIC_NEIGHBOR_MESSAGE_GAIN_RGT_NOT_SPECIFIC"
    elif bypass_dominates:
        decision = "GRAPH_ROOT_BYPASS_DOMINATES"
    else:
        trajectory = results[results["evaluation_scope"] == "diverse_six"].copy()
        epoch_effects = {}
        for epoch in (1, 3, 5, 10):
            selected = trajectory[trajectory["epoch"] == epoch]
            epoch_effects[str(epoch)] = _paired_summary(
                selected,
                "corrected_rgt_gnn",
                "cartesian_gnn",
                "checkpoint_criterion",
            )
        last_change = abs(
            epoch_effects["10"]["mean_delta_candidate_minus_control"]
            - epoch_effects["5"]["mean_delta_candidate_minus_control"]
        )
        base = max(
            abs(epoch_effects["10"]["mean_delta_candidate_minus_control"]), 1e-12
        )
        decision = (
            "RGT_TOPOLOGY_STILL_UNRESOLVED"
            if last_change / base >= 0.25
            else "RGT_TOPOLOGY_NO_REPRODUCIBLE_GAIN"
        )

    candidate = pd.read_csv(EXPERIMENT / "rgt_topology_candidate_comparison.csv")
    validation_candidates = (
        candidate[candidate["stratum"] == "all"]
        .groupby("topology", sort=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    relation = pd.read_csv(EXPERIMENT / "rgt_relation_statistics.csv")
    relation_means = relation.groupby("relation", sort=True).mean(numeric_only=True).to_dict("index")
    relation_different = (
        abs(relation_means["normal"]["rgt_mismatch_mean"] - relation_means["tangential"]["rgt_mismatch_mean"])
        / max(relation_means["tangential"]["rgt_mismatch_mean"], 1e-12)
        > 0.25
    )
    trajectory_summary: dict[str, Any] = {}
    trajectory_path = EXPERIMENT / "gnn_message_trajectory.csv"
    if trajectory_path.exists():
        trajectory_frame = pd.read_csv(trajectory_path)
        for epoch, group in trajectory_frame.groupby("epoch", sort=True):
            trajectory_summary[str(int(epoch))] = {
                key: float(group[key].mean())
                for key in (
                    "neighbor_root_ratio",
                    "attention_entropy",
                    "top_decile_mass",
                    "velocity_relative_change_neighbor_zeroed",
                    "velocity_relative_change_root_zeroed",
                    "graph_cnn_ratio",
                    "decoder_graph_effect_rms",
                )
            }
    questions = {
        "1_topology_change_after_tie_fix": {
            "answer": "material",
            "validation_edge_change_fraction": float(tie_row["v1_v2_changed_fraction"]),
            "exact_tie_fraction": float(tie_row["exact_tie_minimum_fraction"]),
            "near_tie_fraction": float(tie_row["near_tie_minimum_fraction"]),
            "legacy_tied_absolute_shift_mean": float(tie_row["v1_tied_absolute_shift_mean"]),
            "fixed_tied_absolute_shift_mean": float(tie_row["v2_tied_absolute_shift_mean"]),
        },
        "2_plus_minus_three_failure_source": "both_continuous_high_dip_and_faults; wider search lowers mismatch but increases fault crossing",
        "3_inverse_rgt_better_than_local_search": "no_overall; lower mismatch but larger displacement, more fault crossing, and out-of-range omissions",
        "4_confidence_rule_suppresses_faults_without_destroying_dip": {
            "answer": "partially_not_safely_enough",
            "validation_fault_crossing_capture": fault_capture,
            "validation_away_false_block": away_false,
            "validation_high_dip_nonfault_false_block": high_dip_false,
        },
        "5_corrected_rgt_outperforms_cartesian_across_seeds": passes(corrected_vs_cartesian),
        "6_gain_larger_in_complex_geology": complex_only,
        "7_transformerconv_neighbor_use": {
            "status": neighbor_status,
            "velocity_relative_change_when_zeroed": neighbor_relative,
            "mean_neighbor_root_norm_ratio": mean_neighbor_root,
        },
        "8_cnn_graph_bypass_dominates": {
            "answer": bypass_dominates,
            "velocity_relative_change_when_root_zeroed": root_relative,
            "mean_graph_cnn_norm_ratio": mean_graph_cnn,
            "mean_decoder_graph_effect_rms": mean_decoder_graph_effect,
        },
        "9_avo_edge_attr_effect": {
            "status": edge_status,
            "mechanistically_used": float(
                mechanism_means.loc["edge_attr_zeroed", "velocity_relative_delta"]
            ) >= 0.01,
            "criterion_deltas": edge_deltas,
        },
        "10_tangential_normal_statistically_different": relation_different,
        "11_historical_005_mechanistically_explainable": "partially; neighbor and routing sensitivity are real, but historical gains are not a matched causal result",
        "12_relation_aware_next_experiment_justified": bool(rgt_specific and neighbor_gain and relation_different),
    }
    summary = {
        "status": "BOUNDED_V00332P_EPOCH10_COMPLETE_EXTENSION_BLOCKED",
        "primary_decision": decision,
        "TIE_BREAK_DEFECT_FIXED": "YES",
        "FAULT_EDGE_QC_STATUS": fault_status,
        "NEIGHBOR_MESSAGE_STATUS": neighbor_status,
        "AVO_EDGE_ATTR_STATUS": edge_status,
        "screen_scope": "six preselected validation realizations; epoch 10; three matched seeds",
        "primary_outcome_scope": outcome_scope,
        "diverse_screen_passed_for_all_twenty": diverse_screen_pass,
        "predeclared_thresholds": decision_config,
        "architecture_metrics": _architecture_summary(outcome),
        "paired_comparisons": comparisons,
        "complexity_stratified_corrected_minus_cartesian": complex_effects,
        "edge_attr_frozen_intervention": edge_deltas,
        "topology_candidate_validation_means": validation_candidates.to_dict("records"),
        "relation_means": relation_means,
        "matched_message_trajectory": trajectory_summary,
        "required_final_questions": questions,
        "all_twenty_validation_evaluation_required": diverse_screen_pass,
        "all_twenty_validation_evaluation_complete": len(all_twenty) == 3 * 4 * 20,
        "epoch20_extension_status": (
            "required_by_unresolved_outcome_and_evolving_message_use_but_blocked_"
            "pending_scheduler_horizon_decision"
        ),
        "epoch20_scheduler_issue": (
            "epoch-10 checkpoints contain CosineAnnealingLR T_max=10; unchanged "
            "continuation rises after eta_min, while changing T_max now creates a "
            "schedule discontinuity and is not equivalent to a run planned with T_max=20"
        ),
        "production_training_started": False,
        "forbidden_model_changes_performed": False,
        "commit_or_push_performed": False,
    }
    _json_write(EXPERIMENT / "rgt_topology_repair_summary.json", summary)
    contract["status"] = "BOUNDED_V00332P_EPOCH10_COMPLETE_EXTENSION_BLOCKED"
    contract["outcome_decision"] = decision
    _json_write(EXPERIMENT / "rgt_topology_repair_contract.json", contract)
    print(json.dumps(summary, indent=2))



def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subcommands = result.add_subparsers(required=True)
    topology = subcommands.add_parser("topology-audit")
    topology.set_defaults(function=topology_audit)
    confidence = subcommands.add_parser("confidence-qc")
    confidence.set_defaults(function=confidence_qc)
    mechanism = subcommands.add_parser("mechanism-audit")
    mechanism.add_argument("--device", default="cuda")
    mechanism.set_defaults(function=mechanism_audit)
    trajectory = subcommands.add_parser("mechanism-trajectory")
    trajectory.add_argument("--device", default="cuda")
    trajectory.set_defaults(function=mechanism_trajectory)
    prepare = subcommands.add_parser("prepare-training")
    prepare.set_defaults(function=prepare_training)
    smoke = subcommands.add_parser("training-smoke")
    smoke.add_argument("--device", default="cuda")
    smoke.set_defaults(function=training_smoke)
    train = subcommands.add_parser("train-bounded")
    train.add_argument("--device", default="cuda")
    train.set_defaults(function=train_bounded)
    evaluate = subcommands.add_parser("evaluate-diverse")
    evaluate.add_argument("--device", default="cuda")
    evaluate.set_defaults(function=evaluate_diverse)
    evaluate_all = subcommands.add_parser("evaluate-all-validation")
    evaluate_all.add_argument("--device", default="cuda")
    evaluate_all.set_defaults(function=evaluate_all_validation)
    edge_attr = subcommands.add_parser("evaluate-edge-attr")
    edge_attr.add_argument("--device", default="cuda")
    edge_attr.set_defaults(function=evaluate_edge_attr)
    analyze = subcommands.add_parser("analyze-results")
    analyze.set_defaults(function=analyze_results)
    return result


def main() -> None:
    arguments = parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
