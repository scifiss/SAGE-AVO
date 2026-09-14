#!/usr/bin/env python3
"""Calibrate and evaluate v00332x native-RGT graph geometry; never train."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import run_skeleton_graph_v00332u as u
from sage_avo.config import load_config
from sage_avo.diagnostics.flattened_component_graph import (
    detect_reflector_components,
    flatten_rgt,
)
from sage_avo.diagnostics.native_rgt_graph import (
    NativeRGT,
    build_barrier_graph,
    freeze_thresholds,
    link_observables,
    refine_candidates,
)
from sage_avo.diagnostics.rgt_topology_repair import structural_fields
from sage_avo.diagnostics.skeleton_graph import path_fault_qc, sample

BRANCH = "experiment/v00332x-native-rgt-geometry"
CONFIG_PATH = u.REPO / "configs/development_diagnostics_v00332x.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = u.BASE / "stage04" / CONFIG["experiment_name"]
u.OUT = OUT
PROTECTED_SOURCE = [
    CONFIG_PATH,
    Path(__file__),
    u.REPO / "src/sage_avo/diagnostics/native_rgt_graph.py",
    u.REPO / "tests/test_native_rgt_graph.py",
]


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=u.REPO, text=True).strip()


def verify_provenance(expected_commit: str) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    parent = git("rev-parse", "HEAD^")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    remote_ref = git("rev-parse", f"refs/remotes/origin/{BRANCH}")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, found {branch}")
    if head != expected_commit:
        raise RuntimeError(f"HEAD {head} does not equal contract commit {expected_commit}")
    if remote_ref != expected_commit:
        raise RuntimeError(f"Remote-tracking ref {remote_ref} is not {expected_commit}")
    if tracked_status:
        raise RuntimeError("Tracked working tree must be clean before execution")
    source_hashes = {str(path.relative_to(u.REPO)): u.sha(path) for path in PROTECTED_SOURCE}
    return {
        "repository": str(u.REPO),
        "branch": branch,
        "commit_sha": head,
        "parent_commit_sha": parent,
        "remote_tracking_commit_sha": remote_ref,
        "source_config_test_sha256": source_hashes,
        "tracked_worktree_clean_at_start": True,
    }


def load_arrays(realization_id: int) -> dict[str, np.ndarray]:
    path = u.DATASET / "realizations" / f"realization_{realization_id:07d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key]) for key in ["avo", "rgt", "valid_mask", "reservoir_mask"]
        }


def quantile(values: Any, probability: float, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, probability)) if values.size else default


def calibration_run(
    realization_id: int, resolution: int, width: float
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    arrays = load_arrays(realization_id)
    native = NativeRGT(arrays["rgt"])
    time_error, tau_error = native.roundtrip()
    mapping = flatten_rgt(arrays["rgt"], arrays["avo"], arrays["valid_mask"], n_tau=resolution)
    detection = detect_reflector_components(mapping, CONFIG)
    refined = refine_candidates(
        native,
        arrays["avo"],
        mapping,
        detection,
        width_steps=width,
        samples=CONFIG["refinement_samples"],
    )
    displacement = [abs(row["time_displacement"]) for row in refined]
    gain = [
        (row["refined_strength"] - row["coarse_strength"]) / max(row["coarse_strength"], 1e-12)
        for row in refined
    ]
    row = {
        "realization_id": realization_id,
        "discovery_resolution": resolution,
        "refinement_width_steps": width,
        "candidate_count": len(refined),
        "native_time_error_max": float(np.max(time_error)),
        "native_tau_error_max": float(np.max(tau_error)),
        "coarse_flattened_time_error_p99": quantile(
            mapping["grid_resampling_time_error_samples"], 0.99
        ),
        "coarse_flattened_ava_error_p99": quantile(mapping["roundtrip_avo_abs"], 0.99),
        "ridge_time_displacement_p50": quantile(displacement, 0.50),
        "ridge_time_displacement_p99": quantile(displacement, 0.99),
        "ridge_relative_strength_gain_p50": quantile(gain, 0.50),
        "refinement_boundary_fraction": float(
            np.mean([row["refinement_at_boundary"] for row in refined])
        )
        if refined
        else 1.0,
    }
    details = {
        "arrays": arrays,
        "native": native,
        "mapping": mapping,
        "detection": detection,
        "refined": refined,
    }
    return row, details


def choose_discovery(calibration: list[dict[str, Any]]) -> tuple[int, float, list[dict[str, Any]]]:
    aggregate = []
    for resolution in CONFIG["discovery_resolutions"]:
        for width in CONFIG["refinement_width_steps"]:
            subset = [
                row
                for row in calibration
                if row["discovery_resolution"] == resolution
                and row["refinement_width_steps"] == width
            ]
            record = {
                "discovery_resolution": resolution,
                "refinement_width_steps": width,
                "worst_ridge_time_displacement_p99": max(
                    row["ridge_time_displacement_p99"] for row in subset
                ),
                "median_relative_strength_gain": float(
                    np.median([row["ridge_relative_strength_gain_p50"] for row in subset])
                ),
                "maximum_refinement_boundary_fraction": max(
                    row["refinement_boundary_fraction"] for row in subset
                ),
            }
            record["stable"] = bool(
                record["worst_ridge_time_displacement_p99"]
                <= CONFIG["decision"]["maximum_refinement_time_p99"]
                and record["maximum_refinement_boundary_fraction"]
                <= CONFIG["decision"]["maximum_refinement_boundary_fraction"]
            )
            aggregate.append(record)
    stable = [row for row in aggregate if row["stable"]]
    # If no setting meets the stability gate, retain the least-displaced setting
    # for diagnostic validation while keeping RIDGE_REFINEMENT_STATUS=FAIL. This
    # completes fault/graph QC without making the graph eligible for training.
    pool = stable or aggregate
    selected = min(
        pool,
        key=lambda row: (
            0 if row["stable"] else 1,
            row["worst_ridge_time_displacement_p99"]
            if not row["stable"]
            else -row["median_relative_strength_gain"],
            row["maximum_refinement_boundary_fraction"],
            row["discovery_resolution"],
            row["refinement_width_steps"],
        ),
    )
    return (
        int(selected["discovery_resolution"]),
        float(selected["refinement_width_steps"]),
        aggregate,
    )


def build_case(realization_id: int, resolution: int, width: float) -> dict[str, Any]:
    arrays = load_arrays(realization_id)
    native = NativeRGT(arrays["rgt"])
    time_error, tau_error = native.roundtrip()
    mapping = flatten_rgt(arrays["rgt"], arrays["avo"], arrays["valid_mask"], n_tau=resolution)
    detection = detect_reflector_components(mapping, CONFIG)
    refined = refine_candidates(
        native,
        arrays["avo"],
        mapping,
        detection,
        width_steps=width,
        samples=CONFIG["refinement_samples"],
    )
    links = link_observables(native, arrays["avo"], refined, mapping, CONFIG)
    return {
        "rid": realization_id,
        "arrays": arrays,
        "native": native,
        "native_time_error": time_error,
        "native_tau_error": tau_error,
        "mapping": mapping,
        "detection": detection,
        "refined": refined,
        "links": links,
    }


def annotate_validation(case: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Load truth only after the graph and thresholds have been frozen."""
    faults = u.load_faults(u.STAGE02, case["rid"])
    fields = structural_fields(case["arrays"]["rgt"])
    graph = case["graph"]
    fault_candidates = fault_split = adjacent_safe_crossing = adjacent_safe = 0
    high_candidates = high_retained = curved_candidates = curved_retained = 0
    agreement = []
    for row in graph["links"]:
        if not row["reciprocal"]:
            continue
        source = case["refined"][row["source"]]
        target = case["refined"][row["target"]]
        path = np.asarray(
            [
                [source["refined_time"], source["trace"]],
                [target["refined_time"], target["trace"]],
            ]
        )
        crossing, near = path_fault_qc(path, faults)
        row["fault_crossing"] = bool(crossing)
        row["fault_corridor"] = bool(near)
        fault_candidates += bool(crossing)
        fault_split += bool(crossing and row["fault_barrier"])
        adjacent_safe += bool(row["barrier_safe"])
        adjacent_safe_crossing += bool(row["barrier_safe"] and crossing)
        point = path[:1]
        structural_dip = float(sample(fields["dip"], point)[0])
        structural_curvature = float(sample(fields["curvature"], point)[0])
        shift_high = abs(row["physical_shift"]) >= thresholds["high_shift_magnitude"]
        structural_high = structural_dip >= case["structural_contract"]["dip_q67"]
        agreement.append(shift_high == structural_high)
        high = structural_high and not near
        curved = structural_curvature >= case["structural_contract"]["curvature_q75"] and not near
        high_candidates += bool(high)
        high_retained += bool(high and row["barrier_safe"])
        curved_candidates += bool(curved)
        curved_retained += bool(curved and row["barrier_safe"])

    long_crossing = 0
    for edge in graph["edges"]:
        crossing, near = path_fault_qc(edge["path"], faults)
        edge["fault_crossing"] = bool(crossing)
        edge["fault_corridor"] = bool(near)
        long_crossing += bool(crossing)
    spans = [
        case["refined"][group[-1]]["trace"] - case["refined"][group[0]]["trace"]
        for group in graph["components"]
    ]
    valid_count = int(case["arrays"]["valid_mask"].sum())
    return {
        "faults": faults,
        "fault_candidate_links": fault_candidates,
        "fault_split_recall": fault_split / max(fault_candidates, 1),
        "retained_adjacent_fault_crossing_rate": adjacent_safe_crossing / max(adjacent_safe, 1),
        "final_long_edge_fault_crossing_rate": long_crossing / max(len(graph["edges"]), 1),
        "high_dip_candidates": high_candidates,
        "high_dip_retention": high_retained / max(high_candidates, 1),
        "curved_candidates": curved_candidates,
        "curved_reflector_retention": curved_retained / max(curved_candidates, 1),
        "shift_structural_high_dip_agreement": float(np.mean(agreement)) if agreement else 0.0,
        "component_count": len(spans),
        "small_component_fraction": float(np.mean(np.asarray(spans) < 16)) if spans else 1.0,
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "node_fraction": len(graph["nodes"]) / max(valid_count, 1),
        "two_hop_reach_mean": graph["statistics"]["two_hop_reach_mean"],
        "fault_offset_correspondence_count": len(graph["metadata"]),
    }


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    plt.close(figure)
    u.write(OUT / "figures" / name, buffer.getvalue())


def seismic_background(axis: plt.Axes, data: np.ndarray, ylabel: str = "Time sample") -> None:
    limit = np.quantile(np.abs(data), 0.98)
    axis.imshow(data, cmap="gray", aspect="auto", vmin=-limit, vmax=limit)
    axis.set(xlabel="Trace", ylabel=ylabel)


def link_lines(case: dict[str, Any], key: str | None = None) -> list[np.ndarray]:
    lines = []
    for row in case["graph"]["links"]:
        if not row["reciprocal"] or (key is not None and not row[key]):
            continue
        left, right = case["refined"][row["source"]], case["refined"][row["target"]]
        lines.append(
            np.asarray(
                [
                    [left["trace"], left["refined_time"]],
                    [right["trace"], right["refined_time"]],
                ]
            )
        )
    return lines


def make_figures(cases: list[dict[str, Any]]) -> None:
    example = cases[2]
    fault = cases[-1]
    mapping = example["mapping"]
    figure, axis = plt.subplots(figsize=(10, 6))
    seismic_background(axis, mapping["avo"][0], "Coarse RGT-grid sample")
    axis.set_title("1. Coarse flattened seismic is a discovery image only")
    save_figure(figure, "01_coarse_flattened_seismic.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    seismic_background(axis, mapping["avo"][0], "Coarse RGT-grid sample")
    candidates = example["detection"]["candidates"]
    axis.scatter(
        [row["trace"] for row in candidates], [row["tau_index"] for row in candidates], s=2
    )
    axis.set_title("2. Coarse ridge candidates")
    save_figure(figure, "02_coarse_ridge_candidates.png")

    refined_y = np.interp(
        [row["refined_tau"] for row in example["refined"]],
        mapping["tau_grid"],
        np.arange(len(mapping["tau_grid"])),
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    seismic_background(axis, mapping["avo"][0], "Coarse RGT-grid sample")
    axis.scatter([row["trace"] for row in example["refined"]], refined_y, s=2, c="tab:orange")
    axis.set_title("3. Continuously refined ridge coordinates")
    save_figure(figure, "03_refined_continuous_ridges.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    seismic_background(axis, example["arrays"]["avo"][0])
    axis.scatter(
        [row["trace"] for row in example["refined"]],
        [row["refined_time"] for row in example["refined"]],
        s=2,
        c="tab:orange",
    )
    axis.set_title("4. Refined ridges mapped by native trace-wise inverse")
    save_figure(figure, "04_native_inverse_ridges_tx.png")

    native = fault["native"]
    tau_grid = fault["mapping"]["tau_grid"]
    inverse = np.stack(
        [native.inverse_time(trace, tau_grid) for trace in range(native.tau.shape[1])], axis=1
    )
    shift = np.diff(inverse, axis=1)
    jump = np.abs(shift - median_filter(shift, size=(1, 9), mode="nearest"))
    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(shift, aspect="auto", cmap="coolwarm")
    figure.colorbar(image, ax=axis, label="Native physical shift (samples)")
    axis.set(title="5. Native inverse shift field", xlabel="Trace boundary", ylabel="Tau")
    save_figure(figure, "05_native_inverse_shift_field.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(jump, aspect="auto", cmap="magma")
    figure.colorbar(image, ax=axis, label="Shift discontinuity (samples)")
    axis.set(title="6. Native shift-discontinuity evidence", xlabel="Trace boundary", ylabel="Tau")
    save_figure(figure, "06_shift_discontinuity_barrier.png")

    for number, key, title in [
        (7, None, "Reflector links before barrier"),
        (8, "barrier_safe", "Barrier-safe reflector links after pre-union filter"),
    ]:
        figure, axis = plt.subplots(figsize=(10, 6))
        seismic_background(axis, fault["arrays"]["avo"][0])
        lines = link_lines(fault, key)
        if lines:
            axis.add_collection(LineCollection(lines, colors="tab:cyan", linewidths=0.5))
        axis.set_title(f"{number}. {title}")
        save_figure(
            figure, f"{number:02d}_{'links_before' if key is None else 'links_after'}_barrier.png"
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    seismic_background(axis, example["arrays"]["avo"][0])
    long_paths = [edge["path"][:, ::-1] for edge in example["graph"]["edges"] if edge["span"] >= 16]
    if long_paths:
        axis.add_collection(
            LineCollection(
                long_paths[:: max(1, len(long_paths) // 1500)], colors="tab:cyan", linewidths=0.5
            )
        )
    axis.set_title("9. Long sparse native-geometry graph")
    save_figure(figure, "09_long_sparse_graph_physical.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    seismic_background(axis, fault["arrays"]["avo"][0])
    metadata = []
    for row in fault["graph"]["metadata"]:
        left, right = fault["refined"][row["source"]], fault["refined"][row["target"]]
        metadata.append(
            np.asarray(
                [[left["trace"], left["refined_time"]], [right["trace"], right["refined_time"]]]
            )
        )
    if metadata:
        axis.add_collection(LineCollection(metadata, colors="red", linewidths=0.8))
    for truth_fault in fault["truth"]["faults"]:
        time_axis = np.arange(fault["arrays"]["rgt"].shape[0])
        axis.plot(
            float(truth_fault["column"]) + float(truth_fault["dip"]) * time_axis,
            time_axis,
            "y--",
            linewidth=1,
        )
    axis.set_title("10. Fault-offset correspondences excluded from propagation")
    save_figure(figure, "10_fault_offset_correspondence.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"Output exists; refusing overwrite: {OUT}")
    provenance = verify_provenance(args.expected_commit)
    q_contract = json.loads(u.Q.read_text(encoding="utf-8"))
    train_ids = q_contract["split_ids"]["train"]
    indices = np.linspace(0, len(train_ids) - 1, CONFIG["calibration_training_count"], dtype=int)
    calibration_ids = [train_ids[index] for index in indices]
    validation_ids = q_contract["diverse_validation_subset"]["all_ids"]
    contract = {
        "revision": "v00332x",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **provenance,
        "config": CONFIG,
        "calibration_training_ids": calibration_ids,
        "calibration_selection": "six evenly spaced positions in frozen training split",
        "validation_ids": validation_ids,
        "fault_truth_usage": "validation QC only after choices and thresholds were frozen",
        "coarse_grid_role": "display and candidate discovery only; never authoritative geometry",
        "native_inverse_role": "authoritative nodes, correspondence, shift, and edge paths",
        "training": False,
    }
    u.json_file("v00332x_experiment_contract.json", contract)
    if json.loads((OUT / "v00332x_experiment_contract.json").read_text())["commit_sha"] != git(
        "rev-parse", "HEAD"
    ):
        raise RuntimeError("Written experiment contract does not match HEAD")

    calibration = []
    for resolution in CONFIG["discovery_resolutions"]:
        for realization_id in calibration_ids:
            for width in CONFIG["refinement_width_steps"]:
                u.log(f"calibration {realization_id}: grid={resolution}, width={width}")
                row, _ = calibration_run(realization_id, resolution, width)
                calibration.append(row)
    resolution, width, aggregate = choose_discovery(calibration)
    u.csv_file("calibration_grid.csv", calibration)
    u.csv_file("calibration_choice.csv", aggregate)

    u.log(f"frozen discovery choice: grid={resolution}, width={width}")
    training_links = []
    for realization_id in calibration_ids:
        case = build_case(realization_id, resolution, width)
        training_links.extend(case["links"])
    thresholds = freeze_thresholds(training_links, CONFIG)
    thresholds["discovery_resolution"] = resolution
    thresholds["refinement_width_steps"] = width
    u.json_file("frozen_observable_thresholds.json", thresholds)

    cases = []
    native_rows = []
    localization_rows = []
    node_rows = []
    edge_rows = []
    link_rows = []
    fault_rows = []
    structural_contract = q_contract["adaptive_search"]
    for realization_id in validation_ids:
        u.log(f"validation graph {realization_id}")
        case = build_case(realization_id, resolution, width)
        case["structural_contract"] = structural_contract
        case["graph"] = build_barrier_graph(case["refined"], case["links"], thresholds, CONFIG)
        case["truth"] = annotate_validation(case, thresholds)
        cases.append(case)
        native_rows.append(
            {
                "realization_id": realization_id,
                "native_time_error_max": float(np.max(case["native_time_error"])),
                "native_tau_error_max": float(np.max(case["native_tau_error"])),
                "coarse_flattened_time_error_p99": quantile(
                    case["mapping"]["grid_resampling_time_error_samples"], 0.99
                ),
                "coarse_flattened_ava_error_p99": quantile(
                    case["mapping"]["roundtrip_avo_abs"], 0.99
                ),
                "ambiguous_plateau_fraction": case["mapping"]["ambiguous_plateau_fraction"],
            }
        )
        for row in case["refined"]:
            localization_rows.append({"realization_id": realization_id, **row})
        for row in case["graph"]["nodes"]:
            node_rows.append({"realization_id": realization_id, **row})
        for row in case["graph"]["edges"]:
            edge_rows.append(
                {
                    "realization_id": realization_id,
                    **{key: value for key, value in row.items() if key != "path"},
                }
            )
        for row in case["graph"]["links"]:
            link_rows.append({"realization_id": realization_id, **row})
        fault_rows.append(
            {
                "realization_id": realization_id,
                **{key: value for key, value in case["truth"].items() if key != "faults"},
            }
        )

    u.csv_file("native_geometry_qc.csv", native_rows)
    u.csv_file("ridge_localization_qc.csv", localization_rows)
    u.csv_file("node_statistics.csv", node_rows)
    u.csv_file("edge_statistics.csv", edge_rows)
    u.csv_file("adjacent_link_qc.csv", link_rows)
    u.csv_file("fault_qc.csv", fault_rows)
    make_figures(cases)

    native_frame = pd.DataFrame(native_rows)
    fault_frame = pd.DataFrame(fault_rows)
    localization_frame = pd.DataFrame(localization_rows)
    decision_config = CONFIG["decision"]
    native_ok = bool(
        (native_frame.native_time_error_max <= decision_config["maximum_native_time_error"]).all()
        and (native_frame.native_tau_error_max <= decision_config["maximum_native_tau_error"]).all()
    )
    refinement_ok = bool(
        quantile(abs(localization_frame.time_displacement), 0.99)
        <= decision_config["maximum_refinement_time_p99"]
        and float(localization_frame.refinement_at_boundary.mean())
        <= decision_config["maximum_refinement_boundary_fraction"]
    )
    fault_sections = fault_frame[fault_frame.fault_candidate_links > 0]
    fault_ok = bool(
        len(fault_sections)
        and (
            fault_sections.fault_split_recall >= decision_config["minimum_fault_split_recall"]
        ).all()
        and (
            fault_sections.retained_adjacent_fault_crossing_rate
            <= decision_config["maximum_adjacent_fault_crossing"]
        ).all()
    )
    long_ok = bool(
        len(fault_sections)
        and (
            fault_sections.final_long_edge_fault_crossing_rate
            <= decision_config["maximum_long_fault_crossing"]
        ).all()
    )
    high_sections = fault_frame[fault_frame.high_dip_candidates > 0]
    high_ok = bool(
        len(high_sections)
        and (
            high_sections.high_dip_retention >= decision_config["minimum_high_dip_retention"]
        ).all()
    )
    curved_sections = fault_frame[fault_frame.curved_candidates > 0]
    curved_ok = bool(
        len(curved_sections)
        and (
            curved_sections.curved_reflector_retention
            >= decision_config["minimum_curved_retention"]
        ).all()
    )
    sparse_long_ok = bool(
        (fault_frame.node_fraction <= decision_config["maximum_node_fraction"]).all()
        and (
            fault_frame.small_component_fraction
            <= decision_config["maximum_small_component_fraction"]
        ).all()
        and (fault_frame.two_hop_reach_mean >= decision_config["minimum_two_hop_reach"]).all()
    )
    implementation_ok = bool(len(node_rows) and len(edge_rows))
    if not implementation_ok or not native_ok:
        decision = "IMPLEMENTATION_PROBLEM"
    elif not refinement_ok:
        decision = "RIDGE_REFINEMENT_UNSTABLE"
    elif not fault_ok or not long_ok:
        decision = "FAULT_SPLIT_STILL_WEAK"
    elif not high_ok:
        decision = "HIGH_DIP_OVERPRUNED"
    elif not curved_ok or not sparse_long_ok:
        decision = "COMPONENTS_TOO_FRAGMENTED"
    else:
        decision = "NATIVE_RGT_COMPONENT_GRAPH_PROMISING"
    summary = {
        "decision": decision,
        "native_geometry_status": "ACCURATE" if native_ok else "FAIL",
        "ridge_refinement_status": "STABLE" if refinement_ok else "FAIL",
        "fault_split_status": "PASS" if fault_ok else "FAIL",
        "long_edge_fault_status": "PASS" if long_ok else "FAIL",
        "high_dip_retention": "PASS" if high_ok else "FAIL",
        "curved_reflector_retention": "PASS" if curved_ok else "FAIL",
        "sparse_long_range_status": "PASS" if sparse_long_ok else "FAIL",
        "selected_discovery_resolution": resolution,
        "selected_refinement_width_steps": width,
        "frozen_thresholds": thresholds,
        "training_ready": decision == "NATIVE_RGT_COMPONENT_GRAPH_PROMISING",
        "training_performed": False,
        "aggregate_validation": {
            column: float(fault_frame[column].mean())
            for column in [
                "fault_split_recall",
                "retained_adjacent_fault_crossing_rate",
                "final_long_edge_fault_crossing_rate",
                "high_dip_retention",
                "curved_reflector_retention",
                "shift_structural_high_dip_agreement",
                "small_component_fraction",
                "node_fraction",
                "two_hop_reach_mean",
            ]
        },
    }
    u.json_file("v00332x_summary.json", summary)
    report = f"""# v00332x — native-RGT geometry and coarse flattened discovery

Decision: **{decision}**

NATIVE_GEOMETRY_STATUS: {summary["native_geometry_status"]}
RIDGE_REFINEMENT_STATUS: {summary["ridge_refinement_status"]}
FAULT_SPLIT_STATUS: {summary["fault_split_status"]}
LONG_EDGE_FAULT_STATUS: {summary["long_edge_fault_status"]}
HIGH_DIP_RETENTION: {summary["high_dip_retention"]}

The authoritative graph geometry is the direct per-trace monotone native RGT inverse. The selected
{resolution}-sample common grid is used only as a reflector-search image; it does not define nodes,
shift barriers, cross-trace correspondence, or edge paths. Training-only observable calibration
selected a refinement half-width of {width} grid steps and froze the following thresholds:

```json
{json.dumps(thresholds, indent=2)}
```

Native, coarse-image, and ridge-localization errors are reported separately. Fault truth was opened
only for validation QC after all choices and thresholds were frozen. A proposed reciprocal reflector
match can be retained, blocked before union, or preserved as FAULT_OFFSET_CORRESPONDENCE metadata.
Every long edge requires a complete barrier-safe adjacent path; shared labels alone are insufficient.

Mean validation QC:

```json
{json.dumps(summary["aggregate_validation"], indent=2)}
```

No model, checkpoint, optimizer, or training path was used. Training is {"eligible for a future matched ablation" if summary["training_ready"] else "not authorized by these QC results"}.
"""
    u.write(OUT / "v00332x_native_rgt_component_graph_report.md", report.encode())

    for relative, digest in provenance["source_config_test_sha256"].items():
        if u.sha(u.REPO / relative) != digest:
            raise RuntimeError(f"Protected source changed during run: {relative}")
    if git("rev-parse", "HEAD") != args.expected_commit:
        raise RuntimeError("HEAD changed during scientific run")
    u.log(f"{decision}; no training: {OUT}")


if __name__ == "__main__":
    main()
