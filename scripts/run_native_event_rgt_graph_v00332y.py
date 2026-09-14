#!/usr/bin/env python3
"""Calibrate and validate the v00332y native seismic-event graph; no training."""

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

import run_skeleton_graph_v00332u as u
from sage_avo.config import load_config
from sage_avo.diagnostics.native_event_graph import (
    build_event_graph,
    candidate_event_pairs,
    detect_physical_events,
    event_repeatability,
    freeze_event_barriers,
    normal_event_contrasts,
    reciprocal_matches,
    robust_match_scales,
)
from sage_avo.diagnostics.native_rgt_graph import NativeRGT
from sage_avo.diagnostics.rgt_topology_repair import structural_fields
from sage_avo.diagnostics.skeleton_graph import path_fault_qc, sample

BRANCH = "experiment/v00332y-native-event-rgt-graph"
CONFIG_PATH = u.REPO / "configs/development_diagnostics_v00332y.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = u.BASE / "stage04" / CONFIG["experiment_name"]
X_OUT = u.BASE / "stage04/sage_avo_s01_v00332x_native_rgt_component_graph"
u.OUT = OUT
PROTECTED_SOURCE = [
    CONFIG_PATH,
    Path(__file__),
    u.REPO / "scripts/run_skeleton_graph_v00332u.py",
    u.REPO / "src/sage_avo/diagnostics/native_event_graph.py",
    u.REPO / "src/sage_avo/diagnostics/native_rgt_graph.py",
    u.REPO / "src/sage_avo/diagnostics/rgt_topology_repair.py",
    u.REPO / "src/sage_avo/diagnostics/skeleton_graph.py",
    u.REPO / "tests/test_native_event_graph.py",
]


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=u.REPO, text=True).strip()


def provenance(expected_commit: str) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("rev-parse", f"refs/remotes/origin/{BRANCH}")
    status = git("status", "--porcelain", "--untracked-files=no")
    if branch != BRANCH or head != expected_commit or remote != expected_commit or status:
        raise RuntimeError(
            f"Provenance gate failed: branch={branch}, HEAD={head}, remote={remote}, dirty={bool(status)}"
        )
    return {
        "repository": str(u.REPO),
        "branch": branch,
        "commit_sha": head,
        "parent_commit_sha": git("rev-parse", "HEAD^"),
        "remote_tracking_commit_sha": remote,
        "tracked_worktree_clean_at_start": True,
        "source_config_test_sha256": {
            str(path.relative_to(u.REPO)): u.sha(path) for path in PROTECTED_SOURCE
        },
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


def detector_config(candidate: dict[str, Any]) -> dict[str, Any]:
    return {**CONFIG, **candidate}


def add_structural_attributes(events: list[dict[str, Any]], tau: np.ndarray) -> None:
    fields = structural_fields(tau)
    for event in events:
        point = np.asarray([[event["time"], event["trace"]]])
        event["structural_dip"] = float(sample(fields["dip"], point)[0])
        event["structural_curvature"] = float(sample(fields["curvature"], point)[0])


def detect_case(realization_id: int, selected_detector: dict[str, Any]) -> dict[str, Any]:
    arrays = load_arrays(realization_id)
    events = detect_physical_events(
        arrays["avo"], arrays["rgt"], arrays["valid_mask"], selected_detector
    )
    repeat_events = detect_physical_events(
        arrays["avo"],
        arrays["rgt"],
        arrays["valid_mask"],
        selected_detector,
        channels=CONFIG["repeatability_channels"],
    )
    add_structural_attributes(events, arrays["rgt"])
    add_structural_attributes(repeat_events, arrays["rgt"])
    return {
        "rid": realization_id,
        "arrays": arrays,
        "native": NativeRGT(arrays["rgt"]),
        "events": events,
        "repeat_events": repeat_events,
        "repeatability": event_repeatability(
            events, repeat_events, CONFIG["repeatability_tolerance_samples"]
        ),
    }


def matched_event_fraction(events: list[dict[str, Any]], rows: list[dict[str, Any]]) -> float:
    used = set()
    for row in rows:
        if row["reflector_match"]:
            used.update((row["source"], row["target"]))
    return len(used) / max(len(events), 1)


def association_margin(rows: list[dict[str, Any]]) -> float:
    by_source: dict[int, list[float]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row["match_cost"])
    margins = []
    for costs in by_source.values():
        costs.sort()
        if len(costs) > 1:
            margins.append(costs[1] - costs[0])
    return float(np.median(margins)) if margins else 0.0


def calibration(
    train_ids: list[int], q_contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    detector_rows = []
    for candidate in CONFIG["event_detector_candidates"]:
        configured = detector_config(candidate)
        for realization_id in train_ids:
            arrays = load_arrays(realization_id)
            events = detect_physical_events(
                arrays["avo"], arrays["rgt"], arrays["valid_mask"], configured
            )
            repeat = detect_physical_events(
                arrays["avo"],
                arrays["rgt"],
                arrays["valid_mask"],
                configured,
                channels=CONFIG["repeatability_channels"],
            )
            detector_rows.append(
                {
                    "detector": candidate["name"],
                    "realization_id": realization_id,
                    **event_repeatability(
                        events, repeat, CONFIG["repeatability_tolerance_samples"]
                    ),
                    "event_fraction": len(events) / arrays["valid_mask"].sum(),
                }
            )
    detector_summary = []
    for candidate in CONFIG["event_detector_candidates"]:
        subset = [row for row in detector_rows if row["detector"] == candidate["name"]]
        detector_summary.append(
            {
                **candidate,
                "mean_repeatability": float(np.mean([row["repeatability"] for row in subset])),
                "worst_error_p95": max(row["error_p95"] for row in subset),
                "mean_event_fraction": float(np.mean([row["event_fraction"] for row in subset])),
            }
        )
    selected_detector = min(
        detector_summary,
        key=lambda row: (-row["mean_repeatability"], row["worst_error_p95"], row["event_quantile"]),
    )
    configured_detector = detector_config(selected_detector)
    train_cases = [detect_case(realization_id, configured_detector) for realization_id in train_ids]

    association_rows = []
    association_cache: dict[
        tuple[float, str], tuple[list[list[dict[str, Any]]], dict[str, float]]
    ] = {}
    for radius in CONFIG["search_radius_candidates"]:
        candidates_by_case = [
            candidate_event_pairs(case["native"], case["events"], radius) for case in train_cases
        ]
        scales = robust_match_scales([row for rows in candidates_by_case for row in rows])
        for weighting in CONFIG["match_weightings"]:
            matched_by_case = [
                reciprocal_matches(rows, scales, weighting["weights"], CONFIG)
                for rows in candidates_by_case
            ]
            association_cache[(radius, weighting["name"])] = (matched_by_case, scales)
            fractions = [
                matched_event_fraction(case["events"], rows)
                for case, rows in zip(train_cases, matched_by_case)
            ]
            association_rows.append(
                {
                    "search_radius": radius,
                    "weighting": weighting["name"],
                    "mean_matched_event_fraction": float(np.mean(fractions)),
                    "minimum_matched_event_fraction": min(fractions),
                    "median_cost_margin": float(
                        np.median([association_margin(rows) for rows in matched_by_case])
                    ),
                    "scales_json": json.dumps(scales, sort_keys=True),
                }
            )
    selected_association = min(
        association_rows,
        key=lambda row: (
            -row["minimum_matched_event_fraction"],
            -row["median_cost_margin"],
            row["search_radius"],
            row["weighting"],
        ),
    )
    radius = selected_association["search_radius"]
    weighting_name = selected_association["weighting"]
    selected_weighting = next(
        row for row in CONFIG["match_weightings"] if row["name"] == weighting_name
    )
    matched_by_case, scales = association_cache[(radius, weighting_name)]
    thresholds = freeze_event_barriers([row for rows in matched_by_case for row in rows], CONFIG)
    curvature_threshold = q_contract["adaptive_search"]["curvature_q75"]
    spacing_rows = []
    for spacing in CONFIG["node_spacing_candidates"]:
        fractions, reaches, fragmentation = [], [], []
        for case, matches in zip(train_cases, matched_by_case):
            graph = build_event_graph(
                case["events"],
                matches,
                thresholds,
                CONFIG,
                node_spacing=spacing,
                curvature_threshold=curvature_threshold,
            )
            spans = [
                case["events"][group[-1]]["trace"] - case["events"][group[0]]["trace"]
                for group in graph["components"]
            ]
            fractions.append(len(graph["nodes"]) / case["arrays"]["valid_mask"].sum())
            reaches.append(graph["statistics"]["two_hop_reach_mean"])
            fragmentation.append(float(np.mean(np.asarray(spans) < 16)) if spans else 1.0)
        spacing_rows.append(
            {
                "node_spacing": spacing,
                "mean_node_fraction": float(np.mean(fractions)),
                "minimum_two_hop_reach": min(reaches),
                "maximum_small_component_fraction": max(fragmentation),
            }
        )
    target = 0.5 * (
        CONFIG["decision"]["minimum_node_fraction"] + CONFIG["decision"]["maximum_node_fraction"]
    )
    eligible = [
        row
        for row in spacing_rows
        if CONFIG["decision"]["minimum_node_fraction"]
        <= row["mean_node_fraction"]
        <= CONFIG["decision"]["maximum_node_fraction"]
        and row["minimum_two_hop_reach"] >= CONFIG["decision"]["minimum_two_hop_reach"]
    ]
    selected_spacing = min(
        eligible or spacing_rows,
        key=lambda row: (abs(row["mean_node_fraction"] - target), row["node_spacing"]),
    )
    frozen = {
        "detector": configured_detector,
        "search_radius": radius,
        "weighting": selected_weighting,
        "match_scales": scales,
        "barrier_thresholds": thresholds,
        "node_spacing": selected_spacing["node_spacing"],
        "curvature_preservation_threshold": curvature_threshold,
    }
    tables = {
        "detector_rows": detector_rows,
        "detector_summary": detector_summary,
        "association_rows": association_rows,
        "spacing_rows": spacing_rows,
    }
    return frozen, tables, train_cases, selected_detector


def annotate_truth(case: dict[str, Any], q_contract: dict[str, Any]) -> dict[str, Any]:
    faults = u.load_faults(u.STAGE02, case["rid"])
    graph = case["graph"]
    fault_candidates = split = retained = retained_crossing = 0
    high_count = high_kept = curved_count = curved_kept = 0
    for row in graph["matches"]:
        if not row["reflector_match"]:
            continue
        left, right = case["events"][row["source"]], case["events"][row["target"]]
        path = np.asarray([[left["time"], left["trace"]], [right["time"], right["trace"]]])
        crossing, near = path_fault_qc(path, faults)
        row["fault_crossing"] = bool(crossing)
        row["fault_corridor"] = bool(near)
        fault_candidates += bool(crossing)
        split += bool(crossing and row["fault_barrier"])
        retained += bool(row["barrier_safe"])
        retained_crossing += bool(row["barrier_safe"] and crossing)
        high = row["source_dip"] >= q_contract["adaptive_search"]["dip_q67"] and not near
        curved = (
            row["source_curvature"] >= q_contract["adaptive_search"]["curvature_q75"] and not near
        )
        high_count += bool(high)
        high_kept += bool(high and row["barrier_safe"])
        curved_count += bool(curved)
        curved_kept += bool(curved and row["barrier_safe"])
    long_crossing = 0
    for edge in graph["edges"]:
        crossing, near = path_fault_qc(edge["path"], faults)
        edge["fault_crossing"] = bool(crossing)
        edge["fault_corridor"] = bool(near)
        long_crossing += bool(crossing)
    spans = [
        case["events"][group[-1]]["trace"] - case["events"][group[0]]["trace"]
        for group in graph["components"]
    ]
    return {
        "faults": faults,
        "fault_candidate_links": fault_candidates,
        "fault_split_recall": split / max(fault_candidates, 1),
        "retained_adjacent_crossing_rate": retained_crossing / max(retained, 1),
        "long_edge_fault_crossing_rate": long_crossing / max(len(graph["edges"]), 1),
        "high_dip_candidates": high_count,
        "high_dip_retention": high_kept / max(high_count, 1),
        "curved_candidates": curved_count,
        "curved_retention": curved_kept / max(curved_count, 1),
        "event_count": len(case["events"]),
        "matched_event_fraction": matched_event_fraction(case["events"], graph["matches"]),
        "unmatched_event_fraction": 1 - matched_event_fraction(case["events"], graph["matches"]),
        "component_count": len(spans),
        "median_component_span": float(np.median(spans)) if spans else 0.0,
        "small_component_fraction": float(np.mean(np.asarray(spans) < 16)) if spans else 1.0,
        "node_fraction": len(graph["nodes"]) / case["arrays"]["valid_mask"].sum(),
        "long_edge_count": sum(edge["span"] >= 16 for edge in graph["edges"]),
        "two_hop_reach_mean": graph["statistics"]["two_hop_reach_mean"],
        "fault_offset_correspondence_count": len(graph["metadata"]),
        "component_span_stability": case["component_span_stability"],
        "repeat_component_count": len(case["repeat_graph"]["components"]),
    }


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170)
    plt.close(figure)
    u.write(OUT / "figures" / name, buffer.getvalue())


def background(axis: plt.Axes, case: dict[str, Any]) -> None:
    seismic = case["arrays"]["avo"][0]
    limit = np.quantile(abs(seismic), 0.98)
    axis.imshow(seismic, cmap="gray", aspect="auto", vmin=-limit, vmax=limit)
    axis.set(xlabel="Trace", ylabel="Time sample")


def match_lines(case: dict[str, Any], selector: str) -> list[np.ndarray]:
    lines = []
    for row in case["graph"]["matches"]:
        if not row.get(selector, False):
            continue
        left, right = case["events"][row["source"]], case["events"][row["target"]]
        lines.append(np.asarray([[left["trace"], left["time"]], [right["trace"], right["time"]]]))
    return lines


def figures(cases: list[dict[str, Any]]) -> None:
    example, high_dip, fault = cases[2], cases[3], cases[-1]
    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    axis.scatter(
        [e["trace"] for e in example["events"]],
        [e["time"] for e in example["events"]],
        s=2,
        c="tab:orange",
    )
    axis.set_title("1. Physical-time seismic event candidates")
    save_figure(figure, "01_physical_event_candidates.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    scatter = axis.scatter(
        [e["trace"] for e in example["events"]],
        [e["time"] for e in example["events"]],
        s=3,
        c=[e["tau"] for e in example["events"]],
        cmap="turbo",
    )
    figure.colorbar(scatter, ax=axis, label="Native continuous RGT")
    axis.set_title("2. Events colored by native RGT")
    save_figure(figure, "02_events_native_rgt.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    predictions = []
    for row in example["graph"]["matches"][::8]:
        source = example["events"][row["source"]]
        predictions.append(
            np.asarray(
                [[source["trace"], source["time"]], [source["trace"] + 1, row["predicted_time"]]]
            )
        )
    if predictions:
        axis.add_collection(LineCollection(predictions, colors="tab:purple", linewidths=0.6))
    axis.set_title("3. Continuous RGT-predicted cross-trace positions")
    save_figure(figure, "03_rgt_predicted_correspondence.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    reciprocal = match_lines(example, "reflector_match")
    if reciprocal:
        axis.add_collection(LineCollection(reciprocal, colors="tab:cyan", linewidths=0.5))
    axis.set_title("4. Reciprocal physical-event matches")
    save_figure(figure, "04_reciprocal_event_matches.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, fault)
    safe, rejected = match_lines(fault, "barrier_safe"), match_lines(fault, "fault_barrier")
    if safe:
        axis.add_collection(LineCollection(safe, colors="tab:cyan", linewidths=0.5))
    if rejected:
        axis.add_collection(LineCollection(rejected, colors="red", linewidths=0.7))
    axis.set_title("5. Barrier-safe (cyan) versus rejected (red) matches")
    save_figure(figure, "05_barrier_safe_rejected.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    for component, group in enumerate(example["graph"]["components"]):
        event = [example["events"][index] for index in group]
        axis.plot(
            [e["trace"] for e in event],
            [e["time"] for e in event],
            linewidth=0.8,
            color=plt.get_cmap("tab20")(component % 20),
        )
    axis.set_title("6. Connected reflector components")
    save_figure(figure, "06_connected_components.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    nodes = example["graph"]["nodes"]
    axis.scatter(
        [n["trace"] for n in nodes],
        [n["time"] for n in nodes],
        s=7,
        c=[n["component"] for n in nodes],
        cmap="tab20",
    )
    axis.set_title("7. Sparse component nodes")
    save_figure(figure, "07_sparse_nodes.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, example)
    paths = [edge["path"][:, ::-1] for edge in example["graph"]["edges"] if edge["span"] >= 16]
    if paths:
        axis.add_collection(
            LineCollection(paths[:: max(1, len(paths) // 1500)], colors="tab:cyan", linewidths=0.5)
        )
    axis.set_title("8. Variable-length long event-component edges")
    save_figure(figure, "08_variable_length_long_edges.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, high_dip)
    paths = [edge["path"][:, ::-1] for edge in high_dip["graph"]["edges"]]
    if paths:
        axis.add_collection(
            LineCollection(paths[:: max(1, len(paths) // 1500)], colors="tab:cyan", linewidths=0.5)
        )
    axis.set_title("9. High-dip continuous event graph")
    save_figure(figure, "09_high_dip_continuous.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    background(axis, fault)
    rejected = match_lines(fault, "fault_barrier")
    if rejected:
        axis.add_collection(LineCollection(rejected, colors="red", linewidths=0.8))
    time_axis = np.arange(fault["arrays"]["rgt"].shape[0])
    for item in fault["truth"]["faults"]:
        axis.plot(
            float(item["column"]) + float(item["dip"]) * time_axis, time_axis, "y--", linewidth=1
        )
    axis.set_title("10. FAULT_OFFSET_CORRESPONDENCE (red); fault truth is QC only")
    save_figure(figure, "10_fault_offset_correspondence.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"Output exists; refusing overwrite: {OUT}")
    git_record = provenance(args.expected_commit)
    q_contract = json.loads(u.Q.read_text(encoding="utf-8"))
    all_train = q_contract["split_ids"]["train"]
    positions = np.linspace(0, len(all_train) - 1, CONFIG["calibration_training_count"], dtype=int)
    train_ids = [all_train[position] for position in positions]
    validation_ids = q_contract["diverse_validation_subset"]["all_ids"]
    u.json_file(
        "v00332y_experiment_contract.json",
        {
            "revision": "v00332y",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **git_record,
            "config": CONFIG,
            "calibration_training_ids": train_ids,
            "validation_ids": validation_ids,
            "fault_truth_usage": "validation QC only after all choices were frozen",
            "event_location_domain": "observed physical time; no flattened grid",
            "rgt_role": "continuous coordinate and adjacent-trace prediction",
            "training": False,
        },
    )
    frozen, tables, _, selected_detector = calibration(train_ids, q_contract)
    for name, rows in tables.items():
        u.csv_file(f"calibration_{name}.csv", rows)
    u.json_file("v00332y_frozen_contract.json", frozen)
    selected_weighting = frozen["weighting"]["weights"]
    cases = []
    event_rows = []
    match_rows = []
    component_rows = []
    node_rows = []
    edge_rows = []
    normal_rows = []
    qc_rows = []
    for realization_id in validation_ids:
        u.log(f"v00332y validation {realization_id}")
        case = detect_case(realization_id, frozen["detector"])
        pairs = candidate_event_pairs(case["native"], case["events"], frozen["search_radius"])
        matches = reciprocal_matches(pairs, frozen["match_scales"], selected_weighting, CONFIG)
        graph = build_event_graph(
            case["events"],
            matches,
            frozen["barrier_thresholds"],
            CONFIG,
            node_spacing=frozen["node_spacing"],
            curvature_threshold=frozen["curvature_preservation_threshold"],
        )
        case["graph"] = graph
        repeat_pairs = candidate_event_pairs(
            case["native"], case["repeat_events"], frozen["search_radius"]
        )
        repeat_matches = reciprocal_matches(
            repeat_pairs, frozen["match_scales"], selected_weighting, CONFIG
        )
        repeat_graph = build_event_graph(
            case["repeat_events"],
            repeat_matches,
            frozen["barrier_thresholds"],
            CONFIG,
            node_spacing=frozen["node_spacing"],
            curvature_threshold=frozen["curvature_preservation_threshold"],
        )
        case["repeat_graph"] = repeat_graph
        full_span = sum(
            case["events"][group[-1]]["trace"] - case["events"][group[0]]["trace"]
            for group in graph["components"]
        )
        repeat_span = sum(
            case["repeat_events"][group[-1]]["trace"] - case["repeat_events"][group[0]]["trace"]
            for group in repeat_graph["components"]
        )
        case["component_span_stability"] = min(full_span, repeat_span) / max(
            full_span, repeat_span, 1
        )
        case["truth"] = annotate_truth(case, q_contract)
        cases.append(case)
        for row in case["events"]:
            event_rows.append(
                {
                    "realization_id": realization_id,
                    **{key: value for key, value in row.items() if key != "waveform"},
                }
            )
        for row in graph["matches"]:
            match_rows.append({"realization_id": realization_id, **row})
        for component, group in enumerate(graph["components"]):
            traces = [case["events"][index]["trace"] for index in group]
            component_rows.append(
                {
                    "realization_id": realization_id,
                    "component": component,
                    "event_count": len(group),
                    "trace_start": min(traces),
                    "trace_end": max(traces),
                    "trace_span": max(traces) - min(traces),
                }
            )
        for row in graph["nodes"]:
            node_rows.append(
                {
                    "realization_id": realization_id,
                    **{key: value for key, value in row.items() if key != "waveform"},
                }
            )
        for row in graph["edges"]:
            edge_rows.append(
                {
                    "realization_id": realization_id,
                    **{key: value for key, value in row.items() if key != "path"},
                }
            )
        for row in normal_event_contrasts(
            case["arrays"]["rgt"],
            case["arrays"]["avo"],
            graph["nodes"],
            CONFIG["normal_offset_samples"],
        ):
            normal_rows.append({"realization_id": realization_id, **row})
        qc_rows.append(
            {
                "realization_id": realization_id,
                **case["repeatability"],
                **{key: value for key, value in case["truth"].items() if key != "faults"},
            }
        )
    for name, rows in {
        "event_statistics.csv": event_rows,
        "match_statistics.csv": match_rows,
        "component_statistics.csv": component_rows,
        "node_statistics.csv": node_rows,
        "edge_statistics.csv": edge_rows,
        "normal_contrast_statistics.csv": normal_rows,
        "validation_qc.csv": qc_rows,
    }.items():
        u.csv_file(name, rows)
    figures(cases)

    qc = pd.DataFrame(qc_rows)
    decision_config = CONFIG["decision"]
    localization_ok = bool(
        (qc.repeatability >= decision_config["minimum_event_repeatability"]).all()
        and (qc.error_p95 <= decision_config["maximum_repeatability_error_p95_samples"]).all()
    )
    association_ok = bool(
        (qc.matched_event_fraction >= decision_config["minimum_matched_event_fraction"]).all()
    )
    fault_cases = qc[qc.fault_candidate_links > 0]
    fault_ok = bool(
        len(fault_cases)
        and (fault_cases.fault_split_recall >= decision_config["minimum_fault_split_recall"]).all()
        and (
            fault_cases.retained_adjacent_crossing_rate
            <= decision_config["maximum_adjacent_fault_crossing"]
        ).all()
    )
    long_ok = bool(
        len(fault_cases)
        and (
            fault_cases.long_edge_fault_crossing_rate
            <= decision_config["maximum_long_fault_crossing"]
        ).all()
    )
    high_cases = qc[qc.high_dip_candidates > 0]
    high_ok = bool(
        len(high_cases)
        and (high_cases.high_dip_retention >= decision_config["minimum_high_dip_retention"]).all()
    )
    curved_cases = qc[qc.curved_candidates > 0]
    curved_ok = bool(
        len(curved_cases)
        and (curved_cases.curved_retention >= decision_config["minimum_curved_retention"]).all()
    )
    sparse_ok = bool(
        (qc.node_fraction <= decision_config["maximum_node_fraction"]).all()
        and (qc.node_fraction >= decision_config["minimum_node_fraction"]).all()
        and (qc.two_hop_reach_mean >= decision_config["minimum_two_hop_reach"]).all()
    )
    fragmented = bool(
        (qc.small_component_fraction > decision_config["maximum_small_component_fraction"]).any()
        or (qc.component_span_stability < decision_config["minimum_component_span_stability"]).any()
    )
    if not len(event_rows) or not len(edge_rows):
        decision = "IMPLEMENTATION_PROBLEM"
    elif not localization_ok:
        decision = "EVENT_DETECTION_UNSTABLE"
    elif not association_ok:
        decision = "RGT_ASSOCIATION_WEAK"
    elif not fault_ok or not long_ok:
        decision = "FAULT_SPLIT_WEAK"
    elif not high_ok:
        decision = "HIGH_DIP_OVERPRUNED"
    elif fragmented or not sparse_ok or not curved_ok:
        decision = "COMPONENTS_TOO_FRAGMENTED"
    else:
        decision = "NATIVE_EVENT_RGT_GRAPH_PROMISING"

    x_summary = json.loads((X_OUT / "v00332x_summary.json").read_text(encoding="utf-8"))
    comparison = [
        {"metric": "boundary_hit_concept", "v00332x": "applicable and failed", "v00332y": "N/A"},
        {
            "metric": "event_localization_repeatability",
            "v00332x": "coarse/refined displacement unstable",
            "v00332y": float(qc.repeatability.mean()),
        },
        {
            "metric": "component_fragmentation",
            "v00332x": x_summary["aggregate_validation"]["small_component_fraction"],
            "v00332y": float(qc.small_component_fraction.mean()),
        },
        {
            "metric": "component_span_stability",
            "v00332x": "not measured",
            "v00332y": float(qc.component_span_stability.mean()),
        },
        {
            "metric": "fault_split_recall",
            "v00332x": x_summary["aggregate_validation"]["fault_split_recall_fault_bearing_mean"],
            "v00332y": float(fault_cases.fault_split_recall.mean()),
        },
        {
            "metric": "high_dip_retention",
            "v00332x": x_summary["aggregate_validation"]["high_dip_retention"],
            "v00332y": float(qc.high_dip_retention.mean()),
        },
        {
            "metric": "two_hop_reach",
            "v00332x": x_summary["aggregate_validation"]["two_hop_reach_mean"],
            "v00332y": float(qc.two_hop_reach_mean.mean()),
        },
        {
            "metric": "node_fraction",
            "v00332x": x_summary["aggregate_validation"]["node_fraction"],
            "v00332y": float(qc.node_fraction.mean()),
        },
    ]
    u.csv_file("v00332x_vs_v00332y.csv", comparison)
    aggregate = {
        "event_repeatability_mean": float(qc.repeatability.mean()),
        "event_repeatability_error_p95_worst": float(qc.error_p95.max()),
        "matched_event_fraction_mean": float(qc.matched_event_fraction.mean()),
        "fault_split_recall_fault_bearing_mean": float(fault_cases.fault_split_recall.mean()),
        "long_edge_fault_crossing_fault_bearing_mean": float(
            fault_cases.long_edge_fault_crossing_rate.mean()
        ),
        "high_dip_retention_mean": float(qc.high_dip_retention.mean()),
        "curved_retention_mean": float(qc.curved_retention.mean()),
        "node_fraction_mean": float(qc.node_fraction.mean()),
        "two_hop_reach_mean": float(qc.two_hop_reach_mean.mean()),
        "component_span_stability_mean": float(qc.component_span_stability.mean()),
    }
    summary = {
        "decision": decision,
        "event_localization_status": "STABLE" if localization_ok else "FAIL",
        "rgt_association_status": "PASS" if association_ok else "FAIL",
        "fault_split_status": "PASS" if fault_ok else "FAIL",
        "long_edge_fault_status": "PASS" if long_ok else "FAIL",
        "high_dip_retention": "PASS" if high_ok else "FAIL",
        "curved_retention_status": "PASS" if curved_ok else "FAIL",
        "sparsity_long_range_status": "PASS" if sparse_ok else "FAIL",
        "selected_detector": selected_detector,
        "frozen_choices": frozen,
        "aggregate_validation": aggregate,
        "training_ready": decision == "NATIVE_EVENT_RGT_GRAPH_PROMISING",
        "training_performed": False,
    }
    u.json_file("v00332y_summary.json", summary)
    report = f"""# v00332y — native seismic-event graph with RGT-guided association

Decision: **{decision}**

EVENT_LOCALIZATION_STATUS: {summary["event_localization_status"]}
RGT_ASSOCIATION_STATUS: {summary["rgt_association_status"]}
FAULT_SPLIT_STATUS: {summary["fault_split_status"]}
LONG_EDGE_FAULT_STATUS: {summary["long_edge_fault_status"]}
HIGH_DIP_RETENTION: {summary["high_dip_retention"]}
SPARSITY_LONG_RANGE_STATUS: {summary["sparsity_long_range_status"]}

Physical-time, phase-stable signed-amplitude extrema define every event. Native RGT is evaluated
continuously at that event and predicts the adjacent-trace search position; no flattened grid is
constructed or used. Training-only calibration froze the detector, search radius, robust cost
scales, one predeclared weighting, barrier thresholds, and node spacing before validation truth was
opened.

```json
{json.dumps(aggregate, indent=2)}
```

Compared with v00332x, the boundary-hit concept is N/A because v00332y never refines a flattened
candidate. The direct comparison is in `v00332x_vs_v00332y.csv`. Long edges require the complete
intervening barrier-safe event path, and local normal descriptors use above/below samples from the
same reflector. Edge records expose the future positional encoding
`[delta_tau, delta_t, delta_x, geodesic_distance, waveform_similarity, phase_similarity,
dip_difference, curvature]`; no hard RGT attention prior is imposed.

No model, checkpoint, optimizer, or training path was used. Training is {"eligible for a future matched ablation" if summary["training_ready"] else "not authorized by these QC results"}.
"""
    u.write(OUT / "v00332y_native_event_rgt_graph_report.md", report.encode())
    for relative, digest in git_record["source_config_test_sha256"].items():
        if u.sha(u.REPO / relative) != digest:
            raise RuntimeError(f"Protected source changed during run: {relative}")
    if git("rev-parse", "HEAD") != args.expected_commit:
        raise RuntimeError("HEAD changed during run")
    u.log(f"{decision}; no training: {OUT}")


if __name__ == "__main__":
    main()
