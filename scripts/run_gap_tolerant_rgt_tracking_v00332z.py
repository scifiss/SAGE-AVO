#!/usr/bin/env python3
"""Calibrate v00332z on train realizations and QC validation; never train."""

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
from sage_avo.diagnostics.gap_tolerant_graph import (
    candidate_links,
    detect_events,
    freeze_scales,
    normal_event_contrasts,
    score_links,
    sparse_graph,
    track_paths,
)
from sage_avo.diagnostics.native_event_graph import event_repeatability
from sage_avo.diagnostics.native_rgt_graph import NativeRGT
from sage_avo.diagnostics.rgt_topology_repair import structural_fields
from sage_avo.diagnostics.skeleton_graph import path_fault_qc, sample

BRANCH = "experiment/v00332z-gap-tolerant-rgt-tracking"
CONFIG_PATH = u.REPO / "configs/development_diagnostics_v00332z.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = u.BASE / "stage04" / CONFIG["experiment_name"]
u.OUT = OUT
Y_OUT = u.BASE / "stage04/sage_avo_s01_v00332y_native_event_rgt_graph"
DENSE_CSV = u.BASE / "stage04/sage_avo_s01_v00332u_sparse_skeleton/v00332u_current_vs_skeleton.csv"
PROTECTED = [
    CONFIG_PATH,
    Path(__file__),
    u.REPO / "scripts/run_skeleton_graph_v00332u.py",
    u.REPO / "src/sage_avo/diagnostics/gap_tolerant_graph.py",
    u.REPO / "src/sage_avo/diagnostics/native_event_graph.py",
    u.REPO / "src/sage_avo/diagnostics/native_rgt_graph.py",
    u.REPO / "src/sage_avo/diagnostics/rgt_topology_repair.py",
    u.REPO / "src/sage_avo/diagnostics/skeleton_graph.py",
    u.REPO / "tests/test_gap_tolerant_graph.py",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=u.REPO, text=True).strip()


def provenance(expected: str) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("rev-parse", f"refs/remotes/origin/{BRANCH}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if head != expected or branch != BRANCH or remote != expected or dirty:
        raise RuntimeError(
            f"Provenance gate failed: {branch=} {head=} {remote=} dirty={bool(dirty)}"
        )
    return {
        "repository": str(u.REPO),
        "branch": branch,
        "commit_sha": head,
        "parent_commit_sha": git("rev-parse", "HEAD^"),
        "protected_sha256": {str(path.relative_to(u.REPO)): u.sha(path) for path in PROTECTED},
        "tracked_worktree_clean_at_start": True,
    }


def arrays(rid: int) -> dict[str, np.ndarray]:
    path = u.DATASET / "realizations" / f"realization_{rid:07d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in ("avo", "rgt", "valid_mask")}


def case_events(rid: int, weak: dict[str, Any], *, repeat: bool = False) -> dict[str, Any]:
    data = arrays(rid)
    channels = (0, 2) if repeat else (0, 1, 2)
    events, counts = detect_events(
        data["avo"],
        data["rgt"],
        data["valid_mask"].astype(bool),
        CONFIG["strong_detector"],
        weak,
        CONFIG,
        channels=channels,
    )
    fields = structural_fields(data["rgt"])
    for event in events:
        position = np.asarray([[event["time"], event["trace"]]])
        event["structural_dip"] = float(sample(fields["dip"], position)[0])
        event["structural_curvature"] = float(sample(fields["curvature"], position)[0])
    return {
        "rid": rid,
        "arrays": data,
        "events": events,
        "counts": counts,
        "native": NativeRGT(data["rgt"]),
    }


def assigned_fraction(events: list[dict[str, Any]], graph: dict[str, Any]) -> float:
    return len({index for path in graph["components"] for index in path}) / max(len(events), 1)


def calibrate(
    train_ids: list[int], q: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    detector_rows = []
    for weak in CONFIG["weak_candidates"]:
        for rid in train_ids:
            full = case_events(rid, weak)
            subset = case_events(rid, weak, repeat=True)
            metrics = event_repeatability(
                full["events"], subset["events"], CONFIG["repeatability_tolerance_samples"]
            )
            detector_rows.append(
                {
                    "detector": weak["name"],
                    "realization_id": rid,
                    **metrics,
                    "strong_count": full["counts"]["strong"],
                    "weak_count": full["counts"]["weak"],
                    "overlap_count": full["counts"]["overlap"],
                    "events_per_trace": len(full["events"]) / full["arrays"]["rgt"].shape[1],
                }
            )
    detector_summary = []
    for weak in CONFIG["weak_candidates"]:
        rows = [row for row in detector_rows if row["detector"] == weak["name"]]
        detector_summary.append(
            {
                "name": weak["name"],
                "minimum_repeatability": min(row["repeatability"] for row in rows),
                "mean_repeatability": float(np.mean([row["repeatability"] for row in rows])),
                "mean_weak_count": float(np.mean([row["weak_count"] for row in rows])),
            }
        )
    eligible = [
        row
        for row in detector_summary
        if row["minimum_repeatability"] >= CONFIG["decision"]["minimum_repeatability"]
    ]
    selected = max(
        eligible or detector_summary,
        key=lambda row: (row["mean_weak_count"], row["minimum_repeatability"]),
    )
    weak = next(row for row in CONFIG["weak_candidates"] if row["name"] == selected["name"])
    train_cases = [case_events(rid, weak) for rid in train_ids]
    association_rows = []
    candidates = {}
    for radius in CONFIG["search_radius_candidates"]:
        link_lists = [
            candidate_links(case["native"], case["events"], radius, CONFIG["maximum_gap_traces"])
            for case in train_cases
        ]
        scales = freeze_scales([row for links in link_lists for row in links], CONFIG)
        graphs = []
        for case, links in zip(train_cases, link_lists):
            scored = score_links(links, scales, CONFIG)
            graphs.append(track_paths(case["events"], scored, CONFIG))
        candidate = {
            "radius": radius,
            "minimum_assigned_fraction": min(
                assigned_fraction(case["events"], graph) for case, graph in zip(train_cases, graphs)
            ),
            "mean_assigned_fraction": float(
                np.mean(
                    [
                        assigned_fraction(case["events"], graph)
                        for case, graph in zip(train_cases, graphs)
                    ]
                )
            ),
            "mean_gap_fraction": float(
                np.mean(
                    [
                        sum(link["span"] - 1 for link in graph["accepted_links"])
                        / max(len(graph["accepted_links"]), 1)
                        for graph in graphs
                    ]
                )
            ),
            "scales": scales,
        }
        candidates[radius] = (graphs, scales)
        association_rows.append(candidate)
    best = max(
        association_rows,
        key=lambda row: (
            row["minimum_assigned_fraction"],
            -row["mean_gap_fraction"],
            -row["radius"],
        ),
    )
    graphs, scales = candidates[best["radius"]]
    spacing_rows = []
    for spacing in CONFIG["node_spacing_candidates"]:
        built = [
            sparse_graph(
                case["events"], graph, spacing, CONFIG, q["adaptive_search"]["curvature_q75"]
            )
            for case, graph in zip(train_cases, graphs)
        ]
        spacing_rows.append(
            {
                "spacing": spacing,
                "minimum_two_hop_reach": min(
                    graph["statistics"]["two_hop_reach_mean"] for graph in built
                ),
                "mean_node_fraction": float(
                    np.mean(
                        [
                            len(graph["nodes"]) / case["arrays"]["valid_mask"].sum()
                            for case, graph in zip(train_cases, built)
                        ]
                    )
                ),
            }
        )
    spacing = max(spacing_rows, key=lambda row: (row["minimum_two_hop_reach"], -row["spacing"]))[
        "spacing"
    ]
    frozen = {
        "weak_detector": weak,
        "search_radius": best["radius"],
        "scales": scales,
        "node_spacing": spacing,
        "curvature_threshold": q["adaptive_search"]["curvature_q75"],
    }
    return frozen, {
        "detector": detector_rows,
        "detector_summary": detector_summary,
        "association": association_rows,
        "spacing": spacing_rows,
    }


def crossing(
    row: dict[str, Any], events: list[dict[str, Any]], faults: list[dict[str, Any]]
) -> tuple[bool, bool]:
    path = np.asarray(
        [
            [events[row["source"]]["time"], events[row["source"]]["trace"]],
            [events[row["target"]]["time"], events[row["target"]]["trace"]],
        ],
        float,
    )
    return path_fault_qc(path, faults)


def fault_qc(
    case: dict[str, Any], links: list[dict[str, Any]], graph: dict[str, Any]
) -> dict[str, Any]:
    faults = u.load_faults(u.STAGE02, case["rid"])
    events = case["events"]
    support = opportunity = proposed = rejected = barrier_rejected = retained = 0
    for fault in faults:
        left = right = 0
        for event in events:
            signed = event["trace"] - float(fault["column"]) - float(fault["dip"]) * event["time"]
            left += -4 <= signed < 0
            right += 0 <= signed <= 4
        support += int(min(left, right))
    accepted = {(row["source"], row["target"]) for row in graph["accepted_links"]}
    for row in links:
        cross, _ = crossing(row, events, faults)
        if cross:
            opportunity += 1
            proposed += int(row["plausible"])
            rejected += int(row["plausible"] and (row["source"], row["target"]) not in accepted)
            barrier_rejected += int(row["plausible"] and row["barrier"])
            retained += int((row["source"], row["target"]) in accepted)
    long_cross = sum(
        bool(path_fault_qc(edge["path"], faults)[0])
        for edge in graph["edges"]
        if edge["span"] >= 16
    )
    long_total = sum(edge["span"] >= 16 for edge in graph["edges"])
    status = (
        "COVERAGE_INSUFFICIENT"
        if not support
        else "OPPORTUNITY_INSUFFICIENT"
        if not opportunity
        else "EVALUABLE"
    )
    return {
        "event_support_around_fault": support,
        "crossing_opportunity_count": opportunity,
        "crossing_links_proposed": proposed,
        "crossing_links_rejected": rejected,
        "crossing_links_barrier_rejected": barrier_rejected,
        "crossing_links_retained": retained,
        "fault_evaluability": status,
        "fault_split_recall": barrier_rejected / proposed if proposed else None,
        "long_edge_fault_crossing_rate": long_cross / max(long_total, 1),
        "fault_count": len(faults),
    }


def metrics(
    case: dict[str, Any], links: list[dict[str, Any]], graph: dict[str, Any], q: dict[str, Any]
) -> dict[str, Any]:
    events = case["events"]
    accepted = graph["accepted_links"]
    faults = u.load_faults(u.STAGE02, case["rid"])
    high = {
        row["source"]
        for row in links
        if row["plausible"]
        and events[row["source"]]["structural_dip"] >= q["adaptive_search"]["dip_q67"]
        and not crossing(row, events, faults)[1]
    }
    curved = {
        row["source"]
        for row in links
        if row["plausible"]
        and events[row["source"]]["structural_curvature"] >= q["adaptive_search"]["curvature_q75"]
        and not crossing(row, events, faults)[1]
    }
    accepted_set = {(row["source"], row["target"]) for row in accepted}
    tracked_sources = {source for source, _ in accepted_set}
    spans = [events[path[-1]]["trace"] - events[path[0]]["trace"] for path in graph["components"]]
    traces = {events[index]["trace"] for path in graph["components"] for index in path}
    return {
        "realization_id": case["rid"],
        **case["repeatability"],
        "strong_event_count": case["counts"]["strong"],
        "weak_event_count": case["counts"]["weak"],
        "strong_weak_overlap": case["counts"]["overlap"],
        "events_per_trace": len(events) / case["arrays"]["rgt"].shape[1],
        "event_count": len(events),
        "matched_event_fraction": len(
            {index for row in accepted for index in (row["source"], row["target"])}
        )
        / max(len(events), 1),
        "assigned_event_fraction": assigned_fraction(events, graph),
        "trace_coverage_fraction": len(traces) / case["arrays"]["rgt"].shape[1],
        "component_count": len(spans),
        "median_component_span": float(np.median(spans)) if spans else 0.0,
        "maximum_component_span": max(spans, default=0),
        "small_component_fraction": float(np.mean(np.asarray(spans) < 16)) if spans else 1.0,
        "maximum_gap_length": max((row["span"] - 1 for row in accepted), default=0),
        "node_fraction": len(graph["nodes"]) / case["arrays"]["valid_mask"].sum(),
        "two_hop_reach_mean": graph["statistics"]["two_hop_reach_mean"],
        "high_dip_candidates": len(high),
        "high_dip_retention": len(high & tracked_sources) / max(len(high), 1),
        "curved_candidates": len(curved),
        "curved_retention": len(curved & tracked_sources) / max(len(curved), 1),
        **fault_qc(case, links, graph),
    }


def figure(case: dict[str, Any], name: str) -> None:
    fig, axis = plt.subplots(figsize=(11, 6))
    seismic = case["arrays"]["avo"][0]
    limit = np.quantile(abs(seismic), 0.98)
    axis.imshow(seismic, cmap="gray", aspect="auto", vmin=-limit, vmax=limit)
    paths = [
        np.asarray([[case["events"][i]["trace"], case["events"][i]["time"]] for i in path])
        for path in case["graph"]["components"]
    ]
    if paths:
        axis.add_collection(LineCollection(paths, colors="cyan", linewidths=0.7))
    axis.set(title=f"v00332z event tracks: {case['rid']}", xlabel="Trace", ylabel="Time sample")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    u.write(OUT / "figures" / name, buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if OUT.exists():
        raise RuntimeError(f"Output exists; refusing overwrite: {OUT}")
    identity = provenance(args.expected_commit)
    q = json.loads(u.Q.read_text(encoding="utf-8"))
    all_train = q["split_ids"]["train"]
    train_ids = [
        all_train[index]
        for index in np.linspace(
            0, len(all_train) - 1, CONFIG["calibration_training_count"], dtype=int
        )
    ]
    validation_ids = q["diverse_validation_subset"]["all_ids"]
    u.json_file(
        "v00332z_experiment_contract.json",
        {
            **identity,
            "revision": "v00332z",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": CONFIG,
            "calibration_training_ids": train_ids,
            "validation_ids": validation_ids,
            "q_contract_sha256": u.sha(u.Q),
            "y_summary_sha256": u.sha(Y_OUT / "v00332y_summary.json"),
            "fault_truth_use": "validation QC only after calibration",
            "training": False,
        },
    )
    frozen, calibration_tables = calibrate(train_ids, q)
    for name, rows in calibration_tables.items():
        u.csv_file(f"calibration_{name}.csv", rows)
    u.json_file("v00332z_frozen_contract.json", frozen)
    cases = []
    qc_rows = []
    table_rows: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "events",
            "candidate_links",
            "components",
            "nodes",
            "edges",
            "normal_contrasts",
        )
    }
    for rid in validation_ids:
        u.log(f"v00332z validation {rid}")
        case = case_events(rid, frozen["weak_detector"])
        repeat = case_events(rid, frozen["weak_detector"], repeat=True)
        case["repeatability"] = event_repeatability(
            case["events"], repeat["events"], CONFIG["repeatability_tolerance_samples"]
        )
        links = score_links(
            candidate_links(
                case["native"],
                case["events"],
                frozen["search_radius"],
                CONFIG["maximum_gap_traces"],
            ),
            frozen["scales"],
            CONFIG,
        )
        tracking = track_paths(case["events"], links, CONFIG)
        graph = sparse_graph(
            case["events"], tracking, frozen["node_spacing"], CONFIG, frozen["curvature_threshold"]
        )
        case["graph"] = graph
        cases.append(case)
        qc_rows.append(metrics(case, links, graph, q))
        table_rows["events"].extend(
            {
                "realization_id": rid,
                **{key: value for key, value in row.items() if key != "waveform"},
            }
            for row in case["events"]
        )
        table_rows["candidate_links"].extend({"realization_id": rid, **row} for row in links)
        table_rows["components"].extend(
            {
                "realization_id": rid,
                "component": i,
                "event_count": len(path),
                "trace_span": case["events"][path[-1]]["trace"] - case["events"][path[0]]["trace"],
            }
            for i, path in enumerate(graph["components"])
        )
        table_rows["nodes"].extend(
            {
                "realization_id": rid,
                **{key: value for key, value in row.items() if key != "waveform"},
            }
            for row in graph["nodes"]
        )
        table_rows["edges"].extend(
            {"realization_id": rid, **{key: value for key, value in row.items() if key != "path"}}
            for row in graph["edges"]
        )
        table_rows["normal_contrasts"].extend(
            {"realization_id": rid, **row}
            for row in normal_event_contrasts(
                case["arrays"]["rgt"],
                case["arrays"]["avo"],
                graph["nodes"],
                CONFIG["normal_offset_samples"],
            )
        )
        figure(case, f"tracks_{rid}.png")
    for name, rows in table_rows.items():
        u.csv_file(f"{name}.csv", rows)
    u.csv_file("validation_qc.csv", qc_rows)
    y = pd.read_csv(Y_OUT / "validation_qc.csv")
    z = pd.DataFrame(qc_rows)
    common = [
        "repeatability",
        "matched_event_fraction",
        "small_component_fraction",
        "median_component_span",
        "high_dip_retention",
        "curved_retention",
        "fault_split_recall",
        "node_fraction",
        "two_hop_reach_mean",
    ]
    comparison = z.merge(y[["realization_id", *common]], on="realization_id", suffixes=("_z", "_y"))
    u.csv_file("v00332y_vs_v00332z.csv", comparison.to_dict("records"))
    dense = pd.read_csv(DENSE_CSV)
    structural = q["diverse_validation_subset"]["high_dip_continuous_ids"]
    reach = {
        int(row.realization_id): float(row.two_hop_reach_mean)
        for row in dense.itertuples()
        if row.graph == "current" and row.region == "all"
    }
    cfg = CONFIG["decision"]
    coverage_ok = bool(
        (z.repeatability >= cfg["minimum_repeatability"]).all()
        and (z.matched_event_fraction >= cfg["minimum_matched_fraction"]).all()
    )
    fault_cases = z[z.realization_id.isin(q["diverse_validation_subset"]["fault_rich_ids"])]
    evaluable = bool(len(fault_cases) and (fault_cases.fault_evaluability == "EVALUABLE").all())
    fault_ok = bool(
        evaluable and (fault_cases.fault_split_recall >= cfg["minimum_fault_split_recall"]).all()
    )
    long_ok = bool(
        (fault_cases.long_edge_fault_crossing_rate <= cfg["maximum_long_edge_fault_crossing"]).all()
    )
    high_cases = z[z.realization_id.isin(structural)]
    high_ok = bool(
        len(high_cases)
        and (high_cases.high_dip_retention >= cfg["minimum_high_dip_retention"]).all()
    )
    curved_ok = bool(
        (
            z.loc[z.curved_candidates > 0, "curved_retention"] >= cfg["minimum_curved_retention"]
        ).all()
    )
    long_range = bool(
        all(
            z.loc[z.realization_id == rid, "two_hop_reach_mean"].iloc[0] > reach[rid]
            for rid in structural
        )
    )
    fragmentation_better = bool(
        (
            comparison.small_component_fraction_z.mean()
            < comparison.small_component_fraction_y.mean()
        )
        and (comparison.median_component_span_z.mean() > comparison.median_component_span_y.mean())
    )
    sparse = bool((z.node_fraction <= cfg["maximum_node_fraction"]).all())
    if not table_rows["events"] or not table_rows["edges"]:
        decision = "IMPLEMENTATION_PROBLEM"
    elif not coverage_ok:
        decision = "EVENT_COVERAGE_WEAK"
    elif not long_ok:
        decision = "GAP_BRIDGING_CROSSES_FAULTS"
    elif not fault_ok:
        decision = "FAULT_SPLIT_WEAK"
    elif not high_ok:
        decision = "HIGH_DIP_OVERPRUNED"
    elif not (fragmentation_better and curved_ok and long_range and sparse):
        decision = "COMPONENTS_STILL_FRAGMENTED"
    else:
        decision = "GAP_TOLERANT_EVENT_GRAPH_PROMISING"
    summary = {
        "decision": decision,
        "EVENT_COVERAGE_STATUS": "PASS" if coverage_ok else "FAIL",
        "FAULT_EVALUABILITY_STATUS": "PASS" if evaluable else "INSUFFICIENT",
        "FAULT_SPLIT_STATUS": "PASS" if fault_ok else "FAIL_OR_UNEVALUABLE",
        "HIGH_DIP_STATUS": "PASS" if high_ok else "FAIL",
        "CURVED_STATUS": "PASS" if curved_ok else "FAIL",
        "SPARSITY_LONG_RANGE_STATUS": "PASS" if sparse and long_range else "FAIL",
        "fragmentation_improved": fragmentation_better,
        "long_edge_fault_safe": long_ok,
        "frozen_choices": frozen,
        "validation_case_ids": validation_ids,
        "mean_metrics": {
            key: float(z[key].mean()) for key in common if pd.api.types.is_numeric_dtype(z[key])
        },
    }
    u.json_file("v00332z_summary.json", summary)
    report = [
        "# v00332z gap-tolerant RGT-guided reflector tracking",
        "",
        f"Decision: `{decision}`",
        "",
        "No training was performed. Fault truth was used only for validation QC.",
        "",
        "## Frozen training-only choices",
        "",
        "```json",
        json.dumps(frozen, indent=2),
        "```",
        "",
        "## Validation and comparison",
        "",
        "```csv",
        z.to_csv(index=False).strip(),
        "```",
        "",
        "v00332y versus v00332z:",
        "",
        "```csv",
        comparison.to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        f"Event coverage: {summary['EVENT_COVERAGE_STATUS']}; fault evaluability: {summary['FAULT_EVALUABILITY_STATUS']}; fault split: {summary['FAULT_SPLIT_STATUS']}.",
        f"High dip: {summary['HIGH_DIP_STATUS']}; curved: {summary['CURVED_STATUS']}; sparsity and long range: {summary['SPARSITY_LONG_RANGE_STATUS']}.",
        "",
        "Fault support, crossing opportunity, proposed, rejected, and retained links are reported separately. Zero opportunity is not counted as a successful split.",
        "",
    ]
    u.write(OUT / "v00332z_report.md", "\n".join(report).encode())
    print(
        json.dumps(
            {
                "output": str(OUT),
                "decision": decision,
                **{key: summary[key] for key in summary if key.endswith("STATUS")},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
