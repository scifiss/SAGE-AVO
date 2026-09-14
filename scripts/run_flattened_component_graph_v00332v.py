#!/usr/bin/env python3
"""Build and QC v00332v; no model, checkpoint, optimizer, or training."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

import run_skeleton_graph_v00332u as u
from sage_avo.config import load_config
from sage_avo.diagnostics.flattened_component_graph import (
    build_component_graph,
    component_graph_statistics,
    detect_reflector_components,
    flatten_rgt,
    normal_contrasts,
)
from sage_avo.diagnostics.skeleton_graph import path_fault_qc

CONFIG_PATH = u.REPO / "configs/development_diagnostics_v00332v.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = u.BASE / "stage04" / CONFIG["experiment_name"]
u.OUT = OUT


def save(fig, name):
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170)
    plt.close(fig)
    u.write(OUT / "figures" / name, buffer.getvalue())


def background(ax, data, *, flattened=False):
    limit = np.quantile(abs(data), 0.98)
    ax.imshow(data, cmap="gray", aspect="auto", vmin=-limit, vmax=limit)
    ax.set(xlabel="Trace", ylabel="RGT sample" if flattened else "Time sample")


def annotate_truth(case):
    """Truth enters only here, after mapping, detection, and graph construction."""
    faults = u.load_faults(u.STAGE02, case["rid"])
    graph, mapping, detection = case["graph"], case["mapping"], case["detection"]
    crossing_edges = 0
    for edge in graph["edges"]:
        crossing, near = path_fault_qc(edge["path"], faults)
        edge["fault_crossing"] = bool(crossing)
        edge["fault_corridor"] = bool(near)
        crossing_edges += bool(crossing)
    crossing_candidates = kept_crossing_candidates = 0
    high_dip_candidates = high_dip_kept = 0
    for link in detection["links"]:
        a = detection["candidates"][link["source_candidate"]]
        b = detection["candidates"][link["target_candidate"]]
        path = np.array(
            [
                [mapping["inverse_t"][a["tau_index"], a["trace"]], a["trace"]],
                [mapping["inverse_t"][b["tau_index"], b["trace"]], b["trace"]],
            ]
        )
        crossing, near = path_fault_qc(path, faults)
        link["fault_crossing"] = bool(crossing)
        link["fault_corridor"] = bool(near)
        crossing_candidates += bool(crossing and link["reciprocal"])
        kept_crossing_candidates += bool(crossing and link["kept"])
        high = link["reciprocal"] and abs(path[1, 0] - path[0, 0]) >= 0.5 and not near
        high_dip_candidates += bool(high)
        high_dip_kept += bool(high and link["kept"])
    return {
        "faults": faults,
        "edge_fault_crossing_rate": crossing_edges / max(len(graph["edges"]), 1),
        "fault_candidate_links": crossing_candidates,
        "fault_split_recall": 1 - kept_crossing_candidates / max(crossing_candidates, 1),
        "high_dip_candidates": high_dip_candidates,
        "high_dip_retention": high_dip_kept / max(high_dip_candidates, 1),
    }


def make_figures(cases):
    example = cases[2]
    fault = cases[4]
    arr, mapping, detection, graph = (
        example["arrays"],
        example["mapping"],
        example["detection"],
        example["graph"],
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, arr["avo"][0])
    ax.set_title("A. Original near-angle seismic (t,x)")
    save(fig, "01_original_tx.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, mapping["avo"][0], flattened=True)
    ax.set_title("B. RGT-flattened near-angle seismic (tau,x)")
    save(fig, "02_flattened_taux.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, mapping["avo"][0], flattened=True)
    accepted = [c for c in detection["candidates"] if c["accepted"]]
    ax.scatter(
        [c["trace"] for c in accepted], [c["tau_index"] for c in accepted], s=2, c="tab:orange"
    )
    ax.set_title("C. Observable reflector-ridge samples")
    save(fig, "03_detected_ridges.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, mapping["avo"][0], flattened=True)
    ax.scatter(
        [c["trace"] for c in accepted],
        [c["tau_index"] for c in accepted],
        s=3,
        c=[c["component"] for c in accepted],
        cmap="turbo",
    )
    ax.set_title("D. Connected reflector components")
    save(fig, "04_connected_components.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, mapping["avo"][0], flattened=True)
    ax.scatter(
        [n["trace"] for n in graph["nodes"]],
        [n["tau_index"] for n in graph["nodes"]],
        s=8,
        c=[n["component"] for n in graph["nodes"]],
        cmap="turbo",
    )
    ax.set_title("E. Sparse component nodes")
    save(fig, "05_sparse_nodes.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def flattened_path(edge):
        result = []
        for physical_t, trace in edge["path"]:
            x = int(round(trace))
            y = np.interp(
                physical_t,
                mapping["inverse_t"][:, x],
                np.arange(len(mapping["tau_grid"])),
            )
            result.append([trace, y])
        return np.asarray(result)

    for ax, (title, spans) in zip(axes, [("short", {4, 8}), ("medium", {16}), ("long", {32, 64})]):
        background(ax, mapping["avo"][0], flattened=True)
        lines = []
        for e in graph["edges"]:
            if e["span"] in spans:
                lines.append(flattened_path(e))
        if lines:
            ax.add_collection(
                LineCollection(
                    lines[:: max(1, len(lines) // 1000)], colors="tab:cyan", linewidths=0.5
                )
            )
        ax.set_title(f"F. {title} component edges")
    save(fig, "06_edge_ranges_flattened.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, arr["avo"][0])
    lines = [e["path"][:, ::-1] for e in graph["edges"]]
    if lines:
        ax.add_collection(
            LineCollection(lines[:: max(1, len(lines) // 2000)], colors="tab:cyan", linewidths=0.45)
        )
    ax.set_title("G. Same component graph mapped to (t,x)")
    save(fig, "07_graph_mapped_tx.png")
    arr, graph = fault["arrays"], fault["graph"]
    fig, ax = plt.subplots(figsize=(10, 6))
    background(ax, arr["avo"][0])
    kept = [e["path"][:, ::-1] for e in graph["edges"] if not e["fault_crossing"]]
    crossed = [e["path"][:, ::-1] for e in graph["edges"] if e["fault_crossing"]]
    if kept:
        ax.add_collection(
            LineCollection(kept[:: max(1, len(kept) // 1800)], colors="tab:cyan", linewidths=0.45)
        )
    if crossed:
        ax.add_collection(LineCollection(crossed, colors="red", linewidths=1.2))
    y = np.arange(arr["rgt"].shape[0])
    for f in fault["truth"]["faults"]:
        ax.plot(float(f["column"]) + float(f["dip"]) * y, y, "y--", linewidth=1)
    ax.set_title("H. Fault QC: components break before edges (red = failures)")
    save(fig, "08_fault_component_break.png")


def main():
    if OUT.exists():
        raise RuntimeError(f"Output exists; refusing overwrite: {OUT}")
    q = json.loads(u.Q.read_text())
    ids = q["diverse_validation_subset"]["all_ids"]
    sources = [
        CONFIG_PATH,
        Path(__file__),
        u.REPO / "src/sage_avo/diagnostics/flattened_component_graph.py",
        u.REPO / "tests/test_flattened_component_graph.py",
    ]
    protected = {
        str(p): u.sha(p)
        for p in [u.Q] + [u.DATASET / "realizations" / f"realization_{rid:07d}.npz" for rid in ids]
    }
    u.json_file(
        "v00332v_contract.json",
        {
            "revision": "v00332v",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=u.REPO, text=True
            ).strip(),
            "config": CONFIG,
            "source_sha256": {str(p.relative_to(u.REPO)): u.sha(p) for p in sources},
            "input_sha256": protected,
            "fault_truth_usage": "loaded only after graph construction for QC",
            "cnn_coherence": "descriptive only; not computed and not a gate",
            "training": False,
        },
    )
    cases = []
    node_rows = []
    edge_rows = []
    comp_rows = []
    roundtrip = []
    fault_rows = []
    normal_rows = []
    link_rows = []
    for rid in ids:
        u.log(f"v00332v flatten/detect/components: {rid}")
        with np.load(
            u.DATASET / "realizations" / f"realization_{rid:07d}.npz", allow_pickle=False
        ) as f:
            arrays = {k: f[k] for k in ["avo", "rgt", "valid_mask", "reservoir_mask"]}
        mapping = flatten_rgt(
            arrays["rgt"], arrays["avo"], arrays["valid_mask"], CONFIG["tau_samples"]
        )
        detection = detect_reflector_components(mapping, CONFIG)
        graph = build_component_graph(mapping, detection, CONFIG)
        truth = annotate_truth(
            {"rid": rid, "graph": graph, "mapping": mapping, "detection": detection}
        )
        stats = component_graph_statistics(graph)
        for n in graph["nodes"]:
            node_rows.append(dict(realization_id=rid, **n))
        for e in graph["edges"]:
            edge_rows.append({"realization_id": rid, **{k: v for k, v in e.items() if k != "path"}})
        for cid, group in enumerate(detection["components"]):
            xs = [detection["candidates"][i]["trace"] for i in group]
            comp_rows.append(
                {
                    "realization_id": rid,
                    "component": cid,
                    "candidate_points": len(group),
                    "trace_start": min(xs),
                    "trace_end": max(xs),
                    "trace_span": max(xs) - min(xs),
                }
            )
        for row in detection["links"]:
            link_rows.append(dict(realization_id=rid, **row))
        roundtrip.append(
            {
                "realization_id": rid,
                "tau_error_steps_p50": float(np.quantile(mapping["roundtrip_tau_steps"], 0.5)),
                "tau_error_steps_p99": float(np.quantile(mapping["roundtrip_tau_steps"], 0.99)),
                "time_error_samples_p50": float(
                    np.quantile(mapping["roundtrip_time_samples"], 0.5)
                ),
                "time_error_samples_p99": float(
                    np.quantile(mapping["roundtrip_time_samples"], 0.99)
                ),
                "grid_resampling_time_error_samples_p99": float(
                    np.quantile(mapping["grid_resampling_time_error_samples"], 0.99)
                ),
                "avo_abs_error_p99": float(np.quantile(mapping["roundtrip_avo_abs"], 0.99)),
                "ambiguous_plateau_fraction": mapping["ambiguous_plateau_fraction"],
            }
        )
        accepted = sum(c["accepted"] for c in detection["candidates"])
        spans = [
            max(detection["candidates"][i]["trace"] for i in g)
            - min(detection["candidates"][i]["trace"] for i in g)
            for g in detection["components"]
        ]
        fault_rows.append(
            {
                "realization_id": rid,
                **{k: v for k, v in truth.items() if k != "faults"},
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["edges"]),
                "component_count": len(spans),
                "accepted_candidate_fraction": accepted / max(len(detection["candidates"]), 1),
                "node_fraction": len(graph["nodes"]) / arrays["valid_mask"].sum(),
                "median_component_span": float(np.median(spans)) if spans else 0.0,
                "small_component_fraction": float(np.mean(np.array(spans) < 16)) if spans else 1.0,
                "two_hop_reach_mean": stats["two_hop_reach_mean"],
            }
        )
        for row in normal_contrasts(
            arrays["rgt"], arrays["avo"], graph["nodes"], CONFIG["normal_offset_samples"]
        ):
            normal_rows.append(dict(realization_id=rid, **row))
        cases.append(
            {
                "rid": rid,
                "arrays": arrays,
                "mapping": mapping,
                "detection": detection,
                "graph": graph,
                "truth": truth,
            }
        )
    for path, digest in protected.items():
        if u.sha(path) != digest:
            raise RuntimeError("Input changed: " + path)
    frames = {
        "v00332v_node_statistics.csv": node_rows,
        "v00332v_edge_statistics.csv": edge_rows,
        "v00332v_component_statistics.csv": comp_rows,
        "v00332v_roundtrip_qc.csv": roundtrip,
        "v00332v_fault_qc.csv": fault_rows,
        "v00332v_normal_contrast.csv": normal_rows,
        "v00332v_component_link_qc.csv": link_rows,
    }
    for name, rows in frames.items():
        u.csv_file(name, rows)
    make_figures(cases)
    rt = pd.DataFrame(roundtrip)
    qc = pd.DataFrame(fault_rows)
    d = CONFIG["decision"]
    implementation = (
        not len(node_rows)
        or not len(edge_rows)
        or not np.isfinite(
            pd.DataFrame(edge_rows)
            .select_dtypes("number")
            .drop(columns=["reflector_continuity"], errors="ignore")
        )
        .all()
        .all()
    )
    flatten_ok = bool(
        (rt.tau_error_steps_p99 <= d["maximum_roundtrip_tau_steps_p99"]).all()
        and (rt.time_error_samples_p99 <= d["maximum_roundtrip_time_samples_p99"]).all()
    )
    detection_ok = bool((qc.accepted_candidate_fraction >= d["minimum_component_coverage"]).all())
    sparse_ok = bool(
        (qc.node_fraction >= d["minimum_node_fraction"]).all()
        and (qc.node_fraction <= d["maximum_node_fraction"]).all()
    )
    fault_sections = qc[qc.fault_candidate_links > 0]
    fault_ok = bool(
        len(fault_sections) > 0
        and (fault_sections.edge_fault_crossing_rate <= d["maximum_fault_crossing_rate"]).all()
        and (fault_sections.fault_split_recall >= d["minimum_fault_split_recall"]).all()
    )
    fragmentation_ok = bool(
        (qc.small_component_fraction <= d["maximum_small_component_fraction"]).all()
        and (qc.median_component_span >= d["minimum_median_component_span"]).all()
    )
    dip_ok = bool((qc.high_dip_retention >= d["minimum_high_dip_retention"]).all())
    reach_ok = bool((qc.two_hop_reach_mean >= d["minimum_two_hop_reach"]).all())
    if implementation:
        decision = "IMPLEMENTATION_PROBLEM"
    elif not flatten_ok:
        decision = "RGT_FLATTENING_UNSTABLE"
    elif not detection_ok:
        decision = "INTERFACE_DETECTION_WEAK"
    elif not fault_ok:
        decision = "FAULT_COMPONENT_SPLIT_WEAK"
    elif not fragmentation_ok:
        decision = "GRAPH_TOO_FRAGMENTED"
    elif not sparse_ok:
        decision = "GRAPH_NOT_SPARSE"
    elif not dip_ok:
        decision = "INTERFACE_DETECTION_WEAK"
    elif not reach_ok:
        decision = "GRAPH_TOO_FRAGMENTED"
    else:
        decision = "FLATTENED_COMPONENT_GRAPH_PROMISING"
    summary = {
        "decision": decision,
        "gates": {
            "implementation": not implementation,
            "flattening": flatten_ok,
            "interface_detection": detection_ok,
            "fault_component_split": fault_ok,
            "fragmentation": fragmentation_ok,
            "sparsity": sparse_ok,
            "high_dip_retention": dip_ok,
            "two_hop_reach": reach_ok,
            "cnn_coherence_gate": False,
        },
        "training_performed": False,
        "checkpoint_loaded": False,
        "fault_truth_qc_only": True,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    u.json_file("v00332v_summary.json", summary)
    report = f"""# v00332v — RGT-flattened reflector-component graph

Decision: **{decision}**.

No training, checkpoint/CNN execution, production-source edit, or truth-guided construction occurred.
RGT is mapped column-wise with unique plateau collapse; ambiguous plateau samples are excluded from
time round-trip scoring. AVA ridges are detected independently per trace in flattened coordinates,
then connected only by reciprocal adjacent-trace matches passing fixed waveform, phase, strength,
support, and flattened-jump criteria. Components shorter than the predeclared observable extent are
discarded. Sparse nodes sample each accepted ridge trace; sparsity comes from selecting only ridge
locations rather than dense pixels. Edges span 4/8/16/32/64 traces inside one component.
Fault truth is loaded only after all construction and is used solely for QC.

## Gate results

```json
{json.dumps(summary["gates"], indent=2)}
```

Section metrics are in `v00332v_fault_qc.csv`; interpolation errors are in
`v00332v_roundtrip_qc.csv`. Every edge exports delta-tau/x/t, interface geodesic distance,
dip difference, curvature statistics and minimum waveform continuity. Above/below AVA and
P/G/curvature contrasts sampled along the local physical RGT gradient are descriptive only.
CNN feature coherence is neither computed nor used as a training gate.

## Limits

The component detector uses fixed observable thresholds, not fault labels, but one six-section
synthetic QC set is not independent evidence of field robustness. Plateau collapse is invertible
only outside explicitly reported ambiguous samples. The direct mapping round trip and the
additional finite uniform-grid resampling error are reported separately. RGT consistency is
geometry, not evidence of
correct geological identity. A passing topology screen would authorize only a future matched
ablation; training remains unauthorized here.
"""
    u.write(OUT / "v00332v_report.md", report.encode())
    u.log(f"{decision}; complete: {OUT}")


if __name__ == "__main__":
    main()
