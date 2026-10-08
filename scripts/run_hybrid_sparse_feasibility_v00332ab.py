#!/usr/bin/env python3
"""No-training patch and mechanism audit for optional sparse reflector messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch

from sage_avo.config import load_config
from sage_avo.data.patches import resize_channels_first
from sage_avo.diagnostics.gap_tolerant_graph import (
    candidate_links,
    detect_events,
    score_links,
    sparse_graph,
    track_paths,
)
from sage_avo.diagnostics.native_rgt_graph import NativeRGT
from sage_avo.diagnostics.rgt_topology_repair import load_faults, structural_fields
from sage_avo.diagnostics.skeleton_graph import graph_statistics, sample
from sage_avo.evaluation.inference import tile_starts
from sage_avo.models.hybrid_sparse import (
    HybridSparseSAGEAVO,
    SparsePatchGraph,
    crop_accepted_graph,
)
from sage_avo.models.sage_avo import SAGEAVO


REPO = Path(__file__).resolve().parents[1]
BRANCH = "experiment/v00332ab-hybrid-sparse-graph"
PARENT = "8be05862de02bcf6d9838848fae1e74d98f3d910"
CONFIG_PATH = REPO / "configs/development_diagnostics_v00332ab.yaml"
CONFIG = load_config(CONFIG_PATH)
PROTECTED = (
    "configs/development_diagnostics_v00332ab.yaml",
    "scripts/run_hybrid_sparse_feasibility_v00332ab.py",
    "src/sage_avo/models/hybrid_sparse.py",
    "src/sage_avo/evaluation/inference.py",
    "tests/test_hybrid_sparse_graph.py",
    "src/sage_avo/diagnostics/gap_tolerant_graph.py",
    "configs/development_diagnostics_v00332z.yaml",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def provenance(expected: str) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = git("rev-parse", f"refs/remotes/origin/{BRANCH}")
    parent = git("rev-parse", "HEAD^")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if (head, branch, remote, parent) != (expected, BRANCH, expected, PARENT) or dirty:
        raise RuntimeError("v00332ab pushed-commit/branch/parent/clean-worktree gate failed")
    return {
        "repository": str(REPO),
        "branch": BRANCH,
        "commit_sha": head,
        "parent_commit_sha": PARENT,
        "protected_source_config_test_sha256": {name: sha(REPO / name) for name in PROTECTED},
        "tracked_worktree_clean_at_start": True,
    }


def grouped(path: Path) -> dict[int, list[dict[str, Any]]]:
    frame = pd.read_csv(path)
    return {
        int(rid): group.drop(columns="realization_id").to_dict("records")
        for rid, group in frame.groupby("realization_id", sort=False)
    }


def observable_full_graph(
    arrays: dict[str, np.ndarray], frozen: dict[str, Any], z_config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run unchanged frozen z detector/tracker on train observables only."""
    avo = arrays["avo"]
    rgt = arrays["rgt"]
    events, _ = detect_events(
        avo, rgt, arrays["valid_mask"].astype(bool),
        z_config["strong_detector"], frozen["weak_detector"], z_config,
    )
    curvature = structural_fields(rgt)["curvature"]
    for event in events:
        event["structural_curvature"] = float(
            sample(curvature, np.asarray([[event["time"], event["trace"]]]))[0]
        )
    candidates = candidate_links(
        NativeRGT(rgt), events, frozen["search_radius"], z_config["maximum_gap_traces"]
    )
    links = score_links(candidates, frozen["scales"], z_config)
    tracking = track_paths(events, links, z_config)
    graph = sparse_graph(
        events, tracking, frozen["node_spacing"], z_config, frozen["curvature_threshold"]
    )
    return graph["nodes"], graph["edges"]


def rebuilt_patch_graph(
    arrays: dict[str, np.ndarray], patch: dict[str, Any],
    frozen: dict[str, Any], z_config: dict[str, Any],
) -> SparsePatchGraph:
    top, left = int(patch["top"]), int(patch["left"])
    height, width = int(patch["raw_height"]), int(patch["raw_width"])
    output = (int(patch["output_height"]), int(patch["output_width"]))
    spatial = np.s_[top : top + height, left : left + width]
    local = {
        "avo": arrays["avo"][(slice(None),) + spatial],
        "rgt": arrays["rgt"][spatial],
        "valid_mask": arrays["valid_mask"][spatial],
    }
    nodes, edges = observable_full_graph(local, frozen, z_config)
    return crop_accepted_graph(
        nodes, edges, top=0, left=0, raw_shape=(height, width), output_shape=output
    )


def support_mask(graph: SparsePatchGraph) -> np.ndarray:
    """Component-local bilinear footprint; collisions have no support."""
    height, width = graph.shape
    owner = np.full((height, width), -1, int)
    if graph.edge_index.numel() == 0:
        return owner >= 0
    active = np.unique(graph.edge_index.numpy())
    for index in active:
        row, col = graph.coordinates[index].numpy()
        r0, c0 = int(np.floor(row)), int(np.floor(col))
        dr, dc = row - r0, col - c0
        component = int(graph.components[index])
        for dy, wy in ((0, 1 - dr), (1, dr)):
            for dx, wx in ((0, 1 - dc), (1, dc)):
                if wy * wx <= 0:
                    continue
                r, c = min(max(r0 + dy, 0), height - 1), min(max(c0 + dx, 0), width - 1)
                if owner[r, c] == -1:
                    owner[r, c] = component
                elif owner[r, c] != component:
                    owner[r, c] = -2
    return owner >= 0


def region_masks(
    arrays: dict[str, np.ndarray], faults: list[dict[str, Any]],
    top: int, left: int, raw_shape: tuple[int, int], output_shape: tuple[int, int],
    dip_cut: float,
) -> dict[str, np.ndarray]:
    height, width = raw_shape
    spatial = np.s_[top : top + height, left : left + width]
    valid = arrays["valid_mask"][spatial].astype(bool)
    dip = arrays["structural_dip"][spatial]
    time, trace = np.indices((height, width))
    time += top
    trace += left
    fault = np.zeros((height, width), bool)
    for item in faults:
        fault |= abs(trace - item["column"] - item["dip"] * time) <= 3
    raw = {
        "valid": valid,
        "reservoir": arrays["reservoir_mask"][spatial].astype(bool) & valid,
        "high_dip": (dip >= dip_cut) & valid & ~fault,
        "fault_corridor": fault & valid,
    }
    return {
        name: resize_channels_first(mask[None].astype(np.float32), output_shape, order=0)[0]
        > 0.5
        for name, mask in raw.items()
    }


def graph_metrics(
    graph: SparsePatchGraph, masks: dict[str, np.ndarray], long_limit: int
) -> dict[str, Any]:
    points = graph.coordinates.numpy()
    pairs = graph.edge_index.numpy()[:, ::2].T.reshape(-1, 2)
    stats = graph_statistics(points, pairs)
    spans = np.abs(graph.edge_attr.numpy()[::2, 2]) * 100.0
    footprint = support_mask(graph)
    rows: dict[str, Any] = {
        "node_count": len(points),
        "edge_count": len(pairs),
        "isolated_node_fraction": stats["isolated_node_fraction"],
        "edge_span_lt4": int(np.count_nonzero(spans < 4 - 1e-5)),
        "edge_span_4_to15": int(np.count_nonzero((spans >= 4 - 1e-5) & (spans < long_limit - 1e-5))),
        "edge_span_16_to31": int(np.count_nonzero((spans >= 16 - 1e-5) & (spans < 32 - 1e-5))),
        "edge_span_32_to63": int(np.count_nonzero((spans >= 32 - 1e-5) & (spans < 64 - 1e-5))),
        "edge_span_ge64": int(np.count_nonzero(spans >= 64 - 1e-5)),
        "long_edge_count": int(np.count_nonzero(spans >= long_limit - 1e-5)),
        "longest_edge_trace_span": float(spans.max()) if len(spans) else 0.0,
        "one_hop_reach_model_pixels": stats["one_hop_reach_mean"],
        "two_hop_reach_model_pixels": stats["two_hop_reach_mean"],
        "support_pixel_fraction": float(footprint.mean()),
        "component_count": len(set(graph.components.tolist())),
        "zero_usable_edge": len(pairs) == 0,
        "zero_long_edge": not np.any(spans >= long_limit - 1e-5),
    }
    for name, mask in masks.items():
        rows[f"{name}_pixel_count"] = int(mask.sum())
        rows[f"{name}_supported_pixel_fraction"] = (
            float((footprint & mask).sum() / mask.sum()) if mask.any() else None
        )
        rows[f"{name}_any_sparse_support"] = bool((footprint & mask).any()) if mask.any() else None
    return rows


def node_alignment(first: SparsePatchGraph, second: SparsePatchGraph) -> float | None:
    if first.coordinates.shape[0] == 0:
        return None
    if second.coordinates.shape[0] == 0:
        return 0.0
    distance, _ = cKDTree(second.coordinates.numpy()).query(first.coordinates.numpy())
    return float(np.mean(distance <= 1.5))


def public_mechanism_smoke() -> dict[str, Any]:
    """One forward/backward on a tiny public tensor; no optimizer or checkpoint."""
    torch.manual_seed(332)
    dense = SAGEAVO(hidden_channels=8, graph_layers=1, graph_heads=2)
    baseline = HybridSparseSAGEAVO(dense, channels=8, gamma=0)
    active = HybridSparseSAGEAVO(dense, channels=8, gamma=0.1)
    nodes = [
        {"node": 0, "component": 0, "time": 4.2, "trace": 2},
        {"node": 1, "component": 0, "time": 4.4, "trace": 11},
    ]
    edges = [{
        "source": 0, "target": 1, "component": 0, "delta_tau": 0.001,
        "delta_t": 0.2, "delta_x": 9, "geodesic_length": 9.1,
        "waveform_continuity": 0.95, "phase_continuity": 0.9,
        "shift_continuity": 0.01, "curvature": 0.02, "gap_count": 0,
    }]
    graph = crop_accepted_graph(nodes, edges, top=0, left=0, raw_shape=(12, 16), output_shape=(12, 16))
    avo = torch.randn(1, 3, 12, 16)
    low = torch.randn(1, 3, 12, 16)
    rgt = torch.arange(12, dtype=torch.float32)[None, :, None].expand(1, 12, 16).clone()
    t = torch.tensor([0.37])
    original = dense(low, t, avo, low, rgt)
    zero = baseline(low, t, avo, low, rgt, [graph])
    difference = float((original.velocity - zero.velocity).abs().max())
    active_result = active(low, t, avo, low, rgt, [graph])
    support = active.last_support.clone()
    outside = support.expand_as(active_result.velocity) == 0
    outside_difference = float((active_result.velocity[outside] - original.velocity[outside]).abs().max())
    segmentation_difference = float(
        (active_result.segmentation_logits - original.segmentation_logits).abs().max()
    )
    loss = active_result.velocity.square().mean()
    loss.backward()
    grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in active.sparse.parameters()
        if parameter.grad is not None
    )
    with torch.no_grad():
        second = active(low + 0.1, torch.tensor([0.83]), avo, low, rgt, [graph])
        sample_result = active.sample(avo, low, rgt, [graph], steps=2)
    if (
        difference != 0 or outside_difference != 0 or segmentation_difference != 0
        or grad <= 0 or not torch.isfinite(active_result.velocity).all()
    ):
        raise AssertionError("Sparse mechanism zero-parity/gradient/finite smoke failed")
    if not torch.isfinite(second.velocity).all() or not torch.isfinite(sample_result).all():
        raise AssertionError("Multi-time/Heun sparse smoke failed")
    return {
        "zero_scale_max_absolute_difference": difference,
        "active_outside_support_max_absolute_difference": outside_difference,
        "active_segmentation_max_absolute_difference": segmentation_difference,
        "sparse_parameter_gradient_l1": grad,
        "support_pixel_fraction": float(active.last_support.mean()),
        "finite_multiple_flow_times": True,
        "finite_two_step_heun": True,
        "optimizer_steps": 0,
        "checkpoint_loads": 0,
    }


def write_figure(
    output: Path, name: str, arrays: dict[str, np.ndarray],
    full_nodes: list[dict[str, Any]], full_edges: list[dict[str, Any]],
    patch: dict[str, Any], cropped: SparsePatchGraph, rebuilt: SparsePatchGraph,
) -> None:
    top, left = int(patch["top"]), int(patch["left"])
    height, width = int(patch["raw_height"]), int(patch["raw_width"])
    image = resize_channels_first(
        arrays["avo"][:, top : top + height, left : left + width],
        (int(patch["output_height"]), int(patch["output_width"])),
    )[0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    limit = float(np.quantile(np.abs(arrays["avo"][0]), 0.98))
    axes[0].imshow(arrays["avo"][0], cmap="gray", aspect="auto", vmin=-limit, vmax=limit)
    for edge in full_edges:
        if edge["span"] < 16:
            continue
        a, b = full_nodes[int(edge["source"])], full_nodes[int(edge["target"])]
        axes[0].plot((a["trace"], b["trace"]), (a["time"], b["time"]), color="cyan", lw=0.25)
    axes[0].add_patch(plt.Rectangle((left, top), width, height, fill=False, ec="red", lw=1.5))
    axes[0].set_title("Full graph; red actual patch")
    for axis, graph, title in (
        (axes[1], cropped, "Full graph cropped to patch"),
        (axes[2], rebuilt, "Graph rebuilt from patch AVO/RGT"),
    ):
        axis.imshow(image, cmap="gray", aspect="auto", vmin=-limit, vmax=limit)
        xy = graph.coordinates.numpy()
        for source, target in graph.edge_index.numpy()[:, ::2].T:
            if abs(xy[source, 1] - xy[target, 1]) >= 16:
                axis.plot((xy[source, 1], xy[target, 1]), (xy[source, 0], xy[target, 0]), color="cyan", lw=0.5)
        if len(xy):
            axis.scatter(xy[:, 1], xy[:, 0], s=3, c="yellow")
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output / name, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    identity = provenance(args.expected_commit)
    private = args.private_root.resolve()
    if not private.is_dir() or private.is_relative_to(REPO):
        raise RuntimeError("Require existing private artifact root outside Git")
    base = private / "stage_artifacts"
    dataset = base / "stage03/ds_v00331_production100_support_aware/dataset"
    stage02 = base / "stage02/v00331_production100_support_aware/realizations"
    z = base / "stage04/sage_avo_s01_v00332z_gap_tolerant_rgt_tracking"
    q_path = base / "stage04/sage_avo_s01_v00332q_clean_20epoch_corrected_rgt/v00332q_contract.json"
    final_output = private / "scientific_reconciliation/v00332ab_hybrid_sparse_graph"
    output = final_output.with_name(final_output.name + ".incomplete")
    if output.exists() or final_output.exists():
        raise RuntimeError(f"Refusing to overwrite existing private audit: {final_output}")
    q = json.loads(q_path.read_text())
    z_contract = json.loads((z / "v00332z_experiment_contract.json").read_text())
    if z_contract["commit_sha"] != "30e0b012467b3553b6aef44089175dad24a2019a":
        raise RuntimeError("Frozen v00332z artifacts do not match reviewed tracker")
    frozen = json.loads((z / "v00332z_frozen_contract.json").read_text())
    z_config = z_contract["config"]
    train_ids = [int(value) for value in CONFIG["training_audit_realization_ids"]]
    validation_ids = [int(value) for value in CONFIG["reused_validation_realization_ids"]]
    if not set(train_ids) <= set(q["split_ids"]["train"]):
        raise RuntimeError("Audit train IDs are not in immutable train split")
    if not set(validation_ids) <= set(q["split_ids"]["validation"]):
        raise RuntimeError("Audit validation IDs are not in immutable validation split")
    ids = train_ids + validation_ids
    patch_index = pd.read_csv(dataset / "patch_index.csv")
    indexed = patch_index[patch_index.realization_id.isin(ids)]
    if indexed.groupby("realization_id").size().to_dict() != {rid: 200 for rid in ids}:
        raise RuntimeError("Unexpected immutable patch-index counts")
    nodes_by_case = grouped(z / "nodes.csv")
    edges_by_case = grouped(z / "edges.csv")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    inputs = [dataset / "patch_index.csv", q_path, z / "nodes.csv", z / "edges.csv", z / "v00332z_frozen_contract.json"]
    inputs += [dataset / "realizations" / f"realization_{rid:07d}.npz" for rid in ids]
    contract = {
        **identity,
        "revision": "v00332ab",
        "frozen_v00332z_sha": z_contract["commit_sha"],
        "input_sha256": {str(path.relative_to(private)): sha(path) for path in inputs},
        "train_ids": train_ids,
        "reused_validation_ids": validation_ids,
        "training": False,
        "checkpoint_loads": 0,
        "truth_use": "reservoir/fault/high-dip masks used only after observable graph construction for QC",
    }
    (output / "v00332ab_experiment_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    patch_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    illustrated = False
    support_illustrated = False
    for rid in ids:
        print(f"[v00332ab] full graph and indexed patches: {rid}", flush=True)
        path = dataset / "realizations" / f"realization_{rid:07d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in ("avo", "rgt", "valid_mask", "reservoir_mask")}
        arrays["structural_dip"] = structural_fields(arrays["rgt"])["dip"]
        faults = load_faults(stage02, rid)
        if rid in nodes_by_case:
            nodes, edges = nodes_by_case[rid], edges_by_case[rid]
            origin = "frozen_v00332z_csv"
        else:
            nodes, edges = observable_full_graph(arrays, frozen, z_config)
            origin = "frozen_z_algorithm_on_train_observables"
        full_shape = tuple(map(int, arrays["rgt"].shape))
        full = crop_accepted_graph(
            nodes, edges, top=0, left=0, raw_shape=full_shape, output_shape=full_shape
        )
        full_masks = region_masks(
            arrays, faults, 0, 0, full_shape, full_shape, q["adaptive_search"]["dip_q67"]
        )
        full_rows.append({
            "realization_id": rid, "split": "train" if rid in train_ids else "validation_reused",
            "source": origin,
            **graph_metrics(full, full_masks, int(CONFIG["long_edge_minimum_trace_span"])),
        })
        cases = []
        for row_index, patch_row in indexed[indexed.realization_id == rid].iterrows():
            patch = patch_row.to_dict()
            patch.update(
                output_height=int(patch["output_height"]), output_width=int(patch["output_width"])
            )
            cases.append((f"indexed_{row_index}", "indexed_training" if rid in train_ids else "indexed_validation", patch))
        for top in tile_starts(full_shape[0], 50, int(CONFIG["inference_stride"][0])):
            for left in tile_starts(full_shape[1], 100, int(CONFIG["inference_stride"][1])):
                cases.append((f"tile_{top}_{left}", "tiled_inference", {
                    "top": top, "left": left, "raw_height": 50, "raw_width": 100,
                    "output_height": 50, "output_width": 100,
                }))
        for patch_id, kind, patch in cases:
            top, left = int(patch["top"]), int(patch["left"])
            raw_shape = (int(patch["raw_height"]), int(patch["raw_width"]))
            output_shape = (int(patch["output_height"]), int(patch["output_width"]))
            cropped = crop_accepted_graph(
                nodes, edges, top=top, left=left, raw_shape=raw_shape, output_shape=output_shape
            )
            rebuilt = rebuilt_patch_graph(arrays, patch, frozen, z_config)
            masks = region_masks(
                arrays, faults, top, left, raw_shape, output_shape,
                q["adaptive_search"]["dip_q67"],
            )
            common = {
                "realization_id": rid, "split": "train" if rid in train_ids else "validation_reused",
                "patch_id": patch_id, "patch_kind": kind, "top": top, "left": left,
                "raw_height": raw_shape[0], "raw_width": raw_shape[1],
                "output_height": output_shape[0], "output_width": output_shape[1],
            }
            crop_metrics = graph_metrics(cropped, masks, int(CONFIG["long_edge_minimum_trace_span"]))
            rebuild_metrics = graph_metrics(rebuilt, masks, int(CONFIG["long_edge_minimum_trace_span"]))
            patch_rows.append({**common, "graph": "full_precompute_cropped", **crop_metrics})
            patch_rows.append({**common, "graph": "patch_observables_rebuilt", **rebuild_metrics})
            qc_rows.append({
                **common,
                "full_crop_nodes": crop_metrics["node_count"],
                "rebuilt_nodes": rebuild_metrics["node_count"],
                "full_crop_edges": crop_metrics["edge_count"],
                "rebuilt_edges": rebuild_metrics["edge_count"],
                "cropped_long_edges": crop_metrics["long_edge_count"],
                "rebuilt_long_edges": rebuild_metrics["long_edge_count"],
                "node_alignment_within_1p5_model_pixels": node_alignment(cropped, rebuilt),
                "fault_support_from_frozen_safe_components_only": True,
                "full_graph_edges_with_outside_endpoint_used": False,
            })
            if not illustrated and rid == 3400075 and kind == "indexed_validation" and crop_metrics["long_edge_count"]:
                write_figure(output / "figures", "01_full_crop_rebuild_fault_case.png", arrays, nodes, edges, patch, cropped, rebuilt)
                illustrated = True
            if rid == 3400075 and kind == "indexed_validation" and not support_illustrated:
                plt.imsave(output / "figures/02_sparse_support_mask.png", support_mask(cropped), cmap="viridis")
                support_illustrated = True
        print(f"[v00332ab] completed {len(cases)} actual indexed/tiled patches: {rid}", flush=True)
    frame = pd.DataFrame(patch_rows)
    full_frame = pd.DataFrame(full_rows)
    qc_frame = pd.DataFrame(qc_rows)
    frame.to_csv(output / "v00332ab_patch_vs_full_coverage.csv", index=False)
    full_frame.to_csv(output / "v00332ab_full_graph_statistics.csv", index=False)
    qc_frame.to_csv(output / "v00332ab_crop_rebuild_alignment.csv", index=False)
    frame[[
        "realization_id", "patch_id", "patch_kind", "graph", "node_count", "edge_count",
        "long_edge_count", "longest_edge_trace_span", "isolated_node_fraction",
        "one_hop_reach_model_pixels", "two_hop_reach_model_pixels",
    ]].to_csv(output / "v00332ab_node_edge_reach_statistics.csv", index=False)
    frame[[
        "realization_id", "patch_id", "patch_kind", "graph", "support_pixel_fraction",
        "valid_supported_pixel_fraction", "reservoir_supported_pixel_fraction",
        "high_dip_supported_pixel_fraction", "fault_corridor_supported_pixel_fraction",
        "fault_corridor_any_sparse_support",
    ]].to_csv(output / "v00332ab_fusion_support_fault_qc.csv", index=False)
    smoke = public_mechanism_smoke()
    (output / "v00332ab_mechanism_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")
    train = frame[(frame.patch_kind == "indexed_training") & (frame.graph == "full_precompute_cropped")]
    tiled = frame[(frame.patch_kind == "tiled_inference") & (frame.graph == "full_precompute_cropped")]
    train_any = float((train.edge_count > 0).mean())
    train_long = float((train.long_edge_count > 0).mean())
    tile_long = float((tiled.long_edge_count > 0).mean())
    whole_long = int(full_frame.long_edge_count.sum())
    if not len(train) or not len(tiled) or whole_long == 0:
        decision = "IMPLEMENTATION_PROBLEM"
    elif train_long < float(CONFIG["minimum_training_patch_fraction_with_long_edge"]):
        decision = "PATCH_CROPPING_DESTROYS_LONG_RANGE_VALUE"
    elif train_any < float(CONFIG["minimum_training_patch_fraction_with_any_edge"]):
        decision = "SPARSE_GRAPH_INSUFFICIENT_SUPPORT"
    elif smoke["zero_scale_max_absolute_difference"] != 0 or smoke["sparse_parameter_gradient_l1"] <= 0:
        decision = "GRADIENT_OR_NUMERICAL_FAILURE"
    else:
        decision = "HYBRID_GRAPH_READY_FOR_MATCHED_ABLATION"
    summary = {
        "decision": decision,
        "contract": contract,
        "indexed_training_patches": len(train),
        "tiled_inference_patches": len(tiled),
        "training_fraction_with_any_cropped_edge": train_any,
        "training_fraction_with_long_cropped_edge": train_long,
        "tiled_fraction_with_long_cropped_edge": tile_long,
        "whole_graph_long_edges_total": whole_long,
        "cropped_patch_zero_edge_fraction": float((train.edge_count == 0).mean()),
        "cropped_patch_zero_long_edge_fraction": float((train.long_edge_count == 0).mean()),
        "median_crop_rebuild_node_alignment": float(qc_frame.node_alignment_within_1p5_model_pixels.median()),
        "mechanism_smoke": smoke,
        "long_edge_figure_available": illustrated,
        "independent_test_cases_used": False,
        "training": False,
    }
    (output / "v00332ab_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    proposal = [
        "# Future matched ablation proposal — not executed", "",
        "A: frozen existing dense SAGE-AVO; B: same dense model plus sparse branch "
        "with component-wise degree-preserving feature-source permutation; C: same dense "
        "model plus genuine accepted v00332z tangential graph. B and C use identical node "
        "counts, destination degrees, component membership, edge descriptors distribution, "
        "parameter count, initialization, optimizer, loss, seeds, data split, patch schedule, "
        "tiled inference, and whole-realization metrics. Permutation affects only source "
        "feature assignment inside each safe component; no fault-offset link is promoted. "
        "Predeclare permutation seed and test that B truly destroys geological correspondence.",
        "", "Use the immutable v00331 train/validation/test realization split. Freeze all "
        "architecture and hyperparameters on train/validation only; reserve untouched test "
        "realizations for independent evaluation. Previously reused v00332z validation "
        "cases remain exploratory. Report whole-realization Vp/Vs/density, segmentation, "
        "physics consistency, high-dip, curved, reservoir and fault-corridor strata, "
        "including no-sparse-context regions. No superiority claim is made here.", "",
    ]
    (output / "v00332ab_matched_ablation_proposal.md").write_text("\n".join(proposal))
    report = [
        "# v00332ab hybrid dense+sparse feasibility", "",
        f"Decision: `{decision}`", "",
        "No training, optimizer update or checkpoint load. Dense graph, conditional flow, segmentation and exact-PP operator source remain unchanged; the sparse research wrapper is optional.",
        "", "## Patch gate", "",
        f"Audited all {len(train)} actual indexed training patches from six train realizations, "
        f"all {len(frame[(frame.patch_kind == 'indexed_validation') & (frame.graph == 'full_precompute_cropped')])} "
        f"indexed patches on six already-used validation cases, and {len(tiled)} actual 50×100 inference tiles. "
        "Both full-graph crop and local observable rebuild were measured for every audited patch/tile.",
        f"Full graphs contained {whole_long} edges at least 16 native traces long. "
        f"A cropped training patch retained any edge in {train_any:.1%} and a >=16-trace edge in {train_long:.1%}; "
        f"tiled inference retained a >=16-trace edge in {tile_long:.1%}.",
        f"Median cropped-versus-rebuilt node alignment within 1.5 model pixels: {summary['median_crop_rebuild_node_alignment']:.1%}. "
        "Rebuilding on patches changes detector global quantiles, track eligibility and boundary context; full-observable precomputation plus endpoint-safe crop is the only proposed consistent topology protocol.",
        "All native 40×80, 50×100 and 64×128 indexed crops were mapped to 50×100 CNN coordinates using scipy.zoom endpoint alignment. No edge with an outside endpoint is used.",
        "", "## Sparse mechanism", "",
        f"Zero-scale dense velocity parity max absolute difference: {smoke['zero_scale_max_absolute_difference']}; "
        f"sparse parameter gradient L1: {smoke['sparse_parameter_gradient_l1']:.6g}; "
        f"finite at multiple flow times and two-step Heun: {smoke['finite_multiple_flow_times'] and smoke['finite_two_step_heun']}.",
        "At active scale, each flow evaluation samples the current CNN latent with differentiable bilinear interpolation. Two TransformerConv layers use accepted tangential edges and nine observable descriptors. Fault-offset correspondences never become messages. Component-local four-pixel deposition rejects conflicting-component pixels; final velocity is exactly dense-only outside support even though the decoder has convolutions. The dense segmentation branch is deliberately unchanged.",
        "", "## Limits", "",
        "This is engineering feasibility, not inversion performance. No trained hybrid checkpoint or matched ablation exists. Region masks and fault truth are QC-only and never graph inputs. The six v00332z validation cases have been reused for development; no independent test claims. Full-topology precomputation requires whole-section AVO/RGT/support to be available before both patch training and tiled inference; the graph provider passes only in-crop endpoints to CNN features. Sparse support is local to selected reflector nodes and cannot fill all pixels. The existing training sampler draws indexed patches with replacement using foreground/structure/AVO weights; these patch tables cover every indexed candidate in the audit cohort but are not weighted by realized epoch frequency. The existing 50% horizontal-flip augmentation requires the graph's tested horizontal-mirror transform to receive the same draw, which the current generic patch loader does not yet emit. A future matched runner must wire this explicitly, or disable augmentation identically for A/B/C; this step does neither.",
        "", "The future A/B/C matched protocol is in v00332ab_matched_ablation_proposal.md. Stop before training.", "",
    ]
    (output / "v00332ab_report.md").write_text("\n".join(report))
    output.rename(final_output)
    print(json.dumps({"output": str(final_output), "decision": decision, "train_long_fraction": train_long}, indent=2))


if __name__ == "__main__":
    main()
