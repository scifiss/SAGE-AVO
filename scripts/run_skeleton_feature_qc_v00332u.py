#!/usr/bin/env python3
"""Authorized read-only CNN extraction and diagnostic-only skeleton QC."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import run_skeleton_graph_v00332u as u
import run_attention_seg_coupling_v00332r as frozen
from sage_avo.config import load_config
from sage_avo.diagnostics.skeleton_graph import (
    build_skeleton,
    graph_statistics,
    path_fault_qc,
    sample,
)
from sage_avo.diagnostics.skeleton_completion import frozen_cnn_anchors
from sage_avo.diagnostics.skeleton_feature_qc import (
    feature_distance,
    filter_decisions,
    path_observables,
    snap_candidates,
    verify_pre_gnn_tensor,
)
from sage_avo.runtime import print_torch_runtime, select_torch_device

ORIGINAL = u.OUT
PREVIOUS = ORIGINAL.with_name(ORIGINAL.name + "_requested_qc_completion")
CONFIG_PATH = u.REPO / "configs/development_diagnostics_v00332u_features.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = ORIGINAL.with_name(CONFIG["experiment_name"])
u.OUT = OUT


def feature_pairs(graph, features, strength, low_cut, high_cut, rid):
    points = graph["points"]
    sid = graph["surface_ids"]
    rows = []
    pairs = []
    for e in graph["edges"]:
        if e["relation"] == "tangential":
            group = (
                "same_short"
                if e["span"] <= 8
                else "same_medium"
                if e["span"] <= 16
                else "same_long"
            )
            pairs.append((e["source"], e["target"], group, e["kept"]))
    across = set()
    for i, p in enumerate(points):
        distance = np.linalg.norm(points - p, axis=1)
        candidates = np.flatnonzero(
            (sid != sid[i]) & (distance <= CONFIG["near_different_interface_radius"])
        )
        if len(candidates):
            j = int(candidates[np.argmin(distance[candidates])])
            across.add(tuple(sorted((i, j))))
    pairs.extend((i, j, "near_different", True) for i, j in sorted(across))
    center = features.mean(0)
    for i, j, group, kept in pairs:
        cosine, distance, normalized = feature_distance(features[i], features[j])
        centered = feature_distance(features[i] - center, features[j] - center)[0]
        quality = (
            "strong"
            if min(strength[i], strength[j]) >= high_cut
            else "weak"
            if max(strength[i], strength[j]) <= low_cut
            else "mixed"
        )
        rows.append(
            {
                "realization_id": rid,
                "source": i,
                "target": j,
                "group": group,
                "base_kept": kept,
                "anchor_strength_group": quality,
                "cartesian_distance": float(np.linalg.norm(points[j] - points[i])),
                "lateral_separation": float(abs(points[j, 1] - points[i, 1])),
                "cosine": float(cosine),
                "centered_cosine": float(centered),
                "euclidean_feature_distance": float(distance),
                "normalized_feature_distance": float(normalized),
            }
        )
    return rows


def save_figure(fig, name):
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    u.write(OUT / "figures" / name, buffer.getvalue())


def figures(cases, pairs, paths):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["cosine", "centered_cosine"]):
        for rid, g in pairs[pairs.base_kept].groupby("realization_id"):
            means = (
                g.groupby("group")[metric]
                .mean()
                .reindex(["same_short", "same_medium", "same_long", "near_different"])
            )
            ax.plot(range(4), means, marker="o", label=str(rid))
        ax.set_xticks(
            range(4), ["same short", "same medium", "same long", "near different"], rotation=15
        )
        ax.set_title(metric + " — section means, not independent edge replicates")
    axes[0].legend(fontsize=7)
    save_figure(fig, "01_cnn_feature_coherence.png")

    def background(ax, case):
        a = case["arrays"]["avo"][0]
        v = np.quantile(abs(a), 0.98)
        ax.imshow(a, cmap="gray", vmin=-v, vmax=v, aspect="auto")
        ax.set(xlabel="Trace", ylabel="Time sample")

    case = cases[2]
    g = case["graph"]
    points = g["points"]
    node = case["nodes"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, group, color in zip(axes, ["strong", "weak"], ["lime", "orange"]):
        background(ax, case)
        mask = node.strength_group == group
        ax.scatter(points[mask, 1], points[mask, 0], s=12, c=color)
        ax.set_title(group + " reflector anchors; observed AVO quartiles")
    save_figure(fig, "02_strong_weak_anchors.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax in axes:
        background(ax, case)
    axes[0].scatter(points[:, 1], points[:, 0], s=5, c="cyan")
    axes[0].set_title("Original anchors — unchanged")
    moved = case["snapped"]
    axes[1].scatter(moved[:, 1], moved[:, 0], s=5, c="orange")
    axes[1].quiver(
        points[:, 1],
        points[:, 0],
        moved[:, 1] - points[:, 1],
        moved[:, 0] - points[:, 0],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.002,
    )
    axes[1].set_title("Diagnostic snapping proposals — not adopted")
    save_figure(fig, "03_reflector_snapping.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    background(axes[0], case)
    curve = case["probe_path"]
    axes[0].plot(curve[:, 1], curve[:, 0], color="lime", linewidth=2)
    axes[0].set_title("Long curved RGT path; frozen CNN sampled along path")
    feature = case["path_features"]
    cos, _, _ = feature_distance(feature, feature[0])
    axes[1].plot(curve[:, 1], cos)
    axes[1].set(xlabel="Trace", ylabel="CNN cosine to start", ylim=(-1, 1))
    save_figure(fig, "04_long_curved_cnn_path.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    safe = paths[paths.base_kept & paths.fault_corridor.eq(False)]
    for rid, g in safe.groupby("realization_id"):
        for ax, metric in zip(axes, ["old_endpoint_prior", "path_prior"]):
            mean = g.groupby("span")[metric].median()
            ax.plot(mean.index, mean.values, marker="o", label=str(rid))
            ax.set_title(metric)
            ax.set(xlabel="Lateral span", ylabel="Median prior", ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=7)
    save_figure(fig, "05_endpoint_vs_path_prior.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, case in zip(axes, cases[4:]):
        local = paths[paths.realization_id == case["rid"]]
        for ax, keep in zip(row, [True, False]):
            background(ax, case)
            chosen = local[local.COMBINED.eq(keep) & local.base_kept].edge_id.to_numpy(int)
            chosen = chosen[:: max(1, len(chosen) // 700)]
            ax.add_collection(
                LineCollection(
                    [case["graph"]["paths"][i][:, ::-1] for i in chosen],
                    colors="cyan" if keep else "red",
                    linewidths=0.6,
                )
            )
            h, w = case["arrays"]["rgt"].shape
            y = np.arange(h)
            for f in case["faults"]:
                ax.plot(f["column"] + f["dip"] * y, y, "y--", linewidth=1)
            ax.set(
                xlim=(-0.5, w - 0.5),
                ylim=(h - 0.5, -0.5),
                title=f"{case['rid']} fixed COMBINED {'retained' if keep else 'cut'}; truth QC only",
            )
    save_figure(fig, "06_fault_retained_cut.png")


def main():
    if OUT.exists():
        raise RuntimeError("Output exists; refusing to overwrite")
    original = json.loads((ORIGINAL / "v00332u_skeleton_graph_contract.json").read_text())
    for rel, digest in original["source_sha256"].items():
        if u.sha(u.REPO / rel) != digest:
            raise RuntimeError("Frozen prototype source changed: " + rel)
    protected = {
        str(p): u.sha(p) for root in [ORIGINAL, PREVIOUS] for p in root.rglob("*") if p.is_file()
    }
    protected.update(original["input_sha256"])
    checkpoint = frozen.checkpoint(CONFIG["checkpoint_seed"], CONFIG["checkpoint_epoch"])
    protected[str(checkpoint)] = u.sha(checkpoint)
    norm_path = u.DATASET / "normalization.json"
    protected[str(norm_path)] = u.sha(norm_path)
    source_paths = [
        Path(__file__),
        CONFIG_PATH,
        u.REPO / "src/sage_avo/diagnostics/skeleton_feature_qc.py",
        u.REPO / "tests/test_skeleton_feature_qc.py",
        u.REPO / "src/sage_avo/diagnostics/skeleton_completion.py",
    ]
    runtime = print_torch_runtime()
    device = select_torch_device(
        "cuda", require_cuda=True, context="authorized frozen pre-GNN CNN inference ONLY"
    )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    contract = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": CONFIG,
        "authorization": "user explicitly authorized frozen CNN-only checkpoint inference and diagnostic snapping/path/fault QC",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": protected[str(checkpoint)],
        "source_sha256": {str(p.relative_to(u.REPO)): u.sha(p) for p in source_paths},
        "protected_sha256": protected,
        "runtime": runtime,
        "validation_ids": original["validation_ids"],
        "feature_location": "src/sage_avo/models/sage_avo.py SAGEAVO.forward:1098, before tokens at1108 and graph projection at1116",
        "feature_state": "t=0 and normalized low-frequency prior, no target elastic tensor",
        "thresholds": "fixed premeasurement choices, no inversion results and no fault-label fitting",
        "normal_search_units": "row/trace index metric; approximately gradient-normal, not physical metre normal",
        "coherence_comparison": "same-section pair groups, raw/centered cosines; descriptive not independent paired causal evidence",
        "no_training": True,
        "no_optimizer": True,
        "no_backward": True,
    }
    u.json_file("frozen_cnn_path_contract.json", contract)
    model, _ = frozen.load_frozen_model(
        CONFIG["checkpoint_seed"], CONFIG["checkpoint_epoch"], device
    )
    model.eval()
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    normalization = json.loads(norm_path.read_text())
    q = json.loads(u.Q.read_text())
    old_att = pd.read_csv(PREVIOUS / "all_candidate_attention.csv")
    node_rows = []
    pairs = []
    snaps = []
    path_rows = []
    cases = []
    feature_gates = []
    anchor_gates = []
    for rid in original["validation_ids"]:
        u.log(f"Authorized CNN/path QC: {rid}")
        with np.load(
            u.DATASET / "realizations" / f"realization_{rid:07d}.npz", allow_pickle=False
        ) as a:
            arrays = {k: a[k] for k in ["avo", "low", "rgt", "valid_mask", "reservoir_mask"]}
        tau = arrays["rgt"]
        graph = build_skeleton(tau, arrays["avo"], arrays["valid_mask"], u.CONFIG)
        original_nodes = pd.read_csv(ORIGINAL / f"node_features_{rid}.csv")
        np.testing.assert_allclose(
            graph["points"], original_nodes[["row", "trace"]].to_numpy(), rtol=0, atol=1e-10
        )
        proposed, snap = snap_candidates(tau, arrays["avo"], arrays["valid_mask"], graph, CONFIG)
        long = [
            i
            for i, e in enumerate(graph["edges"])
            if e["kept"] and e["relation"] == "tangential" and e["span"] >= 32
        ]
        probe = graph["paths"][long[len(long) // 2]]
        points = np.concatenate([graph["points"], proposed, probe])
        values, info = frozen_cnn_anchors(
            model, arrays["avo"], arrays["low"], points, normalization, device
        )
        n = len(graph["points"])
        cnn = values[:n]
        snap_cnn = values[n : 2 * n]
        if not cases:

            def norm_tile(name, mean, std):
                arr = arrays[name][:, :50, :100]
                return torch.from_numpy(
                    (
                        (arr - np.asarray(normalization[mean], np.float32)[:, None, None])
                        / np.asarray(normalization[std], np.float32)[:, None, None]
                    )[None]
                ).to(device)

            verification = verify_pre_gnn_tensor(
                model,
                norm_tile("avo", "x_mean", "x_std"),
                norm_tile("low", "y_mean", "y_std"),
                torch.from_numpy(tau[None, :50, :100]).to(device),
            )
            u.json_file(
                "exact_pre_gnn_verification.json",
                dict(
                    verification,
                    source=contract["feature_location"],
                    checkpoint_sha256=contract["checkpoint_sha256"],
                ),
            )
        u.json_file(f"cnn_sampling_{rid}.json", info)
        strength = np.sqrt(np.mean(arrays["avo"].astype(float) ** 2, axis=0))
        weak, strong = np.quantile(
            strength[arrays["valid_mask"].astype(bool)],
            [CONFIG["weak_reflector_quantile"], CONFIG["strong_reflector_quantile"]],
        )
        node_strength = sample(strength, graph["points"])
        groups = np.where(
            node_strength <= weak,
            "weak",
            np.where(node_strength >= strong, "strong", "intermediate"),
        )
        frame = original_nodes.copy()
        frame["realization_id"] = rid
        frame["node"] = np.arange(n)
        frame["cnn_norm"] = np.linalg.norm(cnn, axis=1)
        frame["cnn_rms"] = np.sqrt(np.mean(cnn**2, axis=1))
        frame["strength_group"] = groups
        frame["weak_cut"] = weak
        frame["strong_cut"] = strong
        frame = pd.concat(
            [frame, pd.DataFrame(cnn, columns=[f"cnn_{j:03d}" for j in range(cnn.shape[1])])],
            axis=1,
        )
        node_rows.append(frame)
        pair = feature_pairs(graph, cnn, node_strength, weak, strong, rid)
        pairs.extend(pair)
        a = pd.DataFrame(pair)
        a = a[a.base_kept]
        long_values = a[a.group == "same_long"].cosine
        other = a[a.group == "near_different"].cosine
        long_mean = float(long_values.mean()) if len(long_values) else -1.0
        other_mean = float(other.mean()) if len(other) else 1.0
        thresholds = CONFIG["training_gate"]
        feature_gates.append(
            {
                "realization_id": rid,
                "long_same_cosine": long_mean,
                "near_different_cosine": other_mean,
                "difference": long_mean - other_mean,
                "long_pairs": len(long_values),
                "near_different_pairs": len(other),
                "passed": bool(
                    len(long_values) > 0
                    and len(other) > 0
                    and long_mean >= thresholds["minimum_long_same_cosine"]
                    and long_mean - other_mean
                    >= thresholds["minimum_long_minus_near_different_cosine"]
                ),
            }
        )
        supported = float(np.mean((node_strength >= weak) & (frame.cnn_norm.to_numpy() > 1e-10)))
        anchor_gates.append(
            {
                "realization_id": rid,
                "supported_fraction": supported,
                "passed": supported >= thresholds["minimum_supported_anchor_fraction"],
            }
        )
        cosine, distance, normalized = feature_distance(cnn, snap_cnn)
        for i, row in enumerate(snap):
            row.update(
                realization_id=rid,
                strength_group=groups[i],
                cnn_cosine=float(cosine[i]),
                cnn_euclidean_change=float(distance[i]),
                cnn_normalized_change=float(normalized[i]),
            )
        snaps.extend(snap)
        # Construct every observable diagnostic first, then load truth for scoring.
        local = []
        lookup = old_att[old_att.realization_id == rid].set_index(["source", "target"])
        for i, (edge, path) in enumerate(zip(graph["edges"], graph["paths"])):
            if edge["relation"] != "tangential":
                continue
            obs = path_observables(tau, arrays["avo"], path, graph["scale"], CONFIG)
            row = dict(
                obs,
                realization_id=rid,
                edge_id=i,
                source=edge["source"],
                target=edge["target"],
                span=edge["span"],
                base_kept=edge["kept"],
                reciprocal_rgt_error_samples=edge["reciprocal_error_samples"],
                rgt_dip_residual=edge["dip_residual"],
                old_endpoint_prior=float(lookup.loc[(edge["source"], edge["target"]), "prior"]),
            )
            row.update(filter_decisions(obs, edge["kept"], CONFIG))
            local.append(row)
        faults = u.load_faults(u.STAGE02, rid)
        masks = u.strata(arrays, faults, q["adaptive_search"])
        for row in local:
            path = graph["paths"][row["edge_id"]]
            cross, near = path_fault_qc(path, faults)
            high = bool(
                np.isfinite(path).all()
                and (sample(masks["high_dip"].astype(float), path) >= 1 - 1e-7).all()
            )
            row.update(
                fault_crossing=cross,
                fault_corridor=near,
                high_dip_continuous=high,
                curved_long_safe=bool(
                    near is False and row["span"] >= 32 and row.get("path_chord_ratio", 1) > 1.02
                ),
            )
        path_rows.extend(local)
        cases.append(
            {
                "rid": rid,
                "arrays": arrays,
                "graph": graph,
                "faults": faults,
                "nodes": frame,
                "snapped": proposed,
                "probe_path": probe,
                "path_features": values[2 * n :],
            }
        )
    nodes = pd.concat(node_rows, ignore_index=True)
    pair_frame = pd.DataFrame(pairs)
    path_frame = pd.DataFrame(path_rows)
    u.csv_file("v00332u_cnn_feature_sampling.csv", nodes)
    u.csv_file("v00332u_interface_feature_similarity.csv", pairs)
    u.csv_file("v00332u_weak_anchor_qc.csv", snaps)
    u.csv_file("v00332u_reflector_snapping_qc.csv", snaps)
    u.csv_file("v00332u_curved_path_prior_qc.csv", path_rows)
    u.csv_file("feature_coherence_section_summary.csv", feature_gates)
    u.csv_file("anchor_support_section_summary.csv", anchor_gates)
    qc = []
    filter_pass = {}
    reach_rows = []
    current = (
        pd.read_csv(ORIGINAL / "v00332u_current_vs_skeleton.csv")
        .query('graph=="current" and region=="all"')
        .set_index("realization_id")
    )
    for rule in CONFIG["filter_order"]:
        eligible = True
        for case in cases:
            rid = case["rid"]
            local = path_frame[path_frame.realization_id == rid]
            base = local[local.base_kept]
            kept = local[local[rule]]
            crossings = local[local.fault_crossing.eq(True)]
            recall = 1 - float(crossings[rule].mean()) if len(crossings) else None
            rate = float(kept.fault_crossing.eq(True).mean()) if len(kept) else 1.0

            def retention(mask):
                sub = base[mask]
                return float(sub[rule].mean()) if len(sub) else None

            steep = retention(base.high_dip_continuous)
            curved = retention(base.curved_long_safe)
            far = retention(base.fault_corridor.eq(False))
            row = {
                "realization_id": rid,
                "filter": rule,
                "candidate_tangential_edges": len(local),
                "base_retained": len(base),
                "retained_edges": len(kept),
                "crossing_candidates": len(crossings),
                "fault_crossing_recall": recall,
                "retained_fault_crossing_rate": rate,
                "steep_continuous_retention_vs_base": steep,
                "curved_long_safe_retention_vs_base": curved,
                "away_fault_retention_vs_base": far,
            }
            qc.append(row)
            edge_ids = set(kept.edge_id)
            g = case["graph"]
            links = [
                [e["source"], e["target"]]
                for i, e in enumerate(g["edges"])
                if (e["relation"] == "normal" and e["kept"]) or i in edge_ids
            ]
            stat = graph_statistics(g["points"], links)
            ratio = stat["two_hop_reach_mean"] / current.loc[rid, "two_hop_reach_mean"]
            reach_rows.append(
                dict(realization_id=rid, filter=rule, reach_ratio=float(ratio), **stat)
            )
            if len(crossings) and (
                recall < thresholds["minimum_fault_crossing_recall"]
                or rate > thresholds["maximum_retained_crossing_rate_per_fault_section"]
            ):
                eligible = False
            if steep is not None and steep < thresholds["minimum_steep_safe_retention"]:
                eligible = False
            if curved is not None and curved < thresholds["minimum_curved_long_safe_retention"]:
                eligible = False
            if ratio < thresholds["minimum_reach_ratio"]:
                eligible = False
        filter_pass[rule] = bool(eligible)
    u.csv_file("v00332u_fault_filter_qc_updated.csv", qc)
    u.csv_file("filtered_graph_reach.csv", reach_rows)
    # Normal edges are deliberately not filtered using a same-interface rule.
    normal_rows = []
    for case in cases:
        edges = [
            (e, p)
            for e, p in zip(case["graph"]["edges"], case["graph"]["paths"])
            if e["relation"] == "normal" and e["kept"]
        ]
        crossings = sum(bool(path_fault_qc(p, case["faults"])[0]) for e, p in edges)
        normal_rows.append(
            {
                "realization_id": case["rid"],
                "retained_normal_edges": len(edges),
                "crossing_normal_edges": crossings,
                "crossing_rate": crossings / max(len(edges), 1),
            }
        )
    u.csv_file("normal_relation_fault_safety.csv", normal_rows)
    anchor_ok = all(r["passed"] for r in anchor_gates)
    coherence_ok = (
        sum(r["passed"] for r in feature_gates) >= thresholds["minimum_favorable_feature_sections"]
    )
    safe = path_frame[path_frame.base_kept & path_frame.curved_long_safe]
    path_fraction = (
        float((safe.path_prior >= thresholds["high_confidence_path_prior"]).mean())
        if len(safe)
        else 0.0
    )
    path_ok = bool(
        len(safe) > 0 and path_fraction >= thresholds["minimum_high_confidence_path_fraction"]
    )
    # A tangential-only fix cannot authorize a graph whose normal relation still crosses faults.
    normal_ok = all(
        r["crossing_rate"] <= thresholds["maximum_retained_crossing_rate_per_fault_section"]
        for r in normal_rows
    )
    fault_ok = any(filter_pass.values()) and normal_ok
    sparse_ok = all(
        len(c["graph"]["points"]) / c["arrays"]["valid_mask"].sum()
        <= thresholds["maximum_node_fraction"]
        for c in cases
    )
    blockers = []
    if not (anchor_ok and coherence_ok):
        blockers.append("ANCHORING_STILL_WEAK")
    if not path_ok:
        blockers.append("CURVED_PATH_PRIOR_STILL_FAILS")
    if not fault_ok:
        blockers.append("FAULT_FILTER_STILL_FAILS")
    if not sparse_ok:
        blockers.append("IMPLEMENTATION_PROBLEM")
    decision = (
        "SKELETON_READY_FOR_TRAINING"
        if not blockers
        else blockers[0]
        if len(blockers) == 1
        else "MULTIPLE_BLOCKERS_REMAIN"
    )
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[name]):
            raise RuntimeError("Checkpoint weights changed: " + name)
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected gradient")
    for path, digest in protected.items():
        if u.sha(path) != digest:
            raise RuntimeError("Protected input changed: " + path)
    for rel, digest in original["protected_production_sha256"].items():
        if u.sha(u.REPO / rel) != digest:
            raise RuntimeError("Production source changed: " + rel)
    summary = {
        "decision": decision,
        "blockers": blockers,
        "anchor_support_gate": anchor_ok,
        "feature_coherence_gate": coherence_ok,
        "path_prior_gate": path_ok,
        "safe_curved_long_high_prior_fraction": path_fraction,
        "fault_filter_trials_pass": filter_pass,
        "normal_fault_gate": normal_ok,
        "sparsity_gate": sparse_ok,
        "training_justified_next": not blockers,
        "training_performed": False,
        "weights_unchanged": True,
        "anchors_unchanged": True,
        "old_outputs_unchanged": True,
        "graph_message_passing_executed": False,
        "flow_or_loss_executed": False,
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "cnn_channels": int(cnn.shape[1]),
        "sampled_nodes": len(nodes),
        "snapped_fraction_diagnostic_only": float(pd.DataFrame(snaps).changed.mean()),
        "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    figures(cases, pair_frame, path_frame)
    report = f"""# v00332u — authorized frozen CNN, anchor and path QC

Decision: **{decision}**.

## Exact feature provenance

Checkpoint: `{checkpoint.name}`, seed {CONFIG["checkpoint_seed"]}, epoch {CONFIG["checkpoint_epoch"]}.
SHA-256: `{contract["checkpoint_sha256"]}`.
Source: `src/sage_avo/models/sage_avo.py`, `SAGEAVO.forward`, encoder output line1098,
before flatten-to-tokens at1108 and graph projection/message passing at1116.
The real forward was stopped by an encoder-output hook; its tensor was bitwise
equal to the direct encoder extraction. No graph/decoder call occurred.
Dense tile shape {verification["shape"]}; anchor features [{len(nodes)},{cnn.shape[1]}].
Flow time=0, current state=normalized low prior, conditioned on normalized AVA
and prior through existing embeddings. This is one observable initial state,
not a sample of all flow times, and no target elastic tensor was used.
Batch1 float32 CUDA, model.eval and inference_mode, existing train-only
normalization, 50x100 tiles/25x50 stride and Hann stitching. GroupNorm makes
features tile-conditioned; stitched features are not a hypothetical full-section
encoder forward. No optimizer, backward, loss, flow, or training was executed.

## Feature coherence and support

Strong/weak bins are per-section observable-pixel RMS quartiles, fixed before
CNN measurement. Same-interface pairs use existing tangential candidates,
with base-kept status retained. Nearby different-interface pairs are unique
nearest cross-interface nodes within12 grid units. Short<=8 traces, medium16,
long>=32. Cosine, L2 and normalized L2 are per pair; centered cosine subtracts
the section anchor-mean feature to expose common activation bias.
Comparisons are descriptive: distance and interface occupancy differ, edges
are dependent, and reused validation sections are not new independent evidence.
Section-level gate requires mean long-same cosine>=.5 and advantage>=.05 over
near-different in at least5/6 sections. This is not a significance test or proof
of inversion benefit. Both all-candidate and base-kept groups remain in CSV.

Feature gate: {coherence_ok}. Anchor support gate: {anchor_ok}.
Tables: v00332u_cnn_feature_sampling.csv (full CNN vector, norm/RMS, AVA/P/G/C,
tau/dip/curvature, strength, interface and coordinates),
v00332u_interface_feature_similarity.csv, and section summary CSVs.

## Weak anchors and diagnostic snapping

Search +/-2 grid units in .25 increments along normalized local RGT gradient.
Tolerance=min(.75 reference vertical RGT increment, .24 adjacent-interface tau
gap). Every intervening point must be supported, remain within this tau band
and between neighboring inverse surfaces, retain gradient direction cosine>=.9
and gradient magnitude ratio<=4. These checks preserve selected-interface
ordering and avoid OBSERVABLY detected discontinuities, not undetectable faults.
Ties prefer no displacement; strongest compatible observed reflector wins.
CNN similarity is measured afterward, not used to select candidates.
Original nodes/graph remain unchanged. Snapped fraction: {summary["snapped_fraction_diagnostic_only"]:.2%}.
All anchors, including weak ones, retain displacement, strength gain, RGT
deviation/tolerance and CNN change. No proposals were adopted.

## Curved path prior (diagnostic, not deployed)

On every segment of the actual inverse-RGT path, sample midpoint gradient and
local unit tangent. Let c=cos(angle(gradient,tangent)); let e be the mean
absolute tau deviation along vertices AND midpoints divided by section median
positive vertical RGT increment. p_path=exp(-.5(e/.5)^2-.5(RMS(c)/.2)^2).
There is no endpoint-gradient-dot-long-chord term or accumulated length penalty.
CSV retains old endpoint prior, endpoint/mean/max tau errors, RMS/max angular
consistency, geodesic/chord lengths, ratio, and curvature statistics.
Analytic constant-RGT curved-path regression requires prior>.999 up to64 traces.
High-confidence fraction on base-kept long curved away-fault paths:
{path_fraction:.2%}; operational pass requires>=90% with prior>=.5.
This path-local consistency is derived from the same RGT as topology; it cannot
by itself detect a geologically incorrect but internally consistent RGT surface.

## Fixed observable fault-filter trials

BASELINE preserves u; COHERENCE adds path waveform cosine p10>=.8; CONTINUITY
adds reflector RMS p10/p90>=.25 and shift-median residual<=2; COMBINED adds both.
Waveforms are three-band +/-3 local-normal samples at each path point, with
invalid/zero-energy windows failing closed. RGT inverse roundtrip and dip
residual are also exported. No independently stored PWD product is available.
The new waveform coherence is inference-observable, not fault truth.
Thresholds were frozen before outcomes; none were fitted to inversion metrics
or fault labels. Truth only scores crossing recall, retained crossing rate,
steep continuous retention and curved-long safe retention. No trial is adopted.
Operational gates require>=90% crossing recall, <=2% retained crossings per
fault section, >=90% steep/curved safe retention, and >2x local graph reach.
Normal edges are scored separately and not given a same-interface filter.
They remain a hard safety blocker if their crossings exceed the same2% bound.

Trial passes: {filter_pass}. Normal-relation safety pass: {normal_ok}.
Candidate recall and conditional retention have different denominators; missing
inverse endpoints precede the candidate set. Empty QC strata are N/A, not passes
supported by evidence. Full trial reach/degree and normal-safety tables included.

## Training gate and limits

```json
{json.dumps(summary, indent=2)}
```

No conclusion here establishes a superior trained skeleton GNN. The dense CNN
was not optimized for this sparse graph, and coherence may reflect priors,
normalization or shared amplitudes. No production model/source, checkpoint,
existing anchor or prior result was modified. Six diagnostic figures are saved
under figures/. Stop after report, including if the operational gate passes.
"""
    u.write(OUT / "v00332u_completion_report.md", report.encode())
    u.json_file("v00332u_completion_summary.json", summary)
    u.log(f"{decision}; finished: {OUT}")


if __name__ == "__main__":
    main()
