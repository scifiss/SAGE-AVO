#!/usr/bin/env python3
"""No-training sparse skeleton prototype; only writes a new private u directory."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import hashlib
import json
from pathlib import Path
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch

from sage_avo.config import load_config
from sage_avo.diagnostics.rgt_topology_repair import load_faults, structural_fields
from sage_avo.diagnostics.skeleton_graph import (
    attention_quantities,
    build_skeleton,
    graph_statistics,
    path_fault_qc,
    sample,
)
from sage_avo.models.graph import build_experimental_rgt_relations

REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/development_diagnostics_v00332u.yaml"
CONFIG = load_config(CONFIG_PATH)
PRIVATE = Path(load_config(REPO / "configs/paths.yaml")["private_artifact_root"])
BASE = PRIVATE / "stage_artifacts"
OUT = BASE / "stage04" / CONFIG["experiment_name"]
DATASET = BASE / "stage03/ds_v00331_production100_support_aware/dataset"
STAGE02 = BASE / "stage02/v00331_production100_support_aware/realizations"
Q = BASE / "stage04/sage_avo_s01_v00332q_clean_20epoch_corrected_rgt/v00332q_contract.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, content):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("Refusing output outside new u artifacts")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".incomplete")
    with tmp.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def json_file(name, payload):
    write(OUT / name, (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def csv_file(name, rows):
    write(OUT / name, pd.DataFrame(rows).to_csv(index=False).encode())


def log(message):
    print(f"[v00332u {time.strftime('%H:%M:%S')}] {message}", flush=True)


def strata(arrays, faults, thresholds):
    tau = arrays["rgt"]
    valid = arrays["valid_mask"].astype(bool)
    y, x = np.indices(tau.shape)
    corridor = np.zeros_like(valid)
    for fault in faults:
        corridor |= abs(x - float(fault["column"]) - float(fault["dip"]) * y) <= 3
    dip = structural_fields(tau)["dip"]
    return {
        "all": valid,
        "low_dip": valid & (dip < thresholds["dip_q33"]),
        "high_dip": valid & (dip >= thresholds["dip_q67"]) & ~corridor,
        "fault_corridor": valid & corridor,
        "reservoir": valid & arrays["reservoir_mask"].astype(bool),
    }


def current_graph(tau, valid, thresholds):
    h, w = tau.shape
    tang, normal = build_experimental_rgt_relations(
        torch.from_numpy(tau[None]),
        topology="rgt_v3_confidence_blocked",
        max_shift=3,
        confidence_normalized_mismatch_threshold=thresholds["normalized_best_mismatch"],
        confidence_normalized_discontinuity_threshold=thresholds[
            "normalized_cartesian_discontinuity"
        ],
        confidence_dip_residual_threshold=thresholds["rgt_dip_residual"],
    )
    points = np.column_stack(np.indices((h, w)).reshape(2, -1))
    pairs = []
    relations = []
    for relation, edges in [("tangential", tang[0]), ("normal", normal[0])]:
        edge = edges.numpy().T
        edge = edge[: len(edge) // 2]
        edge = edge[valid.ravel()[edge].all(1)]
        pairs.extend(edge.tolist())
        relations.extend([relation] * len(edge))
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    used = np.flatnonzero(valid.ravel())
    mapping = np.full(h * w, -1)
    mapping[used] = np.arange(len(used))
    return points[used].astype(float), mapping[pairs], relations


def comparison_rows(rid, name, points, pairs, tau, masks, faults):
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    rows = []
    for region, mask in masks.items():
        selected = sample(mask.astype(float), points) >= 1 - 1e-7
        nodes = np.flatnonzero(selected)
        mapping = np.full(len(points), -1)
        mapping[nodes] = np.arange(len(nodes))
        local = pairs[selected[pairs].all(1)]
        record = graph_statistics(points[nodes], mapping[local])
        lengths = np.linalg.norm(points[local[:, 1]] - points[local[:, 0]], axis=1)
        mismatch = abs(sample(tau, points[local[:, 1]]) - sample(tau, points[local[:, 0]]))
        crossing = [path_fault_qc(points[pair], faults)[0] for pair in local]
        record.update(
            {
                "realization_id": rid,
                "graph": name,
                "region": region,
                "mean_rgt_mismatch": float(mismatch.mean()) if len(local) else None,
                "non_cartesian_adjacency_fraction": float(np.mean(lengths > np.sqrt(2) + 1e-6))
                if len(local)
                else None,
                "endpoint_chord_fault_crossing_fraction": float(np.mean(crossing))
                if crossing
                else None,
            }
        )
        rows.append(record)
    return rows


def plots(cases):
    def background(ax, case, rgt=False):
        a = case["arrays"]
        data = a["rgt"] if rgt else a["avo"][0]
        limit = np.quantile(abs(data), 0.98)
        ax.imshow(
            data,
            cmap="viridis" if rgt else "gray",
            aspect="auto",
            **({} if rgt else {"vmin": -limit, "vmax": limit}),
        )
        ax.set(xlabel="Trace index", ylabel="Time sample")

    def edges(ax, case, relation="tangential", kept=True, color="tab:blue", maximum=1000):
        g = case["graph"]
        indices = [
            i
            for i, e in enumerate(g["edges"])
            if e["relation"] == relation and e["kept"] == kept and np.isfinite(g["paths"][i]).all()
        ]
        indices = indices[:: max(1, int(np.ceil(len(indices) / maximum)))]
        collection = LineCollection(
            [g["paths"][i][:, ::-1] for i in indices], colors=color, linewidths=0.55, alpha=0.65
        )
        ax.add_collection(collection)
        return indices

    def save(fig, name):
        import io

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160)
        plt.close(fig)
        write(OUT / "figures" / name, buffer.getvalue())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, case in zip(axes.ravel(), cases):
        background(ax, case)
        g = case["graph"]
        ax.scatter(g["points"][:, 1], g["points"][:, 0], c=g["surface_ids"], cmap="tab20", s=5)
        ax.set_title(f"{case['rid']}: {len(g['points'])} sparse interface nodes")
    save(fig, "01_sparse_nodes.png")
    case = cases[2]
    g = case["graph"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (name, spans) in zip(axes, [("Short", [4, 8]), ("Medium", [16]), ("Long", [32, 64])]):
        background(ax, case)
        chosen = {}
        for i, e in enumerate(g["edges"]):
            sid = g["surface_ids"][e["source"]]
            x = g["points"][e["source"], 1]
            if e["kept"] and e["span"] in spans and x >= (sid % 4) * 16:
                chosen.setdefault((sid, e["span"]), i)
        for (sid, span), i in chosen.items():
            path = g["paths"][i]
            color = plt.get_cmap("tab20")(int(sid) % 20)
            ax.plot(path[:, 1], path[:, 0], color=color, linewidth=1.7)
            ax.scatter(path[0, 1], path[0, 0], color=color, marker="o", s=15)
            ax.scatter(path[-1, 1], path[-1, 0], color=color, marker=">", s=25)
        ax.set_title(f"{name}: spans {spans}; sampled links")
    save(fig, "02_variable_length_edges.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, case in zip(axes, cases[2:4]):
        background(ax, case, rgt=True)
        edges(ax, case, color="white")
        ax.set_title(f"High-dip section {case['rid']}")
    save(fig, "03_high_dip_graph.png")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row, case in zip(axes, cases[4:]):
        for ax, kept in zip(row, [True, False]):
            background(ax, case)
            edges(ax, case, kept=kept, color="tab:cyan" if kept else "tab:red")
            h = case["arrays"]["rgt"].shape[0]
            y = np.arange(h)
            for f in case["faults"]:
                ax.plot(float(f["column"]) + float(f["dip"]) * y, y, "y--", linewidth=1)
            ax.set_xlim(-0.5, case["arrays"]["rgt"].shape[1] - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            ax.set_title(
                f"{case['rid']}: {'retained' if kept else 'removed'} edges; truth faults QC only"
            )
    save(fig, "04_fault_graph.png")
    case = cases[2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax in axes:
        background(ax, case)
    p, pairs, _ = case["current"]
    step = max(1, len(pairs) // 7000)
    axes[0].add_collection(
        LineCollection(p[pairs[::step]][:, :, ::-1], colors="tab:cyan", linewidths=0.35)
    )
    axes[0].set_title("Current dense graph (edges subsampled for display)")
    edges(axes[1], case)
    axes[1].set_title("Sparse variable-length interface skeleton")
    save(fig, "05_current_vs_skeleton.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, case in zip(axes, cases[::2]):
        background(ax, case)
        lookup = {
            (r["source"], r["target"]): r["prior"]
            for r in case["attention"]
            if r["relation"] == "tangential"
        }
        g = case["graph"]
        ix = [i for i, e in enumerate(g["edges"]) if e["kept"] and e["relation"] == "tangential"][
            ::3
        ]
        lc = LineCollection([g["paths"][i][:, ::-1] for i in ix], cmap="plasma", linewidths=0.8)
        lc.set_array(
            np.array([lookup[g["edges"][i]["source"], g["edges"][i]["target"]] for i in ix])
        )
        lc.set_clim(0, 1)
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label="Fixed p_RGT (not learned alpha)")
        ax.set_title(str(case["rid"]))
    save(fig, "06_rgt_attention_prior.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    case = cases[2]
    for ax, relation in zip(axes, ["tangential", "normal"]):
        background(ax, case)
        edges(
            ax,
            case,
            relation=relation,
            color="tab:cyan" if relation == "tangential" else "tab:orange",
        )
        ax.set_title(relation + " relation (separate families)")
    save(fig, "07_tangential_vs_normal.png")


def main():
    torch.set_num_threads(2)
    if OUT.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite prototype: {OUT}")
    q = json.loads(Q.read_text())
    ids = q["diverse_validation_subset"]["all_ids"]
    input_paths = [Q] + [DATASET / "realizations" / f"realization_{rid:07d}.npz" for rid in ids]
    input_paths += [STAGE02 / f"realization_{rid:07d}.json" for rid in ids]
    protected = {
        str(p.relative_to(REPO)): sha(p)
        for root in ["src/sage_avo/models", "src/sage_avo/training", "src/sage_avo/forward"]
        for p in (REPO / root).glob("*.py")
    }
    sources = [
        CONFIG_PATH,
        Path(__file__),
        REPO / "src/sage_avo/diagnostics/skeleton_graph.py",
        REPO / "tests/test_skeleton_graph.py",
    ]
    contract = {
        "revision": "v00332u",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "config": CONFIG,
        "source_sha256": {str(p.relative_to(REPO)): sha(p) for p in sources},
        "protected_production_sha256": protected,
        "input_sha256": {str(p): sha(p) for p in input_paths},
        "validation_ids": ids,
        "region_selection": "same six frozen q sections; low/high dip/fault/reservoir masks only QC",
        "truth_usage": "faults and reservoir labels used only AFTER graph construction for QC/plots",
        "device": "CPU topology and statistics; no GPU training/inference or checkpoint load",
        "cnn_features": "optional sampler implemented and tested; no CNN feature map supplied",
        "units": "sample/trace index, Euclidean grid units, not metres; no physical isotropy claimed",
        "attention": "fixed lambda=1, content logits=0, b_geom=0; relation-separated prior-only demonstration",
        "normal_relation": "next selected interface at same trace; contrast connectivity, not exact geometric normal",
        "confidence": "new fixed observable path checks; no fault labels or validation-fitted thresholds",
        "independent_pwd_or_coherence": "not available as stored model input; RGT-derived dip is not independent PWD",
        "decision_thresholds_status": "operational topology screen declared before measurements, not statistical significance",
        "training": False,
    }
    json_file("v00332u_skeleton_graph_contract.json", contract)
    nodes = []
    edge_rows = []
    qc = []
    comparison = []
    attention = []
    cases = []
    for rid in ids:
        log(f"Building observable-only sparse graph for {rid}")
        with np.load(
            DATASET / "realizations" / f"realization_{rid:07d}.npz", allow_pickle=False
        ) as f:
            arrays = {name: f[name] for name in ["rgt", "avo", "valid_mask", "reservoir_mask"]}
        tau = arrays["rgt"]
        valid = arrays["valid_mask"].astype(bool)
        g = build_skeleton(tau, arrays["avo"], valid, CONFIG)
        att = attention_quantities(tau, g["points"], g["edges"], CONFIG)
        if len(g["points"]) == 0 or not att:
            raise RuntimeError("No usable skeleton nodes/edges")
        # Truth is loaded only after observable-only construction and attention.
        faults = load_faults(STAGE02, rid)
        masks = strata(arrays, faults, q["adaptive_search"])
        cur = current_graph(tau, valid, q["confidence_rule"]["thresholds"])
        for sid, level in enumerate(g["levels"]):
            count = int((g["surface_ids"] == sid).sum())
            nodes.append(
                {
                    "realization_id": rid,
                    "interface_id": sid,
                    "tau": float(level),
                    "node_count": count,
                    "dense_valid_node_count": int(valid.sum()),
                    "skeleton_node_count": len(g["points"]),
                    "compression_ratio": float(valid.sum() / len(g["points"])),
                    "node_fraction": float(len(g["points"]) / valid.sum()),
                }
            )
        for e, path in zip(g["edges"], g["paths"]):
            cross, near = path_fault_qc(path, faults)
            finite = np.isfinite(path).all()
            high = (
                bool((sample(masks["high_dip"].astype(float), path) >= 1 - 1e-7).all())
                if finite
                else False
            )
            row = dict(
                e,
                realization_id=rid,
                fault_crossing=cross,
                fault_corridor=near,
                high_dip_continuous=high,
                normalized_rgt_mismatch=e["rgt_mismatch"] / g["scale"],
            )
            edge_rows.append(row)
        local = pd.DataFrame([r for r in edge_rows if r["realization_id"] == rid])
        for relation in ["tangential", "normal"]:
            sub = local[local.relation == relation]
            for name, mask in [
                ("all", np.ones(len(sub), bool)),
                ("fault_crossing", sub.fault_crossing.eq(True)),
                ("safe_away_from_fault", sub.fault_corridor.eq(False)),
                ("high_dip_continuous", sub.high_dip_continuous.eq(True)),
            ]:
                part = sub[mask]
                kept = part[part.kept]
                qc.append(
                    {
                        "realization_id": rid,
                        "relation": relation,
                        "region": name,
                        "candidate_edges": len(part),
                        "retained_edges": len(kept),
                        "retention": len(kept) / len(part) if len(part) else None,
                        "retained_path_fault_crossing_rate": float(
                            kept.fault_crossing.eq(True).mean()
                        )
                        if len(kept)
                        else None,
                    }
                )
        pairs = [[e["source"], e["target"]] for e in g["edges"] if e["kept"]]
        comparison += comparison_rows(rid, "current", cur[0], cur[1], tau, masks, faults)
        comparison += comparison_rows(rid, "skeleton", g["points"], pairs, tau, masks, faults)
        attention.extend(dict(r, realization_id=rid) for r in att)
        degree = np.bincount(np.asarray(pairs, int).ravel(), minlength=len(g["points"]))
        strength = np.sqrt(np.mean(arrays["avo"].astype(float) ** 2, axis=0))
        csv_file(
            f"node_features_{rid}.csv",
            pd.DataFrame(
                g["features"],
                columns=["near", "mid", "far", "P", "G", "C", "tau", "dip", "curvature"],
            ).assign(
                row=g["points"][:, 0],
                trace=g["points"][:, 1],
                interface=g["surface_ids"],
                degree=degree,
                reflector_strength=sample(strength, g["points"]),
            ),
        )
        csv_file(
            f"interface_selection_{rid}.csv",
            pd.DataFrame(
                {
                    "tau": g["selection"]["candidate_tau"],
                    "reflector_strength": g["selection"]["reflector_score"],
                    "coverage": g["selection"]["coverage"],
                    "selected": np.isin(
                        np.arange(CONFIG["level_samples"]), g["selection"]["selected_indices"]
                    ),
                }
            ),
        )
        cases.append(
            {
                "rid": rid,
                "arrays": arrays,
                "graph": g,
                "attention": att,
                "faults": faults,
                "current": cur,
            }
        )
        log(f"{rid}: {len(g['points'])} nodes, {len(pairs)} retained edges")
    csv_file("v00332u_node_statistics.csv", nodes)
    csv_file("v00332u_edge_statistics.csv", edge_rows)
    csv_file("v00332u_fault_qc.csv", qc)
    csv_file("v00332u_current_vs_skeleton.csv", comparison)
    csv_file("v00332u_attention_quantities.csv", attention)
    plots(cases)
    frame = pd.DataFrame(edge_rows)
    tan = frame[frame.relation == "tangential"]
    kept = tan[tan.kept]
    histogram = []
    boundaries = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for (rid, relation, keep), part in frame.groupby(["realization_id", "relation", "kept"]):
        counts, _ = np.histogram(part.cartesian_length, bins=boundaries)
        for low, high, count in zip(boundaries[:-1], boundaries[1:], counts):
            histogram.append(
                {
                    "realization_id": rid,
                    "relation": relation,
                    "kept": bool(keep),
                    "lower_length": low,
                    "upper_length": high,
                    "edge_count": int(count),
                }
            )
    csv_file("v00332u_edge_length_histogram.csv", histogram)
    comp = pd.DataFrame(comparison)
    whole = comp[comp.region == "all"].set_index(["realization_id", "graph"])
    ratio = [
        whole.loc[(rid, "skeleton"), "two_hop_reach_mean"]
        / max(whole.loc[(rid, "current"), "two_hop_reach_mean"], 1e-12)
        for rid in ids
    ]

    def retention(mask):
        part = tan[mask]
        return float(part.kept.mean()) if len(part) else 0.0

    fault_candidates = tan[tan.fault_crossing.eq(True)]
    fault_reduction = 1 - float(fault_candidates.kept.mean()) if len(fault_candidates) else 0.0
    priors = pd.DataFrame(attention)
    prior = priors[priors.relation == "tangential"].prior
    metrics = {
        "maximum_node_fraction": float(max(n["node_fraction"] for n in nodes)),
        "minimum_two_hop_reach_ratio": float(min(ratio)),
        "nonlocal_tangential_fraction": float((kept.cartesian_length > np.sqrt(2)).mean()),
        "long_tangential_fraction": float((kept.span >= 32).mean()),
        "normalized_tangential_mismatch_p95": float(kept.normalized_rgt_mismatch.quantile(0.95)),
        "safe_retention": retention(tan.fault_corridor.eq(False)),
        "high_dip_safe_retention": retention(tan.high_dip_continuous.eq(True)),
        "fault_crossing_reduction_same_candidate_set": fault_reduction,
        "retained_tangential_fault_crossing_fraction": float(kept.fault_crossing.eq(True).mean()),
        "tangential_prior_median": float(prior.median()),
        "tangential_prior_below_epsilon_fraction": float(
            (prior < CONFIG["attention_epsilon"]).mean()
        ),
        "normal_prior_median": float(priors[priors.relation == "normal"].prior.median()),
        "attention_finite": bool(np.isfinite(priors[["logit", "alpha"]]).all().all()),
    }
    d = CONFIG["decision"]
    if not metrics["attention_finite"]:
        decision = "IMPLEMENTATION_PROBLEM"
    elif metrics["maximum_node_fraction"] > d["maximum_node_fraction"]:
        decision = "SKELETON_GRAPH_TOO_DENSE"
    elif (
        metrics["normalized_tangential_mismatch_p95"] > d["maximum_normalized_mismatch_p95"]
        or metrics["tangential_prior_median"] < d["minimum_tangential_prior_median"]
    ):
        decision = "SKELETON_GRAPH_POOR_INTERFACE_ALIGNMENT"
    elif (
        fault_reduction < d["minimum_fault_crossing_reduction"]
        or metrics["retained_tangential_fault_crossing_fraction"]
        > d["maximum_retained_fault_crossing_fraction"]
        or metrics["safe_retention"] < d["minimum_safe_retention"]
        or metrics["high_dip_safe_retention"] < d["minimum_high_dip_safe_retention"]
    ):
        decision = "SKELETON_GRAPH_FAULT_FAILURE"
    elif (
        min(ratio) < d["minimum_two_hop_reach_ratio"]
        or metrics["long_tangential_fraction"] < d["minimum_long_edge_fraction"]
        or metrics["nonlocal_tangential_fraction"] < d["minimum_nonlocal_fraction"]
    ):
        decision = "SKELETON_GRAPH_NOT_DISTINCT_FROM_CURRENT"
    else:
        decision = "SKELETON_GRAPH_PROMISING"
    for rel, digest in protected.items():
        if sha(REPO / rel) != digest:
            raise RuntimeError("Protected production source changed: " + rel)
    for path, digest in contract["input_sha256"].items():
        if sha(path) != digest:
            raise RuntimeError("Input artifact changed: " + path)
    summary = {
        "decision": decision,
        "metrics": metrics,
        "training_performed": False,
        "production_source_unchanged": True,
        "inputs_unchanged": True,
        "controlled_training_justified_by_topology_screen": decision == "SKELETON_GRAPH_PROMISING",
        "no_claim_of_inversion_gain": True,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    display = comp[comp.region == "all"]
    table = "| " + " | ".join(display.columns) + " |\n"
    table += "| " + " | ".join(["---"] * len(display.columns)) + " |\n"
    table += "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in display.itertuples(index=False, name=None)
    )
    report = f"""# v00332u — sparse stratigraphic skeleton, no training

Decision: **{decision}**.

## Design and scope

Same six frozen q validation realizations: {ids}. No outcome/checkpoint selection,
CNN execution, model initialization, optimizer, training, flow/physics/loss edit,
commit or push. Production source and data hashes verified unchanged.
CPU construction/statistics only. Four new source/config/test files; all products private.

Sparse anchors lie on up to 16 separated peaks of observed three-band RMS
reflector strength indexed by RGT; fractional inverse crossing every 4 traces.
Edges span 4,8,16,32,64 traces (1,2,4,8,16 node spacings). Geometry is evaluated
at every intervening trace. Ambiguous/nonmonotone/plateau inverse branches are
omitted, not sorted. Selected interface count may be smaller than 16.
Node features are [near,mid,far,P,G,C,tau,dip,curvature]; optional frozen CNN
feature sampling is implemented and tested but unused, not replaced by random features.

Full-path support, inverse roundtrip, positive RGT scale, maximum shift,
shift-field median residual and RGT-derived dip residual cut edges. Thresholds
are fixed operational prototype choices in the contract, not optimized on QC.
Independent stored PWD/coherence is unavailable; RGT-derived dip is NOT independent
evidence. Fault geometry and reservoir masks enter QC only after construction.
Normal edges connect consecutive selected interfaces at the same trace: a distinct
cross-layer relation, but not a solved geometric-normal trajectory.

## Quantitative comparison

Undirected unique edges; both directions would be used for message passing.
Counts use valid support. Regional rows use induced subgraphs; whole-section
edges are never reconstructed from the masks. All length/reach values use
row/trace grid units, not metres. Edge density E/(N(N-1)/2), sparsity=1-density.
One/two-hop reach is the per-node maximum Cartesian distance, averaged over
ALL nodes including isolated nodes. Dense graph is the unchanged confidence-
blocked corrected-RGT builder plus vertical edges. No inference improvement is measured.

{table}

## Operational screen metrics

```json
{json.dumps(metrics, indent=2)}
```

CSV edge statistics retain each candidate, relation, length/span, geodesic
distance, mismatch, reason for removal and post-hoc fault QC. Fault crossing
in skeleton QC tests every piecewise interface segment, including crossings
followed by returns. Comparison CSV reports endpoint-chord crossing separately:
these are different geometries/denominators, not interchangeable percentages.
Safe/high-dip retention use the SAME pre-filter candidate set. An edge's tiny
RGT mismatch is partly guaranteed by inversion and is NOT fault identity proof.

## Explicit RGT attention prior

delta_tau=tau_j-tau_i; r=(delta_row,delta_trace).
log(p)=-0.5(delta_tau/sigma_tau)^2-0.5(grad(tau)_i dot r/sigma_g)^2.
sigma_tau=one median positive vertical RGT increment; sigma_g=four increments.
alpha=softmax within (source,relation) of q_i.k_j/sqrt(d)+lambda*log(p+eps)+b_geom.
Here q.k=0, lambda=1, eps=1e-12 and b_geom=0. Geometry features (row/trace offset,
length, delta_tau, dip/curvature differences) are exported, not fitted.
Stable logaddexp/log-softmax avoids underflow-related NaN; the epsilon floor can
still suppress useful discrimination. Endpoint tangent priors can penalize long
CURVED but aligned paths. Normal delta-tau suppression is a deliberate diagnostic
counterexample: a same-interface prior must NOT be blindly shared with contrast edges.

## Answers

1. Longer communication: measured minimum two-hop ratio versus current GNN is
{min(ratio):.3f}. This compares graph hops, NOT the total multiscale CNN receptive field.
2. Sparsity: maximum node fraction {metrics["maximum_node_fraction"]:.4%}.
3. Alignment: normalized tangential mismatch p95 {metrics["normalized_tangential_mismatch_p95"]:.4g};
inverse interpolation guarantees much of this. Discontinuity QC remains essential.
4. Fault crossings: candidate crossing removal {fault_reduction:.2%}, retained
tangential crossing rate {metrics["retained_tangential_fault_crossing_fraction"]:.2%}.
5. RGT use: sparse surfaces and long-range links differ qualitatively from pixel
adjacency; explicit additive prior is implemented, not merely an edge feature.
6. Numerical attention status: finite={metrics["attention_finite"]}; prior median
{metrics["tangential_prior_median"]:.4g}. Curvature/normal-relation limitations above apply.
7. Controlled training next justified by this topology screen:
{summary["controlled_training_justified_by_topology_screen"]}. No training is authorized or run here.

## Visual review

Figures 01–07 contain actual seismic/RGT overlays, kept/removed tangential paths,
normal paths and prior strengths. Dense edges and some skeleton edges are
deterministically subsampled for legibility; the CSV metrics use ALL edges.
Visual inspection and validation results will be appended after completion.
"""
    write(OUT / "v00332u_report.md", report.encode())
    json_file("v00332u_summary.json", summary)
    log(f"{decision}; complete: {OUT}")


if __name__ == "__main__":
    main()
