#!/usr/bin/env python3
"""Complete requested v00332u QC without redesigning the frozen failed topology.

Writes a separate completion package. Original u outputs and source stay intact.
Does not load a checkpoint. Frozen CNN sampling awaits explicit authorization.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import run_skeleton_graph_v00332u as u
from sage_avo.diagnostics.skeleton_graph import attention_quantities, build_skeleton
from sage_avo.diagnostics.skeleton_completion import (
    neighborhood_summary,
    signed_prior_softmax,
)
from sage_avo.runtime import print_torch_runtime, select_torch_device

OLD = u.OUT
OUT = OLD.with_name(OLD.name + "_requested_qc_completion")
# Redirect only the imported diagnostic's output guard in this isolated process.
# Do not invoke its main(), train a model, or overwrite its original artifacts.
u.OUT = OUT


def extra_figure(cases):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, curved in zip(axes[0], [False, True]):
        x = np.arange(160)
        y = np.arange(301)[:, None]
        offset = 0.004 * (x - 80) ** 2 if curved else np.zeros_like(x)
        tau = y + offset
        seismic = sum(np.exp(-0.5 * ((tau - level) / 2) ** 2) for level in [60, 120, 180, 240])
        ax.imshow(seismic, cmap="gray", aspect="auto")
        for level in [60, 120, 180, 240]:
            row = level - offset
            for start, span in [(12, 4), (40, 16), (70, 64)]:
                xx = x[start : start + span + 1]
                yy = row[start : start + span + 1]
                ax.plot(xx, yy, linewidth=2)
                ax.scatter(xx[[0, -1]], yy[[0, -1]], s=15)
        ax.set_title(
            ("Analytic curved" if curved else "Analytic flat") + " equal-RGT paths; QC only"
        )
        ax.set(xlabel="Trace index", ylabel="Time sample")
    curved_case = cases[2]
    reservoir_case = max(cases, key=lambda c: int(c["arrays"]["reservoir_mask"].sum()))
    for ax, case, reservoir in [
        (axes[1, 0], curved_case, False),
        (axes[1, 1], reservoir_case, True),
    ]:
        avo = case["arrays"]["avo"][0]
        limit = np.quantile(abs(avo), 0.98)
        ax.imshow(avo, cmap="gray", vmin=-limit, vmax=limit, aspect="auto")
        g = case["graph"]
        indices = [
            i for i, e in enumerate(g["edges"]) if e["kept"] and e["relation"] == "tangential"
        ][::3]
        ax.add_collection(
            LineCollection(
                [g["paths"][i][:, ::-1] for i in indices], colors="tab:cyan", linewidths=0.6
            )
        )
        ax.scatter(g["points"][:, 1], g["points"][:, 0], s=4, color="tab:orange")
        if reservoir:
            mask = case["arrays"]["reservoir_mask"].astype(bool)
            yy, xx = np.where(mask)
            ax.contour(mask, levels=[0.5], colors=["lime"], linewidths=1)
            if len(xx):
                ax.set_xlim(max(0, xx.min() - 10), min(avo.shape[1] - 1, xx.max() + 10))
                ax.set_ylim(min(avo.shape[0] - 1, yy.max() + 10), max(0, yy.min() - 10))
        ax.set_title(
            f"{case['rid']}: {'reservoir outline QC only' if reservoir else 'curved interfaces'}"
        )
        ax.set(xlabel="Trace index", ylabel="Time sample")
    import io

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    u.write(OUT / "figures/04_curved_interfaces.png", buffer.getvalue())


def main():
    if OUT.exists():
        raise RuntimeError("Completion output already exists; refusing to overwrite")
    original = json.loads((OLD / "v00332u_skeleton_graph_contract.json").read_text())
    for rel, digest in original["source_sha256"].items():
        if u.sha(u.REPO / rel) != digest:
            raise RuntimeError("Frozen u implementation changed: " + rel)
    ids = original["validation_ids"]
    protected = {str(p): u.sha(p) for p in OLD.rglob("*") if p.is_file()}
    protected.update(original["input_sha256"])
    protected[str(u.DATASET / "normalization.json")] = u.sha(u.DATASET / "normalization.json")
    runtime = print_torch_runtime()
    select_torch_device("cpu", context="topology/AVA QC only; frozen CNN sampling blocked")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = [
        Path(__file__),
        u.REPO / "src/sage_avo/diagnostics/skeleton_completion.py",
        u.REPO / "tests/test_skeleton_completion.py",
    ]
    contract = {
        "scope": "QC completion of existing u; no topology or prior retuning",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_contract_sha256": u.sha(OLD / "v00332u_skeleton_graph_contract.json"),
        "source_sha256": {str(p.relative_to(u.REPO)): u.sha(p) for p in sources},
        "protected_sha256": protected,
        "runtime": runtime,
        "cnn_checkpoint_loaded": False,
        "cnn_sampling": "BLOCKED_BY_SAFETY_REVIEW_PENDING_EXPLICIT_AUTHORIZATION",
        "attention_sign": "both additive and literal-subtractive prior-only diagnostics; no model rule selected",
        "candidate_attention": "all candidates, including removed edges; kept-only normalization separate",
        "degree_budget": "existing maximum ten tangential plus two normal neighbors; no changed pruning",
        "decision": "retain original predeclared topology decision",
        "truth_usage": "fault/facies/reservoir visualization and QC only, never graph inputs",
        "no_training": True,
        "no_flow_graph_or_decoder_inference": True,
    }
    u.json_file("completion_contract.json", contract)
    cases = []
    all_attention = []
    degrees = []
    cnn_info = []
    for rid in ids:
        u.log(f"Completion topology/AVA QC (no CNN): {rid}")
        with np.load(
            u.DATASET / "realizations" / f"realization_{rid:07d}.npz", allow_pickle=False
        ) as f:
            arrays = {
                name: f[name] for name in ["avo", "low", "rgt", "valid_mask", "reservoir_mask"]
            }
        g = build_skeleton(arrays["rgt"], arrays["avo"], arrays["valid_mask"], u.CONFIG)
        prior_nodes = pd.read_csv(OLD / f"node_features_{rid}.csv")
        np.testing.assert_allclose(
            g["points"], prior_nodes[["row", "trace"]].to_numpy(), rtol=0, atol=1e-10
        )
        old_edges = pd.read_csv(OLD / "v00332u_edge_statistics.csv")
        old_edges = old_edges[old_edges.realization_id == rid]
        assert np.array_equal([e["kept"] for e in g["edges"]], old_edges.kept.to_numpy())
        cnn_info.append(
            {
                "realization_id": rid,
                "status": "BLOCKED_PENDING_AUTHORIZATION",
                "cnn_channels_sampled": 0,
            }
        )
        u.csv_file(f"ava_node_features_{rid}.csv", prior_nodes)
        degrees.append(dict(realization_id=rid, **neighborhood_summary(g["points"], g["edges"])))
        candidates = [dict(e, kept=True) for e in g["edges"]]
        rows = attention_quantities(arrays["rgt"], g["points"], candidates, u.CONFIG)
        metadata = {tuple(sorted((e["source"], e["target"]))): e for e in g["edges"]}
        df = pd.DataFrame(rows)
        df["kept"] = [metadata[tuple(sorted((s, t)))]["kept"] for s, t in zip(df.source, df.target)]
        df["rejection_reason"] = [
            metadata[tuple(sorted((s, t)))]["reason"] for s, t in zip(df.source, df.target)
        ]
        df["alpha_additive_all_candidates"] = df.pop("alpha")
        df["alpha_subtractive_all_candidates"] = signed_prior_softmax(
            df.log_prior, df.source, df.relation, sign=-1
        )
        for name, sign in [("alpha_additive_kept", 1), ("alpha_subtractive_kept", -1)]:
            df[name] = np.nan
            keep = df.kept
            df.loc[keep, name] = signed_prior_softmax(
                df.loc[keep, "log_prior"],
                df.loc[keep, "source"],
                df.loc[keep, "relation"],
                sign=sign,
            )
        df["realization_id"] = rid
        all_attention.append(df)
        cases.append({"rid": rid, "arrays": arrays, "graph": g})
    u.csv_file("all_candidate_attention.csv", pd.concat(all_attention, ignore_index=True))
    u.csv_file("degree_and_local_support.csv", degrees)
    u.json_file(
        "frozen_cnn_sampling.json",
        {
            "runs": cnn_info,
            "checkpoint_loaded": False,
            "status": "BLOCKED_BY_SAFETY_REVIEW_PENDING_EXPLICIT_AUTHORIZATION",
            "gradients_created": False,
            "cuda_inference_performed": False,
        },
    )
    file_mapping = {
        "v00332u_node_statistics.csv": "node_statistics.csv",
        "v00332u_edge_statistics.csv": "edge_statistics.csv",
        "v00332u_current_vs_skeleton.csv": "current_vs_skeleton.csv",
        "v00332u_edge_length_histogram.csv": "edge_length_histogram.csv",
    }
    for source, destination in file_mapping.items():
        u.write(OUT / destination, (OLD / source).read_bytes())
    qc = pd.read_csv(OLD / "v00332u_fault_qc.csv")
    qc["fault_crossing_recall"] = np.where(qc.region == "fault_crossing", 1 - qc.retention, np.nan)
    qc["false_edge_removal_fraction"] = np.where(
        qc.region.isin(["safe_away_from_fault", "high_dip_continuous"]), 1 - qc.retention, np.nan
    )
    u.csv_file("fault_qc.csv", qc)
    fig_mapping = {
        "01_sparse_nodes.png": "01_nodes.png",
        "02_variable_length_edges.png": "02_variable_length_edges.png",
        "03_high_dip_graph.png": "03_high_dip.png",
        "04_fault_graph.png": "05_faults.png",
        "06_rgt_attention_prior.png": "06_rgt_attention_prior.png",
        "05_current_vs_skeleton.png": "07_current_vs_skeleton.png",
        "07_tangential_vs_normal.png": "08_relations_separate.png",
    }
    for source, destination in fig_mapping.items():
        u.write(OUT / "figures" / destination, (OLD / "figures" / source).read_bytes())
    extra_figure(cases)
    all_df = pd.concat(all_attention, ignore_index=True)
    tang = all_df[(all_df.relation == "tangential") & all_df.kept].copy()
    tang["span"] = abs(tang.delta_trace)
    span = tang.groupby("span").prior.agg(["count", "median"]).reset_index()
    u.csv_file("curved_long_prior_by_span.csv", span)
    for path, digest in protected.items():
        if u.sha(path) != digest:
            raise RuntimeError("Protected input or original output changed: " + path)
    for rel, digest in original["protected_production_sha256"].items():
        if u.sha(u.REPO / rel) != digest:
            raise RuntimeError("Production source changed: " + rel)
    summary = json.loads((OLD / "v00332u_summary.json").read_text())
    summary.update(
        {
            "package": "requested_qc_completion",
            "topology_recomputed_identical": True,
            "original_u_preserved": True,
            "frozen_cnn_descriptors_sampled": False,
            "cnn_status": "BLOCKED_BY_SAFETY_REVIEW_PENDING_EXPLICIT_AUTHORIZATION",
            "all_candidate_attention_computed": True,
            "model_or_checkpoint_loaded": False,
            "attention_sign_selected_for_new_model": None,
            "degree_and_local_support": degrees,
            "cnn_sampling": cnn_info,
            "training_performed": False,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    original_report = (OLD / "v00332u_report.md").read_text()
    report = f"""# v00332u — requested sparse geological skeleton QC completion

Decision: **{summary["decision"]}**.

This package completes the topology/AVA portion of the repeated/revised u checklist while preserving the
original experiment and its failed decision. It is NOT a silently redesigned
or retuned graph. Original node coordinates and edge keep decisions were
recomputed and verified identical on all six sections. No training was run.

## What is newly completed

- Exact requested report, CSV and figure filenames, in this separate directory.
- AVA features are sampled at every anchor. The frozen CNN sampling helper is
  implemented and unit-tested but was NOT run on a scientific checkpoint.
  Safety review rejected that step under the topology-only restrictions; explicit
  user authorization was requested. No checkpoint was loaded and no CUDA model
  inference was performed. This is an incomplete requirement, not substituted
  random CNN features. The rest of the topology/QC work proceeds independently.
- Prior and geometry quantities for EVERY candidate, including removed edges;
  candidate versus retained normalization is labeled separately.
- Additive versus literal-subtractive prior-only alpha for sign QC. Content
  score and geometric bias remain zero, lambda=1. A negative log prior added
  to logits suppresses inconsistent edges; subtracting it rewards them.
  Neither sign comparison changes connectivity or constitutes a new model.
- Degree min/median/p95/max, links outside 3x3 and encoder-only 9x9 support,
  explicit fault-crossing recall and steep-continuous false-removal columns.
- Flat/curved analytic geometry controls and actual curved/reservoir views in
  figure04; separate normal/tangential families remain in figure08.

## Answers to the seven questions

1. Truly sparse: yes, 608–640 versus 48,160 nodes (75–79x compression).
2. Longer explicit graph communication: yes, 18–38x mean two-hop reach. This
   is not a claim that a complete CNN with normalization has no global dependence.
3. Geometrically equal-RGT: yes by fractional inverse construction; some anchors
   remain weak reflectors and the diagnostic cannot guarantee fault identity.
4. Faults respected sufficiently: no. Recall is 57.6%, but retained tangential
   crossing rates are 19.7% and 31.2% in the fault-rich sections. Steep continuous
   false removal is about 0.21% conditional on valid inverse candidate endpoints.
5. RGT controls connectivity: yes, explicit sparse surfaces and variable lengths;
   candidates include spans 4/8/16/32/64 traces (equivalent logarithmic spacing
   because node spacing is 4), not dense adjacent-trace message passing.
6. Prior meaningful for long curves: no. It remains finite but the endpoint
   gradient-dot-chord term collapses many valid long curved links. Normal edges
   must not inherit the same same-interface message/prior rule.
7. Matched training ablation ready: no. The prior, weak-interface anchoring and
   discontinuity safeguards need reviewed design changes first, not more training.

The skeleton has a fixed construction upper bound of ten tangential neighbors
(five spans in each direction) plus two cross-interface neighbors. This is a
bounded logarithmic-range design, NOT a demonstrated graph-theoretic small-world
property (no clustering/path-length comparison to random graphs is claimed).
Degree/local-support values are in degree_and_local_support.csv. The 9x9 check
measures only direct four-convolution encoder support; it excludes GroupNorm,
decoder and iterative-flow effects. It cannot establish information that a CNN
is mathematically incapable of representing.

## Preserved original topology evidence

The following embedded report records the ORIGINAL run's scope, including its
no-checkpoint-load statement. That statement applies to the original prototype;
this completion also did not load a checkpoint. CNN descriptors remain pending.

{original_report}
"""
    u.write(OUT / "v00332u_skeleton_graph_report.md", report.encode())
    u.json_file("v00332u_summary.json", summary)
    u.log(f"QC completion finished: {OUT}; {summary['decision']}")


if __name__ == "__main__":
    main()
