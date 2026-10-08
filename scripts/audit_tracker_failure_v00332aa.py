#!/usr/bin/env python3
"""Audit frozen v00332z CSVs; regenerate only missing detector-cap counts.

All output is private. This script does not train or change production science.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert

from sage_avo.diagnostics.gap_tolerant_graph import detect_events
from sage_avo.diagnostics.rgt_topology_repair import load_faults
from sage_avo.diagnostics.skeleton_graph import path_fault_qc
from sage_avo.diagnostics.tracker_failure_audit import classify_events, masked_eligible_endpoints


REPO = Path(__file__).resolve().parents[1]
BRANCH = "experiment/v00332aa-tracker-failure-audit"
PARENT = "30e0b012467b3553b6aef44089175dad24a2019a"
Z_NAME = "sage_avo_s01_v00332z_gap_tolerant_rgt_tracking"
PROTECTED = (
    "scripts/audit_tracker_failure_v00332aa.py",
    "src/sage_avo/diagnostics/tracker_failure_audit.py",
    "tests/test_tracker_failure_audit.py",
    "configs/development_diagnostics_v00332z.yaml",
    "src/sage_avo/diagnostics/gap_tolerant_graph.py",
    "src/sage_avo/diagnostics/native_event_graph.py",
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
    dirty = git("status", "--porcelain", "--untracked-files=no")
    parent = git("rev-parse", "HEAD^")
    if (head, branch, remote, parent) != (expected, BRANCH, expected, PARENT) or dirty:
        raise RuntimeError("Experiment provenance/clean-worktree gate failed")
    return {
        "repository": str(REPO),
        "branch": BRANCH,
        "commit_sha": head,
        "parent_commit_sha": parent,
        "protected_source_config_test_sha256": {name: sha(REPO / name) for name in PROTECTED},
        "tracked_worktree_clean_at_start": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def strong_gate_counts(avo: np.ndarray, valid: np.ndarray, config: dict[str, Any]) -> dict[str, int]:
    """Count pre-refinement peaks at the existing global envelope gate.

    These are candidate peaks, not true-reflector recall or final strong events.
    """
    selected = np.asarray(avo, float)
    scales = np.quantile(np.abs(selected[:, valid]), 0.75, axis=1)
    normalized = selected / np.maximum(scales[:, None, None], 1e-8)
    envelope = np.sqrt(np.mean(np.abs(hilbert(normalized, axis=1)) ** 2, axis=0))
    evidence = gaussian_filter1d(envelope, sigma=0.75, axis=0, mode="nearest")
    threshold = float(np.quantile(evidence[valid], config["event_quantile"]))
    prominence = config["event_prominence_fraction"] * float(np.quantile(evidence[valid], 0.95))
    before = after = after_cap = 0
    for trace in range(evidence.shape[1]):
        peaks, _ = find_peaks(
            np.where(valid[:, trace], evidence[:, trace], 0.0),
            prominence=prominence,
            distance=int(config["minimum_distance"]),
        )
        before += len(peaks)
        filtered = peaks[evidence[peaks, trace] >= threshold]
        after += len(filtered)
        after_cap += min(len(filtered), int(config["maximum_events_per_trace"]))
    return {
        "strong_pre_global_height_peak_count": before,
        "strong_post_global_height_peak_count": after,
        "strong_post_hard_cap_peak_count": after_cap,
    }


def detector_counts(
    rid: int, dataset: Path, frozen: dict[str, Any], config: dict[str, Any]
) -> dict[str, int]:
    """Regenerate only the unrecorded weak pre-cap and strong-gate counts."""
    with np.load(dataset / "realizations" / f"realization_{rid:07d}.npz", allow_pickle=False) as archive:
        avo = np.asarray(archive["avo"])
        rgt = np.asarray(archive["rgt"])
        valid = np.asarray(archive["valid_mask"], bool)
    uncapped = {**config, "weak_maximum_per_trace": 1_000_000}
    events, counts = detect_events(
        avo, rgt, valid, config["strong_detector"], frozen["weak_detector"], uncapped
    )
    return {
        "strong_count_recomputed": counts["strong"],
        "weak_pre_top_four_count": counts["weak"],
        "weak_strong_overlap_suppressed": counts["overlap"],
        "uncapped_detected_event_count": len(events),
        **strong_gate_counts(avo, valid, config["strong_detector"]),
    }


def grouped(path: Path) -> dict[int, list[dict[str, Any]]]:
    frame = pd.read_csv(path)
    result = {}
    for rid, rows in frame.groupby("realization_id", sort=False):
        result[int(rid)] = rows.drop(columns="realization_id").to_dict("records")
    return result


def fault_rows(
    rid: int,
    events: list[dict[str, Any]],
    links: list[dict[str, Any]],
    accepted: dict[str, Any],
    faults: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    selected = {(row["source"], row["target"]) for row in accepted["accepted_links"]}
    for row in links:
        physical = np.asarray(
            [
                [events[row["source"]]["time"], events[row["source"]]["trace"]],
                [events[row["target"]]["time"], events[row["target"]]["trace"]],
            ],
            float,
        )
        crossing, _ = path_fault_qc(physical, faults)
        if not crossing:
            continue
        pair = (int(row["source"]), int(row["target"]))
        selected_link = pair in selected
        exclusive = bool({*pair} & accepted["used"])
        blocked = id(row) in accepted["blocked"]
        if not row["plausible"]:
            reason = "cost_threshold"
        elif row["barrier"]:
            reason = "direct_barrier"
        elif selected_link:
            reason = "retained_adjacent_track"
        elif exclusive:
            reason = "exclusive_nodes_after_greedy_selection"
        elif blocked:
            reason = "noncrossing_constraint"
        else:
            reason = "path_eligibility_or_objective"
        rows.append(
            {
                "realization_id": rid,
                "source": pair[0],
                "target": pair[1],
                "span": int(row["span"]),
                "plausible": bool(row["plausible"]),
                "barrier": bool(row["barrier"]),
                "safe": bool(row["safe"]),
                "accepted": selected_link,
                "exclusive_overlap": exclusive,
                "noncrossing_overlap": blocked,
                "first_rejection_reason": reason,
            }
        )
    return rows


def figure(path: Path, events: list[dict[str, Any]], reasons: list[dict[str, Any]], rid: int) -> None:
    palette = {
        "accepted": "#087e8b",
        "no_association_candidate": "#9e9e9e",
        "cost_threshold_rejection": "#d5a021",
        "fault_or_discontinuity_barrier": "#b23a48",
        "inadequate_event_count": "#724cf9",
        "inadequate_path_length": "#724cf9",
        "joint_count_span_ineligibility": "#724cf9",
        "exclusive_node_conflict_or_greedy_selection": "#e07a31",
        "noncrossing_constraint": "#e07a31",
        "objective_or_selection_unresolved": "#e07a31",
    }
    fig, axis = plt.subplots(figsize=(12, 5))
    reason_by_event = {row["event"]: row["first_decisive_reason"] for row in reasons}
    for name, color in palette.items():
        subset = [event for index, event in enumerate(events) if reason_by_event[index] == name]
        if subset:
            axis.scatter(
                [event["trace"] for event in subset],
                [event["time"] for event in subset],
                s=4 if name != "accepted" else 7,
                alpha=0.65,
                c=color,
                label=f"{name} ({len(subset)})",
            )
    axis.invert_yaxis()
    axis.set(xlabel="Trace", ylabel="Time sample", title=f"v00332aa frozen-event survival: {rid}")
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    identity = provenance(args.expected_commit)
    private = args.private_root.resolve()
    if not private.is_dir() or private.is_relative_to(REPO):
        raise RuntimeError("Require an existing private artifact root outside the Git worktree")
    base = private / "stage_artifacts"
    z = base / "stage04" / Z_NAME
    final_output = private / "scientific_reconciliation" / "v00332aa_tracker_failure_audit"
    output = final_output.with_name(final_output.name + ".incomplete")
    if output.exists() or final_output.exists():
        raise RuntimeError(f"Refusing to overwrite existing private audit: {final_output}")
    required = [z / f"{name}.csv" for name in ("events", "candidate_links", "components", "nodes", "edges", "validation_qc")]
    required += [z / "v00332z_frozen_contract.json", z / "v00332z_experiment_contract.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    original_contract = json.loads((z / "v00332z_experiment_contract.json").read_text())
    if original_contract["commit_sha"] != PARENT:
        raise RuntimeError("Frozen v00332z artifact parent does not match reviewed commit")
    frozen = json.loads((z / "v00332z_frozen_contract.json").read_text())
    config = original_contract["config"]
    events_by_case = grouped(z / "events.csv")
    links_by_case = grouped(z / "candidate_links.csv")
    nodes_by_case = grouped(z / "nodes.csv")
    components_by_case = grouped(z / "components.csv")
    edges_by_case = grouped(z / "edges.csv")
    qc = pd.read_csv(z / "validation_qc.csv").set_index("realization_id")
    ids = [int(value) for value in original_contract["validation_ids"]]
    if set(ids) != set(events_by_case) or set(ids) != set(links_by_case):
        raise RuntimeError("Frozen realization IDs disagree across input tables")
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    contract = {
        **identity,
        "revision": "v00332aa",
        "input_artifact_sha256": {path.name: sha(path) for path in required},
        "frozen_calibration_sha256": sha(z / "v00332z_frozen_contract.json"),
        "validation_ids": ids,
        "scientific_status": "exploratory development QC on previously used validation cases",
        "new_regeneration": "uncapped weak-detector and pre-global strong-gate counts only",
        "training": False,
    }
    (output / "v00332aa_experiment_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    survival, rejected, fault, coverage = [], [], [], []
    masked_rows: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    all_reason_rows: list[dict[str, Any]] = []
    q = json.loads(
        (base / "stage04/sage_avo_s01_v00332q_clean_20epoch_corrected_rgt/v00332q_contract.json").read_text()
    )
    dip_cut = q["adaptive_search"]["dip_q67"]
    curve_cut = q["adaptive_search"]["curvature_q75"]
    for rid in ids:
        print(f"[v00332aa] auditing frozen case {rid}", flush=True)
        events = sorted(events_by_case[rid], key=lambda row: int(row["event"]))
        links = links_by_case[rid]
        nodes = nodes_by_case[rid]
        reasons, replayed = classify_events(events, links, nodes, config)
        masked = masked_eligible_endpoints(events, links, config)
        masked_rows.extend({"realization_id": rid, **row} for row in masked)
        for name, reward, gap_delta in (
            ("frozen", 8.0, 0.0),
            ("reward_6", 6.0, 0.0),
            ("reward_10", 10.0, 0.0),
            ("gap_penalty_0", 8.0, -1.0),
            ("gap_penalty_2", 8.0, 1.0),
        ):
            varied_links = (
                links if gap_delta == 0
                else [dict(row, cost=float(row["cost"]) + gap_delta * (row["span"] - 1)) for row in links]
            )
            varied_config = {**config, "path_step_reward": reward}
            sensitivity.append(
                {
                    "realization_id": rid,
                    "objective_only_variant": name,
                    "masked_eligible_endpoint_count": len(
                        masked if name == "frozen"
                        else masked_eligible_endpoints(events, varied_links, varied_config)
                    ),
                }
            )
        counts = Counter(row["first_decisive_reason"] for row in reasons)
        accepted = sum(row["accepted_track"] for row in reasons)
        if accepted != round(float(qc.loc[rid, "assigned_event_fraction"]) * len(events)):
            raise AssertionError(f"Frozen accepted-event count mismatch in {rid}")
        if len(replayed["paths"]) != int(qc.loc[rid, "component_count"]):
            raise AssertionError(f"Frozen component count mismatch in {rid}")
        if len(nodes) != len({node["event"] for node in nodes}):
            raise AssertionError(f"Duplicate sparse event in {rid}")
        detector = detector_counts(
            rid,
            base / "stage03/ds_v00331_production100_support_aware/dataset",
            frozen,
            config,
        )
        if detector["strong_count_recomputed"] != int(qc.loc[rid, "strong_event_count"]):
            raise AssertionError(f"Strong-event regeneration mismatch in {rid}")
        if detector["weak_pre_top_four_count"] < int(qc.loc[rid, "weak_event_count"]):
            raise AssertionError(f"Uncapped weak count below frozen weak count in {rid}")
        if detector["weak_strong_overlap_suppressed"] != int(qc.loc[rid, "strong_weak_overlap"]):
            raise AssertionError(f"Weak-overlap regeneration mismatch in {rid}")
        fault_truth = load_faults(
            base / "stage02/v00331_production100_support_aware/realizations", rid
        )
        faults_for_case = fault_rows(rid, events, links, replayed, fault_truth)
        fault.extend(faults_for_case)
        for row in reasons:
            rejected.append({"realization_id": rid, **row})
            all_reason_rows.append(row)
        survival.append(
            {
                "realization_id": rid,
                "detected_events": len(events),
                "candidate_events": sum(row["has_candidate"] for row in reasons),
                "plausible_events": sum(row["has_plausible"] for row in reasons),
                "barrier_safe_events": sum(row["has_safe"] for row in reasons),
                "candidate_track_events": sum(row["candidate_track"] for row in reasons),
                "accepted_track_events": accepted,
                "sparse_node_events": sum(row["sparse_node"] for row in reasons),
                "accepted_over_detected": accepted / len(events),
                "first_reason_counts_json": json.dumps(dict(counts), sort_keys=True),
                "structurally_eligible_but_unaccepted": sum(
                    row["structurally_eligible"] and not row["accepted_track"] for row in reasons
                ),
                "initial_best_path_members_but_unaccepted": sum(
                    row["on_initial_best_eligible_path"] and not row["accepted_track"]
                    for row in reasons
                ),
                "masked_eligible_endpoint_count": len(masked),
                "weak_post_top_four_count": int(qc.loc[rid, "weak_event_count"]),
                "weak_removed_by_top_four": detector["weak_pre_top_four_count"]
                - int(qc.loc[rid, "weak_event_count"]),
                **detector,
            }
        )
        accepted_events = {row["event"] for row in reasons if row["accepted_track"]}
        traces = {int(events[index]["trace"]) for index in accepted_events}
        near_fault = {
            index
            for index, event in enumerate(events)
            if any(
                abs(event["trace"] - item["column"] - item["dip"] * event["time"]) <= 4
                for item in fault_truth
            )
        }
        high = {index for index, event in enumerate(events) if event["structural_dip"] >= dip_cut}
        curved = {
            index for index, event in enumerate(events)
            if event["structural_curvature"] >= curve_cut
        }
        edge_rows = edges_by_case[rid]
        long_edges = [row for row in edge_rows if int(row["span"]) >= 16]
        coverage.append(
            {
                "realization_id": rid,
                "detected_event_count": len(events),
                "accepted_event_count": accepted,
                "accepted_event_fraction": accepted / len(events),
                "trace_coverage_fraction": float(qc.loc[rid, "trace_coverage_fraction"]),
                "traces_without_accepted_event": 160 - len(traces),
                "near_fault_detected": len(near_fault),
                "near_fault_accepted": len(near_fault & accepted_events),
                "high_dip_detected": len(high),
                "high_dip_accepted": len(high & accepted_events),
                "curved_detected": len(curved),
                "curved_accepted": len(curved & accepted_events),
                "component_count": len(components_by_case[rid]),
                "median_component_span": float(qc.loc[rid, "median_component_span"]),
                "long_sparse_edge_count": len(long_edges),
                "long_sparse_edge_mean_span": float(np.mean([row["span"] for row in long_edges]))
                if long_edges else 0.0,
                "sparse_two_hop_reach_mean": float(qc.loc[rid, "two_hop_reach_mean"]),
                "undetected_true_reflector_fraction": "UNKNOWN_NO_INDEPENDENT_REFLECTOR_TRUTH",
            }
        )
        if rid in (3400078, 3400075):
            figure(output / "figures" / f"event_survival_{rid}.png", events, reasons, rid)
    write_csv(output / "v00332aa_event_survival.csv", survival)
    write_csv(output / "v00332aa_tracker_rejection_reasons.csv", rejected)
    write_csv(output / "v00332aa_fault_rejection_attribution.csv", fault)
    write_csv(output / "v00332aa_component_coverage.csv", coverage)
    write_csv(output / "v00332aa_masked_eligible_endpoints.csv", masked_rows)
    write_csv(output / "v00332aa_objective_sensitivity.csv", sensitivity)
    reason_counts = Counter(row["first_decisive_reason"] for row in all_reason_rows)
    detected = len(all_reason_rows)
    accepted = reason_counts["accepted"]
    fault_plausible = [row for row in fault if row["plausible"]]
    fault_summary = Counter(row["first_rejection_reason"] for row in fault_plausible)
    root_causes = sorted(
        [(name, count) for name, count in reason_counts.items() if name != "accepted"],
        key=lambda pair: -pair[1],
    )
    # Selection categories are observational. Only pre-tracker gates and exact
    # structural ineligibility are direct exclusions; do not overclaim causality.
    confirmed_groups = [
        (name, count) for name, count in root_causes
        if name in {
            "no_association_candidate", "cost_threshold_rejection",
            "fault_or_discontinuity_barrier", "inadequate_event_count",
            "inadequate_path_length", "joint_count_span_ineligibility",
        }
        and count / detected >= 0.05
    ]
    decision = (
        "MULTIPLE_CAUSES_ESTABLISHED" if len(confirmed_groups) >= 2
        else "EVENT_CANDIDATE_COVERAGE_DOMINANT" if confirmed_groups and confirmed_groups[0][0]
        == "no_association_candidate"
        else "ROOT_CAUSE_NOT_ESTABLISHED"
    )
    summary = {
        "decision": decision,
        "provenance": contract,
        "detected_events": detected,
        "accepted_events": accepted,
        "accepted_fraction_pooled": accepted / detected,
        "excluded_events": detected - accepted,
        "first_decisive_reasons": dict(reason_counts),
        "ranked_exclusion_reasons": root_causes,
        "fault_plausible_crossing_reasons": dict(fault_summary),
        "known_synthetic_one_predecessor_defect": True,
        "frozen_masked_eligible_endpoint_count": len(masked_rows),
        "frozen_case_event_coverage_impact_of_one_predecessor_defect": "NOT_CAUSALLY_IDENTIFIED",
        "geological_true_reflector_recall": "UNKNOWN_NO_INDEPENDENT_REFLECTOR_TRUTH",
        "validation_status": "EXPLORATORY_DEVELOPMENT_QC",
        "no_training": True,
    }
    (output / "v00332aa_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    frozen_examples = "\n".join(
        f"- Case {row['realization_id']}, endpoint {row['endpoint_event']}: "
        f"best path {row['best_path_event_count']} events/{row['best_path_span']} traces, "
        f"score {row['best_score']:.3f}; eligible alternative score "
        f"{row['eligible_alternative_score']:.3f}."
        for row in masked_rows[:10]
    )
    (output / "v00332aa_dp_counterexamples.md").write_text(
        "# One-predecessor DP counterexample\n\n"
        "Two safe paths share trace-12 endpoint. A seven-event path at traces "
        "0,2,4,6,8,10,12 has six cost-1 links; an eight-event path at "
        "0,1,3,5,7,9,11,12 has seven. Both span 12 traces. With reward 8 per "
        "trace-span, the seven-event path scores 90 and the eligible eight-event "
        "path scores 89. The production DP stores only the former predecessor "
        "chain at the endpoint, rejects it under the eight-event gate, and "
        "returns no component. See the strict-xfail regression in "
        "tests/test_tracker_failure_audit.py. This proves an algorithmic defect "
        "on a deterministic graph.\n\n"
        f"Frozen one-predecessor masked eligible endpoints: {len(masked_rows)}. "
        "These endpoint counts are not independent recovered events and do not "
        "establish a coverage gain under competing exclusive/noncrossing tracks.\n\n"
        f"{frozen_examples if frozen_examples else 'No frozen masked endpoint found.'}\n\n"
        "Objective-only reward/gap sensitivity on fixed frozen safe links is in "
        "v00332aa_objective_sensitivity.csv; it is not parameter tuning.\n"
    )
    report = [
        "# v00332aa frozen reflector-coverage root-cause audit",
        "",
        f"Decision: `{decision}`",
        "",
        "Exploratory development QC on previously used validation realizations; no training, threshold tuning, or production-model changes.",
        "",
        f"Detected {detected} events; accepted {accepted} ({accepted / detected:.1%}); excluded {detected - accepted}.",
        "",
        "## First decisive exclusions (mutually exclusive)",
        "",
    ]
    report += [f"- {name}: {count} ({count / detected:.1%} of detected)" for name, count in root_causes]
    report += [
        "", "## Interpretation", "",
        "Candidate/plausible/safe are link-incidence stages. `candidate_track` means positive-best-path participation in the initial DP scan, not a geological truth label.",
        "Final selection categories are observational counterfactuals; exclusive-node and greedy order are not cleanly separable without a different tracker.",
        f"The deterministic one-predecessor counterexample confirms a tracker defect. {len(masked_rows)} frozen endpoints also mask positive eligible alternatives, but endpoint counts do not equal recoverable events.",
        "Weak pre-cap counts were the only missing detector data regenerated; the complete six-case link/component pipeline was not rerun.",
        "Global-envelope pre-height peaks are an upper-bound candidate measure, not true-reflector recall. The fraction of real reflectors never detected is unknown without independent labels.",
        "Fault truth was used only after frozen calibration for QC. Direct barrier rejections and final graph safety are separate effects.",
        "v00332y comparisons in the previous report have non-equivalent event denominators; no cross-version event-recall claim is made here.",
        "Sparse long edges may be useful as a supplementary branch, but no neural integration or benefit was tested. Untouched realizations are needed after freezing any correction.",
        "", "## Fault crossing attribution", "",
    ]
    report += [f"- {name}: {count}" for name, count in sorted(fault_summary.items())]
    report += ["", "## Provenance", "", f"Branch `{BRANCH}` at `{identity['commit_sha']}`; private outputs only.", ""]
    (output / "v00332aa_report.md").write_text("\n".join(report))
    output.rename(final_output)
    print(
        json.dumps(
            {"output": str(final_output), "decision": decision, "accepted_fraction": accepted / detected},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
