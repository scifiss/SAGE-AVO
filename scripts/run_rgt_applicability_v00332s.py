#!/usr/bin/env python3
"""Frozen-output v00332s stop/go screen. No training or model redesign in this runner.

Phase-2 failure is a terminal scientific result. A passing screen authorizes the
separate, still-to-be-implemented causal topology comparison, not an automatic
change to frozen q/r outputs. All writes stay in a new private s directory.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sage_avo_matplotlib")

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_attention_seg_coupling_v00332r as frozen
from sage_avo.config import load_config
from sage_avo.diagnostics.rgt_applicability import (
    applicability_fields, applicability_signal, heldout_binned_prediction, score_bins,
)
from sage_avo.diagnostics.rgt_topology_repair import load_faults
from sage_avo.evaluation.inference import blend_window, tile_starts

REPO = frozen.REPOSITORY
CONFIG_PATH = REPO / "configs/development_diagnostics_v00332s.yaml"
CONFIG = load_config(CONFIG_PATH)
R = frozen.OUT
OUT = frozen.STAGE04 / CONFIG["experiment_name"]
PROPS = frozen.PROPERTIES
A = "RGT_TRANSFORMER_SHARED"
D = "CARTESIAN_TRANSFORMER_SHARED"


def read(path):
    return json.loads(Path(path).read_text())


def log(message):
    print(f"[v00332s {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@contextmanager
def atomic(path):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("Only the new private v00332s directory is writable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    with temporary.open("wb") as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, payload):
    with atomic(path) as stream:
        stream.write((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def write_csv(path, rows):
    with atomic(path) as stream:
        stream.write(pd.DataFrame(rows).to_csv(index=False).encode())


def write_text(path, text):
    with atomic(path) as stream:
        stream.write(text.encode())


def save_figure(fig, name):
    fig.tight_layout()
    with atomic(OUT / "figures" / name) as stream:
        fig.savefig(stream, format="png", dpi=140)
    plt.close(fig)


def contract():
    return read(OUT / "v00332s_applicability_contract.json")


def prediction_path(seed, epoch, condition, rid):
    return R / "predictions" / f"seed_{seed}_{condition}_epoch_{epoch:04d}_{rid}.npz"


def source_paths():
    return [Path(__file__), CONFIG_PATH, REPO / "src/sage_avo/diagnostics/rgt_applicability.py"]


def verify():
    saved = contract()
    for path, sha in saved["protected_inputs_sha256"].items():
        if frozen.file_sha(path) != sha:
            raise RuntimeError(f"Frozen input changed: {path}")
    for path, sha in saved["screen_sources_sha256"].items():
        if frozen.file_sha(REPO / path) != sha:
            raise RuntimeError(f"Predeclared screen changed: {path}")
    return saved


def prepare():
    if (OUT / "v00332s_applicability_contract.json").exists():
        verify()
        return
    expected = CONFIG["review_commit"]
    for ref in ("HEAD", "review/rgt-gnn-reconciliation"):
        actual = subprocess.check_output(["git", "rev-parse", ref], cwd=REPO, text=True).strip()
        if actual != expected:
            raise RuntimeError(f"Unexpected frozen reference: {ref}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if branch != "experiment/v00332s-adaptive-rgt-applicability":
        raise RuntimeError("Expected the new local s branch")
    if read(R / "runner_status.json")["status"] != "command_complete":
        raise RuntimeError("r is not complete; do not overlap or overwrite experiments")
    q = frozen.q_contract()
    r_contract = read(R / "v00332r_baseline_contract.json")
    if q["confidence_rule"] != r_contract["confidence_rule"]:
        raise RuntimeError("Frozen q/r confidence thresholds differ")
    model = frozen.training_config(CONFIG["seeds"][0], A)["model"]
    if model["max_rgt_shift_samples"] != CONFIG["score"]["max_shift"]:
        raise RuntimeError("Unexpected RGT search radius")
    experiment = model["experimental_graph"]
    if (experiment["rgt_topology"] != "rgt_v3_confidence_blocked"
            or experiment.get("attention_mode", "learned") != "learned"
            or experiment.get("segmentation_detach_graph", False)):
        raise RuntimeError("The frozen r baseline is not learned/shared confidence-blocked RGT")
    protected = {}
    # r's work was uncommitted. Preserve actual source bytes, not only Git HEAD.
    for relative, expected_sha in read(R / "matched_training_source_contract.json")["source_sha256"].items():
        path = REPO / relative
        if frozen.file_sha(path) != expected_sha:
            raise RuntimeError(f"r source changed before freezing: {relative}")
        protected[str(path)] = expected_sha
        with atomic(OUT / "frozen_r_source_snapshot" / relative) as stream:
            stream.write(path.read_bytes())
    protected_paths = [R / name for name in (
        "v00332r_baseline_contract.json", "v00332r_summary.json", "v00332r_report.md",
        "fixed_patch_schedule.json", "v00332r_multiseed_results.csv", "validation.json",
    )]
    protected_paths += [frozen.Q / "v00332q_contract.json", frozen.DATASET / "normalization.json"]
    for seed in CONFIG["seeds"]:
        for epoch in CONFIG["milestones"]:
            for condition in (A, D):
                protected_paths.extend(prediction_path(seed, epoch, condition, rid)
                                       for rid in q["diverse_validation_subset"]["all_ids"])
    protected.update({str(p): frozen.file_sha(p) for p in protected_paths})
    write_text(OUT / "frozen_r_worktree.patch", subprocess.check_output(
        ["git", "diff", "--", "src/sage_avo/models/sage_avo.py", "src/sage_avo/models/variants.py"],
        cwd=REPO, text=True,
    ))
    payload = {
        "status": "PREDECLARED_BEFORE_VALIDATION_OUTCOME_BINNING",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "review_commit": expected, "branch": branch, "config": CONFIG,
        "protected_inputs_sha256": protected,
        "screen_sources_sha256": {str(p.relative_to(REPO)): frozen.file_sha(p) for p in source_paths()},
        "confidence_thresholds": q["confidence_rule"]["thresholds"],
        "split_ids": q["split_ids"], "validation_screen_ids": q["diverse_validation_subset"]["all_ids"],
        "training_bins_file": "training_only_bins.json",
        "topology_implementation_changed": False, "adaptive_training_authorized": False,
        "applicability_threshold": None,
        "threshold_status": "Deferred until phase-2 signal; 0.5 sample is proposed a priori, not validation-tuned",
        "model_or_checkpoint_loaded": False,
    }
    write_json(OUT / "v00332s_applicability_contract.json", payload)
    write_text(OUT / "v00332s_applicability_definition.md", DEFINITION)
    log("Frozen r source snapshot and predeclared phase-2 contract saved; no model changes")


DEFINITION = """# v00332s structural-applicability screen

For every rightward adjacent-trace source, j_C is the same-depth next-trace
sample and j_R is the unchanged radius-3, lexicographic tie-fixed candidate.
S = (|tau_i-tau_jC| - |tau_i-tau_jR|)/(delta_tau_local + epsilon).
delta_tau_local is the median of five absolute forward vertical increments on
the source trace (centered at the source row, nearest padding at boundaries).
epsilon = 16*float32_eps*section median positive vertical increment. Undefined
local scale (delta <= epsilon) is recorded, excluded from the mechanism screen,
and must never be regarded as high applicability. Raw small negative S caused
by the frozen near-tie tolerance is retained, not clamped or redesigned.

Reliability is a DIFFERENT quantity: confidence_safe is taken directly from the
unchanged production Torch confidence-blocked edge builder with frozen p/q/r
thresholds. A high S does not override blocking. Fault truth, facies boundaries,
reservoir/plume labels are QC strata ONLY; the measurement function accepts RGT
and existing confidence thresholds only. Valid support is used for evaluation,
not as a new model feature. Dip and curvature retain existing frozen QC formulas.

Training only: fit two tertile cuts to positive S on eligible safe training
correspondences, alongside a distinct S<=0 bin; fit dip quartiles on the same
support. Save/hash cuts BEFORE opening validation predictions for analysis.
Use all 70 training sections. Degenerate cuts cannot justify an invented gate.
Full-section maps cover all training and validation sections; per-source values
are in measurement_maps/*.npz, with readable strata statistics in CSV.

Primary screen: frozen r A versus D, fixed epoch20, the same six-section screen,
all three seeds. Valid source AND Cartesian/RGT endpoints, confidence-safe and
identifiable scale. Compare high-S bin3 versus nonpositive bin0 on the SAME
realizations, requiring >=50 pixels in each bin and >=4 paired realizations.
Compute per-property normalized RMSE per realization, then equal-realization
mean of the three properties. Require high-S relative elastic gain >0 in ALL
three seeds, high-minus-zero gain >0 in ALL three seeds, and average contrast
>=0.005 (0.5 percentage points). This is an operational observational screen,
not a significance test. No pixels or q/r repeated runs count as independent
replicates. Strict monotonicity is descriptive, not required if contrast is
substantial. Other milestones and fault/away/reliability strata are descriptive.

Compare S with absolute dip using the frozen training bins and leave-one-section-
out prediction of per-pixel mean normalized squared-error improvement. Fit only
the bin response on the other five validation sections, never bin thresholds;
report equal-section MSE against a cross-fitted intercept. This exploratory
comparison is NOT a tuning path and cannot rescue a failed primary screen.

Full-section S describes observed geology; stitched predictions use overlapping
50x100 graphs. Explicitly audit Hann-weighted full-grid/tile correspondence and
confidence agreement. Pixel associations are not proof that one edge caused an
error change: messages and CNN receptive fields couple nearby pixels.

If the primary signal is not established, STOP before adaptive topology,
random-control implementation, new model initialization or training. Unexecuted
downstream comparisons must be N/A, not invented negative/equivalent results.
If it passes, select/freeze a threshold from training statistics only, then
implement and validate the separately authorized matched topology experiment.
"""


def load_arrays(rid, *, outcomes=False):
    names = ["rgt", "valid_mask", "segmentation", "reservoir_mask", "plume_mask"]
    if outcomes:
        names.append("elastic")
    with np.load(frozen.DATASET / "realizations" / f"realization_{rid:07d}.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in names}


def measure_one(rid, saved):
    path = OUT / "measurement_maps" / f"realization_{rid}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    arrays = load_arrays(rid)
    fields = applicability_fields(arrays["rgt"], saved["confidence_thresholds"])
    rows, cols = fields["source_row"], fields["source_column"]
    valid = arrays["valid_mask"].astype(bool)
    fields["valid_source"] = valid[:, :-1]
    fields["eligible"] = valid[:, :-1] & valid[:, 1:] & valid[fields["target_row"], cols + 1] & fields["scale_valid"]
    strata = frozen.frozen_helpers._pixel_strata(arrays, load_faults(frozen.STAGE02, rid), frozen.q_contract()["adaptive_search"])
    fields.update({name: mask[:, :-1] for name, mask in strata.items() if name != "all"})
    labels = arrays["segmentation"]
    fields["facies_crossing"] = labels[rows, cols] != labels[fields["target_row"], cols + 1]
    with atomic(path) as stream:
        np.savez_compressed(stream, **fields)
    return fields


def measure():
    saved = verify()
    bins_path = OUT / "training_only_bins.json"
    if not bins_path.exists():
        positives, dips, all_scores = [], [], []
        n_source = n_eligible = n_safe = n_half = n_blocked = 0
        for rid in saved["split_ids"]["train"]:
            f = measure_one(rid, saved)
            support = f["eligible"] & f["confidence_safe"]
            scores = f["S"][support]
            positives.append(scores[scores > 0])
            dips.append(f["dip"][support])
            all_scores.append(scores)
            n_source += int(f["valid_source"].sum())
            n_eligible += int(f["eligible"].sum())
            n_safe += int(support.sum())
            n_blocked += int((f["valid_source"] & ~f["confidence_safe"]).sum())
            n_half += int((support & (f["S"] >= .5)).sum())
            log(f"Training-only structural measurements: {rid}")
        cuts = np.quantile(np.concatenate(positives), [1/3, 2/3]).tolist()
        dip_cuts = np.quantile(np.concatenate(dips), [.25, .5, .75]).tolist()
        score_bins(np.array([0., 1.]), cuts)  # Fail closed if the proposed bins are degenerate.
        write_json(bins_path, {
            "fit_split": "train only", "validation_predictions_opened": False,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "positive_S_tertile_cuts": cuts, "dip_quartile_cuts": dip_cuts,
            "S_quantiles": dict(zip(["q0", "q25", "q50", "q75", "q95", "q99", "q100"],
                                    np.quantile(np.concatenate(all_scores), [0, .25, .5, .75, .95, .99, 1]).tolist())),
            "valid_source_edges": n_source, "eligible_edges": n_eligible, "confidence_safe_eligible_edges": n_safe,
            "confidence_blocked_valid_source_fraction": n_blocked/n_source,
            "proposed_half_sample_applicable_valid_source_fraction": n_half/n_source,
            "proposed_half_sample_cartesian_valid_source_fraction": 1-n_half/n_source,
            "phase2_contract_sha256": frozen.file_sha(OUT / "v00332s_applicability_contract.json"),
        })
        write_json(OUT / "training_bins_freeze.json", {"sha256": frozen.file_sha(bins_path)})
        log("Training-only bins frozen BEFORE validation prediction differences")
    bins = read(bins_path)
    if frozen.file_sha(bins_path) != read(OUT / "training_bins_freeze.json")["sha256"]:
        raise RuntimeError("Frozen training-only bins changed")
    summary = []
    for split in ("train", "validation"):
        for rid in saved["split_ids"][split]:
            f = measure_one(rid, saved)
            b = score_bins(f["S"], bins["positive_S_tertile_cuts"])
            masks = {"all_valid_source": f["valid_source"], "eligible": f["eligible"],
                     "confidence_safe": f["eligible"] & f["confidence_safe"],
                     "confidence_blocked": f["eligible"] & ~f["confidence_safe"]}
            masks.update({name: f["eligible"] & f[name] for name in (
                "low_dip", "high_dip_continuous", "fault_corridor", "away_from_fault", "facies_boundary", "reservoir", "plume")})
            masks.update({f"S_bin_{i}": f["eligible"] & f["confidence_safe"] & (b == i) for i in range(4)})
            masks.update({f"abs_displacement_{i}": f["eligible"] & (np.abs(f["displacement"]) == i) for i in range(4)})
            for name, mask in masks.items():
                if not mask.any():
                    continue
                finite = mask & np.isfinite(f["S"])
                summary.append({"split": split, "realization_id": rid, "stratum": name, "edge_count": int(mask.sum()),
                                "identifiable_scale_fraction": float(f["scale_valid"][mask].mean()),
                                "S_mean": float(f["S"][finite].mean()) if finite.any() else np.nan,
                                "m_C_mean": float(f["m_C"][mask].mean()), "m_R_mean": float(f["m_R"][mask].mean()),
                                "local_scale_mean": float(f["delta_tau_local"][mask].mean()),
                                "dip_mean": float(f["dip"][mask].mean()), "curvature_mean": float(f["curvature"][mask].mean()),
                                "abs_displacement_mean": float(np.abs(f["displacement"][mask]).mean()),
                                "confidence_blocked_fraction": float((~f["confidence_safe"])[mask].mean()),
                                "fault_qc_fraction": float(f["fault_corridor"][mask].mean()),
                                "facies_crossing_qc_fraction": float(f["facies_crossing"][mask].mean()),
                                "reservoir_qc_fraction": float(f["reservoir"][mask].mean()),
                                "plume_qc_fraction": float(f["plume"][mask].mean()),
                                "proposed_half_sample_fraction": float((f["confidence_safe"][mask] & (f["S"][mask] >= .5)).mean())})
            log(f"Structural/QC summary {split}: {rid}")
    write_csv(OUT / "v00332s_structural_informativeness.csv", summary)
    write_csv(OUT / "v00332s_topology_statistics.csv", summary)
    tile_context_audit(saved)


def tile_context_audit(saved):
    destination = OUT / "full_grid_vs_tiled_context.csv"
    if destination.exists():
        return
    records = []
    for rid in saved["split_ids"]["train"][:6] + saved["validation_screen_ids"]:
        arrays = load_arrays(rid)
        full = measure_one(rid, saved)
        tau = arrays["rgt"]
        sums = np.zeros(4)
        window = blend_window((50, 100))[:, :-1]
        for top in tile_starts(tau.shape[0], 50, 25):
            for left in tile_starts(tau.shape[1], 100, 50):
                local = applicability_fields(tau[top:top+50, left:left+100], saved["confidence_thresholds"])
                region = np.s_[top:top+50, left:left+99]
                weight = window * full["eligible"][region]
                sums += [weight.sum(),
                         (weight * (local["target_row"] + top != full["target_row"][region])).sum(),
                         (weight * (local["confidence_safe"] != full["confidence_safe"][region])).sum(),
                         (weight * ((local["S"] >= .5) != (full["S"][region] >= .5))).sum()]
        records.append({"realization_id": rid, "weighted_support": sums[0],
                        "target_disagreement_fraction": sums[1]/sums[0],
                        "confidence_disagreement_fraction": sums[2]/sums[0],
                        "half_sample_score_disagreement_fraction": sums[3]/sums[0]})
        log(f"Frozen tile-context correspondence audit: {rid}")
    write_csv(destination, records)


def analyze():
    saved = verify()
    bins = read(OUT / "training_only_bins.json")
    if frozen.file_sha(OUT / "training_only_bins.json") != read(OUT / "training_bins_freeze.json")["sha256"]:
        raise RuntimeError("Training-only cuts no longer match their frozen hash")
    y_std = np.asarray(read(frozen.DATASET / "normalization.json")["y_std"])[:, None, None]
    records, cv_records = [], []
    for seed in CONFIG["seeds"]:
        cv_groups = {"S": [], "dip": []}
        for epoch in CONFIG["milestones"]:
            for rid in saved["validation_screen_ids"]:
                f = measure_one(rid, saved)
                arrays = load_arrays(rid, outcomes=True)
                errors = []
                for condition in (A, D):
                    with np.load(prediction_path(seed, epoch, condition, rid), allow_pickle=False) as archive:
                        prediction = archive["prediction"].astype(np.float64)
                    errors.append(((prediction[:, :, :-1] - arrays["elastic"][:, :, :-1])/y_std)**2)
                if not all(np.isfinite(e[:, f["eligible"]]).all() for e in errors):
                    raise RuntimeError("Nonfinite frozen predictions on eligible support")
                choices = {"S": score_bins(f["S"], bins["positive_S_tertile_cuts"]),
                           "dip": np.searchsorted(bins["dip_quartile_cuts"], f["dip"], side="right"),
                           "displacement": np.abs(f["displacement"]),
                           "confidence_blocked": (~f["confidence_safe"]).astype(int)}
                scopes = {"all_eligible": f["eligible"], "confidence_safe": f["eligible"] & f["confidence_safe"],
                          "fault_corridor": f["eligible"] & f["confidence_safe"] & f["fault_corridor"],
                          "away_from_fault": f["eligible"] & f["confidence_safe"] & f["away_from_fault"]}
                for variable, assignment in choices.items():
                    for scope, support in scopes.items():
                        for index in range(2 if variable == "confidence_blocked" else 4):
                            mask = support & (assignment == index)
                            if not mask.any():
                                continue
                            rmse_r, rmse_c = [np.sqrt(e[:, mask].mean(axis=1)) for e in errors]
                            row = {"seed": seed, "epoch": epoch, "realization_id": rid, "variable": variable,
                                   "bin": index, "scope": scope, "pixel_count": int(mask.sum()),
                                   "S_mean": float(f["S"][mask].mean()), "dip_mean": float(f["dip"][mask].mean()),
                                   "rgt_mean_elastic": float(rmse_r.mean()), "cartesian_mean_elastic": float(rmse_c.mean()),
                                   "mean_elastic_gain": float(1-rmse_r.mean()/rmse_c.mean())}
                            for channel, prop in enumerate(PROPS):
                                row.update({prop+"_rgt_normalized_rmse": rmse_r[channel],
                                            prop+"_cartesian_normalized_rmse": rmse_c[channel],
                                            prop+"_delta_rgt_minus_cartesian": rmse_r[channel]-rmse_c[channel],
                                            prop+"_relative_gain": 1-rmse_r[channel]/rmse_c[channel]})
                            records.append(row)
                    if epoch == 20 and variable in cv_groups:
                        values = (errors[1]-errors[0]).mean(axis=0)
                        support = scopes["confidence_safe"]
                        parts = []
                        for index in range(4):
                            v = values[support & (assignment == index)]
                            parts.append([v.size, v.sum(), np.square(v).sum()])
                        cv_groups[variable].append(parts)
                log(f"Read-only frozen outcome binning seed={seed} epoch={epoch} realization={rid}")
        for variable, groups in cv_groups.items():
            cv_records.append({"seed": seed, "variable": variable, **heldout_binned_prediction(np.asarray(groups))})
    frame = pd.DataFrame(records)
    write_csv(OUT / "v00332s_informativeness_vs_rgt_gain.csv", records)
    write_csv(OUT / "S_vs_dip_heldout_prediction.csv", cv_records)
    last = frame[(frame.epoch == 20) & (frame.scope == "confidence_safe") & (frame.variable == "S")]
    effects = []
    for seed in CONFIG["seeds"]:
        local = last[(last.seed == seed) & (last.pixel_count >= 50)]
        lo = local[local.bin == 0].set_index("realization_id")
        hi = local[local.bin == 3].set_index("realization_id")
        paired = sorted(set(lo.index) & set(hi.index))
        if not paired:
            effects.append({"seed": seed, "paired_realizations": 0, "high_gain": None, "zero_gain": None, "high_minus_zero_gain": None})
            continue
        high_gain = 1-hi.loc[paired].rgt_mean_elastic.mean()/hi.loc[paired].cartesian_mean_elastic.mean()
        zero_gain = 1-lo.loc[paired].rgt_mean_elastic.mean()/lo.loc[paired].cartesian_mean_elastic.mean()
        effects.append({"seed": seed, "paired_realizations": len(paired), "paired_ids": paired,
                        "high_gain": float(high_gain), "zero_gain": float(zero_gain),
                        "high_minus_zero_gain": float(high_gain-zero_gain)})
    gate = applicability_signal([e["high_gain"] for e in effects], [e["high_minus_zero_gain"] for e in effects],
                               [e["paired_realizations"] for e in effects])
    signal = "RGT_APPLICABILITY_SIGNAL_PRESENT" if gate["passed"] else "RGT_APPLICABILITY_SIGNAL_NOT_ESTABLISHED"
    verify()
    summary = {"status": "PHASE_2_SCREEN_COMPLETE", "decision": signal, "phase2_gate": gate,
               "primary_seed_effects": effects, "S_vs_dip_heldout_prediction": cv_records,
               "RGT_INFORMATIVENESS_STATUS": "PREDICTIVE" if gate["passed"] else "WEAK" if any((e["high_minus_zero_gain"] or 0)>0 for e in effects) else "NOT_PREDICTIVE",
               "HIGH_DIP_STATUS": "UNRESOLVED", "GLOBAL_ELASTIC_STATUS": None, "DENSITY_STATUS": None,
               "FAULT_STATUS": "UNRESOLVED", "RANDOM_CONTROL_STATUS": None,
               "downstream_status_reason": "Adaptive and random models have not been implemented/trained; their effects are untested, not negative or equivalent",
               "training_performed": False, "model_or_checkpoint_loaded": False,
               "topology_changed": False, "r_inputs_unchanged": True,
               "bins_sha256": frozen.file_sha(OUT / "training_only_bins.json"),
               "stop_required": not gate["passed"],
               "next_action": "Proceed to phases3–6 with training-only threshold, then matched tests" if gate["passed"] else "STOP: do not invent another gate or train adaptive models"}
    write_json(OUT / "v00332s_summary.json", summary)
    plots(frame, saved)
    write_text(OUT / "v00332s_report.md", f"""# v00332s — structural applicability, phase-2 stop/go screen

Decision: **{signal}**.

This is a frozen-output analysis, not a new trained-model result. All 70 training
sections set observable-only S/dip bins before validation prediction differences
were opened. The six frozen validation sections and three seeds are reused;
q/r are not independent replications. Epoch20 is primary; other milestones are
descriptive. Eligibility and all decision criteria are in
v00332s_applicability_definition.md and v00332s_applicability_contract.json.

## Predeclared primary screen

Positive gain favors RGT. Each comparison uses the same eligible realizations in
the high and zero-S bins, with >=50 pixels per bin. Realizations, not individual
pixels, have equal weight. A 0.005 gain separation is 0.5 percentage points.

{frozen.markdown_table(pd.DataFrame(effects).drop(columns=["paired_ids"], errors="ignore"))}

```json
{json.dumps(gate, indent=2)}
```

## Does S predict value better than dip?

The following fixed-bin predictors are evaluated by leaving out each realization
in turn. Positive improvement means lower held-out squared prediction error than
an intercept. These exploratory validation fits do NOT alter bins, a threshold,
or the stop/go decision. S and dip use exactly the same pixel support.

{frozen.markdown_table(pd.DataFrame(cv_records))}

Per-property signed RGT-minus-Cartesian RMSE differences, all six milestones,
dip/displacement/confidence groups, and fault/away-from-fault QC comparisons
are in v00332s_informativeness_vs_rgt_gain.csv. Fault/facies/reservoir/plume truth
never enters S or the confidence rule. Undefined local RGT scale is retained as
undefined, not converted into an artificial high score. Full-grid versus tiled
correspondence disagreement is separately quantified in
full_grid_vs_tiled_context.csv. Stitched pixel errors cannot be attributed
causally to a single source edge; this is a prerequisite signal screen only.

## Scope and stop rule

{summary['next_action']}.

No adaptive topology, random-matched control, model initialization, optimizer,
new training, all20 evaluation, commit, push, or original-reference change was
performed. TransformerConv, segmentation, graph reinjection, flow, exact-PP,
losses, confidence thresholds and tie-break remain byte-for-byte unchanged.
The uncommitted r source is preserved in frozen_r_source_snapshot with hashes.

Adaptive HIGH_DIP_STATUS / GLOBAL_ELASTIC_STATUS / DENSITY_STATUS /
RANDOM_CONTROL_STATUS are **NOT TESTED**. They cannot honestly be classified
as gain, harm or equivalence without the conditional downstream experiment.
Initialization/multiseed/adaptive-stratified/random-control/all20 result CSVs and
adaptive causal effect figures are intentionally not fabricated.

Validation command results are in validation.json. All writes are confined to
new s source/test/config files and the new private s output directory.
""")
    log(f"{signal}; training_performed=False; report={OUT / 'v00332s_report.md'}")


def plots(frame, saved):
    final = frame[(frame.epoch == 20) & (frame.scope == "confidence_safe") & (frame.pixel_count >= 50)]
    for variable, number in (("S", "01"), ("dip", "02")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, prop in zip(axes, PROPS):
            for seed in CONFIG["seeds"]:
                part = final[(final.variable == variable) & (final.seed == seed)].groupby("bin").mean(numeric_only=True)
                gain = 100*(1-part[prop+"_rgt_normalized_rmse"]/part[prop+"_cartesian_normalized_rmse"])
                ax.plot(gain.index, gain, marker="o", label=str(seed))
            ax.axhline(0, color="grey", linewidth=.7)
            ax.set(xlabel=f"Training-only {variable} bin", ylabel="RGT relative RMSE gain (%)", title=prop)
        axes[0].legend(title="Seed")
        save_figure(fig, f"{number}_rgt_advantage_vs_{variable}.png")
    rid = saved["validation_screen_ids"][2]
    f = measure_one(rid, saved)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, name in zip(axes, ("S", "dip", "confidence_safe")):
        values = f[name].astype(float).copy()
        values[~f["valid_source"]] = np.nan
        upper = 3 if name in ("S", "dip") else 1
        plot = ax.imshow(values, vmin=0, vmax=upper, aspect="auto", interpolation="nearest")
        ax.set_title(f"{rid}: {name}")
        fig.colorbar(plot, ax=ax)
    save_figure(fig, "03_observable_informativeness_and_reliability.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rgt_mode in zip(axes, (False, True)):
        ax.imshow(f["S"], vmin=0, vmax=3, aspect="auto", cmap="Greys")
        for row in range(40, 100, 5):
            for col in range(40, 80, 3):
                if rgt_mode and not f["confidence_safe"][row, col]:
                    continue
                target = int(f["target_row"][row, col]) if rgt_mode else row
                ax.plot([col, col+1], [row, target], color="tab:blue", linewidth=.7)
        ax.set(xlim=(39, 81), ylim=(102, 38), title="Frozen full RGT" if rgt_mode else "Frozen Cartesian")
    save_figure(fig, "04_frozen_cartesian_rgt_edges_not_adaptive.png")


def validate():
    results = []
    for command in ([sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
                    [sys.executable, "-m", "pytest", "-ra"], ["git", "diff", "--check"]):
        result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
        print(result.stdout, flush=True)
        results.append({"command": command, "exit_code": result.returncode, "output": result.stdout})
        write_json(OUT / "validation.json", {"commands": results, "all_passed": len(results)==3 and all(x["exit_code"]==0 for x in results)})
        if result.returncode:
            raise RuntimeError("Validation failed; no scientific screen or adaptive training may proceed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "measure", "analyze", "validate", "run-screen"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen.print_torch_runtime()
    log("Screen uses NumPy/CPU statistics on saved predictions; no model inference or training. CUDA is checked by the test suite.")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(OUT / "runner_status.json", {"status": "running", "pid": os.getpid(), "command": args.command})
        try:
            if args.command == "run-screen":
                validate()
                prepare()
                measure()
                analyze()
            else:
                {"prepare": prepare, "measure": measure, "analyze": analyze, "validate": validate}[args.command]()
        except BaseException as error:
            write_json(OUT / "runner_status.json", {"status": "stopped_on_error", "error": str(error), "error_type": type(error).__name__})
            raise
        write_json(OUT / "runner_status.json", {"status": "command_complete", "command": args.command,
                                                "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
