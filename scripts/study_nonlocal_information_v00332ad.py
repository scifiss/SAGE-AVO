#!/usr/bin/env python3
"""Pushed-source-gated frozen-CNN information study. Train small ridge probes ONLY."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch

from sage_avo.config import load_config, seed_everything
from sage_avo.diagnostics.nonlocal_information import (
    MODES, PRIMARY, PROPERTIES, comparison, decision, feature_bank,
    fit_probes, paired_uncertainty, realization_metrics, region_masks, sample_observables,
)
from sage_avo.diagnostics.rgt_topology_repair import load_faults
from sage_avo.experiments.manifest import file_sha256
from sage_avo.experiments.prediction import load_controlled_model
from sage_avo.runtime import print_torch_runtime, select_torch_device


REPO = Path(__file__).resolve().parents[1]
BRANCH = "experiment/v00332ad-nonlocal-information-value"
PARENT = "64f3c9f9243c84ec61abcd32e06d027f63d7ace2"
CONFIG = REPO / "configs/nonlocal_information_v00332ad.yaml"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def dump(path, value):
    """Atomic private-only report/cache metadata writes."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def safe_output(output):
    output = Path(output).resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Scientific artifacts must be outside the Git worktree")
    # Reject every registered Git worktree, not just this script's repository.
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            root = Path(line.removeprefix("worktree ")).resolve()
            if output == root or root in output.parents:
                raise ValueError("Output cannot be inside any Git worktree")
    return output


def protected_hashes(expected):
    if git("rev-parse", "HEAD") != expected or git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Exact experiment branch and contract SHA required")
    if git("rev-parse", f"origin/{BRANCH}") != expected:
        raise RuntimeError("Experiment must already be pushed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked source must be clean before execution")
    git("merge-base", "--is-ancestor", PARENT, expected)
    paths = git("ls-files", "src", "configs", "scripts", "tests", "docs", "pyproject.toml").splitlines()
    hashes = {}
    for name in paths:
        digest = file_sha256(REPO / name)
        blob = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=REPO)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise RuntimeError(f"Protected source mismatch: {name}")
        hashes[name] = digest
    return hashes


def make_contract(args):
    hashes = protected_hashes(args.commit)
    output = safe_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "experiment_contract.json"
    if destination.exists():
        raise FileExistsError("Contract already exists; use run to resume exactly that contract")
    dataset, checkpoint = args.dataset.resolve(), args.checkpoint.resolve()
    config = load_config(CONFIG)
    splits = json.loads((dataset / "split_ids.json").read_text())
    groups = json.loads((dataset / "split_group_ids.json").read_text())
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(splits[a]) & set(splits[b]) or set(groups[a]) & set(groups[b]):
            raise RuntimeError("Realization/geology-group splits overlap")
    checkpoint_metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_metadata["epoch"] != config["checkpoint_epoch"]:
        raise RuntimeError("Frozen checkpoint is not the predeclared epoch 40")
    model_config = checkpoint_metadata["config"]
    if model_config["model"].get("hidden_channels") != 64:
        raise RuntimeError("Expected frozen validated 64-channel dense CNN checkpoint")
    input_paths = [checkpoint, dataset / "dataset_manifest.json", dataset / "split_ids.json",
                   dataset / "split_group_ids.json", dataset / "normalization.json"]
    record = {
        "repository": str(REPO), "branch": BRANCH, "commit_sha": args.commit,
        "parent_commit_sha": PARENT, "source_sha256": hashes, "config": config,
        "dataset": str(dataset), "checkpoint": str(checkpoint),
        "output": str(output), "stage02": str(args.stage02.resolve()),
        "input_sha256": {str(p): file_sha256(p) for p in input_paths},
        "checkpoint_config": model_config, "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_selection": "historical validation-only best whole-realization, epoch 40",
        "splits": splits, "geology_group_splits": groups,
        "development_role": "exploratory: validation repeatedly examined and used for checkpoint selection",
        "test_executed": False, "scientific_training": "ridge regression probes ONLY",
        "inference_state": "flow time 0, state = supplied low-frequency prior",
        "cnn_path": "unchanged time_embedding/condition_embedding/encoder; no GNN or flow steps",
    }
    dump(destination, record)
    print(f"Frozen contract: {destination}", flush=True)


def verify_contract(record):
    if protected_hashes(record["commit_sha"]) != record["source_sha256"]:
        raise RuntimeError("Experiment source hashes differ from contract")
    for path, digest in record["input_sha256"].items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"Frozen input changed: {Path(path).name}")
    if str(safe_output(record["output"])) != record["output"]:
        raise RuntimeError("Output does not match frozen contract")
    if record["config"] != load_config(CONFIG):
        raise RuntimeError("Private contract parameters differ from pushed experiment configuration")


def model_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def collect(record, device):
    output, dataset = Path(record["output"]), Path(record["dataset"])
    config = record["config"]
    normalization = json.loads((dataset / "normalization.json").read_text())
    model = load_controlled_model("full", record["checkpoint_config"], Path(record["checkpoint"]),
                                  device, normalization)
    model.requires_grad_(False)
    before = model_digest(model)
    # Loading must not silently substitute normalization for different inputs.
    for key, buffer in (("x_mean", "X_mean_buf"), ("x_std", "X_std_buf"),
                        ("y_mean", "Y_mean_buf"), ("y_std", "Y_std_buf")):
        if not np.allclose(getattr(model, buffer).cpu().numpy().ravel(), normalization[key], rtol=1e-5):
            raise RuntimeError(f"Frozen checkpoint normalization mismatch: {key}")
    records, qc_rows = {}, []
    cache = output / "query_samples"
    cache.mkdir(exist_ok=True)
    for split in ("train", "validation"):
        records[split] = []
        for number, rid in enumerate(record["splits"][split], start=1):
            path = cache / f"realization_{rid}.npz"
            qc_path = path.with_suffix(".json")
            source = dataset / "realizations" / f"realization_{rid:07d}.npz"
            fault_file = Path(record["stage02"]) / f"realization_{rid:07d}.json"
            if path.exists() and qc_path.exists():
                qc = json.loads(qc_path.read_text())
                if qc["commit_sha"] != record["commit_sha"] or qc["checkpoint_sha256"] != record["input_sha256"][record["checkpoint"]]:
                    raise RuntimeError("Cannot reuse a sample cache from another source/checkpoint")
                if qc["source_archive_sha256"] != file_sha256(source) or qc["fault_qc_sidecar_sha256"] != file_sha256(fault_file):
                    raise RuntimeError("Private input changed since sample cache was created")
                if qc["sample_sha256"] != file_sha256(path):
                    raise RuntimeError("Private sample cache was modified or corrupted")
                with np.load(path, allow_pickle=False) as archive:
                    samples = {k: archive[k] for k in archive.files}
            else:
                started = time.monotonic()
                with np.load(source, allow_pickle=False) as archive:
                    # ONLY inference-available arrays enter correspondence/feature extraction.
                    avo, low = archive["avo"], archive["low"]
                    rgt, valid = archive["rgt"], archive["valid_mask"].astype(bool)
                    cnn, normalized_avo, normalized_low = feature_bank(
                        model, avo, low, normalization, tuple(config["patch_shape"]),
                        tuple(config["stride"]), device,
                    )
                    samples, qc = sample_observables(cnn, normalized_avo, normalized_low, rgt,
                                                     valid, config, rid)
                    t, x = samples["t"], samples["x"]
                    # Supervision and region truth are read ONLY AFTER selection is frozen.
                    samples["target"] = ((archive["elastic"][:, t, x] - low[:, t, x])
                                          / np.asarray(normalization["y_std"])[:, None]).T
                    samples["reservoir"] = archive["reservoir_mask"][t, x].astype(bool)
                    samples["geology_id"] = np.full(len(t), int(archive["geology_realization_id"]))
                faults = load_faults(Path(record["stage02"]), rid)
                samples["fault_adjacent"] = np.zeros(len(t), bool)
                for fault in faults:
                    boundary = float(fault["column"]) + float(fault["dip"]) * t
                    samples["fault_adjacent"] |= np.abs(x - boundary) <= config["fault_corridor_traces"]
                samples["realization_id"] = np.full(len(t), rid)
                # Pair correspondence QC uses no geological truth.
                tau_source = rgt[t, x]
                errors = []
                for k, mode in enumerate(MODES):
                    times = samples["remote_times"][k]
                    columns = samples["remote_x"]
                    remote_tau = np.empty_like(times)
                    for j in range(times.shape[1]):
                        lower = np.floor(times[:, j]).astype(int)
                        upper = np.ceil(times[:, j]).astype(int)
                        f = times[:, j] - lower
                        remote_tau[:, j] = rgt[lower, columns[:, j]] * (1 - f) + rgt[upper, columns[:, j]] * f
                    errors.append(float(np.median(np.abs(remote_tau - tau_source[:, None]))))
                qc.update(realization_id=rid, split=split, commit_sha=record["commit_sha"],
                          checkpoint_sha256=record["input_sha256"][record["checkpoint"]],
                          source_archive_sha256=file_sha256(source),
                          fault_qc_sidecar_sha256=file_sha256(fault_file),
                          seconds=time.monotonic() - started,
                          query_count=len(t), median_cartesian_delta_tau=errors[0],
                          median_rgt_delta_tau=errors[1], median_disrupted_delta_tau=errors[2],
                          rgt_shift_jump_median=float(np.median(samples["rgt_shift_jump"])),
                          rgt_shift_jump_nonfinite=int((~np.isfinite(samples["rgt_shift_jump"])).sum()))
                # JSON infinity is not portable; nonfinite jumps are counted above.
                if not np.isfinite(qc["rgt_shift_jump_median"]):
                    qc["rgt_shift_jump_median"] = None
                temporary = path.with_name(path.stem + ".tmp.npz")
                np.savez_compressed(temporary, **samples)
                os.replace(temporary, path)
                qc["sample_sha256"] = file_sha256(path)
                dump(qc_path, qc)
                del cnn
            records[split].append(samples)
            qc_rows.append(qc)
            print(f"{split} {number}/{len(record['splits'][split])}: {rid}; queries={len(samples['t'])}", flush=True)
        # Only regression/sample arrays survive; coordinates remain in private per-case caches.
        excluded = {"remote_x", "remote_times"}
        records[split] = {k: np.concatenate([s[k] for s in records[split]], axis=0)
                          for k in records[split][0] if k not in excluded}
    if model_digest(model) != before:
        raise RuntimeError("Frozen model changed during feature extraction")
    if set(records["train"]["geology_id"]) & set(records["validation"]["geology_id"]):
        raise RuntimeError("Observed geological IDs cross probe train/development boundary")
    pd.DataFrame(qc_rows).to_csv(output / "correspondence_qc.csv", index=False)
    dump(output / "model_freeze_verification.json", {"before": before, "after": before,
                                                    "parameters_updated": False,
                                                    "flow_time": 0, "teacher_forcing_used": False})
    return records, normalization, pd.DataFrame(qc_rows)


def md_table(frame):
    # Self-contained: no optional tabulate installation required for audit reporting.
    columns = list(frame.columns)
    def clean(value):
        return f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
                      *["| " + " | ".join(clean(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def analyze(record, records, normalization, qc):
    output, config = Path(record["output"]), record["config"]
    train, dev = records["train"], records["validation"]
    dip_threshold = float(np.quantile(train["dip"], config["high_dip_training_quantile"]))
    finite_jump = train["rgt_shift_jump"][np.isfinite(train["rgt_shift_jump"])]
    jump_threshold = float(np.quantile(finite_jump, config["discontinuity_training_quantile"]))
    predictions, coefficients, fitting = fit_probes(train, dev, config)
    masks = region_masks(dev, dip_threshold, jump_threshold)
    metrics = realization_metrics(dev, predictions, normalization, masks)
    paired = paired_uncertainty(metrics, config)
    metrics.to_csv(output / "per_realization_metrics.csv", index=False)
    paired.to_csv(output / "paired_realization_uncertainty.csv", index=False)
    summary = metrics.groupby(["region", "probe"])[["joint_nrmse", *[f"{p}_rmse" for p in PROPERTIES]]].mean().reset_index()
    summary.to_csv(output / "probe_summary.csv", index=False)
    pd.DataFrame(fitting["capacities"]).to_csv(output / "probe_capacity.csv", index=False)
    np.savez_compressed(output / "private_probe_coefficients.npz", **coefficients)
    np.savez_compressed(output / "private_development_predictions.npz", **predictions,
                        target=dev["target"], realization_id=dev["realization_id"])
    adequate = (
        len(np.unique(dev["realization_id"])) >= config["minimum_development_realizations"]
        and float((qc.common_support_count / qc.candidate_count).min()) >= config["minimum_common_support_fraction"]
        and fitting["local_feature_rank"] >= config["local_common_components"]
        and fitting["remote_feature_rank"] == config["extra_components"]
        and bool(np.all(train["target"].var(axis=0) > 1e-6))
    )
    chosen = decision(paired, config, adequate=adequate)
    highlights = [comparison(paired, a, b) for a, b in (
        ("local_only", "local_rgt"), ("local_only", "local_cartesian"),
        ("local_cartesian", "local_rgt"), ("local_disrupted", "local_rgt"),
        ("local_rgt_prior", "local_rgt"), ("local_rgt_ava", "local_rgt"),
        ("local_only", "conditional_rgt"), ("conditional_cartesian", "conditional_rgt"),
    )]
    train_elastic_mean = (train["target"] + train["prior"]).mean(axis=0)
    elastic_dev = dev["target"] + dev["prior"]
    reference_mse = np.mean((elastic_dev - train_elastic_mean) ** 2, axis=0)
    prior_analysis = []
    for name in ("supplied_prior", "prior_calibration", *PRIMARY, "local_rgt_prior"):
        mse = np.mean((predictions[name] - dev["target"]) ** 2, axis=0)
        for j, prop in enumerate(PROPERTIES):
            prior_analysis.append({"probe": name, "property": prop,
                                   "full_elastic_R2_against_training_mean": float(1 - mse[j] / reference_mse[j]),
                                   "residual_mse": float(mse[j])})
    pd.DataFrame(prior_analysis).to_csv(output / "prior_explanatory_value.csv", index=False)
    coverage = [{"region": name, "queries": int(mask.sum()),
                 "realizations_with_at_least_four_points": int(sum(np.sum(mask & (dev["realization_id"] == rid)) >= 4 for rid in np.unique(dev["realization_id"]))) }
                for name, mask in masks.items()]
    record_out = {
        "decision": chosen, "adequate": bool(adequate), "branch": BRANCH,
        "commit_sha": record["commit_sha"], "parent_commit_sha": PARENT,
        "checkpoint_sha256": record["input_sha256"][record["checkpoint"]],
        "checkpoint_epoch": 40, "flow_time": 0, "SAGE_AVO_training_performed": False,
        "graph_construction_modified": False, "probe_fitting": fitting,
        "splits": record["splits"], "test_used": False,
        "development_role": record["development_role"],
        "queries_per_case": config["queries_per_realization"], "signed_offset_policy": "inward-facing 16/32/64 traces, identical across modes",
        "region_thresholds_training_only": {"high_dip": dip_threshold, "shift_jump": jump_threshold},
        "region_coverage": coverage, "primary_comparisons": highlights,
        "prior_explanatory_value": prior_analysis, "source_changed_after_run_started": False,
    }
    verify_contract(record)
    dump(output / "nonlocal_information_summary.json", record_out)
    overview = summary[summary.region.eq("all")]
    region_table = summary[summary.probe.isin(PRIMARY)]
    paired_table = pd.DataFrame(highlights)[["baseline", "candidate", "mean_improvement", "ci_low", "ci_high", "relative_improvement", "positive_realizations"]]
    text = f"""# v00332ad: nonlocal information value study

Decision: `{chosen}`

## Scope and provenance

Branch: `{BRANCH}`. Source SHA: `{record['commit_sha']}`. Parent: `{PARENT}`.
Frozen epoch-40 dense checkpoint SHA256: `{record_out['checkpoint_sha256']}`.
Only small linear ridge probes were fitted. No SAGE-AVO training, graph modifications,
Heun sampling, intermediate truth-derived states, or test evaluation occurred.
All source/config/test hashes and exact split IDs are in `experiment_contract.json`.

## Protocol

Training: {len(record['splits']['train'])} realizations. Development: {len(record['splits']['validation'])}
realizations, **exploratory**, repeatedly examined and historically used for checkpoint
selection. Both realization IDs and geology IDs are disjoint. Test: unused.
{config['queries_per_realization']} common-support queries/case, selected from observable
support without fault/reservoir/elastic truth; inward-facing offsets 16/32/64 traces
match across all modes. Native RGT correspondence preserves plateaus as ambiguous.
Ambiguity and observable shift-discontinuity records are in `correspondence_qc.csv`
and the private query caches. Low-confidence routes are reported, not truth-pruned.

CNN extraction reproduces the exact unmodified pre-GNN modules at flow time zero,
with state equal to the supplied prior, production 50x100 tiles and 10x25 stride,
Hann stitching and FP32 CUDA. Frozen parameter digests match before/after extraction.
CNN **GroupNorm is tile-wide**, so a local query representation is not spatially
independent of remote seismic inside a shared tile. This conservatively tests added
information beyond that existing representation, not beyond an ideal local operator.

Targets are normalized `elastic - supplied prior`; truth enters only supervised probe
targets and post-selection regional QC. This is not an endpoint-performance study.
Every primary probe has 64 regressors plus an intercept per property (195 coefficients).
Local-only uses 64 training-fitted local PCs; each remote probe replaces the last 16
with 16 remote PCs, retaining the same first 48 local PCs. A single shared remote
PCA is fitted on pooled TRAINING correspondences, never development. This sacrifices
some local dimensions rather than increasing capacity. Additional 80-regressor
conditional-innovation probes retain ALL 64 local PCs and remove training-estimated
linear local predictability from remote features: these are supporting diagnostics,
not part of the equal-capacity primary decision. Remote-prior and AVA-only controls
also have 64 regressors. Prior calibration has three inputs and is descriptive.

One shared ridge coefficient, {fitting['shared_alpha']}, is selected on a realization-disjoint
56/14 internal training split, averaging the four primary objectives, then all transforms
and probes are refitted on all training realizations. No development tuning occurred.
Effective degrees of freedom and feature ranks are in `probe_capacity.csv`/summary.
Scores are per-realization RMSE in training-standard-deviation units (joint = mean
of the three property NRMSEs). Physical Vp/Vs RMSE is m/s; density is g/cm^3.
Paired 95% intervals resample realizations, not pixels. Region intervals are exploratory,
unadjusted for multiple comparisons. Positive improvement means lower candidate error.

## Overall development performance

{md_table(overview)}

## Paired realization differences and uncertainty

{md_table(paired_table)}

The predeclared primary gate requires at least 1% mean relative improvement and
a strictly positive 95% paired lower bound over local and disrupted controls; RGT
specific support additionally requires that gate over Cartesian. Failing to show
superiority is not evidence of equivalence or proof that RGT/GNN cannot help.
Adequacy checks passed: {adequate}.

## Regional results: unfavorable regions are retained

{md_table(region_table)}

{md_table(pd.DataFrame(coverage))}

High-dip threshold {dip_threshold:.6g} and discontinuity threshold {jump_threshold:.6g}
were fitted from training observables only. Reservoir/fault masks are evaluation-only,
with a three-trace fault corridor. Regions overlap; ordinary excludes all three.
At least four sampled queries/case/region are required; sparse fault coverage weakens
regional conclusions. Complete property-specific paired tables remain private.

## Does the prior already explain the structure?

{md_table(pd.DataFrame(prior_analysis))}

Full-elastic R2 uses the training-mean elastic predictor as reference. High prior R2
alone does not establish that the remaining residual is unimportant or unpredictable.
CNN features already encode the supplied prior; compare RGT remote CNN/AVA against
the remote-prior-only and remote-AVA-only controls before attributing gains to seismic.
The immutable synthetic prior is truth-derived by its original dataset procedure;
it is accepted as an inference input here, not regenerated or used to order matches.

## Limitations and recommendations

This linear, fixed-budget information screen uses one frozen historical checkpoint,
time zero only, and an existing synthetic corpus. PCA and linear probes can miss
nonlinear information. Tile stitching/GroupNorm, noisy observed AVA, native plateau
ambiguity, prior strength, and geometry-dependent common-support selection limit
generalization. The deliberately disrupted control preserves lateral positions and
source-trace tau marginals but does not destroy all broad geological correlations.
The endpoint and full GNN representation were not compared; no architectural causal
claim is licensed. Regional QC uses sparse uniformly chosen points, not whole-section
error maps. Repeated development use makes this decision exploratory, not confirmation.

If equal-capacity gains are credible, first reproduce on new realization-disjoint
data and check remote-prior attribution; do not automatically redesign or train a GNN.
If gains are absent, investigate probe nonlinear capacity, prior strength, alignment,
and noisy AVA before concluding that geological nonlocality is useless. Any region
where RGT harms is a constraint on future correspondence use, not to be concealed.
This study stops here. No next experiment is started.
"""
    (output / "nonlocal_information_report.md").write_text(text)
    dump(output / "execution_complete.json", {"decision": chosen, "commit_sha": record["commit_sha"],
                                             "private_results_only": True, "completed_unix": time.time()})
    print(overview.to_string(index=False), flush=True)
    print(chosen, flush=True)


def run(args):
    record = json.loads((Path(args.output) / "experiment_contract.json").read_text())
    verify_contract(record)
    print_torch_runtime()
    device = select_torch_device("cuda", require_cuda=True, context="frozen CNN information study")
    seed_everything(record["config"]["seed"], deterministic_torch=True)
    records, normalization, qc = collect(record, device)
    analyze(record, records, normalization, qc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    for name in ("dataset", "checkpoint", "stage02", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    freeze.add_argument("--commit", required=True)
    freeze.set_defaults(function=make_contract)
    execute = commands.add_parser("run")
    execute.add_argument("--output", type=Path, required=True)
    execute.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
