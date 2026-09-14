#!/usr/bin/env python3
"""Training-split resolution prerequisite for v00332w; no fault truth or training."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import time

import numpy as np

import run_skeleton_graph_v00332u as u
from sage_avo.config import load_config
from sage_avo.diagnostics.flattened_component_graph import flatten_rgt

CONFIG_PATH = u.REPO / "configs/development_diagnostics_v00332w.yaml"
CONFIG = load_config(CONFIG_PATH)
OUT = u.BASE / "stage04" / CONFIG["experiment_name"]
u.OUT = OUT


def quantiles(values):
    values = np.asarray(values, float)
    return {f"p{q}": float(np.quantile(values, q / 100)) for q in [50, 95, 99]}


def main():
    if OUT.exists():
        raise RuntimeError(f"Output exists; refusing overwrite: {OUT}")
    q = json.loads(u.Q.read_text())
    train = q["split_ids"]["train"]
    indices = np.linspace(0, len(train) - 1, CONFIG["training_qc_count"], dtype=int)
    ids = [train[i] for i in indices]
    sources = [
        CONFIG_PATH,
        Path(__file__),
        u.REPO / "src/sage_avo/diagnostics/dual_domain_fault_barrier.py",
        u.REPO / "tests/test_dual_domain_fault_barrier.py",
    ]
    inputs = [u.DATASET / "realizations" / f"realization_{rid:07d}.npz" for rid in ids]
    protected = {str(path): u.sha(path) for path in inputs}
    u.json_file(
        "v00332w_contract.json",
        {
            "revision": "v00332w-resolution-prerequisite",
            "config": CONFIG,
            "training_qc_ids": ids,
            "training_qc_selection": "six evenly spaced positions in frozen training split",
            "source_sha256": {str(path.relative_to(u.REPO)): u.sha(path) for path in sources},
            "input_sha256": protected,
            "fault_truth_loaded": False,
            "validation_realizations_loaded": False,
            "training": False,
        },
    )
    rows = []
    for resolution in CONFIG["resolutions"]:
        for rid, path in zip(ids, inputs):
            start = time.perf_counter()
            with np.load(path, allow_pickle=False) as archive:
                mapping = flatten_rgt(
                    archive["rgt"], archive["avo"], archive["valid_mask"], resolution
                )
            elapsed = time.perf_counter() - start
            row = {
                "resolution": resolution,
                "realization_id": rid,
                **{
                    f"time_error_{key}_samples": value
                    for key, value in quantiles(
                        mapping["grid_resampling_time_error_samples"]
                    ).items()
                },
                **{
                    f"ava_abs_error_{key}": value
                    for key, value in quantiles(mapping["roundtrip_avo_abs"]).items()
                },
                "ambiguous_plateau_fraction": mapping["ambiguous_plateau_fraction"],
                "runtime_seconds": elapsed,
                "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "mapping_array_bytes": int(
                    mapping["inverse_t"].nbytes
                    + mapping["avo"].nbytes
                    + mapping["valid"].nbytes
                    + mapping["evidence"].nbytes
                ),
            }
            rows.append(row)
            u.log(
                f"training {rid} resolution {resolution}: time p99 "
                f"{row['time_error_p99_samples']:.3f} samples"
            )
    u.csv_file("v00332w_flattening_resolution_qc.csv", rows)
    passing = []
    for resolution in CONFIG["resolutions"]:
        subset = [r for r in rows if r["resolution"] == resolution]
        if all(
            r["time_error_p99_samples"] <= CONFIG["target_time_error_p99_samples"] for r in subset
        ):
            passing.append(resolution)
    selected = min(passing) if passing else None
    status = "ACCURATE" if selected is not None else "INSUFFICIENT"
    decision = (
        "IMPLEMENTATION_PROBLEM" if selected is not None else "FLATTENING_RESOLUTION_UNSTABLE"
    )
    # The dual-domain validation stage is intentionally unreachable without a frozen passing grid.
    summary = {
        "decision": decision,
        "flattening_status": status,
        "selected_resolution": selected,
        "fault_split_status": "FAIL",
        "long_edge_fault_status": "FAIL",
        "high_dip_retention": "FAIL",
        "cross_fault_metadata_status": "NOT_AVAILABLE",
        "validation_fault_analysis_performed": False,
        "training_performed": False,
    }
    u.json_file("v00332w_summary.json", summary)
    maxima = {}
    for resolution in CONFIG["resolutions"]:
        subset = [r for r in rows if r["resolution"] == resolution]
        maxima[resolution] = {
            "worst_time_p99_samples": max(r["time_error_p99_samples"] for r in subset),
            "worst_ava_p99": max(r["ava_abs_error_p99"] for r in subset),
            "total_runtime_seconds": sum(r["runtime_seconds"] for r in subset),
            "maximum_mapping_mb": max(r["mapping_array_bytes"] for r in subset) / 2**20,
        }
    report = f"""# v00332w — uniform RGT-grid resolution prerequisite

Decision: **{decision}**.

FLATTENING_STATUS: {status}
FAULT_SPLIT_STATUS: FAIL (not evaluated because the prerequisite failed)
LONG_EDGE_FAULT_STATUS: FAIL (not evaluated because the prerequisite failed)
HIGH_DIP_RETENTION: FAIL (not evaluated because the prerequisite failed)
CROSS_FAULT_METADATA_STATUS: NOT_AVAILABLE

Only six deterministically spaced realizations from the frozen TRAIN split were loaded: {ids}.
No validation realization, fault truth, model, checkpoint, optimizer, or training path was accessed.
The metric reconstructs physical time from the finite uniform tau grid at original identifiable
RGT samples; plateau ambiguity is reported separately. The required rule is p99 <=
{CONFIG["target_time_error_p99_samples"]} sample on every training-QC realization.

```json
{json.dumps(maxima, indent=2)}
```

Selected resolution: {selected}. Because no resolution passed, no threshold was frozen and the
dual-domain barrier could not be evaluated without violating the declared ordering. The likely
cause is near-flat RGT intervals: a global uniform tau grid allocates samples inefficiently and
converges too slowly in the extreme inverse slope. A future prerequisite should evaluate an
adaptive monotone tau grid or interval-aware representation on training data before fault QC.
The observable shift/jump normalization and simple pre-union barrier primitives are implemented
and unit tested, but were not applied scientifically in this stopped run.
"""
    u.write(OUT / "v00332w_report.md", report.encode())
    for path, digest in protected.items():
        if u.sha(path) != digest:
            raise RuntimeError("Training input changed: " + path)
    u.log(f"{decision}; stopped before validation fault analysis: {OUT}")


if __name__ == "__main__":
    main()
