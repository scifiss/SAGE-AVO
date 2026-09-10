"""Observable-only structural applicability measurements; no model changes."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
import torch

from sage_avo.diagnostics.rgt_topology_repair import discrete_topology, structural_fields
from sage_avo.models.graph import build_horizon_edges_tie_fixed


def applicability_fields(rgt: np.ndarray, thresholds: dict[str, float], *, max_shift: int = 3):
    """Measure each rightward physical correspondence on the native RGT grid.

    The confidence mask comes from the unchanged production Torch builder.
    The five-increment median and numerical epsilon define S, NOT confidence.
    Undefined local vertical scale gives NaN S; it must never imply applicability.
    No elastic, facies, fault, reservoir, or plume information is accepted.
    """
    tau = np.asarray(rgt, dtype=np.float32)
    if tau.ndim != 2 or min(tau.shape) < 2 or not np.isfinite(tau).all():
        raise ValueError("RGT must be a finite two-dimensional grid with both axes >= 2")
    topology = discrete_topology(tau, max_shift=max_shift, tie_fixed=True)
    height, width = tau.shape
    shape = (height, width - 1)
    target = topology.target_row.reshape(shape)
    rows, columns = np.indices(shape)
    increments = np.abs(np.diff(tau.astype(np.float64), axis=0))
    increments = np.concatenate((increments, increments[-1:]), axis=0)
    local = median_filter(increments, size=(5, 1), mode="nearest")[:, :-1]
    positive = increments[increments > 0]
    epsilon = 16 * np.finfo(np.float32).eps * (float(np.median(positive)) if positive.size else 1.)
    m_c = np.abs(tau[:, :-1] - tau[:, 1:]).astype(np.float64)
    m_r = topology.mismatch.reshape(shape).astype(np.float64)
    scale_valid = local > epsilon
    score = np.full(shape, np.nan)
    np.divide(m_c - m_r, local + epsilon, out=score, where=scale_valid)
    # Obtain exactly the source support retained by the frozen confidence rule.
    edges = build_horizon_edges_tie_fixed(
        torch.from_numpy(tau[None]), max_shift=max_shift,
        normalized_mismatch_threshold=thresholds["normalized_best_mismatch"],
        normalized_discontinuity_threshold=thresholds["normalized_cartesian_discontinuity"],
        dip_residual_threshold=thresholds["rgt_dip_residual"],
    )[0].numpy()
    forward = edges[:, :edges.shape[1] // 2]
    safe = np.zeros(shape, dtype=bool)
    source_row, source_column = forward[0] // width, forward[0] % width
    safe[source_row, source_column] = True
    if not np.array_equal(target[source_row, source_column], forward[1] // width):
        raise RuntimeError("Diagnostic candidate selection differs from frozen production topology")
    fields = structural_fields(tau)
    return {
        "S": score, "m_C": m_c, "m_R": m_r, "delta_tau_local": local,
        "epsilon": np.full(shape, epsilon), "scale_valid": scale_valid,
        "confidence_safe": safe, "displacement": target - rows,
        "target_row": target, "source_row": rows, "source_column": columns,
        "dip": fields["dip"][:, :-1], "curvature": fields["curvature"][:, :-1],
    }


def score_bins(score: np.ndarray, positive_cuts: list[float]) -> np.ndarray:
    """Zero/nonpositive bin then three positive training-quantile bins; -1 undefined."""
    cuts = np.asarray(positive_cuts, dtype=np.float64)
    if cuts.shape != (2,) or not (0 < cuts[0] < cuts[1]):
        raise ValueError("Two distinct positive training cuts are required")
    result = np.full(np.shape(score), -1, dtype=np.int8)
    finite = np.isfinite(score)
    result[finite & (score <= 0)] = 0
    positive = finite & (score > 0)
    result[positive] = 1 + np.searchsorted(cuts, score[positive], side="right")
    return result


def applicability_signal(high_gain, high_minus_low, covered_realizations, *, minimum_separation=.005):
    """Predeclared observational screen; not a significance test or model gate."""
    high = np.asarray(high_gain, dtype=float)
    contrast = np.asarray(high_minus_low, dtype=float)
    coverage = np.asarray(covered_realizations, dtype=int)
    if high.shape != (3,) or contrast.shape != (3,) or coverage.shape != (3,):
        raise ValueError("The fixed three-seed screen requires exactly three observations")
    checks = {
        "finite_all_three_seeds": bool(np.isfinite(high).all() and np.isfinite(contrast).all()),
        "high_S_positive_gain_all_seeds": bool((high > 0).all()),
        "high_S_better_than_zero_S_all_seeds": bool((contrast > 0).all()),
        "mean_gain_separation_at_least_half_percentage_point": bool(contrast.mean() >= minimum_separation),
        "at_least_four_paired_realizations_per_seed": bool((coverage >= 4).all()),
    }
    return {"passed": all(checks.values()), "checks": checks}


def heldout_binned_prediction(group_sums: np.ndarray) -> dict[str, float]:
    """Leave-one-realization-out prediction of pixelwise normalized elastic SE gain.

    Input [realization, bin, (count,sum(y),sum(y*y))]. Fits bin means on the other
    realizations, with their global mean for empty bins. Reports equal-realization
    MSE and improvement over an equally cross-fitted intercept. This is exploratory
    validation analysis, not threshold fitting or pixel-level significance.
    """
    groups = np.asarray(group_sums, dtype=float)
    if groups.ndim != 3 or groups.shape[2] != 3 or groups.shape[0] < 2:
        raise ValueError("Expected at least two realizations and three sufficient statistics")
    errors, null_errors = [], []
    for index, held in enumerate(groups):
        training = np.delete(groups, index, axis=0).sum(axis=0)
        count = training[:, 0].sum()
        if count <= 0 or held[:, 0].sum() <= 0:
            raise ValueError("Heldout and fitting support must be nonempty")
        intercept = training[:, 1].sum() / count
        fitted = np.full(training.shape[0], intercept)
        np.divide(training[:, 1], training[:, 0], out=fitted, where=training[:, 0] > 0)
        n, total, square = held.T
        errors.append(float(np.sum(square - 2 * fitted * total + fitted**2 * n) / n.sum()))
        null_errors.append(float(np.sum(square - 2 * intercept * total + intercept**2 * n) / n.sum()))
    mse, null = float(np.mean(errors)), float(np.mean(null_errors))
    return {"heldout_equal_realization_mse": mse, "intercept_mse": null,
            "relative_mse_improvement_over_intercept": (null - mse) / max(null, 1e-15)}
