"""Read-only topology QC for the v00332p RGT repair experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


LEGACY = "RGT_V1_LEGACY"
TIE_FIXED = "RGT_V2_TIE_FIXED"
ADAPTIVE = "T2B_ADAPTIVE_SEARCH"
INVERSE = "T2C_INVERSE_RGT"
CARTESIAN = "CARTESIAN"


@dataclass(frozen=True)
class Topology:
    source_row: np.ndarray
    source_column: np.ndarray
    target_row: np.ndarray
    target_column: np.ndarray
    shift: np.ndarray
    mismatch: np.ndarray
    tied: np.ndarray
    exact_tied: np.ndarray
    boundary_hit: np.ndarray
    valid: np.ndarray
    search_radius: np.ndarray


def near_tie_tolerance(source: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Sixteen local ULPs, matching the experimental Torch graph builder."""
    dtype = np.result_type(source.dtype, candidates.dtype)
    scale = np.maximum.reduce(
        (np.abs(source), np.max(np.abs(candidates), axis=-1), np.ones_like(source))
    )
    return 16.0 * np.finfo(dtype).eps * scale


def _candidate_arrays(rgt: np.ndarray, max_shift: int) -> tuple[np.ndarray, ...]:
    height, width = rgt.shape
    rows, columns = np.meshgrid(
        np.arange(height), np.arange(width - 1), indexing="ij"
    )
    shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int64)
    target_rows = rows[..., None] + shifts
    valid = (target_rows >= 0) & (target_rows < height)
    clipped = np.clip(target_rows, 0, height - 1)
    target_columns = np.broadcast_to((columns + 1)[..., None], clipped.shape)
    candidates = rgt[clipped, target_columns]
    source = rgt[rows, columns]
    mismatch = np.abs(source[..., None] - candidates)
    return rows, columns, shifts, clipped, valid, source, candidates, mismatch


def _balanced_lexicographic_choice(
    mismatch: np.ndarray,
    valid: np.ndarray,
    shifts: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    source: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(valid, mismatch, np.inf)
    minimum = masked.min(axis=-1, keepdims=True)
    tolerance = near_tie_tolerance(source, candidates)[..., None]
    near = valid & (masked <= minimum + tolerance)
    absolute = np.broadcast_to(np.abs(shifts), mismatch.shape)
    minimum_absolute = np.where(near, absolute, np.iinfo(np.int64).max).min(
        axis=-1, keepdims=True
    )
    finalists = near & (absolute == minimum_absolute)
    positive_preferred = ((rows + columns) % 2 == 0)[..., None]
    shift_grid = np.broadcast_to(shifts, mismatch.shape)
    sign_penalty = np.where(
        shift_grid == 0, 0, ((shift_grid > 0) != positive_preferred).astype(np.int64)
    )
    choice = np.where(finalists, sign_penalty, np.iinfo(np.int64).max).argmin(axis=-1)
    return choice, near.sum(axis=-1) > 1


def discrete_topology(
    rgt: np.ndarray,
    *,
    max_shift: int,
    tie_fixed: bool,
    adaptive_margin: int | None = None,
) -> Topology:
    rows, columns, shifts, targets, valid, source, candidates, mismatch = _candidate_arrays(
        np.asarray(rgt, dtype=np.float32), max_shift
    )
    if adaptive_margin is None:
        radius = np.full(rows.shape, max_shift, dtype=np.int64)
    else:
        vertical = np.abs(np.gradient(rgt.astype(np.float64), axis=0))[:, :-1]
        lateral = np.abs(np.diff(rgt.astype(np.float64), axis=1))
        predicted = lateral / np.maximum(vertical, 16.0 * np.finfo(np.float32).eps)
        radius = np.clip(
            np.ceil(predicted + adaptive_margin).astype(np.int64),
            min(3, max_shift),
            max_shift,
        )
        valid = valid & (np.abs(shifts)[None, None, :] <= radius[..., None])
    masked = np.where(valid, mismatch, np.inf)
    minimum = masked.min(axis=-1, keepdims=True)
    tolerance = near_tie_tolerance(source, candidates)[..., None]
    exact_tied = (valid & (masked == minimum)).sum(axis=-1) > 1
    tied = (valid & (masked <= minimum + tolerance)).sum(axis=-1) > 1
    if tie_fixed:
        choice, _ = _balanced_lexicographic_choice(
            mismatch, valid, shifts, rows, columns, source, candidates
        )
    else:
        choice = masked.argmin(axis=-1)
    target = np.take_along_axis(targets, choice[..., None], axis=-1)[..., 0]
    selected_mismatch = np.take_along_axis(mismatch, choice[..., None], axis=-1)[..., 0]
    selected_shift = shifts[choice]
    return Topology(
        rows.ravel(),
        columns.ravel(),
        target.ravel(),
        (columns + 1).ravel(),
        selected_shift.ravel(),
        selected_mismatch.ravel(),
        tied.ravel(),
        exact_tied.ravel(),
        (np.abs(selected_shift) == radius).ravel(),
        np.ones(rows.size, dtype=bool),
        radius.ravel(),
    )


def inverse_rgt_topology(rgt: np.ndarray) -> Topology:
    """Invert each monotonic next-trace RGT column and omit out-of-range tau."""
    rgt = np.asarray(rgt, dtype=np.float32)
    height, width = rgt.shape
    source_rows: list[np.ndarray] = []
    source_columns: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    for column in range(width - 1):
        source_tau = rgt[:, column]
        target_tau = rgt[:, column + 1]
        insertion = np.searchsorted(target_tau, source_tau, side="left")
        lower = np.clip(insertion - 1, 0, height - 1)
        upper = np.clip(insertion, 0, height - 1)
        lower_difference = np.abs(source_tau - target_tau[lower])
        upper_difference = np.abs(source_tau - target_tau[upper])
        tolerance = 16.0 * np.finfo(np.float32).eps * np.maximum.reduce(
            (np.abs(source_tau), np.abs(target_tau[lower]), np.abs(target_tau[upper]), np.ones(height))
        )
        choose_upper = upper_difference + tolerance < lower_difference
        tied = np.abs(upper_difference - lower_difference) <= tolerance
        rows = np.arange(height)
        lower_distance = np.abs(lower - rows)
        upper_distance = np.abs(upper - rows)
        choose_upper |= tied & (upper_distance < lower_distance)
        equal = tied & (upper_distance == lower_distance) & (upper != lower)
        choose_upper = np.where(equal, (rows + column) % 2 == 0, choose_upper)
        selected = np.where(choose_upper, upper, lower)
        source_rows.append(rows)
        source_columns.append(np.full(height, column))
        target_rows.append(selected)
        valid_parts.append((source_tau >= target_tau[0]) & (source_tau <= target_tau[-1]))
    source_row = np.concatenate(source_rows)
    source_column = np.concatenate(source_columns)
    target_row = np.concatenate(target_rows)
    valid = np.concatenate(valid_parts)
    target_column = source_column + 1
    mismatch = np.abs(rgt[source_row, source_column] - rgt[target_row, target_column])
    shift = target_row - source_row
    return Topology(
        source_row,
        source_column,
        target_row,
        target_column,
        shift,
        mismatch,
        np.zeros_like(valid),
        np.zeros_like(valid),
        np.zeros_like(valid),
        valid,
        np.full_like(source_row, -1),
    )


def cartesian_topology(rgt: np.ndarray) -> Topology:
    height, width = rgt.shape
    rows, columns = np.meshgrid(np.arange(height), np.arange(width - 1), indexing="ij")
    mismatch = np.abs(rgt[rows, columns] - rgt[rows, columns + 1])
    zeros = np.zeros(rows.size, dtype=np.int64)
    return Topology(
        rows.ravel(), columns.ravel(), rows.ravel(), (columns + 1).ravel(), zeros,
        mismatch.ravel(), zeros.astype(bool), zeros.astype(bool), zeros.astype(bool),
        np.ones(rows.size, bool), zeros,
    )


def topology_for(
    name: str, rgt: np.ndarray, *, bounded_shift: int = 3, adaptive_shift: int = 3
) -> Topology:
    if name == LEGACY:
        return discrete_topology(rgt, max_shift=bounded_shift, tie_fixed=False)
    if name == TIE_FIXED:
        return discrete_topology(rgt, max_shift=bounded_shift, tie_fixed=True)
    if name == ADAPTIVE:
        return discrete_topology(
            rgt, max_shift=adaptive_shift, tie_fixed=True, adaptive_margin=1
        )
    if name == INVERSE:
        return inverse_rgt_topology(rgt)
    if name == CARTESIAN:
        return cartesian_topology(rgt)
    raise ValueError(name)


def fault_masks(
    topology: Topology, faults: Iterable[dict[str, float]], *, corridor: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    near = np.zeros(topology.source_row.shape, dtype=bool)
    crossing = np.zeros_like(near)
    for fault in faults:
        column = float(fault["column"])
        dip = float(fault["dip"])
        source_boundary = column + dip * topology.source_row
        target_boundary = column + dip * topology.target_row
        source_side = topology.source_column > source_boundary
        target_side = topology.target_column > target_boundary
        crossing |= source_side != target_side
        near |= np.minimum(
            np.abs(topology.source_column - source_boundary),
            np.abs(topology.target_column - target_boundary),
        ) <= corridor
    return near, crossing


def structural_fields(rgt: np.ndarray) -> dict[str, np.ndarray]:
    vertical, lateral = np.gradient(rgt.astype(np.float64))
    dip = np.abs(lateral) / np.maximum(np.abs(vertical), 1e-8)
    curvature = np.abs(np.gradient(lateral, axis=1))
    return {"dip": dip, "curvature": curvature}


def edge_observables(rgt: np.ndarray, topology: Topology) -> dict[str, np.ndarray]:
    """Return inference-available confidence features for lateral links."""
    rgt64 = np.asarray(rgt, dtype=np.float64)
    vertical = np.abs(np.gradient(rgt64, axis=0))
    positive_vertical = vertical[vertical > 16.0 * np.finfo(np.float32).eps]
    reference_step = float(np.median(positive_vertical)) if positive_vertical.size else 1.0
    source_scale = vertical[topology.source_row, topology.source_column]
    target_scale = vertical[topology.target_row, topology.target_column]
    local_scale = np.maximum(0.5 * (source_scale + target_scale), reference_step)
    cartesian_difference = np.abs(
        rgt64[topology.source_row, topology.source_column]
        - rgt64[topology.source_row, topology.target_column]
    )
    signed_lateral = (
        rgt64[topology.source_row, topology.target_column]
        - rgt64[topology.source_row, topology.source_column]
    )
    predicted_shift = -signed_lateral / np.maximum(source_scale, reference_step)
    return {
        "normalized_best_mismatch": topology.mismatch / local_scale,
        "absolute_displacement": np.abs(topology.shift).astype(np.float64),
        "normalized_cartesian_discontinuity": cartesian_difference / local_scale,
        "rgt_dip_residual": np.abs(topology.shift - predicted_shift),
    }


def confidence_block_mask(
    observables: dict[str, np.ndarray], thresholds: dict[str, float]
) -> np.ndarray:
    """Apply the predeclared interpretable V3 hard-blocking rule."""
    mismatch_failure = (
        observables["normalized_best_mismatch"]
        > thresholds["normalized_best_mismatch"]
    )
    discontinuity_and_inconsistency = (
        observables["normalized_cartesian_discontinuity"]
        > thresholds["normalized_cartesian_discontinuity"]
    ) & (observables["rgt_dip_residual"] > thresholds["rgt_dip_residual"])
    return mismatch_failure | discontinuity_and_inconsistency


def fit_structural_contract(dataset: Path, train_ids: Iterable[int]) -> dict[str, Any]:
    dip_values: list[np.ndarray] = []
    curvature_values: list[np.ndarray] = []
    inverse_displacements: list[np.ndarray] = []
    for realization_id in train_ids:
        with np.load(dataset / "realizations" / f"realization_{realization_id:07d}.npz") as archive:
            rgt = np.asarray(archive["rgt"], dtype=np.float32)
        fields = structural_fields(rgt)
        dip_values.append(fields["dip"].ravel())
        curvature_values.append(fields["curvature"].ravel())
        inverse = inverse_rgt_topology(rgt)
        inverse_displacements.append(np.abs(inverse.shift[inverse.valid]))
    dip = np.concatenate(dip_values)
    curvature = np.concatenate(curvature_values)
    displacement = np.concatenate(inverse_displacements)
    adaptive_cap = max(3, int(np.ceil(np.quantile(displacement, 0.995))))
    return {
        "dip_q33": float(np.quantile(dip, 1.0 / 3.0)),
        "dip_q67": float(np.quantile(dip, 2.0 / 3.0)),
        "curvature_q75": float(np.quantile(curvature, 0.75)),
        "adaptive_max_shift_samples": min(adaptive_cap, 24),
        "adaptive_cap_quantile": 0.995,
        "adaptive_cap_hard_safety_limit": 24,
    }


def strata_masks(
    rgt: np.ndarray,
    segmentation: np.ndarray,
    reservoir: np.ndarray,
    plume: np.ndarray,
    topology: Topology,
    faults: Iterable[dict[str, float]],
    contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    fields = structural_fields(rgt)
    dip = fields["dip"][topology.source_row, topology.source_column]
    curvature = fields["curvature"][topology.source_row, topology.source_column]
    near_fault, _ = fault_masks(topology, faults)
    source_seg = segmentation[topology.source_row, topology.source_column]
    target_seg = segmentation[topology.target_row, topology.target_column]
    return {
        "all": np.ones_like(topology.valid),
        "low_dip": dip < contract["dip_q33"],
        "medium_dip": (dip >= contract["dip_q33"]) & (dip < contract["dip_q67"]),
        "high_dip": dip >= contract["dip_q67"],
        "high_rgt_curvature": curvature >= contract["curvature_q75"],
        "reservoir": reservoir[topology.source_row, topology.source_column].astype(bool),
        "plume": plume[topology.source_row, topology.source_column].astype(bool),
        "facies_boundary": source_seg != target_seg,
        "fault_corridor": near_fault,
        "away_from_fault": ~near_fault,
    }


def _json_histogram(values: np.ndarray, support: Iterable[int]) -> str:
    return json.dumps({str(v): int(np.sum(values == v)) for v in support}, sort_keys=True)


def graph_connectivity(topology: Topology, height: int, width: int) -> dict[str, float | int]:
    keep = topology.valid
    source = topology.source_row[keep] * width + topology.source_column[keep]
    target = topology.target_row[keep] * width + topology.target_column[keep]
    vertical_source = np.arange((height - 1) * width)
    vertical_source = (vertical_source // width) * width + vertical_source % width
    vertical_target = vertical_source + width
    row = np.concatenate((source, target, vertical_source, vertical_target))
    column = np.concatenate((target, source, vertical_target, vertical_source))
    nodes = height * width
    adjacency = coo_matrix((np.ones(row.size), (row, column)), shape=(nodes, nodes)).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    degree = np.asarray((adjacency > 0).sum(axis=1)).ravel()
    sizes = np.bincount(labels)
    pairs = set(zip(row.tolist(), column.tolist()))
    symmetric = sum((b, a) in pairs for a, b in pairs) / max(len(pairs), 1)
    return {
        "degree_mean": float(degree.mean()),
        "degree_std": float(degree.std()),
        "degree_min": int(degree.min()),
        "degree_max": int(degree.max()),
        "connected_components": int(count),
        "largest_component_fraction": float(sizes.max() / nodes),
        "forward_reverse_symmetry": float(symmetric),
    }


def topology_summary_rows(
    *,
    topology: Topology,
    topology_name: str,
    split: str,
    realization_id: int,
    rgt: np.ndarray,
    masks: dict[str, np.ndarray],
    fault_crossing: np.ndarray,
) -> list[dict[str, Any]]:
    connectivity = graph_connectivity(topology, *rgt.shape)
    rows: list[dict[str, Any]] = []
    support = range(int(topology.shift.min()), int(topology.shift.max()) + 1)
    for stratum, mask in masks.items():
        selected = mask & topology.valid
        count = int(selected.sum())
        if count == 0:
            continue
        shift = topology.shift[selected]
        mismatch = topology.mismatch[selected]
        tied = topology.tied[selected]
        row: dict[str, Any] = {
            "split": split,
            "realization_id": realization_id,
            "topology": topology_name,
            "stratum": stratum,
            "edge_count_one_direction": count,
            "valid_fraction": float(count / max(int(mask.sum()), 1)),
            "tie_minimum_fraction": float(tied.mean()),
            "exact_tie_minimum_fraction": float(topology.exact_tied[selected].mean()),
            "selected_shift_histogram": _json_histogram(shift, support),
            "tied_selected_shift_histogram": _json_histogram(shift[tied], support),
            "search_boundary_hit_fraction": float(topology.boundary_hit[selected].mean()),
            "absolute_shift_mean": float(np.abs(shift).mean()),
            "absolute_shift_p50": float(np.quantile(np.abs(shift), 0.5)),
            "absolute_shift_p95": float(np.quantile(np.abs(shift), 0.95)),
            "absolute_shift_p99": float(np.quantile(np.abs(shift), 0.99)),
            "rgt_mismatch_mean": float(mismatch.mean()),
            "rgt_mismatch_std": float(mismatch.std()),
            "rgt_mismatch_p50": float(np.quantile(mismatch, 0.5)),
            "rgt_mismatch_p95": float(np.quantile(mismatch, 0.95)),
            "rgt_mismatch_p99": float(np.quantile(mismatch, 0.99)),
            "fault_crossing_fraction": float(fault_crossing[selected].mean()),
        }
        if stratum == "all":
            row.update(connectivity)
        rows.append(row)
    return rows


def edge_jaccard(left: Topology, right: Topology) -> float:
    left_pairs = set(
        zip(
            (left.source_row[left.valid] * 10000 + left.source_column[left.valid]).tolist(),
            (left.target_row[left.valid] * 10000 + left.target_column[left.valid]).tolist(),
        )
    )
    right_pairs = set(
        zip(
            (right.source_row[right.valid] * 10000 + right.source_column[right.valid]).tolist(),
            (right.target_row[right.valid] * 10000 + right.target_column[right.valid]).tolist(),
        )
    )
    return len(left_pairs & right_pairs) / max(len(left_pairs | right_pairs), 1)


def load_realization(dataset: Path, realization_id: int) -> dict[str, np.ndarray]:
    path = dataset / "realizations" / f"realization_{realization_id:07d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_faults(stage02: Path, realization_id: int) -> list[dict[str, float]]:
    sidecar = stage02 / f"realization_{realization_id:07d}.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return list(payload["geology"]["deformation"]["faults"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
