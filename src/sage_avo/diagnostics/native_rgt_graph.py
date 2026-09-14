"""Native trace-wise RGT geometry and pre-union dual-domain barriers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.signal import hilbert

from sage_avo.diagnostics.skeleton_graph import graph_statistics, sample


class NativeRGT:
    """Piecewise-monotone per-trace RGT inverse; plateaus remain ambiguous."""

    def __init__(self, tau: np.ndarray) -> None:
        tau = np.asarray(tau, dtype=np.float64)
        if tau.ndim != 2 or not np.isfinite(tau).all() or np.any(np.diff(tau, axis=0) < 0):
            raise ValueError("RGT must be finite, 2-D, and nondecreasing")
        self.tau = tau
        self.tau_knots: list[np.ndarray] = []
        self.time_knots: list[np.ndarray] = []
        self.identifiable = np.zeros_like(tau, dtype=bool)
        times = np.arange(tau.shape[0], dtype=np.float64)
        for trace in range(tau.shape[1]):
            values, inverse, counts = np.unique(
                tau[:, trace], return_inverse=True, return_counts=True
            )
            # The mean is a deterministic generalized inverse at a plateau. Exact
            # plateau samples remain marked ambiguous and are excluded from QC.
            sums = np.bincount(inverse, weights=times)
            self.tau_knots.append(values)
            self.time_knots.append(sums / counts)
            self.identifiable[:, trace] = counts[inverse] == 1

    def inverse_time(self, trace: int, tau_query: Any) -> np.ndarray:
        """Evaluate t_x(tau) directly on native knots."""
        trace = int(trace)
        return np.interp(tau_query, self.tau_knots[trace], self.time_knots[trace])

    def forward_tau(self, trace: int, time_query: Any) -> np.ndarray:
        """Evaluate tau_x(t) directly on original samples."""
        return np.interp(
            time_query,
            np.arange(self.tau.shape[0], dtype=np.float64),
            self.tau[:, int(trace)],
        )

    def roundtrip(self) -> tuple[np.ndarray, np.ndarray]:
        """Return time and tau errors only at identifiable native samples."""
        time_error: list[float] = []
        tau_error: list[float] = []
        for trace in range(self.tau.shape[1]):
            mask = self.identifiable[:, trace]
            times = np.flatnonzero(mask).astype(np.float64)
            values = self.tau[mask, trace]
            reconstructed_time = self.inverse_time(trace, values)
            time_error.extend(np.abs(reconstructed_time - times))
            reconstructed_tau = self.forward_tau(trace, reconstructed_time)
            tau_error.extend(np.abs(reconstructed_tau - values))
        return np.asarray(time_error), np.asarray(tau_error)


def _normal_waveform(
    avo: np.ndarray, tau: np.ndarray, point: np.ndarray, radius: int = 2
) -> np.ndarray | None:
    gradient = sample(np.stack(np.gradient(tau.astype(float))), point[None])[0]
    direction = gradient / max(np.linalg.norm(gradient), 1e-12)
    locations = point[None] + np.arange(-radius, radius + 1)[:, None] * direction
    if (
        np.any(locations < 0)
        or np.any(locations[:, 0] > tau.shape[0] - 1)
        or np.any(locations[:, 1] > tau.shape[1] - 1)
    ):
        return None
    return sample(avo, locations).ravel()


def refine_candidates(
    native: NativeRGT,
    avo: np.ndarray,
    mapping: Mapping[str, np.ndarray],
    detection: Mapping[str, Any],
    width_steps: float,
    samples: int = 17,
) -> list[dict[str, Any]]:
    """Refine coarse envelope ridges continuously in native tau geometry."""
    avo = np.asarray(avo, dtype=np.float64)
    envelope = np.sqrt(np.mean(np.abs(hilbert(avo, axis=1)) ** 2, axis=0))
    tau_grid = np.asarray(mapping["tau_grid"])
    step = float(np.median(np.diff(tau_grid)))
    rows = []
    for candidate in detection["candidates"]:
        trace = int(candidate["trace"])
        tau_index = int(candidate["tau_index"])
        coarse_tau = float(tau_grid[tau_index])
        queries = np.linspace(
            coarse_tau - width_steps * step,
            coarse_tau + width_steps * step,
            samples,
        )
        queries = np.clip(queries, native.tau_knots[trace][0], native.tau_knots[trace][-1])
        times = native.inverse_time(trace, queries)
        strengths = np.interp(
            times, np.arange(native.tau.shape[0], dtype=float), envelope[:, trace]
        )
        best = int(np.argmax(strengths))
        refined_tau = float(queries[best])
        refined_time = float(times[best])
        coarse_time = float(native.inverse_time(trace, coarse_tau))
        rows.append(
            {
                **candidate,
                "coarse_tau": coarse_tau,
                "refined_tau": refined_tau,
                "coarse_time": coarse_time,
                "refined_time": refined_time,
                "time_displacement": refined_time - coarse_time,
                "coarse_strength": float(
                    np.interp(
                        coarse_time,
                        np.arange(native.tau.shape[0], dtype=float),
                        envelope[:, trace],
                    )
                ),
                "refined_strength": float(strengths[best]),
                "refinement_at_boundary": best in (0, len(queries) - 1),
            }
        )
    return rows


def link_observables(
    native: NativeRGT,
    avo: np.ndarray,
    refined: list[dict[str, Any]],
    mapping: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build refined reciprocal matches and native physical barrier evidence."""
    by_trace: dict[int, list[int]] = {}
    for index, row in enumerate(refined):
        by_trace.setdefault(int(row["trace"]), []).append(index)
    avo = np.asarray(avo, dtype=np.float64)
    analytic = hilbert(avo, axis=1)
    tau_step = float(np.median(np.diff(mapping["tau_grid"])))
    possible: list[dict[str, Any]] = []
    for trace in range(native.tau.shape[1] - 1):
        for source in by_trace.get(trace, []):
            for target in by_trace.get(trace + 1, []):
                left, right = refined[source], refined[target]
                delta_tau_steps = abs(right["refined_tau"] - left["refined_tau"]) / tau_step
                if delta_tau_steps > config["maximum_refined_tau_steps"]:
                    continue
                left_point = np.array([left["refined_time"], trace])
                right_point = np.array([right["refined_time"], trace + 1])
                left_wave = _normal_waveform(avo, native.tau, left_point)
                right_wave = _normal_waveform(avo, native.tau, right_point)
                if left_wave is None or right_wave is None:
                    continue
                waveform = float(
                    left_wave
                    @ right_wave
                    / max(np.linalg.norm(left_wave) * np.linalg.norm(right_wave), 1e-12)
                )
                left_complex = np.array(
                    [
                        np.interp(
                            left_point[0],
                            np.arange(native.tau.shape[0]),
                            analytic[channel, :, trace],
                        )
                        for channel in range(3)
                    ]
                )
                right_complex = np.array(
                    [
                        np.interp(
                            right_point[0],
                            np.arange(native.tau.shape[0]),
                            analytic[channel, :, trace + 1],
                        )
                        for channel in range(3)
                    ]
                )
                phase = float(np.mean(np.cos(np.angle(left_complex) - np.angle(right_complex))))
                strength_ratio = min(left["refined_strength"], right["refined_strength"]) / max(
                    left["refined_strength"], right["refined_strength"], 1e-12
                )
                representative_tau = 0.5 * (left["refined_tau"] + right["refined_tau"])
                boundaries = range(max(0, trace - 1), min(native.tau.shape[1] - 1, trace + 2))
                shifts = np.asarray(
                    [
                        native.inverse_time(boundary + 1, representative_tau)
                        - native.inverse_time(boundary, representative_tau)
                        for boundary in boundaries
                    ],
                    dtype=float,
                )
                center_index = 0 if trace == 0 else 1
                physical_shift = float(shifts[center_index])
                shift_jump = float(abs(physical_shift - np.median(shifts)))
                second_difference = (
                    float(abs(shifts[2] - 2 * shifts[1] + shifts[0])) if len(shifts) == 3 else 0.0
                )
                cost = delta_tau_steps + 0.5 * (1 - waveform) + 0.25 * (1 - phase)
                possible.append(
                    {
                        "source": source,
                        "target": target,
                        "boundary": trace,
                        "delta_tau_steps": float(delta_tau_steps),
                        "original_waveform_cosine": waveform,
                        "original_phase_cosine": phase,
                        "strength_ratio": float(strength_ratio),
                        "physical_shift": physical_shift,
                        "shift_jump": shift_jump,
                        "shift_second_difference": second_difference,
                        "cost": float(cost),
                    }
                )
    for row in possible:
        same_source = [candidate for candidate in possible if candidate["source"] == row["source"]]
        same_target = [candidate for candidate in possible if candidate["target"] == row["target"]]
        best_source = min(same_source, key=lambda item: (item["cost"], item["target"]))
        best_target = min(same_target, key=lambda item: (item["cost"], item["source"]))
        row["reciprocal"] = row is best_source and row is best_target
    return possible


def freeze_thresholds(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, float]:
    """Freeze observable-only robust thresholds from training links."""
    reciprocal = [row for row in rows if row["reciprocal"]]
    if not reciprocal:
        raise ValueError("Cannot freeze thresholds without reciprocal training links")

    def robust_high(name: str) -> float:
        values = np.asarray([row[name] for row in reciprocal], dtype=float)
        median = np.median(values)
        mad = 1.4826 * np.median(np.abs(values - median))
        robust_limit = median + config["barrier_robust_sigma"] * max(mad, 1e-8)
        # A zero-inflated shift distribution can have median=MAD=0 despite a
        # legitimate smooth nonzero tail. Never let that degeneracy turn
        # floating-point noise into a geological barrier: retain at least the
        # observable 95th percentile while retaining the declared upper cap.
        lower_quantile = np.quantile(values, 0.95)
        upper_quantile = np.quantile(values, config["barrier_quantile_cap"])
        return float(max(lower_quantile, min(upper_quantile, robust_limit)))

    def robust_low(name: str, floor: float) -> float:
        values = np.asarray([row[name] for row in reciprocal], dtype=float)
        return float(max(floor, np.quantile(values, 0.01)))

    return {
        "shift_jump": robust_high("shift_jump"),
        "shift_second_difference": robust_high("shift_second_difference"),
        "original_waveform_cosine": robust_low(
            "original_waveform_cosine",
            config["minimum_original_waveform_cosine_floor"],
        ),
        "original_phase_cosine": robust_low(
            "original_phase_cosine", config["minimum_original_phase_cosine_floor"]
        ),
        "strength_ratio": robust_low("strength_ratio", config["minimum_strength_ratio_floor"]),
        "high_shift_magnitude": float(
            np.quantile(np.abs([row["physical_shift"] for row in reciprocal]), 2 / 3)
        ),
    }


def build_barrier_graph(
    refined: list[dict[str, Any]],
    links: list[dict[str, Any]],
    thresholds: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the observable barrier before union and build safe-path long edges."""
    parent = list(range(len(refined)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    safe_pairs: dict[tuple[int, int], dict[str, Any]] = {}
    metadata = []
    for row in links:
        reflector_match = bool(row["reciprocal"])
        barrier = bool(
            row["shift_jump"] > thresholds["shift_jump"]
            or row["shift_second_difference"] > thresholds["shift_second_difference"]
            or row["original_waveform_cosine"] < thresholds["original_waveform_cosine"]
            or row["original_phase_cosine"] < thresholds["original_phase_cosine"]
            or row["strength_ratio"] < thresholds["strength_ratio"]
        )
        row["reflector_match"] = reflector_match
        row["fault_barrier"] = bool(reflector_match and barrier)
        row["barrier_safe"] = bool(reflector_match and not barrier)
        if row["barrier_safe"]:
            left, right = root(row["source"]), root(row["target"])
            parent[max(left, right)] = min(left, right)
            safe_pairs[(row["source"], row["target"])] = row
        elif row["fault_barrier"]:
            metadata.append({**row, "relation": "FAULT_OFFSET_CORRESPONDENCE"})

    raw_groups: dict[int, list[int]] = {}
    for index in range(len(refined)):
        raw_groups.setdefault(root(index), []).append(index)
    groups = [
        sorted(group, key=lambda index: refined[index]["trace"]) for group in raw_groups.values()
    ]
    groups = [
        group
        for group in groups
        if len(group) >= config["minimum_component_points"]
        and refined[group[-1]]["trace"] - refined[group[0]]["trace"]
        >= config["minimum_component_trace_span"]
    ]
    groups.sort(
        key=lambda group: (
            refined[group[0]]["trace"],
            refined[group[0]]["refined_tau"],
        )
    )

    nodes: list[dict[str, Any]] = []
    lookup: dict[int, int] = {}
    for component, group in enumerate(groups):
        for index in group:
            node = len(nodes)
            lookup[index] = node
            nodes.append({"node": node, "component": component, **refined[index]})

    edges = []
    for component, group in enumerate(groups):
        by_trace = {int(refined[index]["trace"]): index for index in group}
        for trace, source in by_trace.items():
            for span in config["edge_spans_traces"]:
                target = by_trace.get(trace + span)
                if target is None or not all(
                    (by_trace[position], by_trace[position + 1]) in safe_pairs
                    for position in range(trace, trace + span)
                ):
                    continue
                path_ids = [by_trace[position] for position in range(trace, trace + span + 1)]
                path = np.asarray(
                    [
                        [refined[index]["refined_time"], refined[index]["trace"]]
                        for index in path_ids
                    ],
                    dtype=float,
                )
                tau_path = np.asarray(
                    [refined[index]["refined_tau"] for index in path_ids], dtype=float
                )
                slope = np.diff(path[:, 0])
                curvature = np.diff(slope)
                adjacent = [
                    safe_pairs[(path_ids[offset], path_ids[offset + 1])] for offset in range(span)
                ]
                edges.append(
                    {
                        "edge": len(edges),
                        "source": lookup[source],
                        "target": lookup[target],
                        "component": component,
                        "span": span,
                        "delta_tau": float(tau_path[-1] - tau_path[0]),
                        "delta_x": span,
                        "delta_t": float(path[-1, 0] - path[0, 0]),
                        "cartesian_length": float(np.linalg.norm(path[-1] - path[0])),
                        "geodesic_length": float(
                            np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                        ),
                        "dip_difference": float(slope[-1] - slope[0]) if len(slope) > 1 else 0.0,
                        "curvature_mean": float(np.mean(np.abs(curvature)))
                        if len(curvature)
                        else 0.0,
                        "curvature_max": float(np.max(np.abs(curvature)))
                        if len(curvature)
                        else 0.0,
                        "reflector_continuity": float(
                            min(row["original_waveform_cosine"] for row in adjacent)
                        ),
                        "path": path,
                    }
                )

    points = np.asarray(
        [[node["refined_time"], node["trace"]] for node in nodes], dtype=float
    ).reshape(-1, 2)
    pairs = np.asarray([[edge["source"], edge["target"]] for edge in edges], dtype=int).reshape(
        -1, 2
    )
    statistics = graph_statistics(points, pairs)
    return {
        "nodes": nodes,
        "edges": edges,
        "links": links,
        "metadata": metadata,
        "components": groups,
        "statistics": statistics,
    }
