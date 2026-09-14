"""Physical seismic events associated by continuous native RGT geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, hilbert

from sage_avo.diagnostics.native_rgt_graph import NativeRGT
from sage_avo.diagnostics.skeleton_graph import graph_statistics, sample


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def _interpolate_channels(array: np.ndarray, time: float, trace: int) -> np.ndarray:
    axis = np.arange(array.shape[1], dtype=float)
    return np.asarray([np.interp(time, axis, channel[:, trace]) for channel in array])


def _pgc(amplitudes: np.ndarray) -> tuple[float, float, float]:
    angle = np.sin(np.deg2rad([10.0, 24.0, 38.0])) ** 2
    centered = angle - angle.mean()
    gradient = float(
        ((amplitudes - amplitudes.mean()) * centered).sum() / max((centered**2).sum(), 1e-12)
    )
    intercept = float(amplitudes.mean() - gradient * angle.mean())
    curvature = float(amplitudes[0] - 2 * amplitudes[1] + amplitudes[2])
    return intercept, gradient, curvature


def detect_physical_events(
    avo: np.ndarray,
    tau: np.ndarray,
    valid: np.ndarray,
    config: Mapping[str, Any],
    *,
    channels: Sequence[int] = (0, 1, 2),
) -> list[dict[str, Any]]:
    """Detect phase-stable signed-amplitude extrema directly in physical time."""
    avo = np.asarray(avo, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if avo.shape != (3, *tau.shape) or valid.shape != tau.shape:
        raise ValueError("Require AVA[3,H,W], RGT[H,W], and valid[H,W]")
    selected = avo[np.asarray(channels, dtype=int)]
    scales = np.quantile(np.abs(selected[:, valid]), 0.75, axis=1)
    normalized = selected / np.maximum(scales[:, None, None], 1e-8)
    analytic = hilbert(avo, axis=1)
    selected_analytic = hilbert(normalized, axis=1)
    envelope = np.sqrt(np.mean(np.abs(selected_analytic) ** 2, axis=0))
    evidence = gaussian_filter1d(envelope, sigma=0.75, axis=0, mode="nearest")
    stack = normalized.mean(axis=0)
    phase_concentration = np.abs(np.mean(np.exp(1j * np.angle(selected_analytic)), axis=0))
    threshold = float(np.quantile(evidence[valid], config["event_quantile"]))
    prominence = config["event_prominence_fraction"] * float(np.quantile(evidence[valid], 0.95))
    native = NativeRGT(tau)
    events: list[dict[str, Any]] = []
    time_axis = np.arange(tau.shape[0], dtype=float)
    radius = int(config["event_refinement_radius"])
    wave_radius = int(config["event_waveform_radius"])
    for trace in range(tau.shape[1]):
        peaks, properties = find_peaks(
            np.where(valid[:, trace], evidence[:, trace], 0.0),
            height=threshold,
            prominence=prominence,
            distance=int(config["minimum_distance"]),
        )
        if len(peaks) > config["maximum_events_per_trace"]:
            order = np.argsort(properties["peak_heights"])[
                -int(config["maximum_events_per_trace"]) :
            ]
            peaks = np.sort(peaks[order])
        refined_indices: set[int] = set()
        for peak in peaks:
            start, stop = max(1, peak - radius), min(tau.shape[0] - 1, peak + radius + 1)
            score = np.abs(stack[start:stop, trace]) * phase_concentration[start:stop, trace]
            center = int(start + np.argmax(score))
            if center in refined_indices or not valid[center, trace]:
                continue
            refined_indices.add(center)
            left = abs(stack[center - 1, trace]) * phase_concentration[center - 1, trace]
            middle = abs(stack[center, trace]) * phase_concentration[center, trace]
            right = abs(stack[center + 1, trace]) * phase_concentration[center + 1, trace]
            denominator = left - 2 * middle + right
            offset = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
            event_time = float(center + np.clip(offset, -0.5, 0.5))
            concentration = float(np.interp(event_time, time_axis, phase_concentration[:, trace]))
            if concentration < config["minimum_phase_concentration"]:
                continue
            amplitudes = _interpolate_channels(avo, event_time, trace)
            phases = np.angle(_interpolate_channels(analytic, event_time, trace))
            event_envelope = float(
                np.sqrt(np.mean(np.abs(_interpolate_channels(analytic, event_time, trace)) ** 2))
            )
            p_value, g_value, c_value = _pgc(amplitudes)
            offsets = np.arange(-wave_radius, wave_radius + 1, dtype=float)
            waveform = np.concatenate(
                [
                    np.interp(event_time + offsets, time_axis, avo[channel, :, trace])
                    for channel in range(3)
                ]
            )
            events.append(
                {
                    "event": len(events),
                    "trace": trace,
                    "time": event_time,
                    "tau": float(native.forward_tau(trace, event_time)),
                    "a_near": float(amplitudes[0]),
                    "a_mid": float(amplitudes[1]),
                    "a_far": float(amplitudes[2]),
                    "envelope": event_envelope,
                    "phase_near": float(phases[0]),
                    "phase_mid": float(phases[1]),
                    "phase_far": float(phases[2]),
                    "phase_concentration": concentration,
                    "p": p_value,
                    "g": g_value,
                    "c": c_value,
                    "waveform": waveform,
                }
            )
    return events


def event_repeatability(
    reference: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    """Symmetric same-trace repeatability under an observable angle perturbation."""
    by_trace: dict[int, list[float]] = {}
    for event in comparison:
        by_trace.setdefault(int(event["trace"]), []).append(float(event["time"]))
    errors = []
    for event in reference:
        candidates = by_trace.get(int(event["trace"]), [])
        if candidates:
            errors.append(min(abs(float(event["time"]) - value) for value in candidates))
    matched_errors = [error for error in errors if error <= tolerance]
    forward_fraction = len(matched_errors) / max(len(reference), 1)
    by_reference: dict[int, list[float]] = {}
    for event in reference:
        by_reference.setdefault(int(event["trace"]), []).append(float(event["time"]))
    reverse_errors = []
    for event in comparison:
        candidates = by_reference.get(int(event["trace"]), [])
        if candidates:
            reverse_errors.append(min(abs(float(event["time"]) - value) for value in candidates))
    reverse_fraction = sum(error <= tolerance for error in reverse_errors) / max(len(comparison), 1)
    return {
        "repeatability": float(0.5 * (forward_fraction + reverse_fraction)),
        # Localization error and missing-event repeatability are distinct. Do
        # not turn a missing event into an artificial multi-sample position error.
        "error_p50": float(np.quantile(matched_errors, 0.50)) if matched_errors else float("inf"),
        "error_p95": float(np.quantile(matched_errors, 0.95)) if matched_errors else float("inf"),
        "reference_count": len(reference),
        "comparison_count": len(comparison),
    }


def candidate_event_pairs(
    native: NativeRGT,
    events: list[dict[str, Any]],
    search_radius: float,
) -> list[dict[str, Any]]:
    """Enumerate observable targets around a native-RGT physical prediction."""
    by_trace: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        by_trace.setdefault(int(event["trace"]), []).append(index)
    fields = np.stack(np.gradient(native.tau))
    rows = []
    for trace in range(native.tau.shape[1] - 1):
        for source in by_trace.get(trace, []):
            left = events[source]
            predicted_time = float(native.inverse_time(trace + 1, left["tau"]))
            local_step = abs(
                native.forward_tau(trace, left["time"] + 0.5)
                - native.forward_tau(trace, left["time"] - 0.5)
            )
            for target in by_trace.get(trace + 1, []):
                right = events[target]
                time_residual = abs(right["time"] - predicted_time)
                if time_residual > search_radius:
                    continue
                phase_left = np.asarray([left["phase_near"], left["phase_mid"], left["phase_far"]])
                phase_right = np.asarray(
                    [right["phase_near"], right["phase_mid"], right["phase_far"]]
                )
                ava_left = np.asarray([left["a_near"], left["a_mid"], left["a_far"]])
                ava_right = np.asarray([right["a_near"], right["a_mid"], right["a_far"]])
                representative_tau = 0.5 * (left["tau"] + right["tau"])
                boundaries = range(max(0, trace - 1), min(native.tau.shape[1] - 1, trace + 2))
                shifts = np.asarray(
                    [
                        native.inverse_time(boundary + 1, representative_tau)
                        - native.inverse_time(boundary, representative_tau)
                        for boundary in boundaries
                    ]
                )
                center_index = 0 if trace == 0 else 1
                physical_shift = float(shifts[center_index])
                shift_jump = float(abs(physical_shift - np.median(shifts)))
                second = (
                    float(abs(shifts[2] - 2 * shifts[1] + shifts[0])) if len(shifts) == 3 else 0.0
                )
                left_point = np.asarray([[left["time"], trace]])
                right_point = np.asarray([[right["time"], trace + 1]])
                gradients_left = sample(fields, left_point)[0]
                gradients_right = sample(fields, right_point)[0]
                dip_left = -gradients_left[1] / max(abs(gradients_left[0]), 1e-12)
                dip_right = -gradients_right[1] / max(abs(gradients_right[0]), 1e-12)
                curvature = structural_curvature(native.tau, left_point)[0]
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "boundary": trace,
                        "predicted_time": predicted_time,
                        "d_tau": float(abs(right["tau"] - left["tau"]) / max(local_step, 1e-12)),
                        "d_time": float(time_residual),
                        "waveform_cosine": _cosine(left["waveform"], right["waveform"]),
                        "phase_cosine": float(np.mean(np.cos(phase_left - phase_right))),
                        "ava_cosine": _cosine(ava_left, ava_right),
                        "physical_shift": physical_shift,
                        "shift_jump": shift_jump,
                        "shift_second_difference": second,
                        "dip_difference": float(abs(dip_right - dip_left)),
                        "source_dip": float(abs(dip_left)),
                        "source_curvature": float(curvature),
                    }
                )
    return rows


def structural_curvature(tau: np.ndarray, points: np.ndarray) -> np.ndarray:
    lateral = np.gradient(np.asarray(tau, dtype=float), axis=1)
    return sample(np.abs(np.gradient(lateral, axis=1)), points)


def robust_match_scales(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Freeze robust positive cost scales using training candidate pairs."""
    if not rows:
        raise ValueError("No training candidate pairs")
    names = ["d_tau", "d_time", "waveform_penalty", "phase_penalty", "ava_penalty"]
    values = {
        "d_tau": [row["d_tau"] for row in rows],
        "d_time": [row["d_time"] for row in rows],
        "waveform_penalty": [1 - row["waveform_cosine"] for row in rows],
        "phase_penalty": [1 - row["phase_cosine"] for row in rows],
        "ava_penalty": [1 - row["ava_cosine"] for row in rows],
    }
    return {name: float(max(np.quantile(np.asarray(values[name]), 0.75), 1e-6)) for name in names}


def reciprocal_matches(
    rows: list[dict[str, Any]],
    scales: Mapping[str, float],
    weights: Mapping[str, float],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply one fixed interpretable cost and reciprocal/no-forcing rule."""
    result = []
    for row in rows:
        cost = (
            weights["tau"] * row["d_tau"] / scales["d_tau"]
            + weights["time"] * row["d_time"] / scales["d_time"]
            + weights["waveform"] * (1 - row["waveform_cosine"]) / scales["waveform_penalty"]
            + weights["phase"] * (1 - row["phase_cosine"]) / scales["phase_penalty"]
            + weights["ava"] * (1 - row["ava_cosine"]) / scales["ava_penalty"]
        )
        result.append({**row, "match_cost": float(cost)})
    for row in result:
        source_options = [item for item in result if item["source"] == row["source"]]
        target_options = [item for item in result if item["target"] == row["target"]]
        best_source = min(source_options, key=lambda item: (item["match_cost"], item["target"]))
        best_target = min(target_options, key=lambda item: (item["match_cost"], item["source"]))
        reciprocal = row is best_source and row is best_target
        row["reciprocal"] = reciprocal
        row["reflector_match"] = bool(
            reciprocal
            and row["match_cost"] <= config["maximum_normalized_match_cost"]
            and row["waveform_cosine"] >= config["minimum_match_waveform_cosine"]
            and row["phase_cosine"] >= config["minimum_match_phase_cosine"]
            and row["ava_cosine"] >= config["minimum_match_ava_cosine"]
        )
    return result


def freeze_event_barriers(
    rows: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, float]:
    """Freeze observable barrier thresholds from training reflector matches."""
    matched = [row for row in rows if row["reflector_match"]]
    if not matched:
        raise ValueError("No training reflector matches")

    def high(name: str) -> float:
        values = np.asarray([row[name] for row in matched])
        median = np.median(values)
        mad = 1.4826 * np.median(abs(values - median))
        robust = median + config["barrier_robust_sigma"] * max(mad, 1e-8)
        return float(
            max(
                np.quantile(values, 0.95),
                min(np.quantile(values, config["barrier_quantile_cap"]), robust),
            )
        )

    def low(name: str, floor: float) -> float:
        return float(max(floor, np.quantile([row[name] for row in matched], 0.01)))

    return {
        "shift_jump": high("shift_jump"),
        "shift_second_difference": high("shift_second_difference"),
        "dip_difference": high("dip_difference"),
        "waveform_cosine": low("waveform_cosine", config["minimum_barrier_waveform_cosine"]),
        "phase_cosine": low("phase_cosine", config["minimum_barrier_phase_cosine"]),
        "ava_cosine": low("ava_cosine", config["minimum_barrier_ava_cosine"]),
    }


def build_event_graph(
    events: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    thresholds: Mapping[str, float],
    config: Mapping[str, Any],
    *,
    node_spacing: int,
    curvature_threshold: float,
) -> dict[str, Any]:
    """Apply barriers before union, subsample components, and form safe long edges."""
    parent = list(range(len(events)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    safe_pairs: dict[tuple[int, int], dict[str, Any]] = {}
    metadata = []
    for row in matches:
        barrier = bool(
            row["shift_jump"] > thresholds["shift_jump"]
            or row["shift_second_difference"] > thresholds["shift_second_difference"]
            or row["dip_difference"] > thresholds["dip_difference"]
            or row["waveform_cosine"] < thresholds["waveform_cosine"]
            or row["phase_cosine"] < thresholds["phase_cosine"]
            or row["ava_cosine"] < thresholds["ava_cosine"]
        )
        row["fault_barrier"] = bool(row["reflector_match"] and barrier)
        row["barrier_safe"] = bool(row["reflector_match"] and not barrier)
        if row["barrier_safe"]:
            left, right = root(row["source"]), root(row["target"])
            parent[max(left, right)] = min(left, right)
            safe_pairs[(row["source"], row["target"])] = row
        elif row["fault_barrier"]:
            metadata.append({**row, "relation": "FAULT_OFFSET_CORRESPONDENCE"})
    raw_groups: dict[int, list[int]] = {}
    for index in range(len(events)):
        raw_groups.setdefault(root(index), []).append(index)
    groups = [
        sorted(group, key=lambda index: events[index]["trace"]) for group in raw_groups.values()
    ]
    groups = [
        group
        for group in groups
        if len(group) >= config["minimum_component_points"]
        and events[group[-1]]["trace"] - events[group[0]]["trace"]
        >= config["minimum_component_trace_span"]
    ]
    groups.sort(key=lambda group: (events[group[0]]["trace"], events[group[0]]["time"]))
    nodes = []
    lookup: dict[int, int] = {}
    selected_by_component: dict[int, list[int]] = {}
    for component, group in enumerate(groups):
        selected = [group[0]]
        for index in group[1:-1]:
            if (
                events[index]["trace"] - events[selected[-1]]["trace"] >= node_spacing
                or events[index].get("structural_curvature", 0.0) >= curvature_threshold
            ):
                selected.append(index)
        if group[-1] != selected[-1]:
            selected.append(group[-1])
        selected_by_component[component] = selected
        for index in selected:
            node = len(nodes)
            lookup[index] = node
            nodes.append({"node": node, "component": component, **events[index]})
    edges = []
    for component, group in enumerate(groups):
        by_trace = {int(events[index]["trace"]): index for index in group}
        selected_by_trace = {
            int(events[index]["trace"]): index for index in selected_by_component[component]
        }
        for trace, source in selected_by_trace.items():
            for span in config["edge_spans_traces"]:
                target = selected_by_trace.get(trace + span)
                if target is None or not all(
                    position in by_trace
                    and position + 1 in by_trace
                    and (by_trace[position], by_trace[position + 1]) in safe_pairs
                    for position in range(trace, trace + span)
                ):
                    continue
                path_ids = [by_trace[position] for position in range(trace, trace + span + 1)]
                path = np.asarray(
                    [[events[index]["time"], events[index]["trace"]] for index in path_ids]
                )
                tau_path = np.asarray([events[index]["tau"] for index in path_ids])
                adjacent = [
                    safe_pairs[(path_ids[offset], path_ids[offset + 1])] for offset in range(span)
                ]
                slopes = np.diff(path[:, 0])
                curvature = np.diff(slopes)
                edges.append(
                    {
                        "edge": len(edges),
                        "source": lookup[source],
                        "target": lookup[target],
                        "component": component,
                        "span": span,
                        "delta_tau": float(tau_path[-1] - tau_path[0]),
                        "delta_t": float(path[-1, 0] - path[0, 0]),
                        "delta_x": span,
                        "geodesic_distance": float(
                            np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
                        ),
                        "waveform_similarity": float(
                            min(row["waveform_cosine"] for row in adjacent)
                        ),
                        "phase_similarity": float(min(row["phase_cosine"] for row in adjacent)),
                        "dip_difference": float(slopes[-1] - slopes[0]) if len(slopes) > 1 else 0.0,
                        "curvature_mean": float(np.mean(abs(curvature))) if len(curvature) else 0.0,
                        "path": path,
                    }
                )
    points = np.asarray([[node["time"], node["trace"]] for node in nodes]).reshape(-1, 2)
    pairs = np.asarray([[edge["source"], edge["target"]] for edge in edges]).reshape(-1, 2)
    return {
        "nodes": nodes,
        "edges": edges,
        "components": groups,
        "matches": matches,
        "metadata": metadata,
        "statistics": graph_statistics(points, pairs),
    }


def normal_event_contrasts(
    tau: np.ndarray, avo: np.ndarray, nodes: list[dict[str, Any]], offset: float
) -> list[dict[str, Any]]:
    """Sample local AVA/P/G/C contrast across the same physical reflector."""
    gradients = np.stack(np.gradient(np.asarray(tau, dtype=float)))
    rows = []
    for node in nodes:
        point = np.asarray([node["time"], node["trace"]], dtype=float)
        gradient = sample(gradients, point[None])[0]
        direction = gradient / max(np.linalg.norm(gradient), 1e-12)
        above, below = point - offset * direction, point + offset * direction
        upper = sample(avo, above[None])[0]
        lower = sample(avo, below[None])[0]
        delta = lower - upper
        p_value, g_value, c_value = _pgc(delta)
        rows.append(
            {
                "node": node["node"],
                "component": node["component"],
                "delta_ava_near": float(delta[0]),
                "delta_ava_mid": float(delta[1]),
                "delta_ava_far": float(delta[2]),
                "delta_p": p_value,
                "delta_g": g_value,
                "delta_c": c_value,
            }
        )
    return rows
