"""Gap-tolerant physical seismic-event tracking with native-RGT geometry."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.signal import find_peaks, hilbert

from sage_avo.diagnostics.native_event_graph import (
    _pgc,
    detect_physical_events,
    normal_event_contrasts,
)
from sage_avo.diagnostics.native_rgt_graph import NativeRGT
from sage_avo.diagnostics.skeleton_graph import graph_statistics


def detect_events(
    avo: np.ndarray,
    tau: np.ndarray,
    valid: np.ndarray,
    strong_config: Mapping[str, Any],
    weak_config: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    channels: tuple[int, ...] = (0, 1, 2),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Merge v00332y strong events with local phase-consistent signed extrema."""
    strong = detect_physical_events(avo, tau, valid, strong_config, channels=channels)
    for row in strong:
        row["source_type"] = "strong"
    avo = np.asarray(avo, float)
    selected = avo[list(channels)]
    scales = np.quantile(np.abs(selected[:, valid]), 0.75, axis=1)
    normalized_analytic = hilbert(selected / np.maximum(scales[:, None, None], 1e-8), axis=1)
    analytic = hilbert(avo, axis=1)
    envelope = np.sqrt(np.mean(np.abs(normalized_analytic) ** 2, axis=0))
    phase = np.abs(np.mean(np.exp(1j * np.angle(normalized_analytic)), axis=0))
    signed = (selected / np.maximum(scales[:, None, None], 1e-8)).mean(axis=0)
    native = NativeRGT(tau)
    local_window = int(config["weak_local_window"])
    weak: list[dict[str, Any]] = []
    overlap = 0
    strong_by_trace: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in strong:
        strong_by_trace[int(row["trace"])].append(row)
    for x in range(tau.shape[1]):
        signal = signed[:, x]
        peaks = np.concatenate(
            [
                find_peaks(signal, distance=config["weak_minimum_distance"])[0],
                find_peaks(-signal, distance=config["weak_minimum_distance"])[0],
            ]
        )
        accepted = []
        for center in np.sort(peaks):
            if center < 3 or center >= tau.shape[0] - 3 or not valid[center, x]:
                continue
            lo, hi = (
                max(0, center - local_window // 2),
                min(tau.shape[0], center + local_window // 2 + 1),
            )
            local = envelope[lo:hi, x][valid[lo:hi, x]]
            if not len(local) or envelope[center, x] < np.quantile(
                local, weak_config["local_envelope_quantile"]
            ):
                continue
            prominence = abs(signal[center]) - min(abs(signal[center - 2]), abs(signal[center + 2]))
            if prominence < weak_config["local_prominence_fraction"] * max(
                abs(signal[center]), 1e-9
            ):
                continue
            if phase[center, x] < weak_config["minimum_phase_concentration"]:
                continue
            strength = abs(signal[center - 1 : center + 2]) * phase[center - 1 : center + 2, x]
            denominator = strength[0] - 2 * strength[1] + strength[2]
            offset = (
                0.5 * (strength[0] - strength[2]) / denominator if abs(denominator) > 1e-12 else 0.0
            )
            time = float(center + np.clip(offset, -0.5, 0.5))
            if any(
                abs(row["time"] - time) <= config["weak_overlap_samples"]
                for row in strong_by_trace[x]
            ):
                overlap += 1
                continue
            if any(abs(row["time"] - time) <= config["weak_overlap_samples"] for row in accepted):
                continue
            amplitude = np.asarray(
                [np.interp(time, np.arange(tau.shape[0]), avo[c, :, x]) for c in range(3)]
            )
            z = np.asarray(
                [np.interp(time, np.arange(tau.shape[0]), analytic[c, :, x]) for c in range(3)]
            )
            waveform = np.concatenate(
                [
                    np.interp(time + np.arange(-2, 3), np.arange(tau.shape[0]), avo[c, :, x])
                    for c in range(3)
                ]
            )
            p, g, c = _pgc(amplitude)
            accepted.append(
                {
                    "trace": x,
                    "time": time,
                    "tau": float(native.forward_tau(x, time)),
                    "a_near": float(amplitude[0]),
                    "a_mid": float(amplitude[1]),
                    "a_far": float(amplitude[2]),
                    "envelope": float(envelope[center, x]),
                    "phase_near": float(np.angle(z[0])),
                    "phase_mid": float(np.angle(z[1])),
                    "phase_far": float(np.angle(z[2])),
                    "phase_concentration": float(phase[center, x]),
                    "p": p,
                    "g": g,
                    "c": c,
                    "waveform": waveform,
                    "source_type": "weak",
                }
            )
        accepted.sort(key=lambda row: -row["envelope"])
        weak.extend(accepted[: config["weak_maximum_per_trace"]])
    combined = sorted(
        strong + weak, key=lambda row: (row["trace"], row["time"], row["source_type"])
    )
    for index, row in enumerate(combined):
        row["event"] = index
    return combined, {"strong": len(strong), "weak": len(weak), "overlap": overlap}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def candidate_links(
    native: NativeRGT, events: list[dict[str, Any]], radius: float, max_gap: int
) -> list[dict[str, Any]]:
    """Enumerate one- and two-trace links with native shift evidence at every crossed boundary."""
    by_trace: dict[int, list[int]] = defaultdict(list)
    for i, event in enumerate(events):
        by_trace[int(event["trace"])].append(i)
    rows = []
    for x in range(native.tau.shape[1]):
        for span in range(1, max_gap + 2):
            if x + span >= native.tau.shape[1]:
                continue
            for i in by_trace[x]:
                left = events[i]
                predicted = float(native.inverse_time(x + span, left["tau"]))
                tau_step = abs(
                    native.forward_tau(x, left["time"] + 0.5)
                    - native.forward_tau(x, left["time"] - 0.5)
                )
                for j in by_trace[x + span]:
                    right = events[j]
                    residual = abs(right["time"] - predicted)
                    if residual > radius * span:
                        continue
                    representative = 0.5 * (left["tau"] + right["tau"])
                    shifts = np.asarray(
                        [
                            native.inverse_time(k + 1, representative)
                            - native.inverse_time(k, representative)
                            for k in range(
                                max(0, x - 1), min(native.tau.shape[1] - 1, x + span + 1)
                            )
                        ],
                        float,
                    )
                    mid = 1 if x > 0 else 0
                    crossed = shifts[mid : mid + span]
                    local_jumps = []
                    for k in range(span):
                        position = mid + k
                        neighborhood = shifts[max(0, position - 1) : min(len(shifts), position + 2)]
                        local_jumps.append(abs(shifts[position] - np.median(neighborhood)))
                    second = abs(np.diff(shifts, n=2)) if len(shifts) > 2 else np.array([0.0])
                    pa = np.asarray([left[f"phase_{v}"] for v in ("near", "mid", "far")])
                    pb = np.asarray([right[f"phase_{v}"] for v in ("near", "mid", "far")])
                    aa = np.asarray([left[f"a_{v}"] for v in ("near", "mid", "far")])
                    ab = np.asarray([right[f"a_{v}"] for v in ("near", "mid", "far")])
                    rows.append(
                        {
                            "source": i,
                            "target": j,
                            "span": span,
                            "boundary": x,
                            "predicted_time": predicted,
                            "d_tau": float(abs(right["tau"] - left["tau"]) / max(tau_step, 1e-12)),
                            "d_time": float(residual),
                            "waveform_cosine": _cosine(left["waveform"], right["waveform"]),
                            "phase_cosine": float(np.mean(np.cos(pa - pb))),
                            "ava_cosine": _cosine(aa, ab),
                            "shift_jump": float(max(local_jumps)),
                            "shift_second_difference": float(max(second)),
                            "shift_continuity": float(np.max(abs(crossed - np.median(shifts)))),
                            "physical_shift": float(np.sum(crossed)),
                        }
                    )
    return rows


def freeze_scales(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, float]:
    """Robust observable scales and barriers from training candidate links only."""
    if not rows:
        raise ValueError("No training candidate links")
    names = ("d_tau", "d_time", "shift_jump", "shift_second_difference", "shift_continuity")
    return {
        name: float(max(np.quantile([row[name] for row in rows], 0.75), 1e-6)) for name in names
    } | {
        "waveform_penalty": float(
            max(np.quantile([1 - row["waveform_cosine"] for row in rows], 0.75), 1e-6)
        ),
        "phase_penalty": float(
            max(np.quantile([1 - row["phase_cosine"] for row in rows], 0.75), 1e-6)
        ),
        "ava_penalty": float(max(np.quantile([1 - row["ava_cosine"] for row in rows], 0.75), 1e-6)),
        "barrier_shift_jump": float(
            np.quantile([row["shift_jump"] for row in rows], config["barrier_quantile_cap"])
        ),
        "barrier_shift_second": float(
            np.quantile(
                [row["shift_second_difference"] for row in rows], config["barrier_quantile_cap"]
            )
        ),
    }


def score_links(
    rows: list[dict[str, Any]], scales: Mapping[str, float], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Preclude structural barriers, then assign fixed normalized DAG edge costs."""
    scored = []
    for original in rows:
        row = original.copy()
        row["barrier"] = bool(
            row["shift_jump"] > scales["barrier_shift_jump"]
            or row["shift_second_difference"] > scales["barrier_shift_second"]
            or row["waveform_cosine"] < config["minimum_waveform_cosine"]
            or row["phase_cosine"] < config["minimum_phase_cosine"]
            or row["ava_cosine"] < config["minimum_ava_cosine"]
        )
        row["cost"] = float(
            row["d_tau"] / scales["d_tau"]
            + row["d_time"] / scales["d_time"]
            + (1 - row["waveform_cosine"]) / scales["waveform_penalty"]
            + (1 - row["phase_cosine"]) / scales["phase_penalty"]
            + (1 - row["ava_cosine"]) / scales["ava_penalty"]
            + config["gap_penalty"] * (row["span"] - 1)
            + row["shift_continuity"] / scales["shift_continuity"]
        )
        row["plausible"] = bool(row["cost"] <= config["maximum_normalized_cost"])
        row["safe"] = bool(row["plausible"] and not row["barrier"])
        row["relation"] = (
            "FAULT_OFFSET_CORRESPONDENCE" if row["plausible"] and row["barrier"] else "CANDIDATE"
        )
        scored.append(row)
    return scored


def _crosses(a: Mapping[str, Any], b: Mapping[str, Any], events: list[dict[str, Any]]) -> bool:
    """Check order reversal of two segments on their common trace interval."""
    left, right = (
        max(events[a["source"]]["trace"], events[b["source"]]["trace"]),
        min(events[a["target"]]["trace"], events[b["target"]]["trace"]),
    )
    if right <= left:
        return False

    def height(row: Mapping[str, Any], x: int) -> float:
        start, end = events[row["source"]], events[row["target"]]
        return float(
            start["time"]
            + (end["time"] - start["time"]) * (x - start["trace"]) / (end["trace"] - start["trace"])
        )

    delta_left = height(a, left) - height(b, left)
    delta_right = height(a, right) - height(b, right)
    return bool(
        delta_left * delta_right < 0
        or (delta_left == 0 and delta_right != 0)
        or (delta_right == 0 and delta_left != 0)
    )


def track_paths(
    events: list[dict[str, Any]], links: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Repeated best DAG paths with exclusive nodes and noncrossing selected links."""
    incoming: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_boundary: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in links:
        if row["safe"]:
            incoming[row["target"]].append(row)
            for boundary in range(events[row["source"]]["trace"], events[row["target"]]["trace"]):
                by_boundary[boundary].append(row)
    used: set[int] = set()
    blocked: set[int] = set()
    accepted: list[dict[str, Any]] = []
    paths: list[list[int]] = []
    path_links: list[list[dict[str, Any]]] = []
    event_order = sorted(range(len(events)), key=lambda i: (events[i]["trace"], events[i]["time"]))
    while True:
        best_score = np.zeros(len(events), float)
        predecessor: dict[int, dict[str, Any]] = {}
        for j in event_order:
            if j in used:
                continue
            for row in incoming[j]:
                i = row["source"]
                if i in used or id(row) in blocked:
                    continue
                score = best_score[i] + config["path_step_reward"] * row["span"] - row["cost"]
                if score > best_score[j] + 1e-10:
                    best_score[j] = score
                    predecessor[j] = row
        if not predecessor:
            break
        eligible = []
        for end in sorted(range(len(events)), key=lambda i: (-best_score[i], i)):
            if best_score[end] <= 0:
                break
            chain = []
            cursor = end
            while cursor in predecessor:
                row = predecessor[cursor]
                chain.append(row)
                cursor = row["source"]
            chain.reverse()
            if not chain:
                continue
            indices = [chain[0]["source"]] + [row["target"] for row in chain]
            if (
                len(indices) >= config["minimum_component_points"]
                and events[indices[-1]]["trace"] - events[indices[0]]["trace"]
                >= config["minimum_component_span"]
            ):
                eligible.append((best_score[end], indices, chain))
                break
        if not eligible:
            break
        _, indices, chain = eligible[0]
        paths.append(indices)
        path_links.append(chain)
        used.update(indices)
        accepted.extend(chain)
        for selected in chain:
            for boundary in range(
                events[selected["source"]]["trace"], events[selected["target"]]["trace"]
            ):
                for candidate in by_boundary[boundary]:
                    if id(candidate) not in blocked and _crosses(candidate, selected, events):
                        blocked.add(id(candidate))
    return {
        "components": paths,
        "path_links": path_links,
        "accepted_links": accepted,
        "metadata": [row for row in links if row["relation"] == "FAULT_OFFSET_CORRESPONDENCE"],
    }


def sparse_graph(
    events: list[dict[str, Any]],
    tracking: dict[str, Any],
    spacing: int,
    config: Mapping[str, Any],
    curvature_limit: float,
) -> dict[str, Any]:
    """Subsample complete event tracks, then add long edges along their safe links."""
    nodes, edges = [], []
    for component, (path, links) in enumerate(zip(tracking["components"], tracking["path_links"])):
        chosen = [0]
        for k in range(1, len(path) - 1):
            if (
                events[path[k]]["trace"] - events[path[chosen[-1]]]["trace"] >= spacing
                or events[path[k]].get("structural_curvature", 0) >= curvature_limit
            ):
                chosen.append(k)
        if chosen[-1] != len(path) - 1:
            chosen.append(len(path) - 1)
        index_to_node = {}
        for k in chosen:
            index_to_node[k] = len(nodes)
            nodes.append({"node": len(nodes), "component": component, **events[path[k]]})
        for start in chosen:
            for end in chosen:
                if end <= start:
                    continue
                span = events[path[end]]["trace"] - events[path[start]]["trace"]
                if (
                    end != chosen[chosen.index(start) + 1]
                    and span not in config["edge_spans_traces"]
                ):
                    continue
                segment = links[start:end]
                coordinates = np.asarray(
                    [
                        [events[path[k]]["time"], events[path[k]]["trace"]]
                        for k in range(start, end + 1)
                    ],
                    float,
                )
                gaps = sum(link["span"] - 1 for link in segment)
                slopes = np.diff(coordinates[:, 0]) / np.diff(coordinates[:, 1])
                edges.append(
                    {
                        "edge": len(edges),
                        "source": index_to_node[start],
                        "target": index_to_node[end],
                        "component": component,
                        "span": span,
                        "delta_tau": float(events[path[end]]["tau"] - events[path[start]]["tau"]),
                        "delta_t": float(coordinates[-1, 0] - coordinates[0, 0]),
                        "delta_x": span,
                        "geodesic_length": float(
                            np.linalg.norm(np.diff(coordinates, axis=0), axis=1).sum()
                        ),
                        "gap_count": int(gaps),
                        "waveform_continuity": float(
                            min(row["waveform_cosine"] for row in segment)
                        ),
                        "phase_continuity": float(min(row["phase_cosine"] for row in segment)),
                        "shift_continuity": float(max(row["shift_continuity"] for row in segment)),
                        "curvature": float(max(abs(np.diff(slopes)))) if len(slopes) > 1 else 0.0,
                        "path": coordinates,
                    }
                )
    points = np.asarray([[node["time"], node["trace"]] for node in nodes], float).reshape(-1, 2)
    pairs = np.asarray([[edge["source"], edge["target"]] for edge in edges], int).reshape(-1, 2)
    return {
        **tracking,
        "nodes": nodes,
        "edges": edges,
        "statistics": graph_statistics(points, pairs),
    }


__all__ = [
    "detect_events",
    "candidate_links",
    "freeze_scales",
    "score_links",
    "track_paths",
    "sparse_graph",
    "normal_event_contrasts",
]
