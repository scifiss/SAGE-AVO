"""RGT-flattened observable reflector components; topology/QC only."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, hilbert

from sage_avo.diagnostics.skeleton_graph import graph_statistics, sample


def _unique_column(values, *fields):
    """Collapse RGT plateaus without sorting or inventing a crossing."""
    unique, inverse = np.unique(values, return_inverse=True)
    count = np.bincount(inverse).astype(float)
    reduced = []
    for field in fields:
        field = np.asarray(field)
        if field.ndim == 1:
            reduced.append(np.bincount(inverse, weights=field) / count)
        else:
            reduced.append(np.stack([np.bincount(inverse, weights=f) / count for f in field]))
    return unique, reduced, count[inverse] == 1, count


def flatten_rgt(tau, avo, valid, n_tau=301):
    """Construct column-wise (t,x)<->(tau,x) interpolation and flatten AVA.

    RGT must be nondecreasing. Plateau locations are flagged as ambiguous for
    round-trip scoring rather than perturbed into an artificial ordering.
    """
    tau = np.asarray(tau, float)
    avo = np.asarray(avo, float)
    valid = np.asarray(valid, bool)
    if tau.ndim != 2 or avo.shape != (3, *tau.shape) or valid.shape != tau.shape:
        raise ValueError("Require tau[H,W], AVA[3,H,W], valid[H,W]")
    if not np.isfinite(tau).all() or not np.isfinite(avo).all() or (np.diff(tau, axis=0) < 0).any():
        raise ValueError("RGT must be finite and nondecreasing in every trace")
    columns = []
    for x in range(tau.shape[1]):
        unique, reduced, identifiable, unique_counts = _unique_column(
            tau[:, x], np.arange(tau.shape[0], dtype=float), avo[:, :, x], valid[:, x].astype(float)
        )
        columns.append((unique, *reduced, identifiable, unique_counts))
    lo = max(c[0][0] for c in columns)
    hi = min(c[0][-1] for c in columns)
    if not hi > lo:
        raise ValueError("No common invertible RGT range")
    grid = np.linspace(lo, hi, int(n_tau))
    inverse_t = np.empty((len(grid), tau.shape[1]))
    flat = np.empty((3, len(grid), tau.shape[1]))
    flat_valid = np.empty((len(grid), tau.shape[1]), bool)
    time_errors, grid_time_errors, avo_errors = [], [], []
    for x, (unique, times, amplitudes, support, identifiable, unique_counts) in enumerate(columns):
        inverse_t[:, x] = np.interp(grid, unique, times)
        for c in range(3):
            flat[c, :, x] = np.interp(grid, unique, amplitudes[c])
        segment = np.clip(np.searchsorted(unique, grid, side="right") - 1, 0, len(unique) - 2)
        invertible_segment = (unique_counts[segment] == 1) & (unique_counts[segment + 1] == 1)
        flat_valid[:, x] = (np.interp(grid, unique, support) >= 1 - 1e-7) & invertible_segment
        inside = identifiable & (tau[:, x] >= lo) & (tau[:, x] <= hi) & valid[:, x]
        if inside.any():
            direct_t = np.interp(tau[inside, x], unique, times)
            reconstructed_t = np.interp(tau[inside, x], grid, inverse_t[:, x])
            time_errors.extend(abs(direct_t - np.flatnonzero(inside)))
            grid_time_errors.extend(abs(reconstructed_t - np.flatnonzero(inside)))
            for c in range(3):
                reconstructed = np.interp(tau[inside, x], grid, flat[c, :, x])
                avo_errors.extend(abs(reconstructed - avo[c, inside, x]))
    coords = np.indices(inverse_t.shape)
    tau_back = sample(tau, np.column_stack((inverse_t.ravel(), coords[1].ravel()))).reshape(
        inverse_t.shape
    )
    tau_step = float(np.median(np.diff(grid)))
    tau_error_steps = abs(tau_back - grid[:, None]) / tau_step
    return {
        "tau_grid": grid,
        "inverse_t": inverse_t,
        "avo": flat,
        "valid": flat_valid,
        "evidence": gaussian_filter(
            np.sqrt(np.mean(abs(hilbert(flat, axis=1)) ** 2, axis=0)), (1, 0.5)
        ),
        "roundtrip_tau_steps": tau_error_steps[flat_valid],
        "roundtrip_time_samples": np.asarray(time_errors),
        "grid_resampling_time_error_samples": np.asarray(grid_time_errors),
        "roundtrip_avo_abs": np.asarray(avo_errors),
        "ambiguous_plateau_fraction": float(1 - np.mean(np.concatenate([c[-2] for c in columns]))),
    }


def _waveform(flat_avo, y, x, radius):
    lo, hi = y - radius, y + radius + 1
    if lo < 0 or hi > flat_avo.shape[1]:
        return None
    return flat_avo[:, lo:hi, x].ravel()


def detect_reflector_components(mapping, config):
    """Detect per-trace ridges and join only reciprocal observable matches."""
    flat, valid, evidence = mapping["avo"], mapping["valid"], mapping["evidence"]
    analytic = hilbert(flat, axis=1)
    threshold = float(np.quantile(evidence[valid], config["ridge_quantile"]))
    prominence = config["ridge_prominence_fraction"] * float(np.quantile(evidence[valid], 0.95))
    candidates = []
    by_trace = {}
    for x in range(flat.shape[2]):
        peaks, props = find_peaks(
            np.where(valid[:, x], evidence[:, x], 0.0),
            height=threshold,
            prominence=prominence,
            distance=config["ridge_minimum_distance"],
        )
        if len(peaks) > config["maximum_candidates_per_trace"]:
            order = np.argsort(props["peak_heights"])[-config["maximum_candidates_per_trace"] :]
            peaks = np.sort(peaks[order])
        by_trace[x] = []
        for y in peaks:
            i = len(candidates)
            candidates.append(
                {"candidate": i, "tau_index": int(y), "trace": x, "strength": float(evidence[y, x])}
            )
            by_trace[x].append(i)
    parent = list(range(len(candidates)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    links = []
    radius = config["waveform_radius"]
    for x in range(flat.shape[2] - 1):
        left, right = by_trace[x], by_trace[x + 1]
        possible = []
        for i in left:
            yi = candidates[i]["tau_index"]
            wi = _waveform(flat, yi, x, radius)
            if wi is None:
                continue
            for j in right:
                yj = candidates[j]["tau_index"]
                jump = abs(yj - yi)
                if jump > config["maximum_flattened_jump_samples"]:
                    continue
                wj = _waveform(flat, yj, x + 1, radius)
                if wj is None:
                    continue
                waveform = float(wi @ wj / max(np.linalg.norm(wi) * np.linalg.norm(wj), 1e-12))
                phase = float(
                    np.mean(np.cos(np.angle(analytic[:, yi, x]) - np.angle(analytic[:, yj, x + 1])))
                )
                ratio = min(candidates[i]["strength"], candidates[j]["strength"]) / max(
                    candidates[i]["strength"], candidates[j]["strength"], 1e-12
                )
                cost = jump + 0.5 * (1 - waveform) + 0.25 * (1 - phase)
                possible.append((i, j, cost, jump, waveform, phase, ratio))
        best_l = {
            i: min((p for p in possible if p[0] == i), key=lambda p: (p[2], p[1]), default=None)
            for i in left
        }
        best_r = {
            j: min((p for p in possible if p[1] == j), key=lambda p: (p[2], p[0]), default=None)
            for j in right
        }
        for p in possible:
            i, j, _, jump, waveform, phase, ratio = p
            reciprocal = best_l.get(i) == p and best_r.get(j) == p
            keep = (
                reciprocal
                and waveform >= config["minimum_waveform_cosine"]
                and phase >= config["minimum_phase_cosine"]
                and ratio >= config["minimum_strength_ratio"]
            )
            links.append(
                {
                    "source_candidate": i,
                    "target_candidate": j,
                    "jump": jump,
                    "waveform_cosine": waveform,
                    "phase_cosine": phase,
                    "strength_ratio": ratio,
                    "reciprocal": reciprocal,
                    "kept": bool(keep),
                }
            )
            if keep:
                union(i, j)
    groups = {}
    for i in range(len(candidates)):
        groups.setdefault(root(i), []).append(i)
    accepted = [
        g
        for g in groups.values()
        if len(g) >= config["minimum_component_points"]
        and candidates[g[-1]]["trace"] - candidates[g[0]]["trace"]
        >= config["minimum_component_trace_span"]
    ]
    accepted.sort(key=lambda g: (candidates[g[0]]["trace"], candidates[g[0]]["tau_index"], g[0]))
    component = np.full(len(candidates), -1, int)
    for cid, group in enumerate(accepted):
        component[group] = cid
    for row, cid in zip(candidates, component):
        row["component"] = int(cid)
        row["accepted"] = bool(cid >= 0)
    return {
        "candidates": candidates,
        "links": links,
        "components": accepted,
        "threshold": threshold,
        "prominence": prominence,
    }


def build_component_graph(mapping, detection, config):
    """Sparse nodes and multi-range edges restricted to one accepted component."""
    candidates = detection["candidates"]
    nodes, lookup, component_paths = [], {}, {}
    for cid, group in enumerate(detection["components"]):
        ordered = sorted(group, key=lambda i: candidates[i]["trace"])
        component_paths[cid] = ordered
        chosen = [ordered[0]]
        for i in ordered[1:]:
            if (
                candidates[i]["trace"] - candidates[chosen[-1]]["trace"]
                >= config["node_spacing_traces"]
            ):
                chosen.append(i)
        for i in chosen:
            c = candidates[i]
            n = len(nodes)
            lookup[cid, c["trace"]] = n
            nodes.append(
                {
                    "node": n,
                    "component": cid,
                    "candidate": i,
                    "tau_index": c["tau_index"],
                    "tau": float(mapping["tau_grid"][c["tau_index"]]),
                    "trace": c["trace"],
                    "time": float(mapping["inverse_t"][c["tau_index"], c["trace"]]),
                    "strength": c["strength"],
                }
            )
    edges = []
    for (cid, x), i in sorted(lookup.items()):
        group = component_paths[cid]
        by_x = {candidates[k]["trace"]: k for k in group}
        for span in config["edge_spans_traces"]:
            j = lookup.get((cid, x + span))
            if j is None or not all(xx in by_x for xx in range(x, x + span + 1)):
                continue
            path_ids = [by_x[xx] for xx in range(x, x + span + 1)]
            path = np.array(
                [
                    [
                        mapping["inverse_t"][candidates[k]["tau_index"], candidates[k]["trace"]],
                        candidates[k]["trace"],
                    ]
                    for k in path_ids
                ]
            )
            tau_path = np.array([mapping["tau_grid"][candidates[k]["tau_index"]] for k in path_ids])
            slopes = np.diff(path[:, 0])
            curvature = np.diff(slopes)
            link_rows = [
                r
                for r in detection["links"]
                if r["kept"]
                and r["source_candidate"] in path_ids[:-1]
                and r["target_candidate"] in path_ids[1:]
            ]
            edges.append(
                {
                    "edge": len(edges),
                    "source": i,
                    "target": j,
                    "component": cid,
                    "span": span,
                    "delta_tau": float(tau_path[-1] - tau_path[0]),
                    "delta_x": span,
                    "delta_t": float(path[-1, 0] - path[0, 0]),
                    "cartesian_length": float(np.linalg.norm(path[-1] - path[0])),
                    "geodesic_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                    "dip_difference": float(slopes[-1] - slopes[0]) if len(slopes) > 1 else 0.0,
                    "curvature_mean": float(np.mean(abs(curvature))) if len(curvature) else 0.0,
                    "curvature_max": float(np.max(abs(curvature))) if len(curvature) else 0.0,
                    "reflector_continuity": float(
                        min((r["waveform_cosine"] for r in link_rows), default=np.nan)
                    ),
                    "path": path,
                }
            )
    return {"nodes": nodes, "edges": edges, "component_paths": component_paths}


def normal_contrasts(tau, avo, nodes, offset=1.5):
    """Observable above/below contrasts along the local physical RGT normal."""
    gradients = np.stack(np.gradient(np.asarray(tau, float)))
    rows = []
    for node in nodes:
        p = np.array([node["time"], node["trace"]], float)
        g = sample(gradients, p[None])[0]
        direction = g / max(np.linalg.norm(g), 1e-12)
        above, below = p - offset * direction, p + offset * direction
        a, b = sample(avo, above[None])[0], sample(avo, below[None])[0]
        x = np.sin(np.deg2rad([10.0, 24.0, 38.0])) ** 2

        def h(v):
            slope = float(
                ((v - v.mean()) * (x - x.mean())).sum() / (((x - x.mean()) ** 2).sum() + 1e-12)
            )
            return np.r_[v, v.mean() - slope * x.mean(), slope, v[0] - 2 * v[1] + v[2]]

        rows.append(
            {
                "node": node["node"],
                "component": node["component"],
                "above_time": above[0],
                "above_trace": above[1],
                "below_time": below[0],
                "below_trace": below[1],
                **{f"delta_ava_{k}": float(b[k] - a[k]) for k in range(3)},
                **{f"delta_h_{k}": float(v) for k, v in enumerate(h(b) - h(a))},
            }
        )
    return rows


def component_graph_statistics(graph):
    points = np.array([[n["time"], n["trace"]] for n in graph["nodes"]], float)
    pairs = [[e["source"], e["target"]] for e in graph["edges"]]
    return graph_statistics(points, pairs)
