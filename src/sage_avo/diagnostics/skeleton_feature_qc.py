"""Frozen-CNN, observable-only snapping and path diagnostics; no graph updates."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
import torch

from sage_avo.diagnostics.skeleton_graph import sample


def feature_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    distance = np.linalg.norm(a - b, axis=-1)
    cosine = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return cosine, distance, distance / np.maximum(0.5 * (na + nb), 1e-12)


@torch.inference_mode()
def verify_pre_gnn_tensor(model, avo, low, rgt):
    """Abort actual forward at encoder hook, before tokens or any graph call."""
    if model.training:
        raise ValueError("Require eval mode")

    class EncoderStop(Exception):
        pass

    captured = {}

    def capture(module, inputs, output):
        captured["cnn"] = output.detach().clone()
        raise EncoderStop()

    def forbid(module, inputs):
        raise RuntimeError("Forbidden graph/decoder execution")

    handles = [model.encoder.register_forward_hook(capture)]
    if model.graph is not None:
        handles.append(model.graph.register_forward_pre_hook(forbid))
    handles.append(model.decoder.register_forward_pre_hook(forbid))
    t = torch.zeros(low.shape[0], device=low.device, dtype=low.dtype)
    try:
        try:
            model(low, t, avo, low, rgt)
        except EncoderStop:
            pass
    finally:
        for h in handles:
            h.remove()
    time = model.time_embedding(t[:, None])[:, :, None, None].expand(-1, -1, *low.shape[-2:])
    condition = model.condition_embedding(torch.cat((avo, low), dim=1))
    actual = model.encoder(torch.cat((low, time, condition), dim=1))
    torch.testing.assert_close(actual, captured["cnn"], rtol=0, atol=0)
    return {
        "shape": list(actual.shape),
        "bitwise_equal_to_forward_pre_gnn": True,
        "graph_called": False,
        "decoder_called": False,
        "flow_time": 0.0,
        "flow_state": "normalized prior",
    }


def snap_candidates(tau, avo, valid, graph, config):
    """Diagnostic normal-line proposals constrained to disjoint RGT bands.

    Never mutates nodes. No fault labels, elastic truth or CNN input. Reject
    locally detectable orientation/gradient jumps, out-of-support points and
    crossings of adjacent selected interfaces. All distances in sample/trace units.
    """
    points = graph["points"]
    levels = graph["levels"]
    ids = graph["surface_ids"]
    gradients = np.stack(np.gradient(tau.astype(float)))
    strength = np.sqrt(np.mean(avo.astype(float) ** 2, axis=0))
    settings = config["snapping"]
    rows = []
    snapped = []
    offsets = np.arange(
        -settings["normal_radius_samples"],
        settings["normal_radius_samples"] + 1e-9,
        settings["normal_step_samples"],
    )
    for i, p in enumerate(points):
        grad = sample(gradients, p[None])[0]
        norm = np.linalg.norm(grad)
        original_strength = float(sample(strength, p[None])[0])
        sid = ids[i]
        gap = min(
            levels[sid] - levels[sid - 1] if sid else np.inf,
            levels[sid + 1] - levels[sid] if sid + 1 < len(levels) else np.inf,
        )
        tolerance = min(
            settings["rgt_tolerance_reference_steps"] * graph["scale"],
            settings["maximum_interface_gap_fraction"] * gap,
        )
        best = p.copy()
        best_strength = original_strength
        eligible = 1
        if norm > 1e-10:
            direction = grad / norm
            # Deterministic ties favor the smallest displacement, then signed offset.
            for offset in sorted(offsets, key=lambda s: (abs(s), s)):
                if abs(offset) < 1e-9:
                    continue
                candidate = p + offset * direction
                path = np.linspace(p, candidate, max(2, int(np.ceil(abs(offset) / 0.25)) + 1))
                if (
                    (path < 0).any()
                    or (path[:, 0] > tau.shape[0] - 1).any()
                    or (path[:, 1] > tau.shape[1] - 1).any()
                ):
                    continue
                if (sample(valid.astype(float), path) < 1 - 1e-7).any():
                    continue
                delta = sample(tau, path) - levels[sid]
                if (abs(delta) > tolerance).any():
                    continue
                # Preserve physical ordering against neighboring inverse surfaces.
                ordered = True
                for adjacent, sign in [(sid - 1, 1), (sid + 1, -1)]:
                    if 0 <= adjacent < len(levels):
                        other = np.interp(
                            path[:, 1], np.arange(tau.shape[1]), graph["curves"][adjacent]
                        )
                        if not np.isfinite(other).all() or ((path[:, 0] - other) * sign <= 0).any():
                            ordered = False
                if not ordered:
                    continue
                local = sample(gradients, path)
                length = np.linalg.norm(local, axis=1)
                cos = local @ direction / np.maximum(length, 1e-12)
                if (cos < settings["minimum_gradient_direction_cosine"]).any():
                    continue
                if (
                    length.max() / max(length.min(), 1e-12)
                    > settings["maximum_gradient_norm_ratio"]
                ):
                    continue
                eligible += 1
                value = float(sample(strength, candidate[None])[0])
                if value > best_strength + 1e-12:
                    best = candidate
                    best_strength = value
        snapped.append(best)
        rows.append(
            {
                "node": i,
                "row": p[0],
                "trace": p[1],
                "interface_id": int(sid),
                "reflector_strength": original_strength,
                "strongest_compatible_strength": best_strength,
                "strength_gain": best_strength - original_strength,
                "relative_strength_gain": (best_strength - original_strength)
                / max(original_strength, 1e-12),
                "candidate_row": best[0],
                "candidate_trace": best[1],
                "displacement": float(np.linalg.norm(best - p)),
                "rgt_deviation": float(sample(tau, best[None])[0] - levels[sid]),
                "rgt_tolerance": float(tolerance),
                "compatible_candidates": eligible,
                "changed": bool(np.linalg.norm(best - p) > 1e-7),
                "adopted": False,
            }
        )
    return np.asarray(snapped), rows


def path_observables(tau, avo, path, reference_scale, config):
    """Path-local consistency, independent of endpoint chord approximation.

    AVO coherence uses three-band +/-3-sample gradient-normal waveform windows.
    RGT-derived dip is not independent PWD. Low-energy/invalid windows fail closed.
    No fault truth is accepted by this function.
    """
    if len(path) < 2 or not np.isfinite(path).all():
        return {
            "valid_observable_path": False,
            "path_prior": 0.0,
            "coherence_p10": -1.0,
            "reflector_continuity_ratio": 0.0,
            "shift_residual": float("inf"),
        }
    gradients = np.stack(np.gradient(tau.astype(float)))
    dr = np.diff(path, axis=0)
    length = np.linalg.norm(dr, axis=1)
    mid = 0.5 * (path[:-1] + path[1:])
    local = sample(gradients, mid)
    cos = np.sum(local * dr, axis=1) / np.maximum(np.linalg.norm(local, axis=1) * length, 1e-12)
    node_tau = sample(tau, path)
    mid_tau = sample(tau, mid)
    deviation = abs(np.concatenate((node_tau, mid_tau)) - node_tau[0])
    scaled = deviation / reference_scale
    mean = float(scaled.mean())
    rms = float(np.sqrt(np.mean(cos**2)))
    prior_cfg = config["path_prior"]
    log_prior = (
        -0.5 * (mean / prior_cfg["sigma_mean_rgt_reference_steps"]) ** 2
        - 0.5 * (rms / prior_cfg["sigma_local_tangent_cosine"]) ** 2
    )
    tangent = dr / np.maximum(length[:, None], 1e-12)
    turn = np.arccos(np.clip(np.sum(tangent[:-1] * tangent[1:], axis=1), -1, 1))
    curvature = turn / np.maximum(0.5 * (length[:-1] + length[1:]), 1e-12)
    at_nodes = sample(gradients, path)
    norm = np.linalg.norm(at_nodes, axis=1)
    direction = at_nodes / np.maximum(norm[:, None], 1e-12)
    radius = config["path_observables"]["normal_waveform_radius"]
    offsets = np.arange(-radius, radius + 1)
    locations = path[:, None, :] + offsets[None, :, None] * direction[:, None, :]
    in_bounds = (
        (locations[:, :, 0] >= 0)
        & (locations[:, :, 0] <= tau.shape[0] - 1)
        & (locations[:, :, 1] >= 0)
        & (locations[:, :, 1] <= tau.shape[1] - 1)
    )
    wave = sample(avo, locations.reshape(-1, 2)).reshape(len(path), -1)
    wn = np.linalg.norm(wave, axis=1)
    cosine = np.sum(wave[:-1] * wave[1:], axis=1) / np.maximum(wn[:-1] * wn[1:], 1e-12)
    reliable = in_bounds[:-1].all(1) & in_bounds[1:].all(1) & (wn[:-1] > 1e-10) & (wn[1:] > 1e-10)
    cosine = np.where(reliable, cosine, -1.0)
    strength = np.sqrt(np.mean(sample(avo, path) ** 2, axis=1))
    continuity = float(np.quantile(strength, 0.1) / max(float(np.quantile(strength, 0.9)), 1e-12))
    shifts = dr[:, 0] / np.maximum(abs(dr[:, 1]), 1e-12)
    residual = float(np.max(abs(shifts - median_filter(shifts, size=5, mode="nearest"))))
    chord = float(np.linalg.norm(path[-1] - path[0]))
    geodesic = float(length.sum())
    return {
        "valid_observable_path": True,
        "endpoint_delta_rgt": float(node_tau[-1] - node_tau[0]),
        "mean_path_delta_rgt": float(deviation.mean()),
        "max_path_delta_rgt": float(deviation.max()),
        "mean_path_delta_rgt_reference_steps": mean,
        "max_path_delta_rgt_reference_steps": float(scaled.max()),
        "tangent_gradient_cosine_rms": rms,
        "tangent_gradient_cosine_max": float(abs(cos).max()),
        "path_prior": float(np.exp(max(log_prior, -745))),
        "path_log_prior": float(log_prior),
        "geodesic_length": geodesic,
        "chord_length": chord,
        "path_chord_ratio": geodesic / max(chord, 1e-12),
        "curvature_mean": float(curvature.mean()) if len(curvature) else 0.0,
        "curvature_max": float(curvature.max()) if len(curvature) else 0.0,
        "coherence_p10": float(np.quantile(cosine, 0.1)),
        "coherence_min": float(cosine.min()),
        "waveform_valid_fraction": float(reliable.mean()),
        "reflector_continuity_ratio": continuity,
        "shift_residual": residual,
    }


def filter_decisions(row, base_keep, config):
    p = config["path_observables"]
    coherence = row["coherence_p10"] >= p["minimum_path_coherence_p10"]
    continuity = (
        row["reflector_continuity_ratio"] >= p["minimum_reflector_continuity_ratio"]
        and row["shift_residual"] <= p["maximum_shift_median_residual"]
    )
    return {
        "BASELINE": bool(base_keep),
        "COHERENCE": bool(base_keep and coherence),
        "CONTINUITY": bool(base_keep and continuity),
        "COMBINED": bool(base_keep and coherence and continuity),
    }
