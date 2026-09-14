"""Observable-only sparse interface graph. No learned model or fault-label inputs.

Coordinates are (row sample, trace); lengths are grid-sample distances, NOT metres.
Inverse correspondences are fractional: near-zero delta-tau is by construction,
not independent proof of correct geological identity across a discontinuity.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, median_filter
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix


def sample(field, points):
    """Bilinear sampling of [H,W] or [C,H,W]; no coordinate rounding."""
    field = np.asarray(field)
    points = np.asarray(points).reshape(-1, 2)
    if field.ndim == 3:
        return np.stack([sample(channel, points) for channel in field], axis=1)
    return map_coordinates(field.astype(float, copy=False), points.T, order=1, mode="nearest")


def inverse_surfaces(tau, levels):
    """Unique increasing crossing per column. Plateaus/folds/ambiguities -> NaN.

    Never sort a nonmonotonic trace into an invented stratigraphic sequence.
    Half-open intervals avoid counting a single exact sample crossing twice.
    """
    tau = np.asarray(tau, dtype=float)
    surfaces = np.full((len(levels), tau.shape[1]), np.nan)
    for i, level in enumerate(levels):
        cross = (tau[:-1] <= level) & (tau[1:] > level)
        # Reject even a unique upward crossing when another downward one exists.
        down = (tau[:-1] > level) & (tau[1:] <= level)
        flat = (tau[:-1] == level) & (tau[1:] == level)
        valid = (cross.sum(0) == 1) & ~down.any(0) & ~flat.any(0)
        cols = np.flatnonzero(valid)
        rows = cross[:, cols].argmax(0)
        surfaces[i, cols] = rows + (level - tau[rows, cols]) / (
            tau[rows + 1, cols] - tau[rows, cols]
        )
    return surfaces


def observable_fields(tau, avo):
    vertical, lateral = np.gradient(np.asarray(tau, dtype=float))
    positive = np.diff(tau.astype(float), axis=0)
    positive = positive[positive > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    floor = 16 * np.finfo(np.float32).eps * scale
    dip = -lateral / np.maximum(vertical, floor)
    return {
        "vertical": vertical,
        "lateral": lateral,
        "dip": dip,
        "curvature": np.gradient(dip, axis=1),
        "scale": scale,
        "floor": floor,
        "strength": np.sqrt(np.mean(np.asarray(avo, float) ** 2, axis=0)),
    }


def select_interfaces(tau, avo, valid, config):
    """Rank separated local peaks of observable amplitude versus RGT level.

    Only observed AVO, RGT and acquisition support are inputs. No truth elastic,
    segmentation or reservoir maps influence node placement or thresholds.
    """
    levels = np.linspace(*np.quantile(tau[valid], [0.03, 0.97]), config["level_samples"])
    surfaces = inverse_surfaces(tau, levels)
    strength = observable_fields(tau, avo)["strength"]
    scores, coverage = [], []
    for curve in surfaces:
        columns = np.flatnonzero(np.isfinite(curve))
        points = np.column_stack((curve[columns], columns))
        supported = sample(valid.astype(float), points) >= 1 - 1e-7
        coverage.append(float(supported.sum() / tau.shape[1]))
        scores.append(
            float(np.mean(sample(strength, points)[supported])) if supported.any() else 0.0
        )
    scores, coverage = np.asarray(scores), np.asarray(coverage)
    eligible = coverage >= config["minimum_surface_coverage"]
    candidates, _ = find_peaks(
        np.where(eligible, scores, -np.inf), distance=config["minimum_level_separation"]
    )
    selected = sorted(sorted(candidates, key=lambda j: (-scores[j], j))[: config["max_interfaces"]])
    return (
        levels[selected],
        surfaces[selected],
        {
            "candidate_tau": levels,
            "reflector_score": scores,
            "coverage": coverage,
            "selected_indices": np.asarray(selected, int),
        },
    )


def node_features(tau, avo, points, *, cnn_features=None):
    """[optional frozen CNN, near,mid,far,P,G,C,tau,dip,curvature].

    P/G/C follow existing angular_features at (10,24,38) degrees. The prototype
    does not initialize a CNN or substitute random features for trained ones.
    """
    obs = sample(avo, points)
    x = np.sin(np.deg2rad([10.0, 24.0, 38.0])) ** 2
    gradient = ((obs - obs.mean(1, keepdims=True)) * (x - x.mean())).sum(1) / (
        ((x - x.mean()) ** 2).sum() + 1e-6
    )
    intercept = obs.mean(1) - gradient * x.mean()
    curvature = obs[:, 0] - 2 * obs[:, 1] + obs[:, 2]
    fields = observable_fields(tau, avo)
    extra = np.column_stack(
        (
            intercept,
            gradient,
            curvature,
            sample(tau, points),
            sample(fields["dip"], points),
            sample(fields["curvature"], points),
        )
    )
    features = np.column_stack((obs, extra))
    if cnn_features is not None:
        features = np.column_stack((sample(cnn_features, points), features))
    return features


def build_skeleton(tau, avo, valid, config):
    tau, avo, valid = np.asarray(tau), np.asarray(avo), np.asarray(valid, bool)
    if (
        tau.ndim != 2
        or min(tau.shape) < 3
        or avo.shape != (3, *tau.shape)
        or valid.shape != tau.shape
    ):
        raise ValueError("Require RGT[H,W], AVO[3,H,W], support[H,W]")
    if not np.isfinite(tau).all() or not np.isfinite(avo).all() or not valid.any():
        raise ValueError("Nonfinite observables or empty support")
    if config["node_spacing"] < 1:
        raise ValueError("Node spacing must be positive")
    fields = observable_fields(tau, avo)
    levels, curves, selection = select_interfaces(tau, avo, valid, config)
    points, surface_ids, lookup = [], [], {}
    for surface, curve in enumerate(curves):
        for x in range(0, tau.shape[1], config["node_spacing"]):
            if (
                np.isfinite(curve[x])
                and sample(valid.astype(float), [[curve[x], x]])[0] >= 1 - 1e-7
            ):
                lookup[surface, x] = len(points)
                points.append([curve[x], x])
                surface_ids.append(surface)
    points = np.asarray(points, float).reshape(-1, 2)
    records, paths = [], []

    def add(i, j, path, relation, span):
        finite = np.isfinite(path).all()
        reasons = []
        shift_residual = dip_residual = reciprocal = float("nan")
        if not finite:
            reasons.append("missing_inverse")
        else:
            if (sample(valid.astype(float), path) < 1 - 1e-7).any():
                reasons.append("unsupported_path")
            if relation == "tangential":
                delta = np.diff(path[:, 0])
                local = median_filter(delta, size=5, mode="nearest")
                shift_residual = float(np.max(np.abs(delta - local)))
                slopes = sample(fields["dip"], path)
                dip_residual = float(np.max(np.abs(delta - (slopes[:-1] + slopes[1:]) / 2)))
                inverse_error = np.abs(sample(tau, path) - levels[surface_ids[i]])
                vertical = sample(fields["vertical"], path)
                reciprocal = float(np.max(inverse_error / np.maximum(vertical, fields["floor"])))
                if (vertical <= fields["floor"]).any():
                    reasons.append("unidentifiable_scale")
                if np.max(np.abs(delta)) > config["maximum_step_samples"]:
                    reasons.append("large_shift_step")
                if shift_residual > config["maximum_shift_residual_samples"]:
                    reasons.append("shift_discontinuity")
                if dip_residual > config["maximum_dip_residual_samples"]:
                    reasons.append("rgt_dip_inconsistency")
                if reciprocal > config["maximum_reciprocal_error_samples"]:
                    reasons.append("roundtrip_error")
            else:
                vertical = sample(fields["vertical"], path)
                if (vertical <= fields["floor"]).any():
                    reasons.append("unidentifiable_scale")
        dr = points[j] - points[i]
        records.append(
            {
                "source": i,
                "target": j,
                "relation": relation,
                "span": span,
                "kept": not reasons,
                "reason": "|".join(reasons),
                "cartesian_length": float(np.linalg.norm(dr)),
                "geodesic_length": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if finite
                else np.nan,
                "rgt_mismatch": float(
                    abs(sample(tau, points[[j]])[0] - sample(tau, points[[i]])[0])
                ),
                "shift_residual": shift_residual,
                "dip_residual": dip_residual,
                "reciprocal_error_samples": reciprocal,
            }
        )
        paths.append(path)

    for (surface, x), i in lookup.items():
        for multiple in config["range_multipliers"]:
            span = config["node_spacing"] * multiple
            j = lookup.get((surface, x + span))
            if j is not None:
                columns = np.arange(x, x + span + 1)
                add(i, j, np.column_stack((curves[surface, columns], columns)), "tangential", span)
        # Across the next selected interface, not an adjacent image sample.
        j = lookup.get((surface + 1, x))
        if j is not None:
            rows = np.linspace(
                points[i, 0],
                points[j, 0],
                max(2, int(np.ceil(abs(points[j, 0] - points[i, 0]))) + 1),
            )
            add(i, j, np.column_stack((rows, np.full(len(rows), x))), "normal", 0)
    return {
        "points": points,
        "surface_ids": np.asarray(surface_ids, int),
        "levels": levels,
        "curves": curves,
        "selection": selection,
        "edges": records,
        "paths": paths,
        "features": node_features(tau, avo, points),
        "scale": fields["scale"],
    }


def attention_quantities(tau, points, edges, config, *, q=None, k=None, geometry_bias=None):
    """Directed RGT-prior attention; softmax separately by source AND relation.

    Default q.k=0, geometry bias=0: this is a fixed prior visualization, NOT
    trained Transformer attention. Direction-dependent endpoint gradient retained.
    Normal priors are reported as a counterexample, not endorsed for layer contrast.
    """
    grad = np.gradient(np.asarray(tau, float))
    geometry = observable_fields(tau, np.zeros((3, *tau.shape)))
    increments = np.diff(tau.astype(float), axis=0)
    positive = increments[increments > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    rows = []
    for edge in edges:
        if not edge["kept"]:
            continue
        for i, j in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
            r = points[j] - points[i]
            dtau = float(sample(tau, points[[j]])[0] - sample(tau, points[[i]])[0])
            projected = float(sum(sample(g, points[[i]])[0] * r[a] for a, g in enumerate(grad)))
            logp = (
                -0.5 * (dtau / (scale * config["sigma_tau_steps"])) ** 2
                - 0.5 * (projected / (scale * config["sigma_gradient_steps"])) ** 2
            )
            prior = float(np.exp(max(logp, -745)))
            content = 0.0 if q is None else float(q[i] @ k[j] / np.sqrt(q.shape[1]))
            geom = 0.0 if geometry_bias is None else float(geometry_bias(i, j))
            rows.append(
                {
                    "source": i,
                    "target": j,
                    "relation": edge["relation"],
                    "delta_row": float(r[0]),
                    "delta_trace": float(r[1]),
                    "delta_tau": dtau,
                    "gradient_dot_displacement": projected,
                    "edge_length": float(np.linalg.norm(r)),
                    "dip_difference": float(
                        sample(geometry["dip"], points[[j]])[0]
                        - sample(geometry["dip"], points[[i]])[0]
                    ),
                    "curvature_difference": float(
                        sample(geometry["curvature"], points[[j]])[0]
                        - sample(geometry["curvature"], points[[i]])[0]
                    ),
                    "log_prior": logp,
                    "prior": prior,
                    "geometry_bias": geom,
                    "logit": content
                    + config["attention_lambda"]
                    * float(np.logaddexp(logp, np.log(config["attention_epsilon"])))
                    + geom,
                }
            )
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((row["source"], row["relation"]), []).append(index)
    for indices in groups.values():
        logits = np.array([rows[i]["logit"] for i in indices])
        alpha = np.exp(logits - logits.max())
        alpha /= alpha.sum()
        for i, value in zip(indices, alpha):
            rows[i]["alpha"] = float(value)
    return rows


def path_fault_qc(path, faults):
    """Truth-only QC on every path segment, not just its endpoints."""
    if not np.isfinite(path).all():
        return None, None
    crossing = near = False
    for fault in faults:
        signed = path[:, 1] - float(fault["column"]) - float(fault["dip"]) * path[:, 0]
        crossing |= bool(((signed[1:] > 0) != (signed[:-1] > 0)).any())
        near |= bool((np.abs(signed) <= 3).any())
    # A segment passing through a fault necessarily enters its corridor even
    # if neither sampled endpoint lies inside the corridor.
    return crossing, near or crossing


def graph_statistics(points, pairs):
    """Unique undirected edges; exact per-node one/two-hop reach in grid units."""
    n = len(points)
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if len(pairs):
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    src = np.concatenate((pairs[:, 0], pairs[:, 1]))
    dst = np.concatenate((pairs[:, 1], pairs[:, 0]))
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    reach1, reach2 = np.zeros(n), np.zeros(n)
    for i in range(n):
        one = adjacency.indices[adjacency.indptr[i] : adjacency.indptr[i + 1]]
        if not len(one):
            continue
        two = np.unique(
            np.concatenate(
                [one]
                + [adjacency.indices[adjacency.indptr[j] : adjacency.indptr[j + 1]] for j in one]
            )
        )
        reach1[i] = np.linalg.norm(points[one] - points[i], axis=1).max()
        reach2[i] = np.linalg.norm(points[two] - points[i], axis=1).max()
    lengths = np.linalg.norm(points[pairs[:, 1]] - points[pairs[:, 0]], axis=1)
    return {
        "node_count": n,
        "edge_count_undirected": len(pairs),
        "average_degree": 2 * len(pairs) / max(n, 1),
        "edge_density": 2 * len(pairs) / max(n * (n - 1), 1),
        "graph_sparsity": 1 - 2 * len(pairs) / max(n * (n - 1), 1),
        "isolated_node_fraction": float(np.mean(adjacency.getnnz(axis=1) == 0)) if n else 1.0,
        "mean_cartesian_length": float(lengths.mean()) if len(lengths) else 0.0,
        "max_cartesian_length": float(lengths.max()) if len(lengths) else 0.0,
        "one_hop_reach_mean": float(reach1.mean()) if n else 0.0,
        "two_hop_reach_mean": float(reach2.mean()) if n else 0.0,
        "two_hop_reach_max": float(reach2.max()) if n else 0.0,
    }
