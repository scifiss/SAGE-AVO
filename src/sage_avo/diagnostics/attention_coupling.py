"""Pure, destination-wise measurements for the fixed-topology v00332r audit."""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations

import numpy as np
import torch

from sage_avo.diagnostics.rgt_topology_repair import (
    _balanced_lexicographic_choice,
    _candidate_arrays,
    near_tie_tolerance,
)


@contextmanager
def graph_intervention(model, *, attention="learned", edge_attr="current"):
    """Restore transient controls even if a diagnostic forward raises."""
    graph = model.graph
    previous = graph.attention_mode, graph.diagnostic_edge_attr_mode
    layer_modes = [layer.attention_mode for layer in graph.layers]
    try:
        graph.attention_mode, graph.diagnostic_edge_attr_mode = attention, edge_attr
        yield
    finally:
        graph.attention_mode, graph.diagnostic_edge_attr_mode = previous
        for layer, mode in zip(graph.layers, layer_modes):
            layer.attention_mode = mode


def headwise_uniformity(edge_index, alpha, node_count):
    """Average incoming-neighborhood statistics separately for every head.

    Singleton targets have no choice and are excluded from selectivity means.
    Top-decile mass uses ceil(0.1*degree); the corresponding uniform mass is
    reported so low-degree neighborhoods are not misread as concentrated.
    """
    targets = edge_index[1].detach().cpu().numpy()
    values = alpha.detach().double().cpu().numpy()
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("alpha must contain one row per edge and one column per head")
    order = np.argsort(targets, kind="stable")
    _, starts, degrees = np.unique(targets[order], return_index=True, return_counts=True)
    values = values[order]
    buckets = {key: [] for key in (
        "normalized_entropy", "kl_from_uniform", "coefficient_of_variation",
        "top_decile_mass", "uniform_top_decile_mass", "max_mean_attention",
        "min_mean_attention", "normalization_error",
    )}
    for degree in np.unique(degrees):
        if degree < 2:
            continue
        indices = starts[degrees == degree, None] + np.arange(degree)[None, :]
        p = values[indices]
        sums = p.sum(axis=1)
        if not np.isfinite(p).all() or (p < 0).any() or (sums <= 0).any():
            raise ValueError("attention must be finite, nonnegative and nonempty")
        buckets["normalization_error"].append(np.abs(sums - 1.0))
        p = p / sums[:, None, :]
        entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
        buckets["normalized_entropy"].append(entropy / np.log(degree))
        buckets["kl_from_uniform"].append(np.log(degree) - entropy)
        buckets["coefficient_of_variation"].append(p.std(axis=1) * degree)
        k = max(1, int(np.ceil(0.1 * degree)))
        buckets["top_decile_mass"].append(np.sort(p, axis=1)[:, -k:].sum(axis=1))
        buckets["uniform_top_decile_mass"].append(np.full(sums.shape, k / degree))
        buckets["max_mean_attention"].append(p.max(axis=1) * degree)
        buckets["min_mean_attention"].append(p.min(axis=1) * degree)
    rows = []
    for head in range(values.shape[1]):
        row = {
            "head": head, "multineighbor_destinations": int((degrees > 1).sum()),
            "singleton_destinations": int((degrees == 1).sum()),
            "isolated_destinations": int(node_count - len(degrees)),
        }
        for key, chunks in buckets.items():
            array = np.concatenate(chunks)[:, head] if chunks else np.array([])
            row[key] = float(array.mean()) if array.size else float("nan")
            if key in {"kl_from_uniform", "coefficient_of_variation"}:
                row[key + "_p95"] = float(np.quantile(array, .95)) if array.size else float("nan")
        rows.append(row)
    return rows


def final_tie_usage(rgt, max_shift=3):
    """Expose the frozen lexicographic stages, without changing their choice."""
    rows, columns, shifts, targets, valid, source, candidates, mismatch = _candidate_arrays(
        np.asarray(rgt, np.float32), max_shift
    )
    masked = np.where(valid, mismatch, np.inf)
    minimum = masked.min(axis=-1, keepdims=True)
    exact = valid & (masked == minimum)
    near = valid & (masked <= minimum + near_tie_tolerance(source, candidates)[..., None])
    stage1 = near.sum(axis=-1) > 1
    absolute = np.broadcast_to(np.abs(shifts), mismatch.shape)
    minimum_absolute = np.where(near, absolute, np.iinfo(np.int64).max).min(axis=-1, keepdims=True)
    finalists = near & (absolute == minimum_absolute)
    parity = finalists.sum(axis=-1) > 1
    choice, _ = _balanced_lexicographic_choice(mismatch, valid, shifts, rows, columns, source, candidates)
    selected = shifts[choice]
    return {
        "stage1_tie": stage1, "stage1_exact_tie": exact.sum(axis=-1) > 1,
        "resolved_by_min_abs_shift": stage1 & ~parity,
        "parity": parity, "shift": selected,
        "target_rows": np.take_along_axis(targets, choice[..., None], axis=-1)[..., 0],
    }


def task_gradient_rows(model, objectives):
    """Norm shares and pairwise cosines of effective objective gradients."""
    rows = []
    for prefix in ("graph.node_projection.", "graph.layers.0.", "graph.layers.1."):
        params = [p for name, p in model.named_parameters() if name.startswith(prefix)]
        vectors = {}
        for name, objective in objectives.items():
            grads = (
                torch.autograd.grad(objective, params, retain_graph=True, allow_unused=True)
                if objective.requires_grad else [None] * len(params)
            )
            vectors[name] = torch.cat([
                (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                for p, g in zip(params, grads)
            ])
        norms = {name: float(vector.norm()) for name, vector in vectors.items()}
        total = sum(norms.values())
        for first, second in combinations(vectors, 2):
            denominator = norms[first] * norms[second]
            rows.append({
                "parameter_group": prefix.rstrip("."), "objective_a": first, "objective_b": second,
                "gradient_norm_a": norms[first], "gradient_norm_b": norms[second],
                "relative_norm_share_a": norms[first] / total if total else 0.0,
                "relative_norm_share_b": norms[second] / total if total else 0.0,
                "cosine_similarity": float(torch.dot(vectors[first], vectors[second])) / denominator
                if denominator else float("nan"),
            })
    return rows


def elastic_extension_gate(candidate_gains, decoupling_gains, *, seed_count=3,
                           material_margin=.01, density_degradation_margin=.01):
    """Gate expanded validation using only paired elastic errors, never mIoU.

    Each gain array has [seed, vp/vs/density]. Positive means improvement in
    normalized RMSE; mean elastic gains must be supplied separately because
    mean(relative gains) is not the relative gain of mean elastic RMSE.
    """
    def checked(values):
        values = np.asarray(values, dtype=float)
        if values.shape != (seed_count, 4) or not np.isfinite(values).all():
            raise ValueError("Require finite [seed, vp/vs/density/mean-elastic] paired gains")
        return values

    candidates = {name: checked(values) for name, values in candidate_gains.items()}
    detached = checked(decoupling_gains)
    return {
        "A_vp_and_vs_all_three_seeds": [name for name, values in candidates.items()
                                        if (values[:, :2] > 0).all()],
        "B_mean_elastic_all_three_seeds": [name for name, values in candidates.items()
                                           if (values[:, 3] > 0).all()],
        "C_clear_reproducible_decoupling_without_major_density_degradation": bool(
            (detached[:, 3] > material_margin).all()
            and (detached[:, 2] >= -density_degradation_margin).all()
        ),
    }
