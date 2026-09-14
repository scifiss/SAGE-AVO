"""Read-only completion probes for the frozen sparse-skeleton prototype."""

from __future__ import annotations

import numpy as np
import torch

from sage_avo.diagnostics.skeleton_graph import sample
from sage_avo.evaluation.inference import blend_window, tile_starts


@torch.inference_mode()
def frozen_cnn_anchors(
    model, avo, low, points, normalization, device, *, patch_shape=(50, 100), stride=(25, 50)
):
    """Sample the existing encoder at t=0, state=normalized low-frequency prior.

    No graph, decoder, flow integration, truth tensor or optimizer is invoked.
    Hann-stitch frozen encoder features before bilinear node sampling. GroupNorm
    makes these tile-conditioned descriptors, not full-section encoder outputs.
    """
    if model.training:
        raise ValueError("Frozen feature extraction requires eval mode")
    xmean = np.asarray(normalization["x_mean"], np.float32)[:, None, None]
    xstd = np.asarray(normalization["x_std"], np.float32)[:, None, None]
    ymean = np.asarray(normalization["y_mean"], np.float32)[:, None, None]
    ystd = np.asarray(normalization["y_std"], np.float32)[:, None, None]
    if np.any(xstd <= 0) or np.any(ystd <= 0):
        raise ValueError("Normalization scales must be positive")
    x = (np.asarray(avo, np.float32) - xmean) / xstd
    normalized_low = (np.asarray(low, np.float32) - ymean) / ystd
    h, w = normalized_low.shape[1:]
    ph, pw = patch_shape
    if h < ph or w < pw:
        raise ValueError("Section smaller than frozen tile shape")
    window = blend_window(patch_shape)
    weight = np.zeros((h, w), float)
    total = None
    tiles = 0
    for top in tile_starts(h, ph, stride[0]):
        for left in tile_starts(w, pw, stride[1]):
            region = np.s_[top : top + ph, left : left + pw]
            a = torch.from_numpy(x[:, top : top + ph, left : left + pw][None]).to(device)
            state = torch.from_numpy(normalized_low[:, top : top + ph, left : left + pw][None]).to(
                device
            )
            t = torch.zeros((1, 1), device=device, dtype=state.dtype)
            time = model.time_embedding(t)[:, :, None, None].expand(-1, -1, ph, pw)
            condition = model.condition_embedding(torch.cat((a, state), dim=1))
            features = model.encoder(torch.cat((state, time, condition), dim=1))[0].cpu().numpy()
            if total is None:
                total = np.zeros((features.shape[0], h, w), float)
            total[:, region[0], region[1]] += features * window
            weight[region] += window
            tiles += 1
    if not np.all(weight > 0):
        raise RuntimeError("Uncovered pixels in frozen CNN tiling")
    values = sample(total / weight[None], points)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite frozen encoder descriptors")
    return values, {
        "tiles": tiles,
        "channels": values.shape[1],
        "nodes": len(points),
        "state": "normalized prior",
        "time": 0.0,
        "device": str(device),
        "patch_shape": list(patch_shape),
        "stride": list(stride),
        "graph_or_flow_executed": False,
    }


def signed_prior_softmax(log_prior, source, relation, *, sign=1.0, strength=1.0, epsilon=1e-12):
    """Compare additive consistency prior and literal subtraction without training."""
    logits = sign * strength * np.logaddexp(np.asarray(log_prior, float), np.log(epsilon))
    result = np.zeros_like(logits)
    groups = {}
    for index, key in enumerate(zip(source, relation)):
        groups.setdefault(key, []).append(index)
    for indices in groups.values():
        values = logits[indices]
        weights = np.exp(values - values.max())
        result[indices] = weights / weights.sum()
    return result


def neighborhood_summary(points, edges):
    """Count actual degrees and links beyond 3x3/9x9 local convolution support.

    9x9 refers only to the four 3x3 encoder convolutions; not the entire network
    or GroupNorm's spatial-statistic coupling. No claim of CNN impossibility.
    """
    pairs = np.asarray([[e["source"], e["target"]] for e in edges if e["kept"]], int).reshape(-1, 2)
    degree = np.bincount(pairs.ravel(), minlength=len(points))
    delta = abs(points[pairs[:, 1]] - points[pairs[:, 0]])
    return {
        "degree_min": int(degree.min()) if len(degree) else 0,
        "degree_median": float(np.median(degree)) if len(degree) else 0.0,
        "degree_p95": float(np.quantile(degree, 0.95)) if len(degree) else 0.0,
        "degree_max": int(degree.max()) if len(degree) else 0,
        "outside_3x3_fraction": float((delta > 1 + 1e-6).any(1).mean()) if len(delta) else 0.0,
        "outside_encoder_9x9_fraction": float((delta > 4 + 1e-6).any(1).mean())
        if len(delta)
        else 0.0,
    }
