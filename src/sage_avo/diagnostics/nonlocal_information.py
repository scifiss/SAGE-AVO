"""Frozen observable CNN extraction and capacity-matched nonlocal regression probes.

No graph builder, model optimizer, teacher-forced state, or flow integration is
used here. Labels enter only the regression target and post-selection region QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
import pandas as pd
import torch

from sage_avo.diagnostics.native_rgt_graph import NativeRGT
from sage_avo.evaluation.inference import blend_window, tile_starts


MODES = ("cartesian", "rgt", "disrupted")
PRIMARY = ("local_only", "local_cartesian", "local_rgt", "local_disrupted")
PROPERTIES = ("Vp", "Vs", "density")


@torch.inference_mode()
def observable_cnn(model: torch.nn.Module, avo: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """Exactly the current model's pre-GNN encoder at t=0, state=prior.

Executing only the existing upstream modules avoids needlessly constructing a
graph whose output is not used. A regression test compares this to an encoder
hook on the unmodified complete dense forward pass.
"""
    time = torch.zeros((low.shape[0], 1), dtype=low.dtype, device=low.device)
    temporal = model.time_embedding(time)[:, :, None, None].expand(
        -1, -1, *low.shape[-2:]
    )
    condition = model.condition_embedding(torch.cat((avo, low), dim=1))
    return model.encoder(torch.cat((low, temporal, condition), dim=1))


def feature_bank(model, avo, low, normalization, patch_shape, stride, device):
    """Hann-stitch t=0 CNN features using the existing production tile protocol."""
    x_mean = np.array(normalization["x_mean"], np.float32)[:, None, None]
    x_std = np.array(normalization["x_std"], np.float32)[:, None, None]
    y_mean = np.array(normalization["y_mean"], np.float32)[:, None, None]
    y_std = np.array(normalization["y_std"], np.float32)[:, None, None]
    a = (avo - x_mean) / x_std
    m = (low - y_mean) / y_std
    height, width = avo.shape[-2:]
    window = blend_window(patch_shape)
    accumulated = None
    weights = np.zeros((height, width), np.float64)
    model.eval()
    for top in tile_starts(height, patch_shape[0], stride[0]):
        for left in tile_starts(width, patch_shape[1], stride[1]):
            sl = np.s_[top:top + patch_shape[0], left:left + patch_shape[1]]
            at = torch.from_numpy(a[(slice(None),) + sl][None].copy()).to(device)
            mt = torch.from_numpy(m[(slice(None),) + sl][None].copy()).to(device)
            cnn = observable_cnn(model, at, mt).cpu().numpy()[0]
            if accumulated is None:
                accumulated = np.zeros((cnn.shape[0], height, width), np.float64)
            accumulated[(slice(None),) + sl] += cnn * window[None]
            weights[sl] += window
    if accumulated is None or np.any(weights == 0):
        raise RuntimeError("CNN tiles did not cover the section")
    return (accumulated / weights[None]).astype(np.float32), a, m


def interpolate_features(features, time, column):
    """Linear time sampling on integer trace indices; never clamp out-of-domain queries."""
    time = np.asarray(time, np.float64)
    column = np.asarray(column, int)
    if np.any(time < 0) or np.any(time > features.shape[1] - 1):
        raise ValueError("Feature sample outside native time support")
    lower = np.floor(time).astype(int)
    upper = np.minimum(lower + 1, features.shape[1] - 1)
    fraction = time - lower
    return (
        features[:, lower, column] * (1 - fraction)[None]
        + features[:, upper, column] * fraction[None]
    ).T


def correspondences(rgt, valid, offsets, *, seed, times_per_trace):
    """Observable-only source candidates and three paired correspondence geometries.

For edge traces use the same inward-facing signed offsets for ALL modes.
Disruption permutes source tau among candidates on the SAME source trace;
therefore remote trace positions and tau marginals remain matched. Plateaus
are recorded, not assigned an invented order. Common-support selection rejects
unsupported/ambiguous native inverse coordinates equally across all modes.
"""
    native = NativeRGT(rgt)
    height, width = rgt.shape
    offsets = np.asarray(offsets, int)
    if offsets.min() <= 0 or offsets.max() > width // 2:
        raise ValueError("Offsets must be positive and fit both inward half-sections")
    rng = np.random.default_rng(seed)
    rows, columns, wrong_tau = [], [], []
    for x in range(width):
        supported = np.flatnonzero(valid[:, x])
        if supported.size < times_per_trace:
            continue
        t = np.sort(rng.choice(supported, times_per_trace, replace=False))
        # Nonzero cyclic permutation avoids a fixed point without reading labels.
        wrong = rgt[np.roll(t, rng.integers(1, len(t))), x]
        rows.extend(t.tolist())
        columns.extend([x] * len(t))
        wrong_tau.extend(wrong.tolist())
    t = np.asarray(rows, int)
    x = np.asarray(columns, int)
    tau = rgt[t, x].astype(np.float64)
    wrong_tau = np.asarray(wrong_tau)
    signed = np.where(x < width // 2, 1, -1)[:, None] * offsets[None]
    remote_x = x[:, None] + signed
    mapped = {"cartesian": np.repeat(t[:, None], len(offsets), axis=1).astype(float)}
    ambiguity = np.zeros((len(t), len(offsets)), bool)
    support = np.repeat(native.identifiable[t, x, None], len(offsets), axis=1)
    for mode, query in (("rgt", tau), ("disrupted", wrong_tau)):
        times = np.zeros_like(remote_x, dtype=np.float64)
        for trace in np.unique(remote_x):
            positions = remote_x == trace
            queried = np.broadcast_to(query[:, None], remote_x.shape)[positions]
            inside = (queried >= native.tau_knots[trace][0]) & (
                queried <= native.tau_knots[trace][-1]
            )
            times[positions] = native.inverse_time(trace, queried)
            support[positions] &= inside
            # Both bracketing original samples must be individually identifiable.
            lo = np.floor(times[positions]).astype(int)
            hi = np.ceil(times[positions]).astype(int)
            ambiguous = ~(native.identifiable[lo, trace] & native.identifiable[hi, trace])
            ambiguity[positions] |= ambiguous
            support[positions] &= ~ambiguous
        mapped[mode] = times
    for times in mapped.values():
        lo = np.floor(times).astype(int)
        hi = np.ceil(times).astype(int)
        support &= valid[lo, remote_x].astype(bool) & valid[hi, remote_x].astype(bool)
    keep = np.all(support, axis=1)
    # Direct inverse shift field along the remote path, observables only.
    jump = np.zeros((len(t), len(offsets)), float)
    for i in np.flatnonzero(keep):
        for k, target_x in enumerate(remote_x[i]):
            path_x = np.arange(min(x[i], target_x), max(x[i], target_x) + 1)
            path_t = np.array([native.inverse_time(int(px), tau[i]) for px in path_x])
            # Out-of-support interior traces reduce confidence instead of pruning.
            domain_ok = all(native.tau_knots[px][0] <= tau[i] <= native.tau_knots[px][-1]
                            for px in path_x)
            shifts = np.diff(path_t)
            jump[i, k] = float(np.max(np.abs(np.diff(shifts)))) if len(shifts) > 1 else 0
            if not domain_ok:
                jump[i, k] = np.inf
    return {
        "t": t, "x": x, "tau": tau, "remote_x": remote_x, "signed_offsets": signed,
        "mapped": mapped, "keep": keep, "ambiguity": ambiguity, "shift_jump": jump,
        "candidate_count": len(t), "common_support_count": int(keep.sum()),
        "ambiguous_candidate_count": int(np.any(ambiguity, axis=1).sum()),
        "source_ambiguous_count": int((~native.identifiable[t, x]).sum()),
    }


def sample_observables(cnn, avo, low, rgt, valid, config, realization_id):
    """Select and sample every input before any elastic/fault/reservoir label is read."""
    geometry = correspondences(
        rgt, valid, config["lateral_offsets"], seed=config["seed"] + realization_id,
        times_per_trace=config["candidate_times_per_trace"],
    )
    available = np.flatnonzero(geometry["keep"])
    rng = np.random.default_rng(config["seed"] + realization_id + 100000)
    count = config["queries_per_realization"]
    if len(available) < count:
        raise RuntimeError(f"Insufficient common support for realization {realization_id}")
    chosen = np.sort(rng.choice(available, count, replace=False))
    t, x = geometry["t"][chosen], geometry["x"][chosen]
    vertical, lateral = np.gradient(rgt.astype(np.float64))
    tau_scale = max(float(np.ptp(rgt)), 1e-8)
    local = np.column_stack((
        cnn[:, t, x].T, avo[:, t, x].T, low[:, t, x].T,
        (rgt[t, x] - rgt.min()) / tau_scale,
        vertical[t, x] / tau_scale, lateral[t, x] / tau_scale,
        t / (rgt.shape[0] - 1), x / (rgt.shape[1] - 1),
    ))
    features = np.concatenate((cnn, avo), axis=0)
    remotes, ava_remotes = {}, {}
    coordinates = []
    for mode in MODES:
        times = geometry["mapped"][mode][chosen]
        traces = geometry["remote_x"][chosen]
        remotes[mode] = interpolate_features(features, times.ravel(), traces.ravel()).reshape(count, -1)
        # Waveform-only controls distinguish remote AVA from CNN/prior context.
        ava_remotes[mode] = np.concatenate([
            interpolate_features(avo, np.clip(times + shift, 0, rgt.shape[0] - 1).ravel(),
                                 traces.ravel()).reshape(count, -1)
            for shift in (-2, -1, 0, 1, 2)
        ], axis=1)
        coordinates.append(times)
    rgt_times, traces = geometry["mapped"]["rgt"][chosen], geometry["remote_x"][chosen]
    prior_remote = np.concatenate([
        interpolate_features(low, np.clip(rgt_times + shift, 0, rgt.shape[0] - 1).ravel(),
                             traces.ravel()).reshape(count, -1)
        for shift in (-4, 0, 4)
    ], axis=1)
    qc = {key: geometry[key] for key in (
        "candidate_count", "common_support_count", "ambiguous_candidate_count",
        "source_ambiguous_count",
    )}
    return {
        "local": local.astype(np.float32), "prior": low[:, t, x].T.astype(np.float32),
        "remote_prior": prior_remote.astype(np.float32),
        **{f"remote_{m}": remotes[m].astype(np.float32) for m in MODES},
        **{f"ava_{m}": ava_remotes[m].astype(np.float32) for m in MODES},
        "t": t, "x": x, "remote_x": traces,
        "remote_times": np.stack(coordinates),
        "rgt_shift_jump": geometry["shift_jump"][chosen].max(axis=1),
        "dip": np.abs(lateral[t, x]) / np.maximum(np.abs(vertical[t, x]), 1e-8),
    }, qc


@dataclass
class Projection:
    mean: np.ndarray
    scale: np.ndarray
    vectors: np.ndarray
    eigenvalues: np.ndarray

    @classmethod
    def fit(cls, data, components):
        data = np.asarray(data, np.float64)
        if components > data.shape[1] or components >= data.shape[0]:
            raise ValueError("Projection budget exceeds available dimensions/samples")
        mean, scale = data.mean(axis=0), data.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1)
        standardized = (data - mean) / scale
        values, vectors = np.linalg.eigh(standardized.T @ standardized / len(data))
        order = np.argsort(values)[::-1][:components]
        return cls(mean, scale, vectors[:, order], values[order])

    def transform(self, data):
        # Whiten nondegenerate components: one shared ridge penalty across modes.
        return ((data - self.mean) / self.scale @ self.vectors) / np.sqrt(
            np.maximum(self.eigenvalues, 1e-8)
        )


@dataclass
class Ridge:
    mean: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    effective_degrees: float

    @classmethod
    def fit(cls, x, target, alpha):
        x, target = np.asarray(x, np.float64), np.asarray(target, np.float64)
        mean, target_mean = x.mean(axis=0), target.mean(axis=0)
        centered = x - mean
        gram = centered.T @ centered / len(x)
        weights = np.linalg.solve(gram + alpha * np.eye(x.shape[1]),
                                  centered.T @ (target - target_mean) / len(x))
        values = np.maximum(np.linalg.eigvalsh(gram), 0)
        return cls(mean, target_mean, weights, float(1 + np.sum(values / (values + alpha))))

    def predict(self, x):
        return (x - self.mean) @ self.weights + self.target_mean


def probe_design(train, other, config):
    """Same 64 regressors (plus intercept) for all primary probes.

Local-only gets 64 local PCs; remote probes get 48 of those exact local PCs
plus 16 remote PCs. This is a conservative information-budget replacement,
not extra parameters. All remote modes share ONE training-only PCA basis.
"""
    common, extra = config["local_common_components"], config["extra_components"]
    local = Projection.fit(train["local"], common + extra)
    remote = Projection.fit(np.concatenate([train[f"remote_{m}"] for m in MODES]), extra)
    ava = Projection.fit(np.concatenate([train[f"ava_{m}"] for m in MODES]), extra)
    prior = Projection.fit(train["remote_prior"], extra)
    train_local, other_local = local.transform(train["local"]), local.transform(other["local"])
    train_design = {"local_only": train_local}
    other_design = {"local_only": other_local}
    for name, basis, key in (
        *[(f"local_{m}", remote, f"remote_{m}") for m in MODES],
        *[(f"local_{m}_ava", ava, f"ava_{m}") for m in MODES],
        ("local_rgt_prior", prior, "remote_prior"),
    ):
        train_design[name] = np.column_stack((train_local[:, :common], basis.transform(train[key])))
        other_design[name] = np.column_stack((other_local[:, :common], basis.transform(other[key])))
    train_design["prior_calibration"] = train["prior"]
    other_design["prior_calibration"] = other["prior"]
    # Conditional innovation check: KEEP ALL 64 local PCs and remove the
    # training-estimated linear local predictability from each remote vector.
    # Compare RGT/Cartesian/disrupted at identical expanded capacity, explicitly
    # separate from the primary fixed-capacity decision.
    for m in MODES:
        remote_train = remote.transform(train[f"remote_{m}"])
        remote_other = remote.transform(other[f"remote_{m}"])
        nuisance = Ridge.fit(train_local, remote_train, alpha=1e-4)
        train_design[f"conditional_{m}"] = np.column_stack((
            train_local, remote_train - nuisance.predict(train_local),
        ))
        other_design[f"conditional_{m}"] = np.column_stack((
            other_local, remote_other - nuisance.predict(other_local),
        ))
    return train_design, other_design, {
        "local_eigenvalues": local.eigenvalues.tolist(),
        "remote_eigenvalues": remote.eigenvalues.tolist(),
        "local_feature_rank": int(np.sum(local.eigenvalues > 1e-8)),
        "remote_feature_rank": int(np.sum(remote.eigenvalues > 1e-8)),
    }


def subset(samples, mask):
    return {key: value[mask] for key, value in samples.items()}


def fit_probes(train, development, config):
    ids = np.unique(train["realization_id"])
    shuffled = np.random.default_rng(config["seed"]).permutation(ids)
    cutoff = int(len(ids) * config["inner_training_fraction"])
    fit_ids, tune_ids = shuffled[:cutoff], shuffled[cutoff:]
    if cutoff == 0 or len(tune_ids) == 0:
        raise ValueError("Need realization-disjoint internal fit/tuning sets")
    fit = subset(train, np.isin(train["realization_id"], fit_ids))
    tune = subset(train, np.isin(train["realization_id"], tune_ids))
    fit_design, tune_design, _ = probe_design(fit, tune, config)
    tuning = []
    for alpha in config["ridge_alphas"]:
        scores = [np.mean((Ridge.fit(fit_design[m], fit["target"], alpha).predict(tune_design[m])
                            - tune["target"]) ** 2) for m in PRIMARY]
        tuning.append({"alpha": alpha, "mean_primary_mse": float(np.mean(scores))})
    alpha = min(tuning, key=lambda row: row["mean_primary_mse"])["alpha"]
    train_design, dev_design, ranks = probe_design(train, development, config)
    predictions, coefficients, capacities = {}, {}, []
    for name, x in train_design.items():
        model = Ridge.fit(x, train["target"], alpha)
        predictions[name] = model.predict(dev_design[name])
        coefficients[name] = model.weights
        capacities.append({"probe": name, "inputs": x.shape[1],
                           "coefficients_including_intercepts": 3 * (x.shape[1] + 1),
                           "effective_degrees_per_property": model.effective_degrees})
    predictions["supplied_prior"] = np.zeros_like(development["target"])
    return predictions, coefficients, {
        "shared_alpha": alpha, "tuning": tuning, "inner_fit_ids": fit_ids.tolist(),
        "inner_tune_ids": tune_ids.tolist(), "capacities": capacities, **ranks,
    }


def region_masks(samples, dip_threshold, jump_threshold):
    high_dip = samples["dip"] >= dip_threshold
    fault = samples["fault_adjacent"].astype(bool)
    reservoir = samples["reservoir"].astype(bool)
    jump = samples["rgt_shift_jump"]
    reliable = np.isfinite(jump) & (jump <= jump_threshold)
    return {
        "all": np.ones(len(high_dip), bool), "high_dip": high_dip,
        "reservoir": reservoir, "fault_adjacent": fault,
        "ordinary": ~(high_dip | fault | reservoir),
        "observable_reliable": reliable, "observable_discontinuous": ~reliable,
    }


def realization_metrics(samples, predictions, normalization, masks):
    rows = []
    scales = np.asarray(normalization["y_std"])
    for rid in np.unique(samples["realization_id"]):
        for region, mask in masks.items():
            selected = mask & (samples["realization_id"] == rid)
            if selected.sum() < 4:
                continue
            target = samples["target"][selected]
            for name, prediction in predictions.items():
                mse = np.mean((prediction[selected] - target) ** 2, axis=0)
                rmse = np.sqrt(mse)
                row = {"realization_id": int(rid), "region": region, "probe": name,
                       "points": int(selected.sum()), "joint_nrmse": float(rmse.mean())}
                for i, prop in enumerate(PROPERTIES):
                    row[f"{prop}_nrmse"] = float(rmse[i])
                    row[f"{prop}_rmse"] = float(rmse[i] * scales[i])
                rows.append(row)
    return pd.DataFrame(rows)


def paired_uncertainty(metrics, config):
    """Paired whole-realization bootstrap; never treat pixels as independent units."""
    rng = np.random.default_rng(config["seed"])
    rows = []
    probes = [*PRIMARY, "supplied_prior", "prior_calibration", "local_rgt_prior",
              *[f"local_{m}_ava" for m in MODES], *[f"conditional_{m}" for m in MODES]]
    for region in metrics.region.unique():
        for metric in ("joint_nrmse", *[f"{p}_nrmse" for p in PROPERTIES]):
            wide = metrics[metrics.region == region].pivot(index="realization_id", columns="probe", values=metric)
            for baseline, candidate in combinations(probes, 2):
                values = wide[[baseline, candidate]].dropna()
                if len(values) < 2:
                    continue
                difference = (values[baseline] - values[candidate]).to_numpy()
                choices = rng.integers(0, len(values), (config["bootstrap_repetitions"], len(values)))
                boot = difference[choices].mean(axis=1)
                tail = (1 - config["confidence"]) / 2
                lo, hi = np.quantile(boot, [tail, 1 - tail])
                rows.append({
                    "region": region, "metric": metric, "baseline": baseline,
                    "candidate": candidate, "realizations": len(values),
                    "mean_improvement": float(difference.mean()), "ci_low": float(lo),
                    "ci_high": float(hi), "relative_improvement": float(difference.mean() / max(values[baseline].mean(), 1e-12)),
                    "baseline_mean": float(values[baseline].mean()),
                    "candidate_mean": float(values[candidate].mean()),
                    "positive_realizations": int((difference > 0).sum()),
                    "negative_realizations": int((difference < 0).sum()),
                })
    return pd.DataFrame(rows)


def comparison(paired, baseline, candidate, region="all", metric="joint_nrmse"):
    direct = paired[(paired.region == region) & (paired.metric == metric)
                    & (paired.baseline == baseline) & (paired.candidate == candidate)]
    if len(direct):
        return direct.iloc[0].to_dict()
    reverse = paired[(paired.region == region) & (paired.metric == metric)
                     & (paired.baseline == candidate) & (paired.candidate == baseline)]
    if len(reverse):
        row = reverse.iloc[0].to_dict()
        old_lo, old_hi = row["ci_low"], row["ci_high"]
        old_baseline, old_candidate = row["baseline_mean"], row["candidate_mean"]
        old_positive, old_negative = row["positive_realizations"], row["negative_realizations"]
        row.update(baseline=baseline, candidate=candidate,
                   mean_improvement=-row["mean_improvement"], ci_low=-old_hi, ci_high=-old_lo,
                   relative_improvement=float(-row["mean_improvement"] / max(old_candidate, 1e-12)),
                   baseline_mean=old_candidate, candidate_mean=old_baseline,
                   positive_realizations=old_negative, negative_realizations=old_positive)
        return row
    raise ValueError(f"Missing paired comparison: {baseline}/{candidate}/{region}")


def decision(paired, config, *, adequate):
    if not adequate:
        return "PROBE_INCONCLUSIVE"
    def supported(baseline, candidate):
        row = comparison(paired, baseline, candidate)
        return row["ci_low"] > 0 and row["relative_improvement"] >= config["minimum_relative_improvement"]
    rgt = supported("local_only", "local_rgt") and supported("local_disrupted", "local_rgt")
    if rgt and supported("local_cartesian", "local_rgt"):
        return "RGT_NONLOCAL_INFORMATION_SUPPORTED"
    cartesian = supported("local_only", "local_cartesian") and supported("local_disrupted", "local_cartesian")
    if rgt or cartesian:
        return "NONLOCAL_INFORMATION_USEFUL_BUT_NOT_RGT_SPECIFIC"
    return "NONLOCAL_INFORMATION_NOT_ESTABLISHED"
