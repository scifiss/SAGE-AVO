"""Public synthetic-fixture checks; no private dataset/checkpoint is loaded."""

import numpy as np
import pandas as pd
import pytest
import torch

from sage_avo.diagnostics.nonlocal_information import (
    PRIMARY, Projection, Ridge, comparison, correspondences, decision,
    fit_probes, interpolate_features, observable_cnn, paired_uncertainty,
    probe_design, realization_metrics,
)
from sage_avo.models.sage_avo import SAGEAVO


def test_observable_t0_encoder_matches_unmodified_dense_forward_and_freezes_weights():
    torch.manual_seed(4)
    model = SAGEAVO(hidden_channels=16, graph_heads=4).eval()
    model.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    low, avo = torch.randn(1, 3, 7, 9), torch.randn(1, 3, 7, 9)
    rgt = torch.arange(7)[None, :, None].expand(1, 7, 9).float()
    captured = []
    hook = model.encoder.register_forward_hook(lambda module, inputs, output: captured.append(output))
    with torch.inference_mode():
        model(low, torch.zeros(1), avo, low, rgt)
    hook.remove()
    observed = observable_cnn(model, avo, low)
    torch.testing.assert_close(observed, captured[0], rtol=0, atol=0)
    assert not observed.requires_grad
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_correspondences_equal_offsets_native_rgt_dipping_alignment_and_disruption():
    t, x = np.indices((80, 32))
    rgt = t.astype(float) - 0.4 * x
    geometry = correspondences(rgt, np.ones_like(rgt, bool), [4, 8, 12], seed=7, times_per_trace=12)
    keep = geometry["keep"]
    qx, qt = geometry["x"][keep], geometry["t"][keep]
    expected = qt[:, None] + 0.4 * (geometry["remote_x"][keep] - qx[:, None])
    np.testing.assert_allclose(geometry["mapped"]["rgt"][keep], expected)
    assert np.all(np.abs(geometry["signed_offsets"]) == [4, 8, 12])
    assert np.all(geometry["mapped"]["cartesian"] == geometry["t"][:, None])
    assert np.median(np.abs(geometry["mapped"]["rgt"][keep]
                            - geometry["mapped"]["disrupted"][keep])) > 5
    assert geometry["common_support_count"] > 0


def test_plateaus_are_reported_ambiguous_not_given_artificial_order():
    rgt = np.repeat(np.arange(60, dtype=float)[:, None], 24, axis=1)
    rgt[20:30] = 20
    result = correspondences(rgt, np.ones_like(rgt, bool), [4, 8], seed=12, times_per_trace=40)
    assert result["source_ambiguous_count"] > 0
    assert np.all(~result["keep"][(result["t"] >= 20) & (result["t"] < 30)])


def test_feature_interpolation_is_fractional_and_rejects_hidden_clamping():
    features = np.arange(20).reshape(1, 10, 2).astype(float)
    np.testing.assert_allclose(interpolate_features(features, [2.5, 4], [0, 1]), [[5], [9]])
    with pytest.raises(ValueError, match="outside"):
        interpolate_features(features, [-0.01], [0])


def synthetic_samples(seed, cases=10, points=32):
    rng = np.random.default_rng(seed)
    n = cases * points
    local = rng.normal(size=(n, 12))
    rgt = rng.normal(size=(n, 9))
    return {
        "local": local, "remote_rgt": rgt,
        "remote_cartesian": rng.normal(size=(n, 9)),
        "remote_disrupted": rng.normal(size=(n, 9)),
        "ava_rgt": rgt.copy(), "ava_cartesian": rng.normal(size=(n, 9)),
        "ava_disrupted": rng.normal(size=(n, 9)),
        "remote_prior": rng.normal(size=(n, 9)), "prior": local[:, :3],
        "realization_id": np.repeat(np.arange(cases) + seed * 100, points),
        "target": np.column_stack((rgt[:, 0], rgt[:, 1], rgt[:, 2])),
    }


def fixture_config():
    return {"seed": 12, "local_common_components": 4, "extra_components": 4,
            "ridge_alphas": [0.001, 0.01], "inner_training_fraction": 0.8,
            "bootstrap_repetitions": 300, "confidence": 0.95,
            "minimum_relative_improvement": 0.01}


def test_probe_capacity_equal_and_pca_never_depends_on_development_values():
    train, dev, config = synthetic_samples(1), synthetic_samples(2), fixture_config()
    design, other, _ = probe_design(train, dev, config)
    assert all(design[m].shape[1] == 8 for m in PRIMARY)
    for m in PRIMARY[1:]:
        np.testing.assert_allclose(design[m][:, :4], design["local_only"][:, :4])
    shifted = {k: v + 1000 if k != "realization_id" else v for k, v in dev.items()}
    second, _, _ = probe_design(train, shifted, config)
    for key in design:
        np.testing.assert_array_equal(design[key], second[key])
    assert other["local_rgt"].shape[1] == 8


def test_positive_control_detects_remote_information_without_model_training():
    train, dev, config = synthetic_samples(1), synthetic_samples(2), fixture_config()
    # Keep all nine remote dimensions in the fixture's information bottleneck.
    config.update(local_common_components=2, extra_components=9)
    prediction, _, fitting = fit_probes(train, dev, config)
    local_error = np.mean((prediction["local_only"] - dev["target"]) ** 2)
    aligned_error = np.mean((prediction["local_rgt"] - dev["target"]) ** 2)
    disrupted_error = np.mean((prediction["local_disrupted"] - dev["target"]) ** 2)
    assert aligned_error < local_error * 0.02
    assert aligned_error < disrupted_error * 0.02
    assert set(fitting["inner_fit_ids"]).isdisjoint(fitting["inner_tune_ids"])
    assert set(fitting["inner_tune_ids"]).isdisjoint(np.unique(dev["realization_id"]))
    assert len({row["coefficients_including_intercepts"] for row in fitting["capacities"] if row["probe"] in PRIMARY}) == 1


def test_realization_uncertainty_and_reversed_comparison_decision():
    rows = []
    probes = {"local_only": 1.0, "local_cartesian": 0.95,
              "local_rgt": 0.7, "local_disrupted": 1.01}
    for rid in range(12):
        for name, value in probes.items():
            row = {"realization_id": rid, "region": "all", "probe": name,
                   "joint_nrmse": value + rid / 1000}
            row.update({f"{p}_nrmse": row["joint_nrmse"] for p in ("Vp", "Vs", "density")})
            rows.append(row)
    # paired_uncertainty intentionally includes descriptive controls as well.
    for rid in range(12):
        template = rows[4 * rid]
        for name in ("supplied_prior", "prior_calibration", "local_rgt_prior",
                     "local_rgt_ava", "local_cartesian_ava", "local_disrupted_ava",
                     "conditional_rgt", "conditional_cartesian", "conditional_disrupted"):
            rows.append({**template, "probe": name})
    paired = paired_uncertainty(pd.DataFrame(rows), fixture_config())
    reverse = comparison(paired, "local_disrupted", "local_rgt")
    assert reverse["ci_low"] > 0 and reverse["relative_improvement"] > 0
    assert reverse["realizations"] == 12
    assert decision(paired, fixture_config(), adequate=True) == "RGT_NONLOCAL_INFORMATION_SUPPORTED"
    assert decision(paired, fixture_config(), adequate=False) == "PROBE_INCONCLUSIVE"


def test_prior_relative_targets_and_property_units_are_separate():
    samples = {"realization_id": np.repeat([1, 2], 5), "target": np.ones((10, 3))}
    metrics = realization_metrics(samples, {"supplied_prior": np.zeros((10, 3))},
                                  {"y_std": [100, 50, 0.1]}, {"all": np.ones(10, bool)})
    assert (metrics.Vp_rmse == 100).all()
    assert (metrics.Vs_rmse == 50).all()
    assert (metrics.density_rmse == 0.1).all()
    assert (metrics.joint_nrmse == 1).all()


def test_ridge_and_projection_remain_finite_on_constant_columns():
    rng = np.random.default_rng(5)
    x = np.column_stack((rng.normal(size=(100, 4)), np.ones(100)))
    projection = Projection.fit(x, 5)
    z = projection.transform(x)
    assert np.isfinite(z).all()
    model = Ridge.fit(z, x[:, :3], 0.01)
    assert np.isfinite(model.predict(z)).all()
