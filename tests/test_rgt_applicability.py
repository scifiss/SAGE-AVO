"""Regression tests for the observable-only v00332s applicability screen."""
import numpy as np
import pytest

from sage_avo.diagnostics.rgt_applicability import (
    applicability_fields, applicability_signal, heldout_binned_prediction, score_bins,
)

THRESHOLDS = {"normalized_best_mismatch": 11.291821481956012,
              "normalized_cartesian_discontinuity": 14.577536450477627,
              "rgt_dip_residual": 14.3273604826546}


def test_horizontal_layers_have_no_structural_advantage():
    tau = np.broadcast_to(np.arange(20)[:, None], (20, 10)).astype(np.float32)
    f = applicability_fields(tau, THRESHOLDS)
    assert np.all(f["S"] == 0)
    assert np.all(f["displacement"] == 0)
    assert f["confidence_safe"].all()


def test_dipping_linear_layers_improve_by_one_sample():
    y, x = np.indices((20, 10))
    f = applicability_fields((y + x).astype(np.float32), THRESHOLDS)
    np.testing.assert_allclose(f["S"][3:-3], 1., rtol=3e-6)
    assert np.all(f["displacement"][3:-3] == -1)
    assert np.all(f["m_R"][3:-3] == 0)


def test_unidentifiable_scale_is_not_high_informativeness():
    f = applicability_fields(np.zeros((12, 8), np.float32), THRESHOLDS)
    assert np.isnan(f["S"]).all()
    assert not f["scale_valid"].any()
    assert np.all(score_bins(f["S"], [.5, 1.]) == -1)


def test_numerical_score_scale_is_unit_invariant_and_inputs_not_mutated():
    y, x = np.indices((20, 10))
    tau = (y + x).astype(np.float32)
    original = tau.copy()
    a = applicability_fields(tau, THRESHOLDS)
    b = applicability_fields(tau * 8, THRESHOLDS)
    np.testing.assert_allclose(a["S"], b["S"])
    np.testing.assert_array_equal(tau, original)


def test_score_and_reliability_are_separate():
    y, x = np.indices((20, 10))
    tau = (y + 20 * x).astype(np.float32)
    f = applicability_fields(tau, THRESHOLDS)
    assert (f["S"][3:-3] > 2.9).all()
    assert not f["confidence_safe"][3:-3].any()


def test_bins_keep_zero_and_undefined_separate():
    values = np.array([np.nan, -.01, 0., .1, .5, 1., 2.])
    np.testing.assert_array_equal(score_bins(values, [.5, 1.]), [-1, 0, 0, 1, 2, 3, 3])
    with pytest.raises(ValueError):
        score_bins(values, [1., 1.])


def test_signal_screen_requires_reproducibility_and_coverage():
    assert applicability_signal([.02]*3, [.01]*3, [6]*3)["passed"]
    assert not applicability_signal([.02, -.01, .02], [.01]*3, [6]*3)["passed"]
    assert not applicability_signal([.02]*3, [.01]*3, [3]*3)["passed"]
    assert not applicability_signal([.02]*3, [.001]*3, [6]*3)["passed"]
    assert not applicability_signal([np.nan]*3, [.01]*3, [6]*3)["passed"]


def test_heldout_bin_predictor_detects_stable_signal():
    groups = np.array([[[10, -10, 10], [10, 10, 10]]] * 3, dtype=float)
    result = heldout_binned_prediction(groups)
    assert result["heldout_equal_realization_mse"] == 0.
    assert result["relative_mse_improvement_over_intercept"] == 1.
