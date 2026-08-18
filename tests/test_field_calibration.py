from pathlib import Path

import numpy as np
import pytest

from sage_avo.evaluation.field_calibration import (
    FieldDiagnosticThresholds,
    FieldTransferSpecification,
    apply_field_transfer,
    field_domain_diagnostics,
    load_passing_field_calibration,
    prepare_calibrated_field_input,
    prepare_calibrated_field_observation,
    save_field_calibration_manifest,
)


def _data() -> np.ndarray:
    rows, columns = np.indices((128, 25))
    base = np.sin(2.0 * np.pi * rows / 16.0 + columns / 40.0)
    return np.stack((base, 0.8 * base, 0.6 * base)).astype(np.float32)


def _thresholds() -> FieldDiagnosticThresholds:
    return FieldDiagnosticThresholds(
        minimum_polarity_correlation=0.99,
        maximum_absolute_phase_degrees=1.0,
        amplitude_scale_minimum=0.99,
        amplitude_scale_maximum=1.01,
        minimum_normalized_percentile_overlap=0.99,
        maximum_spatial_rms_coefficient_of_variation=0.02,
        maximum_spectral_peak_difference_hz=0.1,
    )


def _identity_transfer() -> FieldTransferSpecification:
    return FieldTransferSpecification(
        transfer_id="reviewed_identity",
        gain_by_band=(1.0, 1.0, 1.0),
        phase_degrees_by_band=(0.0, 0.0, 0.0),
        polarity_by_band=(1, 1, 1),
    )


def test_paired_identity_diagnostics_can_pass_explicit_thresholds():
    data = _data()
    report = field_domain_diagnostics(
        data,
        data.copy(),
        dt_seconds=0.004,
        synthetic_x_mean=(0.0, 0.0, 0.0),
        synthetic_x_std=(1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    assert report["status"] == "pass"
    assert all(all(item["checks"].values()) for item in report["bands"])


def test_unpaired_diagnostics_cannot_pass_phase_and_polarity():
    data = _data()
    report = field_domain_diagnostics(
        data[:, :, :-1],
        data,
        dt_seconds=0.004,
        synthetic_x_mean=(0.0, 0.0, 0.0),
        synthetic_x_std=(1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    assert report["status"] == "fail"
    assert not report["bands"][0]["checks"]["polarity"]


def test_manifest_guard_blocks_missing_or_mismatched_calibration(tmp_path: Path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="no saved calibration"):
        load_passing_field_calibration(
            missing, expected_forward_specification_sha256="forward-a"
        )
    report = field_domain_diagnostics(
        _data(),
        _data(),
        dt_seconds=0.004,
        synthetic_x_mean=(0.0, 0.0, 0.0),
        synthetic_x_std=(1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    path = tmp_path / "calibration.json"
    save_field_calibration_manifest(
        path,
        diagnostics=report,
        transfer=_identity_transfer(),
        forward_specification_sha256="forward-a",
        approved_by="unit-test review",
    )
    with pytest.raises(RuntimeError, match="specifications differ"):
        load_passing_field_calibration(
            path, expected_forward_specification_sha256="forward-b"
        )


def test_field_is_transferred_before_training_normalization(tmp_path: Path):
    data = _data()
    transfer = FieldTransferSpecification(
        transfer_id="gain-polarity-test",
        gain_by_band=(2.0, 0.5, 1.0),
        phase_degrees_by_band=(0.0, 0.0, 0.0),
        polarity_by_band=(-1, 1, 1),
    )
    transferred = apply_field_transfer(data, transfer)
    np.testing.assert_allclose(transferred[0], -2.0 * data[0])
    report = field_domain_diagnostics(
        transferred,
        transferred.copy(),
        dt_seconds=0.004,
        synthetic_x_mean=(0.0, 0.0, 0.0),
        synthetic_x_std=(1.0, 1.0, 1.0),
        thresholds=_thresholds(),
    )
    path = tmp_path / "calibration.json"
    save_field_calibration_manifest(
        path,
        diagnostics=report,
        transfer=transfer,
        forward_specification_sha256="forward-a",
        approved_by="unit-test review",
    )
    normalized = prepare_calibrated_field_input(
        data,
        calibration_manifest=path,
        expected_forward_specification_sha256="forward-a",
        normalization={"x_mean": [0.0, 0.0, 0.0], "x_std": [2.0, 2.0, 2.0]},
    )
    np.testing.assert_allclose(normalized, transferred / 2.0)
    transferred_only = prepare_calibrated_field_observation(
        data,
        calibration_manifest=path,
        expected_forward_specification_sha256="forward-a",
    )
    np.testing.assert_allclose(transferred_only, transferred)
