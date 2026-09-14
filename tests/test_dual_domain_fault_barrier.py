import numpy as np

from sage_avo.diagnostics.dual_domain_fault_barrier import (
    barrier_safe,
    inverse_shift_fields,
)


def test_shift_barrier_responds_to_abrupt_change_not_constant_dip():
    tau, x = np.indices((20, 30))
    smooth = 0.8 * x + 0.01 * tau
    fields = inverse_shift_fields(smooth, np.ones_like(smooth, bool))
    assert np.quantile(fields["normalized_shift_jump"], 0.95) < 1e-6
    broken = smooth.copy()
    broken[:, 15:] += 8
    fields = inverse_shift_fields(broken, np.ones_like(broken, bool))
    assert fields["normalized_shift_jump"][:, 14].mean() > 10


def test_preunion_barrier_preserves_metadata_eligible_match():
    thresholds = {
        "flattened_waveform_cosine": 0.8,
        "flattened_phase_cosine": 0.5,
        "strength_ratio": 0.2,
        "normalized_shift_jump": 4.0,
        "normalized_shift_second_difference": 4.0,
        "original_waveform_cosine": 0.7,
        "original_phase_cosine": 0.4,
    }
    row = {
        "flattened_waveform_cosine": 0.95,
        "flattened_phase_cosine": 0.9,
        "strength_ratio": 0.8,
        "normalized_shift_jump": 8.0,
        "normalized_shift_second_difference": 7.0,
        "original_waveform_cosine": 0.9,
        "original_phase_cosine": 0.8,
    }
    matched, safe = barrier_safe(row, thresholds)
    assert matched
    assert not safe
