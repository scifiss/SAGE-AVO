from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sage_avo.data.field import (
    FieldLineStacks,
    load_field_line_stacks,
    save_field_line_stacks,
    stack_segy_line,
)
from sage_avo.data.layout import DataLayout
from sage_avo.geology.field_conditioning import interval_mask
from sage_avo.geology.field_conditioning import (
    blend_horizon_conditioned_background,
    build_horizon_conditioned_fields,
    build_reservoir_training_table,
)
from sage_avo.data.interpretation import PreparedWell
from sage_avo.structure.rgt import (
    PwdRgtResult,
    load_pwd_rgt,
    repair_rgt_monotonicity,
    refine_rgt_with_horizons,
    save_pwd_rgt,
)


def test_field_stack_archive_round_trip(tmp_path: Path):
    stacks = FieldLineStacks(
        avo=np.arange(3 * 4 * 2, dtype=np.float32).reshape(3, 4, 2),
        seismic_structure=np.ones((4, 2), dtype=np.float32),
        band_fold=np.full((3, 2), 10, dtype=np.int32),
        structure_fold=np.full(2, 30, dtype=np.int32),
        time_ms=np.arange(4, dtype=np.float32) * 4.0,
        cdps=np.array([100, 101], dtype=np.int32),
        line_xy=np.array([[0.0, 1.0], [2.0, 3.0]]),
        band_names=("near", "mid", "far"),
        band_limits_degrees=((3.0, 17.0), (17.0, 31.0), (31.0, 45.0)),
    )
    path = tmp_path / "field.npz"
    save_field_line_stacks(path, stacks)
    loaded = load_field_line_stacks(path)
    np.testing.assert_array_equal(loaded.avo, stacks.avo)
    assert loaded.band_names == stacks.band_names


def test_pwd_rgt_archive_round_trip(tmp_path: Path):
    result = PwdRgtResult(
        dip=np.zeros((4, 2), dtype=np.float32),
        rgt=np.arange(4, dtype=np.float32)[:, None].repeat(2, axis=1),
        structure_oriented_seismic=np.ones((4, 2), dtype=np.float32),
        reference_sample=2,
    )
    path = tmp_path / "rgt.npz"
    save_pwd_rgt(path, result)
    loaded = load_pwd_rgt(path)
    np.testing.assert_array_equal(loaded.rgt, result.rgt)
    assert loaded.reference_sample == 2


def test_offset_header_requires_explicit_angle_semantics():
    with pytest.raises(ValueError, match="Refusing to interpret"):
        stack_segy_line(
            "not_read.segy",
            bands_degrees={"near": (3.0, 17.0)},
            time_window_ms=(2000.0, 3200.0),
            midpoint_y_max=None,
            angle_header_semantics="unknown",
        )


def test_data_layout_separates_raw_and_outputs(tmp_path: Path):
    raw = tmp_path / "external_raw"
    layout = DataLayout(tmp_path / "work", "s01data", "v001", raw_root=raw)
    assert layout.raw == raw.resolve()
    assert layout.usable == (tmp_path / "work" / "s01data" / "usable" / "v001").resolve()
    layout.ensure_outputs("usable", "attributes")
    assert layout.usable.is_dir()
    assert not raw.exists()


def test_rgt_repair_is_strictly_monotonic():
    raw = np.array([[0.0, 0.0], [1.0, 0.8], [0.7, 1.6], [2.0, 2.4]])
    repaired, qc = repair_rgt_monotonicity(raw, minimum_step=1e-5)
    assert np.all(np.diff(repaired, axis=0) >= 9e-6)
    assert qc["adjustment_max_absolute"] > 0.0


def test_regularized_horizon_refinement_is_optional_monotonic_and_improves_ties():
    time_ms = np.arange(80, dtype=float) * 4.0
    columns = np.arange(21, dtype=float)
    base = np.arange(80, dtype=float)[:, None] / 79.0
    base = base + 0.025 * np.sin(columns / 4.0)[None]
    top = 100.0 + 18.0 * np.sin(columns / 5.0)
    base_horizon = 220.0 + 22.0 * np.sin(columns / 5.0 + 0.4)
    disabled = refine_rgt_with_horizons(
        base,
        time_ms,
        {"T6": top, "T7": base_horizon},
        enabled=False,
    )
    np.testing.assert_array_equal(disabled.rgt, base.astype(np.float32))
    result = refine_rgt_with_horizons(
        base,
        time_ms,
        {"T6": top, "T7": base_horizon},
        horizon_weight=0.6,
        maximum_correction_rgt=0.08,
    )
    assert np.all(np.diff(result.rgt, axis=0) > 0.0)
    for name in ("T6", "T7"):
        pre = result.qc["pre_horizon_residuals"][name]["rmse_ms"]
        post = result.qc["post_horizon_residuals"][name]["rmse_ms"]
        assert post < pre
    assert result.qc["adjustment_max_absolute"] <= 0.081


def test_interval_mask_uses_interpreted_surfaces():
    mask = interval_mask(
        np.array([0.0, 4.0, 8.0, 12.0]),
        np.array([4.0, 8.0]),
        np.array([8.0, 12.0]),
    )
    np.testing.assert_array_equal(mask.sum(axis=0), np.array([2, 2]))


def test_horizon_conditioned_coordinate_honors_both_surfaces():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    line_xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    rgt = time[:, None] + np.array([[0.0, 0.2, 0.4]])
    top = np.array([1.0, 1.0, 1.0])
    base = np.array([2.0, 2.0, 2.0])
    logs = np.zeros(1)
    wells = [
        PreparedWell("A", 0.0, 0.0, logs, "A", {}),
        PreparedWell("B", 2.0, 0.0, logs, "B", {}),
    ]
    well_table = pd.DataFrame(
        {
            "WELL": ["A", "B"],
            "LINE_INDEX": [0, 2],
            "T6_TWT_MS": [1.0, 1.0],
            "T7_TWT_MS": [2.0, 2.0],
        }
    )
    training = pd.DataFrame(
        {
            "WELL": ["A"] * 4 + ["B"] * 4,
            "TIME_MS": np.tile(time, 2),
            "RGT": np.concatenate([rgt[:, 0], rgt[:, 2]]),
            "DELTA": np.tile([0.2, 0.3, 0.7, 0.8], 2),
            "PORO": np.tile([0.05, 0.06, 0.04, 0.03], 2),
        }
    )
    fields = build_horizon_conditioned_fields(
        training,
        wells,
        well_table,
        line_xy,
        time,
        rgt,
        top,
        base,
        n_strat=11,
        minimum_interval_ms=0.5,
        smooth_sigma=(0.0, 0.0),
    )
    np.testing.assert_allclose(fields.strat_coordinate[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(fields.strat_coordinate[2], 1.0, atol=1e-6)
    assert np.isnan(fields.delta_time[0]).all()
    assert np.isnan(fields.delta_time[3]).all()


def test_reservoir_blend_has_no_hard_horizon_seam():
    background = np.zeros((5, 1))
    reservoir = np.ones((5, 1))
    coordinate = np.array([[np.nan], [0.0], [0.5], [1.0], [np.nan]])
    confidence = np.ones((5, 1))
    composite, weight = blend_horizon_conditioned_background(
        background, reservoir, coordinate, confidence, edge_fraction=0.25
    )
    np.testing.assert_allclose(weight[:, 0], [0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(composite[:, 0], [0.0, 0.0, 1.0, 0.0, 0.0])


def test_reservoir_training_uses_only_completely_tied_interval_rows():
    training = pd.DataFrame(
        {
            "WELL": ["A"] * 5 + ["B"] * 5,
            "TIME_MS": np.tile(np.arange(5.0), 2),
            "RGT": np.tile(np.arange(5.0), 2),
            "DELTA": 0.5,
            "PORO": 0.1,
            "VP": 3000.0,
            "VS": 1700.0,
            "RHOB": 2.3,
        }
    )
    ties = pd.DataFrame(
        {
            "WELL": ["A", "B"],
            "T6_TWT_MS": [1.0, np.nan],
            "T7_TWT_MS": [3.0, 3.0],
        }
    )
    reservoir = build_reservoir_training_table(
        training, ties, minimum_interval_ms=1.0
    )
    assert reservoir["WELL"].unique().tolist() == ["A"]
    np.testing.assert_allclose(reservoir["STRAT_FRACTION"], [0.0, 0.5, 1.0])
