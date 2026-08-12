"""Well-conditioned Wheeler fields and grouped elastic-property models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from sage_avo.data.interpretation import PreparedWell


@dataclass(frozen=True)
class WheelerFields:
    rgt_axis: np.ndarray
    delta_wheeler: np.ndarray
    porosity_wheeler: np.ndarray
    delta_time: np.ndarray
    sand_probability_time: np.ndarray
    porosity_time: np.ndarray


@dataclass(frozen=True)
class HorizonConditionedFields:
    """Reservoir properties in a local T6--T7 stratigraphic coordinate.

    ``strat_coordinate`` is zero on T6 and one on T7.  Property arrays are
    deliberately NaN outside the geometrically valid T6--T7 interval; the
    separate ``support_mask`` distinguishes well-supported interpolation from
    lateral extrapolation inside that interval.
    """

    strat_axis: np.ndarray
    strat_coordinate: np.ndarray
    delta_strat: np.ndarray
    porosity_strat: np.ndarray
    delta_time: np.ndarray
    sand_probability_time: np.ndarray
    porosity_time: np.ndarray
    reservoir_mask: np.ndarray
    support_mask: np.ndarray
    support_confidence: np.ndarray
    geometry_valid: np.ndarray
    nearest_well_distance_m: np.ndarray
    effective_well_count: np.ndarray
    wells_used: tuple[str, ...]


@dataclass(frozen=True)
class ElasticModelSet:
    models: dict[str, object]
    cross_validation: pd.DataFrame
    target_ranges: dict[str, tuple[float, float]]
    feature_columns: tuple[str, ...] = ("DELTA", "PORO", "RGT")


def _interpolate_curve(time: np.ndarray, values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    valid = np.isfinite(time) & np.isfinite(values)
    output = np.full(axis.shape, np.nan, dtype=float)
    if valid.sum() < 2:
        return output
    frame = pd.DataFrame({"time": time[valid], "value": values[valid]})
    grouped = frame.groupby("time", as_index=False)["value"].median().sort_values("time")
    inside = (axis >= grouped["time"].iloc[0]) & (axis <= grouped["time"].iloc[-1])
    output[inside] = np.interp(axis[inside], grouped["time"], grouped["value"])
    return output


def build_well_training_table(
    wells: list[PreparedWell],
    well_table: pd.DataFrame,
    time_ms: np.ndarray,
    rgt: np.ndarray,
) -> pd.DataFrame:
    """Resample logs to seismic time, preventing sub-sample pseudoreplication."""
    rows = []
    axes = np.asarray(time_ms, dtype=float)
    for well in wells:
        match = well_table[well_table["WELL"] == well.name]
        if match.empty:
            continue
        line_index = int(match.iloc[0]["LINE_INDEX"])
        logs = well.logs
        curves = {
            name: _interpolate_curve(
                logs["TWT_MS"].to_numpy(float), logs[name].to_numpy(float), axes
            )
            for name in ("DELTA", "PORO", "VP", "VS", "RHOB")
        }
        frame = pd.DataFrame(
            {
                "WELL": well.name,
                "TIME_MS": axes,
                "RGT": rgt[:, line_index],
                **curves,
            }
        )
        rows.append(frame)
    if not rows:
        raise ValueError("No wells could be resampled onto the seismic grid")
    return pd.concat(rows, ignore_index=True)


def build_reservoir_training_table(
    training: pd.DataFrame,
    well_table: pd.DataFrame,
    *,
    rgt_steering_weight: float = 0.35,
    minimum_interval_ms: float = 40.0,
    minimum_rgt_separation: float = 1e-3,
) -> pd.DataFrame:
    """Add local T6--T7 stratigraphic position to completely tied wells.

    Returned rows lie strictly inside each well's interpreted interval.  This
    prevents a reservoir elastic model from relearning the global RGT trend
    that the local horizon normalization was designed to correct.
    """
    if not 0.0 <= rgt_steering_weight <= 1.0:
        raise ValueError("rgt_steering_weight must lie in [0, 1]")
    rows = []
    for _, tie in well_table.iterrows():
        name = str(tie["WELL"])
        top = float(tie.get("T6_TWT_MS", np.nan))
        base = float(tie.get("T7_TWT_MS", np.nan))
        if not np.isfinite(top) or not np.isfinite(base) or base - top < minimum_interval_ms:
            continue
        frame = training[training["WELL"] == name].copy().sort_values("TIME_MS")
        valid_rgt = np.isfinite(frame["TIME_MS"]) & np.isfinite(frame["RGT"])
        if valid_rgt.sum() < 2:
            continue
        tau_top = float(np.interp(top, frame.loc[valid_rgt, "TIME_MS"], frame.loc[valid_rgt, "RGT"]))
        tau_base = float(np.interp(base, frame.loc[valid_rgt, "TIME_MS"], frame.loc[valid_rgt, "RGT"]))
        if tau_base - tau_top < minimum_rgt_separation:
            continue
        time_fraction = (frame["TIME_MS"] - top) / (base - top)
        rgt_fraction = (frame["RGT"] - tau_top) / (tau_base - tau_top)
        frame["STRAT_FRACTION"] = (
            (1.0 - rgt_steering_weight) * time_fraction
            + rgt_steering_weight * rgt_fraction
        )
        frame = frame[(time_fraction >= 0.0) & (time_fraction <= 1.0)]
        frame["STRAT_FRACTION"] = np.clip(frame["STRAT_FRACTION"], 0.0, 1.0)
        rows.append(frame)
    if not rows:
        raise ValueError("No wells have usable complete T6--T7 reservoir training rows")
    return pd.concat(rows, ignore_index=True)


def _well_values_on_rgt(
    training: pd.DataFrame,
    well_names: list[str],
    rgt_axis: np.ndarray,
    column: str,
) -> np.ndarray:
    values = np.full((len(well_names), rgt_axis.size), np.nan, dtype=float)
    for index, name in enumerate(well_names):
        frame = training[training["WELL"] == name]
        values[index] = _interpolate_curve(
            frame["RGT"].to_numpy(float), frame[column].to_numpy(float), rgt_axis
        )
    return values


def _distance_weighted_grid(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(values)
    numerator = np.einsum("wr,wx->rx", np.nan_to_num(values), weights)
    denominator = np.einsum("wr,wx->rx", valid.astype(float), weights)
    output = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-10,
    )
    for column in range(output.shape[1]):
        finite = np.isfinite(output[:, column])
        if finite.any():
            output[:, column] = np.interp(
                np.arange(output.shape[0]), np.flatnonzero(finite), output[finite, column]
            )
    if not np.isfinite(output).all():
        raise ValueError("Wheeler interpolation left unsupported cells")
    return output


def _unflatten(field: np.ndarray, rgt_axis: np.ndarray, rgt: np.ndarray) -> np.ndarray:
    output = np.empty(rgt.shape, dtype=float)
    for column in range(rgt.shape[1]):
        output[:, column] = np.interp(rgt[:, column], rgt_axis, field[:, column])
    return output


def _normalized_gaussian_filter(
    values: np.ndarray,
    sigma: tuple[float, float],
) -> np.ndarray:
    """Smooth finite samples without allowing NaNs to contaminate neighbors."""
    valid = np.isfinite(values)
    numerator = gaussian_filter(np.nan_to_num(values), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(float), sigma=sigma, mode="nearest")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-8,
    )


def _sample_surface_coordinate(
    coordinate: np.ndarray,
    time_ms: np.ndarray,
    surface_ms: np.ndarray,
) -> np.ndarray:
    return np.array(
        [np.interp(surface_ms[index], time_ms, coordinate[:, index]) for index in range(coordinate.shape[1])],
        dtype=float,
    )


def build_horizon_conditioned_fields(
    training: pd.DataFrame,
    wells: list[PreparedWell],
    well_table: pd.DataFrame,
    line_xy: np.ndarray,
    time_ms: np.ndarray,
    rgt: np.ndarray,
    t6_time_ms: np.ndarray,
    t7_time_ms: np.ndarray,
    *,
    n_strat: int = 201,
    lateral_distance_scale_m: float = 1_000.0,
    max_support_distance_m: float = 2_000.0,
    minimum_support_confidence: float = 0.1,
    minimum_interval_ms: float = 40.0,
    minimum_rgt_separation: float = 1e-3,
    rgt_steering_weight: float = 0.35,
    coordinate_smooth_sigma: tuple[float, float] = (1.5, 8.0),
    smooth_sigma: tuple[float, float] = (2.0, 6.0),
) -> HorizonConditionedFields:
    """Interpolate reservoir logs in a T6--T7-normalized RGT coordinate.

    For each trace, the local coordinate is

    ``s(t, x) = (tau(t, x) - tau_T6(x)) / (tau_T7(x) - tau_T6(x))``.

    Consequently T6 is exactly ``s=0`` and T7 is exactly ``s=1`` even when
    neither interpreted surface is constant RGT.  Only wells with finite T6
    and T7 time ties contribute.  Physical-distance weights are retained in
    absolute form so unsupported parts of the line can be reported rather
    than silently normalized into apparent certainty.
    """
    if n_strat < 2:
        raise ValueError("n_strat must be at least two")
    if lateral_distance_scale_m <= 0 or max_support_distance_m <= 0:
        raise ValueError("distance scales must be positive")
    if not 0.0 <= minimum_support_confidence <= 1.0:
        raise ValueError("minimum_support_confidence must lie in [0, 1]")
    if not 0.0 <= rgt_steering_weight <= 1.0:
        raise ValueError("rgt_steering_weight must lie in [0, 1]")

    times = np.asarray(time_ms, dtype=float)
    tau = np.asarray(rgt, dtype=float)
    line = np.asarray(line_xy, dtype=float)
    top = np.asarray(t6_time_ms, dtype=float)
    base = np.asarray(t7_time_ms, dtype=float)
    if tau.shape != (times.size, line.shape[0]):
        raise ValueError("rgt must have shape [time, line position]")
    if top.shape != (line.shape[0],) or base.shape != top.shape:
        raise ValueError("T6 and T7 surfaces must match the line length")

    well_lookup = {well.name: well for well in wells}
    strat_axis = np.linspace(0.0, 1.0, n_strat)
    curves: dict[str, list[np.ndarray]] = {"DELTA": [], "PORO": []}
    used_names: list[str] = []
    used_xy: list[tuple[float, float]] = []

    for _, row in well_table.iterrows():
        name = str(row["WELL"])
        if name not in well_lookup:
            continue
        well_top = float(row.get("T6_TWT_MS", np.nan))
        well_base = float(row.get("T7_TWT_MS", np.nan))
        line_index = int(row["LINE_INDEX"])
        if not np.isfinite(well_top) or not np.isfinite(well_base):
            continue
        if well_base - well_top < minimum_interval_ms:
            continue
        tau_top = float(np.interp(well_top, times, tau[:, line_index]))
        tau_base = float(np.interp(well_base, times, tau[:, line_index]))
        if tau_base - tau_top < minimum_rgt_separation:
            continue

        frame = training[training["WELL"] == name].copy()
        time_fraction = (frame["TIME_MS"] - well_top) / (well_base - well_top)
        rgt_fraction = (frame["RGT"] - tau_top) / (tau_base - tau_top)
        frame["STRAT_FRACTION"] = (
            (1.0 - rgt_steering_weight) * time_fraction
            + rgt_steering_weight * rgt_fraction
        )
        frame = frame[(time_fraction >= 0.0) & (time_fraction <= 1.0)]
        interpolated = {
            column: _interpolate_curve(
                frame["STRAT_FRACTION"].to_numpy(float),
                frame[column].to_numpy(float),
                strat_axis,
            )
            for column in curves
        }
        if any(np.isfinite(values).sum() < max(5, n_strat // 10) for values in interpolated.values()):
            continue
        for column, values in interpolated.items():
            curves[column].append(values)
        used_names.append(name)
        used_xy.append((well_lookup[name].x, well_lookup[name].y))

    if not used_names:
        raise ValueError("No wells have complete, usable T6--T7 ties")

    well_xy = np.asarray(used_xy, dtype=float)
    distances = np.linalg.norm(well_xy[:, None, :] - line[None, :, :], axis=2)
    weights = np.exp(-((distances / lateral_distance_scale_m) ** 2))
    nearest_distance = np.min(distances, axis=0)
    sum_weights = np.sum(weights, axis=0)
    support_confidence_1d = np.clip(sum_weights, 0.0, 1.0)
    effective_count = np.divide(
        sum_weights**2,
        np.sum(weights**2, axis=0),
        out=np.zeros_like(sum_weights),
        where=np.sum(weights**2, axis=0) > 0,
    )

    strat_fields: dict[str, np.ndarray] = {}
    for column, rows in curves.items():
        values = np.asarray(rows, dtype=float)
        valid = np.isfinite(values)
        numerator = np.einsum("ws,wx->sx", np.nan_to_num(values), weights)
        denominator = np.einsum("ws,wx->sx", valid.astype(float), weights)
        grid = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-10,
        )
        strat_fields[column] = _normalized_gaussian_filter(grid, smooth_sigma)

    tau_top = _sample_surface_coordinate(tau, times, top)
    tau_base = _sample_surface_coordinate(tau, times, base)
    interval_thickness = base - top
    tau_separation = tau_base - tau_top
    geometry_valid = (
        np.isfinite(top)
        & np.isfinite(base)
        & (interval_thickness >= minimum_interval_ms)
        & np.isfinite(tau_separation)
        & (tau_separation >= minimum_rgt_separation)
    )
    # Raw normalized RGT can contain short-wavelength lateral jumps even after
    # monotonic repair.  Resample it on normalized interval time, smooth there,
    # then blend it with the artifact-free normalized-time coordinate.  This
    # retains structural steering without letting unstable RGT dominate.
    raw_rgt_fraction = np.full((n_strat, line.shape[0]), np.nan, dtype=float)
    for index in np.flatnonzero(geometry_valid):
        local_times = top[index] + strat_axis * interval_thickness[index]
        local_tau = np.interp(local_times, times, tau[:, index])
        raw_rgt_fraction[:, index] = (local_tau - tau_top[index]) / tau_separation[index]
    smooth_rgt_fraction = _normalized_gaussian_filter(
        raw_rgt_fraction, coordinate_smooth_sigma
    )
    steered_fraction = np.full_like(smooth_rgt_fraction, np.nan)
    for index in np.flatnonzero(geometry_valid):
        curve = np.clip(smooth_rgt_fraction[:, index], 0.0, 1.0)
        curve = np.maximum.accumulate(curve)
        span = curve[-1] - curve[0]
        if not np.isfinite(span) or span < 1e-8:
            curve = strat_axis.copy()
        else:
            curve = (curve - curve[0]) / span
        curve = (1.0 - rgt_steering_weight) * strat_axis + rgt_steering_weight * curve
        curve[0], curve[-1] = 0.0, 1.0
        steered_fraction[:, index] = np.maximum.accumulate(curve)

    strat_coordinate = np.full(tau.shape, np.nan, dtype=float)
    for index in np.flatnonzero(geometry_valid):
        time_fraction = (times - top[index]) / interval_thickness[index]
        strat_coordinate[:, index] = np.interp(
            np.clip(time_fraction, 0.0, 1.0),
            strat_axis,
            steered_fraction[:, index],
        )
    reservoir = interval_mask(times, top, base).astype(bool) & geometry_valid[None, :]
    lateral_support = nearest_distance <= max_support_distance_m
    support = (
        reservoir
        & lateral_support[None, :]
        & (support_confidence_1d >= minimum_support_confidence)[None, :]
    )
    support_confidence = reservoir * support_confidence_1d[None, :]

    def map_reservoir(field: np.ndarray, lower: float, upper: float) -> np.ndarray:
        mapped = np.full(tau.shape, np.nan, dtype=float)
        for index in np.flatnonzero(geometry_valid):
            inside = reservoir[:, index]
            mapped[inside, index] = np.interp(
                np.clip(strat_coordinate[inside, index], 0.0, 1.0),
                strat_axis,
                field[:, index],
            )
        return np.clip(mapped, lower, upper)

    delta_time = map_reservoir(strat_fields["DELTA"], 0.0, 1.0)
    porosity_time = map_reservoir(strat_fields["PORO"], 0.0, 0.6)
    return HorizonConditionedFields(
        strat_axis=strat_axis.astype(np.float32),
        strat_coordinate=strat_coordinate.astype(np.float32),
        delta_strat=np.clip(strat_fields["DELTA"], 0.0, 1.0).astype(np.float32),
        porosity_strat=np.clip(strat_fields["PORO"], 0.0, 0.6).astype(np.float32),
        delta_time=delta_time.astype(np.float32),
        sand_probability_time=(1.0 - delta_time).astype(np.float32),
        porosity_time=porosity_time.astype(np.float32),
        reservoir_mask=reservoir.astype(np.uint8),
        support_mask=support.astype(np.uint8),
        support_confidence=support_confidence.astype(np.float32),
        geometry_valid=geometry_valid.astype(np.uint8),
        nearest_well_distance_m=nearest_distance.astype(np.float32),
        effective_well_count=effective_count.astype(np.float32),
        wells_used=tuple(used_names),
    )


def blend_horizon_conditioned_background(
    background: np.ndarray,
    reservoir: np.ndarray,
    strat_coordinate: np.ndarray,
    support_confidence: np.ndarray,
    *,
    edge_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend reservoir properties into a regional field without hard seams.

    The weight is zero at T6/T7, rises smoothly over ``edge_fraction`` of the
    interval, and is reduced continuously where absolute well support is weak.
    The returned weight is provenance and should be saved with the composite.
    """
    base = np.asarray(background, dtype=float)
    conditioned = np.asarray(reservoir, dtype=float)
    coordinate = np.asarray(strat_coordinate, dtype=float)
    confidence = np.asarray(support_confidence, dtype=float)
    if not (base.shape == conditioned.shape == coordinate.shape == confidence.shape):
        raise ValueError("background, reservoir, coordinate, and confidence must match")
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must lie in (0, 0.5]")

    distance_to_edge = np.minimum(coordinate, 1.0 - coordinate)
    ramp = np.clip(distance_to_edge / edge_fraction, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # cubic smoothstep
    weight = np.where(
        np.isfinite(conditioned) & np.isfinite(coordinate),
        ramp * np.clip(confidence, 0.0, 1.0),
        0.0,
    )
    output = (1.0 - weight) * base + weight * np.nan_to_num(conditioned, nan=0.0)
    return output.astype(np.float32), weight.astype(np.float32)


def build_wheeler_fields(
    training: pd.DataFrame,
    wells: list[PreparedWell],
    line_xy: np.ndarray,
    rgt: np.ndarray,
    *,
    n_rgt: int = 520,
    lateral_distance_scale_m: float = 1_500.0,
    smooth_sigma: tuple[float, float] = (2.0, 8.0),
) -> WheelerFields:
    """Interpolate DELTA and porosity in RGT using physical well distances."""
    if n_rgt < 2 or lateral_distance_scale_m <= 0:
        raise ValueError("n_rgt and lateral_distance_scale_m must be positive")
    well_names = [well.name for well in wells]
    well_xy = np.array([[well.x, well.y] for well in wells], dtype=float)
    line = np.asarray(line_xy, dtype=float)
    distances = np.linalg.norm(well_xy[:, None, :] - line[None, :, :], axis=2)
    weights = np.exp(-((distances / lateral_distance_scale_m) ** 2))
    rgt_axis = np.linspace(float(np.min(rgt)), float(np.max(rgt)), n_rgt)
    delta_values = _well_values_on_rgt(training, well_names, rgt_axis, "DELTA")
    porosity_values = _well_values_on_rgt(training, well_names, rgt_axis, "PORO")
    delta_wheeler = gaussian_filter(
        _distance_weighted_grid(delta_values, weights), sigma=smooth_sigma, mode="nearest"
    )
    porosity_wheeler = gaussian_filter(
        _distance_weighted_grid(porosity_values, weights), sigma=smooth_sigma, mode="nearest"
    )
    delta_wheeler = np.clip(delta_wheeler, 0.0, 1.0)
    porosity_wheeler = np.clip(porosity_wheeler, 0.0, 0.6)
    delta_time = np.clip(_unflatten(delta_wheeler, rgt_axis, rgt), 0.0, 1.0)
    porosity_time = np.clip(_unflatten(porosity_wheeler, rgt_axis, rgt), 0.0, 0.6)
    return WheelerFields(
        rgt_axis=rgt_axis.astype(np.float32),
        delta_wheeler=delta_wheeler.astype(np.float32),
        porosity_wheeler=porosity_wheeler.astype(np.float32),
        delta_time=delta_time.astype(np.float32),
        sand_probability_time=(1.0 - delta_time).astype(np.float32),
        porosity_time=porosity_time.astype(np.float32),
    )


def fit_grouped_elastic_models(
    training: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = ("DELTA", "PORO", "RGT"),
    seed: int = 12345,
    n_estimators: int = 150,
    min_samples_leaf: int = 5,
) -> ElasticModelSet:
    """Validate by leave-one-well-out folds, then fit all authorized wells."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import LeaveOneGroupOut
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("Elastic model fitting requires scikit-learn") from error
    features = list(feature_columns)
    if len(features) != 3:
        raise ValueError("elastic models require exactly three feature columns")
    targets = ["VP", "VS", "RHOB"]
    clean = training.dropna(subset=["WELL", *features, *targets]).copy()
    if clean["WELL"].nunique() < 2:
        raise ValueError("Grouped validation requires at least two wells")
    x = clean[features].to_numpy(float)
    groups = clean["WELL"].to_numpy(str)
    rows = []
    models: dict[str, object] = {}
    ranges: dict[str, tuple[float, float]] = {}
    logo = LeaveOneGroupOut()
    for target in targets:
        y = clean[target].to_numpy(float)
        for train_index, test_index in logo.split(x, y, groups):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_leaf=min_samples_leaf,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(x[train_index], y[train_index])
            prediction = model.predict(x[test_index])
            rows.append(
                {
                    "target": target,
                    "held_out_well": str(groups[test_index][0]),
                    "samples": int(test_index.size),
                    "rmse": float(np.sqrt(mean_squared_error(y[test_index], prediction))),
                    "mae": float(mean_absolute_error(y[test_index], prediction)),
                    "r2": float(r2_score(y[test_index], prediction)),
                }
            )
        final_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )
        final_model.fit(x, y)
        models[target] = final_model
        ranges[target] = (float(np.quantile(y, 0.005)), float(np.quantile(y, 0.995)))
    return ElasticModelSet(models, pd.DataFrame(rows), ranges, tuple(features))


def predict_elastic_fields(
    model_set: ElasticModelSet,
    delta: np.ndarray,
    porosity: np.ndarray,
    rgt: np.ndarray,
) -> np.ndarray:
    """Predict clipped Vp, Vs, and RHOB grids ordered as [3, time, CDP]."""
    shape = np.asarray(delta).shape
    if np.asarray(porosity).shape != shape or np.asarray(rgt).shape != shape:
        raise ValueError("delta, porosity, and rgt must have matching shapes")
    features = np.column_stack([np.ravel(delta), np.ravel(porosity), np.ravel(rgt)])
    valid = np.all(np.isfinite(features), axis=1)
    if not valid.any():
        raise ValueError("no finite feature rows are available for elastic prediction")
    outputs = []
    for target in ("VP", "VS", "RHOB"):
        prediction = np.full(features.shape[0], np.nan, dtype=float)
        prediction[valid] = model_set.models[target].predict(features[valid])
        lower, upper = model_set.target_ranges[target]
        outputs.append(np.clip(prediction, lower, upper).reshape(shape))
    return np.asarray(outputs, dtype=np.float32)


def interval_mask(time_ms: np.ndarray, top_ms: np.ndarray, base_ms: np.ndarray) -> np.ndarray:
    """Return a [time, CDP] mask between two interpreted time surfaces."""
    time = np.asarray(time_ms, dtype=float)[:, None]
    top = np.asarray(top_ms, dtype=float)[None, :]
    base = np.asarray(base_ms, dtype=float)[None, :]
    if top.shape[1] != base.shape[1]:
        raise ValueError("top_ms and base_ms must have matching lengths")
    return ((time >= np.minimum(top, base)) & (time <= np.maximum(top, base))).astype(np.uint8)
