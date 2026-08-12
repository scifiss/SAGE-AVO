"""Streaming readers for the private S01 field line.

The source research notebook loaded the complete multi-gigabyte SEG-Y into
memory before stacking.  This module preserves the numerical operation (a
mean over traces in each CDP/angle bin) while reading traces in bounded chunks.
No field-data path is embedded in the public package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SegySummary:
    """Small, serializable description of a SEG-Y file and its key headers."""

    path: str
    trace_count: int
    sample_count: int
    sample_start_ms: float
    sample_end_ms: float
    sample_interval_ms: float
    cdp_min: int
    cdp_max: int
    angle_header_min: int
    angle_header_max: int
    angle_header_values: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "trace_count": self.trace_count,
            "sample_count": self.sample_count,
            "sample_start_ms": self.sample_start_ms,
            "sample_end_ms": self.sample_end_ms,
            "sample_interval_ms": self.sample_interval_ms,
            "cdp_min": self.cdp_min,
            "cdp_max": self.cdp_max,
            "angle_header_min": self.angle_header_min,
            "angle_header_max": self.angle_header_max,
            "angle_header_values": list(self.angle_header_values),
        }


@dataclass(frozen=True)
class FieldLineStacks:
    """Real S01 line geometry, three AVO bands, and a full structural stack."""

    avo: np.ndarray
    seismic_structure: np.ndarray
    band_fold: np.ndarray
    structure_fold: np.ndarray
    time_ms: np.ndarray
    cdps: np.ndarray
    line_xy: np.ndarray
    band_names: tuple[str, ...]
    band_limits_degrees: tuple[tuple[float, float], ...]

    def validate(self) -> None:
        if self.avo.ndim != 3:
            raise ValueError("avo must have shape [band, time, cdp]")
        n_band, n_time, n_cdp = self.avo.shape
        if self.seismic_structure.shape != (n_time, n_cdp):
            raise ValueError("seismic_structure shape does not match AVO")
        if self.band_fold.shape != (n_band, n_cdp):
            raise ValueError("band_fold must have shape [band, cdp]")
        if self.structure_fold.shape != (n_cdp,):
            raise ValueError("structure_fold must have shape [cdp]")
        if self.time_ms.shape != (n_time,) or self.cdps.shape != (n_cdp,):
            raise ValueError("coordinate axes do not match the data")
        if self.line_xy.shape != (n_cdp, 2):
            raise ValueError("line_xy must have shape [cdp, 2]")
        if len(self.band_names) != n_band or len(self.band_limits_degrees) != n_band:
            raise ValueError("band metadata does not match AVO")


def _require_segyio():
    try:
        import segyio
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Field SEG-Y processing requires segyio. Install the field extra "
            "with `python -m pip install -e \".[field]\"`."
        ) from error
    return segyio


def _coordinate_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values < 0, 1.0 / np.maximum(np.abs(values), 1.0), np.where(values > 0, values, 1.0))


def inspect_segy(path: str | Path, angle_header: str = "offset") -> SegySummary:
    """Inspect the headers used by the S01 workflow without reading trace samples."""
    segyio = _require_segyio()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SEG-Y file not found: {source}")
    if angle_header != "offset":
        raise ValueError("The current field reader supports angle_header='offset' only")

    with segyio.open(str(source), "r", ignore_geometry=True) as handle:
        cdps = np.asarray(handle.attributes(segyio.TraceField.CDP)[:], dtype=np.int64)
        angles = np.asarray(handle.attributes(segyio.TraceField.offset)[:], dtype=np.int64)
        samples = np.asarray(handle.samples, dtype=np.float64)
        values = np.unique(angles)
        return SegySummary(
            path=str(source),
            trace_count=int(handle.tracecount),
            sample_count=int(samples.size),
            sample_start_ms=float(samples[0]),
            sample_end_ms=float(samples[-1]),
            sample_interval_ms=float(np.median(np.diff(samples))),
            cdp_min=int(cdps.min()),
            cdp_max=int(cdps.max()),
            angle_header_min=int(angles.min()),
            angle_header_max=int(angles.max()),
            angle_header_values=tuple(int(value) for value in values),
        )


def stack_segy_line(
    path: str | Path,
    *,
    bands_degrees: Mapping[str, tuple[float, float] | list[float]],
    time_window_ms: tuple[float, float],
    midpoint_y_max: float | None,
    angle_header_semantics: str,
    angle_header: str = "offset",
    robust_clip_percentiles: tuple[float, float] = (1.0, 99.0),
    chunk_traces: int = 32_768,
) -> FieldLineStacks:
    """Build exact mean CDP/angle stacks from SEG-Y using bounded memory.

    The source S01 file stores integer incidence-angle bins in the SEG-Y offset
    word.  Because that convention is not standard SEG-Y semantics, callers
    must explicitly set ``angle_header_semantics='incidence_angle_degrees'``.
    The function otherwise refuses to label the header as an angle.
    """
    if angle_header_semantics != "incidence_angle_degrees":
        raise ValueError(
            "Refusing to interpret the SEG-Y offset word as angle. Set "
            "angle_header_semantics='incidence_angle_degrees' only after "
            "verifying the acquisition/export metadata."
        )
    if angle_header != "offset":
        raise ValueError("The current field reader supports angle_header='offset' only")
    if chunk_traces < 1:
        raise ValueError("chunk_traces must be positive")
    band_items = [(str(name), tuple(float(v) for v in limits)) for name, limits in bands_degrees.items()]
    if not band_items or any(len(limits) != 2 or limits[0] > limits[1] for _, limits in band_items):
        raise ValueError("bands_degrees must map names to ordered [minimum, maximum] limits")

    segyio = _require_segyio()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SEG-Y file not found: {source}")

    with segyio.open(str(source), "r", ignore_geometry=True) as handle:
        cdps = np.asarray(handle.attributes(segyio.TraceField.CDP)[:], dtype=np.int64)
        angles = np.asarray(handle.attributes(segyio.TraceField.offset)[:], dtype=np.float64)
        sx = np.asarray(handle.attributes(segyio.TraceField.SourceX)[:], dtype=np.float64)
        sy = np.asarray(handle.attributes(segyio.TraceField.SourceY)[:], dtype=np.float64)
        gx = np.asarray(handle.attributes(segyio.TraceField.GroupX)[:], dtype=np.float64)
        gy = np.asarray(handle.attributes(segyio.TraceField.GroupY)[:], dtype=np.float64)
        scalars = np.asarray(
            handle.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=np.float64
        )
        scale = _coordinate_scale(scalars)
        midpoint_x = 0.5 * (sx + gx) * scale
        midpoint_y = 0.5 * (sy + gy) * scale

        all_cdps, inverse = np.unique(cdps, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        cdp_x = np.bincount(inverse, weights=midpoint_x) / counts
        cdp_y = np.bincount(inverse, weights=midpoint_y) / counts
        cdp_mask = np.isfinite(cdp_x) & np.isfinite(cdp_y)
        if midpoint_y_max is not None:
            cdp_mask &= cdp_y < float(midpoint_y_max)
        selected_cdps = all_cdps[cdp_mask]
        line_xy = np.column_stack([cdp_x[cdp_mask], cdp_y[cdp_mask]])
        if selected_cdps.size == 0:
            raise ValueError("The line selection contains no CDPs")

        samples_ms = np.asarray(handle.samples, dtype=np.float64)
        t_min, t_max = map(float, time_window_ms)
        if t_min > t_max:
            raise ValueError("time_window_ms must be ordered")
        time_mask = (samples_ms >= t_min) & (samples_ms <= t_max)
        time_indices = np.flatnonzero(time_mask)
        if time_indices.size == 0:
            raise ValueError("The requested time window does not intersect the SEG-Y samples")
        if np.any(np.diff(time_indices) != 1):
            raise ValueError("The selected SEG-Y time samples are not contiguous")
        sample_slice = slice(int(time_indices[0]), int(time_indices[-1]) + 1)
        time_ms = samples_ms[sample_slice]

        cdp_index = np.searchsorted(selected_cdps, cdps)
        valid_cdp = cdp_index < selected_cdps.size
        valid_cdp &= selected_cdps[np.minimum(cdp_index, selected_cdps.size - 1)] == cdps
        angle_min = min(limits[0] for _, limits in band_items)
        angle_max = max(limits[1] for _, limits in band_items)
        selected_trace = valid_cdp & (angles >= angle_min) & (angles <= angle_max)
        trace_indices = np.flatnonzero(selected_trace)
        if trace_indices.size == 0:
            raise ValueError("No traces satisfy the line, time, and angle selections")

        n_band, n_time, n_cdp = len(band_items), time_ms.size, selected_cdps.size
        band_sums = np.zeros((n_band, n_time, n_cdp), dtype=np.float64)
        band_fold = np.zeros((n_band, n_cdp), dtype=np.int64)
        structure_sum = np.zeros((n_time, n_cdp), dtype=np.float64)
        structure_fold = np.zeros(n_cdp, dtype=np.int64)

        first, last = int(trace_indices[0]), int(trace_indices[-1]) + 1
        for start in range(first, last, chunk_traces):
            stop = min(start + chunk_traces, last)
            local_mask = selected_trace[start:stop]
            if not local_mask.any():
                continue
            traces = np.asarray(handle.trace.raw[start:stop], dtype=np.float32)
            traces = traces[local_mask, sample_slice]
            local_cdps = cdp_index[start:stop][local_mask]
            local_angles = angles[start:stop][local_mask]
            np.add.at(structure_sum.T, local_cdps, traces)
            np.add.at(structure_fold, local_cdps, 1)
            for band_index, (_, (minimum, maximum)) in enumerate(band_items):
                in_band = (local_angles >= minimum) & (local_angles <= maximum)
                if in_band.any():
                    np.add.at(band_sums[band_index].T, local_cdps[in_band], traces[in_band])
                    np.add.at(band_fold[band_index], local_cdps[in_band], 1)

    avo = np.divide(
        band_sums,
        band_fold[:, None, :],
        out=np.full_like(band_sums, np.nan),
        where=band_fold[:, None, :] > 0,
    ).astype(np.float32)
    structural = np.divide(
        structure_sum,
        structure_fold[None, :],
        out=np.full_like(structure_sum, np.nan),
        where=structure_fold[None, :] > 0,
    )
    finite = structural[np.isfinite(structural)]
    if finite.size == 0:
        raise ValueError("The structural stack contains no finite samples")
    low, high = np.percentile(finite, robust_clip_percentiles)
    structural = np.clip(structural, low, high)
    structural = (structural - np.nanmean(structural)) / max(np.nanstd(structural), 1e-8)

    result = FieldLineStacks(
        avo=avo,
        seismic_structure=structural.astype(np.float32),
        band_fold=band_fold.astype(np.int32),
        structure_fold=structure_fold.astype(np.int32),
        time_ms=time_ms.astype(np.float32),
        cdps=selected_cdps.astype(np.int32),
        line_xy=line_xy.astype(np.float64),
        band_names=tuple(name for name, _ in band_items),
        band_limits_degrees=tuple(limits for _, limits in band_items),
    )
    result.validate()
    return result


def save_field_line_stacks(path: str | Path, stacks: FieldLineStacks) -> None:
    """Save the compact, lossless output of the expensive SEG-Y stacking step."""
    stacks.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        avo=stacks.avo,
        seismic_structure=stacks.seismic_structure,
        band_fold=stacks.band_fold,
        structure_fold=stacks.structure_fold,
        time_ms=stacks.time_ms,
        cdps=stacks.cdps,
        line_xy=stacks.line_xy,
        band_names=np.asarray(stacks.band_names),
        band_limits_degrees=np.asarray(stacks.band_limits_degrees, dtype=np.float32),
    )


def load_field_line_stacks(path: str | Path) -> FieldLineStacks:
    """Load a cached field-line stack archive and validate its contract."""
    with np.load(Path(path), allow_pickle=False) as archive:
        result = FieldLineStacks(
            avo=archive["avo"],
            seismic_structure=archive["seismic_structure"],
            band_fold=archive["band_fold"],
            structure_fold=archive["structure_fold"],
            time_ms=archive["time_ms"],
            cdps=archive["cdps"],
            line_xy=archive["line_xy"],
            band_names=tuple(str(value) for value in archive["band_names"]),
            band_limits_degrees=tuple(
                tuple(float(v) for v in limits) for limits in archive["band_limits_degrees"]
            ),
        )
    result.validate()
    return result
