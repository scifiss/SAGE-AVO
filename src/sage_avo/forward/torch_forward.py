"""Differentiable Torch counterpart of the exact-Zoeppritz workflow."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .specification import ForwardModelSpecification


PRODUCTION_ANGLE_BANDS = ((3.0, 17.0), (17.0, 31.0), (31.0, 45.0))
# Backward-compatible name for earlier callers. Adjacent production bands use
# the shared-endpoint convention declared in the forward specification.
CURRENT_ANGLE_BANDS = PRODUCTION_ANGLE_BANDS
LEGACY_005_ANGLE_BANDS = PRODUCTION_ANGLE_BANDS


def exact_zoeppritz_pp(vp: Tensor, vs: Tensor, density: Tensor, angles_degrees: Tensor) -> Tensor:
    """Return exact real-component P-P reflectivity as ``[B, angle, H, W]``.

    Complex vertical slownesses are retained above critical incidence so this
    differentiable operator follows the same convention as the NumPy Stage-02
    solver. Clamping the square-root arguments to a real pre-critical branch
    produces incorrect high-angle amplitudes at strong velocity contrasts.
    """
    if vp.ndim == 4:
        vp, vs, density = vp[:, 0], vs[:, 0], density[:, 0]
    if vp.ndim != 3 or vp.shape != vs.shape or vp.shape != density.shape:
        raise ValueError("vp, vs, density must have matching [B, H, W] shapes")
    vp1, vp2 = vp[:, :-1], vp[:, 1:]
    vs1, vs2 = vs[:, :-1], vs[:, 1:]
    rho1, rho2 = density[:, :-1], density[:, 1:]
    outputs = []
    complex_dtype = torch.complex128 if vp.dtype == torch.float64 else torch.complex64
    for angle in angles_degrees:
        theta = torch.deg2rad(angle.to(dtype=vp.dtype, device=vp.device))
        p = torch.sin(theta) / vp1
        p2 = p.square()
        j1 = torch.sqrt((1.0 / vp1.square() - p2).to(complex_dtype))
        j2 = torch.sqrt((1.0 / vp2.square() - p2).to(complex_dtype))
        k1 = torch.sqrt((1.0 / vs1.square() - p2).to(complex_dtype))
        k2 = torch.sqrt((1.0 / vs2.square() - p2).to(complex_dtype))
        a = rho2 * (1.0 - 2.0 * vs2.square() * p2) - rho1 * (1.0 - 2.0 * vs1.square() * p2)
        b = rho2 * (1.0 - 2.0 * vs2.square() * p2) + 2.0 * rho1 * vs1.square() * p2
        c = rho1 * (1.0 - 2.0 * vs1.square() * p2) + 2.0 * rho2 * vs2.square() * p2
        d = 2.0 * (rho2 * vs2.square() - rho1 * vs1.square())
        e_term = b * j1 + c * j2
        f_term = b * k1 + c * k2
        g_term = a - d * j1 * k2
        h_term = a - d * j2 * k1
        denominator = e_term * f_term + g_term * h_term * p2
        numerator = (b * j1 - c * j2) * f_term - (a + d * j1 * k2) * h_term * p2
        outputs.append((numerator / (denominator + 1e-12)).real)
    reflectivity = torch.stack(outputs, dim=1)
    return F.pad(reflectivity, (0, 0, 1, 0))


def _ricker(
    frequency_hz: float,
    dt_seconds: float,
    samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    time = (torch.arange(samples, device=device, dtype=dtype) - samples // 2) * dt_seconds
    argument = (torch.pi * frequency_hz * time).square()
    wavelet = (1.0 - 2.0 * argument) * torch.exp(-argument)
    return (wavelet / (wavelet.abs().sum() + 1e-12)).view(1, 1, samples, 1)


def forward_avo_three_band_torch(
    vp: Tensor,
    vs: Tensor,
    density: Tensor,
    angles_degrees: Tensor | None = None,
    wavelet_hz: float = 14.0,
    dt_seconds: float = 0.004,
    wavelet_samples: int = 81,
    bands_degrees: tuple[tuple[float, float], ...] = CURRENT_ANGLE_BANDS,
    apply_mute: bool = True,
    mute_start: tuple[float, float] = (30.0, 0.0),
    mute_end: tuple[float, float] = (45.0, 0.1),
    taper_samples: int = 5,
    sample_origin: int | Tensor = 0,
    mute_time_origin_seconds: float = 0.0,
) -> Tensor:
    """Generate differentiable exact-PP near/mid/far stacks.

    Adjacent production bands intentionally share the 17° and 31° samples.
    The physics loss must use the same convention as the observations it
    reproduces.
    """
    if angles_degrees is None:
        angles_degrees = torch.arange(3.0, 46.0, device=vp.device, dtype=vp.dtype)
    else:
        angles_degrees = angles_degrees.to(device=vp.device, dtype=vp.dtype)
    if wavelet_samples < 1 or wavelet_samples % 2 == 0:
        raise ValueError("wavelet_samples must be a positive odd integer")
    if len(bands_degrees) != 3:
        raise ValueError("Exactly three angle bands are required")
    reflectivity = exact_zoeppritz_pp(vp, vs, density, angles_degrees)
    batch, angles, height, width = reflectivity.shape
    traces = reflectivity.reshape(batch * angles, 1, height, width)
    seismic = F.conv2d(
        traces,
        _ricker(wavelet_hz, dt_seconds, wavelet_samples, vp.device, vp.dtype),
        padding=(wavelet_samples // 2, 0),
    ).reshape(batch, angles, height, width)

    if apply_mute:
        start_angle, start_time = mute_start
        end_angle, end_time = mute_end
        slope = 0.0 if end_angle == start_angle else (end_time - start_time) / (end_angle - start_angle)
        mute_times = torch.where(
            angles_degrees <= start_angle,
            torch.as_tensor(start_time, device=vp.device, dtype=vp.dtype),
            torch.where(
                angles_degrees >= end_angle,
                torch.as_tensor(end_time, device=vp.device, dtype=vp.dtype),
                start_time + slope * (angles_degrees - start_angle),
            ),
        )
        sample_axis = torch.arange(height, device=vp.device, dtype=vp.dtype)
        origins = torch.as_tensor(sample_origin, device=vp.device, dtype=vp.dtype)
        if origins.ndim == 0:
            origins = origins.expand(batch)
        if origins.shape != (batch,):
            raise ValueError("sample_origin must be scalar or have shape [batch]")
        mute_samples = torch.floor(
            (mute_times - float(mute_time_origin_seconds)) / dt_seconds + 1e-6
        )
        denominator = max(taper_samples - 1, 1)
        taper = torch.clamp(
            (
                sample_axis.view(1, 1, height, 1)
                + origins.view(batch, 1, 1, 1)
                - mute_samples.view(1, -1, 1, 1)
            )
            / denominator,
            0.0,
            1.0,
        )
        if taper_samples == 0:
            taper = (
                sample_axis.view(1, 1, height, 1)
                + origins.view(batch, 1, 1, 1)
                >= mute_samples.view(1, -1, 1, 1)
            ).to(vp.dtype)
        seismic = seismic * taper

    outputs = []
    for minimum, maximum in bands_degrees:
        selection = (angles_degrees >= minimum) & (angles_degrees <= maximum)
        if not bool(selection.any()):
            raise ValueError(f"No configured angles fall in band [{minimum}, {maximum}]")
        outputs.append(seismic[:, selection].mean(dim=1))
    return torch.stack(outputs, dim=1)


def forward_avo_three_band_spec_torch(
    vp: Tensor,
    vs: Tensor,
    density: Tensor,
    specification: ForwardModelSpecification,
    *,
    sample_origin: int | Tensor = 0,
) -> Tensor:
    """Differentiable exact-PP operator using the shared v003 contract."""
    angles_degrees = torch.as_tensor(
        specification.angles_degrees,
        device=vp.device,
        dtype=vp.dtype,
    )
    reflectivity = exact_zoeppritz_pp(vp, vs, density, angles_degrees)
    batch, angle_count, height, width = reflectivity.shape
    convolved = []
    for angle_index, angle in enumerate(specification.angles_degrees):
        samples = specification.wavelet_for_angle(angle).samples_array(
            specification.dt_seconds
        )
        kernel = torch.as_tensor(samples[::-1].copy(), device=vp.device, dtype=vp.dtype)
        kernel = kernel.view(1, 1, -1, 1)
        convolved.append(
            F.conv2d(
                reflectivity[:, angle_index : angle_index + 1],
                kernel,
                padding=(kernel.shape[2] // 2, 0),
            )[:, 0]
        )
    seismic = torch.stack(convolved, dim=1)
    if specification.apply_mute:
        start_angle, start_time = specification.mute_start
        end_angle, end_time = specification.mute_end
        slope = 0.0 if end_angle == start_angle else (end_time - start_time) / (
            end_angle - start_angle
        )
        mute_times = torch.where(
            angles_degrees <= start_angle,
            torch.as_tensor(start_time, device=vp.device, dtype=vp.dtype),
            torch.where(
                angles_degrees >= end_angle,
                torch.as_tensor(end_time, device=vp.device, dtype=vp.dtype),
                start_time + slope * (angles_degrees - start_angle),
            ),
        )
        origins = torch.as_tensor(sample_origin, device=vp.device, dtype=vp.dtype)
        if origins.ndim == 0:
            origins = origins.expand(batch)
        if origins.shape != (batch,):
            raise ValueError("sample_origin must be scalar or have shape [batch]")
        local = torch.arange(height, device=vp.device, dtype=vp.dtype)
        global_samples = local.view(1, 1, height, 1) + origins.view(batch, 1, 1, 1)
        mute_samples = torch.floor(
            (
                mute_times - specification.mute_time_origin_seconds
            )
            / specification.dt_seconds
            + 1e-6
        ).view(1, angle_count, 1, 1)
        if specification.taper_samples == 0:
            taper = (global_samples >= mute_samples).to(vp.dtype)
        else:
            taper = torch.clamp(
                (global_samples - mute_samples) / max(specification.taper_samples - 1, 1),
                0.0,
                1.0,
            )
        seismic = seismic * taper
    stacks = []
    for band in specification.bands:
        selected = (angles_degrees >= band.minimum_degrees) & (
            angles_degrees <= band.maximum_degrees
        )
        if not bool(selected.any()):
            raise ValueError(f"No angles fall in band {band.name!r}")
        stacks.append(seismic[:, selected].mean(dim=1))
    return torch.stack(stacks, dim=1)
