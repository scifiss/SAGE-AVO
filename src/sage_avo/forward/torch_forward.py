"""Differentiable Torch counterpart of the public exact-Zoeppritz workflow."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def exact_zoeppritz_pp(vp: Tensor, vs: Tensor, density: Tensor, angles_degrees: Tensor) -> Tensor:
    """Return pre-critical exact P-P reflectivity as ``[B, angle, H, W]``."""
    if vp.ndim == 4:
        vp, vs, density = vp[:, 0], vs[:, 0], density[:, 0]
    if vp.ndim != 3 or vp.shape != vs.shape or vp.shape != density.shape:
        raise ValueError("vp, vs, density must have matching [B, H, W] shapes")
    vp1, vp2 = vp[:, :-1], vp[:, 1:]
    vs1, vs2 = vs[:, :-1], vs[:, 1:]
    rho1, rho2 = density[:, :-1], density[:, 1:]
    outputs = []
    for angle in angles_degrees:
        theta = torch.deg2rad(angle.to(dtype=vp.dtype, device=vp.device))
        p = torch.sin(theta) / vp1
        p2 = p.square()
        j1 = torch.sqrt(torch.clamp(1.0 / vp1.square() - p2, min=1e-12))
        j2 = torch.sqrt(torch.clamp(1.0 / vp2.square() - p2, min=1e-12))
        k1 = torch.sqrt(torch.clamp(1.0 / vs1.square() - p2, min=1e-12))
        k2 = torch.sqrt(torch.clamp(1.0 / vs2.square() - p2, min=1e-12))
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
        outputs.append(numerator / (denominator + 1e-12))
    reflectivity = torch.stack(outputs, dim=1)
    return F.pad(reflectivity, (0, 0, 1, 0))


def _ricker(frequency_hz: float, dt_seconds: float, samples: int, device: torch.device, dtype: torch.dtype) -> Tensor:
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
) -> Tensor:
    """Generate differentiable, non-overlapping near/mid/far stacks."""
    if angles_degrees is None:
        angles_degrees = torch.arange(3.0, 46.0, device=vp.device, dtype=vp.dtype)
    reflectivity = exact_zoeppritz_pp(vp, vs, density, angles_degrees)
    batch, angles, height, width = reflectivity.shape
    traces = reflectivity.reshape(batch * angles, 1, height, width)
    seismic = F.conv2d(
        traces,
        _ricker( wavelet_hz, dt_seconds, 81, vp.device, vp.dtype),
        padding=(40, 0),
    ).reshape(batch, angles, height, width)

    mute_times = torch.clamp((angles_degrees - 30.0) / 15.0, 0.0, 1.0) * 0.1
    sample_axis = torch.arange(height, device=vp.device, dtype=vp.dtype)
    mute_samples = torch.floor(mute_times / dt_seconds + 1e-6)
    taper = torch.clamp(
        (sample_axis.view(1, 1, height, 1) - mute_samples.view(1, -1, 1, 1)) / 4.0,
        0.0,
        1.0,
    )
    seismic = seismic * taper
    selections = (
        (angles_degrees >= 3) & (angles_degrees <= 17),
        (angles_degrees >= 18) & (angles_degrees <= 31),
        (angles_degrees >= 32) & (angles_degrees <= 45),
    )
    return torch.stack([seismic[:, selection].mean(dim=1) for selection in selections], dim=1)
