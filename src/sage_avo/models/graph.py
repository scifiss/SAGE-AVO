"""Torch RGT graph construction used by TransformerConv message passing."""

from __future__ import annotations

import torch
from torch import Tensor


RGT_V1_LEGACY = "rgt_v1_legacy"
RGT_V2_TIE_FIXED = "rgt_v2_tie_fixed"
RGT_T2B_ADAPTIVE = "rgt_t2b_adaptive"
RGT_T2C_INVERSE = "rgt_t2c_inverse"
RGT_V3_CONFIDENCE_BLOCKED = "rgt_v3_confidence_blocked"


def _near_tie_tolerance(source: Tensor, candidates: Tensor) -> Tensor:
    """Return a scale-aware float tolerance for lexicographic RGT matching.

    Sixteen ULPs at the local RGT magnitude cover CPU/CUDA reduction-order
    noise without treating geologically distinct mismatches as equivalent.
    The tensors are evaluated in their native dtype so the rule is portable.
    """
    if not source.dtype.is_floating_point:
        raise TypeError("RGT must use a floating-point dtype")
    scale = torch.maximum(
        torch.maximum(source.abs(), candidates.abs().amax(dim=-1)),
        torch.ones((), dtype=source.dtype, device=source.device),
    )
    return 16.0 * torch.finfo(source.dtype).eps * scale


def _lexicographic_shift_choice(
    differences: Tensor,
    valid: Tensor,
    shifts: Tensor,
    source_rows: Tensor,
    source_columns: Tensor,
    source_rgt: Tensor,
    candidate_rgt: Tensor,
) -> Tensor:
    """Choose mismatch, then |shift|, then a balanced deterministic sign.

    The last tie is resolved by source-node parity: even nodes prefer positive
    and odd nodes prefer negative displacement. This avoids a global sign bias
    while remaining exactly repeatable on CPU and CUDA.
    """
    masked = differences.masked_fill(~valid, torch.inf)
    minimum = masked.amin(dim=-1, keepdim=True)
    tolerance = _near_tie_tolerance(source_rgt, candidate_rgt).unsqueeze(-1)
    near = valid & (masked <= minimum + tolerance)
    shift_grid = shifts.view(*((1,) * (differences.ndim - 1)), -1)
    absolute_shift = shift_grid.abs().expand_as(differences)
    minimum_absolute = absolute_shift.masked_fill(~near, torch.iinfo(torch.int64).max).amin(
        dim=-1, keepdim=True
    )
    finalists = near & (absolute_shift == minimum_absolute)
    positive_preferred = ((source_rows + source_columns) % 2 == 0).unsqueeze(-1)
    sign_penalty = torch.where(
        shift_grid == 0,
        torch.zeros_like(shift_grid),
        ((shift_grid > 0) != positive_preferred).to(shift_grid.dtype),
    ).expand_as(differences)
    rank = sign_penalty.masked_fill(~finalists, torch.iinfo(torch.int64).max)
    return rank.argmin(dim=-1, keepdim=True)


def build_horizon_edges_tie_fixed(
    rgt: Tensor,
    max_shift: int = 3,
    *,
    adaptive_margin: int | None = None,
    normalized_mismatch_threshold: float | None = None,
    normalized_discontinuity_threshold: float | None = None,
    dip_residual_threshold: float | None = None,
    displacement_threshold: int | None = None,
) -> list[Tensor]:
    """Build tie-fixed tangential edges with optional adaptive search/blocking.

    ``adaptive_margin`` enables a local search radius based on the RGT-gradient
    tangent prediction, capped by ``max_shift``. Confidence thresholds remove
    uncertain links rather than passing a soft feature that attention may
    ignore. Reverse edges are added only after filtering.
    """
    if rgt.ndim != 3:
        raise ValueError("rgt must have shape [B, H, W]")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if adaptive_margin is not None and adaptive_margin < 0:
        raise ValueError("adaptive_margin must be non-negative")
    batch, height, width = rgt.shape
    rows, columns = torch.meshgrid(
        torch.arange(height, device=rgt.device),
        torch.arange(width - 1, device=rgt.device),
        indexing="ij",
    )
    shifts = torch.arange(-max_shift, max_shift + 1, device=rgt.device)
    target_rows = rows.unsqueeze(-1) + shifts
    valid = (target_rows >= 0) & (target_rows < height)
    target_rows = target_rows.clamp(0, height - 1)
    target_columns = (columns + 1).unsqueeze(-1).expand_as(target_rows)
    outputs: list[Tensor] = []
    for item in range(batch):
        source_rgt = rgt[item, rows, columns]
        candidate_rgt = rgt[item, target_rows, target_columns]
        differences = torch.abs(source_rgt.unsqueeze(-1) - candidate_rgt)
        item_valid = valid
        if adaptive_margin is not None:
            vertical_gradient = torch.gradient(rgt[item], dim=0)[0].abs()[:, :-1]
            lateral_gradient = rgt[item, :, 1:] - rgt[item, :, :-1]
            predicted = torch.abs(lateral_gradient) / vertical_gradient.clamp_min(
                16.0 * torch.finfo(rgt.dtype).eps
            )
            radius = torch.ceil(predicted + float(adaptive_margin)).to(torch.int64)
            radius = radius.clamp(min=min(3, max_shift), max=max_shift)
            item_valid = item_valid & (shifts.abs().view(1, 1, -1) <= radius.unsqueeze(-1))
        best = _lexicographic_shift_choice(
            differences,
            item_valid,
            shifts,
            rows,
            columns,
            source_rgt,
            candidate_rgt,
        )
        destination_rows = target_rows.gather(-1, best).squeeze(-1)
        selected_mismatch = differences.gather(-1, best).squeeze(-1)
        selected_shift = shifts[best.squeeze(-1)]
        keep = torch.ones_like(selected_mismatch, dtype=torch.bool)
        if normalized_mismatch_threshold is not None:
            vertical = torch.gradient(rgt[item], dim=0)[0].abs()
            positive_vertical = vertical[
                vertical > 16.0 * torch.finfo(rgt.dtype).eps
            ]
            reference_step = (
                positive_vertical.median()
                if positive_vertical.numel()
                else torch.ones((), dtype=rgt.dtype, device=rgt.device)
            )
            source_scale = vertical[rows, columns]
            target_scale = vertical[destination_rows, columns + 1]
            local_scale = (0.5 * (source_scale + target_scale)).clamp_min(reference_step)
            normalized_mismatch = selected_mismatch / local_scale
            block = normalized_mismatch > float(normalized_mismatch_threshold)
            if (
                normalized_discontinuity_threshold is not None
                and dip_residual_threshold is not None
            ):
                cartesian_difference = torch.abs(
                    rgt[item, rows, columns] - rgt[item, rows, columns + 1]
                )
                normalized_discontinuity = cartesian_difference / local_scale
                signed_lateral = rgt[item, rows, columns + 1] - rgt[item, rows, columns]
                predicted_shift = -signed_lateral / source_scale.clamp_min(reference_step)
                dip_residual = (selected_shift.to(rgt.dtype) - predicted_shift).abs()
                block |= (
                    normalized_discontinuity > float(normalized_discontinuity_threshold)
                ) & (dip_residual > float(dip_residual_threshold))
            keep &= ~block
        if displacement_threshold is not None:
            keep &= selected_shift.abs() <= int(displacement_threshold)
        source = (rows * width + columns)[keep]
        destination = (destination_rows * width + columns + 1)[keep]
        outputs.append(
            torch.stack(
                (
                    torch.cat((source.flatten(), destination.flatten())),
                    torch.cat((destination.flatten(), source.flatten())),
                )
            )
        )
    return outputs


def build_horizon_edges_inverse_rgt(rgt: Tensor) -> list[Tensor]:
    """Steer to the nearest discrete inverse-RGT target without gross clipping."""
    if rgt.ndim != 3:
        raise ValueError("rgt must have shape [B, H, W]")
    batch, height, width = rgt.shape
    rows, columns = torch.meshgrid(
        torch.arange(height, device=rgt.device),
        torch.arange(width - 1, device=rgt.device),
        indexing="ij",
    )
    outputs: list[Tensor] = []
    for item in range(batch):
        source_tau = rgt[item, :, :-1].transpose(0, 1).contiguous()
        target_columns = rgt[item, :, 1:].transpose(0, 1).contiguous()
        insertion = torch.searchsorted(target_columns, source_tau)
        lower = (insertion - 1).clamp(0, height - 1)
        upper = insertion.clamp(0, height - 1)
        lower_tau = target_columns.gather(1, lower)
        upper_tau = target_columns.gather(1, upper)
        lower_difference = (source_tau - lower_tau).abs()
        upper_difference = (source_tau - upper_tau).abs()
        tolerance = 16.0 * torch.finfo(rgt.dtype).eps * torch.maximum(
            torch.maximum(source_tau.abs(), torch.maximum(lower_tau.abs(), upper_tau.abs())),
            torch.ones((), dtype=rgt.dtype, device=rgt.device),
        )
        choose_upper = upper_difference + tolerance < lower_difference
        tied = (upper_difference - lower_difference).abs() <= tolerance
        lower_distance = (lower - rows.transpose(0, 1)).abs()
        upper_distance = (upper - rows.transpose(0, 1)).abs()
        choose_upper |= tied & (upper_distance < lower_distance)
        equal_distance = tied & (upper_distance == lower_distance) & (upper != lower)
        positive_preferred = (rows.transpose(0, 1) + columns.transpose(0, 1)) % 2 == 0
        choose_upper = torch.where(equal_distance, positive_preferred, choose_upper)
        destination_rows = torch.where(choose_upper, upper, lower).transpose(0, 1)
        in_range = (source_tau >= target_columns[:, :1]) & (
            source_tau <= target_columns[:, -1:]
        )
        keep = in_range.transpose(0, 1)
        source = (rows * width + columns)[keep]
        destination = (destination_rows * width + columns + 1)[keep]
        outputs.append(
            torch.stack(
                (
                    torch.cat((source.flatten(), destination.flatten())),
                    torch.cat((destination.flatten(), source.flatten())),
                )
            )
        )
    return outputs


def _deterministically_shuffle_lateral_edges(edges: Tensor) -> Tensor:
    """Permute neighbor identity while preserving source/target degree multisets."""
    half = edges.shape[1] // 2
    forward = edges[:, :half]
    if half <= 1:
        return edges
    permutation = torch.roll(torch.arange(half, device=edges.device), shifts=half // 3 + 1)
    shuffled = torch.stack((forward[0], forward[1, permutation]))
    return torch.cat((shuffled, shuffled.flip(0)), dim=1)


def build_horizon_edges(
    rgt: Tensor,
    max_shift: int = 3,
    steered: bool = True,
) -> list[Tensor]:
    """Return bidirectional adjacent-trace edges tangent to RGT horizons."""
    if rgt.ndim != 3:
        raise ValueError("rgt must have shape [B, H, W]")
    batch, height, width = rgt.shape
    rows, columns = torch.meshgrid(
        torch.arange(height, device=rgt.device),
        torch.arange(width, device=rgt.device),
        indexing="ij",
    )
    outputs: list[Tensor] = []
    for item in range(batch):
        valid_rows = rows[:, :-1]
        valid_columns = columns[:, :-1]
        source = valid_rows * width + valid_columns
        if steered:
            shifts = torch.arange(-max_shift, max_shift + 1, device=rgt.device)
            target_rows = valid_rows.unsqueeze(-1) + shifts
            valid = (target_rows >= 0) & (target_rows < height)
            target_rows = target_rows.clamp(0, height - 1)
            target_columns = (valid_columns + 1).unsqueeze(-1).expand_as(target_rows)
            differences = torch.abs(
                rgt[item, valid_rows, valid_columns].unsqueeze(-1)
                - rgt[item, target_rows, target_columns]
            ).masked_fill(~valid, torch.inf)
            best = differences.argmin(dim=-1, keepdim=True)
            destination_rows = target_rows.gather(-1, best).squeeze(-1)
        else:
            destination_rows = valid_rows
        destination = destination_rows * width + valid_columns + 1
        outputs.append(
            torch.stack(
                (
                    torch.cat((source.flatten(), destination.flatten())),
                    torch.cat((destination.flatten(), source.flatten())),
                )
            )
        )
    return outputs


def build_horizon_candidate_edges(
    rgt: Tensor,
    max_shift: int = 3,
    candidates: int = 2,
    steered: bool = True,
) -> list[Tensor]:
    """Return multiple candidate edges per adjacent-trace tangential relation.

    RGT mode retains the ``candidates`` smallest-mismatch destinations instead
    of committing to one hard argmin. Cartesian mode retains the nearest row
    offsets while preserving the same candidate count for controlled routing
    diagnostics.
    """
    if rgt.ndim != 3:
        raise ValueError("rgt must have shape [B, H, W]")
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    batch, height, width = rgt.shape
    if candidates < 1 or candidates > min(max_shift + 1, height):
        raise ValueError(
            "candidates must lie between 1 and the smallest boundary candidate count"
        )
    rows, columns = torch.meshgrid(
        torch.arange(height, device=rgt.device),
        torch.arange(width - 1, device=rgt.device),
        indexing="ij",
    )
    shifts = torch.arange(-max_shift, max_shift + 1, device=rgt.device)
    target_rows = rows.unsqueeze(-1) + shifts
    valid = (target_rows >= 0) & (target_rows < height)
    target_rows = target_rows.clamp(0, height - 1)
    target_columns = (columns + 1).unsqueeze(-1).expand_as(target_rows)
    outputs: list[Tensor] = []
    for item in range(batch):
        if steered:
            scores = torch.abs(
                rgt[item, rows, columns].unsqueeze(-1)
                - rgt[item, target_rows, target_columns]
            )
            scores = scores + 1e-6 * shifts.abs().to(rgt.dtype)
        else:
            scores = shifts.abs().to(rgt.dtype).view(1, 1, -1).expand_as(target_rows)
        scores = scores.masked_fill(~valid, torch.inf)
        best = torch.topk(scores, k=candidates, dim=-1, largest=False).indices
        destination_rows = target_rows.gather(-1, best)
        destination_columns = target_columns.gather(-1, best)
        source = (rows * width + columns).unsqueeze(-1).expand_as(destination_rows)
        destination = destination_rows * width + destination_columns
        outputs.append(
            torch.stack(
                (
                    torch.cat((source.flatten(), destination.flatten())),
                    torch.cat((destination.flatten(), source.flatten())),
                )
            )
        )
    return outputs


def build_normal_edges(
    rgt: Tensor,
    max_lateral_shift: int = 1,
    steered: bool = True,
) -> list[Tensor]:
    """Return bidirectional edges approximating the local RGT-gradient direction.

    The steered destination maximizes absolute RGT change per sample distance
    among candidates one row deeper and within ``max_lateral_shift`` columns.
    This is a discrete normal-to-horizon relation, distinct from equal-RGT
    tangential steering. Cartesian mode returns ordinary vertical edges.
    """
    if rgt.ndim != 3:
        raise ValueError("rgt must have shape [B, H, W]")
    if max_lateral_shift < 0:
        raise ValueError("max_lateral_shift must be non-negative")
    batch, height, width = rgt.shape
    rows, columns = torch.meshgrid(
        torch.arange(height - 1, device=rgt.device),
        torch.arange(width, device=rgt.device),
        indexing="ij",
    )
    outputs: list[Tensor] = []
    for item in range(batch):
        source = rows * width + columns
        if steered and max_lateral_shift:
            shifts = torch.arange(
                -max_lateral_shift,
                max_lateral_shift + 1,
                device=rgt.device,
            )
            target_columns = columns.unsqueeze(-1) + shifts
            valid = (target_columns >= 0) & (target_columns < width)
            target_columns = target_columns.clamp(0, width - 1)
            target_rows = (rows + 1).unsqueeze(-1).expand_as(target_columns)
            differences = torch.abs(
                rgt[item, rows, columns].unsqueeze(-1)
                - rgt[item, target_rows, target_columns]
            )
            distances = torch.sqrt(1.0 + shifts.to(rgt.dtype).square())
            scores = (differences / distances).masked_fill(~valid, -torch.inf)
            best = scores.argmax(dim=-1, keepdim=True)
            destination_columns = target_columns.gather(-1, best).squeeze(-1)
        else:
            destination_columns = columns
        destination = (rows + 1) * width + destination_columns
        outputs.append(
            torch.stack(
                (
                    torch.cat((source.flatten(), destination.flatten())),
                    torch.cat((destination.flatten(), source.flatten())),
                )
            )
        )
    return outputs


def build_normal_candidate_edges(
    rgt: Tensor,
    max_lateral_shift: int = 1,
    candidates: int = 2,
    steered: bool = True,
) -> list[Tensor]:
    """Return multiple candidates for the discrete normal-to-RGT relation."""
    if rgt.ndim != 3:
        raise ValueError("rgt must have shape [B, H, W]")
    if max_lateral_shift < 0:
        raise ValueError("max_lateral_shift must be non-negative")
    batch, height, width = rgt.shape
    if candidates < 1 or candidates > min(max_lateral_shift + 1, width):
        raise ValueError(
            "candidates must lie between 1 and the smallest boundary candidate count"
        )
    rows, columns = torch.meshgrid(
        torch.arange(height - 1, device=rgt.device),
        torch.arange(width, device=rgt.device),
        indexing="ij",
    )
    shifts = torch.arange(
        -max_lateral_shift,
        max_lateral_shift + 1,
        device=rgt.device,
    )
    target_columns = columns.unsqueeze(-1) + shifts
    valid = (target_columns >= 0) & (target_columns < width)
    target_columns = target_columns.clamp(0, width - 1)
    target_rows = (rows + 1).unsqueeze(-1).expand_as(target_columns)
    distances = torch.sqrt(1.0 + shifts.to(rgt.dtype).square())
    outputs: list[Tensor] = []
    for item in range(batch):
        if steered:
            scores = torch.abs(
                rgt[item, rows, columns].unsqueeze(-1)
                - rgt[item, target_rows, target_columns]
            ) / distances
            scores = scores - 1e-6 * shifts.abs().to(rgt.dtype)
        else:
            scores = -shifts.abs().to(rgt.dtype).view(1, 1, -1).expand_as(target_columns)
        scores = scores.masked_fill(~valid, -torch.inf)
        best = torch.topk(scores, k=candidates, dim=-1, largest=True).indices
        destination_rows = target_rows.gather(-1, best)
        destination_columns = target_columns.gather(-1, best)
        source = (rows * width + columns).unsqueeze(-1).expand_as(destination_columns)
        destination = destination_rows * width + destination_columns
        outputs.append(
            torch.stack(
                (
                    torch.cat((source.flatten(), destination.flatten())),
                    torch.cat((destination.flatten(), source.flatten())),
                )
            )
        )
    return outputs


def build_rgt_edges(
    rgt: Tensor,
    max_shift: int = 3,
    steered: bool = True,
) -> list[Tensor]:
    """Return bidirectional horizontal and vertical edges.

    When ``steered`` is true, adjacent-trace targets minimize RGT difference
    within ``max_shift`` samples. When false, horizontal connections retain the
    same time sample, providing the controlled Cartesian/no-RGT ablation.
    """
    horizon = build_horizon_edges(rgt, max_shift=max_shift, steered=steered)
    normal = build_normal_edges(rgt, max_lateral_shift=0, steered=False)
    return [torch.cat((along, across), dim=1) for along, across in zip(horizon, normal)]


def build_experimental_rgt_edges(
    rgt: Tensor,
    *,
    topology: str,
    max_shift: int = 3,
    adaptive_margin: int = 1,
    confidence_normalized_mismatch_threshold: float | None = None,
    confidence_normalized_discontinuity_threshold: float | None = None,
    confidence_dip_residual_threshold: float | None = None,
    confidence_displacement_threshold: int | None = None,
) -> list[Tensor]:
    """Build an opt-in topology while preserving the V1 production behavior."""
    horizon, normal = build_experimental_rgt_relations(
        rgt,
        topology=topology,
        max_shift=max_shift,
        adaptive_margin=adaptive_margin,
        confidence_normalized_mismatch_threshold=confidence_normalized_mismatch_threshold,
        confidence_normalized_discontinuity_threshold=(
            confidence_normalized_discontinuity_threshold
        ),
        confidence_dip_residual_threshold=confidence_dip_residual_threshold,
        confidence_displacement_threshold=confidence_displacement_threshold,
    )
    return [torch.cat((along, across), dim=1) for along, across in zip(horizon, normal)]


def build_experimental_rgt_relations(
    rgt: Tensor,
    *,
    topology: str,
    max_shift: int = 3,
    adaptive_margin: int = 1,
    confidence_normalized_mismatch_threshold: float | None = None,
    confidence_normalized_discontinuity_threshold: float | None = None,
    confidence_dip_residual_threshold: float | None = None,
    confidence_displacement_threshold: int | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """Return tangential and normal relations separately for diagnostics."""
    if topology == RGT_V1_LEGACY:
        horizon = build_horizon_edges(rgt, max_shift=max_shift, steered=True)
    elif topology == "cartesian":
        horizon = build_horizon_edges(rgt, max_shift=max_shift, steered=False)
    elif topology in {RGT_V2_TIE_FIXED, RGT_T2B_ADAPTIVE, RGT_V3_CONFIDENCE_BLOCKED}:
        horizon = build_horizon_edges_tie_fixed(
            rgt,
            max_shift=max_shift,
            adaptive_margin=adaptive_margin if topology == RGT_T2B_ADAPTIVE else None,
            normalized_mismatch_threshold=(
                confidence_normalized_mismatch_threshold
                if topology == RGT_V3_CONFIDENCE_BLOCKED
                else None
            ),
            normalized_discontinuity_threshold=(
                confidence_normalized_discontinuity_threshold
                if topology == RGT_V3_CONFIDENCE_BLOCKED
                else None
            ),
            dip_residual_threshold=(
                confidence_dip_residual_threshold
                if topology == RGT_V3_CONFIDENCE_BLOCKED
                else None
            ),
            displacement_threshold=(
                confidence_displacement_threshold
                if topology == RGT_V3_CONFIDENCE_BLOCKED
                else None
            ),
        )
    elif topology == RGT_T2C_INVERSE:
        horizon = build_horizon_edges_inverse_rgt(rgt)
    elif topology == "shuffled":
        horizon = [
            _deterministically_shuffle_lateral_edges(edges)
            for edges in build_horizon_edges_tie_fixed(rgt, max_shift=max_shift)
        ]
    else:
        raise ValueError(f"unsupported experimental RGT topology {topology!r}")
    normal = build_normal_edges(rgt, max_lateral_shift=0, steered=False)
    return horizon, normal
