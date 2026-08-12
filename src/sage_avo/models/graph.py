"""Torch RGT graph construction used by TransformerConv message passing."""

from __future__ import annotations

import torch
from torch import Tensor


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

        vertical_source = rows[:-1] * width + columns[:-1]
        vertical_destination = (rows[:-1] + 1) * width + columns[:-1]
        src = torch.cat(
            (source.flatten(), destination.flatten(), vertical_source.flatten(), vertical_destination.flatten())
        )
        dst = torch.cat(
            (destination.flatten(), source.flatten(), vertical_destination.flatten(), vertical_source.flatten())
        )
        outputs.append(torch.stack((src, dst)))
    return outputs
