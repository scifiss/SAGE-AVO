"""Regression tests for the opt-in RGT topology repair experiment."""

from __future__ import annotations

import pytest
import torch

from sage_avo.models.graph import (
    RGT_T2C_INVERSE,
    RGT_V1_LEGACY,
    RGT_V2_TIE_FIXED,
    build_experimental_rgt_edges,
    build_horizon_edges,
    build_horizon_edges_inverse_rgt,
    build_horizon_edges_tie_fixed,
)
from sage_avo.models.sage_avo import SAGEAVO
from sage_avo.experiments.training import FixedEpochIndexSampler


def _forward_shifts(edges: torch.Tensor, height: int, width: int) -> torch.Tensor:
    forward = edges[:, : height * (width - 1)]
    return forward[1].div(width, rounding_mode="floor") - forward[0].div(
        width, rounding_mode="floor"
    )


def test_flat_rgt_legacy_is_frozen_and_tie_fixed_selects_zero() -> None:
    rgt = torch.ones((1, 8, 2), dtype=torch.float32)
    legacy = build_horizon_edges(rgt, max_shift=3)[0]
    frozen = build_experimental_rgt_edges(
        rgt, topology=RGT_V1_LEGACY, max_shift=3
    )[0]
    assert torch.equal(legacy, frozen[:, : legacy.shape[1]])
    repaired = build_horizon_edges_tie_fixed(rgt, max_shift=3)[0]
    assert torch.equal(_forward_shifts(repaired, 8, 2), torch.zeros(8, dtype=torch.long))


def test_symmetric_dip_and_exact_ties_have_no_global_sign_bias() -> None:
    base = torch.arange(9, dtype=torch.float32)[:, None]
    positive = torch.cat((base, base - 1), dim=1)
    negative = torch.cat((base, base + 1), dim=1)
    assert torch.equal(
        _forward_shifts(build_horizon_edges_tie_fixed(positive[None], 2)[0], 9, 2),
        torch.ones(9, dtype=torch.long).clamp_max(8 - torch.arange(9)),
    )
    assert torch.equal(
        _forward_shifts(build_horizon_edges_tie_fixed(negative[None], 2)[0], 9, 2),
        torch.tensor([0, -1, -1, -1, -1, -1, -1, -1, -1]),
    )
    # Equal +/-1 candidates alternate sign by source parity rather than always
    # choosing the negative shift.
    tied = torch.tensor(
        [[[0.0, 3.0], [1.0, 2.0], [2.0, 3.0], [3.0, 2.0], [4.0, 3.0]]]
    )
    shifts = _forward_shifts(build_horizon_edges_tie_fixed(tied, 1)[0], 5, 2)
    assert int((shifts > 0).sum()) > 0
    assert int((shifts < 0).sum()) > 0


def test_near_tie_prefers_minimum_absolute_shift() -> None:
    rgt = torch.tensor([[[0.0, 0.0], [1.0, 1.0 + 2e-7], [2.0, 2.0]]])
    shifts = _forward_shifts(build_horizon_edges_tie_fixed(rgt, 1)[0], 3, 2)
    assert shifts.tolist() == [0, 0, 0]


def test_top_and_bottom_boundaries_remain_valid() -> None:
    rgt = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).repeat(1, 1, 2)
    for builder in (build_horizon_edges_tie_fixed, build_horizon_edges_inverse_rgt):
        edges = builder(rgt)[0]
        assert edges.min().item() >= 0
        assert edges.max().item() < 12


def test_inverse_rgt_does_not_clip_out_of_range_tau() -> None:
    left = torch.arange(5, dtype=torch.float32)
    right = torch.arange(5, dtype=torch.float32) + 1
    edges = build_horizon_edges_inverse_rgt(torch.stack((left, right), dim=1)[None])[0]
    # tau=0 lies below the target column range and is omitted in both directions.
    assert edges.shape[1] == 2 * 4


def test_confidence_block_removes_only_lateral_edges() -> None:
    rgt = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1).repeat(1, 1, 2)
    edges = build_experimental_rgt_edges(
        rgt,
        topology="rgt_v3_confidence_blocked",
        max_shift=1,
        confidence_normalized_mismatch_threshold=-1.0,
    )[0]
    assert edges.shape[1] == 2 * (6 - 1) * 2  # vertical edges on two traces


def test_experimental_topologies_and_root_only_are_parameter_matched() -> None:
    legacy = SAGEAVO(hidden_channels=16, graph_heads=4, rgt_topology=RGT_V1_LEGACY)
    corrected = SAGEAVO(hidden_channels=16, graph_heads=4, rgt_topology=RGT_V2_TIE_FIXED)
    root_only = SAGEAVO(
        hidden_channels=16,
        graph_heads=4,
        rgt_topology=RGT_V2_TIE_FIXED,
        graph_neighbor_scale=0.0,
    )
    assert list(legacy.state_dict()) == list(corrected.state_dict()) == list(root_only.state_dict())
    assert sum(p.numel() for p in legacy.parameters()) == sum(
        p.numel() for p in root_only.parameters()
    )


def test_neighbor_root_instrumentation_and_reinjection_control() -> None:
    torch.manual_seed(123)
    model = SAGEAVO(
        hidden_channels=16,
        graph_heads=4,
        rgt_topology=RGT_V2_TIE_FIXED,
    ).eval()
    state = torch.randn(1, 3, 8, 7)
    avo = torch.randn(1, 3, 8, 7)
    low = torch.randn(1, 3, 8, 7)
    rgt = torch.arange(8, dtype=torch.float32)[None, :, None].expand(1, 8, 7)
    model.graph.diagnostic_capture_contributions = True
    with torch.inference_mode():
        intact = model(state, torch.tensor([0.5]), avo, low, rgt)
        diagnostics = model.graph.last_contribution_diagnostics
        model.graph.diagnostic_neighbor_scale = 0.0
        root_only = model(state, torch.tensor([0.5]), avo, low, rgt)
        model.graph.diagnostic_neighbor_scale = 1.0
        model.diagnostic_graph_reinjection_scale = 0.0
        no_reinjection = model(state, torch.tensor([0.5]), avo, low, rgt)
    assert len(diagnostics) == 2
    assert all(row["root_rms"] > 0 and row["neighbor_rms"] > 0 for row in diagnostics)
    assert not torch.equal(intact.velocity, root_only.velocity)
    assert not torch.equal(intact.velocity, no_reinjection.velocity)


def test_fixed_epoch_sampler_changes_only_when_epoch_is_advanced() -> None:
    sampler = FixedEpochIndexSampler(((3, 1, 4), (2, 0, 5)))
    assert list(sampler) == [3, 1, 4]
    assert list(sampler) == [3, 1, 4]
    sampler.set_epoch(1)
    assert list(sampler) == [2, 0, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("topology", [RGT_V2_TIE_FIXED, RGT_T2C_INVERSE])
def test_cpu_cuda_topology_repeatability(topology: str) -> None:
    generator = torch.Generator().manual_seed(12345)
    increments = torch.rand((1, 15, 8), generator=generator) + 0.05
    rgt = increments.cumsum(dim=1)
    cpu = build_experimental_rgt_edges(rgt, topology=topology, max_shift=4)[0]
    cuda = build_experimental_rgt_edges(rgt.cuda(), topology=topology, max_shift=4)[0].cpu()
    assert torch.equal(cpu, cuda)
