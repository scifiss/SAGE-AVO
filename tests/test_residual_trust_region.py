"""Tests for the training-derived smooth elastic residual trust region."""

from __future__ import annotations

import torch

from sage_avo.models.sage_avo import SAGEAVO


def test_residual_trust_region_is_smooth_and_strictly_bounded() -> None:
    scales = (3.5070113, 3.5530620, 2.7374384)
    model = SAGEAVO(hidden_channels=8, graph_heads=1, residual_trust_region_scales=scales)
    raw = torch.tensor(
        [[[[1e6]], [[-1e6]], [[1e6]]]], dtype=torch.float64, requires_grad=True
    )
    bounded = model._parameterize_velocity(raw)
    scale = model.residual_trust_region_scales.to(raw.dtype)
    assert torch.all(bounded.abs() <= scale * (1.0 + 1e-12))
    assert torch.isfinite(torch.autograd.grad(bounded.sum(), raw)[0]).all()


def test_residual_trust_region_is_locally_identity_like() -> None:
    scales = (3.5070113, 3.5530620, 2.7374384)
    model = SAGEAVO(hidden_channels=8, graph_heads=1, residual_trust_region_scales=scales)
    raw = torch.tensor([[[[1e-5]], [[-1e-5]], [[2e-5]]]])
    torch.testing.assert_close(model._parameterize_velocity(raw), raw, rtol=1e-5, atol=1e-9)


def test_legacy_model_has_no_trust_region_transform() -> None:
    model = SAGEAVO(hidden_channels=8, graph_heads=1)
    raw = torch.randn(2, 3, 4, 5)
    assert model._parameterize_velocity(raw) is raw
