"""Numerical contract tests for masked reductions used by the physics loss."""

from __future__ import annotations

import pytest
import torch

import sage_avo.training.losses as losses
from sage_avo.training.losses import masked_mse, physics_loss_with_context


def _expanded_mask(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if mask.ndim == prediction.ndim - 1:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1:
        mask = mask.expand_as(prediction)
    return mask.to(device=prediction.device, dtype=prediction.dtype)


def _eligible_only_reference(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicitly gather eligible values before squaring them."""
    expanded = _expanded_mask(mask, prediction)
    eligible = expanded != 0
    if bool(eligible.any()):
        residual = prediction[eligible] - target[eligible]
        numerator = (residual.square() * expanded[eligible]).sum()
    else:
        # Preserve a differentiable zero, as required by an inactive training batch.
        numerator = (prediction - target)[eligible].sum()
    denominator = expanded.sum() + 1e-8
    return numerator / denominator, denominator


CASES = {
    "all_eligible": torch.ones(2, 1, 2, 3, dtype=torch.float64),
    "partially_eligible": torch.tensor(
        [[[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]]],
        dtype=torch.float64,
    ),
    "one_eligible": torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
        dtype=torch.float64,
    ),
    "all_ineligible": torch.zeros(2, 1, 2, 3, dtype=torch.float64),
    "ordinary_finite": torch.tensor(
        [[[[0.25, 0.0, 1.5], [0.0, 0.75, 0.0]]], [[[0.0, 1.0, 0.0], [2.0, 0.0, 0.5]]]],
        dtype=torch.float64,
    ),
}


@pytest.mark.parametrize("case", tuple(CASES))
def test_masked_mse_matches_explicit_eligible_only_reference(case: str) -> None:
    prediction = torch.linspace(-2.0, 3.0, 36, dtype=torch.float64).reshape(2, 3, 2, 3)
    prediction.requires_grad_(True)
    target = torch.linspace(1.5, -1.0, 36, dtype=torch.float64).reshape_as(prediction)
    mask = CASES[case]

    actual = masked_mse(prediction, target, mask)
    reference, reference_denominator = _eligible_only_reference(prediction, target, mask)
    actual_gradient = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, prediction)[0]
    expanded = _expanded_mask(mask, prediction)

    torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(
        expanded.sum() + 1e-8,
        reference_denominator,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(actual_gradient[expanded == 0]).item() == 0
    assert torch.isfinite(actual)


def test_extreme_finite_inactive_residual_is_excluded_before_square() -> None:
    prediction = torch.tensor(
        [[[[1.0, torch.finfo(torch.float32).max / 4.0], [2.0, 3.0]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

    actual = masked_mse(prediction, target, mask)
    reference, denominator = _eligible_only_reference(prediction, target, mask)
    gradient = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, prediction)[0]

    legacy = ((prediction - target).square() * mask).sum() / denominator
    assert torch.isnan(legacy)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)
    torch.testing.assert_close(gradient, reference_gradient, rtol=0.0, atol=0.0)
    assert gradient[0, 0, 0, 1].item() == 0.0
    assert gradient[0, 0, 1, 1].item() == 0.0


def test_fully_inactive_mask_returns_autograd_safe_exact_zero() -> None:
    prediction = torch.full(
        (2, 3, 4, 5),
        torch.finfo(torch.float32).max / 4.0,
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 1, 4, 5)

    loss = masked_mse(prediction, target, mask)
    gradient = torch.autograd.grad(loss, prediction)[0]

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    assert torch.count_nonzero(gradient).item() == 0
    assert torch.isfinite(gradient).all()


def test_context_physics_subsets_ineligible_samples_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_forward_batches: list[int] = []

    def finite_forward(
        vp: torch.Tensor,
        vs: torch.Tensor,
        density: torch.Tensor,
        specification: object,
        *,
        sample_origin: torch.Tensor,
    ) -> torch.Tensor:
        del specification
        assert torch.isfinite(vp).all()
        assert torch.isfinite(vs).all()
        assert torch.isfinite(density).all()
        assert sample_origin.shape == (1,)
        observed_forward_batches.append(vp.shape[0])
        return torch.stack((vp * 0.0, vp * 0.0, vp * 0.0), dim=1)

    monkeypatch.setattr(losses, "forward_avo_three_band_spec_torch", finite_forward)
    prediction = torch.zeros(2, 3, 2, 2, requires_grad=True)
    context = torch.zeros(2, 3, 4, 2)
    context[1] = torch.nan
    observed = torch.zeros(2, 3, 2, 2)
    mask = torch.zeros(2, 1, 2, 2)
    mask[0] = 1.0
    channel_statistics = torch.zeros(1, 3, 1, 1)
    channel_scale = torch.ones(1, 3, 1, 1)
    loss = physics_loss_with_context(
        prediction,
        context,
        observed,
        channel_statistics,
        channel_scale,
        channel_statistics,
        channel_scale,
        mask=mask,
        core_start=torch.tensor((1, 1)),
        sample_origin=torch.tensor((10, 20)),
        specification=object(),
    )
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert observed_forward_batches == [1]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[1]).item() == 0
