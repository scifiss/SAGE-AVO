"""Regression and gradient tests for the matrix-solve exact Zoeppritz operator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sage_avo.forward.torch_forward import (
    exact_zoeppritz_pp_closed_form,
    exact_zoeppritz_pp_matrix,
)
from sage_avo.forward.zoeppritz import zoeppritz_pp


CASES = (
    (3000.0, 1600.0, 2.30, 3010.0, 1605.0, 2.305),
    (2700.0, 1450.0, 2.25, 3200.0, 1750.0, 2.42),
    (2200.0, 1200.0, 2.10, 3600.0, 1950.0, 2.50),
    (4200.0, 2500.0, 2.65, 2100.0, 1100.0, 2.10),
)


@pytest.mark.parametrize("case", CASES)
def test_matrix_zoeppritz_matches_numpy_exact_solution(
    case: tuple[float, ...],
) -> None:
    angles = torch.arange(3.0, 46.0, dtype=torch.float64)
    vp = torch.tensor([[[case[0]], [case[3]]]], dtype=torch.float64)
    vs = torch.tensor([[[case[1]], [case[4]]]], dtype=torch.float64)
    density = torch.tensor([[[case[2]], [case[5]]]], dtype=torch.float64)
    actual = exact_zoeppritz_pp_matrix(vp, vs, density, angles)[0, :, 1, 0]
    expected = np.asarray(
        [zoeppritz_pp(*case, float(angle)) for angle in angles],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-8, atol=2e-9)


def test_matrix_and_closed_form_match_on_stable_states() -> None:
    angles = torch.arange(3.0, 46.0, dtype=torch.float64)
    case = CASES[1]
    vp = torch.tensor([[[case[0]], [case[3]]]], dtype=torch.float64)
    vs = torch.tensor([[[case[1]], [case[4]]]], dtype=torch.float64)
    density = torch.tensor([[[case[2]], [case[5]]]], dtype=torch.float64)
    matrix = exact_zoeppritz_pp_matrix(vp, vs, density, angles)
    closed_form = exact_zoeppritz_pp_closed_form(vp, vs, density, angles)
    torch.testing.assert_close(matrix, closed_form, rtol=5e-7, atol=5e-8)


def test_matrix_zoeppritz_gradcheck() -> None:
    angles = torch.tensor([3.0, 17.0, 31.0, 45.0], dtype=torch.float64)
    vp = torch.tensor([[[2700.0], [3200.0]]], dtype=torch.float64, requires_grad=True)
    vs = torch.tensor([[[1450.0], [1750.0]]], dtype=torch.float64, requires_grad=True)
    density = torch.tensor([[[2.25], [2.42]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda *values: exact_zoeppritz_pp_matrix(*values, angles),
        (vp, vs, density),
        eps=1e-5,
        atol=2e-5,
        rtol=2e-4,
    )


@pytest.mark.parametrize("case", CASES)
def test_matrix_zoeppritz_forward_and_backward_are_finite(
    case: tuple[float, ...],
) -> None:
    angles = torch.arange(3.0, 46.0, dtype=torch.float64)
    vp = torch.tensor([[[case[0]], [case[3]]]], dtype=torch.float64, requires_grad=True)
    vs = torch.tensor([[[case[1]], [case[4]]]], dtype=torch.float64, requires_grad=True)
    density = torch.tensor(
        [[[case[2]], [case[5]]]], dtype=torch.float64, requires_grad=True
    )
    output = exact_zoeppritz_pp_matrix(vp, vs, density, angles)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in (vp, vs, density)
    )
