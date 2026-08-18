import numpy as np
import pytest
import torch

from sage_avo.forward.madagascar import (
    madagascar_availability,
    reflectivity_gather_madagascar,
)
from sage_avo.forward.pipeline import forward_avo_dense_spec
from sage_avo.forward.perturbations import (
    ObservationPerturbationConfig,
    apply_observation_perturbations,
)
from sage_avo.forward.specification import (
    ForwardModelSpecification,
    WaveletSpecification,
    forward_specification_from_mapping,
)
from sage_avo.forward.stacks import DEFAULT_BANDS
from sage_avo.forward.torch_forward import (
    exact_zoeppritz_pp,
    forward_avo_three_band_spec_torch,
)
from sage_avo.forward.zoeppritz import reflectivity_gather
from sage_avo.training.losses import physics_loss_with_context


def _elastic(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    height, width = shape
    rows, columns = np.indices(shape)
    vp = 2800.0 + 3.2 * rows + 90.0 * np.sin(columns / 4.0) + 55.0 * np.sin(rows / 8.0)
    vs = 1550.0 + 1.8 * rows + 45.0 * np.cos(columns / 5.0) + 28.0 * np.sin(rows / 9.0)
    density = 2.35 + 0.0005 * rows + 0.012 * np.sin(columns / 6.0)
    return vp, vs, density


def _specification(phase_degrees: float = 0.0) -> ForwardModelSpecification:
    return ForwardModelSpecification(
        specification_id="v003_round_trip_test",
        angles_degrees=tuple(float(value) for value in range(3, 46)),
        bands=DEFAULT_BANDS,
        dt_seconds=0.004,
        wavelets=(
            WaveletSpecification(
                wavelet_id=f"test_phase_{phase_degrees:g}",
                phase_degrees=phase_degrees,
            ),
        ),
    )


def test_mapping_band_order_is_canonical_after_sorted_json_round_trip():
    config = {
        "forward_model": {
            "specification_id": "mapping_order_test",
            "angles_degrees": {"start": 3.0, "stop": 45.0, "step": 1.0},
            "bands": {
                "far": [31.0, 45.0],
                "mid": [17.0, 31.0],
                "near": [3.0, 17.0],
            },
            "dt_seconds": 0.004,
            "wavelet": {"type": "ricker", "frequency_hz": 14.0, "samples": 81},
            "front_mute": {
                "enabled": True,
                "start": [30.0, 0.0],
                "end": [45.0, 0.1],
                "taper_samples": 5,
            },
        }
    }
    specification = forward_specification_from_mapping(config)
    assert tuple(band.name for band in specification.bands) == ("near", "mid", "far")
    assert tuple(
        (band.minimum_degrees, band.maximum_degrees) for band in specification.bands
    ) == ((3.0, 17.0), (17.0, 31.0), (31.0, 45.0))


def test_numpy_and_torch_shared_forward_operator_round_trip():
    vp, vs, density = _elastic((120, 11))
    specification = _specification(phase_degrees=17.0)
    expected = forward_avo_dense_spec(vp, vs, density, specification).stacks
    actual = forward_avo_three_band_spec_torch(
        torch.tensor(vp[None], dtype=torch.float64),
        torch.tensor(vs[None], dtype=torch.float64),
        torch.tensor(density[None], dtype=torch.float64),
        specification,
    )[0].detach().numpy()
    difference = actual - expected
    assert np.max(np.abs(difference)) < 1e-8
    assert np.sqrt(np.mean(difference**2)) < 1e-9


def test_production_specification_has_canonical_bands_wavelet_and_dt():
    specification = _specification()
    assert tuple(band.name for band in specification.bands) == ("near", "mid", "far")
    assert tuple(
        (band.minimum_degrees, band.maximum_degrees) for band in specification.bands
    ) == ((3.0, 17.0), (17.0, 31.0), (31.0, 45.0))
    assert specification.dt_seconds == 0.004
    assert specification.wavelets[0].peak_frequency_hz == 14.0
    assert specification.wavelets[0].samples == 81


def test_numpy_and_torch_match_above_a_pp_critical_angle():
    height, width = 80, 5
    vp = np.full((height, width), 2200.0)
    vs = np.full((height, width), 1200.0)
    density = np.full((height, width), 2.20)
    vp[40:] = 3300.0
    vs[40:] = 1800.0
    density[40:] = 2.45
    specification = _specification()
    expected = forward_avo_dense_spec(vp, vs, density, specification).stacks
    actual = forward_avo_three_band_spec_torch(
        torch.tensor(vp[None], dtype=torch.float64),
        torch.tensor(vs[None], dtype=torch.float64),
        torch.tensor(density[None], dtype=torch.float64),
        specification,
    )[0].detach().numpy()
    difference = actual - expected
    assert np.max(np.abs(difference)) < 1e-6
    assert np.sqrt(np.mean(difference**2)) < 1e-8


def test_postcritical_torch_operator_retains_finite_gradients():
    vp = torch.tensor(
        [[[[2200.0], [2200.0], [3300.0], [3300.0]]]],
        requires_grad=True,
    )
    vs = torch.tensor([[[[1200.0], [1200.0], [1800.0], [1800.0]]]])
    density = torch.tensor([[[[2.20], [2.20], [2.45], [2.45]]]])
    result = forward_avo_three_band_spec_torch(
        vp,
        vs,
        density,
        _specification(),
    )
    result.square().mean().backward()
    assert vp.grad is not None
    assert torch.isfinite(vp.grad).all()


def test_halo_forward_does_not_restart_patch_mute_or_convolution():
    vp, vs, density = _elastic((180, 9))
    specification = _specification()
    full = forward_avo_dense_spec(vp, vs, density, specification).stacks
    top, core_height = 70, 50
    halo = specification.maximum_wavelet_half_length
    context_top = top - halo
    context_bottom = top + core_height + halo
    context = forward_avo_dense_spec(
        vp[context_top:context_bottom],
        vs[context_top:context_bottom],
        density[context_top:context_bottom],
        specification,
        sample_origin=context_top,
    ).stacks
    np.testing.assert_allclose(
        context[:, halo : halo + core_height],
        full[:, top : top + core_height],
        rtol=1e-5,
        atol=1e-9,
    )


@pytest.mark.parametrize("top", [0, 8, 20, 35, 70, 130])
def test_torch_round_trip_across_global_patch_origins_and_boundaries(top: int):
    """Cover top/bottom halos and samples inside/outside the 0–100 ms mute."""
    vp, vs, density = _elastic((180, 9))
    specification = _specification()
    full = forward_avo_dense_spec(vp, vs, density, specification).stacks
    core_height = 50
    halo = specification.maximum_wavelet_half_length
    context_top = max(0, top - halo)
    context_bottom = min(vp.shape[0], top + core_height + halo)
    core_start = top - context_top
    actual = forward_avo_three_band_spec_torch(
        torch.tensor(vp[None, context_top:context_bottom], dtype=torch.float64),
        torch.tensor(vs[None, context_top:context_bottom], dtype=torch.float64),
        torch.tensor(density[None, context_top:context_bottom], dtype=torch.float64),
        specification,
        sample_origin=context_top,
    )[0, :, core_start : core_start + core_height].detach().numpy()
    np.testing.assert_allclose(
        actual,
        full[:, top : top + core_height],
        rtol=1e-5,
        atol=2e-8,
    )


@pytest.mark.skipif(
    not madagascar_availability().available,
    reason="Madagascar sfzoeppritz2 toolchain is not installed",
)
@pytest.mark.parametrize(
    "values",
    [
        (2500.0, 1400.0, 2.25, 2600.0, 1450.0, 2.28),
        (2200.0, 1200.0, 2.10, 3600.0, 1950.0, 2.50),
    ],
)
def test_madagascar_numpy_torch_interface_convention(values: tuple[float, ...]):
    """Validate ordinary and postcritical real reflected-P coefficients."""
    vp1, vs1, rho1, vp2, vs2, rho2 = values
    vp = np.asarray([[vp1], [vp1], [vp2], [vp2]])
    vs = np.asarray([[vs1], [vs1], [vs2], [vs2]])
    density = np.asarray([[rho1], [rho1], [rho2], [rho2]])
    angles = np.arange(3.0, 46.0)
    madagascar = reflectivity_gather_madagascar(vp, vs, density, angles)[:, 2, 0]
    numpy_result = reflectivity_gather(vp, vs, density, angles)[:, 2, 0]
    torch_result = (
        exact_zoeppritz_pp(
            torch.tensor(vp[None], dtype=torch.float64),
            torch.tensor(vs[None], dtype=torch.float64),
            torch.tensor(density[None], dtype=torch.float64),
            torch.tensor(angles, dtype=torch.float64),
        )[0, :, 2, 0]
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(madagascar, numpy_result, rtol=1e-5, atol=1.1e-6)
    np.testing.assert_allclose(madagascar, torch_result, rtol=1e-5, atol=1.1e-6)


def test_observation_perturbations_are_post_forward_deterministic_and_traceable():
    vp, vs, density = _elastic((90, 12))
    clean = forward_avo_dense_spec(vp, vs, density, _specification()).stacks
    config = ObservationPerturbationConfig(
        enabled=True,
        white_noise_fraction_by_band=(0.01, 0.02, 0.03),
        colored_noise_fraction_by_band=(0.0, 0.01, 0.02),
        coherent_noise_fraction=0.01,
        gain_range_by_band=((0.98, 1.02), (0.98, 1.02), (0.9, 0.95)),
        phase_degrees_by_band=((-3.0, 3.0), (-3.0, 3.0), (-5.0, 5.0)),
        far_angle_weakening_range=(0.5, 0.7),
    )
    first = apply_observation_perturbations(clean, np.random.default_rng(17), config)
    second = apply_observation_perturbations(clean, np.random.default_rng(17), config)
    np.testing.assert_array_equal(first.stacks, second.stacks)
    assert not np.array_equal(first.stacks, clean)
    assert first.metadata == second.metadata
    assert 0.5 <= first.metadata["far_angle_scale"] <= 0.7


def test_disabled_observation_perturbation_is_identity():
    clean = np.arange(3 * 12 * 7, dtype=np.float32).reshape(3, 12, 7)
    result = apply_observation_perturbations(
        clean,
        np.random.default_rng(1),
        ObservationPerturbationConfig(enabled=False),
    )
    np.testing.assert_array_equal(result.stacks, clean)


def test_context_physics_loss_round_trip_matches_stored_stage02_stack():
    vp, vs, density = _elastic((180, 9))
    elastic = np.stack((vp, vs, density))
    specification = _specification()
    stored = forward_avo_dense_spec(vp, vs, density, specification).stacks
    top, core_height = 70, 50
    halo = specification.maximum_wavelet_half_length
    context_top = top - halo
    context = elastic[:, context_top : top + core_height + halo]
    core = elastic[:, top : top + core_height]
    observed = stored[:, top : top + core_height]
    y_mean = torch.tensor([3000.0, 1700.0, 2.4], dtype=torch.float64).view(1, 3, 1, 1)
    y_std = torch.tensor([500.0, 300.0, 0.1], dtype=torch.float64).view(1, 3, 1, 1)
    x_mean = torch.tensor(stored.mean(axis=(1, 2)), dtype=torch.float64).view(1, 3, 1, 1)
    x_std = torch.tensor(stored.std(axis=(1, 2)), dtype=torch.float64).view(1, 3, 1, 1)
    loss = physics_loss_with_context(
        (torch.tensor(core[None], dtype=torch.float64) - y_mean) / y_std,
        (torch.tensor(context[None], dtype=torch.float64) - y_mean) / y_std,
        (torch.tensor(observed[None], dtype=torch.float64) - x_mean) / x_std,
        y_mean,
        y_std,
        x_mean,
        x_std,
        mask=torch.ones((1, 1, core_height, 9), dtype=torch.float64),
        core_start=torch.tensor([halo]),
        sample_origin=torch.tensor([context_top]),
        specification=specification,
    )
    assert float(loss) < 1e-12
