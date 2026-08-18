import numpy as np
import pytest

from sage_avo.forward.pipeline import ForwardConfig, forward_avo_three_band
from sage_avo.forward.stacks import (
    DEFAULT_BANDS,
    AngleBand,
    representative_band_angles,
    validate_bands,
)
from sage_avo.forward.zoeppritz import zoeppritz_pp


def test_zoeppritz_matches_normal_incidence_impedance_contrast():
    vp1, rho1 = 2500.0, 2.1
    vp2, rho2 = 3000.0, 2.3
    expected = (vp2 * rho2 - vp1 * rho1) / (vp2 * rho2 + vp1 * rho1)
    actual = zoeppritz_pp(vp1, 1300.0, rho1, vp2, 1600.0, rho2, 0.0)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_default_bands_share_only_the_intentional_endpoints():
    validate_bands(DEFAULT_BANDS)
    assert DEFAULT_BANDS[0].maximum_degrees == DEFAULT_BANDS[1].minimum_degrees == 17.0
    assert DEFAULT_BANDS[1].maximum_degrees == DEFAULT_BANDS[2].minimum_degrees == 31.0
    assert representative_band_angles(DEFAULT_BANDS) == (10.0, 24.0, 38.0)


def test_band_validation_rejects_overlap_beyond_a_shared_endpoint():
    with pytest.raises(ValueError, match="overlap beyond"):
        validate_bands(
            (
                AngleBand("near", 3.0, 18.0),
                AngleBand("mid", 17.0, 31.0),
                AngleBand("far", 31.0, 45.0),
            )
        )


def test_forward_pipeline_shape_and_finiteness():
    vp = np.full((24, 5), 2500.0)
    vs = np.full((24, 5), 1300.0)
    density = np.full((24, 5), 2.1)
    vp[12:] = 3000.0
    vs[12:] = 1600.0
    density[12:] = 2.3
    output = forward_avo_three_band(vp, vs, density, ForwardConfig(apply_mute=False))
    assert output.shape == (3, 24, 5)
    assert np.isfinite(output).all()


def test_numpy_and_torch_forward_operators_agree_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from sage_avo.forward.torch_forward import forward_avo_three_band_torch

    vp = np.full((24, 5), 2500.0, dtype=np.float32)
    vs = np.full_like(vp, 1300.0)
    density = np.full_like(vp, 2.1)
    vp[12:], vs[12:], density[12:] = 3000.0, 1600.0, 2.3
    expected = forward_avo_three_band(vp, vs, density)
    actual = forward_avo_three_band_torch(
        torch.from_numpy(vp)[None],
        torch.from_numpy(vs)[None],
        torch.from_numpy(density)[None],
    )[0].numpy()
    for band in range(3):
        correlation = np.corrcoef(expected[band].ravel(), actual[band].ravel())[0, 1]
        relative_error = np.linalg.norm(expected[band] - actual[band]) / np.linalg.norm(expected[band])
        assert correlation > 0.999
        assert relative_error < 0.02


def test_torch_analytic_zoeppritz_matches_numpy_matrix_solution():
    torch = pytest.importorskip("torch")
    from sage_avo.forward.torch_forward import exact_zoeppritz_pp

    angles = np.array([0.0, 10.0, 25.0, 40.0], dtype=np.float32)
    expected = np.array(
        [zoeppritz_pp(2700.0, 1450.0, 2.25, 3200.0, 1750.0, 2.42, angle) for angle in angles]
    )
    vp = torch.tensor([[[2700.0], [3200.0]]])
    vs = torch.tensor([[[1450.0], [1750.0]]])
    density = torch.tensor([[[2.25], [2.42]]])
    actual = exact_zoeppritz_pp(vp, vs, density, torch.from_numpy(angles))[0, :, 1, 0]
    np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-6)
