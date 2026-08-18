import pytest


torch = pytest.importorskip("torch")
flow = pytest.importorskip("sage_avo.training.flow")
heun_integrate = flow.heun_integrate
straight_path = flow.straight_path


def test_straight_path_endpoints_and_target_velocity():
    low = torch.randn(2, 3, 5, 4)
    target = torch.randn_like(low)
    times = torch.tensor([0.0, 1.0])
    state, velocity = straight_path(low, target, times)
    torch.testing.assert_close(state[0], low[0])
    torch.testing.assert_close(state[1], target[1])
    torch.testing.assert_close(velocity, target - low)


def test_straight_path_matches_closed_form_at_interior_time():
    low = torch.randn(3, 3, 4, 6)
    target = torch.randn_like(low)
    time = torch.tensor([0.2, 0.5, 0.8])
    state, _ = straight_path(low, target, time)
    fraction = time[:, None, None, None]
    torch.testing.assert_close(state, (1.0 - fraction) * low + fraction * target)


def test_heun_integrates_constant_velocity_exactly():
    initial = torch.randn(2, 3, 4, 5)
    constant = torch.full_like(initial, 2.5)

    def velocity(state, time):
        assert state.shape[0] == time.shape[0]
        return constant

    result = heun_integrate(initial, velocity, steps=7)
    torch.testing.assert_close(result, initial + constant)


def test_flow_shape_validation():
    low = torch.zeros(2, 3, 4, 5)
    with pytest.raises(ValueError, match="share a shape"):
        straight_path(low, torch.zeros(1, 3, 4, 5), torch.zeros(2))
    with pytest.raises(ValueError, match="one value"):
        straight_path(low, low, torch.zeros(1))
