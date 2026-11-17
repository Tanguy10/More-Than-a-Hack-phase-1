import torch
from possession.model import PossessionTCN


def make_inputs():
    torch.manual_seed(0)
    xy = torch.rand(2, 48, 7, 2) * 60
    xy[0, :, 0] = float("nan")
    speed = torch.rand(2, 48, 7) * 8
    ball = torch.rand(2, 48, 2) * 60
    ball[:, 10:20] = float("nan")
    team = torch.tensor([[0, 0, 0, 1, 1, 1, -1]] * 2)
    pitch = torch.tensor([[105.0, 68.0]] * 2)
    return dict(xy=xy, speed=speed, ball=ball, team=team, pitch=pitch)


def make_model():
    return PossessionTCN(channels=16, blocks=2, set_width=8, set_output=16, dropout=0).eval()


@torch.no_grad()
def test_set_model_is_permutation_invariant():
    model = make_model()
    inputs = make_inputs()
    a, b = model(**inputs)
    permutation = torch.tensor([6, 2, 1, 5, 0, 4, 3])
    inputs["xy"] = inputs["xy"][:, :, permutation]
    inputs["speed"] = inputs["speed"][:, :, permutation]
    inputs["team"] = inputs["team"][:, permutation]
    other, ob = model(**inputs)
    torch.testing.assert_close(other, a, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(ob, b, atol=2e-6, rtol=1e-5)


@torch.no_grad()
def test_full_model_exchange_swaps_states_and_transition_axes():
    model = make_model()
    inputs = make_inputs()
    a, b = model(**inputs)
    inputs["team"] = torch.where(inputs["team"] >= 0, 1 - inputs["team"], inputs["team"])
    other, ob = model(**inputs)
    order = [1, 0, 2]
    torch.testing.assert_close(other[..., order], a, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(ob[..., order, :][..., :, order], b, atol=2e-6, rtol=1e-5)


def test_all_ball_missing_and_empty_team_have_finite_gradients():
    model = make_model()
    inputs = make_inputs()
    inputs["ball"][:] = float("nan")
    inputs["xy"][:, :, 3:] = float("nan")
    state, boundary = model(**inputs)
    loss = -state[..., 0].mean() + boundary.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@torch.no_grad()
def test_reference_double_precision_runs_without_cuda():
    model = make_model().double()
    inputs = make_inputs()
    inputs = {k: v.double() if v.is_floating_point() else v for k, v in inputs.items()}
    state, boundary = model(**inputs)
    assert state.dtype == boundary.dtype == torch.float64
    assert torch.isfinite(state).all() and torch.isfinite(boundary).all()
