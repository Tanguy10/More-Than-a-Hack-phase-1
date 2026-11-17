import numpy as np
import pytest
import torch

from possession.relational import (
    CAP,
    LENGTH,
    RelationalPossession,
    ball_context,
    predict_relational,
    set_velocity,
)
from possession.schema import MatchSequence


def inputs(n=80):
    generator = torch.Generator().manual_seed(7)
    return dict(
        xy=torch.rand((1, n, 6, 2), generator=generator) * 30,
        speed=torch.ones(1, n, 6),
        ball=torch.rand(1, n, 2, generator=generator) * 30,
        team=torch.tensor([[0, 0, 0, 1, 1, 1]]),
        pitch=torch.tensor([[105.0, 68.0]]),
        ball_context=torch.zeros(1, n, 2),
    )


def small_model():
    return RelationalPossession(
        channels=16, blocks=2, set_width=16, set_output=16, dropout=0
    ).eval()


def test_context_includes_distant_observations_and_excludes_missing_ball():
    ball = np.full((LENGTH + 2 * CAP, 2), np.nan, "f4")
    ball[CAP - 700] = 0
    assert ball_context(ball)[CAP, 0] == pytest.approx(700 / 1500)
    ball[CAP - 700] = np.nan
    assert ball_context(ball)[CAP, 0] == 1
    rng = np.random.default_rng(1)
    observed = rng.normal(size=(10000, 2)).astype("f4")
    observed[rng.random(10000) < 0.99] = np.nan
    expected = ball_context(observed)
    np.testing.assert_array_equal(
        ball_context(observed[2000 - CAP : 3000 + CAP])[CAP : CAP + 1000], expected[2000:3000]
    )


def test_framewise_player_permutation_and_team_exchange():
    torch.manual_seed(9)
    model = small_model()
    values = inputs()
    # Activate relation branches so the test exercises attention geometry.
    for layer in model.set_encoder.relations:
        torch.nn.init.normal_(layer.project.weight, std=0.1)
    original = model(**values)
    permutation = torch.stack(
        [torch.cat([torch.randperm(3), torch.randperm(3) + 3]) for _ in range(80)]
    )
    changed = {key: value.clone() for key, value in values.items()}
    changed["xy"] = values["xy"].gather(2, permutation[None, :, :, None].expand(1, 80, 6, 2))
    changed["speed"] = values["speed"].gather(2, permutation[None])
    for before, after in zip(original, model(**changed)):
        torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)
    changed = dict(values)
    changed["team"] = 1 - values["team"]
    state, edge = model(**changed)
    order = [1, 0, 2]
    torch.testing.assert_close(state, original[0][..., order], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        edge, original[1][..., order, :][..., :, order], atol=2e-5, rtol=2e-5
    )


def test_missing_observations_keep_outputs_and_gradients_finite():
    model = small_model()
    values = inputs()
    for key in ("xy", "speed", "ball"):
        values[key][:] = float("nan")
    state, edge = model(**values, symmetrise=False)
    assert torch.isfinite(state).all() and torch.isfinite(edge).all()
    (state.square().mean() + edge.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_period_padding_and_chunk_size_preserve_predictions():
    torch.manual_seed(13)
    model = small_model()
    rng = np.random.default_rng(12)
    n = 1470
    xy = rng.normal(size=(n, 4, 2)).astype("f4") + 20
    ball = np.full((n, 2), np.nan, "f4")
    ball[100:300] = 0
    ball[1100:] = 40
    period = np.ones(n, "i1")
    period[700:720] = 0
    period[720:] = 2
    sequence = MatchSequence(
        np.arange(n),
        period,
        xy,
        np.array([0, 0, 1, 1]),
        np.ones((n, 4), "f4"),
        ball,
        (105.0, 68.0),
        "synthetic",
    )
    first, first_edge = predict_relational(model, sequence, central=256, batch_size=2)
    second, second_edge = predict_relational(model, sequence, central=512, batch_size=2)
    np.testing.assert_allclose(first, second, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(first_edge, second_edge, atol=2e-5, rtol=2e-5)


def test_double_precision_preserved_in_geometry_and_heads():
    model = small_model().double()
    values = {
        key: value.double() if value.is_floating_point() else value
        for key, value in inputs().items()
    }
    state, edge = model(**values)
    velocity, confidence = set_velocity(
        values["xy"], torch.isfinite(values["xy"]).all(-1), values["team"]
    )
    assert state.dtype == edge.dtype == velocity.dtype == confidence.dtype == torch.float64
