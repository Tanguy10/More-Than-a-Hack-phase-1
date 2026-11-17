from types import SimpleNamespace

import numpy as np
import pytest
from kloppy.domain import Ground

from possession.io import tactical


@pytest.fixture
def tactical_input(tmp_path, monkeypatch):
    folder = tmp_path / "tactical"
    folder.mkdir()
    (folder / "BAR-ATH_tactical_atd.xml").write_bytes(b"<synthetic/>")
    (folder / "BAR-ATH_tactical_atd.txt").write_bytes(b"")
    home = SimpleNamespace(ground=Ground.HOME)
    away = SimpleNamespace(ground=Ground.AWAY)
    home.players = [SimpleNamespace(team=home, player_id="h")]
    away.players = [SimpleNamespace(team=away, player_id="a")]
    metadata = SimpleNamespace(frame_rate=25, teams=[home, away])
    monkeypatch.setattr(tactical, "load_metadata", lambda _: metadata)
    monkeypatch.setattr(tactical, "FRAME_COUNT", 12)
    monkeypatch.setattr(tactical, "GRADED", ((2, 5), (8, 11)))
    rows = [
        dict(
            frame_id=index + 1,
            ball_x=0.1,
            ball_y=0.2,
            player_h_x=0.3,
            player_h_y=0.4,
            player_h_s=5.0,
            player_a_x=0.7,
            player_a_y=0.6,
            player_a_s=4.0,
        )
        for index in range(12)
    ]
    monkeypatch.setattr(tactical, "read_raw_data", lambda *_: iter(rows))
    return tmp_path, rows


def test_units_team_assignment_and_periods(tactical_input):
    path, _ = tactical_input
    sequence = tactical.load_tactical(path)
    np.testing.assert_allclose(sequence.ball_xy[0], [10.53, 13.6], atol=1e-5)
    np.testing.assert_allclose(sequence.players_xy[0, 0], [31.59, 27.2], atol=1e-5)
    np.testing.assert_array_equal(sequence.players_team, [0, 1])
    np.testing.assert_array_equal(sequence.period, [0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0])
    assert sequence.players_speed[0, 0] == 5.0


def test_missing_observations_remain_missing(tactical_input):
    path, rows = tactical_input
    rows[3] = dict(frame_id=4)
    sequence = tactical.load_tactical(path)
    assert np.isnan(sequence.ball_xy[3]).all()
    assert np.isnan(sequence.players_xy[3]).all()
    assert np.isnan(sequence.players_speed[3]).all()


@pytest.mark.parametrize("change", ["duplicate", "missing", "outside"])
def test_invalid_frame_inventory_is_rejected(tactical_input, change):
    path, rows = tactical_input
    if change == "duplicate":
        rows[4]["frame_id"] = rows[3]["frame_id"]
    elif change == "missing":
        rows.pop()
    else:
        rows[-1]["frame_id"] = 13
    with pytest.raises(ValueError):
        tactical.load_tactical(path)
