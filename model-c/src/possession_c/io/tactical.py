"""Stream EPTS using its declared channel map, without materialising Kloppy Frames."""

from pathlib import Path
import numpy as np
from kloppy.domain import Ground
from kloppy.infra.serializers.tracking.metrica_epts.metadata import load_metadata
from kloppy.infra.serializers.tracking.metrica_epts.reader import read_raw_data
from possession_c.schema import MatchSequence

FRAME_COUNT = 199319
GRADED = ((11185, 90652), (110808, 190198))


def load_tactical(data_dir):
    stem = Path(data_dir) / "tactical" / "BAR-ATH_tactical"
    with open(str(stem) + "_atd.xml", "rb") as handle:
        metadata = load_metadata(handle)
    assert metadata.frame_rate == 25
    players = [p for team in metadata.teams for p in team.players]
    team = np.array([0 if p.team.ground == Ground.HOME else 1 for p in players], np.int8)
    xy = np.full((FRAME_COUNT, len(players), 2), np.nan, np.float32)
    speed = np.full((FRAME_COUNT, len(players)), np.nan, np.float32)
    ball = np.full((FRAME_COUNT, 2), np.nan, np.float32)
    seen = np.zeros(FRAME_COUNT, bool)
    with open(str(stem) + "_atd.txt", "rb") as handle:
        for row in read_raw_data(handle, metadata):
            index = int(row["frame_id"]) - 1
            if index < 0 or index >= FRAME_COUNT or seen[index]:
                raise ValueError("ATD frame index violates published contract")
            seen[index] = True
            ball[index] = (row.get("ball_x", np.nan), row.get("ball_y", np.nan))
            for slot, player in enumerate(players):
                prefix = f"player_{player.player_id}_"
                xy[index, slot] = (row.get(prefix + "x", np.nan), row.get(prefix + "y", np.nan))
                speed[index, slot] = row.get(prefix + "s", np.nan)
    if not seen.all():
        raise ValueError("ATD is not exactly one record per tactical video frame")
    xy *= np.array([105.3, 68.0], np.float32)
    ball *= np.array([105.3, 68.0], np.float32)
    period = np.zeros(FRAME_COUNT, np.int8)
    for p, (start, stop) in enumerate(GRADED, 1):
        period[start:stop] = p
    return MatchSequence(
        np.arange(FRAME_COUNT, dtype=np.int64),
        period,
        xy,
        team,
        speed,
        ball,
        (105.3, 68.0),
        "BAR-ATH",
    ).validate()
