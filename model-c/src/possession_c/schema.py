"""Tracking features and period boundaries at 25 Hz."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class MatchSequence:
    frame_index: np.ndarray
    period: np.ndarray
    players_xy: np.ndarray
    players_team: np.ndarray
    players_speed: np.ndarray
    ball_xy: np.ndarray
    pitch: tuple[float, float]
    match_id: str
    fps: int = 25

    @property
    def players_mask(self):
        return np.isfinite(self.players_xy).all(-1)

    @property
    def ball_mask(self):
        return np.isfinite(self.ball_xy).all(-1)

    def validate(self):
        n, slots, dim = self.players_xy.shape
        assert dim == 2 and self.ball_xy.shape == (n, 2)
        assert self.players_speed.shape == (n, slots)
        assert self.players_team.shape == (slots,)
        assert self.frame_index.shape == self.period.shape == (n,)
        assert np.all(np.diff(self.frame_index) > 0)
        assert set(np.unique(self.players_team)) <= {0, 1, -1}
        assert self.fps == 25 and min(self.pitch) > 0
        return self

    def save(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for key in (
            "frame_index",
            "period",
            "players_xy",
            "players_team",
            "players_speed",
            "ball_xy",
        ):
            np.save(root / f"{key}.npy", getattr(self, key), allow_pickle=False)
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "match_id": self.match_id,
                    "pitch": self.pitch,
                    "fps": self.fps,
                    "schema": 1,
                    "feature_allowlist": [
                        "xy",
                        "team_side",
                        "speed",
                        "masks",
                        "observation_quality",
                    ],
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, directory, mmap=True):
        root = Path(directory)
        meta = json.loads((root / "metadata.json").read_text())
        arrays = {
            key: np.load(root / f"{key}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
            for key in (
                "frame_index",
                "period",
                "players_xy",
                "players_team",
                "players_speed",
                "ball_xy",
            )
        }
        return cls(
            **arrays, pitch=tuple(meta["pitch"]), match_id=meta["match_id"], fps=meta["fps"]
        ).validate()


def period_runs(period):
    """Half-open contiguous runs with a positive period; never bridge a break."""
    edges = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a and period[a] > 0]
