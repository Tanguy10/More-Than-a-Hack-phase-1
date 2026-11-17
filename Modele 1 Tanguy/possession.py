"""Predict H/A possession while the ball is alive.

PossessionModel.predict(ctx, dead_result) returns one H/A value per frame.
The pipeline later overrides D frames. dead_result lets models avoid
smoothing across stoppages or use restarting-team evidence. Add a
PossessionModel subclass and pass it to the pipeline to try another method."""

from abc import ABC, abstractmethod

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1
LETTERS = np.array(["H", "A"])


class PossessionModel(ABC):
    name = "possession"

    @abstractmethod
    def predict(self, ctx, dead_result) -> np.ndarray:
        """Return an H/A label for every frame, dtype <U1."""
        ...


# --------------------------------------------------------------------------------------------
# Reusable components
# --------------------------------------------------------------------------------------------

def nearest_player_team(ctx, chunk=20000):
    """Return nearest-player team 0/1 per frame, or NOBODY when the ball is unseen."""
    out = np.full(ctx.n_frames, NOBODY, dtype=np.int8)
    team = ctx.track_team
    ball = ctx.ball_xy_m.astype(np.float32)
    for s in range(0, ctx.n_frames, chunk):
        e = min(ctx.n_frames, s + chunk)
        players = ctx.players_xy_m[s:e]                          # (n, slots, 2)
        d2 = np.sum((players - ball[s:e, None, :]) ** 2, axis=2)  # NaN when absent
        d2 = np.where(np.isnan(d2), np.inf, d2)
        best = np.argmin(d2, axis=1)
        ok = np.isfinite(d2[np.arange(e - s), best])
        out[s:e][ok] = team[best[ok]]
    return out


def vote_within_segments(codes, segments, window):
    """Centred ±window majority vote confined to each alive interval.

Frames with no votes inherit the nearest decision in the same interval;
an interval with no votes defaults to H."""
    labels = np.full(len(codes), HOME, dtype=np.int8)
    for start, end in segments:
        seg = codes[start:end]
        n = len(seg)
        home = np.concatenate(([0], np.cumsum(seg == HOME)))
        away = np.concatenate(([0], np.cumsum(seg == AWAY)))
        idx = np.arange(n)
        lo = np.maximum(0, idx - window)
        hi = np.minimum(n, idx + window + 1)
        h, a = home[hi] - home[lo], away[hi] - away[lo]

        vote = np.where(h + a == 0, NOBODY, np.where(h >= a, HOME, AWAY)).astype(np.int8)
        known = np.flatnonzero(vote != NOBODY)
        if len(known):
            # Assign each frame the nearest known frame value.
            pos = np.searchsorted(known, idx)
            right = known[np.minimum(pos, len(known) - 1)]
            left = known[np.maximum(pos - 1, 0)]
            vote = vote[np.where(np.abs(idx - left) <= np.abs(right - idx), left, right)]
        else:
            vote[:] = HOME
        labels[start:end] = vote
    return labels


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------

class NearestPlayerPossession(PossessionModel):
    """Baseline: nearest player to ball plus a majority vote within alive phases."""
    name = "nearest_player"

    def __init__(self, window=50):
        self.window = window

    def predict(self, ctx, dead_result):
        codes = nearest_player_team(ctx)
        votes = vote_within_segments(codes, dead_result.alive_segments(ctx), self.window)
        return LETTERS[votes]


class ConstantPossession(PossessionModel):
    """Always H: isolate D quality without possession-model noise."""
    name = "constant"

    def __init__(self, label="H"):
        self.label = label

    def predict(self, ctx, dead_result):
        return np.full(ctx.n_frames, self.label, dtype="<U1")
