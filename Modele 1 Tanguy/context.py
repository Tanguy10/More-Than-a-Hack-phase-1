"""MatchContext: raw match data and lazily cached derived features.

Each stoppage, restart or possession detector receives this context.
Add a property to add a feature; it is computed once at first access."""

from functools import cached_property

import numpy as np

import data_io


class MatchContext:
    def __init__(self, tracking=None, set_pieces=None, tagging=None):
        self._tracking = tracking
        self._set_pieces = set_pieces
        self._tagging = tagging
        self.n_frames = data_io.N_FRAMES
        self.fps = data_io.FPS
        self.pitch = np.array(data_io.PITCH, dtype=np.float64)  # (105.3, 68.0) m

    # -- Raw data --

    @cached_property
    def tracking(self):
        return self._tracking if self._tracking is not None else data_io.load_tracking()

    @cached_property
    def tagging(self):
        return self._tagging if self._tagging is not None else data_io.load_smart_tagging()

    @cached_property
    def set_pieces(self):
        if self._set_pieces is not None:
            return self._set_pieces
        return [e for e in self.tagging if e.code.lower() == "set pieces"]

    @cached_property
    def half_blocks(self):
        """Return (half, first frame, last frame) tuples with inclusive bounds."""
        return data_io.half_video_blocks(data_io.GAME)

    @cached_property
    def graded(self):
        return data_io.graded_mask()

    # -- Ball --

    @property
    def ball_xy(self):
        """Normalized 0–1 positions; NaN when the ball is unseen."""
        return self.tracking["ball_xy"]

    @cached_property
    def ball_xy_m(self):
        """Ball position in metres."""
        return self.ball_xy.astype(np.float64) * self.pitch

    @cached_property
    def ball_visible(self):
        return ~np.isnan(self.ball_xy[:, 0])

    @cached_property
    def ball_speed(self):
        """Ball speed in m/s from successive frames; NaN if unknown."""
        speed = np.full(self.n_frames, np.nan)
        speed[1:] = np.linalg.norm(np.diff(self.ball_xy_m, axis=0), axis=1) * self.fps
        return speed

    def ball_in_pitch(self, margin=0.0):
        """True when the ball is observed inside the pitch, with normalized margin."""
        x, y = self.ball_xy[:, 0], self.ball_xy[:, 1]
        with np.errstate(invalid="ignore"):
            return (x >= -margin) & (x <= 1 + margin) & (y >= -margin) & (y <= 1 + margin)

    def ball_out_of_pitch(self, margin=0.0):
        """True when the ball is observed outside the pitch; unseen means False."""
        return self.ball_visible & ~self.ball_in_pitch(margin)

    # -- Tactical-video audio --

    @cached_property
    def audio_band(self):
        """Per-frame 3–5 kHz spectrum: db (N, bins) float16 and freqs (bins,)."""
        import audio
        return audio.load_band_spectrogram()

    @cached_property
    def audio_floor(self):
        """Per-frame 3–5 kHz background level: median bin level in dB."""
        return np.median(self.audio_band["db"], axis=1).astype(np.float32)

    def audio_tone(self, lo_hz, hi_hz):
        """Return emergence above background (dB) and frequency (Hz) of the strongest peak in [lo, hi].

A pure whistle tone emerges by 25–45 dB; crowd noise by about 10 dB."""
        freqs = self.audio_band["freqs"]
        keep = (freqs >= lo_hz) & (freqs <= hi_hz)
        sub = self.audio_band["db"][:, keep]
        k = sub.argmax(axis=1)
        level = sub[np.arange(len(k)), k].astype(np.float32)
        return level - self.audio_floor, freqs[keep][k]

    # -- Players --

    @property
    def players(self):
        """Return (N, slots, 3): normalized x, y and speed in m/s."""
        return self.tracking["players"]

    @property
    def track_team(self):
        """Return (slots,) team codes: 0 = H, 1 = A."""
        return self.tracking["team"]

    @cached_property
    def p_dead(self):
        """Per-frame P(dead ball) from the player-only DFL model in dead_model.py.
NaN outside halves."""
        import dead_model
        return dead_model.predict_p_dead(self)

    @cached_property
    def players_xy_m(self):
        return self.players[:, :, :2].astype(np.float32) * self.pitch.astype(np.float32)

    def half_of(self, frame):
        for half, first, last in self.half_blocks:
            if first <= frame <= last:
                return half
        return None
