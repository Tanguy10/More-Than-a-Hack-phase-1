"""Detect dead ball (D) versus alive play with a two-state machine.

ALIVE -- stop event --> DEAD -- restart event --> ALIVE.
Stop detectors include out-of-bounds, referee whistle and player-only DFL model.
Restart detectors include tagged set pieces, whistle followed by a kick,
tracking-only kicks, model phase endings and timeouts. Fallback/Earliest
combine sources. Sparse or missing tags require fallback detection to avoid
leaving play in D for minutes. All indices are video frames; intervals are
[start, end), with an exclusive end. This is an exploratory pipeline, not
the retained submission model."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

# Set-piece Type values that can restart play, compared in lowercase.
RESTART_TYPES = (
    "throw-in",
    "corner kick",
    "goal kick",
    "free kick",
    "kick off (after goal)",
    "kick off (start)",
)
KICKOFF_START_TYPES = ("kick off (start)",)

TEAM_OF_CLUB = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


# --------------------------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------------------------

@dataclass
class Event:
    """A detector trigger."""
    frame: int
    source: str                 # Detector name
    detail: str = ""            # For example Throw-in or x=-0.01 y=0.35
    team: Optional[str] = None  # H/A if the restarting team is known
    meta: dict = field(default_factory=dict)  # Detector measurements, such as whistle strength


@dataclass
class DeadSegment:
    start: int                  # First D frame
    end: int                    # First subsequent alive frame, excluded
    stop: Event
    restart: Optional[Event]    # None if the half ends before a restart

    @property
    def duration_s(self):
        return (self.end - self.start) / 25.0


@dataclass
class DeadBallResult:
    dead: np.ndarray                                   # Boolean, one value per frame
    segments: List[DeadSegment] = field(default_factory=list)

    def alive_segments(self, ctx):
        """Return alive intervals [start, end) inside the halves."""
        out = []
        for _, first, last in ctx.half_blocks:
            alive = ~self.dead[first:last + 1]
            edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
            starts = np.flatnonzero(edges == 1) + first
            ends = np.flatnonzero(edges == -1) + first
            out.extend(zip(starts.tolist(), ends.tolist()))
        return out


# --------------------------------------------------------------------------------------------
# Stop detectors
# --------------------------------------------------------------------------------------------

class StopDetector(ABC):
    """Return sorted stoppage times from this source.

The state machine ignores stops while already dead, so repeated proposals
do not create duplicate phases."""
    name = "stop"

    @abstractmethod
    def detect(self, ctx) -> List[Event]:
        ...


class OutOfBoundsStop(StopDetector):
    """Detect the ball leaving the pitch.

margin is a normalized line tolerance (0.01 is about 1 m lengthwise or
0.7 m across). min_frames suppresses short out-of-bounds noise.
backdate_gap allows dating a stop to the last observed frame + 1 when
the ball disappears at the line and later reappears outside within that gap."""
    name = "out_of_bounds"

    def __init__(self, margin=0.01, min_frames=3, backdate_gap=25):
        self.margin = margin
        self.min_frames = min_frames
        self.backdate_gap = backdate_gap

    def detect(self, ctx):
        out = ctx.ball_out_of_pitch(self.margin)
        edges = np.diff(np.concatenate(([0], out.astype(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        visible_idx = np.flatnonzero(ctx.ball_visible)
        inside = ctx.ball_in_pitch(0.0)
        second = int(ctx.fps)

        events = []
        for s, e in zip(starts, ends):
            if e - s < self.min_frames:
                continue
            frame = int(s)
            # Last observed ball frame before s
            k = np.searchsorted(visible_idx, s) - 1
            if k >= 0 and 1 < s - visible_idx[k] <= self.backdate_gap:
                frame = int(visible_idx[k]) + 1
            x, y = ctx.ball_xy[s]
            # Was the ball observed crossing the line (inside during the previous second
            # and not far outside)? Otherwise this may be a ball-boy observation or
            # a ball rediscovered after a long gap, already in a player’s hands.
            dist_m = (max(-x, x - 1, 0) * ctx.pitch[0]) + (max(-y, y - 1, 0) * ctx.pitch[1])
            events.append(Event(frame, self.name, "x={:.2f} y={:.2f}".format(x, y),
                                meta={"continuous": bool(inside[max(0, s - second):s].any()),
                                      "dist_m": float(dist_m)}))
        return events


class RefereeStop(StopDetector):
    """Referee stoppage while the ball is inside the pitch.

WhistleStop uses audio. Other possible cues include collective slowing,
a stationary ball, a later Free Kick tag or referee gestures in video."""
    name = "referee"


class NoRefereeStop(RefereeStop):
    """Placeholder: no referee stoppages detected."""
    name = "referee_none"

    def detect(self, ctx):
        return []


def _runs(mask):
    """Return [start, end) intervals of True values."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


class WhistleFinder:
    """Find referee whistles in tactical-video audio.

The observed signature has simultaneous tones near 3.7–3.9 and 4.2–4.3 kHz,
400–520 Hz apart. Crowd noise is mostly below 3 kHz; chants drift and lack
this paired spacing. A whistle frame requires both tones above background
with the correct gap. Merge runs closer than merge_gap and keep at least
min_frames. Shared by whistle stop and restart detectors."""

    def __init__(self, band_a=(3650, 3950), band_b=(4100, 4450), tone_a_db=15.0, tone_b_db=12.0,
                 spacing_hz=(380, 560), min_frames=5, merge_gap=3):
        self.band_a = band_a
        self.band_b = band_b
        self.tone_a_db = tone_a_db
        self.tone_b_db = tone_b_db
        self.spacing_hz = spacing_hz
        self.min_frames = min_frames
        self.merge_gap = merge_gap
        self._cache = {}

    def whistles(self, ctx):
        """Return sorted (start, end, peak emergence dB) whistle intervals [start, end)."""
        key = id(ctx)
        if key in self._cache:
            return self._cache[key]
        level_a, freq_a = ctx.audio_tone(*self.band_a)
        level_b, freq_b = ctx.audio_tone(*self.band_b)
        spacing = freq_b - freq_a
        on = ((level_a >= self.tone_a_db) & (level_b >= self.tone_b_db)
              & (spacing >= self.spacing_hz[0]) & (spacing <= self.spacing_hz[1]))
        # Merge fragments of the same whistle when a tone flickers briefly.
        for s, e in _runs(~on):
            if s > 0 and e < len(on) and e - s <= self.merge_gap:
                on[s:e] = True
        out = [(s, e, float(np.minimum(level_a, level_b)[s:e].max()))
               for s, e in _runs(on) if e - s >= self.min_frames]
        self._cache[key] = out
        return out


class WhistleStop(RefereeStop):
    """STOP: referee whistle with the ball inside the pitch.

in_pitch_only rejects whistles following an observed out-of-bounds ball
within out_window_s; OutOfBoundsStop covers those. offset_s shifts the stop
relative to whistle onset. Some whistles are inaudible and some authorize
restarts. Whistles during D are ignored; a missed stop can therefore make
a restart whistle create a wrongly timed dead phase."""
    name = "whistle"

    def __init__(self, finder=None, in_pitch_only=True, out_window_s=1.0, offset_s=0.0):
        self.finder = finder or WhistleFinder()
        self.in_pitch_only = in_pitch_only
        self.out_window_s = out_window_s
        self.offset_s = offset_s

    def detect(self, ctx):
        out_of_pitch = ctx.ball_out_of_pitch(0.01)
        window = int(round(self.out_window_s * ctx.fps))
        offset = int(round(self.offset_s * ctx.fps))
        events = []
        for s, e, level in self.finder.whistles(ctx):
            if self.in_pitch_only and out_of_pitch[max(0, s - window):s + 1].any():
                continue
            events.append(Event(s + offset, self.name,
                                "{:.1f}s {:.0f}dB".format((e - s) / ctx.fps, level),
                                meta={"duration_s": (e - s) / ctx.fps, "level_db": level}))
        return events


# --------------------------------------------------------------------------------------------
# Timing: locate a restart within a candidate interval.
# --------------------------------------------------------------------------------------------

class KickFinder:
    """Find kick onsets where the in-pitch ball exceeds min_speed for min_frames.

When still_window > 0, require the preceding ball to be slow or unseen
for at least still_ratio of that window, consistent with a placed or held ball."""

    def __init__(self, min_speed=4.0, min_frames=3, margin=0.01,
                 still_window=0, still_speed=2.0, still_ratio=0.8):
        self.min_speed = min_speed
        self.min_frames = min_frames
        self.margin = margin
        self.still_window = still_window
        self.still_speed = still_speed
        self.still_ratio = still_ratio
        self._cache = {}

    def candidates(self, ctx):
        key = id(ctx)
        if key in self._cache:
            return self._cache[key]

        speed = ctx.ball_speed
        with np.errstate(invalid="ignore"):
            fast = (speed >= self.min_speed) & ctx.ball_in_pitch(self.margin)
        # Require min_frames consecutive fast frames starting at f.
        run = np.convolve(fast.astype(np.int16), np.ones(self.min_frames, dtype=np.int16),
                          mode="full")[self.min_frames - 1:]
        sustained = run >= self.min_frames
        rising = sustained & ~np.concatenate(([False], sustained[:-1]))

        if self.still_window > 0:
            with np.errstate(invalid="ignore"):
                still = np.isnan(speed) | (speed < self.still_speed)
            csum = np.concatenate(([0], np.cumsum(still)))
            idx = np.arange(len(speed))
            lo = np.maximum(0, idx - self.still_window)
            ratio = (csum[idx] - csum[lo]) / np.maximum(1, idx - lo)
            rising &= ratio >= self.still_ratio

        frames = np.flatnonzero(rising)
        self._cache[key] = frames
        return frames

    def first_in(self, ctx, lo, hi):
        """Return the first kick in [lo, hi], or None."""
        frames = self.candidates(ctx)
        k = np.searchsorted(frames, lo)
        if k < len(frames) and frames[k] <= hi:
            return int(frames[k])
        return None


class Timing(ABC):
    """Strategy for restart timing within a tag, given dead play since `since`."""

    @abstractmethod
    def locate(self, ctx, tag, since, limit) -> Optional[int]:
        ...


class TagStart(Timing):
    """Restart at tag start plus a seconds offset: the literal strategy."""

    def __init__(self, offset_s=0.0):
        self.offset_s = offset_s

    def locate(self, ctx, tag, since, limit):
        frame = tag.start_frame + int(round(self.offset_s * ctx.fps))
        return frame if since < frame <= limit else None


class FirstKick(Timing):
    """Restart at the first kick inside the tag interval, extended by pre_s.

Tag starts are not restart instants; use them as search windows.
fallback_to_start uses the tag start when no kick is found."""

    def __init__(self, kick=None, pre_s=0.0, fallback_to_start=False):
        self.kick = kick or KickFinder()
        self.pre_s = pre_s
        self.fallback_to_start = fallback_to_start

    def locate(self, ctx, tag, since, limit):
        lo = max(since + 1, tag.start_frame - int(round(self.pre_s * ctx.fps)))
        hi = min(limit, tag.end_frame)
        if lo > hi:
            return None
        frame = self.kick.first_in(ctx, lo, hi)
        if frame is None and self.fallback_to_start and since < tag.start_frame <= limit:
            frame = tag.start_frame
        return frame


# --------------------------------------------------------------------------------------------
# Restart detectors
# --------------------------------------------------------------------------------------------

class RestartDetector(ABC):
    """Given dead play since `since`, return the first restart Event in (since, limit], or None."""
    name = "restart"

    @abstractmethod
    def next_restart(self, ctx, since, limit) -> Optional[Event]:
        ...


class SmartTaggingRestart(RestartDetector):
    """Restart from a Smart Tagging set piece."""
    name = "smart_tagging"

    def __init__(self, types=RESTART_TYPES, timing=None, max_lead_s=2.0):
        self.types = tuple(t.lower() for t in types)
        self.timing = timing or FirstKick()
        # Require a tag to appear after the stop, allowing max_lead_s; otherwise
        # a long interval from an earlier set piece could restart play incorrectly.
        self.max_lead_s = max_lead_s

    def _tags(self, ctx):
        return [e for e in ctx.set_pieces if e.type.lower() in self.types]

    def next_restart(self, ctx, since, limit):
        earliest_start = since - int(round(self.max_lead_s * ctx.fps))
        for tag in self._tags(ctx):           # Sorted by start
            if tag.start_frame < earliest_start:
                continue
            if tag.start_frame > limit:
                break
            frame = self.timing.locate(ctx, tag, since, limit)
            if frame is not None:
                return Event(frame, self.name, tag.type, TEAM_OF_CLUB.get(tag.team.lower()))
        return None


class KickRestart(RestartDetector):
    """Restart at the first untagged kick at least min_delay_s after the stop."""
    name = "kick"

    def __init__(self, kick=None, min_delay_s=3.0):
        self.kick = kick or KickFinder(still_window=15)
        self.min_delay_s = min_delay_s

    def next_restart(self, ctx, since, limit):
        frame = self.kick.first_in(ctx, since + int(round(self.min_delay_s * ctx.fps)), limit)
        return Event(frame, self.name) if frame is not None else None


class ModelSignal:
    """Return DFL-model dead phases from ctx.p_dead.

Smooth P(dead) over smooth_s, then threshold it. Within each half, absorb
alive or dead phases shorter than min_s into neighbours. Shared by
ModelStop and ModelRestart."""

    def __init__(self, smooth_s=1.0, threshold=0.5, min_s=2.0):
        self.smooth_s = smooth_s
        self.threshold = threshold
        self.min_s = min_s
        self._cache = {}

    def dead(self, ctx):
        key = id(ctx)
        if key in self._cache:
            return self._cache[key]
        from dead_model import roll_nanmean
        out = np.zeros(ctx.n_frames, dtype=bool)
        k = int(round(self.min_s * ctx.fps))
        for _, first, last in ctx.half_blocks:
            p = roll_nanmean(ctx.p_dead[first:last + 1], max(1, int(round(self.smooth_s * ctx.fps))))
            dead = p > self.threshold
            for value in (True, False):
                for s, e in _runs(dead == value):
                    if e - s < k and s > 0 and e < len(dead):
                        dead[s:e] = not value
            out[first:last + 1] = dead
        self._cache[key] = out
        return out


class ModelStop(StopDetector):
    """STOP: the model predicts dead for at least min_s seconds.

ball_lost_only keeps stops where the ball is unseen for at least half
of the surrounding ±2 s, where ball-based rules cannot observe it."""
    name = "model"

    def __init__(self, signal=None, min_s=3.0, ball_lost_only=False):
        self.signal = signal or ModelSignal()
        self.min_s = min_s
        self.ball_lost_only = ball_lost_only

    def detect(self, ctx):
        dead = self.signal.dead(ctx)
        k = int(round(self.min_s * ctx.fps))
        w = 2 * ctx.fps
        events = []
        for s, e in _runs(dead):
            if e - s < k:
                continue
            lost = 1 - ctx.ball_visible[max(0, s - w):s + w].mean()
            if self.ball_lost_only and lost < 0.5:
                continue
            events.append(Event(int(s), self.name, "ball missing {:.0%}".format(lost)))
        return events


class ModelRestart(RestartDetector):
    """RESTART: the model dead phase following the stop ends.

First require the model to confirm dead within confirm_s; otherwise defer
to other sources. Snap the phase end to an observed kick in the surrounding
[-kick_before_s, +kick_after_s] interval when available."""
    name = "model"

    def __init__(self, signal=None, confirm_s=10.0, min_delay_s=1.0, kick=None,
                 kick_before_s=2.0, kick_after_s=1.0):
        self.signal = signal or ModelSignal()
        self.confirm_s = confirm_s
        self.min_delay_s = min_delay_s
        self.kick = kick if kick is not None else KickFinder(still_window=15)
        self.kick_before_s = kick_before_s
        self.kick_after_s = kick_after_s

    def next_restart(self, ctx, since, limit):
        dead = self.signal.dead(ctx)
        lo = max(since, 0)
        confirm = np.flatnonzero(dead[lo:min(limit, since + int(round(self.confirm_s * ctx.fps))) + 1])
        if not len(confirm):
            return None
        start = lo + int(confirm[0])
        alive = np.flatnonzero(~dead[start:limit + 1])
        if not len(alive):
            return None
        frame = start + int(alive[0])
        detail = 'model D end'
        if self.kick:
            kick = self.kick.first_in(ctx, max(since + 1, frame - int(round(self.kick_before_s * ctx.fps))),
                                      min(limit, frame + int(round(self.kick_after_s * ctx.fps))))
            if kick is not None:
                frame, detail = kick, 'model D end + kick'
        frame = max(frame, since + int(round(self.min_delay_s * ctx.fps)))
        return Event(frame, self.name, detail) if frame <= limit else None


# --------------------------------------------------------------------------------------------
# Model confirmation: a weak whistle, uncertain line crossing or isolated kick
# requires supporting collective player behavior.
# --------------------------------------------------------------------------------------------

def strong_whistle(ctx, event, min_duration_s=0.3, min_level_db=20.0):
    """A sufficiently long and strong whistle needs no model confirmation.
Historical confirmation rates were 90–95%, versus 60–67% for weak/short whistles."""
    return (event.meta.get("duration_s", 0) >= min_duration_s
            and event.meta.get("level_db", 0) >= min_level_db)


OUT_RESTART_TYPES = ("throw-in", "corner kick", "goal kick")


def reliable_out(ctx, event, max_dist_m=3.0, tag_before_s=5.0, tag_after_s=30.0):
    """Trust an observed line crossing or a nearby throw-in/corner/goal-kick tag.

Crossing requires an observation inside the pitch within the previous
second and no farther than max_dist_m from the line. A ball found far
outside without a tag may be a ball-boy observation and needs model agreement."""
    if event.meta.get("continuous", False) and event.meta.get("dist_m", 99.0) <= max_dist_m:
        return True
    lo = event.frame - int(round(tag_before_s * ctx.fps))
    hi = event.frame + int(round(tag_after_s * ctx.fps))
    return any(lo <= t.start_frame <= hi for t in ctx.set_pieces
               if t.type.lower() in OUT_RESTART_TYPES)


class ModelConfirmedStop(StopDetector):
    """Filter `inner` stops through model confirmation unless trust(ctx, event).

Confirm dead within confirm_s: keep the cue time. Confirm between
confirm_s and late_s: date the stop at model transition. Otherwise ignore it."""

    def __init__(self, inner, signal=None, trust=None, confirm_s=5.0, late_s=15.0):
        self.inner = inner
        self.signal = signal or ModelSignal()
        self.trust = trust or (lambda ctx, event: False)
        self.confirm_s = confirm_s
        self.late_s = late_s
        self.name = inner.name
        self.rejected = []  # Diagnostic information

    def detect(self, ctx):
        dead = self.signal.dead(ctx)
        k_confirm = int(round(self.confirm_s * ctx.fps))
        k_late = int(round(self.late_s * ctx.fps))
        kept, self.rejected = [], []
        for event in self.inner.detect(ctx):
            if self.trust(ctx, event):
                kept.append(event)
                continue
            onset = np.flatnonzero(dead[event.frame:event.frame + k_late + 1])
            if not len(onset):
                self.rejected.append(event)
            elif onset[0] <= k_confirm:
                kept.append(event)
            else:
                kept.append(Event(event.frame + int(onset[0]), event.source,
                                  event.detail + ' -> model +{:.1f}s'.format(onset[0] / ctx.fps),
                                  event.team, event.meta))
        return sorted(kept, key=lambda e: e.frame)


class ModelConfirmedRestart(RestartDetector):
    """Accept an `inner` restart only if the model becomes alive within confirm_s.

Otherwise try the next candidate. If the model never saw a brief stop and
is already alive, no additional delay is imposed."""

    def __init__(self, inner, signal=None, confirm_s=3.0, max_tries=50):
        self.inner = inner
        self.signal = signal or ModelSignal()
        self.confirm_s = confirm_s
        self.max_tries = max_tries
        self.name = inner.name

    def next_restart(self, ctx, since, limit):
        dead = self.signal.dead(ctx)
        k = int(round(self.confirm_s * ctx.fps))
        for _ in range(self.max_tries):
            event = self.inner.next_restart(ctx, since, limit)
            if event is None or not dead[event.frame:min(limit, event.frame + k) + 1].all():
                return event
            since = max(event.frame, since + 1)
        return None


class WhistleRestart(RestartDetector):
    """Restart whistle followed by a kick.

Search whistles at least min_delay_s after the stop, avoiding repeated stop
whistles. Use the first in-pitch kick above 4 m/s within kick_window_s.
With require_kick=True, ignore whistles without a visible subsequent kick;
otherwise use whistle end if no kick is observed."""
    name = "whistle"

    def __init__(self, finder=None, kick=None, min_delay_s=3.0, kick_window_s=5.0,
                 require_kick=True):
        self.finder = finder or WhistleFinder()
        self.kick = kick or KickFinder()
        self.min_delay_s = min_delay_s
        self.kick_window_s = kick_window_s
        self.require_kick = require_kick

    def next_restart(self, ctx, since, limit):
        earliest = since + int(round(self.min_delay_s * ctx.fps))
        window = int(round(self.kick_window_s * ctx.fps))
        for s, e, level in self.finder.whistles(ctx):
            if s < earliest:
                continue
            if s > limit:
                break
            detail = "{:.1f}s {:.0f}dB".format((e - s) / ctx.fps, level)
            kick = self.kick.first_in(ctx, s, min(limit, s + window))
            if kick is not None:
                return Event(kick, self.name, detail + " + frappe")
            if not self.require_kick and e <= limit:
                return Event(e, self.name, detail)
        return None


class TimeoutRestart(RestartDetector):
    """Force a restart after `seconds` seconds of D."""
    name = "timeout"

    def __init__(self, seconds=60.0):
        self.seconds = seconds

    def next_restart(self, ctx, since, limit):
        frame = since + int(round(self.seconds * ctx.fps))
        return Event(frame, self.name) if frame <= limit else None


class Earliest(RestartDetector):
    """Return the earliest restart from multiple detectors."""
    name = "earliest"

    def __init__(self, *detectors):
        self.detectors = detectors

    def next_restart(self, ctx, since, limit):
        found = [d.next_restart(ctx, since, limit) for d in self.detectors]
        found = [e for e in found if e is not None]
        return min(found, key=lambda e: e.frame) if found else None


class Fallback(RestartDetector):
    """Use `primary` if it restarts within horizon_s of the stop; otherwise
choose the earliest primary/secondary proposal in the remaining half."""
    name = "fallback"

    def __init__(self, primary, secondary, horizon_s=45.0):
        self.primary = primary
        self.secondary = secondary
        self.horizon_s = horizon_s

    def next_restart(self, ctx, since, limit):
        horizon = min(limit, since + int(round(self.horizon_s * ctx.fps)))
        event = self.primary.next_restart(ctx, since, horizon)
        if event is not None:
            return event
        return Earliest(self.primary, self.secondary).next_restart(ctx, since, limit)


# --------------------------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------------------------

class DeadBallDetector:
    """Combine stop and restart detectors into a per-frame D mask.

stops supplies StopDetector objects; restart is the subsequent restart
source. half_start detects kickoff when a half starts dead; None means
start alive. refractory_s ignores stops too soon after a restart, such
as a ball still held outside immediately after a throw-in."""

    def __init__(self, stops: Sequence[StopDetector], restart: RestartDetector,
                 half_start: Optional[RestartDetector] = None, refractory_s=1.0):
        self.stops = list(stops)
        self.restart = restart
        self.half_start = half_start
        self.refractory_s = refractory_s

    def run(self, ctx) -> DeadBallResult:
        dead = np.ones(ctx.n_frames, dtype=bool)   # Outside halves: D, which is ungraded anyway.
        segments = []

        stop_events = sorted((e for d in self.stops for e in d.detect(ctx)),
                             key=lambda e: e.frame)
        stop_frames = np.array([e.frame for e in stop_events], dtype=np.int64)
        refractory = int(round(self.refractory_s * ctx.fps))

        for half, first, last in ctx.half_blocks:
            dead[first:last + 1] = False
            limit = last + 1  # A restart at last+1 means no restart in this half.

            # Start the half dead until kickoff.
            alive_from = first
            if self.half_start is not None:
                kickoff = self.half_start.next_restart(ctx, first - 1, last)
                if kickoff is not None and kickoff.frame > first:
                    dead[first:kickoff.frame] = True
                    segments.append(DeadSegment(first, kickoff.frame,
                                                Event(first, "half_start", "half {}".format(half)),
                                                kickoff))
                    alive_from = kickoff.frame

            # Alternate stop and restart events.
            while True:
                k = np.searchsorted(stop_frames, alive_from + refractory)
                if k >= len(stop_frames) or stop_frames[k] > last:
                    break
                stop = stop_events[k]
                restart = self.restart.next_restart(ctx, stop.frame, last)
                end = restart.frame if restart is not None else limit
                dead[stop.frame:end] = True
                segments.append(DeadSegment(stop.frame, end, stop, restart))
                if restart is None:
                    break
                alive_from = end

        return DeadBallResult(dead, segments)
