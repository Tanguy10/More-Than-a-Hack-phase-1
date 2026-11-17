"""Exploratory pipeline: dead ball first, then H/A possession on remaining frames.

D overrides possession_model(frame). Run this module to write and evaluate
the default preset, --dead all --no-write to inspect presets, or --segments
to list D phases. Public-label checks must not be used for model selection.
Add a preset in DEAD_BALL_PRESETS/POSSESSION_PRESETS or pass objects to
build_labels. The retained submission is the separate modele-bnz package."""

import argparse
import os

import numpy as np

import data_io
from context import MatchContext
from dead_ball import (DeadBallDetector, OutOfBoundsStop, NoRefereeStop, WhistleFinder,
                       WhistleStop, WhistleRestart, SmartTaggingRestart, KickRestart, Fallback,
                       Earliest, TagStart, FirstKick, KickFinder, KICKOFF_START_TYPES,
                       ModelSignal, ModelStop, ModelRestart, ModelConfirmedStop,
                       ModelConfirmedRestart, strong_whistle, reliable_out)
from possession import NearestPlayerPossession, ConstantPossession
import evaluate

RESULTS_DIR = os.path.join(data_io.ROOT, "evaluation", "results")


# --------------------------------------------------------------------------------------------
# Dead-ball presets
# --------------------------------------------------------------------------------------------

def dead_literal():
    """Literal rule: out of bounds means D until the next tagged set-piece start."""
    return DeadBallDetector(
        stops=[OutOfBoundsStop(), NoRefereeStop()],
        restart=SmartTaggingRestart(timing=TagStart()),
        half_start=SmartTaggingRestart(KICKOFF_START_TYPES, timing=TagStart(), max_lead_s=10.0),
    )


def dead_tag_kick():
    """As literal, but restart at the first kick inside the tag interval."""
    return DeadBallDetector(
        stops=[OutOfBoundsStop(), NoRefereeStop()],
        restart=SmartTaggingRestart(timing=FirstKick(fallback_to_start=True)),
        half_start=SmartTaggingRestart(KICKOFF_START_TYPES, timing=FirstKick(), max_lead_s=10.0),
    )


def _tagged_restart():
    return SmartTaggingRestart(timing=FirstKick(fallback_to_start=True))


def _kick_fallback():
    """Untagged fallback: a kick after a still/unseen ball, at least 3 s after stop."""
    return KickRestart(KickFinder(still_window=15), min_delay_s=3.0)


def _kickoff():
    # Do not use kickoff whistles here: the second graded half starts during
    # warmup with whistles and a tracked warmup ball, before actual kickoff.
    return SmartTaggingRestart(KICKOFF_START_TYPES, timing=FirstKick(), max_lead_s=10.0)


def dead_no_whistle():
    """tag_kick plus fallback: untagged kick if no tagged restart arrives within 45 s."""
    return DeadBallDetector(
        stops=[OutOfBoundsStop(), NoRefereeStop()],
        restart=Fallback(primary=_tagged_restart(), secondary=_kick_fallback(), horizon_s=45.0),
        half_start=_kickoff(),
    )


def dead_whistle_stop():
    """no_whistle plus audio-whistle stops with the ball inside the pitch."""
    return DeadBallDetector(
        stops=[OutOfBoundsStop(), WhistleStop()],
        restart=Fallback(primary=_tagged_restart(), secondary=_kick_fallback(), horizon_s=45.0),
        half_start=_kickoff(),
    )


def dead_default():
    """Rules-only preset: out-of-bounds/whistle stops and earliest tagged or
whistle-plus-kick restart, with tracking-kick fallback after 45 s."""
    whistles = WhistleFinder()  # Shared: detect only once.
    return DeadBallDetector(
        stops=[OutOfBoundsStop(), WhistleStop(whistles)],
        restart=Fallback(
            primary=Earliest(_tagged_restart(), WhistleRestart(whistles)),
            secondary=_kick_fallback(),
            horizon_s=45.0,
        ),
        half_start=_kickoff(),
    )


def dead_hybrid(confirm_restart_s=3.0, confirm_stop_s=5.0):
    """Default exploratory preset: rules plus the player-only DFL model.

Stops combine line crossings, whistles and model transitions when the ball
is lost. Restarts take the earliest tag, whistle-plus-kick or model phase
end snapped to a kick, then a kick-only fallback. Weak whistles or uncertain
line crossings require model dead confirmation within 15 s; confirmation
after 5 s sets the stop time. Restarts require alive confirmation within
3 s, otherwise try the next candidate."""
    whistles, signal = WhistleFinder(), ModelSignal()
    return DeadBallDetector(
        stops=[ModelConfirmedStop(OutOfBoundsStop(), signal, reliable_out, confirm_stop_s),
               ModelConfirmedStop(WhistleStop(whistles), signal, strong_whistle, confirm_stop_s),
               ModelStop(signal, ball_lost_only=True)],
        restart=ModelConfirmedRestart(
            Fallback(primary=Earliest(_tagged_restart(), WhistleRestart(whistles),
                                      ModelRestart(signal)),
                     secondary=_kick_fallback(), horizon_s=45.0),
            signal, confirm_restart_s),
        half_start=_kickoff(),
    )


def dead_never():
    """Never predict D inside halves: baseline for isolating detector effects."""
    return DeadBallDetector(stops=[], restart=SmartTaggingRestart())


DEAD_BALL_PRESETS = {
    "hybrid": dead_hybrid,          # Default exploratory preset
    # Rules-only reference presets without the model
    "rules": dead_default,
    "whistle_stop": dead_whistle_stop,
    "no_whistle": dead_no_whistle,
    "tag_kick": dead_tag_kick,
    "literal": dead_literal,
    "never": dead_never,
}

POSSESSION_PRESETS = {
    "nearest": lambda window: NearestPlayerPossession(window=window),
    "constant": lambda window: ConstantPossession("H"),
}


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------

def build_labels(ctx, dead_detector, possession_model):
    """Return (labels, dead_result), with D overriding possession."""
    dead = dead_detector.run(ctx)
    possession = possession_model.predict(ctx, dead)
    labels = np.where(dead.dead, "D", possession).astype("<U1")
    return labels, dead


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", default="hybrid",
                        choices=sorted(DEAD_BALL_PRESETS) + ["all"])
    parser.add_argument("--possession", default="nearest", choices=sorted(POSSESSION_PRESETS))
    parser.add_argument("--window", type=int, default=50, help="vote majoritaire, +/- frames")
    parser.add_argument("--name", default=None, help='CSV name (default: modele1_<preset>)')
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--segments", action="store_true", help="List D phases")
    args = parser.parse_args()

    ctx = MatchContext()
    gt = data_io.load_public_gt()
    presets = sorted(DEAD_BALL_PRESETS) if args.dead == "all" else [args.dead]

    for preset in presets:
        labels, dead = build_labels(ctx, DEAD_BALL_PRESETS[preset](),
                                    POSSESSION_PRESETS[args.possession](args.window))
        print("\n" + "=" * 90)
        print("dead-ball preset: {}   |   possession: {}".format(preset, args.possession))
        print("=" * 90)
        evaluate.report(ctx, labels, dead, gt, show_segments=args.segments)

        if not args.no_write:
            name = args.name or "modele1_{}".format(preset)
            path = data_io.write_submission(os.path.join(args.out, name + ".csv"), labels)
            print("\n-> {}".format(os.path.relpath(path, data_io.ROOT)))


if __name__ == "__main__":
    main()
