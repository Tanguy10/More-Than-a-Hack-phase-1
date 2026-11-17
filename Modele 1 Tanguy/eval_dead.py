"""Evaluate dead-ball detection alone, without H/A possession classification.

python "Modele 1 Tanguy/eval_dead.py" compares exploratory presets.
--dead selects a preset; --review writes predicted phases for inspection.

Frame metrics use the official alive/dead F1 plus D precision/recall/F1.
Phase diagnostics greedily match D intervals by IoU 0.3..0.7 and report
stop/restart precision/recall within ±1 s and phase-edge errors. Full-match
checks without ground truth describe D share, phase lengths and tagged
set-piece coverage within ±5 s. These are diagnostics, not target truth.

The public window has only four D phases. Never train, tune or select
models on it, and never create additional human labels on the provided game.
The legacy --annotations loader is not permitted for challenge-game labels;
external-game studies must use separate inputs and follow the official rules.
Its generic CSV schema is kind,start_s,end_s, with window rows denoting
fully annotated ranges and D rows denoting dead intervals inside them."""

import argparse
import csv
import os
from collections import Counter

import numpy as np

import data_io
from context import MatchContext
from dead_ball import RESTART_TYPES
from evaluate import runs
import pipeline

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
TOL = 25            # Transition tolerance in frames: one second.
COVER_S = 5.0       # Tag coverage requires D within ±COVER_S seconds of its start.
LONG_S = 90.0       # Long D phases deserve inspection, allowing goals, injuries and half endings.
NOT_ANNOTATED, ALIVE, DEAD = -1, 0, 1


# --------------------------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------------------------

def load_truth(annotations=None):
    """Return int8 per frame: -1 unlabelled, 0 alive, 1 dead."""
    gt = data_io.load_public_gt()
    truth = np.full(len(gt), NOT_ANNOTATED, dtype=np.int8)
    truth[(gt == "H") | (gt == "A")] = ALIVE
    truth[gt == "D"] = DEAD
    if annotations:
        with open(annotations) as handle:
            rows = list(csv.DictReader(handle))
        to_frame = lambda s: int(round(float(s) * data_io.FPS))  # noqa: E731
        for row in rows:                       # Apply annotated windows first, defaulting to alive.
            if row["kind"].strip().lower() == "window":
                truth[to_frame(row["start_s"]):to_frame(row["end_s"])] = ALIVE
        for row in rows:                       # Then overlay D intervals.
            if row["kind"].strip().upper() == "D":
                truth[to_frame(row["start_s"]):to_frame(row["end_s"])] = DEAD
    return truth


# --------------------------------------------------------------------------------------------
# 1. Frames
# --------------------------------------------------------------------------------------------

def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def frame_scores(truth, dead):
    m = truth != NOT_ANNOTATED
    t, d = truth[m] == DEAD, dead[m]
    _, _, alive_f1 = _f1(np.sum(~t & ~d), np.sum(t & ~d), np.sum(~t & d))
    dp, dr, df1 = _f1(np.sum(t & d), np.sum(~t & d), np.sum(t & ~d))
    return {"alive_dead_f1": alive_f1, "D_precision": dp, "D_recall": dr, "D_f1": df1,
            "n_frames": int(m.sum())}


# --------------------------------------------------------------------------------------------
# 2. D phases and transitions
# --------------------------------------------------------------------------------------------

def _phases(truth, dead):
    """Return true phases, predicted phases and annotated windows, clipped to those windows."""
    windows = runs(truth != NOT_ANNOTATED)
    gt, pred = [], []
    for ws, we in windows:
        gt += runs(truth[ws:we] == DEAD, ws)
        pred += runs(dead[ws:we], ws)
    return gt, pred, windows


def _iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, inter) / (max(a[1], b[1]) - min(a[0], b[0]))


def phase_f1(gt, pred, ious=IOUS):
    """Mean F1 over IoU thresholds using greedy one-to-one matching, as for Spell F1."""
    pairs = sorted(((_iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if _iou(g, p) > 0), reverse=True)
    f1s = []
    for thr in ious:
        used_g, used_p = set(), set()
        for iou, i, j in pairs:
            if iou >= thr and i not in used_g and j not in used_p:
                used_g.add(i)
                used_p.add(j)
        f1s.append(_f1(len(used_g), len(pred) - len(used_p), len(gt) - len(used_g))[2])
    return float(np.mean(f1s)) if f1s else 0.0


def _match(gt_frames, pred_frames, tol):
    """Chronological one-to-one matching within ±tol frames; return (tp, fp, fn)."""
    i = j = tp = 0
    while i < len(gt_frames) and j < len(pred_frames):
        delta = pred_frames[j] - gt_frames[i]
        if abs(delta) <= tol:
            tp += 1
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return tp, len(pred_frames) - tp, len(gt_frames) - tp


def transition_scores(gt, pred, windows, tol=TOL):
    """Stop/restart precision and recall within ±tol frames.
A phase edge coinciding with an annotation-window edge is not a transition."""
    starts = {ws for ws, _ in windows}
    ends = {we for _, we in windows}
    out = {}
    for name, k, edge in (("stop", 0, starts), ("restart", 1, ends)):
        g = sorted(p[k] for p in gt if p[k] not in edge)
        q = sorted(p[k] for p in pred if p[k] not in edge)
        p_, r_, _ = _f1(*_match(g, q, tol))
        out[name + "_P"], out[name + "_R"] = p_, r_
    return out


# --------------------------------------------------------------------------------------------
# 3. Full match
# --------------------------------------------------------------------------------------------

def full_match_checks(ctx, result):
    graded = ctx.graded
    durations = np.array([s.duration_s for s in result.segments
                          if s.stop.source != "half_start"])
    tags = [t for t in ctx.set_pieces if t.type.lower() in RESTART_TYPES
            and t.type.lower() != "kick off (start)"]
    pad = int(round(COVER_S * ctx.fps))
    covered = [bool(result.dead[max(0, t.start_frame - pad):t.start_frame + pad].any())
               for t in tags]
    missed = Counter(t.type for t, c in zip(tags, covered) if not c)
    return {
        "D_share": float(result.dead[graded].mean()),
        "n_phases": int(len(durations)),
        "median_s": float(np.median(durations)) if len(durations) else 0.0,
        "n_long": int((durations > LONG_S).sum()),
        "tag_cover": float(np.mean(covered)) if covered else 0.0,
        "n_tags": len(tags),
        "tags_missed": missed,
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------

def evaluate_dead(ctx, result, truth):
    gt, pred, windows = _phases(truth, result.dead)
    scores = frame_scores(truth, result.dead)
    scores["phase_f1"] = phase_f1(gt, pred)
    scores.update(transition_scores(gt, pred, windows))
    scores.update(full_match_checks(ctx, result))
    scores["n_gt_phases"] = len(gt)
    return scores, gt, pred


COLUMNS = [  # Title, key, format
    ("alive/dead F1", "alive_dead_f1", "{:.3f}"), ('D prec', "D_precision", "{:.3f}"),
    ("D rappel", "D_recall", "{:.3f}"), ("phaseF1", "phase_f1", "{:.3f}"),
    ('stops P/R', ("stop_P", "stop_R"), "{:.2f}/{:.2f}"),
    ("reprises P/R", ("restart_P", "restart_R"), "{:.2f}/{:.2f}"),
    ("| D match", "D_share", "{:.1%}"), ("phases", "n_phases", "{}"),
    ("> 90 s", "n_long", "{}"), ("tags couverts", "tag_cover", "{:.0%}"),
]


def print_table(rows):
    def cell(scores, key, fmt):
        return fmt.format(*(scores[k] for k in key)) if isinstance(key, tuple) \
            else fmt.format(scores[key])
    header = ["preset"] + [c[0] for c in COLUMNS]
    body = [[name] + [cell(s, k, f) for _, k, f in COLUMNS] for name, s in rows]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    for r in [header] + body:
        print("  ".join(v.rjust(w) for v, w in zip(r, widths)))


def print_detail(scores, gt, pred):
    print("\nphases D vraies ({}) :".format(len(gt)))
    for s, e in gt:
        hits = [(ps, pe) for ps, pe in pred if ps < e and pe > s]
        txt = ", ".join('{}-{}  start {:+.1f}s  end {:+.1f}s'.format(
            ps, pe, (ps - s) / 25, (pe - e) / 25) for ps, pe in hits) or 'MISSED'
        print("  {:>6}-{:<6} ({:>7}) {:>5.1f}s   {}".format(s, e, _mmss(s), (e - s) / 25, txt))
    extra = [(ps, pe) for ps, pe in pred if not any(ps < e and pe > s for s, e in gt)]
    for ps, pe in extra:
        print("  {:>30}   {}-{} FAUX D ({:.1f}s)".format("", ps, pe, (pe - ps) / 25))
    if scores["tags_missed"]:
        print('\ntagged set pieces without nearby D ({}/{}): {}'.format(
            sum(scores["tags_missed"].values()), scores["n_tags"],
            ", ".join("{} {}".format(k, v) for k, v in scores["tags_missed"].most_common())))


def _mmss(frame):
    s = frame / 25
    return "{:d}:{:04.1f}".format(int(s // 60), s % 60)


def write_review(path, result):
    """Predicted D phases for tactical-video inspection, with mm:ss times."""
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["start_frame", "end_frame", "start_video", "end_video", "duration_s",
                    "stop", "stop_detail", "restart", "restart_detail", 'ok (fill in)'])
        for s in result.segments:
            w.writerow([s.start, s.end, _mmss(s.start), _mmss(s.end), round(s.duration_s, 1),
                        s.stop.source, s.stop.detail,
                        s.restart.source if s.restart else "", s.restart.detail if s.restart else "",
                        ""])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", default="all",
                        choices=sorted(pipeline.DEAD_BALL_PRESETS) + ["all"])
    parser.add_argument("--annotations", default=None, help="CSV kind,start_s,end_s (voir doc)")
    parser.add_argument("--review", default=None, help='Write predicted D phases to this CSV')
    args = parser.parse_args()

    ctx = MatchContext()
    truth = load_truth(args.annotations)
    presets = sorted(pipeline.DEAD_BALL_PRESETS) if args.dead == "all" else [args.dead]

    rows, last = [], None
    for name in presets:
        result = pipeline.DEAD_BALL_PRESETS[name]().run(ctx)
        scores, gt, pred = evaluate_dead(ctx, result, truth)
        rows.append((name, scores))
        last = (scores, gt, pred, result)

    print('ground truth: {} labelled frames, {} D phases\n'.format(
        rows[0][1]["n_frames"], rows[0][1]["n_gt_phases"]))
    print_table(rows)
    if len(presets) == 1:
        print_detail(*last[:3])
    if args.review:
        print("\n-> " + write_review(args.review, last[3]))


if __name__ == "__main__":
    main()
