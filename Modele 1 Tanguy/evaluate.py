"""Official public-window scores and dead-ball diagnostics for the exploratory pipeline.

Five minutes contain only four D phases. Full-match D share, durations and
restart-source summaries are checks without private ground truth, not
full-match quality measurements. Do not tune on the public labels."""

import os
import sys
from collections import Counter

import numpy as np

import data_io

sys.path.insert(0, os.path.join(data_io.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


def runs(mask, offset=0):
    """Return [start, end) intervals of True values in a Boolean mask."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip((np.flatnonzero(edges == 1) + offset).tolist(),
                    (np.flatnonzero(edges == -1) + offset).tolist()))


def report(ctx, labels, dead, gt, show_segments=False):
    # -- Official scores on the public window --
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    d = result["per_class"]["D"]
    print("public 5 min   spellF1 {:.4f}   macroF1 {:.4f}   alive/dead F1 {:.4f}".format(
        result["spell_f1_mean"], result["macro_f1"], result["alive_dead_f1"]))
    print('               D: precision {:.3f}  recall {:.3f}  F1 {:.3f}   '
          '(predicted possessions {} / true {})'.format(
              d["precision"], d["recall"], d["f1"], result["n_pred_spells"],
              result["n_gt_spells"]))

    # -- Ground-truth versus predicted D phases --
    graded_gt = gt != "X"
    first, last = np.flatnonzero(graded_gt)[[0, -1]]
    gt_dead = runs((gt == "D")[first:last + 1], first)
    pred_dead = runs(dead.dead[first:last + 1], first)
    print('\n  ground-truth D          overlapping predicted D')
    for s, e in gt_dead:
        hits = [(ps, pe) for ps, pe in pred_dead if ps < e and pe > s]
        txt = ", ".join('{}-{} (start {:+d}, end {:+d})'.format(ps, pe, ps - s, pe - e)
                        for ps, pe in hits) or 'MISSED'
        print("  {:>6}-{:<6} {:>5.1f}s   {}".format(s, e, (e - s) / 25, txt))
    extra = [(ps, pe) for ps, pe in pred_dead
             if not any(ps < e and pe > s for s, e in gt_dead)]
    for ps, pe in extra:
        print("  {:>21}   {}-{} FAUX D ({:.1f}s)".format("", ps, pe, (pe - ps) / 25))

    # -- Full-match diagnostic checks --
    graded = ctx.graded
    share = dead.dead[graded].mean()
    in_half = [sg for sg in dead.segments]
    durations = np.array([sg.duration_s for sg in in_half]) if in_half else np.zeros(1)
    print('\nfull match  D on {:.1%} of graded frames   {} D phases   '
          'median duration {:.1f}s   max {:.1f}s'.format(
              share, len(in_half), np.median(durations), durations.max()))
    stops = Counter(sg.stop.source for sg in in_half)
    # Set-piece type is useful for tags; other events need only the source.
    restarts = Counter((sg.restart.source + (":" + sg.restart.detail
                                             if sg.restart.source == "smart_tagging" else ""))
                       if sg.restart else "half end" for sg in in_half)
    print('               stops    : ' + ", ".join("{} {}".format(k, v) for k, v in stops.most_common()))
    print("               reprises : " + ", ".join("{} {}".format(k, v) for k, v in restarts.most_common()))
    long_ones = [sg for sg in in_half if sg.duration_s > 90]
    if long_ones:
        print("               ATTENTION {} phase(s) D > 90 s : {}".format(
            len(long_ones), ", ".join("{}-{} ({:.0f}s)".format(sg.start, sg.end, sg.duration_s)
                                      for sg in long_ones[:6])))

    if show_segments:
        print("\n  all D phases:")
        for sg in in_half:
            print('  {:>6}-{:<6} {:>6.1f}s   stop {:<14} {:<16} restart {}'.format(
                sg.start, sg.end, sg.duration_s, sg.stop.source, sg.stop.detail,
                "{} {} {}".format(sg.restart.source, sg.restart.detail, sg.restart.team or "")
                if sg.restart else "-"))
    return result
