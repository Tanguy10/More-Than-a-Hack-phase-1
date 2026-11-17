"""Plot ball-tracking presence throughout the match.

One panel per half, in match minutes. Second-half minute 45 is actual
kickoff; earlier graded frames are warmup. A strip shows per-second
visibility, the curve a one-minute moving percentage and grey shading
predicted D. Writes a local PNG outside the repository; do not publish
match-derived visualizations with private tracking."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import data_io  # noqa: E402
import pipeline  # noqa: E402
from context import MatchContext  # noqa: E402

OUT = os.path.join(os.path.dirname(data_io.CACHE_DIR), "visu", "presence_ballon.png")
FPS = data_io.FPS
WINDOW_S = 60

# Colours for a light-theme visualization.
SERIES = "#2a78d6"      # Tracked ball
SURFACE = "#fcfcfb"
DEAD = "#e9e8e3"        # Neutral D-phase background
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def match_minutes(frames, first, kickoff, offset_min):
    """Match minute: offset plus time since kickoff; negative beforehand."""
    return offset_min + (frames - kickoff) / FPS / 60.0


def main():
    ctx = MatchContext()
    dead = pipeline.dead_default().run(ctx)
    vis = ctx.ball_visible
    graded = ctx.graded

    # Actual kickoff is the end of the initial D phase in each half.
    kickoffs = {}
    for seg in dead.segments:
        if seg.stop.source == "half_start":
            kickoffs[ctx.half_of(seg.start)] = seg.end

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(16, 8.6), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, hspace=0.42, left=0.06, right=0.985, top=0.86, bottom=0.08)
    strip_cmap = LinearSegmentedColormap.from_list("presence", ["#eef0f2", SERIES])

    summary = []
    for row, (half, first, last) in enumerate(ctx.half_blocks):
        frames = np.arange(first, last + 1)
        v = vis[first:last + 1].astype(float)
        d = dead.dead[first:last + 1]
        kickoff = kickoffs.get(half, first)
        offset = 0.0 if half == 1 else 45.0
        x = match_minutes(frames, first, kickoff, offset)

        # Centred one-minute moving mean
        k = WINDOW_S * FPS
        roll = np.convolve(v, np.ones(k) / k, mode="same") * 100
        # Per-second presence
        n_sec = len(v) // FPS
        per_sec = v[:n_sec * FPS].reshape(n_sec, FPS).mean(axis=1)

        p_all, p_alive, p_dead = 100 * v.mean(), 100 * v[~d].mean(), 100 * v[d].mean()
        summary.append((half, p_all, p_alive, p_dead))

        gs = outer[row].subgridspec(2, 1, height_ratios=[1, 7], hspace=0.08)
        ax_strip = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_strip)

        # Presence strip
        ax_strip.imshow(per_sec[None, :], aspect="auto", cmap=strip_cmap, vmin=0, vmax=1,
                        extent=(x[0], x[0] + n_sec / 60.0, 0, 1), interpolation="nearest")
        ax_strip.set_yticks([])
        ax_strip.tick_params(axis="x", labelbottom=False, length=0)
        for s in ax_strip.spines.values():
            s.set_visible(False)
        ax_strip.set_title('half {}   ball tracked on {:.0f}% of frames   (predicted alive: {:.0f}%,'
                           ' predicted dead: {:.0f}%)'.format(
                               "1re" if half == 1 else "2e", p_all, p_alive, p_dead),
                           loc="left", fontsize=11, color=INK, pad=6, fontweight="bold")

        # D-phase background
        edges = np.diff(np.concatenate(([0], d.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            ax.axvspan(x[s], x[min(e, len(x) - 1)], color=DEAD, lw=0, zorder=0)

        ax.plot(x, roll, color=SERIES, lw=1.6, zorder=3)
        ax.set_ylim(0, 102)
        ax.set_xlim(x[0], x[-1])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for name, s in ax.spines.items():
            s.set_visible(name == "bottom")
            s.set_color(AXIS)
        ax.tick_params(length=0)
        ticks = np.arange(np.ceil(x[0] / 5) * 5, x[-1] + 0.01, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["{:d}'".format(int(round(t))) for t in ticks])
        ax.set_ylabel("ball tracked\n(1-minute mean)", fontsize=9)

        # Reference markers
        if half == 1:
            gt = data_io.load_public_gt()
            known = np.flatnonzero(gt != "X")
            g0 = match_minutes(known[0], first, kickoff, offset)
            g1 = match_minutes(known[-1], first, kickoff, offset)
            ax.annotate("", xy=(g0, 6), xytext=(g1, 6),
                        arrowprops=dict(arrowstyle="|-|", color=INK2, lw=1, shrinkA=0, shrinkB=0))
            ax.text((g0 + g1) / 2, 9, 'public ground truth (5 min)', ha="center", va="bottom",
                    fontsize=8.5, color=INK2)
        if half == 2 and kickoff > first:
            ax.axvline(offset, color=INK2, lw=1, zorder=4)
            ax.text(offset + 0.2, 3, 'kickoff (earlier graded frames: warmup)', ha="left",
                    va="bottom", fontsize=8.5, color=INK2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        if row == 1:
            ax.set_xlabel("minute de match", fontsize=9)

    total = 100 * vis[graded].mean()
    fig.text(0.06, 0.955, 'Ball-tracking presence throughout the match',
             fontsize=15, fontweight="bold", color=INK)
    fig.text(0.06, 0.918, 'Ball tracked on {:.0f}% of graded frames. The blue line shows the fraction'
             ' of frames with an observed ball; grey shading shows dead-ball phases'
             ' predicted by the model.'.format(total), fontsize=10, color=INK2)
    fig.legend(handles=[Line2D([], [], color=SERIES, lw=2, label="ball tracked (% of frames, 1 min)"),
                        Patch(color=DEAD, label='predicted dead ball (D)'),
                        Patch(color=SERIES, label='strip: per-second presence')],
               loc="upper right", bbox_to_anchor=(0.985, 0.975), ncol=3, frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print("-> {}".format(OUT))
    for half, p_all, p_alive, p_dead in summary:
        print('half {}: ball tracked {:.0f}% | predicted alive {:.0f}% | predicted D {:.0f}%'.format(
            half, p_all, p_alive, p_dead))


if __name__ == "__main__":
    main()
