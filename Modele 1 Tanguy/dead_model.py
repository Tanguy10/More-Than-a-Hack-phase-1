"""Player-only DFL model for per-frame P(dead ball).

DFL provides seven corrected 25 Hz Bundesliga matches with ball status.
Training uses other games, never the challenge labels. Ball inputs are
excluded because the DFL ball is frozen during stoppages, a shortcut absent
from automatic challenge tracking. Player behavior remains observable when
the ball is lost; ball/audio/tag cues stay in dead_ball.py rules.

About 170 columns describe player speeds, spatial spread, team distances,
field-region counts and nearest opponents. Context uses 2/10/30 s windows
and ±10 s offsets. Per-half 0–1 ranks reduce source speed-scale differences.
Training corruption removes 7% of positions in roughly five-second gaps,
adds 20 cm noise and recomputes speeds.

python dead_model.py trains/caches the seven-match model; --cv runs
leave-one-match-out DFL validation. ctx.p_dead exposes predictions to
ModelStop and ModelRestart. This exploratory model is not the submission."""

import argparse
import os
import sys
import warnings
import zlib

import numpy as np

import data_io

sys.path.insert(0, os.path.join(data_io.HERE, "external"))

FPS = 25
HALF_L, HALF_W = 52.5, 34.0
MODEL_PATH = os.path.join(data_io.CACHE_DIR, "dead_model_dfl.txt")
FEATURE_VERSION = 1  # Increment when features change to invalidate caches.
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=8, seed=0, deterministic=True, force_row_wise=True, verbose=-1)
N_ROUNDS = 300
TRAIN_STEP = 5  # Sample DFL at 5 Hz for training.


# --------------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------------

def roll_nanmean(a, w):
    """Centred w-frame moving mean, ignoring NaN."""
    a = np.asarray(a, dtype=np.float64)
    ok = ~np.isnan(a)
    cs = np.concatenate([[0], np.cumsum(np.where(ok, a, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    n = len(a)
    lo = np.clip(np.arange(n) - w // 2, 0, n)
    hi = np.clip(np.arange(n) - w // 2 + w, 0, n)
    c = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)


def _shift(a, k):
    """Return out[t] = a[t+k], with NaN at edges."""
    out = np.full(len(a), np.nan)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def frame_features(px, py, speed, team):
    """Compute per-frame features for one half.

px, py are (N, P) metres from pitch centre; speed is m/s.
NaN means absent player; team has (P,) codes 0/1."""
    vis = ~np.isnan(px)
    nv = np.maximum(vis.sum(1), 1)
    f = {"n_vis": vis.sum(1).astype(float)}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        f["sp_med"] = np.nanmedian(speed, 1)
        f["sp_mean"] = np.nanmean(speed, 1)
        f["sp_q90"] = np.nanpercentile(speed, 90, axis=1)
        f["frac_slow"] = ((speed < 1.0) & vis).sum(1) / nv
        f["frac_walk"] = ((speed < 2.0) & vis).sum(1) / nv
        f["frac_fast"] = ((speed > 4.0) & vis).sum(1) / nv
        f["frac_sprint"] = ((speed > 6.0) & vis).sum(1) / nv
        f["cx_abs"] = np.abs(np.nanmean(px, 1))
        f["cy_abs"] = np.abs(np.nanmean(py, 1))
        f["sx"] = np.nanstd(px, 1)
        f["sy"] = np.nanstd(py, 1)
        f["frac_touch"] = ((np.abs(py) > HALF_W - 6) & vis).sum(1) / nv
        f["frac_box"] = ((np.abs(px) > HALF_L - 16.5) & (np.abs(py) < 20.16) & vis).sum(1) / nv
        f["frac_out"] = (((np.abs(px) > HALF_L + 1) | (np.abs(py) > HALF_W + 1)) & vis).sum(1) / nv
        f["frac_center"] = ((np.hypot(px, py) < 9.15) & vis).sum(1) / nv
        centres, same_side, spread = [], [], []
        for t in (0, 1):
            m = team == t
            cx, cy = np.nanmean(px[:, m], 1), np.nanmean(py[:, m], 1)
            centres.append((cx, cy))
            # Team fraction on the same midfield side as its centre (1.0 at kickoff).
            same = (np.sign(px[:, m]) == np.sign(cx)[:, None]) & vis[:, m]
            same_side.append(same.sum(1) / np.maximum(vis[:, m].sum(1), 1))
            spread.append(np.nanstd(px[:, m], 1))
        f["same_side_min"] = np.minimum(*same_side)
        f["sx_team_mean"] = (spread[0] + spread[1]) / 2
        f["d_cent"] = np.hypot(centres[0][0] - centres[1][0], centres[0][1] - centres[1][1])
        h, a = team == 0, team == 1
        d = np.hypot(px[:, h, None] - px[:, None, a], py[:, h, None] - py[:, None, a])
        f["opp_near_med"] = np.nanmedian(np.concatenate([np.nanmin(d, 2), np.nanmin(d, 1)], 1), 1)
    return f


TEMPORAL = ("sp_med", "sp_mean", "frac_slow", "frac_fast", "frac_sprint", "sx", "sy", "d_cent",
            "opp_near_med", "frac_touch", "frac_box")
OFFSETS_S = (-10, -6, -3, -1, 0, 1, 3, 6, 10)


def half_features(px, py, speed, team):
    """Return (X, names) for one half: ranked frame features and past/future context."""
    base = frame_features(px, py, speed, team)
    out = dict(base)
    for k in TEMPORAL:
        r2 = roll_nanmean(base[k], 2 * FPS)
        for o in OFFSETS_S:
            out["%s_w2_%+d" % (k, o)] = _shift(r2, o * FPS)
        r10 = roll_nanmean(base[k], 10 * FPS)
        for o in (-10, 0, 10):
            out["%s_w10_%+d" % (k, o)] = _shift(r10, o * FPS)
        out["%s_w30" % k] = roll_nanmean(base[k], 30 * FPS)
        out["%s_fut_minus_past" % k] = out["%s_w2_+3" % k] - out["%s_w2_-3" % k]
    names = sorted(out)
    X = np.column_stack([pct_rank(out[n]) for n in names]).astype(np.float32)
    return X, names


def pct_rank(v):
    """Return per-half 0–1 ranks with average ties and preserved NaNs."""
    v = np.asarray(v, dtype=np.float64)
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return out
    _, inv, cnt = np.unique(v[ok], return_inverse=True, return_counts=True)
    out[ok] = ((np.cumsum(cnt) - cnt + 0.5 * cnt) / ok.sum())[inv]
    return out


# --------------------------------------------------------------------------------------------
# Training data: DFL corrupted to resemble automatic tracking.
# --------------------------------------------------------------------------------------------

def _speed_from_positions(px, py):
    out = np.empty_like(px)
    for j in range(px.shape[1]):
        xs, ys = roll_nanmean(px[:, j], 5), roll_nanmean(py[:, j], 5)
        out[:, j] = np.hypot(np.gradient(xs), np.gradient(ys)) * FPS
    return np.where(np.isnan(px), np.nan, out)


def _drop_players(n, p, rng, miss_rate=0.07, mean_len_s=5.0):
    """Visibility mask with roughly five-second gaps (ATD sees 20.5 of 22 players)."""
    vis = np.ones((n, p), bool)
    n_seg = int(miss_rate * n * p / (mean_len_s * FPS))
    starts, cols = rng.integers(0, n, n_seg), rng.integers(0, p, n_seg)
    lens = rng.exponential(mean_len_s * FPS, n_seg).astype(int) + 1
    for s, c, length in zip(starts, cols, lens):
        vis[s:s + length, c] = False
    return vis


def dfl_training_set(match_id):
    """Return cached (X, y_dead, period) for one DFL match at 5 Hz."""
    import dfl
    cache = os.path.join(dfl.cache_dir(), "dead_features_v{}_{}.npz".format(FEATURE_VERSION, match_id))
    if os.path.isfile(cache):
        d = np.load(cache)
        return d["X"], d["y"], d["half"], list(d["names"])
    m = dfl.load_match(match_id)
    rng = np.random.default_rng(zlib.crc32(match_id.encode()))
    keep = m["team"] < 2  # Exclude referees, which are absent from challenge tracking.
    Xs, ys, hs = [], [], []
    for half in (1, 2):
        sl = m["period"] == half
        P = m["players"][sl][:, keep]
        px, py = P[:, :, 0].astype(float), P[:, :, 1].astype(float)
        vis = _drop_players(len(px), px.shape[1], rng)
        px = np.where(vis, px + rng.normal(0, 0.2, px.shape), np.nan)
        py = np.where(vis, py + rng.normal(0, 0.2, py.shape), np.nan)
        X, names = half_features(px, py, _speed_from_positions(px, py), m["team"][keep])
        Xs.append(X[::TRAIN_STEP])
        ys.append((m["alive"][sl] == 0)[::TRAIN_STEP])
        hs.append(np.full(len(Xs[-1]), half))
    X, y, h = np.concatenate(Xs), np.concatenate(ys), np.concatenate(hs)
    np.savez(cache, X=X, y=y, half=h, names=np.array(names))
    return X, y, h, names


def train(match_ids=None, save=True):
    import lightgbm as lgb
    import dfl
    match_ids = match_ids or dfl.list_matches()
    if not match_ids:
        raise SystemExit("No DFL data: run 'Modele 1 Tanguy/external/download_external.sh'")
    parts = [dfl_training_set(mid) for mid in match_ids]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts]).astype(int)
    model = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=parts[0][3]), num_boost_round=N_ROUNDS)
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save_model(MODEL_PATH)
    return model


def load_model():
    import lightgbm as lgb
    if not os.path.isfile(MODEL_PATH):
        return train()
    return lgb.Booster(model_file=MODEL_PATH)


# --------------------------------------------------------------------------------------------
# Application to automatic tracking
# --------------------------------------------------------------------------------------------

def atd_features(ctx):
    """Cached features for each BAR-ATH half; NaN elsewhere."""
    cache = os.path.join(data_io.CACHE_DIR, "dead_features_v{}_tactical.npy".format(FEATURE_VERSION))
    if os.path.isfile(cache):
        return np.load(cache)
    P = ctx.players
    px = (P[:, :, 0].astype(float) - 0.5) * ctx.pitch[0]
    py = (P[:, :, 1].astype(float) - 0.5) * ctx.pitch[1]
    speed = P[:, :, 2].astype(float)  # Speed supplied by ATD
    X = None
    for _, first, last in ctx.half_blocks:
        sl = slice(first, last + 1)
        Xh, _ = half_features(px[sl], py[sl], speed[sl], ctx.track_team)
        if X is None:
            X = np.full((ctx.n_frames, Xh.shape[1]), np.nan, dtype=np.float32)
        X[sl] = Xh
    np.save(cache, X)
    return X


def predict_p_dead(ctx):
    """Return P(dead ball) per frame; NaN outside halves."""
    model = load_model()
    X = atd_features(ctx)
    p = np.full(ctx.n_frames, np.nan)
    for _, first, last in ctx.half_blocks:
        p[first:last + 1] = model.predict(X[first:last + 1])
    return p


# --------------------------------------------------------------------------------------------
# DFL cross-validation
# --------------------------------------------------------------------------------------------

def cross_validate():
    import dfl
    ids = dfl.list_matches()
    rows = []
    for test in ids:
        model = train([m for m in ids if m != test], save=False)
        X, y, half, _ = dfl_training_set(test)
        p = model.predict(X)
        pred = np.concatenate([roll_nanmean(p[half == h], 5) > 0.5 for h in (1, 2)])
        tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
        alive_tp = (~pred & ~y).sum()
        rows.append((test, (pred == y).mean(), 2 * alive_tp / (2 * alive_tp + fn + fp),
                     2 * tp / (2 * tp + fp + fn), y.mean(), pred.mean()))
        print('{} acc {:.3f} aliveF1 {:.3f} deadF1 {:.3f} | D true {:.1%} predicted {:.1%}'.format(*rows[-1]))
    print("moyenne  acc {:.3f} aliveF1 {:.3f} deadF1 {:.3f}".format(*np.mean([r[1:4] for r in rows], 0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cv", action="store_true", help='DFL cross-validation')
    args = parser.parse_args()
    if args.cv:
        cross_validate()
    else:
        train()
        print('model ->', MODEL_PATH)
