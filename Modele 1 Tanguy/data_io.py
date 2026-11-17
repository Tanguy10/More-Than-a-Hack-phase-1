"""Load data with a NumPy tracking cache.

All arrays use video-frame indices 0..199318, matching the submission.
load_tracking() returns NumPy arrays and caches the first read.
load_set_pieces() returns Smart Tagging set-piece events."""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

# Store the cache next to the data to avoid the small home-directory quota.
# MTAH_CACHE_DIR overrides the cache location.
CACHE_DIR = os.environ.get("MTAH_CACHE_DIR") or os.path.join(
    os.path.dirname(os.path.realpath(os.path.join(ROOT, "data", "tactical"))), "cache")
N_FRAMES = source_frames("tactical", GAME)
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]  # Pitch (length, width) in metres
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

HOME, AWAY = 0, 1


# --------------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------------

def _parse_raw_atd(raw_path, n_frames):
    """Read raw ATD records: frame_id:player x,y,s;...:ball x,y,z."""
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = None
    with open(raw_path) as handle:
        for line in handle:
            frame_id, player_part, ball_part = line.rstrip("\n").split(":")
            frame = int(frame_id) - 1  # ATD is one-based; video frames are zero-based.
            groups = player_part.split(";")
            if players is None:
                players = np.full((n_frames, len(groups), 3), np.nan, dtype=np.float32)
            # float("NaN") already handles missing observations.
            players[frame] = [[float(v) for v in g.split(",")] for g in groups]
            ball[frame] = [float(v) for v in ball_part.split(",")]
    return ball, players


def team_of_each_track(meta_path):
    """Team of each tracking slot: 0 = home, 1 = away."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def load_tracking(use_cache=True):
    """Return tactical tracking indexed by video frame.

ball_xy: (N, 2), normalized 0–1 coordinates, NaN when unseen.
ball_z: (N,), the supplied third ball field.
players: (N, S, 3), x, y and speed (m/s), NaN when missing.
team: (S,), 0 = H, 1 = A."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    cache = os.path.join(CACHE_DIR, "tracking_tactical.npz")
    if use_cache and os.path.isfile(cache):
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    ball, players = _parse_raw_atd(raw_path, N_FRAMES)
    out = {
        "ball_xy": ball[:, :2],
        "ball_z": ball[:, 2],
        "players": players,
        "team": team_of_each_track(meta_path),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out


# --------------------------------------------------------------------------------------------
# Smart Tagging
# --------------------------------------------------------------------------------------------

@dataclass
class TaggedEvent:
    code: str
    type: str          # For example Throw-in or Corner Kick; empty if absent.
    team: str          # "Barcelona" / "Atletic Club" / "N/A"
    start_frame: int
    end_frame: int     # Excluded


def load_smart_tagging():
    """Return all Smart Tagging instances sorted by start."""
    path = require(["smart_tagging"])[0]
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append(TaggedEvent(
            code=(inst.findtext("code") or "").strip(),
            type=(tags.get("Type") or "").strip(),
            team=(tags.get("Team") or "N/A").strip(),
            start_frame=int(round(float(inst.findtext("start")) * FPS)),
            end_frame=int(round(float(inst.findtext("end")) * FPS)),
        ))
    return sorted(out, key=lambda e: e.start_frame)


def load_set_pieces():
    """Return set pieces: throw-ins, corners, goal kicks, free kicks and kickoffs."""
    return [e for e in load_smart_tagging() if e.code.lower() == "set pieces"]


# --------------------------------------------------------------------------------------------
# Other helpers
# --------------------------------------------------------------------------------------------

def graded_mask():
    """True on the graded frames in the two halves."""
    mask = np.zeros(N_FRAMES, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def load_public_gt():
    """Public ground-truth labels, with X outside the first five minutes."""
    import pandas as pd
    return pd.read_csv(GT_PUBLIC)["label"].to_numpy().astype("<U1")


def write_submission(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
