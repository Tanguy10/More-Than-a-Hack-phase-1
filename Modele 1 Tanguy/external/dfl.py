"""DFL Bundesliga data: Bassek et al. (2025), Scientific Data, figshare 28196177, CC BY 4.0.

Seven corrected 25 Hz matches include 22 players, three referees and the ball.
BallStatus: 1 alive, 0 dead. BallPossession: 1 home, 2 away.
Download separately with download_external.sh; never commit the data.

load_match() returns ball (N,4): centred x/y/z metres and speed m/s;
alive (N,), poss (N,), players (N,P,3): x/y/speed with NaN when absent;
team (P,): 0 home, 1 away, 2 referee; period (N,): 1/2; and t (N,): UTC seconds.
load_events() returns dictionaries with frame, type, team and decision_frame.
Running this module builds caches in parallel and prints a summary."""

import glob
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import external_dir  # noqa: E402

FPS = 25
PITCH = (105.0, 68.0)
RESTART_TYPES = ("KickOff", "ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty")


def data_dir():
    return os.path.join(external_dir(), "dfl_figshare")


def cache_dir():
    path = os.path.join(data_dir(), "cache")
    os.makedirs(path, exist_ok=True)
    return path


def list_matches():
    files = glob.glob(os.path.join(data_dir(), "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(os.path.basename(f).rsplit("_", 1)[1][:-4] for f in files)


def _file(kind, match_id):
    hits = glob.glob(os.path.join(data_dir(), "DFL_{}_*{}.xml".format(kind, match_id)))
    if not hits:
        raise FileNotFoundError("{} {} : lancer download_external.sh".format(kind, match_id))
    return hits[0]


def _ts(text):
    return datetime.fromisoformat(text).timestamp()


def match_info(match_id):
    from lxml import etree
    root = etree.parse(_file("02_01_matchinformation", match_id)).getroot()
    gen = root.find(".//General")
    teams = {t.get("TeamId"): t for t in root.iter("Team")}
    return {
        "match_id": match_id,
        "title": gen.get("MatchTitle"),
        "competition": gen.get("CompetitionName"),
        "home_id": gen.get("HomeTeamId"),
        "away_id": gen.get("GuestTeamId"),
        "players": {p.get("PersonId"): {"team": tid, "shirt": p.get("ShirtNumber"),
                                        "position": p.get("PlayingPosition")}
                    for tid, t in teams.items() for p in t.iter("Player")},
    }


def _parse_positions(match_id):
    """Stream the roughly 400 MB XML one FrameSet at a time."""
    from lxml import etree
    info = match_info(match_id)
    team_code = {info["home_id"]: 0, info["away_id"]: 1}
    sections = {}  # GameSection -> {"ball": ..., "people": [(pid, team, arr)]}
    for _, fs in etree.iterparse(_file("04_03_positions_raw_observed", match_id), tag="FrameSet"):
        sec = sections.setdefault(fs.get("GameSection"), {"people": []})
        team_id = fs.get("TeamId")
        frames = list(fs.iterchildren("Frame"))
        n = np.array([int(f.get("N")) for f in frames])
        xy = np.array([(float(f.get("X")), float(f.get("Y")), float(f.get("S"))) for f in frames],
                      dtype=np.float32)
        if team_id == "BALL":
            sec["ball_n"] = n
            sec["ball"] = np.column_stack([xy[:, :2], [float(f.get("Z")) for f in frames], xy[:, 2]])
            sec["alive"] = np.array([int(f.get("BallStatus")) for f in frames], dtype=np.int8)
            sec["poss"] = np.array([int(f.get("BallPossession")) for f in frames], dtype=np.int8)
            sec["t"] = np.array([_ts(f.get("T")) for f in frames])
        else:
            team = 2 if team_id == "referee" else team_code[team_id]
            sec["people"].append((fs.get("PersonId"), team, n, xy))
        fs.clear()
        while fs.getprevious() is not None:
            del fs.getparent()[0]

    # Use the same player index for both halves.
    pids = sorted({(team, pid) for s in sections.values() for pid, team, _, _ in s["people"]})
    col = {pid: i for i, (_, pid) in enumerate(pids)}
    out = {k: [] for k in ("ball", "alive", "poss", "t", "players", "period", "frame_n")}
    for period, name in ((1, "firstHalf"), (2, "secondHalf")):
        sec = sections[name]
        n0, n_frames = sec["ball_n"][0], len(sec["ball_n"])
        assert np.all(np.diff(sec["ball_n"]) == 1), 'Noncontiguous ball frames'
        players = np.full((n_frames, len(pids), 3), np.nan, dtype=np.float32)
        for pid, _, n, xy in sec["people"]:
            idx = n - n0
            ok = (idx >= 0) & (idx < n_frames)
            players[idx[ok], col[pid]] = xy[ok]
        for key in ("ball", "alive", "poss", "t"):
            out[key].append(sec[key])
        out["players"].append(players)
        out["period"].append(np.full(n_frames, period, dtype=np.int8))
        out["frame_n"].append(sec["ball_n"])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["ball"] = out["ball"].astype(np.float32)
    out["team"] = np.array([team for team, _ in pids], dtype=np.int8)
    out["player_ids"] = np.array([pid for _, pid in pids])
    return out


def load_match(match_id, use_cache=True):
    cache = os.path.join(cache_dir(), match_id + ".npz")
    if use_cache and os.path.isfile(cache):
        data = np.load(cache)
        return {k: data[k] for k in data.files}
    out = _parse_positions(match_id)
    np.savez_compressed(cache, **out)
    return out


def load_events(match_id, t=None):
    """Return DFL events aligned with per-frame UTC seconds m["t"]."""
    from lxml import etree
    info = match_info(match_id)
    team_code = {info["home_id"]: 0, info["away_id"]: 1}
    out = []
    for ev in etree.parse(_file("03_02_events_raw", match_id)).getroot().iter("Event"):
        child = next(iter(ev), None)
        if child is None:
            continue
        team = child.get("Team") or child.get("WinnerTeam") or child.get("TeamFouled")
        row = {"type": child.tag, "time": _ts(ev.get("EventTime")),
               "team": team_code.get(team, -1),
               "x": float(ev.get("X-Position", "nan")), "y": float(ev.get("Y-Position", "nan")),
               "decision_time": _ts(child.get("DecisionTimestamp")) if child.get("DecisionTimestamp") else None,
               "attrs": dict(child.attrib)}
        sub = next(iter(child), None)
        row["sub"] = sub.tag if sub is not None else ""
        out.append(row)
    if t is not None:
        for row in out:
            row["frame"] = int(np.clip(np.searchsorted(t, row["time"]), 0, len(t) - 1))
            row["decision_frame"] = (int(np.clip(np.searchsorted(t, row["decision_time"]), 0, len(t) - 1))
                                     if row["decision_time"] else None)
    return sorted(out, key=lambda r: r["time"])


def segments(mask):
    """Return [start, end) intervals of True values in a Boolean array."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2], d[1::2]))


def _build(match_id):
    m = load_match(match_id)
    return match_id, len(m["alive"])


if __name__ == "__main__":
    from multiprocessing import Pool
    ids = list_matches()
    with Pool(len(ids)) as pool:
        for mid, n in pool.imap_unordered(_build, ids):
            print("cache", mid, n, "frames", flush=True)
