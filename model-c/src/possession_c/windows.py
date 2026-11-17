"""Padded inference windows and batching."""

import numpy as np
import torch


def extract_window(sequence, start, length, run):
    a, b = run
    lo = max(a, start)
    hi = min(b, start + length)
    slots = len(sequence.players_team)
    xy = np.full((length, slots, 2), np.nan, "f4")
    speed = np.full((length, slots), np.nan, "f4")
    ball = np.full((length, 2), np.nan, "f4")
    if hi > lo:
        offset = lo - start
        xy[offset : offset + hi - lo] = sequence.players_xy[lo:hi]
        speed[offset : offset + hi - lo] = sequence.players_speed[lo:hi]
        ball[offset : offset + hi - lo] = sequence.ball_xy[lo:hi]
    return {
        "xy": xy,
        "speed": speed,
        "ball": ball,
        "team": np.array(sequence.players_team, copy=True),
        "pitch": np.array(sequence.pitch, "f4"),
    }


def collate(samples):
    slots = max(len(s["team"]) for s in samples)
    out = {}
    for key in ("xy", "speed", "team"):
        values = []
        for sample in samples:
            value = sample[key]
            pad = slots - len(sample["team"])
            padding = (
                ((0, 0), (0, pad), (0, 0))
                if key == "xy"
                else (((0, 0), (0, pad)) if key == "speed" else ((0, pad),))
            )
            values.append(np.pad(value, padding, constant_values=-1 if key == "team" else np.nan))
        out[key] = torch.from_numpy(np.stack(values))
    for key in ("ball", "pitch", "labels"):
        if key in samples[0]:
            out[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    return out
