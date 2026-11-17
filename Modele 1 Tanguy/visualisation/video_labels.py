"""Render H/A/D/X overlays on tactical video for local inspection.

Defaults to public frames 11185–18684 with five seconds of surrounding
context and synchronized audio. --pred adds a prediction CSV; --players
adds tracked players; --start/--end select video seconds. Use --format webm
for a VS Code preview.

Banners show ground truth and prediction, green/red agreement outlines,
a ±15 s label timeline, projected tracked ball, whistle indicators and
optional blue-home/red-away player dots. No display server is needed: the
script writes a video under local /Data/.../visu, outside the repository.
Never redistribute challenge footage or derived tracking visualizations."""

import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # Modele 1 Tanguy/
import data_io  # noqa: E402  (also adds the repository root and starter/ to the path)
from load_homography import load as load_homography, project  # noqa: E402

FPS = data_io.FPS
OUT_DIR = os.path.join(os.path.dirname(data_io.CACHE_DIR), "visu")
FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"

COLORS = {"H": (31, 95, 191), "A": (200, 30, 45), "D": (140, 140, 140), "X": (30, 30, 30)}
NAMES = {"H": "H  Barcelona", "A": "A  Athletic", "D": "D  dead ball", "X": 'X  ungraded'}
TIMELINE_S = 15.0


def read_labels(path):
    import pandas as pd
    labels = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()
    if len(labels) != data_io.N_FRAMES:
        raise ValueError("{} : {} lignes, {} attendues".format(path, len(labels), data_io.N_FRAMES))
    return labels


def color_array(labels):
    """Return (N, 3) uint8 colours for each frame."""
    out = np.zeros((len(labels), 3), dtype=np.uint8)
    for lab, col in COLORS.items():
        out[labels == lab] = col
    return out


def whistle_mask():
    """Return whistle-detection mask; empty when audio is unavailable."""
    try:
        from context import MatchContext
        from dead_ball import WhistleFinder
        mask = np.zeros(data_io.N_FRAMES, dtype=bool)
        for s, e, _ in WhistleFinder().whistles(MatchContext()):
            mask[s:e] = True
        return mask
    except Exception as exc:  # Continue without an audio cache or ffmpeg.
        print('(whistles not shown: {})'.format(exc))
        return None


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------------------------
# Frame drawing
# --------------------------------------------------------------------------------------------

class Painter:
    def __init__(self, width, height, gt, pred, tracking, homography, whistles, players):
        self.w, self.h = width, height
        self.scale = width / 1920.0
        self.gt, self.pred = gt, pred
        self.gt_col = color_array(gt)
        self.pred_col = color_array(pred) if pred is not None else None
        self.tracking = tracking
        self.H = homography
        self.whistles = whistles
        self.players = players
        self.f_big, self.f_small = font(int(height * 0.045)), font(int(height * 0.028))
        self.strip_h = max(8, int(height * 0.022))
        self.half = int(TIMELINE_S * FPS)

    def _strip(self, img, colors, frame, y):
        """Draw the full-width label timeline from frame-half to frame+half."""
        idx = np.linspace(frame - self.half, frame + self.half, self.w).round().astype(int)
        ok = (idx >= 0) & (idx < len(colors))
        row = np.zeros((self.w, 3), dtype=np.uint8)
        row[ok] = colors[idx[ok]]
        img[y:y + self.strip_h] = row[None, :, :]

    def _banner(self, draw, x, y, title, label, outline=None):
        text = "{}  {}".format(title, NAMES.get(label, label))
        box = draw.textbbox((0, 0), text, font=self.f_big)
        pad = int(self.h * 0.012)
        rect = (x, y, x + box[2] + 2 * pad, y + box[3] + 2 * pad)
        draw.rectangle(rect, fill=COLORS.get(label, (0, 0, 0)),
                       outline=outline, width=max(2, int(self.h * 0.006)) if outline else 0)
        draw.text((x + pad, y + pad), text, font=self.f_big, fill=(255, 255, 255))
        return rect[3]

    def _point(self, frame, x, y):
        px, py = project(self.H[frame], x, y)
        return px * self.scale, py * self.scale

    def paint(self, rgb, frame):
        # Draw NumPy timelines before text.
        y0 = self.h - 2 * self.strip_h - 6 if self.pred_col is not None else self.h - self.strip_h - 4
        self._strip(rgb, self.gt_col, frame, y0)
        if self.pred_col is not None:
            self._strip(rgb, self.pred_col, frame, y0 + self.strip_h + 2)
        rgb[y0 - 4:self.h, self.w // 2 - 1:self.w // 2 + 1] = 255   # Current time

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        # Tracking projected onto the image
        bx, by = self.tracking["ball_xy"][frame]
        ball_seen = not np.isnan(bx)
        if self.players:
            pl, team = self.tracking["players"][frame], self.tracking["team"]
            r = max(2, int(self.h * 0.006))
            for (x, y, _), t in zip(pl, team):
                if not np.isnan(x):
                    px, py = self._point(frame, x, y)
                    draw.ellipse((px - r, py - r, px + r, py + r),
                                 fill=COLORS["H"] if t == 0 else COLORS["A"], outline=(255, 255, 255))
        if ball_seen:
            px, py = self._point(frame, bx, by)
            r = max(5, int(self.h * 0.016))
            draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 230, 0), width=3)

        # Banners
        m = int(self.h * 0.02)
        g = self.gt[frame]
        bottom = self._banner(draw, m, m + int(self.h * 0.06), 'GROUND TRUTH', g)
        if self.pred is not None:
            p = self.pred[frame]
            outline = None if g == "X" else ((40, 220, 80) if p == g else (255, 60, 60))
            self._banner(draw, m, bottom + m // 2, 'PREDICTION', p, outline)

        # Right-side information
        t = frame / FPS
        info = ["frame {}".format(frame), 'video {:d}:{:05.2f}'.format(int(t // 60), t % 60)]
        if not ball_seen:
            info.append("BALL NOT TRACKED")
        if self.whistles is not None and self.whistles[frame]:
            info.append("WHISTLE")
        y = m + int(self.h * 0.06)
        for line in info:
            box = draw.textbbox((0, 0), line, font=self.f_small)
            x = self.w - m - box[2]
            col = {"BALL NOT TRACKED": (255, 170, 0), "WHISTLE": (255, 255, 0)}.get(line, (255, 255, 255))
            draw.rectangle((x - 6, y - 3, self.w - m + 6, y + box[3] + 5), fill=(0, 0, 0))
            draw.text((x, y), line, font=self.f_small, fill=col)
            y += box[3] + 12
        return img.tobytes()


# --------------------------------------------------------------------------------------------
# Video read/write using ffmpeg
# --------------------------------------------------------------------------------------------

def render(video, out, first, last, width, painter, fmt):
    height = int(round(width * 1080 / 1920 / 2)) * 2
    n = last - first + 1
    t0, dur = first / FPS, n / FPS
    decoder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "{:.3f}".format(t0), "-i", video,
         "-frames:v", str(n), "-vf", "scale={}:{}".format(width, height),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    if fmt == "webm":
        codec = ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                 "-b:v", "0", "-crf", "36", "-pix_fmt", "yuv420p",
                 "-c:a", "libopus", "-b:a", "96k"]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart"]
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "{}x{}".format(width, height), "-r", str(FPS),
         "-i", "-", "-ss", "{:.3f}".format(t0), "-t", "{:.3f}".format(dur), "-i", video,
         "-map", "0:v", "-map", "1:a?", "-shortest"] + codec + [out], stdin=subprocess.PIPE)

    size = width * height * 3
    for k in range(n):
        raw = decoder.stdout.read(size)
        if len(raw) < size:
            print('End of video at frame {}'.format(first + k))
            break
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        encoder.stdin.write(painter.paint(rgb, first + k))
        if k % (30 * FPS) == 0:
            print("  {:5.0f} s / {:.0f} s".format(k / FPS, dur), flush=True)
    decoder.stdout.close()
    decoder.wait()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg encoding failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=data_io.GT_PUBLIC, help='Ground-truth frame,label CSV')
    parser.add_argument("--pred", default=None, help='Optional prediction frame,label CSV')
    parser.add_argument("--start", type=float, default=None, help='Start in video seconds')
    parser.add_argument("--end", type=float, default=None, help='End in video seconds')
    parser.add_argument("--margin", type=float, default=5.0,
                        help='Without --start/--end: seconds of context around ground truth')
    parser.add_argument("--width", type=int, default=960, help="largeur de sortie (px)")
    parser.add_argument("--players", action="store_true", help="Also draw tracked players")
    parser.add_argument("--format", choices=("mp4", "webm"), default="mp4")
    parser.add_argument("--out", default=None, help="Output file")
    args = parser.parse_args()

    gt = read_labels(args.gt)
    pred = read_labels(args.pred) if args.pred else None
    known = np.flatnonzero(gt != "X")
    pad = int(round(args.margin * FPS))
    first = int(round(args.start * FPS)) if args.start is not None else max(0, known[0] - pad)
    last = int(round(args.end * FPS)) - 1 if args.end is not None \
        else min(data_io.N_FRAMES - 1, known[-1] + pad)

    video = data_io.require(["video"])[0]
    tracking = data_io.load_tracking()
    H, _, _ = load_homography()
    painter = Painter(args.width, int(round(args.width * 1080 / 1920 / 2)) * 2, gt, pred,
                      tracking, H, whistle_mask(), args.players)

    out = args.out or os.path.join(OUT_DIR, "labels_{}-{}{}.{}".format(
        first, last, "_pred" if pred is not None else "", args.format))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    print('frames {} -> {} ({:.0f} s), writing {}'.format(first, last, (last - first + 1) / FPS, out))
    render(video, out, first, last, args.width, painter, args.format)
    print("-> {}".format(out))


if __name__ == "__main__":
    main()
