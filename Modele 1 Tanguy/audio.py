"""Tactical-video audio: a 3–5 kHz spectrum for each video frame.

The referee whistle contains two simultaneous tones near 3.85 and 4.3 kHz.
Audio comes from the tactical video, so video frame i corresponds to
(i + 0.5) / 25 seconds. load_band_spectrogram() caches the result.
The returned db array is (N, bins), float16 dB; freqs contains bin frequencies in Hz."""

import os
import shutil
import subprocess

import numpy as np

import data_io

SR = 16000                     # Hz; sufficient because the whistle is below 8 kHz
HOP = SR // data_io.FPS        # 640 samples = one video frame
NFFT = 2048                    # 128 ms window, 7.8 Hz resolution
F_LO, F_HI = 3000.0, 5000.0    # Cached frequency band
CHUNK = 4000                   # Video frames processed together to bound memory


def _decode(video_path):
    """Decode a mono 16 kHz int16 audio track with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", video_path,
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.int16)


def _band_spectrogram(x, n_frames):
    """Return [F_LO, F_HI] dB per video frame, with a frame-centred window."""
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    keep = (freqs >= F_LO) & (freqs <= F_HI)
    window = np.hanning(NFFT).astype(np.float32)

    # Centre frame i on sample i*HOP + HOP/2; zero-pad the edges.
    half = NFFT // 2
    need = (n_frames - 1) * HOP + HOP // 2 + NFFT
    xp = np.zeros(max(need, len(x) + 2 * half), dtype=np.float32)
    xp[half:half + len(x)] = x / 32768.0
    view = np.lib.stride_tricks.sliding_window_view(xp, NFFT)[HOP // 2::HOP]

    out = np.empty((n_frames, keep.sum()), dtype=np.float16)
    for lo in range(0, n_frames, CHUNK):
        hi = min(n_frames, lo + CHUNK)
        spec = np.fft.rfft(view[lo:hi] * window, axis=1)[:, keep]
        out[lo:hi] = 10.0 * np.log10(np.abs(spec) ** 2 + 1e-12)
    return out, freqs[keep]


def load_band_spectrogram(use_cache=True):
    cache = os.path.join(data_io.CACHE_DIR, "audio_band_tactical.npz")
    if use_cache and os.path.isfile(cache):
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    video_path = data_io.require(["video"])[0]
    db, freqs = _band_spectrogram(_decode(video_path), data_io.N_FRAMES)
    out = {"db": db, "freqs": freqs}
    os.makedirs(data_io.CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out
