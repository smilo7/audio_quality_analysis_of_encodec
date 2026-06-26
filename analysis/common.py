"""
analysis/common.py

Shared helpers for the figure scripts: the paper plot style, STFT / spectrogram
utilities, and the BST class-group mapping.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from encodec_eval.plot_style import (  # noqa: E402  re-exported for figure scripts
    apply_paper_style, save_fig, figsize, PALETTE,
)

SR = 24000
RESULTS = _REPO / "results"


# ---------------------------------------------------------------------------
# BST taxonomy: second-level class prefix -> top-level group
# ---------------------------------------------------------------------------

BST_GROUPS = {
    "is": "Instrument",
    "m": "Music",
    "fx": "Sound FX",
    "ss": "Soundscape",
    "sp": "Speech",
}
# stable colour per group (indexes into PALETTE)
GROUP_ORDER = ["Instrument", "Music", "Sound FX", "Soundscape", "Speech"]


def group_of(cls: str) -> str:
    return BST_GROUPS.get(str(cls).split("-")[0], "Other")


def group_color(cls_or_group: str):
    g = cls_or_group if cls_or_group in GROUP_ORDER else group_of(cls_or_group)
    idx = GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
    return PALETTE[idx % len(PALETTE)]


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def stft_mag(x, n_fft=1024, hop=256):
    """Magnitude STFT of a [T] or [1, T] waveform -> [F, T_frames] tensor."""
    x = x.reshape(-1).float()
    win = torch.hann_window(n_fft, device=x.device)
    S = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, center=True, return_complex=True)
    return S.abs()


def db_spectrogram(x, n_fft=1024, hop=256, top_db=80.0):
    """Log-magnitude spectrogram in dB, clipped to `top_db` below the peak."""
    S = stft_mag(x, n_fft, hop)
    db = 20 * torch.log10(S + 1e-8)
    db = db - db.max()
    return db.clamp(min=-top_db).numpy()


def plot_spectrogram(ax, x, sr=SR, n_fft=1024, hop=256, top_db=80.0,
                     fmax=None, title=None, cmap="magma"):
    """Draw a dB spectrogram on `ax`; returns the image handle."""
    db = db_spectrogram(x, n_fft, hop, top_db)
    dur = x.reshape(-1).shape[-1] / sr
    im = ax.imshow(db, origin="lower", aspect="auto", cmap=cmap,
                   extent=[0, dur, 0, sr / 2 / 1000])
    if fmax:
        ax.set_ylim(0, fmax / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    if title:
        ax.set_title(title)
    return im


def avg_mag_spectrum(x, n_fft=2048, hop=512, sr=SR):
    """Time-averaged magnitude spectrum -> (freqs_hz, mag) numpy arrays."""
    S = stft_mag(x, n_fft, hop).mean(dim=-1).numpy()
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    return freqs, S


def analytic_envelope(x, smooth_ms=0.0, sr=SR):
    """|Hilbert| amplitude envelope (ripple-free for tonal content) of a
    [T]/[1,T] waveform -> (times_s, env), full sample rate. Optional moving-average
    smoothing over `smooth_ms`."""
    x = x.reshape(-1).float()
    n = x.shape[-1]
    Xf = torch.fft.fft(x)
    h = torch.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    env = torch.fft.ifft(Xf * h).abs()
    if smooth_ms > 0:
        k = max(1, int(smooth_ms * 1e-3 * sr))
        env = torch.nn.functional.avg_pool1d(
            env.view(1, 1, -1), kernel_size=k, stride=1, padding=k // 2).view(-1)[:n]
    times = np.arange(n) / sr
    return times, env.numpy()


def local_env(x, win_ms=3.0, hop_ms=0.5, sr=SR):
    """Local Hann-weighted RMS envelope -> (times_s, rms). Smooth for both tonal
    and broadband content (no tonal ripple) yet strictly local (unlike Hilbert,
    which rings before sharp onsets and fabricates pre-onset energy)."""
    x = x.reshape(-1).float()
    win = max(8, int(win_ms * 1e-3 * sr))
    hop = max(1, int(hop_ms * 1e-3 * sr))
    w = torch.hann_window(win)
    frames = x.unfold(0, win, hop)                 # [n, win]
    rms = (frames.pow(2) * w).sum(1).div(w.sum()).clamp_min(0).sqrt().numpy()
    times = (np.arange(len(rms)) * hop + win / 2) / sr
    return times, rms


def rms_envelope(x, frame=120, hop=60):
    """Frame-wise RMS envelope of a [T]/[1,T] waveform -> (times_s, rms)."""
    x = x.reshape(-1).float()
    frames = x.unfold(0, frame, hop)               # [n_frames, frame]
    rms = frames.pow(2).mean(dim=1).sqrt().numpy()
    times = (np.arange(len(rms)) * hop + frame / 2) / SR
    return times, rms
