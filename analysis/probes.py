"""
analysis/probes.py

Targeted artifact probes on controlled signals, each producing a thesis figure
and a printed quantitative summary:

    sine        log sweep + single tones -> frequency response, HF rolloff, and
                spurious ("hallucinated") harmonics
    transient   percussive material (amen break) -> transient attenuation/smearing
    preecho     sharp onset after silence -> MP3-style pre-echo (energy before onset)

Run:
    python -m analysis.probes                 # all probes at 6 kbps
    python -m analysis.probes --probe sine --bitrates 1.5 12
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from encodec_eval.roundtrip import roundtrip, load_mono
from .common import (
    SR, RESULTS, apply_paper_style, save_fig, figsize, PALETTE,
    stft_mag, plot_spectrogram, avg_mag_spectrum, rms_envelope, local_env,
)

_REPO = Path(__file__).resolve().parents[1]
FIG_DIR = RESULTS / "probes" / "figures"
AMEN = _REPO / "legacy" / "scratch_audio" / "amen-break.wav"
TRANSIENT_DIR = _REPO / "datasets" / "transients"   # drop dry castanets/cymbals here


# ---------------------------------------------------------------------------
# Signal synthesis
# ---------------------------------------------------------------------------

def log_sweep(f0=20.0, f1=11500.0, dur=4.0, amp=0.5):
    t = torch.arange(int(dur * SR)) / SR
    L = math.log(f1 / f0)
    phase = 2 * math.pi * f0 * dur / L * (torch.exp(t / dur * L) - 1.0)
    return (amp * torch.sin(phase)).unsqueeze(0)


def tone(freq, dur=1.0, amp=0.5):
    t = torch.arange(int(dur * SR)) / SR
    return (amp * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Probe 1: sine response (rolloff + hallucinated harmonics)
# ---------------------------------------------------------------------------

def _spectrogram_grid(x, bitrates, name, title_in, fmax=None):
    """Input + one EnCodec-output spectrogram per bitrate, in a shared-y row."""
    n = 1 + len(bitrates)
    fig, axes = plt.subplots(1, n, figsize=figsize("text", ratio=1.5 / n),
                             sharey=True)
    plot_spectrogram(axes[0], x, title=title_in, fmax=fmax)
    for ax, bw in zip(axes[1:], bitrates):
        plot_spectrogram(ax, roundtrip(x, bw)[1], title=f"{bw:g} kbps", fmax=fmax)
    for ax in axes:
        ax.set_xlabel("")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.supxlabel("Time (s)", fontsize=8)
    save_fig(fig, name, fig_dir=FIG_DIR)
    plt.close(fig)


def probe_sine(bitrates=(6.0,)):
    apply_paper_style()
    f0 = 1000.0

    # (a) sweep spectrograms: input vs codec output, one panel per bitrate
    _spectrogram_grid(log_sweep(), bitrates, "sine_sweep_spectrogram",
                      "Input log sweep")

    # (b) constant-tone spectrograms (spurious harmonics as steady horizontal
    #     lines; coding noise floor over time), one panel per bitrate
    _spectrogram_grid(tone(f0, dur=2.0), bitrates, "sine_tone_spectrogram",
                      f"Input {f0/1000:g} kHz tone")

    # (c) single 1 kHz tone: output spectrum shows fundamental + spurious harmonics
    x = tone(f0)
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.5))
    freqs, mag_in = avg_mag_spectrum(x)
    ax.plot(freqs / 1000, 20 * np.log10(mag_in / mag_in.max() + 1e-9),
            color="0.6", lw=1.0, label="input (1 kHz)")
    ratios = {}
    for i, bw in enumerate(bitrates):
        _, rec = roundtrip(x, bw)
        freqs, mag = avg_mag_spectrum(rec)
        mag_db = 20 * np.log10(mag / mag.max() + 1e-9)
        ax.plot(freqs / 1000, mag_db, color=PALETTE[i % len(PALETTE)],
                label=f"output @ {bw:g} kbps")
        ratios[bw] = _spurious_ratio_db(freqs, mag, f0)
    for k in range(2, 9):  # mark harmonic positions within band
        if k * f0 < SR / 2:
            ax.axvline(k * f0 / 1000, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(0, SR / 2 / 1000)
    ax.set_ylim(-90, 3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB, peak-normalised)")
    ax.set_title("Reconstruction of a pure 1 kHz tone")
    ax.legend()
    save_fig(fig, "sine_harmonics", fig_dir=FIG_DIR)
    plt.close(fig)

    print("[sine] spurious-to-fundamental ratio (dB, higher = more hallucinated energy):")
    for bw, r in ratios.items():
        print(f"        {bw:>5g} kbps : {r:6.1f} dB")
    print(f"[sine] figures -> {FIG_DIR}")


def _spurious_ratio_db(freqs, mag, f0, tol=30.0):
    """Energy outside a narrow band around the fundamental, relative to it."""
    p = mag ** 2
    fund = (np.abs(freqs - f0) <= tol)
    e_fund = p[fund].sum()
    e_spur = p[~fund & (freqs > tol)].sum()
    return 10 * np.log10((e_spur + 1e-12) / (e_fund + 1e-12))


# ---------------------------------------------------------------------------
# Probe 2: transients (attenuation + smearing)
# ---------------------------------------------------------------------------

def probe_transient(bitrates=(1.5, 12.0), window_s=2.0):
    apply_paper_style()
    if not AMEN.exists():
        print(f"[transient] missing {AMEN}; skipping")
        return
    x = load_mono(AMEN)[..., : int(window_s * SR)]
    recons = {bw: roundtrip(x, bw)[1] for bw in bitrates}
    orig = roundtrip(x, bitrates[0])[0]

    # spectrograms: original vs the lowest bitrate (most revealing)
    bw_lo = bitrates[0]
    fig, axes = plt.subplots(2, 1, figsize=figsize("text", ratio=0.7), sharex=True)
    plot_spectrogram(axes[0], orig, fmax=12000, title="Original (drum break)")
    plot_spectrogram(axes[1], recons[bw_lo], fmax=12000,
                     title=f"EnCodec @ {bw_lo:g} kbps")
    axes[0].set_xlabel("")
    save_fig(fig, "transient_spectrogram", fig_dir=FIG_DIR)
    plt.close(fig)

    # energy envelope across bitrates + peak attenuation at onsets
    t_o, e_o = rms_envelope(orig)
    peaks = _local_peaks(e_o, min_rel=0.4)
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.42))
    ax.plot(t_o, e_o, color="0.3", lw=1.4, label="original")
    print(f"[transient] {len(peaks)} onsets; mean peak attenuation by bitrate:")
    for i, bw in enumerate(bitrates):
        t_r, e_r = rms_envelope(recons[bw])
        n = min(len(e_o), len(e_r))
        ax.plot(t_r[:n], e_r[:n], color=PALETTE[i % len(PALETTE)], lw=1.0,
                label=f"@ {bw:g} kbps")
        atten = 20 * np.log10((e_r[peaks] + 1e-9) / (e_o[peaks] + 1e-9))
        print(f"        {bw:>5g} kbps : {atten.mean():+.2f} dB")
    ax.plot(t_o[peaks], e_o[peaks], "v", color="0.3", ms=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS amplitude")
    ax.set_title("Transient energy envelope (drum break)")
    ax.legend(ncol=len(bitrates) + 1, loc="upper right")
    save_fig(fig, "transient_envelope", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[transient] figures -> {FIG_DIR}")


def _local_peaks(e, min_rel=0.4, guard=8):
    thr = min_rel * e.max()
    peaks = []
    for i in range(guard, len(e) - guard):
        if e[i] >= thr and e[i] == e[i - guard:i + guard + 1].max():
            if not peaks or i - peaks[-1] > guard:
                peaks.append(i)
    return np.array(peaks, dtype=int)


# ---------------------------------------------------------------------------
# Probe 3: pre-echo (energy before a sharp onset)
# ---------------------------------------------------------------------------

def _env_db(w):
    """Local Hann-RMS envelope in dB, peak-normalised."""
    t, e = local_env(w, win_ms=5.0, hop_ms=1.0)
    return t, 20 * np.log10(e / (e.max() + 1e-9) + 1e-9)


def _find_onset(w, rel=0.2):
    """First time the local envelope exceeds `rel` x its peak."""
    t, e = local_env(w, win_ms=3.0, hop_ms=0.5)
    idx = int(np.argmax(e >= rel * e.max()))
    return t[idx]


def _preecho_panel(ax, x, bitrates, onset_t, span=0.06, title=""):
    """Plot original + each bitrate's recon envelope (dB) around an onset."""
    t_o, db_o = _env_db(roundtrip(x, bitrates[0])[0])
    mo = (t_o >= onset_t - span) & (t_o <= onset_t + span)
    ax.plot(1000 * (t_o[mo] - onset_t), db_o[mo], color="0.3", lw=1.4, label="original")
    res = {}
    for i, bw in enumerate(bitrates):
        t_r, db_r = _env_db(roundtrip(x, bw)[1])
        mr = (t_r >= onset_t - span) & (t_r <= onset_t + span)
        ax.plot(1000 * (t_r[mr] - onset_t), db_r[mr],
                color=PALETTE[i % len(PALETTE)], label=f"@ {bw:g} kbps")
        res[bw] = db_r[(t_r > onset_t - 0.02) & (t_r < onset_t)].mean()
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_ylim(-90, 3)
    ax.set_xlabel("Time relative to onset (ms)")
    ax.set_ylabel("Level (dB, peak-norm.)")
    if title:
        ax.set_title(title)
    return res


def probe_preecho(bitrates=(1.5, 12.0), sil=0.4, burst=0.12):
    """Synthetic control: silence -> abrupt broadband noise burst -> silence.
    A broadband transient (not a tone) is the canonical pre-echo stimulus and
    gives a ripple-free envelope."""
    apply_paper_style()
    n_sil, n_burst = int(sil * SR), int(burst * SR)
    g = torch.Generator().manual_seed(0)
    b = torch.randn(n_burst, generator=g)              # hard-onset white noise
    b = 0.5 * b / b.abs().max()
    x = torch.cat([torch.zeros(n_sil), b, torch.zeros(n_sil)]).unsqueeze(0)
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.45))
    res = _preecho_panel(ax, x, bitrates, onset_t=sil,
                         title="Pre-echo probe (synthetic noise burst)")
    ax.legend()
    save_fig(fig, "preecho", fig_dir=FIG_DIR)
    plt.close(fig)
    print("[preecho] synthetic pre-onset level (dB below peak):")
    for bw, v in res.items():
        print(f"        {bw:>5g} kbps : {v:6.1f} dB")
    print(f"[preecho] figure -> {FIG_DIR}")


def probe_preecho_files(files, bitrates=(1.5, 12.0), pad=0.3):
    """Pre-echo test on real percussive recordings (castanets, cymbals, ...).

    Each file is normalised and prepended with `pad` s of silence so there is a
    clean pre-onset region; the onset is found automatically and the codec's
    pre-onset energy is compared to the original (which is digital silence there).
    """
    apply_paper_style()
    files = [Path(f) for f in files]
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=figsize("text", ratio=0.34 + 0.06 * (n == 1)),
                             squeeze=False)
    print("[preecho-files] pre-onset level (dB below peak), per file:")
    for ax, f in zip(axes[0], files):
        x = load_mono(f)
        x = x / (x.abs().max() + 1e-9)
        x = torch.cat([torch.zeros(1, int(pad * SR)), x], dim=-1)
        onset_t = _find_onset(x)
        res = _preecho_panel(ax, x, bitrates, onset_t, title=f.stem)
        print(f"  {f.stem:<24} " +
              " ".join(f"{bw:g}k={v:5.1f}" for bw, v in res.items()))
    axes[0][-1].legend(fontsize=7)
    save_fig(fig, "preecho_real", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[preecho-files] figure -> {FIG_DIR}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe",
                    choices=["sine", "transient", "preecho", "preecho_real", "all"],
                    default="all")
    ap.add_argument("--bitrates", type=float, nargs="+", default=[1.5, 12.0])
    ap.add_argument("--files", nargs="+", default=None,
                    help="audio files for preecho_real (default: datasets/transients/*)")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.probe in ("sine", "all"):
        probe_sine(args.bitrates)
    if args.probe in ("transient", "all"):
        probe_transient(args.bitrates)
    if args.probe in ("preecho", "all"):
        probe_preecho(args.bitrates)
    if args.probe in ("preecho_real", "all"):
        files = args.files
        if not files and TRANSIENT_DIR.is_dir():
            files = sorted(str(p) for p in TRANSIENT_DIR.glob("*")
                           if p.suffix.lower() in {".wav", ".flac", ".aif", ".aiff"})
        if files:
            probe_preecho_files(files, args.bitrates)
        elif args.probe == "preecho_real":
            print(f"[preecho-files] no files; drop dry transients in {TRANSIENT_DIR}")


if __name__ == "__main__":
    main()
