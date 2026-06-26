"""
analysis/spectral.py

Frequency-resolved characterisation of EnCodec's reconstruction error. Runs a
roundtrip pass over a subsample of a dataset (reusing an eval config's dataset
spec) and produces two figures:

    spectral_error_profile   mean |Δ log10 |S|| vs frequency (0–12 kHz), one
                             curve per bitrate -> error concentrates toward HF
                             within the band (motivates LF enhancement)
    bandlimit                avg spectrum of the original 48 kHz audio vs the
                             EnCodec recon (empty above 12 kHz) -> the lost HF
                             band (motivates bandwidth extension)

Run:
    python -m analysis.spectral --config configs/temp_subset.yaml --n 60 \
        --bitrates 1.5 6 12
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from encodec_eval.data import load_items
from encodec_eval.roundtrip import load_mono, roundtrip
from .common import (SR, RESULTS, apply_paper_style, save_fig, figsize, PALETTE,
                     stft_mag, avg_mag_spectrum)

N_FFT, HOP = 2048, 512
EPS = 1e-8


def _per_bin_log_err(orig24, rec24):
    So = stft_mag(orig24, N_FFT, HOP)
    Sr = stft_mag(rec24, N_FFT, HOP)
    t = min(So.shape[-1], Sr.shape[-1])
    lo = torch.log10(So[..., :t] + EPS)
    lr = torch.log10(Sr[..., :t] + EPS)
    return (lo - lr).abs().mean(dim=1)             # [F] mean over time


def run(items, bitrates, out_dir):
    apply_paper_style()
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)         # 0..12 kHz bins
    err = {bw: torch.zeros(len(freqs)) for bw in bitrates}
    # band-limit figure accumulators (at the middle bitrate)
    bw_mid = bitrates[len(bitrates) // 2]
    spec_orig48 = spec_rec24 = None
    f48 = f24 = None
    n = 0

    for it in items:
        x24 = load_mono(it.path, SR)
        x48 = load_mono(it.path, 48000)
        for bw in bitrates:
            o, r = roundtrip(x24, bw)
            err[bw] += _per_bin_log_err(o, r)
        # spectra for the band-limit panel
        _, rec_mid = roundtrip(x24, bw_mid)
        f48, s_o = avg_mag_spectrum(x48, sr=48000)
        f24, s_r = avg_mag_spectrum(rec_mid, sr=SR)
        spec_orig48 = s_o if spec_orig48 is None else spec_orig48 + s_o
        spec_rec24 = s_r if spec_rec24 is None else spec_rec24 + s_r
        n += 1

    # --- spectral error profile ---
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.5))
    for i, bw in enumerate(bitrates):
        ax.plot(freqs / 1000, (err[bw] / n).numpy(),
                color=PALETTE[i % len(PALETTE)], label=f"{bw:g} kbps")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(r"mean $|\Delta \log_{10}|S||$")
    ax.set_title(f"Reconstruction error vs frequency (n={n})")
    ax.legend(title="bitrate")
    save_fig(fig, "spectral_error_profile", fig_dir=out_dir)
    plt.close(fig)

    # --- band-limit (lost HF band) ---
    so = 20 * np.log10(spec_orig48 / n + EPS)
    sr_ = 20 * np.log10(spec_rec24 / n + EPS)
    ref = so.max()
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.5))
    ax.plot(f48 / 1000, so - ref, color="0.4", label="original (48 kHz)")
    ax.plot(f24 / 1000, sr_ - ref, color=PALETTE[3],
            label=f"EnCodec recon @ {bw_mid:g} kbps")
    ax.axvline(12, color=PALETTE[1], ls="--", lw=1.0, label="12 kHz band-limit")
    ax.set_xlim(0, 24)
    ax.set_ylim(-90, 3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB, peak-normalised)")
    ax.set_title("Spectrum discarded above the codec band-limit")
    ax.legend()
    save_fig(fig, "bandlimit", fig_dir=out_dir)
    plt.close(fig)
    print(f"spectral figures -> {out_dir} (n={n}, bitrates={bitrates})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/temp_subset.yaml")
    ap.add_argument("--n", type=int, default=60, help="files to sample")
    ap.add_argument("--bitrates", type=float, nargs="+", default=[1.5, 6.0, 12.0])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"][0]
    per_class = max(1, round(args.n / 23))         # spread across all 23 classes
    ds = {**ds, "sample_per_class": per_class, "limit": args.n}
    items = load_items(ds)
    out_dir = RESULTS / "spectral" / "figures"
    run(items, args.bitrates, out_dir)


if __name__ == "__main__":
    main()
