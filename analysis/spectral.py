"""
analysis/spectral.py

Frequency-resolved characterisation of EnCodec's reconstruction error. Runs a
roundtrip pass over a (seeded, class-stratified) subsample of a dataset and
produces:

    profile      mean log-spectral error (dB) vs frequency, one curve per
                 bitrate -> where in the band the codec is least faithful
    bandlimit    avg magnitude spectrum of the original vs the EnCodec recon
                 (empty above 12 kHz) -> the discarded HF band  [best from the
                 48 kHz subset, which carries true content up to 24 kHz]
    convergence  the profile re-estimated at increasing sample sizes, to show
                 the curve has stabilised (justifies the chosen n)

The error is reported in dB using the *same* convention as the LSD metric in
the methods: per frequency bin, the mean absolute difference of the log-power
spectra, 10*log10(|S|^2 + eps), averaged over time frames and files. (LSD is the
RMS-across-frequency aggregate of these same per-bin dB differences.)

Run (from the repo root):
    python -m analysis.spectral --mode profile --config configs/bsd10k_local.yaml --n 300
    python -m analysis.spectral --mode convergence --config configs/bsd10k_local.yaml
    python -m analysis.spectral --mode bandlimit --config configs/bsd_balanced.yaml --n 300
"""

import argparse
import random

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
OUT = RESULTS / "spectral" / "figures"


def _per_bin_db_err(orig24, rec24):
    """Per-frequency mean |Δ| of the log-power spectrum (dB), over time frames.
    Same dB convention as LSD: 10*log10(|S|^2 + eps)."""
    So, Sr = stft_mag(orig24, N_FFT, HOP), stft_mag(rec24, N_FFT, HOP)
    t = min(So.shape[-1], Sr.shape[-1])
    lo = 10 * torch.log10(So[..., :t] ** 2 + EPS)
    lr = 10 * torch.log10(Sr[..., :t] ** 2 + EPS)
    return (lo - lr).abs().mean(dim=1)                 # [F] mean over time, dB


def _freqs_khz():
    return np.fft.rfftfreq(N_FFT, 1 / SR) / 1000.0


# ---------------------------------------------------------------------------
# Error profile (all bitrates)
# ---------------------------------------------------------------------------

def error_profile(items, bitrates, out_dir):
    apply_paper_style()
    fk = _freqs_khz()
    err = {bw: torch.zeros(len(fk)) for bw in bitrates}
    n = skipped = 0
    for it in items:
        try:                                       # skip clips too short for the STFT
            x24 = load_mono(it.path, SR)
            contrib = {bw: _per_bin_db_err(*roundtrip(x24, bw)) for bw in bitrates}
        except Exception:
            skipped += 1
            continue
        for bw in bitrates:
            err[bw] += contrib[bw]
        n += 1

    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.52))
    for i, bw in enumerate(bitrates):
        ax.plot(fk, (err[bw] / n).numpy(), color=PALETTE[i % len(PALETTE)],
                label=f"{bw:g}")
    ax.set_xlim(0, SR / 2 / 1000)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Mean log-spectral error (dB)")
    ax.set_title("EnCodec reconstruction error across frequency")
    ax.legend(title="Bitrate (kbps)", ncol=2)
    ax.text(0.98, 0.04, f"$n = {n}$ files", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="0.4")
    save_fig(fig, "spectral_error_profile", fig_dir=out_dir)
    plt.close(fig)
    print(f"[profile] {out_dir}/spectral_error_profile  (n={n}, bitrates={bitrates})")


# ---------------------------------------------------------------------------
# Convergence check (justify n)
# ---------------------------------------------------------------------------

def convergence(items, bitrate, checkpoints, out_dir, seed=42):
    apply_paper_style()
    items = list(items)
    random.Random(seed).shuffle(items)             # representative prefixes
    cps = sorted(c for c in checkpoints if c <= len(items))
    fk = _freqs_khz()
    err = torch.zeros(len(fk))
    snaps, n = {}, 0
    for it in items:
        try:
            o, r = roundtrip(load_mono(it.path, SR), bitrate)
            e = _per_bin_db_err(o, r)
        except Exception:
            continue
        err += e
        n += 1
        if n in cps:
            snaps[n] = (err / n).clone()
    if n not in snaps:
        snaps[n] = (err / n).clone()

    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.52))
    for i, (k, v) in enumerate(sorted(snaps.items())):
        ax.plot(fk, v.numpy(), color=PALETTE[i % len(PALETTE)], label=f"{k}")
    ax.set_xlim(0, SR / 2 / 1000)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Mean log-spectral error (dB)")
    ax.set_title(f"Error-profile convergence ({bitrate:g} kbps)")
    ax.legend(title="files sampled", ncol=2)
    save_fig(fig, "spectral_error_convergence", fig_dir=out_dir)
    plt.close(fig)
    print(f"[convergence] {out_dir}/spectral_error_convergence  (snaps={sorted(snaps)})")


# ---------------------------------------------------------------------------
# Band-limit (keep generating from the 48 kHz subset)
# ---------------------------------------------------------------------------

def bandlimit(items, bw_mid, out_dir):
    apply_paper_style()
    so_acc = sr_acc = None
    f48 = f24 = None
    n = 0
    for it in items:
        try:
            x24 = load_mono(it.path, SR)
            _, rec = roundtrip(x24, bw_mid)
            f48, s_o = avg_mag_spectrum(load_mono(it.path, 48000), sr=48000)
            f24, s_r = avg_mag_spectrum(rec, sr=SR)
        except Exception:
            continue
        so_acc = s_o if so_acc is None else so_acc + s_o
        sr_acc = s_r if sr_acc is None else sr_acc + s_r
        n += 1
    so = 20 * np.log10(so_acc / n + EPS)
    sr_ = 20 * np.log10(sr_acc / n + EPS)
    ref = so.max()
    fig, ax = plt.subplots(figsize=figsize(5.0, ratio=0.6))
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
    print(f"[bandlimit] {out_dir}/bandlimit  (n={n})")


# ---------------------------------------------------------------------------

def _load(cfg_path, n):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"][0]
    ds = {**ds, "sample_per_class": max(1, round(n / 23)), "limit": n}
    return load_items(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["profile", "convergence", "bandlimit"],
                    default="profile")
    ap.add_argument("--config", default="configs/bsd10k_local.yaml")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bitrates", type=float, nargs="+",
                    default=[1.5, 3.0, 6.0, 12.0, 24.0])
    ap.add_argument("--conv-bitrate", type=float, default=6.0)
    ap.add_argument("--conv-checkpoints", type=int, nargs="+",
                    default=[50, 100, 200, 400])
    args = ap.parse_args()

    if args.mode == "profile":
        error_profile(_load(args.config, args.n), args.bitrates, OUT)
    elif args.mode == "convergence":
        n = max(args.conv_checkpoints)
        convergence(_load(args.config, n), args.conv_bitrate,
                    args.conv_checkpoints, OUT)
    else:
        error_profile_items = _load(args.config, args.n)
        bandlimit(error_profile_items, args.bitrates[len(args.bitrates) // 2], OUT)


if __name__ == "__main__":
    main()
