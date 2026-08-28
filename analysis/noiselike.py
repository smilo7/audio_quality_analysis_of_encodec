"""
analysis/noiselike.py

Why SI-SNR ranks the BST categories differently from the spectral metrics.

LSD / MR-STFT / log-mel L1 all compare *magnitude* spectrograms; SI-SNR is the
only metric computed on the waveform, so it is the only one that sees phase.
EnCodec's decoder is free to regenerate a plausible noise realisation rather
than reproduce the original one — the magnitude metrics score that as a near
match, SI-SNR scores it as near-total error. The variable that separates them
is therefore how noise-like the source is, and this module measures that
directly: per-file spectral flatness of the *reference* audio against each
metric, ignoring the class taxonomy entirely.

    scatter    flatness vs SI-SNR and vs LSD, one point per file, coloured by
               BST group, with binned-median trend and Spearman rho

Flatness is Wiener entropy (geometric / arithmetic mean of the power spectrum)
per frame, averaged over frames weighted by frame energy so that silence does
not dominate. 0 = pure tone, 1 = white noise.

Needs the audio locally (datasets/BSD10K); the per-file flatness is cached next
to the figures so re-plotting is free.

Run (from the repo root):
    python -m analysis.noiselike --csv results/bsd10k/per_file.csv --n 2000
    python -m analysis.noiselike --csv results/bsd10k/per_file.csv --bitrate 12
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from encodec_eval.roundtrip import load_mono
from .common import (SR, apply_paper_style, save_fig, figsize, stft_mag,
                     group_of, group_color, GROUP_ORDER)

N_FFT, HOP = 1024, 256
EPS = 1e-12
MANIFEST = Path("datasets/BSD10K/manifest.csv")
AUDIO_ROOT = Path("datasets/BSD10K")


def spectral_flatness(x):
    """Energy-weighted mean Wiener entropy of a [T] waveform. 0=tonal, 1=noise."""
    S = stft_mag(x, N_FFT, HOP) ** 2 + EPS              # [F, T] power
    gm = torch.exp(torch.log(S).mean(dim=0))            # geometric mean per frame
    am = S.mean(dim=0)                                  # arithmetic mean per frame
    w = am / am.sum()                                   # energy weight per frame
    return float(((gm / am) * w).sum())


def _flatness_table(stems, cache):
    """Per-file flatness for `stems` (sound_id -> relative audio path), cached."""
    have = pd.read_csv(cache, dtype={"stem": str}) if cache.exists() else \
        pd.DataFrame(columns=["stem", "flatness"]).astype({"stem": str})
    todo = stems[~stems.stem.isin(set(have.stem))]
    rows = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        try:
            x = load_mono(str(AUDIO_ROOT / r.link), SR)
        except Exception:
            continue
        if x.reshape(-1).numel() < N_FFT * 4:
            continue
        rows.append({"stem": r.stem, "flatness": spectral_flatness(x)})
        if i % 200 == 0:
            print(f"  flatness {i}/{len(todo)}", flush=True)
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(cache, index=False)
    return have


def scatter(df, bitrate, out_dir, metrics=("si_snr", "lsd")):
    apply_paper_style()
    d = df[df.bandwidth == bitrate]
    fig, axes = plt.subplots(1, len(metrics),
                             figsize=figsize("text", ratio=0.40), squeeze=False)
    for ax, metric in zip(axes.flatten(), metrics):
        sub = d[d[metric].notna()]
        ax.scatter(sub.flatness, sub[metric], s=3, alpha=0.35, linewidths=0,
                   c=[group_color(g) for g in sub.group])
        # median trend over equal-count (quantile) flatness bins, so the line
        # spans the whole range instead of thinning out in the sparse tail
        edges = np.unique(np.quantile(sub.flatness, np.linspace(0, 1, 13)))
        idx = np.clip(np.digitize(sub.flatness, edges) - 1, 0, len(edges) - 2)
        cx, cy = [], []
        for b in range(len(edges) - 1):
            sel = sub[idx == b]
            if len(sel) >= 10:
                cx.append(sel.flatness.median())
                cy.append(sel[metric].median())
        ax.plot(cx, cy, "k-", lw=1.4, zorder=3)
        rho, p = spearmanr(sub.flatness, sub[metric])
        # keep the outlier tail from squashing the bulk of the cloud
        lo, hi = np.percentile(sub[metric], [0.5, 99.5])
        pad = 0.04 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xscale("log")
        ax.set_ylabel(_label(metric))                # shared x label set below
        ax.set_title(f"{_label(metric).split(' ')[0]}   $\\rho={rho:+.2f}$")
    # sample size, bitrate and aggregation belong in the LaTeX caption
    groups = [g for g in GROUP_ORDER if g in set(d.group)]
    fig.legend(handles=[mpatches.Patch(color=group_color(g), label=g)
                        for g in groups],
               loc="outside right center", fontsize=7)
    fig.supxlabel("Spectral flatness of source (tonal $\\rightarrow$ noisy)")
    fig.suptitle("Source noise-likeness vs. reconstruction metrics")
    save_fig(fig, "noiselike_scatter", fig_dir=out_dir)
    plt.close(fig)


def _label(metric):
    return {"lsd": "LSD (dB) ↓", "si_snr": "SI-SNR (dB) ↑",
            "mrstft": "MR-STFT ↓", "mel_l1": "log-mel L1 ↓"}.get(metric, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/bsd10k/per_file.csv")
    ap.add_argument("--bitrate", type=float, default=6.0)
    ap.add_argument("--n", type=int, default=2000, help="files to sample (0 = all)")
    ap.add_argument("--metrics", nargs="+", default=["si_snr", "lsd"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.csv).resolve().parent / "figures"
    df = pd.read_csv(args.csv, dtype={"stem": str})
    man = pd.read_csv(MANIFEST, dtype={"sound_id": str}).rename(
        columns={"sound_id": "stem"})[["stem", "link"]]

    stems = df[["stem"]].drop_duplicates().merge(man, on="stem")
    if args.n:
        stems = stems.sample(min(args.n, len(stems)), random_state=args.seed)
    flat = _flatness_table(stems, out_dir / "noiselike_flatness.csv")

    df = df.merge(flat, on="stem")
    df["group"] = df.cls.map(group_of)
    scatter(df, args.bitrate, out_dir, tuple(args.metrics))

    d = df[df.bandwidth == args.bitrate]
    print(f"\nflatness vs metric, Spearman rho @ {args.bitrate:g} kbps "
          f"(n={d.stem.nunique()} files):")
    for m in [c for c in ("si_snr", "lsd", "mrstft", "mel_l1") if c in d.columns]:
        rho, p = spearmanr(d.flatness, d[m])
        print(f"  {m:7s} {rho:+.2f}  (p={p:.1e})")
    print(f"figure -> {out_dir}")


if __name__ == "__main__":
    main()
