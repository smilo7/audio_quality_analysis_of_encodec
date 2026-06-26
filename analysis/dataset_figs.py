"""
analysis/dataset_figs.py

Dataset-level figures from an encodec_eval per_file.csv:

    class_quality      per-class LSD bars at a fixed bitrate, coloured by BST
                       group, sorted worst->best (which sounds it fails on)
    quality_vs_bitrate LSD and SI-SNR vs bitrate, one line per BST group
                       (bitrate-quality tradeoff; speech bias)

Run:
    python -m analysis.dataset_figs --csv results/temp_subset/per_file.csv
    python -m analysis.dataset_figs --csv results/bsd_balanced/per_file.csv --bitrate 6
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from .common import (apply_paper_style, save_fig, figsize, PALETTE,
                     group_of, group_color, GROUP_ORDER)

ID_COLS = {"dataset", "bandwidth", "cls", "stem"}
HIGHER_BETTER = {"si_snr", "pesq", "stoi"}


def class_quality(df, bitrate, metric, out_dir):
    apply_paper_style()
    d = df[df.bandwidth == bitrate]
    by_cls = d.groupby("cls")[metric].mean().reset_index()
    by_cls["group"] = by_cls.cls.map(group_of)
    asc = metric not in HIGHER_BETTER           # LSD/MR-STFT: lower=better -> worst on right
    by_cls = by_cls.sort_values(metric, ascending=asc)

    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.5))
    colors = [group_color(c) for c in by_cls.cls]
    ax.bar(range(len(by_cls)), by_cls[metric], color=colors)
    ax.set_xticks(range(len(by_cls)))
    ax.set_xticklabels(by_cls.cls, rotation=90)
    ax.set_xlabel("BST class")
    ax.set_ylabel(_label(metric))
    ax.set_title(f"Per-class reconstruction quality @ {bitrate:g} kbps")
    groups = [g for g in GROUP_ORDER if g in set(by_cls.group)]
    ax.legend(handles=[mpatches.Patch(color=group_color(g), label=g) for g in groups],
              ncol=2, fontsize=7)
    save_fig(fig, f"class_quality_{metric}", fig_dir=out_dir)
    plt.close(fig)


def group_quality(df, bitrate, metric, out_dir):
    """Per top-level BST group bars (mean +/- std over the group's files)."""
    apply_paper_style()
    d = df[df.bandwidth == bitrate].copy()
    d["group"] = d.cls.map(group_of)
    agg = d.groupby("group")[metric].agg(["mean", "std"])
    agg = agg.reindex([g for g in GROUP_ORDER if g in agg.index])
    asc = metric not in HIGHER_BETTER
    agg = agg.sort_values("mean", ascending=asc)

    fig, ax = plt.subplots(figsize=figsize("column", ratio=0.85))
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"], capsize=3,
           color=[group_color(g) for g in agg.index])
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg.index, rotation=30, ha="right")
    ax.set_ylabel(_label(metric))
    ax.set_title(f"Quality by category @ {bitrate:g} kbps")
    save_fig(fig, f"group_quality_{metric}", fig_dir=out_dir)
    plt.close(fig)


def quality_vs_bitrate(df, out_dir):
    apply_paper_style()
    df = df.copy()
    df["group"] = df.cls.map(group_of)
    metrics = [("lsd", "Spectral distance (LSD)"),
               ("si_snr", "SI-SNR"),
               ("mrstft", "MR-STFT")]
    fig, axes = plt.subplots(1, 3, figsize=figsize("text", ratio=0.34))
    for ax, (metric, title) in zip(axes, metrics):
        for g in GROUP_ORDER:
            sub = df[df.group == g]
            if sub.empty:
                continue
            curve = sub.groupby("bandwidth")[metric].mean()
            ax.plot(curve.index, curve.values, marker="o", ms=3,
                    color=group_color(g), label=g)
        ovr = df.groupby("bandwidth")[metric].mean()
        ax.plot(ovr.index, ovr.values, "k--", lw=1.2, label="all")
        ax.set_xscale("log")
        ax.set_xticks(sorted(df.bandwidth.unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel("Bitrate (kbps)")
        ax.set_ylabel(_label(metric))
        ax.set_title(title)
    axes[-1].legend(fontsize=6, ncol=2)
    fig.suptitle("Reconstruction quality vs bitrate, by sound category")
    save_fig(fig, "quality_vs_bitrate", fig_dir=out_dir)
    plt.close(fig)


def _label(metric):
    return {"lsd": "LSD (dB) ↓", "si_snr": "SI-SNR (dB) ↑",
            "mrstft": "MR-STFT ↓", "mel_l1": "log-mel L1 ↓",
            "pesq": "PESQ ↑", "stoi": "STOI ↑",
            "mcd": "MCD ↓", "cdpam": "CDPAM ↓"}.get(metric, metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/temp_subset/per_file.csv")
    ap.add_argument("--bitrate", type=float, default=None,
                    help="bitrate for the per-class bars (default: max available)")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    out_dir = Path(args.csv).resolve().parent / "figures"
    bitrate = args.bitrate or max(df.bandwidth.unique())
    metric_cols = [c for c in df.columns if c not in ID_COLS]
    for metric in metric_cols:
        if df[metric].notna().any():
            class_quality(df, bitrate, metric, out_dir)
            group_quality(df, bitrate, metric, out_dir)
    quality_vs_bitrate(df, out_dir)
    print(f"dataset figures -> {out_dir} (metrics: {metric_cols} @ {bitrate:g} kbps)")


if __name__ == "__main__":
    main()
