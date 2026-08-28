"""
analysis/stats.py

Significance testing for per-category metric differences, done correctly for
this data:

  * One bitrate at a time (rows within a bitrate are independent; the same file
    across bitrates is not -> never pool bitrates here).
  * Kruskal-Wallis omnibus per metric across categories (non-parametric: the
    metrics are skewed and heteroscedastic, so ANOVA assumptions fail).
  * Pairwise Mann-Whitney U with Holm correction for the "which pairs differ".
  * Effect sizes (eta-squared for the omnibus, Cliff's delta per pair), because
    with this many files p-values are tiny and not the interesting quantity.

Usage:
    python -m analysis.stats --csv results/bsd_balanced/per_file.csv \
        --metric lsd --by group --bitrate 6
    # --by cls for the 23 second-level classes; --bitrate omitted -> max available
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

from .common import group_of

HIGHER_BETTER = {"si_snr", "pesq", "stoi"}


def eta_squared_H(H, k, n):
    """Eta-squared for Kruskal-Wallis (Tomczak & Tomczak 2014): (H-k+1)/(n-k)."""
    return (H - k + 1) / (n - k)


def cliffs_delta(a, b):
    """Cliff's delta in [-1, 1]: P(a>b) - P(a<b). |d|: .11 small, .28 med, .43 large."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = np.sign(a[:, None] - b[None, :])
    return diff.sum() / (len(a) * len(b))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((m - rank) * p[idx], 1.0))
        adj[idx] = running
    return adj


def analyse(df, metric, by, bitrate, out_dir):
    d = df[(df.bandwidth == bitrate)].dropna(subset=[metric]).copy()
    d["grp"] = d.cls.map(group_of) if by == "group" else d.cls
    groups = {g: sub[metric].values for g, sub in d.groupby("grp") if len(sub) >= 2}
    names = sorted(groups)
    n = sum(len(v) for v in groups.values())

    H, p = kruskal(*groups.values())
    eta2 = eta_squared_H(H, len(names), n)
    print(f"\n=== {metric} @ {bitrate:g} kbps, by {by}  (n={n}, {len(names)} groups) ===")
    print(f"Kruskal-Wallis: H={H:.1f}, p={p:.2e}, eta^2={eta2:.3f} "
          f"({'higher=better' if metric in HIGHER_BETTER else 'lower=better'})")

    rows = []
    for a, b in itertools.combinations(names, 2):
        u, pu = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        rows.append({"a": a, "b": b, "median_a": np.median(groups[a]),
                     "median_b": np.median(groups[b]),
                     "cliffs_delta": cliffs_delta(groups[a], groups[b]), "p_raw": pu})
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p_raw.values)
    res["sig"] = res.p_holm < 0.05
    res = res.sort_values("cliffs_delta", key=np.abs, ascending=False)

    for c in ("median_a", "median_b", "cliffs_delta"):
        res[c] = res[c].round(3)
    res["p_holm"] = res.p_holm.map(lambda x: f"{x:.1e}")
    res["p_raw"] = res.p_raw.map(lambda x: f"{x:.1e}")
    print(res.to_string(index=False))

    out = out_dir / f"stats_{metric}_{by}_{bitrate:g}kbps.csv"
    res.to_csv(out, index=False)
    print(f"-> {out}")


def prepare(df, manifest=None, min_confidence=None, balance=None, cap=None, seed=42):
    """Optional analysis-time views on the per-file table:
      * min_confidence: keep only labels with confidence >= N (needs --manifest,
        joined on stem == sound_id).
      * balance group|class with a per-unit cap: random (seeded) subsample so each
        category (or sub-class) contributes at most `cap` files."""
    df = df.copy()
    if min_confidence is not None:
        if manifest is None:
            raise SystemExit("--min-confidence needs --manifest (for the confidence column)")
        m = pd.read_csv(manifest)[["sound_id", "confidence"]]
        m["sound_id"] = m.sound_id.astype(str)
        df["stem"] = df.stem.astype(str)
        df = df.merge(m, left_on="stem", right_on="sound_id", how="inner")
        df = df[df.confidence >= min_confidence]
    if balance:
        unit = df.cls.map(group_of) if balance == "group" else df.cls
        df = (df.assign(_u=unit)
                .groupby(["bandwidth", "_u"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), cap), random_state=seed))
                .drop(columns="_u"))
    return df


def metric_correlations(df, bitrate, out_dir):
    """Rank correlation between the metrics themselves, at one bitrate.

    Answers whether the metrics are measuring distinct things. Spearman rather
    than Pearson because the metrics are skewed and only their ordering is
    comparable. One bitrate at a time: the same file appears once per bitrate,
    so pooling would duplicate observations.
    """
    d = df[df.bandwidth == bitrate]
    cols = [c for c in ("lsd", "mel_l1", "mrstft", "si_snr", "pesq", "stoi")
            if c in d.columns and d[c].notna().any()]
    corr = d[cols].corr(method="spearman")
    n = len(d.dropna(subset=cols))
    print(f"\n=== metric Spearman correlations @ {bitrate:g} kbps "
          f"(n={n} files) ===")
    print(corr.round(2).to_string())
    mag = [c for c in ("lsd", "mel_l1", "mrstft") if c in cols]
    if len(mag) > 1:
        vals = [corr.loc[a, b] for i, a in enumerate(mag) for b in mag[i + 1:]]
        print(f"\nmagnitude metrics among themselves: "
              f"{min(vals):.2f} to {max(vals):.2f}")
    if "si_snr" in cols and mag:
        vals = [abs(corr.loc["si_snr", m]) for m in mag]
        print(f"si_snr against magnitude metrics: "
              f"{min(vals):.2f} to {max(vals):.2f}")
    path = Path(out_dir) / f"metric_corr_{bitrate:g}kbps.csv"
    corr.to_csv(path)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", action="store_true",
                    help="report metric-to-metric Spearman correlations and exit")
    ap.add_argument("--csv", default="results/bsd_balanced/per_file.csv")
    ap.add_argument("--metric", default="lsd")
    ap.add_argument("--by", choices=["group", "cls"], default="group")
    ap.add_argument("--bitrate", type=float, default=None, help="default: max available")
    ap.add_argument("--manifest", default=None, help="for --min-confidence (sound_id,confidence)")
    ap.add_argument("--min-confidence", type=int, default=None)
    ap.add_argument("--balance", choices=["group", "cls"], default=None)
    ap.add_argument("--cap", type=int, default=None, help="max files per unit when --balance")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = prepare(df, args.manifest, args.min_confidence, args.balance, args.cap, args.seed)
    bitrate = args.bitrate or max(df.bandwidth.unique())
    out_dir = Path(args.csv).resolve().parent
    if args.corr:
        metric_correlations(df, bitrate, out_dir)
        return
    analyse(df, args.metric, args.by, bitrate, out_dir)


if __name__ == "__main__":
    main()
