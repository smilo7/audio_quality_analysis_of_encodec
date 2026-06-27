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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/bsd_balanced/per_file.csv")
    ap.add_argument("--metric", default="lsd")
    ap.add_argument("--by", choices=["group", "cls"], default="group")
    ap.add_argument("--bitrate", type=float, default=None, help="default: max available")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    bitrate = args.bitrate or max(df.bandwidth.unique())
    out_dir = Path(args.csv).resolve().parent
    analyse(df, args.metric, args.by, bitrate, out_dir)


if __name__ == "__main__":
    main()
