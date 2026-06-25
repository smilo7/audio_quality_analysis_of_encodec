"""
encodec_eval/run.py

Config-driven runner. For each dataset x bandwidth x file it runs the EnCodec
roundtrip at 24 kHz, computes the reference metrics against the original, and
writes per-file and summary tables (overall and per sound class).

Usage:
    python -m encodec_eval.run --config configs/bsd_heldout.yaml
    python -m encodec_eval.run --config configs/bsd_balanced.yaml --force

Outputs (under eval.output_dir):
    per_file.csv   one row per (dataset, bandwidth, cls, file, metric...)
    summary.csv    mean/std per (dataset, bandwidth, cls) and per (dataset, bandwidth)
    summary.md     readable markdown version of summary.csv

Resumable: a re-run loads existing per_file.csv and skips rows already computed
(keyed by dataset+bandwidth+stem). Pass --force to recompute everything.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm.auto import tqdm

from . import metrics as M
from .data import load_items
from .roundtrip import load_mono, roundtrip

KEY = ["dataset", "bandwidth", "stem"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _done_keys(per_file_path: Path, force: bool):
    if force or not per_file_path.exists():
        return pd.DataFrame(columns=KEY), set()
    prev = pd.read_csv(per_file_path)
    keys = set(map(tuple, prev[KEY].astype(str).values.tolist()))
    return prev, keys


def run(cfg: dict, force: bool = False):
    ev = cfg["eval"]
    device = ev.get("device", "cpu")
    sr = ev.get("sr", 24000)
    bandwidths = ev["bandwidths"]
    which = cfg["metrics"]
    out_dir = Path(ev["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "per_file.csv"

    prev, done = _done_keys(per_file_path, force)
    rows = [] if force else prev.to_dict("records")

    for ds in cfg["datasets"]:
        items = load_items(ds)
        for bw in bandwidths:
            todo = [it for it in items
                    if (ds["name"], str(bw), str(it.stem)) not in done]
            if not todo:
                continue
            for it in tqdm(todo, desc=f"{ds['name']} @ {bw}kbps", leave=False):
                orig = load_mono(it.path, target_sr=sr)
                orig, recon = roundtrip(orig, bandwidth=bw, device=device)
                vals = M.compute(recon, orig, which, sr=sr)
                rows.append({"dataset": ds["name"], "bandwidth": bw,
                             "cls": it.cls, "stem": it.stem, **vals})
            pd.DataFrame(rows).to_csv(per_file_path, index=False)  # checkpoint

    df = pd.DataFrame(rows)
    _write_summary(df, which, out_dir)
    print(f"\nWrote {len(df)} rows -> {per_file_path}")
    print(f"Summary -> {out_dir / 'summary.csv'} and summary.md")
    return df


def _write_summary(df: pd.DataFrame, which, out_dir: Path):
    metric_cols = [m for m in which if m in df.columns]
    agg = {m: ["mean", "std"] for m in metric_cols}

    by_cls = (df.groupby(["dataset", "bandwidth", "cls"]).agg(agg)
              .round(4).reset_index())
    by_cls.columns = ["_".join(c).rstrip("_") for c in by_cls.columns]

    overall = (df.groupby(["dataset", "bandwidth"]).agg(agg)
               .round(4).reset_index())
    overall.insert(2, "cls", "ALL")
    overall.columns = list(by_cls.columns)

    summary = pd.concat([overall, by_cls], ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)

    arrows = " ".join(f"({m}: {'↑' if M.direction(m) == 'higher' else '↓'} better)"
                      for m in metric_cols)
    md = ["# EnCodec roundtrip — reference metrics\n",
          f"Direction: {arrows}\n",
          "Rows with cls=ALL are the dataset-wide aggregate; the rest are "
          "per sound class.\n",
          _to_markdown(summary)]
    (out_dir / "summary.md").write_text("\n".join(md))


def _to_markdown(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured table (avoids the optional `tabulate` dep)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
