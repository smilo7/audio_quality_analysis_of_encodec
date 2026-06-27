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

Crash-safety for long (multi-day) runs:
  * Atomic writes (write to .tmp then os.replace) so a crash mid-save cannot
    corrupt per_file.csv.
  * Rotating timestamped backups under output_dir/backups/ (keep last N), plus
    optional mirrors to other locations via eval.backup_dirs.
  * A clean snapshot under output_dir/snapshots/ after each bitrate completes.
  * progress.json (done/target/skipped/updated) so you can see where it's at.
  * skipped.txt listing any unreadable files (the run skips, never crashes, on a
    bad file).
  * Resume recovers from the newest backup if per_file.csv is ever unreadable.

Tunable via the eval block: checkpoint_every (default 100), backup_every (1000),
backup_keep (6), backup_dirs ([]).
"""

import argparse
import json
import os
import time
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


# ---------------------------------------------------------------------------
# Crash-safe IO
# ---------------------------------------------------------------------------

def _atomic_write_csv(df: pd.DataFrame, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)                     # atomic on the same filesystem


def _atomic_write_text(text: str, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _load_resume(per_file_path: Path, out_dir: Path, force: bool):
    """Load prior rows, falling back to the newest backup if the main CSV is
    missing/corrupt. Returns (DataFrame, set-of-done-keys)."""
    if force:
        return pd.DataFrame(columns=KEY), set()
    candidates = [per_file_path]
    bdir = out_dir / "backups"
    if bdir.is_dir():
        candidates += sorted(bdir.glob("per_file.*.csv"), reverse=True)
    for c in candidates:
        if not Path(c).exists():
            continue
        try:
            prev = pd.read_csv(c)
            if Path(c) != per_file_path:
                print(f"[resume] per_file.csv unreadable; recovered from {c}")
            keys = set(map(tuple, prev[KEY].astype(str).values.tolist()))
            return prev, keys
        except Exception as e:
            print(f"[resume] could not read {c}: {e}")
    return pd.DataFrame(columns=KEY), set()


def _checkpoint(rows, per_file_path, out_dir, backup_dirs, progress,
                *, backup=False, keep=6):
    df = pd.DataFrame(rows)
    _atomic_write_csv(df, per_file_path)                       # resume source
    _atomic_write_text(json.dumps(progress, indent=2), out_dir / "progress.json")
    if backup:
        bdir = out_dir / "backups"
        bdir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        _atomic_write_csv(df, bdir / f"per_file.{stamp}.csv")
        for old in sorted(bdir.glob("per_file.*.csv"))[:-keep]:
            old.unlink()
        for d in backup_dirs:                                 # extra locations
            d = Path(d)
            d.mkdir(parents=True, exist_ok=True)
            _atomic_write_csv(df, d / "per_file.csv")


def run(cfg: dict, force: bool = False):
    ev = cfg["eval"]
    device = ev.get("device", "cpu")
    sr = ev.get("sr", 24000)
    bandwidths = ev["bandwidths"]
    which = cfg["metrics"]
    out_dir = Path(ev["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = out_dir / "per_file.csv"
    ckpt_every = ev.get("checkpoint_every", 100)
    backup_every = ev.get("backup_every", 1000)
    backup_keep = ev.get("backup_keep", 6)
    backup_dirs = ev.get("backup_dirs", []) or []

    prev, done = _load_resume(per_file_path, out_dir, force)
    rows = [] if force else prev.to_dict("records")

    datasets_items = [(ds, load_items(ds)) for ds in cfg["datasets"]]
    target = sum(len(items) for _, items in datasets_items) * len(bandwidths)
    skipped = 0
    since_ckpt = since_backup = 0
    print(f"[resume] {len(rows)} rows done; target {target} "
          f"({max(target - len(rows), 0)} remaining)")

    def _progress(bw, dsname):
        return {"done": len(rows), "target": target, "skipped": skipped,
                "current_dataset": dsname, "current_bandwidth": bw,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S")}

    for ds, items in datasets_items:
        for bw in bandwidths:
            todo = [it for it in items
                    if (ds["name"], str(bw), str(it.stem)) not in done]
            if not todo:
                continue
            for it in tqdm(todo, desc=f"{ds['name']} @ {bw}kbps", leave=False):
                try:
                    orig = load_mono(it.path, target_sr=sr)
                    orig, recon = roundtrip(orig, bandwidth=bw, device=device)
                    vals = M.compute(recon, orig, which, sr=sr)
                except Exception as e:
                    tqdm.write(f"  [skip] {it.path}: {type(e).__name__}: {e}")
                    with open(out_dir / "skipped.txt", "a") as f:
                        f.write(f"{bw}\t{it.path}\t{type(e).__name__}: {e}\n")
                    skipped += 1
                    continue
                rows.append({"dataset": ds["name"], "bandwidth": bw,
                             "cls": it.cls, "stem": it.stem, **vals})
                since_ckpt += 1
                since_backup += 1
                if since_ckpt >= ckpt_every:
                    do_backup = since_backup >= backup_every
                    _checkpoint(rows, per_file_path, out_dir, backup_dirs,
                                _progress(bw, ds["name"]),
                                backup=do_backup, keep=backup_keep)
                    since_ckpt = 0
                    if do_backup:
                        since_backup = 0
            # bitrate complete: checkpoint + backup + clean stage snapshot
            _checkpoint(rows, per_file_path, out_dir, backup_dirs,
                        _progress(bw, ds["name"]), backup=True, keep=backup_keep)
            since_ckpt = since_backup = 0
            snap = out_dir / "snapshots"
            snap.mkdir(parents=True, exist_ok=True)
            _atomic_write_csv(pd.DataFrame(rows), snap / f"per_file_bw{bw:g}.csv")
            tqdm.write(f"[stage] {ds['name']} @ {bw} kbps done ({len(rows)} rows); "
                       f"snapshot + backup written")

    df = pd.DataFrame(rows)
    _atomic_write_csv(df, per_file_path)
    _write_summary(df, which, out_dir)
    print(f"\nWrote {len(df)} rows -> {per_file_path}  (skipped {skipped}; "
          f"see {out_dir / 'skipped.txt'})")
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
