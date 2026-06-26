"""
encodec_eval/data.py

Resolve a dataset config into a list of (path, cls, stem) items. Audio lives
*outside* this repo (e.g. the BWE repo's data/), so configs point at it by
absolute path; nothing is copied here.

Two modes:
    manifest  read a CSV; each row -> (audio_root / row[path_col], row[class_col])
              (BSD balanced: path_col=link, class_col=cls)
    glob      glob audio_root for audio files; class taken from the filename
              prefix before `class_sep` (BSD heldout files: "<cls>_<id>.wav")
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg"}


class Item:
    __slots__ = ("path", "cls", "stem")

    def __init__(self, path, cls, stem):
        self.path, self.cls, self.stem = Path(path), cls, stem


def _from_manifest(ds: dict):
    root = Path(ds["audio_root"])
    path_col = ds.get("path_col", "link")
    class_col = ds.get("class_col", "cls")
    items = []
    with open(ds["manifest"]) as f:
        for row in csv.DictReader(f):
            p = root / row[path_col]
            items.append(Item(p, row.get(class_col, "unknown"), p.stem))
    return items


def _from_glob(ds: dict):
    root = Path(ds["audio_root"])
    sep = ds.get("class_sep", "_")
    items = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        cls = p.stem.split(sep)[0] if sep in p.stem else "unknown"
        items.append(Item(p, cls, p.stem))
    return items


def _sample_per_class(items, n, seed):
    """Up to `n` items per class, randomly (seeded) for representativeness."""
    rng = random.Random(seed)
    by_cls = defaultdict(list)
    for it in items:
        by_cls[it.cls].append(it)
    out = []
    for cls in sorted(by_cls):
        group = by_cls[cls]
        rng.shuffle(group)
        out.extend(group[:n])
    return out


def load_items(ds: dict):
    mode = ds.get("mode", "manifest")
    items = _from_manifest(ds) if mode == "manifest" else _from_glob(ds)
    missing = [i for i in items if not i.path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{len(items)} audio files in dataset '{ds['name']}' "
            f"do not exist (first: {missing[0].path}). Check audio_root.")
    spc = ds.get("sample_per_class")
    if spc:
        items = _sample_per_class(items, spc, ds.get("seed", 42))
    limit = ds.get("limit")
    return items[:limit] if limit else items
