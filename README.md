# Audio Quality Analysis of EnCodec

Reference-based benchmark of the audio-quality issues introduced by a plain
**EnCodec encode→decode roundtrip**, broken down by sound category. This is the
measurement tooling behind the thesis chapter *"Issues of Audio Quality in
Generative Audio Systems"*.

The "model under test" is the codec itself: `original_24k → codes → recon_24k`,
compared at EnCodec's native 24 kHz rate over a bandwidth sweep.

## Layout

```
encodec_eval/        the benchmark package
  metrics.py         LSD (multiscale), SI-SNR, MR-STFT
  roundtrip.py       EnCodec 24 kHz roundtrip (wraps wrapper_encodec.py)
  data.py            manifest / glob dataset loaders
  run.py             config-driven runner -> per_file.csv + summary.csv/.md
configs/             eval configs (point at datasets by absolute path)
results/             output tables and figures
wrapper_encodec.py   EncodecProcessor (shared with the BWE repo)
notebooks/           codec speed comparison (chapter "Choice of NAC")
legacy/              superseded scripts / scratch audio, kept for reference
```

Datasets live **outside this repo** (in the `encodec_bandwith_extension` repo's
`data/`) and are referenced in place — nothing is copied here.

## Usage

```bash
# fast sanity slice (19 files, ~1 per class)
python -m encodec_eval.run --config configs/bsd_heldout.yaml

# main per-category benchmark (2492 files, 23 classes, class-stratified)
python -m encodec_eval.run --config configs/bsd_balanced.yaml
```

Runs are **resumable**: a re-run skips `(dataset, bandwidth, file)` rows already
in `per_file.csv`; pass `--force` to recompute. Outputs land in
`eval.output_dir`:

- `per_file.csv` — one row per (dataset, bandwidth, cls, file, metric…)
- `summary.csv` / `summary.md` — mean/std per class and dataset-wide (`cls=ALL`)

## Metrics

| Metric  | Meaning | Direction |
|---------|---------|-----------|
| `lsd`    | multiscale log-spectral distance (dB) | lower better |
| `si_snr` | scale-invariant SNR (dB)              | higher better |
| `mrstft` | multi-resolution STFT log-mag distance | lower better |

See [TODO.md](TODO.md) for the planned next metrics (PESQ/STOI on speech,
FAD/KAD, and targeted artifact probes).
