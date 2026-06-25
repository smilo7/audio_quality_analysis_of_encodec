"""
encodec_eval — reference-based benchmark of EnCodec's reconstruction artifacts.

This package characterises the audio-quality issues of a plain EnCodec
encode->decode roundtrip at its native operating rate (24 kHz), broken down by
sound category. It is the measurement tooling behind the thesis chapter
"Issues of Audio Quality in Generative Audio Systems".

Layout:
    metrics.py    reference metrics: LSD (multiscale), SI-SNR, MR-STFT
    roundtrip.py  EnCodec 24 kHz codec roundtrip (wraps wrapper_encodec)
    data.py       manifest / audio loading (data lives outside this repo)
    run.py        config-driven runner -> per_file.csv + summary.csv/.md

Run with:
    python -m encodec_eval.run --config configs/bsd_heldout.yaml
"""
