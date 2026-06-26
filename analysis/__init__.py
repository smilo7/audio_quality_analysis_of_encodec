"""
analysis — thesis-ready figures characterising EnCodec's reconstruction
artifacts. Figures use the shared paper style (encodec_eval/plot_style.py) and
are written as vector PDF + PNG.

Modules:
    common.py    plot wiring, STFT/spectrogram helpers, BST class groups
    probes.py    targeted artifact probes (sine response, transients, pre-echo)
    spectral.py  per-frequency error profile + band-limit average spectrum
    dataset_figs.py  per-class bars + quality-vs-bitrate (from a per_file.csv)
"""
