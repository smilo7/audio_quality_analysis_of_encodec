"""
analysis/probes.py

Targeted artifact probes on controlled signals, each producing a thesis figure
and a printed quantitative summary:

    sine        log sweep + single tones -> frequency response, HF rolloff,
                and spurious energy above the fundamental
    harmonics   is that spurious energy harmonically locked to the input? peak
                prominence at k*f0 against a non-harmonic control baseline
    transient   percussive material (amen break) -> transient attenuation/smearing
    preecho     sharp onset after silence -> MP3-style pre-echo (energy before onset)

Run:
    python -m analysis.probes                 # all probes at 6 kbps
    python -m analysis.probes --probe sine --bitrates 1.5 12
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from encodec_eval.roundtrip import roundtrip, load_mono
from .common import (
    SR, RESULTS, apply_paper_style, save_fig, figsize, PALETTE,
    stft_mag, plot_spectrogram, avg_mag_spectrum, rms_envelope, local_env,
)

_REPO = Path(__file__).resolve().parents[1]
FIG_DIR = RESULTS / "probes" / "figures"
AMEN = _REPO / "legacy" / "scratch_audio" / "amen-break.wav"
TRANSIENT_DIR = _REPO / "datasets" / "transients"   # drop dry castanets/cymbals here


# ---------------------------------------------------------------------------
# Signal synthesis
# ---------------------------------------------------------------------------

def log_sweep(f0=20.0, f1=11500.0, dur=4.0, amp=0.5):
    t = torch.arange(int(dur * SR)) / SR
    L = math.log(f1 / f0)
    phase = 2 * math.pi * f0 * dur / L * (torch.exp(t / dur * L) - 1.0)
    return (amp * torch.sin(phase)).unsqueeze(0)


def tone(freq, dur=1.0, amp=0.5):
    t = torch.arange(int(dur * SR)) / SR
    return (amp * torch.sin(2 * math.pi * freq * t)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Probe 1: sine response (rolloff + hallucinated harmonics)
# ---------------------------------------------------------------------------

def _spectrogram_grid(x, bitrates, name, title_in, fmax=None):
    """Input + one EnCodec-output spectrogram per bitrate, in a shared-y row."""
    n = 1 + len(bitrates)
    fig, axes = plt.subplots(1, n, figsize=figsize("text", ratio=1.5 / n),
                             sharey=True)
    plot_spectrogram(axes[0], x, title=title_in, fmax=fmax)
    for ax, bw in zip(axes[1:], bitrates):
        plot_spectrogram(ax, roundtrip(x, bw)[1], title=f"{bw:g} kbps", fmax=fmax)
    for ax in axes:
        ax.set_xlabel("")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.supxlabel("Time (s)", fontsize=8)
    save_fig(fig, name, fig_dir=FIG_DIR)
    plt.close(fig)


def probe_sine(bitrates=(6.0,)):
    apply_paper_style()
    f0 = 1000.0

    # (a) sweep spectrograms: input vs codec output, one panel per bitrate
    _spectrogram_grid(log_sweep(), bitrates, "sine_sweep_spectrogram",
                      "Input sweep")

    # (b) constant-tone spectrograms (spurious harmonics as steady horizontal
    #     lines; coding noise floor over time), one panel per bitrate
    _spectrogram_grid(tone(f0, dur=2.0), bitrates, "sine_tone_spectrogram",
                      f"Input {f0/1000:g} kHz tone")

    # (c) single 1 kHz tone: output spectrum shows fundamental + spurious harmonics
    x = tone(f0)
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.5))
    freqs, mag_in = avg_mag_spectrum(x)
    ax.plot(freqs / 1000, 20 * np.log10(mag_in / mag_in.max() + 1e-9),
            color="0.6", lw=1.0, label="Input")
    ratios = {}
    for i, bw in enumerate(bitrates):
        _, rec = roundtrip(x, bw)
        freqs, mag = avg_mag_spectrum(rec)
        mag_db = 20 * np.log10(mag / mag.max() + 1e-9)
        ax.plot(freqs / 1000, mag_db, color=PALETTE[i % len(PALETTE)],
                label=f"{bw:g} kbps")
        ratios[bw] = _spurious_ratio_db(freqs, mag, f0)
    for k in range(2, 9):  # mark harmonic positions within band
        if k * f0 < SR / 2:
            ax.axvline(k * f0 / 1000, color="0.85", lw=0.6, zorder=0)
    ax.set_xlim(0, SR / 2 / 1000)
    ax.set_ylim(-90, 3)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB, peak-normalised)")
    ax.set_title(f"Reconstruction of a {f0/1000:g} kHz tone")
    ax.legend()
    save_fig(fig, "sine_harmonics", fig_dir=FIG_DIR)
    plt.close(fig)

    print("[sine] spurious-to-fundamental ratio (dB, higher = more hallucinated energy):")
    for bw, r in ratios.items():
        print(f"        {bw:>5g} kbps : {r:6.1f} dB")
    print(f"[sine] figures -> {FIG_DIR}")


# ---------------------------------------------------------------------------
# Probe 1b: are the spurious tones genuine harmonics, or just noise peaks?
# ---------------------------------------------------------------------------

_HN_FFT, _HN_HOP = 8192, 2048
HARM_F0S = (220.0, 440.0, 587.0, 880.0, 1000.0, 1500.0, 2000.0, 3000.0)


def _fine_spectrum(x):
    """Time-averaged magnitude spectrum at high frequency resolution (~3 Hz)."""
    w = torch.hann_window(_HN_FFT)
    S = torch.stft(x.reshape(-1), n_fft=_HN_FFT, hop_length=_HN_HOP, window=w,
                   center=True, return_complex=True).abs()
    return np.fft.rfftfreq(_HN_FFT, 1 / SR), S.mean(dim=-1).numpy()


def _peak_db(freqs, mag, f, half=60.0):
    return 20 * np.log10(mag[np.abs(freqs - f) <= half].max() + 1e-12)


def _floor_db(freqs, mag, f, near=60.0, span=400.0):
    """Median level of the neighbourhood around f, excluding f itself."""
    sel = (np.abs(freqs - f) <= span) & (np.abs(freqs - f) > near)
    return 20 * np.log10(np.median(mag[sel]) + 1e-12)


def probe_harmonics(bitrates=(1.5, 6.0, 12.0), f0=1000.0, kmax=10, fmax=11000.0):
    """Test whether spurious energy is harmonically locked to the input tone.

    For each harmonic k*f0 we measure *prominence*: the peak level in a narrow
    window minus the median of the surrounding neighbourhood. Taking a max
    against a median is biased upward even for pure noise, so the same quantity
    is measured at (k+0.5)*f0 -- never an integer multiple -- to establish the
    null baseline. Excess prominence over that baseline is the real evidence.

    Also writes an inspection spectrogram with the harmonic positions marked.
    """
    apply_paper_style()

    # (a) inspection spectrogram: input + one panel per bitrate, harmonics marked
    x = tone(f0, dur=2.0)
    n = 1 + len(bitrates)
    fig, axes = plt.subplots(1, n, figsize=figsize("text", ratio=0.42),
                             sharey=True)
    panels = [("Input", x)] + [(f"{bw:g} kbps", roundtrip(x, bw)[1])
                               for bw in bitrates]
    for ax, (name, w) in zip(axes, panels):
        plot_spectrogram(ax, w, fmax=8000, top_db=80, title=name)
        for k in range(2, kmax + 1):
            if k * f0 < 8000:
                ax.axhline(k * f0 / 1000, color="w", lw=0.4, ls=":", alpha=0.55)
        ax.set_ylabel("Frequency (kHz)" if ax is axes[0] else "")
    # the dotted harmonic markers are explained in the LaTeX caption
    fig.suptitle(f"Reconstruction of a {f0/1000:g} kHz tone")
    save_fig(fig, "sine_harmonic_spectrogram", fig_dir=FIG_DIR)
    plt.close(fig)

    # (b) prominence + level measurement over a set of input frequencies
    print("[harmonics] prominence over local floor, mean across "
          f"{len(HARM_F0S)} input tones (excess over non-harmonic control):")
    ks = list(range(2, kmax + 1))
    lvl_curves, exc_curves, bases = {}, {}, {}
    for bw in bitrates:
        harm = {k: [] for k in ks}
        rel = {k: [] for k in ks}
        ctrl = []
        for f in HARM_F0S:
            _, rec = roundtrip(tone(f, dur=2.0), bw)
            fr, mg = _fine_spectrum(rec)
            fund = _peak_db(fr, mg, f)
            for k in ks:
                if k * f <= fmax:
                    harm[k].append(_peak_db(fr, mg, k * f) - _floor_db(fr, mg, k * f))
                    rel[k].append(_peak_db(fr, mg, k * f) - fund)
                fc = (k + 0.5) * f
                if fc <= fmax:
                    ctrl.append(_peak_db(fr, mg, fc) - _floor_db(fr, mg, fc))
        base, sd = np.mean(ctrl), np.std(ctrl)
        bases[bw] = (base, sd)
        lvl_curves[bw] = [np.mean(rel[k]) if rel[k] else np.nan for k in ks]
        exc_curves[bw] = [np.mean(harm[k]) - base if harm[k] else np.nan for k in ks]
        print(f"  {bw:g} kbps  (control {base:.1f} +/- {sd:.1f} dB):")
        for k in ks:
            if harm[k]:
                m = np.mean(harm[k])
                flag = "*" if m - base > 2 * sd else " "
                print(f"     k={k:<3} {m:5.1f} dB   excess {m - base:+5.1f} dB {flag}")

    # (c) summary figure: how loud each harmonic is, and whether it is real
    fig, axes = plt.subplots(1, 2, figsize=figsize("text", ratio=0.38))
    for i, bw in enumerate(bitrates):
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(ks, lvl_curves[bw], marker="o", ms=3, color=c,
                     label=f"{bw:g} kbps")
        axes[1].plot(ks, exc_curves[bw], marker="o", ms=3, color=c,
                     label=f"{bw:g} kbps")
    sd_max = max(sd for _, sd in bases.values())
    axes[1].axhspan(-2 * sd_max, 2 * sd_max, color="0.85", zorder=0)
    axes[1].axhline(0, color="0.4", lw=0.8, ls=":")
    axes[0].set_ylabel("Level relative to fundamental (dB)")
    axes[0].set_title("Harmonic level")
    axes[1].set_ylabel("Excess prominence (dB)")
    axes[1].set_title("Prominence above non-harmonic control")
    for ax in axes:
        ax.set_xlabel("Harmonic order $k$")
        ax.set_xticks(ks)
    axes[0].legend(fontsize=7)          # series are shared; one legend is enough
    fig.suptitle("Harmonic distortion of a reconstructed pure tone")
    save_fig(fig, "sine_harmonic_levels", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[harmonics] figures -> {FIG_DIR}")


def _spurious_ratio_db(freqs, mag, f0, tol=30.0):
    """Energy outside a narrow band around the fundamental, relative to it."""
    p = mag ** 2
    fund = (np.abs(freqs - f0) <= tol)
    e_fund = p[fund].sum()
    e_spur = p[~fund & (freqs > tol)].sum()
    return 10 * np.log10((e_spur + 1e-12) / (e_fund + 1e-12))


# ---------------------------------------------------------------------------
# Probe 1c: noise input — the counterpart to the sine probe
# ---------------------------------------------------------------------------

def _noise(kind="white", dur=2.0, amp=0.5, seed=0):
    """White or pink (1/f) noise, generated in the frequency domain so the
    target spectrum is exact."""
    n = int(dur * SR)
    g = torch.Generator().manual_seed(seed)
    if kind == "white":
        x = torch.randn(n, generator=g)
    else:
        spec = torch.randn(n // 2 + 1, generator=g, dtype=torch.cfloat)
        f = torch.fft.rfftfreq(n, 1 / SR)
        spec = spec / torch.sqrt(torch.clamp(f, min=f[1]))      # 1/f power
        spec[0] = 0
        x = torch.fft.irfft(spec, n=n)
    return (amp * x / x.abs().max()).unsqueeze(0)


def _oct_smooth(freqs, db, frac=1 / 12):
    """Fractional-octave moving average of a dB spectrum (readability only)."""
    out = np.empty_like(db)
    lo, hi = freqs * 2 ** (-frac / 2), freqs * 2 ** (frac / 2)
    il = np.searchsorted(freqs, lo, "left")
    ih = np.searchsorted(freqs, hi, "right")
    csum = np.concatenate([[0.0], np.cumsum(db)])
    for i in range(len(db)):
        a, b = il[i], max(ih[i], il[i] + 1)
        out[i] = (csum[b] - csum[a]) / (b - a)
    return out


def probe_noise(bitrates=(1.5, 6.0, 12.0), kinds=("white", "pink"), n_probe=300):
    """Does EnCodec impose tonal structure on noise, the way it adds harmonics
    to a tone?

    Two measures, both comparing the reconstruction against the *input* noise so
    that the noise's own peakiness is controlled for:

      flatness   Wiener entropy of the average spectrum (1 = ideal noise). A
                 drop means energy has been concentrated into peaks.
      prominence peak-minus-local-median at `n_probe` randomly chosen
                 frequencies, exactly the statistic used for the harmonic test.
                 Higher than the input means added tonal structure.
    """
    apply_paper_style()
    rng = np.random.default_rng(0)
    probe_f = rng.uniform(300.0, 10500.0, n_probe)

    for kind in kinds:
        x = _noise(kind)
        recs = {bw: roundtrip(x, bw)[1] for bw in bitrates}

        # (a) spectrograms, input + one per bitrate
        fig, axes = plt.subplots(1, 1 + len(bitrates),
                                 figsize=figsize("text", ratio=0.42), sharey=True)
        for ax, (name, w) in zip(axes, [("Input", x)] +
                                 [(f"{bw:g} kbps", recs[bw]) for bw in bitrates]):
            plot_spectrogram(ax, w, fmax=12000, top_db=60, title=name)
            ax.set_ylabel("Frequency (kHz)" if ax is axes[0] else "")
        fig.suptitle(f"Reconstruction of {kind} noise")
        save_fig(fig, f"noise_{kind}_spectrogram", fig_dir=FIG_DIR)
        plt.close(fig)

        # (b) average spectrum, input vs reconstructions. Raw noise spectra are
        # unreadable, so smooth over a fixed fractional-octave span.
        fig, ax = plt.subplots(figsize=figsize("text", ratio=0.42))
        fr, mg_in = _fine_spectrum(x)
        ref = 20 * np.log10(mg_in + 1e-12).max()
        ax.plot(fr / 1000, _oct_smooth(fr, 20 * np.log10(mg_in + 1e-12) - ref),
                color="0.6", lw=1.2, label="Input")
        for i, bw in enumerate(bitrates):
            _, mg = _fine_spectrum(recs[bw])
            ax.plot(fr / 1000, _oct_smooth(fr, 20 * np.log10(mg + 1e-12) - ref),
                    color=PALETTE[i % len(PALETTE)], lw=1.0, label=f"{bw:g} kbps")
        ax.set_xlim(0, 12)
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"Average spectrum, {kind} noise")
        ax.legend(fontsize=7)
        save_fig(fig, f"noise_{kind}_spectrum", fig_dir=FIG_DIR)
        plt.close(fig)

        # (c) numbers
        def flatness(w):
            _, m = _fine_spectrum(w)
            p = m.astype(np.float64) ** 2 + 1e-20
            band = (fr >= 300) & (fr <= 10500)
            return float(np.exp(np.log(p[band]).mean()) / p[band].mean())

        def prom(w):
            _, m = _fine_spectrum(w)
            return np.mean([_peak_db(fr, m, f) - _floor_db(fr, m, f)
                            for f in probe_f])

        print(f"[noise/{kind}] flatness (1 = ideal noise) | mean prominence at "
              f"{n_probe} random frequencies")
        print(f"    input      {flatness(x):8.3f} | {prom(x):6.1f} dB")
        for bw in bitrates:
            print(f"    {bw:>5g} kbps {flatness(recs[bw]):8.3f} | "
                  f"{prom(recs[bw]):6.1f} dB")
    print(f"[noise] figures -> {FIG_DIR}")


# ---------------------------------------------------------------------------
# Probe 2: transients (attenuation + smearing)
# ---------------------------------------------------------------------------

def probe_transient(bitrates=(1.5, 12.0), window_s=2.0):
    apply_paper_style()
    if not AMEN.exists():
        print(f"[transient] missing {AMEN}; skipping")
        return
    x = load_mono(AMEN)[..., : int(window_s * SR)]
    recons = {bw: roundtrip(x, bw)[1] for bw in bitrates}
    orig = roundtrip(x, bitrates[0])[0]

    # spectrograms: original vs the lowest bitrate (most revealing)
    bw_lo = bitrates[0]
    fig, axes = plt.subplots(2, 1, figsize=figsize("text", ratio=0.7), sharex=True)
    plot_spectrogram(axes[0], orig, fmax=12000, title="Original")
    plot_spectrogram(axes[1], recons[bw_lo], fmax=12000,
                     title=f"EnCodec, {bw_lo:g} kbps")
    axes[0].set_xlabel("")
    save_fig(fig, "transient_spectrogram", fig_dir=FIG_DIR)
    plt.close(fig)

    # energy envelope across bitrates + peak attenuation at onsets
    t_o, e_o = rms_envelope(orig)
    peaks = _local_peaks(e_o, min_rel=0.4)
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.42))
    ax.plot(t_o, e_o, color="0.3", lw=1.4, label="Original")
    print(f"[transient] {len(peaks)} onsets; mean peak attenuation by bitrate:")
    for i, bw in enumerate(bitrates):
        t_r, e_r = rms_envelope(recons[bw])
        n = min(len(e_o), len(e_r))
        ax.plot(t_r[:n], e_r[:n], color=PALETTE[i % len(PALETTE)], lw=1.0,
                label=f"{bw:g} kbps")
        atten = 20 * np.log10((e_r[peaks] + 1e-9) / (e_o[peaks] + 1e-9))
        print(f"        {bw:>5g} kbps : {atten.mean():+.2f} dB")
    ax.plot(t_o[peaks], e_o[peaks], "v", color="0.3", ms=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS amplitude")
    ax.set_title("Transient energy envelope")
    ax.legend(ncol=len(bitrates) + 1, loc="upper right")
    save_fig(fig, "transient_envelope", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[transient] figures -> {FIG_DIR}")


def unquantised_roundtrip(x):
    """Encoder -> continuous latents -> decoder, with NO vector quantisation.

    Isolates the quantiser from the rest of the autoencoder. Note this is not a
    lossless path: the encoder still downsamples 320x to a 75 Hz, 128-dim
    latent, so what survives here is attributable to the encoder-decoder pair
    rather than to the discrete codes.
    """
    from encodec_eval.roundtrip import get_processor, ENCODEC_SR
    proc = get_processor("cpu")
    lat, meta = proc.audio_to_latents(x, sample_rate=ENCODEC_SR)
    return proc.decode_latents_audio(lat[0], meta).reshape(1, -1)


def probe_unquantised(bitrates=(1.5, 6.0, 24.0), f0=1000.0, n_rand=200):
    """Is the spurious energy produced by the quantiser or by the autoencoder?

    Compares the standard quantised roundtrip against decoding straight from the
    continuous latents. If the artifacts survive without the quantiser, they are
    not quantisation error.
    """
    apply_paper_style()
    rng = np.random.default_rng(0)

    def prom(w, skip=None):
        fr, mg = _fine_spectrum(w)
        f = rng.uniform(400, 11000, n_rand)
        if skip is not None:
            f = f[np.abs(f - skip) > 200]
        return _prominence_at(fr, mg, f).mean()

    xt, xn = tone(f0, dur=2.0), _noise("white")
    un_t, un_n = unquantised_roundtrip(xt), unquantised_roundtrip(xn)

    x1 = tone(f0)
    print("[unquantised] spurious-to-fundamental ratio, "
          f"{f0/1000:g} kHz tone (dB)")
    print(f"    no quantiser : "
          f"{_spurious_ratio_db(*avg_mag_spectrum(unquantised_roundtrip(x1)), f0):6.1f}")
    for bw in bitrates:
        print(f"    {bw:>5g} kbps   : "
              f"{_spurious_ratio_db(*avg_mag_spectrum(roundtrip(x1, bw)[1]), f0):6.1f}")

    print(f"[unquantised] mean peak prominence at {n_rand} random frequencies (dB)")
    for name, x, un, skip in (("tone", xt, un_t, f0), ("white noise", xn, un_n, None)):
        cells = " ".join(f"{bw:g}k {prom(roundtrip(x, bw)[1], skip):5.1f}"
                         for bw in bitrates)
        print(f"    {name:<12} input {prom(x, skip):5.1f} | "
              f"no quantiser {prom(un, skip):5.1f} | {cells}")

    fig, axes = plt.subplots(1, 4, figsize=figsize("text", ratio=0.42), sharey=True)
    panels = [("Input", xt), ("No quantiser", un_t),
              (f"{bitrates[0]:g} kbps", roundtrip(xt, bitrates[0])[1]),
              (f"{bitrates[-1]:g} kbps", roundtrip(xt, bitrates[-1])[1])]
    for ax, (name, w) in zip(axes, panels):
        plot_spectrogram(ax, w, fmax=12000, top_db=80, title=name)
        ax.set_ylabel("Frequency (kHz)" if ax is axes[0] else "")
    fig.suptitle(f"Spurious tones persist without quantisation "
                 f"({f0/1000:g} kHz tone)")
    save_fig(fig, "unquantised_tone", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[unquantised] figure -> {FIG_DIR}")


def _prominence_at(fr, mg, freqs):
    return np.array([_peak_db(fr, mg, f) - _floor_db(fr, mg, f) for f in freqs])


def probe_spurious(bitrates=(1.5, 6.0, 12.0), kmax=10, fmax=11000.0, n_rand=200):
    """What kind of spurious energy does EnCodec add, for tonal and noise input?

    Three panels, answering three separate questions:
      (a) does it add tonal structure at all?  Peak prominence at randomly
          chosen frequencies, input vs output, for a tone and for white noise.
      (b) is that structure harmonic?  Distribution of the distance from each
          detected peak to the nearest integer multiple of f0, against the
          uniform distribution expected under no harmonic preference.
      (c) are the harmonics nonetheless boosted?  Prominence at k*f0 in excess
          of the random-frequency null.
    """
    apply_paper_style()
    from scipy.signal import find_peaks
    rng = np.random.default_rng(0)
    ks = list(range(2, kmax + 1))

    prom_in, prom_out = {}, {bw: {} for bw in bitrates}
    dist, excess, nulls = {bw: [] for bw in bitrates}, {}, {}

    # --- tonal input -------------------------------------------------------
    harm = {bw: {k: [] for k in ks} for bw in bitrates}
    null = {bw: [] for bw in bitrates}
    fr_in, mg_in = _fine_spectrum(tone(1000.0, dur=2.0))
    prom_in["Tone"] = _prominence_at(fr_in, mg_in,
                                     rng.uniform(400, fmax, n_rand)).mean()
    for bw in bitrates:
        vals = []
        for f0 in HARM_F0S:
            _, rec = roundtrip(tone(f0, dur=2.0), bw)
            fr, mg = _fine_spectrum(rec)
            for k in ks:
                if k * f0 <= fmax:
                    harm[bw][k].append(_peak_db(fr, mg, k * f0)
                                       - _floor_db(fr, mg, k * f0))
            rf = rng.uniform(400, fmax, n_rand)
            rf = rf[np.abs(rf - f0) > 200]
            p = _prominence_at(fr, mg, rf)
            null[bw].extend(p)
            vals.extend(p)
            # peak positions relative to the harmonic grid
            db = 20 * np.log10(mg + 1e-12)
            band = (fr > 300) & (fr < fmax) & (np.abs(fr - f0) > 100)
            pk, _ = find_peaks(db[band], prominence=6.0)
            ph = (fr[band][pk] / f0) % 1.0
            dist[bw].extend(np.minimum(ph, 1 - ph))
        prom_out[bw]["Tone"] = np.mean(vals)
        nulls[bw] = (np.mean(null[bw]), np.std(null[bw]))
        excess[bw] = [np.mean(harm[bw][k]) - nulls[bw][0] if harm[bw][k] else np.nan
                      for k in ks]

    # --- noise input -------------------------------------------------------
    xw = _noise("white")
    fr_n, mg_n = _fine_spectrum(xw)
    rf = rng.uniform(400, fmax, n_rand)
    prom_in["White noise"] = _prominence_at(fr_n, mg_n, rf).mean()
    for bw in bitrates:
        _, rec = roundtrip(xw, bw)
        fr, mg = _fine_spectrum(rec)
        prom_out[bw]["White noise"] = _prominence_at(fr, mg, rf).mean()

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize("text", ratio=0.42))
    kinds = ["Tone", "White noise"]
    w, xs = 0.2, np.arange(len(kinds))
    axes[0].bar(xs - 1.5 * w, [prom_in[k] for k in kinds], w,
                color="0.6", label="Input")
    for i, bw in enumerate(bitrates):
        axes[0].bar(xs + (i - 0.5) * w, [prom_out[bw][k] for k in kinds], w,
                    color=PALETTE[i % len(PALETTE)], label=f"{bw:g} kbps")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(kinds)
    axes[0].set_ylabel("Mean prominence (dB)")
    axes[0].set_title("Spurious tonal structure")
    axes[0].legend(fontsize=6)

    d = np.concatenate([dist[bw] for bw in bitrates])
    axes[1].hist(d, bins=20, range=(0, 0.5), density=True,
                 color=PALETTE[0], alpha=0.85)
    axes[1].axhline(2.0, color="0.3", lw=1.2, ls="--", label="Chance")
    axes[1].set_xlabel("Distance to nearest\nharmonic ($\\times f_0$)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Peak position")
    axes[1].legend(fontsize=6)

    for i, bw in enumerate(bitrates):
        axes[2].plot(ks, excess[bw], marker="o", ms=3,
                     color=PALETTE[i % len(PALETTE)], label=f"{bw:g} kbps")
    axes[2].axhline(0, color="0.3", lw=0.8, ls=":")
    axes[2].set_xticks(ks)
    axes[2].set_xlabel("Harmonic order $k$")
    axes[2].set_ylabel("Excess over null (dB)")
    axes[2].set_title("Harmonic boost")
    axes[2].legend(fontsize=6)
    fig.suptitle("Spurious energy added by EnCodec")
    save_fig(fig, "spurious_summary", fig_dir=FIG_DIR)
    plt.close(fig)

    print("[spurious] mean peak prominence at random frequencies (dB)")
    for k in kinds:
        print(f"  {k:<12} input {prom_in[k]:5.1f} | " +
              " ".join(f"{bw:g}k {prom_out[bw][k]:5.1f}" for bw in bitrates))
    print("[spurious] fraction of peaks within 5% of a harmonic (chance 0.100)")
    for bw in bitrates:
        dd = np.array(dist[bw])
        print(f"  {bw:>5g} kbps  {(dd < 0.05).mean():.3f}  (n={len(dd)} peaks)")
    print("[spurious] harmonic excess over random null")
    for bw in bitrates:
        b, s = nulls[bw]
        print(f"  {bw:>5g} kbps (null {b:.1f} +/- {s:.1f} dB): " +
              " ".join(f"k{k}={e:+.1f}" for k, e in zip(ks, excess[bw]) if k <= 7))
    print(f"[spurious] figure -> {FIG_DIR}")


DRUMLOOPS = TRANSIENT_DIR / "6499__rytmenpinnen__drumloops"
_AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff"}


def probe_transient_corpus(bitrates=(1.5, 6.0, 12.0), src=DRUMLOOPS,
                           n_files=None, max_s=8.0, min_onsets=3, seed=0):
    """Transient peak attenuation over a corpus rather than a single excerpt.

    The single-file probe gives 14 onsets from one drum break, too thin to
    support a claim. Here onsets are detected on each original's RMS envelope
    and the reconstruction is read at the same frame. Files yielding fewer than
    `min_onsets` strong onsets are skipped as insufficiently percussive.

    Onsets within a file are not independent, so the headline number is the mean
    over *files* of each file's mean attenuation; the pooled per-onset spread is
    reported alongside it but should not be read as n = number of onsets.

    `src` may be a directory of audio or a CSV manifest with a `subset_path`
    column (paths relative to the manifest's dataset root).
    """
    apply_paper_style()
    src = Path(src)
    if src.is_dir():
        paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in _AUDIO_EXT)
    else:
        import pandas as pd
        man = pd.read_csv(src)
        root = src.resolve().parents[2]
        paths = [root / p for p in man.subset_path]
        paths = [p for p in paths if p.exists()]
    if n_files and n_files < len(paths):
        idx = np.random.default_rng(seed).choice(len(paths), n_files, replace=False)
        paths = [paths[i] for i in sorted(idx)]

    per_onset = {bw: [] for bw in bitrates}
    per_file = {bw: [] for bw in bitrates}
    n_ok = 0
    for i, p in enumerate(paths, 1):
        try:
            x = load_mono(str(p))[..., : int(max_s * SR)]
        except Exception as e:
            print(f"  skip {p.name}: {e}", flush=True)
            continue
        if x.reshape(-1).numel() < SR // 2:
            continue
        _, e_o = rms_envelope(roundtrip(x, bitrates[0])[0])
        peaks = _local_peaks(e_o, min_rel=0.4)
        if len(peaks) < min_onsets:
            continue
        n_ok += 1
        for bw in bitrates:
            _, e_r = rms_envelope(roundtrip(x, bw)[1])
            k = peaks[peaks < min(len(e_o), len(e_r))]
            if not len(k):
                continue
            att = 20 * np.log10((e_r[k] + 1e-9) / (e_o[k] + 1e-9))
            per_onset[bw].extend(att)
            per_file[bw].append(att.mean())
        if i % 10 == 0:
            print(f"  {i}/{len(paths)} files ({n_ok} usable)", flush=True)

    print(f"\n[transient-corpus] {src.name}: {n_ok} usable files, "
          f"{len(per_onset[bitrates[0]])} onsets")
    print("  bitrate    per-file mean +/- sd      pooled median   frac < -1 dB")
    for bw in bitrates:
        f = np.array(per_file[bw])
        v = np.array(per_onset[bw])
        print(f"  {bw:>5g} kbps  {f.mean():+7.2f} +/- {f.std():4.2f} dB "
              f"(n={len(f)})   {np.median(v):+7.2f} dB   {(v < -1).mean():.2f}")

    fig, ax = plt.subplots(figsize=figsize("column", ratio=0.8))
    ax.boxplot([per_onset[bw] for bw in bitrates], showfliers=False,
               medianprops=dict(color=PALETTE[3]))
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xticklabels([f"{bw:g}" for bw in bitrates])
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("Onset peak attenuation (dB)")
    ax.set_title("Transient attenuation across a drum-loop corpus")
    save_fig(fig, "transient_corpus", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[transient-corpus] figure -> {FIG_DIR}")


def _local_peaks(e, min_rel=0.4, guard=8):
    thr = min_rel * e.max()
    peaks = []
    for i in range(guard, len(e) - guard):
        if e[i] >= thr and e[i] == e[i - guard:i + guard + 1].max():
            if not peaks or i - peaks[-1] > guard:
                peaks.append(i)
    return np.array(peaks, dtype=int)


# ---------------------------------------------------------------------------
# Probe 3: pre-echo (energy before a sharp onset)
# ---------------------------------------------------------------------------

def _env_db(w):
    """Local Hann-RMS envelope in dB, peak-normalised."""
    t, e = local_env(w, win_ms=5.0, hop_ms=1.0)
    return t, 20 * np.log10(e / (e.max() + 1e-9) + 1e-9)


def _find_onset(w, rel=0.2):
    """First time the local envelope exceeds `rel` x its peak."""
    t, e = local_env(w, win_ms=3.0, hop_ms=0.5)
    idx = int(np.argmax(e >= rel * e.max()))
    return t[idx]


def _preecho_panel(ax, x, bitrates, onset_t, span=0.06, title=""):
    """Plot original + each bitrate's recon envelope (dB) around an onset."""
    t_o, db_o = _env_db(roundtrip(x, bitrates[0])[0])
    mo = (t_o >= onset_t - span) & (t_o <= onset_t + span)
    ax.plot(1000 * (t_o[mo] - onset_t), db_o[mo], color="0.3", lw=1.4, label="Original")
    res = {}
    for i, bw in enumerate(bitrates):
        t_r, db_r = _env_db(roundtrip(x, bw)[1])
        mr = (t_r >= onset_t - span) & (t_r <= onset_t + span)
        ax.plot(1000 * (t_r[mr] - onset_t), db_r[mr],
                color=PALETTE[i % len(PALETTE)], label=f"{bw:g} kbps")
        res[bw] = db_r[(t_r > onset_t - 0.02) & (t_r < onset_t)].mean()
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_ylim(-90, 3)
    ax.set_xlabel("Time relative to onset (ms)")
    ax.set_ylabel("Level (dB, peak-norm.)")
    if title:
        ax.set_title(title)
    return res


def probe_preecho(bitrates=(1.5, 12.0), sil=0.4, burst=0.12):
    """Synthetic control: silence -> abrupt broadband noise burst -> silence.
    A broadband transient (not a tone) is the canonical pre-echo stimulus and
    gives a ripple-free envelope."""
    apply_paper_style()
    n_sil, n_burst = int(sil * SR), int(burst * SR)
    g = torch.Generator().manual_seed(0)
    b = torch.randn(n_burst, generator=g)              # hard-onset white noise
    b = 0.5 * b / b.abs().max()
    x = torch.cat([torch.zeros(n_sil), b, torch.zeros(n_sil)]).unsqueeze(0)
    fig, ax = plt.subplots(figsize=figsize("text", ratio=0.45))
    res = _preecho_panel(ax, x, bitrates, onset_t=sil,
                         title="Pre-echo probe (synthetic noise burst)")
    ax.legend()
    save_fig(fig, "preecho", fig_dir=FIG_DIR)
    plt.close(fig)
    print("[preecho] synthetic pre-onset level (dB below peak):")
    for bw, v in res.items():
        print(f"        {bw:>5g} kbps : {v:6.1f} dB")
    print(f"[preecho] figure -> {FIG_DIR}")


def probe_preecho_files(files, bitrates=(1.5, 12.0), pad=0.3):
    """Pre-echo test on real percussive recordings (castanets, cymbals, ...).

    Each file is normalised and prepended with `pad` s of silence so there is a
    clean pre-onset region; the onset is found automatically and the codec's
    pre-onset energy is compared to the original (which is digital silence there).
    """
    apply_paper_style()
    files = [Path(f) for f in files]
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=figsize("text", ratio=0.34 + 0.06 * (n == 1)),
                             squeeze=False)
    print("[preecho-files] pre-onset level (dB below peak), per file:")
    for ax, f in zip(axes[0], files):
        x = load_mono(f)
        x = x / (x.abs().max() + 1e-9)
        x = torch.cat([torch.zeros(1, int(pad * SR)), x], dim=-1)
        onset_t = _find_onset(x)
        res = _preecho_panel(ax, x, bitrates, onset_t, title=f.stem)
        print(f"  {f.stem:<24} " +
              " ".join(f"{bw:g}k={v:5.1f}" for bw, v in res.items()))
    axes[0][-1].legend(fontsize=7)
    save_fig(fig, "preecho_real", fig_dir=FIG_DIR)
    plt.close(fig)
    print(f"[preecho-files] figure -> {FIG_DIR}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Audio export: write the probe stimuli and their reconstructions to disk
# ---------------------------------------------------------------------------

AUDIO_DIR = RESULTS / "probes" / "audio"


def export_probe_audio(bitrates=(1.5, 3.0, 6.0, 12.0, 24.0), f0=1000.0):
    """Write every probe stimulus and its reconstructions as WAV, for listening.

    Levels are preserved across the files in each group (no per-file
    normalisation) so that level differences between bitrates stay audible; a
    single shared scale factor is applied only if something would clip.
    """
    import soundfile as sf
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    groups = {
        "tone1k": tone(f0, dur=2.0),
        "sweep": log_sweep(),
        "noise_white": _noise("white"),
        "noise_pink": _noise("pink"),
    }
    if AMEN.exists():
        groups["drums"] = load_mono(AMEN)[..., : int(2.0 * SR)]

    for name, x in groups.items():
        versions = [("input", x)]
        versions += [(f"{bw:g}kbps", roundtrip(x, bw)[1]) for bw in bitrates]
        if name in ("tone1k", "noise_white"):
            versions.append(("unquantised", unquantised_roundtrip(x)))
        peak = max(float(w.abs().max()) for _, w in versions)
        scale = 1.0 / peak if peak > 1.0 else 1.0
        for tag, w in versions:
            p = AUDIO_DIR / f"{name}__{tag}.wav"
            sf.write(str(p), (w.reshape(-1) * scale).numpy(), SR, subtype="PCM_24")
        print(f"  {name:<12} {len(versions)} files, peak {peak:.2f}"
              + ("" if scale == 1.0 else f" (scaled by {scale:.3f})"))
    print(f"[audio] -> {AUDIO_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe",
                    choices=["sine", "harmonics", "spurious", "unquantised", "audio",
                             "noise", "transient",
                             "transient_corpus", "preecho", "preecho_real",
                             "all"],
                    default="all")
    ap.add_argument("--bitrates", type=float, nargs="+", default=[1.5, 12.0])
    ap.add_argument("--src", default=None,
                    help="dir of audio or CSV manifest for transient_corpus")
    ap.add_argument("--files", nargs="+", default=None,
                    help="audio files for preecho_real (default: datasets/transients/*)")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.probe in ("sine", "all"):
        probe_sine(args.bitrates)
    if args.probe in ("harmonics", "all"):
        probe_harmonics(args.bitrates)
    if args.probe in ("noise", "all"):
        probe_noise(args.bitrates)
    if args.probe in ("spurious", "all"):
        probe_spurious(args.bitrates)
    if args.probe in ("unquantised", "all"):
        probe_unquantised()
    if args.probe == "audio":
        export_probe_audio(args.bitrates)
    if args.probe in ("transient", "all"):
        probe_transient(args.bitrates)
    if args.probe == "transient_corpus":
        probe_transient_corpus(args.bitrates,
                               **({'src': args.src} if args.src else {}))
    if args.probe in ("preecho", "all"):
        probe_preecho(args.bitrates)
    if args.probe in ("preecho_real", "all"):
        files = args.files
        if not files and TRANSIENT_DIR.is_dir():
            files = sorted(str(p) for p in TRANSIENT_DIR.glob("*")
                           if p.suffix.lower() in {".wav", ".flac", ".aif", ".aiff"})
        if files:
            probe_preecho_files(files, args.bitrates)
        elif args.probe == "preecho_real":
            print(f"[preecho-files] no files; drop dry transients in {TRANSIENT_DIR}")


if __name__ == "__main__":
    main()
