"""
encodec_eval/metrics.py

Per-file, reference-based metrics for an EnCodec roundtrip. All operate on two
aligned mono waveforms at the codec's native rate (24 kHz), shape [T] or [1, T]:
the original (reference) and the EnCodec reconstruction.

    lsd       multiscale log-spectral distance (dB)   — lower is better
    si_snr    scale-invariant signal-to-noise ratio (dB) — HIGHER is better
    mrstft    multi-resolution STFT log-magnitude distance — lower is better

LSD and MR-STFT are lifted from the BWE repo's evaluation/metrics_ref.py;
SI-SNR follows the standard scale-invariant definition (the SNR helper in the
old calculating_metrics.py was plain, scale-dependent SNR).
"""

import torch

EPS = 1e-8


def _flat(x):
    return x.reshape(-1).float()


def _align(enh, ref):
    enh, ref = _flat(enh), _flat(ref)
    t = min(enh.shape[-1], ref.shape[-1])
    return enh[:t], ref[:t]


# ---------------------------------------------------------------------------
# Log-spectral distance (single scale + multiscale wrapper)
# ---------------------------------------------------------------------------

def _lsd_single(enh, ref, n_fft, hop, eps=1e-10):
    win = torch.hann_window(n_fft, device=enh.device)
    kw = dict(n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win,
              center=True, return_complex=True)
    Se = torch.stft(enh, **kw).abs()
    Sr = torch.stft(ref, **kw).abs()
    le = 10 * torch.log10(Se ** 2 + eps)
    lr = 10 * torch.log10(Sr ** 2 + eps)
    per_frame = ((lr - le) ** 2).mean(dim=0).sqrt()   # mean over freq, per frame
    return per_frame.mean().item()                    # mean over time


def lsd(enh, ref, scales=(2048, 1024, 512), sr=24000):
    """Average LSD (dB) across several FFT sizes (hop = n_fft // 4)."""
    enh, ref = _align(enh, ref)
    vals = [_lsd_single(enh, ref, n, n // 4) for n in scales]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Scale-invariant SNR
# ---------------------------------------------------------------------------

def si_snr(enh, ref, sr=24000):
    """Scale-invariant SNR in dB (higher is better)."""
    enh, ref = _align(enh, ref)
    enh = enh - enh.mean()
    ref = ref - ref.mean()
    s_target = (torch.dot(enh, ref) / (torch.dot(ref, ref) + EPS)) * ref
    e_noise = enh - s_target
    return (10 * torch.log10(
        (s_target.pow(2).sum() + EPS) / (e_noise.pow(2).sum() + EPS))).item()


# ---------------------------------------------------------------------------
# Multi-resolution STFT (auraloss)
# ---------------------------------------------------------------------------

_MRSTFT = None


def mrstft(enh, ref, sr=24000):
    global _MRSTFT
    if _MRSTFT is None:
        import auraloss.freq as af
        _MRSTFT = af.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048], w_sc=0.0, w_log_mag=1.0,
            w_lin_mag=0.0, sample_rate=sr,
        )
    enh, ref = _align(enh, ref)
    return _MRSTFT(enh.view(1, 1, -1), ref.view(1, 1, -1)).item()


# ---------------------------------------------------------------------------
# Mel-domain spectral metrics (perceptual frequency weighting)
# ---------------------------------------------------------------------------

import math

_MEL = None
_MEL_MCD = None
_DCT = None


def _resample(x, sr, target):
    import torchaudio
    if sr == target:
        return x.reshape(-1)
    return torchaudio.functional.resample(x.reshape(1, -1), sr, target).reshape(-1)


def mel_l1(enh, ref, sr=24000):
    """L1 distance between log-mel spectrograms (perceptual frequency axis)."""
    global _MEL
    import torchaudio
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80, power=2.0)
    le = torch.log(_MEL(enh.reshape(-1)) + EPS)
    lr = torch.log(_MEL(ref.reshape(-1)) + EPS)
    t = min(le.shape[-1], lr.shape[-1])
    return (le[..., :t] - lr[..., :t]).abs().mean().item()


def mcd(enh, ref, sr=24000, n_mfcc=24, n_mels=40):
    """Mel-cepstral distortion (dB), c0 excluded. Lower is better.

    Mel-cepstra = orthonormal DCT of the *natural-log* mel spectrum (not
    torchaudio's dB-scaled MFCC, which would inflate the score), so the standard
    MCD scaling (10/ln10)*sqrt(2*sum_d (c_d-ĉ_d)^2) lands in the usual ~0–15 range.
    """
    global _MEL_MCD, _DCT
    import torchaudio
    if _MEL_MCD is None:
        _MEL_MCD = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=n_mels, power=2.0)
        _DCT = torchaudio.functional.create_dct(n_mfcc, n_mels, "ortho")  # [n_mels, n_mfcc]

    def cep(x):
        logmel = torch.log(_MEL_MCD(x.reshape(-1)) + EPS)        # [n_mels, T]
        return (logmel.transpose(0, 1) @ _DCT).transpose(0, 1)   # [n_mfcc, T]

    ce, cr = cep(enh), cep(ref)
    t = min(ce.shape[-1], cr.shape[-1])
    d = ce[1:, :t] - cr[1:, :t]                                  # drop energy (c0)
    per_frame = (2.0 * d.pow(2).sum(0)).sqrt()
    return ((10.0 / math.log(10)) * per_frame.mean()).item()


# ---------------------------------------------------------------------------
# Perceptual reference metrics (speech-validated; exploratory on non-speech)
# ---------------------------------------------------------------------------

def pesq_wb(enh, ref, sr=24000):
    """PESQ wideband (16 kHz), 1.0–4.5, higher better. NaN on failure."""
    from pesq import pesq as _pesq
    r = _resample(ref, sr, 16000).numpy()
    e = _resample(enh, sr, 16000).numpy()
    try:
        return float(_pesq(16000, r, e, "wb"))
    except Exception:
        return float("nan")


def stoi(enh, ref, sr=24000):
    """STOI intelligibility (0–1, higher better). NaN on failure."""
    from pystoi import stoi as _stoi
    r = _resample(ref, sr, 16000).numpy()
    e = _resample(enh, sr, 16000).numpy()
    try:
        return float(_stoi(r, e, 16000, extended=False))
    except Exception:
        return float("nan")


_CDPAM = None


def cdpam(enh, ref, sr=24000):
    """CDPAM learned perceptual distance (22.05 kHz). Lower better. NaN on failure."""
    global _CDPAM
    if _CDPAM is None:
        import cdpam as _c
        # cdpam 0.0.6 calls torch.load without weights_only; torch>=2.6 defaults
        # it to True and fails on the (trusted) packaged checkpoint. Patch just
        # for the model init.
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        try:
            _CDPAM = _c.CDPAM(dev="cpu")
        finally:
            torch.load = _orig

    def _fmt(w):
        return (_resample(w, sr, 22050) * 32768.0).round().reshape(1, -1)

    try:
        with torch.no_grad():
            d = _CDPAM.forward(_fmt(ref), _fmt(enh))
        return float(d.item())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_FUNCS = {
    "lsd": lsd, "si_snr": si_snr, "mrstft": mrstft,
    "mcd": mcd, "mel_l1": mel_l1,
    "pesq": pesq_wb, "stoi": stoi, "cdpam": cdpam,
}
_HIGHER_IS_BETTER = {"si_snr", "pesq", "stoi"}


def direction(name: str) -> str:
    return "higher" if name in _HIGHER_IS_BETTER else "lower"


def compute(enh, ref, which, sr=24000) -> dict:
    """`which` is a list of metric names from _FUNCS, e.g. [lsd, mcd, pesq]."""
    out = {}
    for name in which:
        fn = _FUNCS.get(name)
        if fn is not None:
            out[name] = fn(enh, ref, sr=sr)
    return out
