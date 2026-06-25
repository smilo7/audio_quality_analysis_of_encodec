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


def lsd(enh, ref, scales=(2048, 1024, 512)):
    """Average LSD (dB) across several FFT sizes (hop = n_fft // 4)."""
    enh, ref = _align(enh, ref)
    vals = [_lsd_single(enh, ref, n, n // 4) for n in scales]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Scale-invariant SNR
# ---------------------------------------------------------------------------

def si_snr(enh, ref):
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
# Dispatch
# ---------------------------------------------------------------------------

_HIGHER_IS_BETTER = {"si_snr"}


def direction(name: str) -> str:
    return "higher" if name in _HIGHER_IS_BETTER else "lower"


def compute(enh, ref, which, sr=24000) -> dict:
    """`which` is a list of metric names, e.g. [lsd, si_snr, mrstft]."""
    out = {}
    if "lsd" in which:
        out["lsd"] = lsd(enh, ref)
    if "si_snr" in which:
        out["si_snr"] = si_snr(enh, ref)
    if "mrstft" in which:
        out["mrstft"] = mrstft(enh, ref, sr=sr)
    return out
