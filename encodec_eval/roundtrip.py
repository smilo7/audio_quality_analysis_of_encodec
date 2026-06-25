"""
encodec_eval/roundtrip.py

EnCodec 24 kHz encode->decode roundtrip. The "model under test" for the
artifact analysis is just the codec itself: original_24k -> codes -> recon_24k,
both compared at the codec's native rate.

Wraps the EncodecProcessor from wrapper_encodec.py (repo root). The processor is
cached per (device, bandwidth) so a bandwidth sweep reuses one model instance.
"""

import sys
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

ENCODEC_SR = 24000


@lru_cache(maxsize=4)
def get_processor(device: str = "cpu"):
    from wrapper_encodec import EncodecProcessor
    return EncodecProcessor(sr=ENCODEC_SR, device=device)


def load_mono(path, target_sr: int = ENCODEC_SR) -> torch.Tensor:
    """Load -> mono -> resample to the codec rate. Returns [1, T]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


@torch.no_grad()
def roundtrip(original_24k: torch.Tensor, bandwidth: float, device: str = "cpu"):
    """
    original_24k [1, T] -> EnCodec recon [1, T] at the same rate, trimmed to the
    shorter length so the pair is sample-aligned for metric computation.
    """
    proc = get_processor(device)
    codes, meta = proc.encode_audio_codes(original_24k, kbps=bandwidth,
                                           sample_rate=ENCODEC_SR)
    recon = proc.decode_codes_audio(codes, meta)
    if recon.dim() == 3:
        recon = recon.squeeze(0)
    recon = recon.cpu()
    t = min(original_24k.shape[-1], recon.shape[-1])
    return original_24k[..., :t], recon[..., :t]
