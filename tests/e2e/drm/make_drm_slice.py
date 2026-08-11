"""Regenerate the git-ignored gate asset tests/artifacts/assets/DRM/dw_modeb.cf32:
a real off-air Deutsche Welle DRM recording as 12 kHz complex baseband.

Source: "DW_ModeB_10kHz.flac" from the Dream decoder's published sample set
(https://sourceforge.net/projects/drm/files/samples/DRM%20sample%20recordings/) —
Robustness Mode B, spectrum occupancy 3 (10 kHz), mono 48 kHz real-IF with the DRM
signal centred at a 12 kHz IF, ~44 s. Not committed (.gitignore: artifacts/);
download it and place it at SRC, then run:
    uv run python tests/e2e/drm/make_drm_slice.py

Conversion: mix the 12 kHz IF to DC, low-pass (127-tap Hamming-windowed sinc,
6 kHz cutoff), decimate x4 -> 12 kHz complex64 — the baseband tests/e2e/drm/_drm.py's
ofdm_coherent_sync expects."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "DRM"
SRC = _DIR / "DW_ModeB_10kHz.flac"
OUT = _DIR / "dw_modeb.cf32"

IF_HZ = 12_000.0
LPF_HZ = 6_000.0
TAPS = 127
DECIMATION = 4


def _read_mono_pcm16(flac: Path) -> tuple[np.ndarray, int]:
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "src.wav"
        subprocess.run(["sndfile-convert", str(flac), str(wav_path)], check=True)
        with wave.open(str(wav_path), "rb") as w:
            fs = w.getframerate()
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float64) / 32768.0, fs


def _lowpass_taps(cutoff_hz: float, fs: int, n_taps: int) -> npt.NDArray[np.float64]:
    mid = (n_taps - 1) / 2
    h = np.sinc(2 * (cutoff_hz / fs) * (np.arange(n_taps) - mid))
    h *= np.hamming(n_taps)
    taps: npt.NDArray[np.float64] = h / h.sum()
    return taps


def _to_baseband(x: np.ndarray, fs: int) -> np.ndarray:
    n = np.arange(len(x))
    mixed = x * np.exp(-1j * 2 * np.pi * IF_HZ * n / fs)
    filtered = np.convolve(mixed, _lowpass_taps(LPF_HZ, fs, TAPS), "same")
    return filtered[::DECIMATION].astype(np.complex64)


def main() -> None:
    pcm, fs = _read_mono_pcm16(SRC)
    baseband = _to_baseband(pcm, fs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    baseband.tofile(OUT)
    print(f"wrote {OUT} ({baseband.size} samples @ {fs // DECIMATION} Hz)")


if __name__ == "__main__":
    main()
