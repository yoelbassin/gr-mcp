"""One-time golden generation: survey the 2 MHz band for ACARS channels, emit a
12.5 kHz mono AM-demodulated wav per candidate channel for acarsdec, and print the
JSON skeleton to fill from acarsdec's output. Throwaway numpy — deliberately not
Marconi code, so the golden is independent of the system under test.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_ASSETS = Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "ACRAS"
_CF32 = _ASSETS / "acars.cf32"
_RATE = 2_000_000
_AUDIO_RATE = 12_500  # acarsdec 3.7 `-f` requires exactly this input rate
_KNOWN_ACARS_HZ = [129.125e6, 130.025e6, 130.450e6, 131.550e6, 131.725e6]
_CENTER_HZ = 129.535e6


def band_peaks() -> list[float]:
    iq = np.fromfile(_CF32, dtype=np.complex64, count=_RATE)  # first second
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(iq.size, 1.0 / _RATE))
    kernel = np.ones(2048) / 2048.0
    smooth = np.convolve(spec, kernel, "same")
    floor = np.median(smooth)
    hot = freqs[smooth > 20 * floor]
    return sorted(set(round(float(f) / 1e3) * 1e3 for f in hot))


def _lowpass(x: np.ndarray, decim: int, ntaps: int = 129) -> np.ndarray:
    k = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(k / decim) * np.blackman(ntaps)
    return np.convolve(x, h / h.sum(), "same")[::decim]


def demod_channel(offset_hz: float, out_wav: Path) -> None:
    # chunked: the full capture is 1.3 GB of complex64; keep peak memory ~flat
    chunks: list[np.ndarray] = []
    chunk = 10 * _RATE
    with open(_CF32, "rb") as f:
        k = 0
        while True:
            iq = np.fromfile(f, dtype=np.complex64, count=chunk)
            if iq.size == 0:
                break
            n = np.arange(k, k + iq.size, dtype=np.float64)
            shifted = iq * np.exp(-2j * np.pi * offset_hz * n / _RATE).astype(
                np.complex64
            )
            baseband = _lowpass(_lowpass(shifted, 10), 4)  # 2 MHz -> 50 kHz
            chunks.append(np.abs(baseband).astype(np.float32))
            k += iq.size
    audio = np.concatenate(chunks)
    audio = audio - np.mean(audio)
    n_out = int(audio.size * _AUDIO_RATE / (_RATE / 40))
    audio48 = np.interp(
        np.linspace(0, audio.size - 1, n_out), np.arange(audio.size), audio
    )
    pcm = np.clip(audio48 / (np.max(np.abs(audio48)) + 1e-9), -1, 1)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_AUDIO_RATE)
        w.writeframes((pcm * 32000).astype(np.int16).tobytes())


if __name__ == "__main__":
    peaks = band_peaks()
    print("hot offsets (Hz from 129.535 MHz):", peaks)
    print("known ACARS channels in band:", [f - _CENTER_HZ for f in _KNOWN_ACARS_HZ])
    for off in peaks:
        out = _ASSETS / f"chan_{int(off):+d}.wav"
        demod_channel(float(off), out)
        print("wrote", out)
