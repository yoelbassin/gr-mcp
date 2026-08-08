"""Regenerate the ADS-B off-air slice tests/e2e/adsb/test_adsb_offair.py decodes:
20 s of complex64 carved from a real 1090 MHz off-air recording,
adsb.2021-11-26T15_03_30_573.wav (2.4 Msps, 16-bit stereo PCM, channel 0 = I,
channel 1 = Q). Reproducibility script - adsb_20s.cf32 is already on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "ADS-B"
_WAV = _DIR / "adsb.2021-11-26T15_03_30_573.wav"
_OUT = _DIR / "adsb_20s.cf32"

# 20 s @ 2.4 Msps complex64 (8 bytes/sample) - the byte-length of the
# adsb_20s.cf32 already on disk; the fallback reference on a clean checkout
# where the wav exists but the slice hasn't been carved yet.
_KNOWN_BYTES = 384_000_000
_START_S = 0.0
_DURATION_S = 20.0


def main() -> None:
    if not _WAV.exists():
        raise FileNotFoundError(f"{_WAV} absent - source recording not present")
    rate, pcm = wavfile.read(_WAV, mmap=True)
    assert (
        pcm.ndim == 2 and pcm.shape[1] == 2
    ), f"expected interleaved stereo IQ, got shape {pcm.shape}"
    assert pcm.dtype == np.int16, f"expected 16-bit PCM, got dtype {pcm.dtype}"

    start = int(_START_S * rate)
    stop = start + int(_DURATION_S * rate)
    assert stop <= pcm.shape[0], (
        f"wav too short for a {_DURATION_S}s window: {pcm.shape[0]} frames "
        f"at {rate} Hz"
    )
    window = pcm[start:stop].astype(np.float32) / 32768.0
    sig = (window[:, 0] + 1j * window[:, 1]).astype(np.complex64)

    existing_bytes = _OUT.stat().st_size if _OUT.exists() else _KNOWN_BYTES
    _DIR.mkdir(parents=True, exist_ok=True)
    sig.tofile(_OUT)
    new_bytes = _OUT.stat().st_size

    print(f"wrote {_OUT}: {new_bytes} bytes ({sig.size} samples at {rate} Hz)")
    print(f"existing/reference: {existing_bytes} bytes")
    assert (
        new_bytes == existing_bytes
    ), f"regenerated {new_bytes} bytes != existing {existing_bytes} bytes"


if __name__ == "__main__":
    main()
