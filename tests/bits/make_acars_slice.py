"""One-time asset regeneration: SDRuno IQ wav -> cf32 (complex64) slice.

Provenance: artifacts/assets/ACRAS/SDRuno_20200908_152020Z_129535kHz.wav,
2ch int16 @ 2 MHz, 83.4 s, captured 2020-09-08 at 129.535 MHz.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

_ASSETS = Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "ACRAS"
_WAV = _ASSETS / "SDRuno_20200908_152020Z_129535kHz.wav"
_OUT = _ASSETS / "acars.cf32"
_CHUNK_FRAMES = 1 << 20


def convert(conjugate: bool) -> None:
    with wave.open(str(_WAV)) as w, open(_OUT, "wb") as out:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        remaining = w.getnframes()
        while remaining:
            n = min(remaining, _CHUNK_FRAMES)
            raw = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, 2)
            iq = (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64) / 32768.0
            if conjugate:
                iq = np.conj(iq)
            iq.tofile(out)
            remaining -= n


if __name__ == "__main__":
    convert(conjugate="--conjugate" in sys.argv)
    print(f"wrote {_OUT} ({_OUT.stat().st_size / 1e9:.2f} GB)")
