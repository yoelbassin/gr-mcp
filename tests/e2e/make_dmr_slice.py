"""Regenerate artifacts/assets/DMR/dmr.cf32 from the DMR control-channel capture
(SDRSharp IQ .wav, 39062 Hz, 450.176 MHz, Motorola Capacity Plus). The source is
not committed (.gitignore artifacts/); place the downloaded wav at SRC, then run:
    uv run python tests/e2e/make_dmr_slice.py
Polarity is fixed here: this capture's sync pattern correlates at native polarity, so
the IQ is written un-conjugated. If the gate finds no syncs on a re-derived capture,
wrap the IQ in np.conjugate to flip every dibit's sign."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "DMR"
SRC = _DIR / "capture.wav"
OUT = _DIR / "dmr.cf32"


def main() -> None:
    w = wave.open(str(SRC), "rb")
    pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    scaled = pcm.astype(np.float32) / 32768.0
    iq = (scaled[0::2] + 1j * scaled[1::2]).astype(np.complex64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    iq.tofile(OUT)
    print(f"wrote {OUT} ({iq.size} samples)")


if __name__ == "__main__":
    main()
