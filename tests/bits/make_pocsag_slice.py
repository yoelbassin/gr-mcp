"""Reproduce the asset-gated POCSAG slice from its public sigidwiki source.

Downloads POCSAG_IQ.zip (real off-air pager traffic, 8-bit interleaved IQ WAV at
128 kHz) and writes t=3..10 s as raw complex64 — the slice the off-air gate
(tests/bits/test_pocsag_offair.py) decodes. Ground truth via multimon-ng 1.5.0.
Source: https://www.sigidwiki.com/wiki/POCSAG (File:POCSAG_IQ.zip)."""

from __future__ import annotations

import io
import urllib.request
import wave
import zipfile
from pathlib import Path

import numpy as np

URL = "http://www.sigidwiki.com/images/6/65/POCSAG_IQ.zip"
OUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "POCSAG"
    / "pocsag.cf32"
)
RATE = 128000
START_S, END_S = 3.0, 10.0


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(
        req, timeout=120
    ).read()  # noqa: S310 (documented host)
    zf = zipfile.ZipFile(io.BytesIO(raw))
    wav_name = next(n for n in zf.namelist() if n.lower().endswith(".wav"))
    with wave.open(io.BytesIO(zf.read(wav_name))) as w:
        assert w.getnchannels() == 2, "expected interleaved IQ (2 channels)"
        assert w.getsampwidth() == 1, "expected 8-bit PCM"
        assert w.getframerate() == RATE, f"expected {RATE} Hz"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8)
    iq = pcm.astype(np.float32).reshape(-1, 2) - 127.5
    sig = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    sig = sig[int(START_S * RATE) : int(END_S * RATE)]
    assert sig.size == int((END_S - START_S) * RATE), f"short download: {sig.size}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sig.tofile(OUT)
    print(f"wrote {OUT} ({sig.size} samples, {sig.size / RATE:.1f}s)")


if __name__ == "__main__":
    main()
