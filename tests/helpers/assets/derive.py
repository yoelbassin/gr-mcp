from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

Deriver = Callable[[Path, Path], None]


class DeriveError(Exception):
    pass


_POCSAG_RATE = 128000
_POCSAG_START_S, _POCSAG_END_S = 3.0, 10.0


def _pocsag_slice(src: Path, dst: Path) -> None:
    with wave.open(str(src)) as w:
        if (w.getnchannels(), w.getsampwidth()) != (2, 1):
            raise DeriveError(f"{src}: expected 8-bit interleaved IQ (2 channels)")
        if w.getframerate() != _POCSAG_RATE:
            raise DeriveError(f"{src}: expected {_POCSAG_RATE} Hz")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.uint8)
    # centred on 127.5 and left unscaled: the gate's amplitude contract is
    # calibrated against these magnitudes, and the manifest pins the digest
    iq = pcm.astype(np.float32).reshape(-1, 2) - 127.5
    z = (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64)
    want = int((_POCSAG_END_S - _POCSAG_START_S) * _POCSAG_RATE)
    z = z[int(_POCSAG_START_S * _POCSAG_RATE) : int(_POCSAG_END_S * _POCSAG_RATE)]
    if z.size != want:
        raise DeriveError(f"{src}: short source, got {z.size} of {want} samples")
    dst.parent.mkdir(parents=True, exist_ok=True)
    z.tofile(dst)


DERIVERS: dict[str, Deriver] = {"pocsag_slice": _pocsag_slice}
