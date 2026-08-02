# tests/e2e/make_dab_slice.py
"""Crop the V1 DAB wav (read-only) to a gitignored ~1.5 s cf32 slice.
Run once:
    cd /Users/joel/Clones/gr-mcp-rebuild
    .venv/bin/python tests/e2e/make_dab_slice.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile

_SRC = Path(
    "/Users/joel/Clones/gr-mcp/artifacts/assets/dab.2021-12-16T14_26_44_664.wav"
)
_OUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "DAB"
    / "bbc_slice.cf32"
)


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _, data = wavfile.read(_SRC, mmap=True)
    seg = data[4_000_000 : 4_000_000 + 3_200_000].astype(np.float32)
    (seg[:, 0] + 1j * seg[:, 1]).astype(np.complex64).tofile(_OUT)
    print(f"wrote {_OUT} ({seg.shape[0]} samples)")


if __name__ == "__main__":
    main()
