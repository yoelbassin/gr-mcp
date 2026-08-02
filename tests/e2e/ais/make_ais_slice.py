"""Regenerate the gitignored AIS IQ slice from the stable .sdriq source.

Run once before test_ais_offair.py:
    cd /Users/joel/Clones/gr-mcp-rebuild
    .venv/bin/python tests/e2e/make_ais_slice.py
The 60 s cf32 slice lands in the gitignored artifacts/ tree. (The slice is
reproducible; the GR *demod* of it is not — hence the threshold assertion.)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

SRC = Path(
    "/Users/joel/Clones/gr-mcp/artifacts/assets/ais.2021-05-04T11_30_31_640.sdriq"
)
# parents[3] of tests/e2e/ais/make_ais_slice.py is the repo root.
OUT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "assets"
    / "AIS"
    / "ais_60s.cf32"
)
SECONDS = 60.0


def main() -> None:
    head = SRC.read_bytes()[:32]
    rate = struct.unpack_from("<I", head, 0)[0]
    ssz = struct.unpack_from("<I", head, 20)[0]
    assert ssz == 24, ssz
    n = int(rate * SECONDS)
    mm = np.memmap(SRC, dtype=np.dtype("<i4"), mode="r", offset=32, shape=(n * 2,))
    iq = mm.astype(np.float32).reshape(-1, 2) / float(1 << 23)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (iq[:, 0] + 1j * iq[:, 1]).astype(np.complex64).tofile(OUT)
    print(f"wrote {OUT}  rate={rate}  seconds={SECONDS}")


if __name__ == "__main__":
    main()
