"""Crop the V1 IQ_2 capture (read-only) to a gitignored ~5 s cf32 frame window.

Run once before test_lora_offair.py:
    cd /Users/joel/Clones/gr-mcp-rebuild
    .venv/bin/python tests/bits/make_lora_slice.py

The cf32 slice lands in the gitignored artifacts/ tree. (The slice is
reproducible; the GR demod of it is not — hence the threshold assertion.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SRC = Path("/Users/joel/Clones/gr-mcp/artifacts/captures/IQ_2.sigmf-data")
# parents[2] of tests/bits/make_lora_slice.py is the repo root.
_OUT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "LoRa"
    / "iq2_frame.cf32"
)


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    # 1 Msps; preambles ~3.9/8.9/13.9/18.9 s; one frame ≈ 4.67 s. Window 3.7–9.0 s.
    window = np.fromfile(
        _SRC, dtype=np.complex64, count=int(5.3e6), offset=int(3.7e6) * 8
    )
    window.tofile(_OUT)
    print(f"wrote {_OUT} ({window.size} samples)")


if __name__ == "__main__":
    main()
