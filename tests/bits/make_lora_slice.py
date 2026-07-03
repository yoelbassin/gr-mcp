"""Crop the V1 IQ_2 capture (read-only) to a gitignored ~10 s cf32 window
holding TWO complete frames (issue 03: multi-burst decode must be exercised
by the off-air gate, not hidden by a hand-trimmed single frame).

Run once before test_lora_offair.py:
    cd /Users/joel/Clones/gr-mcp-rebuild
    .venv/bin/python tests/bits/make_lora_slice.py
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
    # 1 Msps; preambles ~3.9/8.9/13.9/18.9 s; preamble+header+payload ≈ 4.97 s,
    # so frame 2 ends ~13.87 s. Window 3.7–15.0 s = two complete frames plus a
    # tail that clears chirp_sync's withheld detector margin (frame 3's
    # preamble at ~13.9 s is included but its frame is truncated — harmless).
    window = np.fromfile(
        _SRC, dtype=np.complex64, count=int(11.3e6), offset=int(3.7e6) * 8
    )
    window.tofile(_OUT)
    print(f"wrote {_OUT} ({window.size} samples)")


if __name__ == "__main__":
    main()
