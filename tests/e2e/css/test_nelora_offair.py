"""NELoRa-Bench off-air gates: real USRP N210 LoRa symbol captures with
per-symbol filename ground truth (see artifacts/assets/NELoRa/README.md).

The captures are concatenated symbol windows aligned from offset 0 — a bare
dechirp from sample 0 is the whole receive chain. Capture facts (LoRa SF7/SF10,
125 kHz bandwidth, 1 Msps => oversample 8) are the asset's SigMF ground truth
and live only in this e2e.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from marconi.mcp.tools import survey

_ASSET_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "NELoRa"
_SF7 = _ASSET_ROOT / "sf7_pkt0.cf32"
_FS = 1_000_000.0
_SF7_RATE = 125_000.0 / (1 << 7)  # 976.5625 Hz — below the old fs/1000 floor


@pytest.mark.skipif(not _SF7.exists(), reason="NELoRa asset absent")
def test_survey_default_floor_reaches_sf7_symbol_rate() -> None:
    # the regression this gate pins: a fixed fs/1000 floor (1000 Hz) hid the
    # 976.56 Hz line entirely; the span-derived default must search below it
    # and surface a candidate within the clock spectrum's own quantization
    out = survey(str(_SF7), _FS)
    sr = cast(dict[str, object], out["symbol_rate"])
    assert cast(float, sr["search_lo_hz"]) < _SF7_RATE
    res = cast(float, sr["clock_resolution_hz"])
    cands = cast(list[float], sr["candidates_hz"])
    assert min(abs(c - _SF7_RATE) for c in cands) < 2.0 * res, (cands, res)
