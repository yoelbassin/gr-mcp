from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from marconi.mcp.tools import survey

_SLICE = (
    Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "DMR" / "dmr.cf32"
)
_FS = 39062.0
_BAUD = 4800.0  # off-air ground truth — DMR 4FSK, lives only in this e2e


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="DMR slice absent — run tests/e2e/dmr/make_dmr_slice.py",
)
def test_survey_recovers_dmr_parameters() -> None:
    out = survey(str(_SLICE), _FS)

    symbol_rate = cast(dict[str, object], out["symbol_rate"])
    cands = cast(list[float], symbol_rate["candidates_hz"])
    assert any(abs(c - _BAUD) < 0.05 * _BAUD for c in cands), cands

    inst_freq = cast(dict[str, object], out["inst_freq"])
    peaks = cast(list[float], inst_freq["peaks_hz"])
    assert 3 <= len(peaks) <= 6, peaks

    spectrum = cast(dict[str, object], out["spectrum"])
    bw = cast(float, spectrum["occupied_bw_hz"])
    assert 4_000.0 < bw < 16_000.0, bw
