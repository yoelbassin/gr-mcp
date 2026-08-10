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

    eyes = cast(list[float], symbol_rate["eye_openness"])
    assert len(eyes) == len(cands), (cands, eyes)

    if symbol_rate["eye_confirmed"]:  # re-rank fired: candidates_hz[0] is trusted
        assert abs(cands[0] - _BAUD) < 0.05 * _BAUD, (cands, eyes)

    envelope = cast(dict[str, object], out["envelope"])
    const_ratio = cast(float, envelope["const_envelope_ratio"])
    assert const_ratio < 0.1, const_ratio

    inst_freq = cast(dict[str, object], out["inst_freq"])
    spread_hz = cast(float, inst_freq["spread_hz"])
    assert spread_hz > 0.1 * _BAUD, spread_hz

    spectrum = cast(dict[str, object], out["spectrum"])
    bw = cast(float, spectrum["occupied_bw_hz"])
    assert 4_000.0 < bw < 16_000.0, bw

    # real off-air 4FSK: the M-th-power carrier block must NOT mislabel it PSK
    # (squaring a carrier-bearing FSK concentrates at order 2 — the jump rule is
    # what keeps psk_order null here instead of a false BPSK).
    carrier = cast(dict[str, object], out["carrier"])
    assert carrier["psk_order"] is None, carrier
