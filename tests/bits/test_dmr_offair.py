"""Real off-air DMR closure gate: Marconi's full front-end (dc_block ->
channelize -> fsk -> mslice) turns the capture into hard dibits, the test
carves DMR framing (sync-from-sign, burst extraction, info-bit gather) as
caller data, and the generic BPTC helper (Task 4) decodes each burst. The
oracle is dsd-neo's independently CRC-confirmed subscriber-ID pairs.

mslice's code=[3,2,0,1] is chosen so each dibit's high bit equals the symbol
sign; bits[0::2] is therefore the per-symbol sign, and correlating it against
_dmr.SYNC_SIGNS reproduces the proven soft-sign sync detection from one GR run.
Polarity is fixed at asset-generation time (make_dmr_slice.py), so the gate does
not search it. All DMR values (sync pattern, thresholds, oracle pairs, field
offsets) are caller data in tests/; none enter src/marconi (test_agnosticism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bits import _dmr

from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
RATE = 39062.0
_SLICE = (
    Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "DMR" / "dmr.cf32"
)

ORACLE_PAIRS = {(3109836, 2247700), (3109823, 2247700), (3169855, 2247700)}


def _dmr_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=4800.0,
        path=[
            ModemStep(conv="dc_block", params={"dc_block_len": 32}),
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 12500.0, "center_hz": 0.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 1944.0, "loop_bw": 0.01}),
            ModemStep(
                conv="mslice",
                params={
                    "thresholds": [-0.667, 0.0, 0.667],
                    "code": [3, 2, 0, 1],
                    "bits": 2,
                },
            ),
        ],
    )


def _decode(bits: np.ndarray) -> list[dict[str, int | str]]:
    signs = 1 - 2 * bits[0::2].astype(np.int64)  # dibit high bit == symbol sign
    w = np.lib.stride_tricks.sliding_window_view(signs, _dmr.SYNC_SIGNS.size)
    syncs = np.flatnonzero((w == _dmr.SYNC_SIGNS).sum(axis=1) >= 21)
    out: list[dict[str, int | str]] = []
    for s in syncs:
        b0 = 2 * (int(s) - 54)
        if b0 < 0 or b0 + 264 > bits.size:
            continue
        r = _dmr.decode_burst_from_bits(bits[b0 : b0 + 264])
        if r:
            out.append(r)
    return out


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="DMR slice absent — run tests/bits/make_dmr_slice.py",
)
def test_dmr_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "dmr_bits.u8"
    pipe = compile_modem(
        _dmr_modem(),
        stage_registry(),
        direction="rx",
        sample_rate=RATE,
        start=IQ,
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=180.0)
    assert r.status == "ok", r

    decoded = _decode(read_bits(snk))
    pairs = {
        (int(d["source"]), int(d["target"]))
        for d in decoded
        if d["kind"] == "data_header" or d.get("csbko") == 61
    }
    assert ORACLE_PAIRS <= pairs, f"missing dsd pairs; got {sorted(pairs)}"
    assert len(decoded) >= 5, f"CRC-valid yield too low: {len(decoded)}"
    sources = {
        int(d["source"])
        for d in decoded
        if (int(d["source"]), int(d["target"])) in ORACLE_PAIRS
    }
    assert len(sources) >= 2
    for d in decoded:
        assert 0 < int(d["source"]) < (1 << 24)
        assert 0 < int(d["target"]) < (1 << 24)
