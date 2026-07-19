"""Real off-air DMR closure gate: Marconi's front-end (channelize -> fsk) turns the
capture into soft symbols, and the test carves DMR framing as caller data. The capture
has a strong LO-leakage DC spike co-located with the baseband signal, so a streaming
DC-blocker would notch the signal itself; DC is removed by whole-signal mean subtraction
(thin capture prep) before the modem. The FM discriminator also has per-burst DC wander
that a memoryless global slicer cannot follow, so slicing is local: detect syncs on the
soft-symbol signs, then per-burst remove DC and rescale before the M-ary decision. The
generic BPTC helper (Task 4) decodes each 132-dibit burst; the oracle is dsd-neo's
independently CRC-confirmed subscriber-ID pairs.

Polarity is fixed at asset-generation time (make_dmr_slice.py), so the gate does not
search it. All DMR values (sync pattern, thresholds, oracle pairs, field offsets) are
caller data in tests/; none enter src/marconi (test_agnosticism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bits import _dmr

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
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 12500.0, "center_hz": 0.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 1944.0, "loop_bw": 0.01}),
        ],
    )


def _decode(sym: np.ndarray) -> list[dict[str, int | str]]:
    signs = np.sign(sym - np.median(sym))
    w = np.lib.stride_tricks.sliding_window_view(signs, _dmr.SYNC_SIGNS.size)
    syncs = np.flatnonzero((w == _dmr.SYNC_SIGNS).sum(axis=1) >= 21)
    out: list[dict[str, int | str]] = []
    for s in syncs:
        lo = int(s) - 54
        if lo < 0 or lo + 132 > sym.size:
            continue
        c = sym[lo : lo + 132] - np.median(sym[lo : lo + 132])
        mag = np.abs(c)
        outer = float(np.mean(mag[mag > np.percentile(mag, 60)]) or 1.0)
        n = c / outer
        dibits = np.array(
            [[3, 2, 0, 1][int(np.searchsorted([-0.667, 0.0, 0.667], v))] for v in n],
            np.uint8,
        )
        r = _dmr.decode_burst(dibits)
        if r:
            out.append(r)
    return out


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="DMR slice absent — run tests/bits/make_dmr_slice.py",
)
def test_dmr_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    iq = np.fromfile(_SLICE, np.complex64)
    iq = (iq - iq.mean()).astype(np.complex64)  # strip the LO-leakage DC spike
    src = tmp_path / "dmr_dc.cf32"
    src.write_bytes(iq.tobytes())
    snk = tmp_path / "dmr_sym.f32"
    pipe = compile_modem(
        _dmr_modem(),
        stage_registry(),
        direction="rx",
        sample_rate=RATE,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=180.0)
    assert r.status == "ok", r

    decoded = _decode(np.fromfile(snk, np.float32))
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
