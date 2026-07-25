"""Real off-air DMR closure gate, the acid test of the phy repartition: the
full frame-coupled chain -- acquisition marks -> per-window normalize ->
m_slice -> symbol_map -> mark_frame windows -- now runs as ONE modem, in
product, through run_rx. Marconi's front-end (channelize -> fsk) turns the
capture into soft symbols; a coding tail of generic stages carries on in the
same pipeline: sync_symbols marks bursts on the soft-symbol signs, normalize
per-burst removes DC and rescales before the m_slice M-ary decision,
symbol_map hardens dibits to bits and scales the marks by data_bits, and
mark_frame reseeds windows from those scaled marks -- carving each 264-bit
burst is a thin test-side helper (framing.carve_fixed) over run_rx's windows.
The generic BPTC helper (Task 4) decodes each burst; the oracle is dsd-neo's
independently CRC-confirmed subscriber-ID pairs.

The capture carries a LO-leakage DC offset co-located with the baseband
signal at 0 Hz, so a streaming DC-blocker would notch the signal itself; DC
is removed by whole-signal mean subtraction (thin capture prep) before the
modem -- unchanged from the old gate.

Polarity is fixed at asset-generation time (make_dmr_slice.py), so the gate
does not search it. All DMR values (sync pattern, thresholds, oracle pairs,
field offsets) are caller data in tests/; none enter src/marconi
(test_agnosticism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from e2e import _dmr
from helpers import framing

from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import ensure_worker_warm
from marconi.phy.engine import run_rx
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
RATE = 39062.0
_SLICE = (
    Path(__file__).resolve().parents[2] / "artifacts" / "assets" / "DMR" / "dmr.cf32"
)

ORACLE_PAIRS = {(3109836, 2247700), (3109823, 2247700), (3169855, 2247700)}
SYNC_PATTERN = _dmr.SYNC_SIGNS.astype(int).tolist()


def _dmr_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=4800.0,
        path=[
            ModemStep(
                conv="channelize",
                params={"decim": 1, "bandwidth_hz": 12500.0, "center_hz": 0.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 1944.0, "loop_bw": 0.01}),
            ModemStep(
                conv="sync_symbols",
                params={"pattern": SYNC_PATTERN, "max_errors": 3, "pre_symbols": 54},
            ),
            ModemStep(
                conv="normalize",
                params={"span_symbols": 132, "dc": "median", "gain_percentile": 60.0},
            ),
            ModemStep(
                conv="m_slice",
                params={"thresholds": [-0.667, 0.0, 0.667], "levels": [3, 2, 0, 1]},
            ),
            ModemStep(
                conv="symbol_map",
                params={"code_bits": 2, "data_bits": 2, "table": [0, 1, 2, 3]},
            ),
            ModemStep(conv="mark_frame", params={"offset_bits": 0}),
        ],
    )


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="DMR slice absent — run tests/e2e/make_dmr_slice.py",
)
def test_dmr_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    iq = np.fromfile(_SLICE, np.complex64)
    iq = (iq - iq.mean()).astype(np.complex64)  # strip the LO-leakage DC spike
    src = tmp_path / "dmr_dc.cf32"
    src.write_bytes(iq.tobytes())

    res = run_rx(
        _dmr_modem(),
        stage_registry(),
        sample_rate=RATE,
        start=IQ,
        workdir=tmp_path,
        source_io={"path": str(src)},
    )
    assert res.status == "ok", res
    assert res.windows, f"no DMR bursts framed; census: {res.census}"
    assert res.bitstream is not None

    bits = read_bits(res.bitstream.path)
    out: list[dict[str, int | str]] = []
    for burst in framing.carve_fixed(bits, res.windows, 264):
        r = _dmr.decode_burst_from_bits(burst)
        if r:
            out.append(r)

    pairs = {
        (int(d["source"]), int(d["target"]))
        for d in out
        if d["kind"] == "data_header" or d.get("csbko") == 61
    }
    assert ORACLE_PAIRS <= pairs, f"missing dsd pairs; got {sorted(pairs)}"
    assert len(out) >= 5, f"CRC-valid yield too low: {len(out)}"
    sources = {
        int(d["source"])
        for d in out
        if (int(d["source"]), int(d["target"])) in ORACLE_PAIRS
    }
    assert len(sources) >= 2
    for d in out:
        assert 0 < int(d["source"]) < (1 << 24)
        assert 0 < int(d["target"]) < (1 << 24)
