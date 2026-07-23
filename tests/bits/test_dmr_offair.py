"""Real off-air DMR closure gate: Marconi's front-end (channelize -> fsk) turns the
capture into soft symbols, and a composed codec of generic stages carves DMR framing
as caller data. The capture carries a LO-leakage DC offset co-located with the
baseband signal at 0 Hz, so a streaming DC-blocker would notch the signal itself; DC
is removed by whole-signal mean subtraction (thin capture prep) before the modem. The
FM discriminator also has a per-burst DC wander that a memoryless global slicer cannot
follow, so slicing is local: sync_symbols marks bursts on the soft-symbol signs, then
normalize per-burst removes DC and rescales before the m_slice M-ary decision;
symbol_map / mark_frame / fixed_frame carve each 264-bit burst. The generic BPTC
helper (Task 4) decodes each burst; the oracle is dsd-neo's independently
CRC-confirmed subscriber-ID pairs.

Polarity is fixed at asset-generation time (make_dmr_slice.py), so the gate does not
search it. All DMR values (sync pattern, thresholds, oracle pairs, field offsets) are
caller data in tests/; none enter src/marconi (test_agnosticism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from bits import _dmr

from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Symbolstream
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
        ],
    )


def _dmr_codec() -> CodecSpec:
    return CodecSpec(
        path=[
            CodecStep(
                conv="sync_symbols",
                params={"pattern": SYNC_PATTERN, "max_errors": 3, "pre_symbols": 54},
            ),
            CodecStep(
                conv="normalize",
                params={"span_symbols": 132, "dc": "median", "gain_percentile": 60.0},
            ),
            CodecStep(
                conv="m_slice",
                params={"thresholds": [-0.667, 0.0, 0.667], "levels": [3, 2, 0, 1]},
            ),
            CodecStep(
                conv="symbol_map",
                params={"code_bits": 2, "data_bits": 2, "table": [0, 1, 2, 3]},
            ),
            CodecStep(conv="mark_frame", params={"offset_bits": 0}),
            CodecStep(conv="fixed_frame", params={"payload_bits": 264}),
        ]
    )


def _decode(snk: Path) -> list[dict[str, int | str]]:
    sym = np.fromfile(snk, np.float32)
    stream = Symbolstream(path=snk, num_symbols=sym.size, item_type="f")
    result = parse_bitstream(stream, _dmr_codec(), registry())
    out: list[dict[str, int | str]] = []
    for fr in result.frames:
        burst = np.unpackbits(np.frombuffer(bytes.fromhex(fr.payload_hex), np.uint8))
        r = _dmr.decode_burst_from_bits(burst[:264])
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

    decoded = _decode(snk)
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
