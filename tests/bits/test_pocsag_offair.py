"""Real off-air POCSAG, phy through a composed codec of generic seeded stages,
multimon-ng as the independent oracle.

The PHY (channelize/fsk/slice) closes with zero new production code, and the
batch/codeword framing now composes from the generic vocabulary alone:
sync_word seeds a frame per 32-bit batch sync, seeded permute gathers every
slot's 31-bit BCH span (dropping each slot's trailing even-parity bit), seeded
block_code (BCH 31,21, emit=data) decodes all 16 codewords per batch, and
fixed_frame carves the 336 decoded bits. This replaces the former hand-rolled
sync-anchored carve that existed only because block_code/permute were
pre-seed-only.

Residual, test-side by design: selecting address codewords (flag bit == 0) and
dropping idle over the DECODED 21-bit words needs a generic "select frames
where bit N == value" primitive that is out of scope for this plan. That
selection is the short loop below; idle (flag 0, non-oracle prefix) and any
noise decode fall to the oracle-membership compare.

multimon-ng 1.5.0 on the same slice decodes three pages, all function 3: RICs
1920569, 1234567, 1209871 (one carrying "THIS IS A TEST PERIODIC PAGE SEQUENTIAL
NUMBER  2379"). The RIC's low 3 bits are positional (the codeword's batch slot);
this gate matches on the unambiguous 18-bit prefix. Cross-codeword message-text
assembly is a documented follow-up, not asserted here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
RATE = 128000.0
_SLICE = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "assets"
    / "POCSAG"
    / "pocsag.cf32"
)

# POCSAG constants (caller data — a protocol lives in the fixture, not production).
SC = 0x7CD215D8  # frame sync codeword
SC_HEX = SC.to_bytes(4, "big").hex()
BCH_GEN = 0x769  # BCH(31,21) generator: x^10+x^9+x^8+x^6+x^5+x^3+1
CODE_BITS, DATA_BITS = 31, 21
NPAR = CODE_BITS - DATA_BITS
BATCH_CODEWORDS = 16
PERM = [slot * 32 + b for slot in range(BATCH_CODEWORDS) for b in range(CODE_BITS)]

# Oracle: multimon-ng's RICs on this slice, as (18-bit prefix, function).
ORACLE = {240071: 3, 154320: 3, 151233: 3}  # 1920569>>3, 1234567>>3, 1209871>>3


def _bch_parity_masks() -> list[int]:
    """Systematic BCH(31,21) parity-check rows in block_code's basis: data in
    stride[0:21] MSB-first, parity in stride[21:31]. mask p bit b is set when
    info bit b feeds parity bit p, from the remainder of (info << NPAR) mod GEN."""
    masks = [0] * NPAR
    for b in range(DATA_BITS):
        reg = (1 << (DATA_BITS - 1 - b)) << NPAR
        for i in range(CODE_BITS - 1, NPAR - 1, -1):
            if reg & (1 << i):
                reg ^= BCH_GEN << (i - NPAR)
        for p in range(NPAR):
            if (reg >> (NPAR - 1 - p)) & 1:
                masks[p] |= 1 << b
    return masks


PARITY_MASKS = _bch_parity_masks()


def _pocsag_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=1200.0,
        path=[
            ModemStep(
                conv="channelize",
                params={"decim": 4, "bandwidth_hz": 14000.0, "center_hz": -10250.0},
            ),
            ModemStep(conv="fsk", params={"deviation": 4500.0}),
            ModemStep(conv="slice", params={}),
        ],
    )


def _pocsag_codec() -> CodecSpec:
    return CodecSpec(
        path=[
            CodecStep(conv="sync_word", params={"sync": SC_HEX, "max_errors": 0}),
            CodecStep(conv="permute", params={"perm": PERM}),
            CodecStep(
                conv="block_code",
                params={
                    "code_bits": CODE_BITS,
                    "data_bits": DATA_BITS,
                    "parity_masks": PARITY_MASKS,
                    "correct": True,
                    "emit": "data",
                },
            ),
            CodecStep(
                conv="fixed_frame",
                params={"payload_bits": BATCH_CODEWORDS * DATA_BITS},
            ),
        ]
    )


def _word(bits: np.ndarray, lo: int, hi: int) -> int:
    return int(bits[lo:hi].dot(1 << np.arange(hi - lo - 1, -1, -1, dtype=np.int64)))


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="POCSAG slice absent — run tests/bits/make_pocsag_slice.py",
)
def test_pocsag_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    snk = tmp_path / "pocsag_bits.u8"
    pipe = compile_modem(
        _pocsag_modem(),
        stage_registry(),
        direction="rx",
        sample_rate=RATE,
        start=IQ,
        source_io={"path": str(_SLICE)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=180.0)
    assert r.status == "ok", r

    stream = Bitstream(
        path=snk, num_bits=int(read_bits(snk).size), source_capture=_SLICE
    )
    result = parse_bitstream(stream, _pocsag_codec(), registry())
    assert result.frames, "no POCSAG batches framed"

    found: dict[int, int] = {}
    for fr in result.frames:
        words = np.unpackbits(np.frombuffer(bytes.fromhex(fr.payload_hex), np.uint8))
        for d in words.reshape(-1, DATA_BITS):
            if int(d[0]) == 0:  # residual select: address codeword (flag bit 0)
                ric, fn = _word(d, 1, 19), _word(d, 19, 21)
                if ric:
                    found[ric] = fn
    found = {ric: fn for ric, fn in found.items() if ric in ORACLE}

    # Closure: the codec's decoded address prefixes are exactly multimon-ng's
    # RICs (>>3), each with the function it reported. A mis-seeded batch, a bad
    # permute gather, or a miscorrected codeword would perturb this set, so
    # exact equality is the oracle. Determinism rests on each RIC recurring
    # across the capture's batches (a dropped clean instance would fail, never
    # falsely pass); observed stable over many runs.
    assert found == ORACLE, f"decoded {found}, oracle {ORACLE}"
