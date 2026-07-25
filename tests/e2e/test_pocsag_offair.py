"""Real off-air POCSAG, one ModemSpec spanning phy through the coding tail,
multimon-ng as the independent oracle.

The PHY (channelize/fsk/slice) and the batch/codeword framing now compose
from the same generic vocabulary in a single pipeline: sync_word seeds a
frame per 32-bit batch sync, seeded permute gathers every slot's 31-bit BCH
span (dropping each slot's trailing even-parity bit), and seeded block_code
(BCH 31,21, emit=data) decodes all 16 codewords per batch. Carving the 336
decoded bits into batches is a test-side helper (framing.carve_fixed) over
run_rx's windows.

Residual, test-side by design: selecting address codewords (flag bit == 0) and
dropping idle over the DECODED 21-bit words needs a generic "select frames
where bit N == value" primitive that is out of scope for this plan. That
selection is the short loop below; it explicitly excludes idle's decoded
value, so any other decode must exactly match the oracle.

multimon-ng 1.5.0 on the same slice decodes three pages, all function 3: RICs
1920569, 1234567, 1209871 (one carrying "THIS IS A TEST PERIODIC PAGE SEQUENTIAL
NUMBER  2379"). The RIC's low 3 bits are positional (the codeword's batch slot);
this gate matches on the unambiguous 18-bit prefix. Cross-codeword message-text
assembly is a documented follow-up, not asserted here.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from helpers import framing

from marconi.engine.backends.gnuradio.runner import ensure_worker_warm
from marconi.engine.bitfile import read_bits
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep
from marconi.engine.run import run_rx
from marconi.engine.stages import stage_registry

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
IDLE = 0x7A89C197  # idle/filler codeword
IDLE_DATA = IDLE >> (NPAR + 1)  # its decoded 21-bit word (drop parity + trailing bit)

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
            ModemStep(conv="agc", params={"window_symbols": 64.0}),
            ModemStep(conv="fsk", params={"deviation": 4500.0}),
            ModemStep(conv="slice", params={}),
            ModemStep(conv="sync_word", params={"sync": SC_HEX, "max_errors": 0}),
            ModemStep(conv="permute", params={"perm": cast("list[float | int]", PERM)}),
            ModemStep(
                conv="block_code",
                params={
                    "code_bits": CODE_BITS,
                    "data_bits": DATA_BITS,
                    "parity_masks": cast("list[float | int]", PARITY_MASKS),
                    "correct": True,
                    "emit": "data",
                },
            ),
        ],
    )


def _word(bits: np.ndarray, lo: int, hi: int) -> int:
    return int(bits[lo:hi].dot(1 << np.arange(hi - lo - 1, -1, -1, dtype=np.int64)))


@pytest.mark.skipif(
    not _SLICE.exists(),
    reason="POCSAG slice absent — run tests/e2e/make_pocsag_slice.py",
)
def test_pocsag_offair(tmp_path: Path) -> None:
    ensure_worker_warm()
    res = run_rx(
        _pocsag_modem(),
        stage_registry(),
        sample_rate=RATE,
        start=IQ,
        workdir=tmp_path,
        source_io={"path": str(_SLICE)},
    )
    assert res.status == "ok", res
    assert res.windows, "no POCSAG batches framed"
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)
    batches = framing.carve_fixed(bits, res.windows, BATCH_CODEWORDS * DATA_BITS)

    found: dict[int, int] = {}
    for batch in batches:
        for d in batch.reshape(-1, DATA_BITS):
            if int(d[0]) != 0:
                continue
            if _word(d, 0, DATA_BITS) == IDLE_DATA:
                continue
            found[_word(d, 1, 19)] = _word(d, 19, 21)

    assert found == ORACLE, f"decoded {found}, oracle {ORACLE}"
