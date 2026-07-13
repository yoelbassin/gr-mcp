"""Real off-air POCSAG, phy through the generic block_code (BCH 31,21) stage,
multimon-ng as the independent oracle.

The PHY (channelize/fsk/slice) closes with zero new production code. block_code
decodes each carved address codeword — its first real consumer; the stage's
syndrome/correction path runs (a no-op on this clean capture, where every
codeword is already valid). POCSAG's batch/codeword framing exceeds Marconi's
generic framing vocabulary (block_code is pre-seed-only, and there is no generic
sync-anchored codeword carve), so per the roadmap (framing is a thin verification
oracle, the PHY is the focus) that carve is thin test-side code here.

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

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import block_code_rx
from marconi.core.bitfile import read_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
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
IDLE = 0x7A89C197  # idle/filler codeword (flag bit is 0, so filtered explicitly)
BCH_GEN = 0x769  # BCH(31,21) generator: x^10+x^9+x^8+x^6+x^5+x^3+1
CODE_BITS, DATA_BITS = 31, 21
NPAR = CODE_BITS - DATA_BITS
BATCH_CODEWORDS = 16

# Oracle: multimon-ng's RICs on this slice, as (18-bit prefix, function).
ORACLE = {240071: 3, 154320: 3, 151233: 3}  # 1920569>>3, 1234567>>3, 1209871>>3


def _bch_parity_masks() -> list[int]:
    """Systematic BCH(31,21) parity-check rows in block_code_rx's basis: data in
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


def _word(bits: np.ndarray, lo: int, hi: int) -> int:
    return int(bits[lo:hi].dot(1 << np.arange(hi - lo - 1, -1, -1, dtype=np.int64)))


def _sync_positions(bits: np.ndarray) -> list[int]:
    pat = np.array([(SC >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
    win = np.lib.stride_tricks.sliding_window_view(bits, 32)
    return np.flatnonzero((win != pat).sum(axis=1) == 0).tolist()


def _address_codewords(bits: np.ndarray) -> list[np.ndarray]:
    """Thin sync-anchored carve: 16 codewords per batch, keep the 31-bit BCH span
    of every non-idle address codeword (flag MSB=0). Validity/correction is
    block_code_rx's job, not this carve's."""
    out: list[np.ndarray] = []
    for p in _sync_positions(bits):
        batch = bits[p + 32 : p + 32 + BATCH_CODEWORDS * 32]
        if batch.size < BATCH_CODEWORDS * 32:
            continue
        for j in range(BATCH_CODEWORDS):
            cw = batch[j * 32 : (j + 1) * 32]
            if int(cw[0]) == 0 and _word(cw, 0, 32) != IDLE:
                out.append(cw[:CODE_BITS])
    return out


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

    codewords = _address_codewords(read_bits(snk))
    assert codewords, "no POCSAG address codewords carved"

    # The generic coding stage decodes (and would correct) every address codeword.
    decoded = block_code_rx(
        RxCarrier(bits=np.concatenate(codewords)),
        code_bits=CODE_BITS,
        data_bits=DATA_BITS,
        parity_masks=PARITY_MASKS,
        correct=True,
    ).bits.reshape(-1, DATA_BITS)
    found = {_word(d, 1, 19): _word(d, 19, 21) for d in decoded}

    # Closure: block_code's decoded address prefixes are exactly multimon-ng's RICs
    # (>>3), each with the function it reported. A miscarried/miscorrected codeword
    # would perturb this set, so exact equality is the oracle. Determinism rests on
    # each RIC recurring across the capture's batches (a dropped clean instance
    # would fail, never falsely pass); observed stable over many runs.
    assert found == ORACLE, f"decoded {found}, oracle {ORACLE}"
