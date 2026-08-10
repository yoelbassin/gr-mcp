from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from helpers._dsp import aligned_ber_best, channel, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.modulation.fsk.stages import FskStep, MskStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_SR, _SYM = 20.0, 1.0  # sps 20 = the ACARS operating point

# fractional 2.5/48000 cycles/sample -- half the +-5 Hz CFO bound
# measured on the real capture at 48 kHz (same fraction
# test_msk_block.py pins as _CFO_PINNED), rescaled to this test's _SR=20.0.
_CFO_HZ_PINNED = 2.5 / 48000.0 * _SR


def _tx_modem() -> Modem:
    return Modem(
        symbol_rate=_SYM,
        path=[FskStep(deviation=_SYM / 4.0), SliceStep()],
    )


def _rx_modem() -> Modem:
    return Modem(
        symbol_rate=_SYM,
        path=[MskStep(), SliceStep()],
    )


def _run(direction: Literal["rx", "tx"], modem: Modem, src: Path, snk: Path) -> None:
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=60.0).status == "ok"


def _best_ber(out: np.ndarray, bits: np.ndarray) -> float:
    # msk's rail decisions are the NRZI (differential) encoding of a
    # non-precoded MSK TX (the fsk stage at h=0.5 used here) -- see
    # test_msk_block.py's _best_ber for the same law. Decoded test-side so
    # the production demod stays protocol-agnostic (issue 22).
    h = out.astype(np.uint8)
    nrzi = (h[1:] ^ h[:-1]).astype(np.uint8)
    return aligned_ber_best([nrzi, 1 - nrzi], bits)


def test_msk_roundtrip_clean_ber0(tmp_path: Path) -> None:
    ensure_worker_warm()
    bits = np.random.default_rng(0).integers(0, 2, 2048).astype(np.uint8)
    bp, ip, op = tmp_path / "in.bits", tmp_path / "sig.iq", tmp_path / "out.bits"
    write_bits(bp, bits)
    _run("tx", _tx_modem(), bp, ip)
    _run("rx", _rx_modem(), ip, op)
    out = read_bits(op)
    assert out.size >= bits.size - 64, f"tail loss: {out.size}/{bits.size}"
    assert _best_ber(out, bits) == 0.0


def test_msk_roundtrip_survives_impairments(tmp_path: Path) -> None:
    ensure_worker_warm()
    bits = np.random.default_rng(3).integers(0, 2, 2048).astype(np.uint8)
    bp = tmp_path / "in.bits"
    clean, imp, op = tmp_path / "clean.iq", tmp_path / "imp.iq", tmp_path / "o.bits"
    write_bits(bp, bits)
    _run("tx", _tx_modem(), bp, clean)
    # cfo bound from the measured tracking range. msk_demod's bit clock
    # free-runs at the nominal baud (embedded/msk.py: the DECOUPLED path) with
    # no active SFO tracking, only sub-sample polyphase timing; empirically
    # BER stays 0 up to ~350ppm drift over this 2048-bit/sps=20 burst, so
    # sfo_ppm=244 both stays inside that budget and keeps the resample from
    # being a silent no-op at this length (>~12ppm here).
    channel(
        clean,
        imp,
        snr_db=20.0,
        cfo_hz=_CFO_HZ_PINNED,
        sto=1.5,
        sfo_ppm=244.0,
        sample_rate=_SR,
        seed=7,
    )
    _run("rx", _rx_modem(), imp, op)
    out = read_bits(op)
    assert out.size >= bits.size - 64
    assert _best_ber(out, bits) == 0.0
