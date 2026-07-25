from math import log2
from pathlib import Path

import numpy as np
import pytest
from engine._dsp import (
    channel,
    read_bits,
    read_complex,
    resolved_ser,
    tx_sym_indices,
    write_bits,
)

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
SYM_C = Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT)
_SR, _SYM = 4.0, 1.0

# order -> (N_bits, cfo_frac, settle) — prototype-locked, SER-0 over 10 seeds
_CFG = {2: (2048, 0.004, 128), 4: (4096, 0.006, 128), 8: (6144, 0.001, 192)}


def _const_points(order: int):
    from gnuradio import digital  # in-process GR for oracle ground truth (allowed)

    c = {
        2: digital.constellation_bpsk,
        4: digital.constellation_qpsk,
        8: digital.constellation_8psk,
    }[order]()
    return np.asarray(c.points()), c.bits_per_symbol()


def _full(order: int) -> ModemSpec:
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="psk_demod", params={"order": order}),
            ModemStep(conv="psk_demap", params={"order": order}),
        ],
    )


def _demod(order: int) -> ModemSpec:
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(
                conv="agc",
                params={
                    "mode": "feedback",
                    # 16/16: agc2_cc limit-cycles at faster rates on a large
                    # gain step (see issue).
                    "attack_symbols": 16.0,
                    "decay_symbols": 16.0,
                },
            ),
            ModemStep(conv="psk_demod", params={"order": order}),
        ],
    )


def _compile(modem, direction, start, src, snk):
    return compile_modem(
        modem,
        _stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=start,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def _stage_registry():
    from marconi.engine.stages.registry import stage_registry

    return stage_registry()


@pytest.mark.parametrize("order", [2, 4, 8])
def test_psk_demod_ser0_under_impairments(order: int, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    n_bits, cfo_frac, settle = _CFG[order]
    points, k = _const_points(order)
    bits = np.random.default_rng(order).integers(0, 2, n_bits).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, sym = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "s.cf32"
    # TX bits -> IQ via the full modem
    assert be.run_pipeline(_compile(_full(order), "tx", IQ, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=25.0,
        cfo_hz=cfo_frac * _SR,
        sto=1.5,
        sfo_ppm=500.0,
        sample_rate=_SR,
        seed=order,
    )
    # RX taps soft complex symbols (demod only)
    assert be.run_pipeline(_compile(_demod(order), "rx", IQ, imp, sym)).status == "ok"
    rx_sym = read_complex(sym)
    tsi = tx_sym_indices(bits, k)
    assert resolved_ser(rx_sym, tsi, points, order, settle=settle) == 0.0


@pytest.mark.parametrize("order", [2, 4, 8])
def test_psk_demap_clean_ber0(order: int, tmp_path: Path) -> None:
    # BITS -> SYMBOLS -> BITS through the hard map/demap, no channel: identity.
    ensure_worker_warm()
    be = GnuRadioBackend()
    n_bits = _CFG[order][0]
    bits = np.random.default_rng(0).integers(0, 2, n_bits).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    sym, op = tmp_path / "s.cf32", tmp_path / "out.bits"
    modem = ModemSpec(
        symbol_rate=_SYM, path=[ModemStep(conv="psk_demap", params={"order": order})]
    )
    assert be.run_pipeline(_compile(modem, "tx", SYM_C, bp, sym)).status == "ok"
    assert be.run_pipeline(_compile(modem, "rx", SYM_C, sym, op)).status == "ok"
    out = read_bits(op)
    k = int(log2(order))
    n = (len(bits) // k) * k
    assert np.array_equal(out[:n], bits[:n])
