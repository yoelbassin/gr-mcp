from math import log2
from pathlib import Path

import numpy as np
import pytest
from phy._dsp import channel, read_bits, resolved_ser_hard, tx_sym_indices, write_bits

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
SYM_B = Descriptor(Level.SYMBOLS, "b", carrier=Carrier.HARD)
_SR, _SYM = 4.0, 1.0

# order -> (N_bits, cfo_frac, snr_db, settle) — prototype-locked, SER-0 over 10 seeds
_CFG = {16: (24000, 0.002, 32.0, 1500), 64: (36000, 0.001, 40.0, 1500)}


def _const_points(order: int):
    from gnuradio import digital  # in-process GR for oracle ground truth (allowed)

    if order == 16:
        c = digital.constellation_16qam()
    else:
        from gnuradio.digital import mod_codes, qam

        c = qam.qam_constellation(
            constellation_points=order,
            differential=False,
            mod_code=mod_codes.GRAY_CODE,
            large_ampls_to_corners=False,
        )
    return np.asarray(c.points()), c.bits_per_symbol()


def _full(order: int) -> ModemSpec:
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="qam_demod", params={"order": order}),
            ModemStep(conv="qam_demap", params={"order": order}),
        ],
    )


def _demod(order: int) -> ModemSpec:
    # window_symbols=128: empirically SER-0 for orders 16/64 under this test's
    # impairments (fine-grained sweep, both n_bits x1 and x4). NOT a robust band:
    # rms_cf zero-inits its running estimate, so its startup transient (shaped by
    # alpha=1/(window*sps)) perturbs constellation_receiver_cb's lock chaotically
    # and non-monotonically in window -- see review-fix report for the sweep.
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="agc", params={"mode": "power", "window_symbols": 128.0}),
            ModemStep(conv="qam_demod", params={"order": order}),
        ],
    )


def _compile(modem, direction, start, src, snk):
    from marconi.phy.stages import stage_registry

    return compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=start,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


@pytest.mark.parametrize("order", [16, 64])
def test_qam_demod_ser0_under_impairments(order: int, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    n_bits, cfo_frac, snr_db, settle = _CFG[order]
    points, k = _const_points(order)
    bits = np.random.default_rng(order).integers(0, 2, n_bits).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, sym = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "s.u8"
    # TX bits -> IQ via the full modem (pack -> chunks_to_symbols -> RRC)
    assert be.run_pipeline(_compile(_full(order), "tx", IQ, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=snr_db,
        cfo_hz=cfo_frac * _SR,
        sto=1.5,
        sfo_ppm=500.0,
        sample_rate=_SR,
        seed=order,
    )
    # RX taps hard symbol indices (demod only) -> byte file
    assert be.run_pipeline(_compile(_demod(order), "rx", IQ, imp, sym)).status == "ok"
    rx_idx = read_bits(sym)
    tsi = tx_sym_indices(bits, k)
    assert resolved_ser_hard(rx_idx, tsi, points, settle=settle) == 0.0


@pytest.mark.parametrize("order", [16, 64])
def test_qam_demap_clean_identity(order: int, tmp_path: Path) -> None:
    # BITS -> symbol-index bytes -> BITS through the pure pack/unpack, no channel.
    ensure_worker_warm()
    be = GnuRadioBackend()
    n_bits = _CFG[order][0]
    bits = np.random.default_rng(0).integers(0, 2, n_bits).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    sym, op = tmp_path / "s.u8", tmp_path / "out.bits"
    modem = ModemSpec(
        symbol_rate=_SYM, path=[ModemStep(conv="qam_demap", params={"order": order})]
    )
    assert be.run_pipeline(_compile(modem, "tx", SYM_B, bp, sym)).status == "ok"
    assert be.run_pipeline(_compile(modem, "rx", SYM_B, sym, op)).status == "ok"
    out = read_bits(op)
    k = int(log2(order))
    n = (len(bits) // k) * k
    assert np.array_equal(out[:n], bits[:n])
