from pathlib import Path

import numpy as np
import pytest
from phy._dsp import aligned_ber, channel, make_preamble, read_bits, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
_SR, _SYM = 4.0, 1.0
# order -> (N_payload_bits, cfo_frac) — same impairment levels as the PSK round-trip
_CFG = {2: (2048, 0.004), 4: (4096, 0.006), 8: (6144, 0.001)}


def _const_points(order: int) -> np.ndarray:
    from gnuradio import digital  # in-process GR for a constellation-valid preamble

    c = {
        2: digital.constellation_bpsk,
        4: digital.constellation_qpsk,
        8: digital.constellation_8psk,
    }[order]()
    return np.asarray(c.points())


def _modem(order: int) -> ModemSpec:
    pi, pq = make_preamble(_const_points(order))
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="psk_demod", params={"order": order}),
            ModemStep(
                conv="preamble_sync",
                params={"preamble_i": pi, "preamble_q": pq, "pad_symbols": 192},
            ),
            ModemStep(conv="psk_demap", params={"order": order}),
        ],
    )


def _compile(modem: ModemSpec, direction: str, src: Path, snk: Path):
    from marconi.phy.stages import stage_registry

    return compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


@pytest.mark.parametrize("order", [2, 4, 8])
def test_preamble_sync_ber0_no_oracle(order: int, tmp_path: Path) -> None:
    """The win: a full bits round-trip with NO rotation resolution. Without
    preamble_sync the demap output is rotated by the M-fold ambiguity (the PSK
    round-trip needs resolved_ser to undo it); preamble_sync derotates in-band,
    so aligned_ber (timing shift only, no rotation search) is 0."""
    ensure_worker_warm()
    be = GnuRadioBackend()
    n_bits, cfo_frac = _CFG[order]
    bits = np.random.default_rng(order).integers(0, 2, n_bits).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op = tmp_path / "c.iq", tmp_path / "i.iq", tmp_path / "out.bits"
    modem = _modem(order)
    assert be.run_pipeline(_compile(modem, "tx", bp, clean)).status == "ok"
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
    assert be.run_pipeline(_compile(modem, "rx", imp, op)).status == "ok"
    out = read_bits(op)
    # preamble stripped: output is the payload minus the demod end-transient and
    # corr_est_cc's withheld filter tail (~104 syms), not payload+ramp+preamble;
    # guards against a passthrough no-op.
    k = order.bit_length() - 1
    assert len(out) < n_bits + 32 * k
    assert len(out) > n_bits - 128 * k
    assert aligned_ber(out, bits, max_shift=16 * k) == 0.0
