from pathlib import Path

import numpy as np
import pytest
from engine._dsp import (
    aligned_ber,
    channel,
    make_preamble,
    read_bits,
    read_complex,
    write_bits,
)

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compiler import compile_modem
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep

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


def _modem(order: int, direction: str) -> ModemSpec:
    pi, pq = make_preamble(_const_points(order))
    demod = [ModemStep(conv="psk_demod", params={"order": order})]
    if direction == "rx":
        # window_symbols=320: empirically measured; BER-0 across a real ~80-wide
        # robust band (272-352) under this test's impairments (see review-fix
        # report's sweep).
        demod = [
            ModemStep(conv="agc", params={"mode": "power", "window_symbols": 320.0})
        ] + demod
    return ModemSpec(
        symbol_rate=_SYM,
        path=demod
        + [
            ModemStep(
                conv="preamble_sync",
                params={"preamble_i": pi, "preamble_q": pq, "pad_symbols": 192},
            ),
            ModemStep(conv="psk_demap", params={"order": order}),
        ],
    )


def _compile(modem: ModemSpec, direction: str, src: Path, snk: Path):
    from marconi.engine.stages import stage_registry

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
    assert (
        be.run_pipeline(_compile(_modem(order, "tx"), "tx", bp, clean)).status == "ok"
    )
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
    assert be.run_pipeline(_compile(_modem(order, "rx"), "rx", imp, op)).status == "ok"
    out = read_bits(op)
    # preamble stripped: output is the payload minus the demod end-transient and
    # corr_est_cc's withheld filter tail (~104 syms), not payload+ramp+preamble;
    # guards against a passthrough no-op.
    k = order.bit_length() - 1
    assert len(out) < n_bits + 32 * k
    assert len(out) > n_bits - 128 * k
    assert aligned_ber(out, bits, max_shift=16 * k) == 0.0


def test_two_burst_capture_decodes_both(tmp_path: Path) -> None:
    """A capture holding two independent bursts decodes both in one run:
    sym_strip re-arms on burst 2's phase_est tag (issue 03). The carrier loop
    re-locks with an arbitrary M-fold ambiguity across the inter-burst gap, so
    burst 2 decodes only if its own tag re-estimates the rotation — the
    pre-fix first-tag latch replays burst 1's phase forever."""
    ensure_worker_warm()
    be = GnuRadioBackend()
    order, k = 4, 2
    n_bits = 2048
    rng = np.random.default_rng(11)
    bits1 = rng.integers(0, 2, n_bits).astype(np.uint8)
    bits2 = rng.integers(0, 2, n_bits).astype(np.uint8)
    tx_modem = _modem(order, "tx")
    f1, f2 = tmp_path / "f1.iq", tmp_path / "f2.iq"
    # 16 pad bits absorb the RRC pulse tail truncated at each burst's end (in
    # single-burst tests those edge symbols hide inside corr_est's withheld
    # EOF tail); the payloads under test then round-trip exactly.
    pad = np.zeros(16, dtype=np.uint8)
    b1 = write_bits(tmp_path / "b1.bits", np.concatenate([bits1, pad]))
    b2 = write_bits(tmp_path / "b2.bits", np.concatenate([bits2, pad]))
    assert be.run_pipeline(_compile(tx_modem, "tx", b1, f1)).status == "ok"
    assert be.run_pipeline(_compile(tx_modem, "tx", b2, f2)).status == "ok"
    noise = 0.02 * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
    gap, tail = noise[:1024], noise[1024:]
    cap = np.concatenate(
        [
            read_complex(f1),
            gap.astype(np.complex64),
            read_complex(f2),
            tail.astype(np.complex64),
        ]
    ).astype(np.complex64)
    cap_p, imp, op = tmp_path / "cap.iq", tmp_path / "imp.iq", tmp_path / "out.bits"
    cap.tofile(cap_p)
    channel(cap_p, imp, snr_db=25.0, cfo_hz=0.004 * _SR, sample_rate=_SR, seed=3)
    assert be.run_pipeline(_compile(_modem(order, "rx"), "rx", imp, op)).status == "ok"
    out = read_bits(op)
    assert len(out) > 2 * n_bits  # both payloads present (plus inter-burst junk)
    assert aligned_ber(out[: n_bits + 64 * k], bits1, max_shift=16 * k) == 0.0
    assert aligned_ber(out[n_bits:], bits2, max_shift=2048) == 0.0
