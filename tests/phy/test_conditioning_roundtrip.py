from pathlib import Path

import numpy as np
import pytest
from phy._dsp import aligned_ber, read_bits, read_complex, upconvert_widen, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
_DEV, _SYM = 0.75, 1.0
_BASE_RATE, _UP = 8.0, 2  # baseband rate 8 -> widened to 16; channelize decim 2 -> 8


def _compile(modem, direction, sample_rate, src, snk):
    from marconi.phy.stages import stage_registry

    return compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def _fsk_modem() -> ModemSpec:
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="fsk", params={"deviation": _DEV}),
            ModemStep(conv="slice", params={}),
        ],
    )


def _chan_rx(center: float) -> ModemSpec:
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(
                conv="channelize",
                params={"decim": 2, "bandwidth_hz": 4.0, "center_hz": center},
            ),
            ModemStep(conv="fsk", params={"deviation": _DEV}),
            ModemStep(conv="slice", params={}),
        ],
    )


def _invert_rx() -> ModemSpec:
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(conv="invert", params={}),
            ModemStep(conv="fsk", params={"deviation": _DEV}),
            ModemStep(conv="slice", params={}),
        ],
    )


@pytest.mark.parametrize("center", [0.0, 4.0, -5.0])
def test_channelize_ber0_offset_capture(center: float, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(1).integers(0, 2, 2048).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    base, wide, op = tmp_path / "b.iq", tmp_path / "w.iq", tmp_path / "out.bits"
    # baseband FSK at rate 8 via the fsk modem TX
    assert (
        be.run_pipeline(_compile(_fsk_modem(), "tx", _BASE_RATE, bp, base)).status
        == "ok"
    )
    # widen x2 to rate 16, place the signal at +center, add AWGN
    upconvert_widen(
        base, wide, up=_UP, offset_hz=center, rate_wide=16.0, snr_db=12.0, seed=2
    )
    # RX: channelize back to rate 8, then demod
    assert (
        be.run_pipeline(_compile(_chan_rx(center), "rx", 16.0, wide, op)).status == "ok"
    )
    assert aligned_ber(read_bits(op), bits) == 0.0


def test_invert_ber0_mirrored_capture(tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(3).integers(0, 2, 2048).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    base, mirror, op = tmp_path / "b.iq", tmp_path / "m.iq", tmp_path / "out.bits"
    assert (
        be.run_pipeline(_compile(_fsk_modem(), "tx", _BASE_RATE, bp, base)).status
        == "ok"
    )
    # mirror the capture (spectral conjugate) — what invert must undo
    np.conj(read_complex(base)).astype(np.complex64).tofile(mirror)
    assert (
        be.run_pipeline(_compile(_invert_rx(), "rx", _BASE_RATE, mirror, op)).status
        == "ok"
    )
    assert aligned_ber(read_bits(op), bits) == 0.0
