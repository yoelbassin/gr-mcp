from pathlib import Path

import numpy as np
import pytest
from engine._dsp import aligned_ber, read_bits, read_complex, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compiler import compile_modem
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
_DEV, _SYM = 0.75, 1.0


def _compile(modem, direction, sample_rate, src, snk):
    from marconi.engine.stages import stage_registry

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


def _resample_rx(cap_rate: int) -> ModemSpec:
    # RX resamples the captured rate back to 8 (the fsk sps rate): rate=8/cap_rate.
    return ModemSpec(
        symbol_rate=_SYM,
        path=[
            ModemStep(
                conv="resample", params={"interpolation": 8, "decimation": cap_rate}
            ),
            ModemStep(conv="fsk", params={"deviation": _DEV}),
            ModemStep(conv="slice", params={}),
        ],
    )


@pytest.mark.parametrize(
    "cap_rate", [7, 9]
)  # 7: upsample 7->8; 9: downsample 9->8 (CSS uses downsample)
def test_resample_ber0(cap_rate: int, tmp_path: Path) -> None:
    from scipy.signal import resample_poly

    ensure_worker_warm()
    be = GnuRadioBackend()
    bits = np.random.default_rng(cap_rate).integers(0, 2, 2048).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    base, off, op = tmp_path / "b.iq", tmp_path / "o.iq", tmp_path / "out.bits"
    assert be.run_pipeline(_compile(_fsk_modem(), "tx", 8.0, bp, base)).status == "ok"
    # model a capture sampled at cap_rate instead of 8
    x = read_complex(base)
    resample_poly(x, cap_rate, 8).astype(np.complex64).tofile(off)
    assert (
        be.run_pipeline(
            _compile(_resample_rx(cap_rate), "rx", float(cap_rate), off, op)
        ).status
        == "ok"
    )
    assert aligned_ber(read_bits(op), bits) == 0.0
