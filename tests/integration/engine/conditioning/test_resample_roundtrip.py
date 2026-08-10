from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from helpers._dsp import aligned_ber, read_bits, read_complex, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.conditioning import ResampleStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

IQ = Descriptor(Level.IQ, ItemType.C)
_DEV, _SYM = 0.75, 1.0


def _compile(
    modem: Modem,
    direction: Literal["rx", "tx"],
    sample_rate: float,
    src: Path,
    snk: Path,
) -> GrPipeline:
    from marconi.engine.stages.registry import stage_registry

    return compile_modem(
        modem,
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def _fsk_modem() -> Modem:
    return Modem(
        symbol_rate=_SYM,
        path=[FskStep(deviation=_DEV), SliceStep()],
    )


def _resample_rx(cap_rate: int) -> Modem:
    # RX resamples the captured rate back to 8 (the fsk sps rate): rate=8/cap_rate.
    return Modem(
        symbol_rate=_SYM,
        path=[
            ResampleStep(interpolation=8, decimation=cap_rate),
            FskStep(deviation=_DEV),
            SliceStep(),
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
