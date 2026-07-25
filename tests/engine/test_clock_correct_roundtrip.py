from pathlib import Path

import numpy as np
import pytest
from engine._dsp import aligned_ber, channel, read_bits, write_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep
from marconi.engine.types.params import ParamValue

IQ = Descriptor(Level.IQ, "c")
_SF, _OS, _ZP, _SYM = 9, 2, 4, 1.0


def _p() -> dict[str, ParamValue]:
    return {
        "sf": _SF,
        "oversample": _OS,
        "zero_pad": _ZP,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }


def _css_steps() -> list[ModemStep]:
    return [
        ModemStep(conv="chirp_sync", params=_p()),
        ModemStep(
            conv="dechirp", params={"sf": _SF, "oversample": _OS, "zero_pad": _ZP}
        ),
        ModemStep(conv="css_demap", params={"sf": _SF}),
    ]


def _compile(path, direction, rate, src, snk):
    from marconi.engine.stages.registry import stage_registry

    return compile_modem(
        ModemSpec(symbol_rate=_SYM, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=rate,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


# SFO is a no-op below ~1 chip of cumulative drift, so use a long frame + an
# exaggerated ppm: 50 ppm over ~150 SF9 symbols (~1024 samp/sym) drifts ~7.7
# samples = ~3.8 chips, which shatters the tail without correction.
@pytest.mark.parametrize("ppm", [50.0, -50.0])
def test_clock_correct_removes_sfo(ppm: float, tmp_path: Path) -> None:
    ensure_worker_warm()
    be = GnuRadioBackend()
    rate = _OS * (1 << _SF) * _SYM
    bits = (
        np.random.default_rng(int(abs(ppm))).integers(0, 2, _SF * 150).astype(np.uint8)
    )
    bp = write_bits(tmp_path / "in.bits", bits)
    clean, imp, op_no, op_cc = (
        tmp_path / n for n in ("c.iq", "i.iq", "o0.bits", "o1.bits")
    )

    tx = _css_steps()
    assert be.run_pipeline(_compile(tx, "tx", rate, bp, clean)).status == "ok"
    channel(
        clean,
        imp,
        snr_db=20.0,
        cfo_hz=0.0,
        sto=0.0,
        sfo_ppm=ppm,
        sample_rate=rate,
        seed=7,
    )

    rx_no = _css_steps()
    assert be.run_pipeline(_compile(rx_no, "rx", rate, imp, op_no)).status == "ok"
    assert (
        aligned_ber(read_bits(op_no), bits, max_shift=8 * _SF) > 0.0
    ), "SFO must break the tail"

    rx_cc = [ModemStep(conv="clock_correct", params={"ppm": ppm})] + rx_no
    assert be.run_pipeline(_compile(rx_cc, "rx", rate, imp, op_cc)).status == "ok"
    assert aligned_ber(read_bits(op_cc), bits, max_shift=8 * _SF) == 0.0
