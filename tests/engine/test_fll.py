"""fll: coarse carrier acquisition ahead of a decision-directed demod.

A costas loop pulls in only a fraction of its loop bandwidth. Measured on this
chain, psk_demod alone falls off a cliff between 5% and 8% of symbol rate --
and a 2 ppm TCXO at 900 MHz is ~1.8 kHz, which is 19% of symbol rate on a
9600-baud link. So every real capture sat outside the pull-in range and the
whole coherent-linear family could only ever decode its own transmitter.

Measured SER over this sweep:

    CFO (% symbol rate)   no fll   with fll
                     0%   0.0000     0.0000
                     5%   0.0000     0.0000
                     8%   0.7246     0.0000
                    20%   0.7230     0.0000
                    45%   0.7223     0.0000
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from engine._dsp import (
    AlignmentNotFound,
    channel,
    read_complex,
    resolved_ser,
    tx_sym_indices,
    write_bits,
)

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep

IQ = Descriptor(Level.IQ, "c")
_SR, _SYM, _ORDER, _NBITS = 8.0, 1.0, 4, 8192
_SETTLE, _MAX_SHIFT = 400, 1200  # max_shift MUST exceed settle or nothing aligns


def _points() -> tuple[np.ndarray, int]:
    from gnuradio import digital

    c = digital.constellation_qpsk()
    return np.asarray(c.points()), int(c.bits_per_symbol())


def _compile(path: list[ModemStep], direction: str, src: Path, snk: Path):
    return compile_modem(
        ModemSpec(symbol_rate=_SYM, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=IQ,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )


def _ser_at(tmp_path: Path, cfo_frac: float, with_fll: bool) -> float:
    be = GnuRadioBackend()
    pts, k = _points()
    bits = np.random.default_rng(4).integers(0, 2, _NBITS).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    clean = tmp_path / "clean.iq"
    tx = [
        ModemStep(conv="psk_demod", params={"order": _ORDER}),
        ModemStep(conv="psk_demap", params={"order": _ORDER}),
    ]
    assert be.run_pipeline(_compile(tx, "tx", bp, clean)).status == "ok"

    imp = tmp_path / f"imp_{cfo_frac}.iq"
    channel(
        clean,
        imp,
        snr_db=30.0,
        cfo_hz=cfo_frac * _SYM,
        sto=0.5,
        sample_rate=_SR,
        seed=1,
    )
    path = [ModemStep(conv="agc", params={"mode": "feedback"})]
    if with_fll:
        path.append(ModemStep(conv="fll", params={"rolloff": 0.35}))
    path.append(ModemStep(conv="psk_demod", params={"order": _ORDER}))
    snk = tmp_path / f"sym_{cfo_frac}_{with_fll}.cf32"
    assert be.run_pipeline(_compile(path, "rx", imp, snk), timeout=180.0).status == "ok"
    return resolved_ser(
        read_complex(snk),
        tx_sym_indices(bits, k),
        pts,
        _ORDER,
        max_shift=_MAX_SHIFT,
        settle=_SETTLE,
    )


@pytest.mark.parametrize("cfo_frac", [0.08, 0.20, 0.45])
def test_fll_acquires_offsets_costas_alone_cannot(
    cfo_frac: float, tmp_path: Path
) -> None:
    ensure_worker_warm()
    with pytest.raises(AlignmentNotFound):
        _ser_at(tmp_path, cfo_frac, with_fll=False)
    assert _ser_at(tmp_path, cfo_frac, with_fll=True) == 0.0


def test_fll_is_transparent_when_there_is_no_offset(tmp_path: Path) -> None:
    ensure_worker_warm()
    assert _ser_at(tmp_path, 0.0, with_fll=True) == 0.0


def test_fll_preserves_the_amplitude_claim() -> None:
    """Rotation only, so an agc upstream still satisfies a demod's contract."""
    from marconi.engine.types.descriptor import Amplitude

    stage = stage_registry()["fll"]
    normalized = Descriptor(Level.IQ, "c", amplitude=Amplitude.RMS_UNITY)
    assert stage.out_descriptor(normalized, {}).amplitude is Amplitude.RMS_UNITY
    assert stage.rate_factor({}) == 1.0


def test_fll_rejects_out_of_range_shaping() -> None:
    from pydantic import ValidationError

    model = stage_registry()["fll"].params_model
    for bad in (
        {"rolloff": 0.0},
        {"rolloff": 1.5},
        {"loop_bw": 0.0},
        {"filter_size": 0},
    ):
        with pytest.raises(ValidationError):
            model.model_validate(bad)
