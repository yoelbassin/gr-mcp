from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import Polar, PolarStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem


def _frozen(n, k):
    from gnuradio.fec.polar.channel_construction import frozen_bit_positions

    return [int(p) for p in frozen_bit_positions(n, k, 0.5)]


def _polar_encode(info, n, k, fpos, fval):
    from gnuradio import blocks as gb
    from gnuradio import fec, gr

    enc = fec.polar_encoder.make(n, k, fpos, fval, False)
    tb = gr.top_block()
    snk = gb.vector_sink_b()
    tb.connect(
        gb.vector_source_b(info.tolist(), False),
        fec.extended_encoder(enc, threading=None, puncpat="11"),
        snk,
    )
    tb.run()
    return np.asarray(snk.data(), np.uint8)


def _roundtrip(tmp_path, *, n, k, list_size, nframes, seed):
    ensure_worker_warm()
    fpos = _frozen(n, k)
    fval = [0] * len(fpos)
    info = np.random.default_rng(seed).integers(0, 2, k * nframes).astype(np.uint8)
    coded = _polar_encode(info, n, k, fpos, fval)
    # engine convention: bit 1 -> negative LLR (the polar stage flips it internally)
    soft = (1.0 - 2.0 * coded).astype(np.float32)
    src = tmp_path / f"s{seed}.f32"
    soft.tofile(src)
    snk = tmp_path / f"b{seed}.u8"
    modem = Modem(
        name="polar",
        symbol_rate=1.0,
        path=[
            PolarStep(
                block_size=n,
                info_bits=k,
                frozen_positions=fpos,
                frozen_values=fval,
                list_size=list_size,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=Descriptor(Level.BITS, ItemType.F, Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=60.0)
    assert r.status == "ok", r
    return read_bits(snk)[: info.size], info


def test_polar_sc_roundtrip(tmp_path):
    out, info = _roundtrip(tmp_path, n=256, k=128, list_size=1, nframes=20, seed=1)
    assert np.array_equal(out, info)


def test_polar_sc_list_roundtrip(tmp_path):
    out, info = _roundtrip(tmp_path, n=512, k=256, list_size=8, nframes=12, seed=2)
    assert np.array_equal(out, info)


_OK: dict[str, Any] = {
    "block_size": 8,
    "info_bits": 4,
    "frozen_positions": [0, 1, 2, 4],
    "frozen_values": [0, 0, 0, 0],
}


def test_polar_params_accept_valid():
    step = Polar().step_model.model_validate(_OK)
    assert step.block_size == 8 and step.info_bits == 4 and step.list_size == 1


@pytest.mark.parametrize(
    "override",
    [
        {"block_size": 6},  # not a power of two
        {"info_bits": 8},  # k >= n
        {"frozen_positions": [0, 1, 2], "frozen_values": [0, 0, 0]},  # wrong count
        {"frozen_values": [0, 0, 0]},  # values length != positions
        {"frozen_positions": [0, 1, 2, 2]},  # not distinct
        {"frozen_positions": [0, 1, 2, 8]},  # out of range
        {"frozen_values": [0, 0, 0, 2]},  # value not a bit
        {"list_size": 0},  # < 1
    ],
)
def test_polar_params_reject_malformed(override):
    with pytest.raises(ValidationError):
        Polar().step_model.model_validate({**_OK, **override})
