from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import Fec, FecStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem


def _roundtrip(
    tmp_path: Path, *, rate_inv: int, polys: list[int], frame_bits: int, seed: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr, trellis

    tail = 6
    rng = np.random.default_rng(seed)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    bits = np.concatenate([info, np.zeros(tail, np.uint8)])
    fsm = trellis.fsm(1, rate_inv, polys)

    class Enc(gr.top_block):
        def __init__(self, data: npt.NDArray[np.uint8]) -> None:
            gr.top_block.__init__(self)
            src = gb.vector_source_b(list(map(int, data)), False)
            enc = trellis.encoder_bb(fsm, 0)
            snk = gb.vector_sink_b()
            self.connect(src, enc, snk)
            self.snk = snk

    e = Enc(bits)
    e.run()
    syms = np.array(e.snk.data(), np.int64)
    soft = np.empty(syms.size * rate_inv, np.float32)
    for i, s in enumerate(syms):
        for d in range(rate_inv):
            soft[i * rate_inv + d] = 1.0 - 2 * ((s >> (rate_inv - 1 - d)) & 1)
    src = tmp_path / f"s{seed}.f32"
    soft.tofile(src)
    snk = tmp_path / f"b{seed}.u8"
    modem = Modem(
        name="fec",
        symbol_rate=1.0,
        path=[
            FecStep(
                scheme="cc",
                rate_inv=rate_inv,
                polys=list(polys),
                frame_bits=frame_bits,
                tail=tail,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=Descriptor(Level.BITS, ItemType.F, Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    return read_bits(snk)[:frame_bits], info


def test_fec_conv_dab_rate_quarter(tmp_path: Path) -> None:
    out, info = _roundtrip(
        tmp_path, rate_inv=4, polys=[0o133, 0o171, 0o145, 0o133], frame_bits=768, seed=1
    )
    assert np.array_equal(out, info)


def test_fec_conv_generic_rate_half(tmp_path: Path) -> None:
    out, info = _roundtrip(
        tmp_path, rate_inv=2, polys=[0o133, 0o171], frame_bits=200, seed=2
    )
    assert np.array_equal(out, info)


_CC_PARAMS: dict[str, Any] = {
    "scheme": "cc",
    "rate_inv": 2,
    "polys": [0o171, 0o133],
    "frame_bits": 100,
    "tail": 6,
}


def test_fec_params_reject_unknown_scheme() -> None:
    with pytest.raises(ValidationError):
        Fec().step_model.model_validate({**_CC_PARAMS, "scheme": "turbo"})


def test_fec_params_reject_nonpositive_and_empty() -> None:
    override: dict[str, Any]
    for override in ({"rate_inv": 0}, {"frame_bits": 0}, {"polys": []}, {"tail": -1}):
        with pytest.raises(ValidationError):
            Fec().step_model.model_validate({**_CC_PARAMS, **override})


def test_fec_params_accept_cc() -> None:
    step = Fec().step_model.model_validate(_CC_PARAMS)
    assert step.scheme == "cc"


def test_fec_conv_unterminated_frame_tail_bits_correct(tmp_path: Path) -> None:
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr, trellis

    frame_bits, rate_inv, polys = 64, 2, [0o7, 0o5]
    rng = np.random.default_rng(7)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    while not (info[-1] or info[-2]):
        info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    fsm = trellis.fsm(1, rate_inv, polys)

    class Enc(gr.top_block):
        def __init__(self, data: npt.NDArray[np.uint8]) -> None:
            gr.top_block.__init__(self)
            src = gb.vector_source_b(list(map(int, data)), False)
            enc = trellis.encoder_bb(fsm, 0)
            snk = gb.vector_sink_b()
            self.connect(src, enc, snk)
            self.snk = snk

    e = Enc(info)
    e.run()
    syms = np.array(e.snk.data(), np.int64)
    soft = np.empty(syms.size * rate_inv, np.float32)
    for i, s in enumerate(syms):
        for d in range(rate_inv):
            soft[i * rate_inv + d] = 1.0 - 2 * ((s >> (rate_inv - 1 - d)) & 1)
    src = tmp_path / "unterm.f32"
    soft.tofile(src)
    snk = tmp_path / "unterm.u8"
    modem = Modem(
        name="fec",
        symbol_rate=1.0,
        path=[
            FecStep(
                scheme="cc",
                rate_inv=rate_inv,
                polys=polys,
                frame_bits=frame_bits,
                tail=0,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=Descriptor(Level.BITS, ItemType.F, Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    out = read_bits(snk)[:frame_bits]
    assert np.array_equal(out, info), np.flatnonzero(out != info)


def test_fec_conv_rate_two_thirds_k2(tmp_path: Path) -> None:
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr, trellis

    k, rate_inv, polys = 2, 3, [0o7, 0o3, 0o5, 0o6, 0o4, 0o7]
    frame_bits, tail = 128, 4
    rng = np.random.default_rng(3)
    info = rng.integers(0, 2, frame_bits).astype(np.uint8)
    padded = np.concatenate([info, np.zeros(tail, np.uint8)])
    syms_in = (padded[0::2] << 1) | padded[1::2]
    fsm = trellis.fsm(k, rate_inv, polys)

    class Enc(gr.top_block):
        def __init__(self, data: npt.NDArray[np.uint8]) -> None:
            gr.top_block.__init__(self)
            src = gb.vector_source_b(list(map(int, data)), False)
            enc = trellis.encoder_bb(fsm, 0)
            snk = gb.vector_sink_b()
            self.connect(src, enc, snk)
            self.snk = snk

    e = Enc(syms_in)
    e.run()
    syms = np.array(e.snk.data(), np.int64)
    soft = np.empty(syms.size * rate_inv, np.float32)
    for i, s in enumerate(syms):
        for d in range(rate_inv):
            soft[i * rate_inv + d] = 1.0 - 2 * ((s >> (rate_inv - 1 - d)) & 1)
    src = tmp_path / "k2.f32"
    soft.tofile(src)
    snk = tmp_path / "k2.u8"
    modem = Modem(
        name="fec",
        symbol_rate=1.0,
        path=[
            FecStep(
                scheme="cc",
                k=k,
                rate_inv=rate_inv,
                polys=polys,
                frame_bits=frame_bits,
                tail=tail,
            )
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=Descriptor(Level.BITS, ItemType.F, Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    assert np.array_equal(read_bits(snk)[:frame_bits], info)
