"""Polar decode earns its keep: a real AWGN channel + the real soft lane, with
the uncoded hard decision as the control arm. This is the cross-stage proof that
the stock gr-fec polar decoder Marconi wires (psk_soft_demap -> polar) corrects
errors the raw link makes -- not just a noiseless self-consistent round-trip."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import PolarStep
from marconi.engine.modulation.psk.stages import PskSoftDemapStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)
_N, _K, _NFRAMES = 256, 128, 8


def _frozen(n: int, k: int) -> list[int]:
    from gnuradio.fec.polar.channel_construction import frozen_bit_positions

    return [int(p) for p in frozen_bit_positions(n, k, 0.5)]


def _polar_encode(
    info: npt.NDArray[np.uint8], n: int, k: int, fpos: list[int], fval: list[int]
) -> npt.NDArray[np.uint8]:
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


def _qpsk_points() -> npt.NDArray[np.complex128]:
    from gnuradio import digital

    return np.asarray(digital.constellation_qpsk().points())


def _run(
    path: list[Step], sym: npt.NDArray[np.complex128], tmp_path: Path
) -> npt.NDArray[np.uint8]:
    src = tmp_path / "sym.cf32"
    np.asarray(sym, np.complex64).tofile(src)
    snk = tmp_path / "out.u8"
    pipe = compile_modem(
        Modem(symbol_rate=1.0, path=path),
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=SYM_C,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=90.0)
    assert r.status == "ok", r
    return read_bits(snk)


def test_polar_corrects_errors_the_uncoded_link_makes(tmp_path: Path) -> None:
    """At Es/N0 4 dB QPSK the hard symbol decision errs on ~6% of coded bits; the
    polar SC decoder recovers the frame exactly. Measured over 4 seeds: uncoded
    0.056-0.066, coded 0.0 every time."""
    ensure_worker_warm()
    pts = _qpsk_points()
    fpos = _frozen(_N, _K)
    fval = [0] * len(fpos)
    rng = np.random.default_rng(11)
    info = rng.integers(0, 2, _K * _NFRAMES).astype(np.uint8)
    coded = _polar_encode(info, _N, _K, fpos, fval)
    pairs = coded.reshape(-1, 2)
    clean = pts[pairs[:, 0] * 2 + pairs[:, 1]]

    es = float(np.mean(np.abs(clean) ** 2))
    noise = np.sqrt(es / (2 * 10 ** (4.0 / 10)))
    g = rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
    noisy = clean + noise * g

    hi = np.argmin(np.abs(noisy[:, None] - pts[None, :]), axis=1)
    uncoded_ber = float(
        np.mean(np.stack([hi >> 1, hi & 1], axis=1).reshape(-1) != coded)
    )
    assert uncoded_ber > 0.02, f"control arm too clean to prove anything: {uncoded_ber}"

    out = _run(
        [
            PskSoftDemapStep(order=PskOrder(4)),
            PolarStep(
                block_size=_N,
                info_bits=_K,
                frozen_positions=fpos,
                frozen_values=fval,
                list_size=1,
            ),
        ],
        noisy,
        tmp_path,
    )[: info.size]
    assert out.size == info.size
    assert np.array_equal(out, info), (
        f"polar failed to correct: BER {float(np.mean(out != info))} "
        f"with uncoded {uncoded_ber}"
    )
