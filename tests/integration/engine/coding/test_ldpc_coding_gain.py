"""LDPC decode earns its keep: a real AWGN channel + the real soft lane, with
the uncoded hard decision as the control arm. This is the cross-stage proof that
the stock gr-fec LDPC decoder Marconi wires (psk_soft_demap -> ldpc) corrects
errors the raw link makes -- not just a noiseless self-consistent round-trip."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from helpers._ldpc import ldpc_codes, parse_check_nodes

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import LdpcStep
from marconi.engine.modulation.psk.stages import PskSoftDemapStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)


def _pick_code() -> tuple[Path, int, int, int] | None:
    codes = ldpc_codes()
    big = [c for c in codes if c[1] >= 256]  # a longer code -> a sharper waterfall
    return big[0] if big else (codes[-1] if codes else None)


_CODE = _pick_code()
pytestmark = pytest.mark.skipif(
    _CODE is None, reason="no gr-fec LDPC alist codes installed"
)


def _encode(path: Path, gap: int, info: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    from gnuradio import blocks as gb
    from gnuradio import fec, gr

    enc = fec.ldpc_par_mtrx_encoder.make(str(path), gap)
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
    r = GnuRadioBackend().run_pipeline(pipe, timeout=120.0)
    assert r.status == "ok", r
    return read_bits(snk)


def test_ldpc_corrects_errors_the_uncoded_link_makes(tmp_path: Path) -> None:
    """At Es/N0 4 dB QPSK the hard symbol decision errs on ~5% of coded bits; the
    LDPC decoder recovers the frame exactly. Measured over 4 seeds (N=300):
    uncoded 0.048-0.059, coded 0.0 every time."""
    ensure_worker_warm()
    assert _CODE is not None
    path, n, k, gap = _CODE
    checks = parse_check_nodes(path, n)
    pts = _qpsk_points()
    rng = np.random.default_rng(11)
    nframes = max(4, 2000 // n)
    info = rng.integers(0, 2, k * nframes).astype(np.uint8)
    coded = _encode(path, gap, info)
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
            LdpcStep(block_size=n, check_nodes=checks, max_iterations=50),
        ],
        noisy,
        tmp_path,
    )[: info.size]
    assert out.size == info.size
    assert np.array_equal(out, info), (
        f"ldpc failed to correct: BER {float(np.mean(out != info))} "
        f"with uncoded {uncoded_ber}"
    )
