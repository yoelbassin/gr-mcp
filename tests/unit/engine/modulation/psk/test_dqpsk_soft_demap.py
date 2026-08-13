from pathlib import Path

import numpy as np
import numpy.typing as npt

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_llrs
from marconi.engine.modulation.ofdm.stages import DqpskSoftDemapStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

NC, DS = 5, 3  # 5 carriers, PRS + 3 FIC symbols


def _demap(
    tmp_path: Path,
    step: DqpskSoftDemapStep,
    points: npt.NDArray[np.complex128],
    syms: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    ensure_worker_warm()
    carriers = np.empty((DS + 1, NC), complex)
    carriers[0] = 1.0  # PRS reference
    for s in range(DS):
        carriers[s + 1] = carriers[s] * points[syms[s]]  # differential encode
    smaj = carriers.reshape(-1).astype(np.complex64)  # symbol-major
    src = tmp_path / "c.cf32"
    smaj.tofile(src)
    snk = tmp_path / "s.f32"
    modem = Modem(name="dm", symbol_rate=1.0, path=[step])
    pipe = compile_modem(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    # Per-block counts, not the RunResult: this chain feeds keep_m_in_n exactly
    # one n-block, so a single item lost upstream writes an empty sink, and the
    # default repr truncates the census that names where it went.
    assert r.status == "ok", (
        " | ".join(f"{c.block}:{c.items_in}->{c.items_out}" for c in (r.census or []))
        + f" err={r.error}"
    )
    soft = read_llrs(snk)
    assert soft.size == DS * NC * 2  # PRS dropped, 2 soft/carrier
    return soft


def test_dqpsk_soft_demap_recovers_bits(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    syms = rng.integers(0, 4, (DS, NC))
    step = DqpskSoftDemapStep(data_syms=DS, n_carriers=NC, scheme="psk", order=4)
    soft = _demap(tmp_path, step, qpsk, syms)
    assert np.all(np.abs(soft) > 0.1)  # clean QPSK -> confident
    # GR's qpsk constellation: index = 2*(Im>0) + (Re>0), soft MSB-first;
    # the engine contract is bit-1 = NEGATIVE LLR, so signs must complement
    # the GR bits (raw GR emits positive-is-one)
    gr_bits = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], np.uint8)
    expected_ones = gr_bits[syms.reshape(-1)].reshape(-1).astype(bool)
    assert np.array_equal(soft < 0, expected_ones)


def test_dqpsk_soft_demap_explicit_points_own_the_mapping(tmp_path: Path) -> None:
    # a protocol whose differential mapping complements GR's stock qpsk
    # declares it as caller points: a point's bit pattern is its index
    rng = np.random.default_rng(3)
    points = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    syms = rng.integers(0, 4, (DS, NC))
    step = DqpskSoftDemapStep(
        data_syms=DS,
        n_carriers=NC,
        scheme="explicit",
        points_i=[float(p.real) for p in points],
        points_q=[float(p.imag) for p in points],
    )
    soft = _demap(tmp_path, step, points, syms)
    assert np.all(np.abs(soft) > 0.1)
    idx_bits = np.array([[i >> 1 & 1, i & 1] for i in range(4)], np.uint8)
    expected_ones = idx_bits[syms.reshape(-1)].reshape(-1).astype(bool)
    assert np.array_equal(soft < 0, expected_ones)
