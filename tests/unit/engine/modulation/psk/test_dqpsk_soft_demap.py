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
# keep_m_in_n emits only on a COMPLETE n-block, and GR can shave items off the
# tail at EOF: a measured CI failure had multiply_conjugate consume 16 of the 20
# items on offer, which left this chain's single block incomplete and the sink
# empty. Frames are independent (each opens its own PRS reference and
# keep_m_in_n's period is one frame), so feeding several turns that shaved tail
# into at most one lost frame instead of the whole run.
FRAMES = 3


def _demap(
    tmp_path: Path,
    step: DqpskSoftDemapStep,
    points: npt.NDArray[np.complex128],
    syms: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    ensure_worker_warm()
    frames, ds, nc = syms.shape
    carriers = np.empty((frames, ds + 1, nc), complex)
    carriers[:, 0] = 1.0  # PRS reference, one per frame
    for s in range(ds):
        carriers[:, s + 1] = carriers[:, s] * points[syms[:, s]]  # differential
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
    # Per-block counts, not the RunResult: the default repr truncates the census
    # that names which block dropped the items.
    assert r.status == "ok", (
        " | ".join(f"{c.block}:{c.items_in}->{c.items_out}" for c in (r.census or []))
        + f" err={r.error}"
    )
    soft = read_llrs(snk)
    quantum = ds * nc * 2  # PRS dropped, 2 soft/carrier
    # a whole number of frames, at least one, never more than were fed: a tail
    # frame may be shaved, a partial or misaligned frame is a real defect
    assert soft.size and soft.size % quantum == 0, f"{soft.size} not a frame multiple"
    assert soft.size <= frames * quantum
    return soft


def test_dqpsk_soft_demap_recovers_bits(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    syms = rng.integers(0, 4, (FRAMES, DS, NC))
    step = DqpskSoftDemapStep(data_syms=DS, n_carriers=NC, scheme="psk", order=4)
    soft = _demap(tmp_path, step, qpsk, syms)
    assert np.all(np.abs(soft) > 0.1)  # clean QPSK -> confident
    # GR's qpsk constellation: index = 2*(Im>0) + (Re>0), soft MSB-first;
    # the engine contract is bit-1 = NEGATIVE LLR, so signs must complement
    # the GR bits (raw GR emits positive-is-one)
    gr_bits = np.array([[1, 1], [0, 1], [1, 0], [0, 0]], np.uint8)
    expected_ones = gr_bits[syms.reshape(-1)].reshape(-1).astype(bool)
    got_ones = soft < 0
    # every frame that survived is checked, not just the first
    assert np.array_equal(got_ones, expected_ones[: got_ones.size])


def test_dqpsk_soft_demap_explicit_points_own_the_mapping(tmp_path: Path) -> None:
    # a protocol whose differential mapping complements GR's stock qpsk
    # declares it as caller points: a point's bit pattern is its index
    rng = np.random.default_rng(3)
    points = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    syms = rng.integers(0, 4, (FRAMES, DS, NC))
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
    got_ones = soft < 0
    assert np.array_equal(got_ones, expected_ones[: got_ones.size])
