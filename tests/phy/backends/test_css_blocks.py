"""Round-trip tests for the CSS chirp embedded blocks.

Chain under test:
    bits_file_source → css_map(sf) → chirp_mod(sf, os)
                     → [stock dechirp composition] → css_demap(sf) → bits_file_sink

The dechirp RX is the production `Dechirp` stage's stock-GR composition
(multiply_cc/fft_vcc/fold/peak_decision — see stages.py), exercised here via
CompileContext so the test stays DRY with production.

All integer-output Python blocks (int16, uint8) are exercised only via the
off-main-thread production runner to avoid the embedded-uint8-output SIGSEGV.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from phy._dsp import aligned_ber, read_bits, write_bits

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compile_context import CompileContext
from marconi.phy.ir import GrBlock, GrConnection, GrPipeline
from marconi.phy.modulation.css.stages import Dechirp

SF, OS, ZP = 7, 2, 4
SAMPLE_NUM = OS * (1 << SF)  # 256
PREAMBLE_LEN = 8

# How many full CSS symbols to round-trip
N_SYMS = 50

_IQ = Descriptor(Level.IQ, "c")


def _bits_to_bits_pipeline(
    bits_in: Path, bits_out: Path, *, sf: int = SF, os: int = OS, zp: int = ZP
) -> GrPipeline:
    """bits_src → css_map → chirp_mod → [stock dechirp chain] → css_demap →
    bits_sink."""
    rate = float(os * (1 << sf))
    ctx = CompileContext(_IQ, rate, 1.0)
    ctx.chain("bits_file_source", path=str(bits_in))
    ctx.chain("css_map", sf=sf)
    ctx.chain("chirp_mod", sf=sf, oversample=os)
    Dechirp().emit_rx(ctx, {"sf": sf, "oversample": os, "zero_pad": zp})
    ctx.chain("css_demap", sf=sf)
    ctx.chain("bits_file_sink", path=str(bits_out))
    return ctx.build("css_roundtrip", rate)


def _prepend_pipeline(
    iq_in: Path,
    iq_out: Path,
    *,
    sf: int = SF,
    os: int = OS,
    preamble_len: int = PREAMBLE_LEN,
) -> GrPipeline:
    """Build a GrPipeline: iq_src → chirp_prepend → iq_sink."""
    return GrPipeline(
        name="css_prepend_len",
        sample_rate=float(os * (1 << sf)),
        blocks=[
            GrBlock(id="src", kind="iq_file_source", params={"path": str(iq_in)}),
            GrBlock(
                id="pre",
                kind="chirp_prepend",
                params={
                    "sf": sf,
                    "oversample": os,
                    "preamble_len": preamble_len,
                    "sfd_symbols": 2.25,
                },
            ),
            GrBlock(id="snk", kind="iq_file_sink", params={"path": str(iq_out)}),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="pre"),
            GrConnection(src_block="pre", dst_block="snk"),
        ],
    )


@pytest.mark.parametrize(
    ("sf", "os_", "n_syms"),
    [
        (SF, OS, N_SYMS),
        # sn = os*2^sf exceeds GR's 8191-item default buffer from sf=12 os=2 up;
        # these pin the whole advertised SF range (mod completes, demod nonempty).
        (12, 2, 12),
        (13, 2, 12),
        (14, 2, 12),
        (11, 4, 12),
    ],
)
def test_css_core_bits_roundtrip(
    sf: int, os_: int, n_syms: int, tmp_path: Path
) -> None:
    """css_map → chirp_mod → [stock dechirp] → css_demap recovers bits exactly."""
    ensure_worker_warm()
    bits = np.random.default_rng(0).integers(0, 2, sf * n_syms).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    op = tmp_path / "out.bits"

    be = GnuRadioBackend()
    result = be.run_pipeline(_bits_to_bits_pipeline(bp, op, sf=sf, os=os_))
    assert result.status == "ok", f"pipeline failed: {result}"

    out = read_bits(op)
    assert len(out) == len(bits), f"length mismatch: {len(out)} != {len(bits)}"
    assert np.array_equal(out, bits), f"bits mismatch at {np.where(out != bits)}"


@pytest.mark.parametrize(
    ("sf", "os_", "zp", "n_syms"),
    [
        # bins = 2^sf * zp > 32768: overflows an int16 argmax index (the
        # stock argmax_fs regression); decision must be full-precision.
        (12, 2, 10, 12),
        (14, 2, 10, 12),
    ],
)
def test_css_roundtrip_wide_fold_bins(
    sf: int, os_: int, zp: int, n_syms: int, tmp_path: Path
) -> None:
    ensure_worker_warm()
    bits = np.random.default_rng(1).integers(0, 2, sf * n_syms).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    op = tmp_path / "out.bits"
    be = GnuRadioBackend()
    result = be.run_pipeline(_bits_to_bits_pipeline(bp, op, sf=sf, os=os_, zp=zp))
    assert result.status == "ok", f"pipeline failed: {result}"
    out = read_bits(op)
    assert len(out) == len(bits)
    assert np.array_equal(out, bits)


def _full_chain_pipeline(
    bits_in: Path,
    bits_out: Path,
    *,
    sf: int = SF,
    os: int = OS,
    zp: int = ZP,
    preamble_len: int = PREAMBLE_LEN,
) -> GrPipeline:
    """TX+RX in one flowgraph: … chirp_prepend → chirp_sync → [stock dechirp] …"""
    rate = float(os * (1 << sf))
    ctx = CompileContext(_IQ, rate, 1.0)
    ctx.chain("bits_file_source", path=str(bits_in))
    ctx.chain("css_map", sf=sf)
    ctx.chain("chirp_mod", sf=sf, oversample=os)
    ctx.chain(
        "chirp_prepend",
        sf=sf,
        oversample=os,
        preamble_len=preamble_len,
        sfd_symbols=2.25,
    )
    ctx.chain(
        "chirp_sync",
        sf=sf,
        oversample=os,
        zero_pad=zp,
        preamble_len=preamble_len,
        bandwidth=float(1 << sf),
        sfd_symbols=2.25,
        sync_symbols=2,
    )
    Dechirp().emit_rx(ctx, {"sf": sf, "oversample": os, "zero_pad": zp})
    ctx.chain("css_demap", sf=sf)
    ctx.chain("bits_file_sink", path=str(bits_out))
    return ctx.build("css_full_chain", rate)


def test_chirp_sync_clean_roundtrip(tmp_path: Path) -> None:
    """chirp_prepend adds preamble+SFD; chirp_sync detects it, strips to payload,
    and derotates the (zero) CFO; payload bits survive the full chain exactly.
    chirp_sync's emission trails its always-on burst hunt by a bounded margin
    (withheld at EOF), so the payload is followed by pad symbols that absorb it."""
    ensure_worker_warm()
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, SF * 40).astype(np.uint8)
    pad = rng.integers(0, 2, SF * (PREAMBLE_LEN + 4)).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", np.concatenate([bits, pad]))
    op = tmp_path / "out.bits"
    be = GnuRadioBackend()
    assert be.run_pipeline(_full_chain_pipeline(bp, op)).status == "ok"
    out = read_bits(op)
    assert len(out) >= len(bits)  # payload fully emitted; pad absorbs the tail
    assert aligned_ber(out, bits) == 0.0


def test_chirp_prepend_output_length(tmp_path: Path) -> None:
    """chirp_prepend emits (preamble_len + 2.25)*sample_num prepend samples
    followed by the input payload samples."""
    ensure_worker_warm()
    payload_samples = 4 * SAMPLE_NUM  # a few chirp symbols worth of IQ
    payload = np.ones(payload_samples, dtype=np.complex64)
    iq_in = tmp_path / "payload.iq"
    payload.tofile(iq_in)
    iq_out = tmp_path / "out.iq"

    be = GnuRadioBackend()
    result = be.run_pipeline(_prepend_pipeline(iq_in, iq_out))
    assert result.status == "ok", f"pipeline failed: {result}"

    out = np.fromfile(iq_out, dtype=np.complex64)
    sfd_len = SAMPLE_NUM + SAMPLE_NUM + SAMPLE_NUM // 4  # 2.25 * sample_num
    expected_prepend = PREAMBLE_LEN * SAMPLE_NUM + sfd_len
    expected_total = expected_prepend + payload_samples
    assert len(out) == expected_total, (
        f"expected {expected_total} samples ({expected_prepend} prepend + "
        f"{payload_samples} payload), got {len(out)}"
    )


def test_chirp_mod_build_allocation_bounded():
    ensure_worker_warm()
    from gnuradio import gr

    from marconi.phy.backends.gnuradio.embedded.chirp import make_chirp_mod

    for sf in (12, 14):
        tracemalloc.start()
        make_chirp_mod(gr, sf, 2)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 10 * 2**20, f"sf={sf} build allocated {peak} bytes"


def test_chirp_sync_noise_bounded_no_lock():
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr

    from marconi.phy.backends.gnuradio.embedded.chirp import make_chirp_sync

    sf, os_, zp, pl = SF, OS, ZP, PREAMBLE_LEN
    sn = os_ * (1 << sf)
    rng = np.random.default_rng(9)
    n = 400 * sn
    sig = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    blk = make_chirp_sync(gr, sf, os_, zp, pl, float(os_ * (1 << sf)), 2.25, 2)
    tb = gr.top_block()
    src = gb.vector_source_c(sig.tolist(), False)
    snk = gb.vector_sink_c()
    tb.connect(src, blk, snk)
    tb.run()
    bound = 2 * (pl + 6) * sn + 8192
    assert blk._buf.size <= bound, f"pre-lock buffer {blk._buf.size} > {bound}"
    assert not blk._armed
    assert len(snk.data()) == 0
