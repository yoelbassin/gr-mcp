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

from pathlib import Path

import numpy as np
import pytest
from helpers import _synth as synth
from helpers._dsp import aligned_ber, read_bits

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.modulation.css.stages import (
    ChirpSyncStep,
    CssDemapStep,
    Dechirp,
    DechirpStep,
)
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level

SF, OS, ZP = 7, 2, 4
SAMPLE_NUM = OS * (1 << SF)  # 256
PREAMBLE_LEN = 8

# How many full CSS symbols to round-trip
N_SYMS = 50

_IQ = Descriptor(Level.IQ, ItemType.C)


def _bits_to_bits_pipeline(
    iq_in: Path, bits_out: Path, *, sf: int = SF, os: int = OS, zp: int = ZP
) -> GrPipeline:
    """iq_src → [stock dechirp chain] → css_demap → bits_sink. The chirps are
    synthesized (helpers/_synth.py) rather than modulated by a sibling block, so
    the decode is measured against an independent waveform."""
    rate = float(os * (1 << sf))
    ctx = CompileContext(_IQ, rate, 1.0)
    ctx.chain("iq_file_source", path=str(iq_in))
    Dechirp().emit_rx(ctx, DechirpStep(sf=sf, oversample=os, zero_pad=zp))
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
    bp = synth.write(
        tmp_path / "in.iq",
        synth.from_steps(
            [DechirpStep(sf=sf, oversample=os_, zero_pad=1), CssDemapStep(sf=sf)],
            bits,
            sample_rate=float(os_ << sf),
            symbol_rate=1.0,
        ),
    )
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
    bp = synth.write(
        tmp_path / "in.iq",
        synth.from_steps(
            [DechirpStep(sf=sf, oversample=os_, zero_pad=zp), CssDemapStep(sf=sf)],
            bits,
            sample_rate=float(os_ << sf),
            symbol_rate=1.0,
        ),
    )
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
    """iq_src (synthesized preamble + payload) → chirp_sync → [stock dechirp] …"""
    rate = float(os * (1 << sf))
    ctx = CompileContext(_IQ, rate, 1.0)
    ctx.chain("iq_file_source", path=str(bits_in))
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
    Dechirp().emit_rx(ctx, DechirpStep(sf=sf, oversample=os, zero_pad=zp))
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
    bp = synth.write(
        tmp_path / "in.iq",
        synth.from_steps(
            [
                ChirpSyncStep(
                    sf=SF,
                    oversample=OS,
                    zero_pad=ZP,
                    preamble_len=PREAMBLE_LEN,
                    sfd_symbols=2.25,
                    sync_symbols=2,
                ),
                DechirpStep(sf=SF, oversample=OS, zero_pad=ZP),
                CssDemapStep(sf=SF),
            ],
            np.concatenate([bits, pad]),
            sample_rate=float(OS << SF),
            symbol_rate=1.0,
        ),
    )
    op = tmp_path / "out.bits"
    be = GnuRadioBackend()
    assert be.run_pipeline(_full_chain_pipeline(bp, op)).status == "ok"
    out = read_bits(op)
    assert len(out) >= len(bits)  # payload fully emitted; pad absorbs the tail
    assert aligned_ber(out, bits) == 0.0


def test_chirp_sync_noise_bounded_no_lock() -> None:
    ensure_worker_warm()
    from gnuradio import blocks as gb
    from gnuradio import gr

    from marconi.engine.backends.gnuradio.embedded.chirp import make_chirp_sync

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
    assert (
        blk._core._buf.size <= bound
    ), f"pre-lock buffer {blk._core._buf.size} > {bound}"
    assert not blk._core.armed
    assert len(snk.data()) == 0
