"""Round-trip tests for the CSS chirp embedded blocks.

Chain under test:
    bits_file_source → css_map(sf) → chirp_mod(sf, os)
                     → chirp_demod(sf, os, zp) → css_demap(sf) → bits_file_sink

All integer-output Python blocks (int16, uint8) are exercised only via the
off-main-thread production runner to avoid the embedded-uint8-output SIGSEGV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from phy._dsp import read_bits, write_bits

from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.ir import GrBlock, GrConnection, GrPipeline

SF, OS, ZP = 7, 2, 4
SAMPLE_NUM = OS * (1 << SF)  # 256
PREAMBLE_LEN = 8

# How many full CSS symbols to round-trip
N_SYMS = 50


def _bits_to_bits_pipeline(
    bits_in: Path,
    bits_out: Path,
    *,
    sf: int = SF,
    os: int = OS,
    zp: int = ZP,
) -> GrPipeline:
    """Build a GrPipeline: bits_src → css_map → chirp_mod → chirp_demod
    → css_demap → bits_sink."""
    return GrPipeline(
        name="css_roundtrip",
        sample_rate=float(os * (1 << sf)),
        blocks=[
            GrBlock(id="src", kind="bits_file_source", params={"path": str(bits_in)}),
            GrBlock(id="map", kind="css_map", params={"sf": sf}),
            GrBlock(id="mod", kind="chirp_mod", params={"sf": sf, "oversample": os}),
            GrBlock(
                id="demod",
                kind="chirp_demod",
                params={"sf": sf, "oversample": os, "zero_pad": zp},
            ),
            GrBlock(id="demap", kind="css_demap", params={"sf": sf}),
            GrBlock(id="snk", kind="bits_file_sink", params={"path": str(bits_out)}),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="map"),
            GrConnection(src_block="map", dst_block="mod"),
            GrConnection(src_block="mod", dst_block="demod"),
            GrConnection(src_block="demod", dst_block="demap"),
            GrConnection(src_block="demap", dst_block="snk"),
        ],
    )


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
                params={"sf": sf, "oversample": os, "preamble_len": preamble_len},
            ),
            GrBlock(id="snk", kind="iq_file_sink", params={"path": str(iq_out)}),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="pre"),
            GrConnection(src_block="pre", dst_block="snk"),
        ],
    )


def test_css_core_bits_roundtrip(tmp_path: Path) -> None:
    """css_map → chirp_mod → chirp_demod → css_demap recovers bits exactly."""
    ensure_worker_warm()
    bits = np.random.default_rng(0).integers(0, 2, SF * N_SYMS).astype(np.uint8)
    bp = write_bits(tmp_path / "in.bits", bits)
    op = tmp_path / "out.bits"

    be = GnuRadioBackend()
    result = be.run_pipeline(_bits_to_bits_pipeline(bp, op))
    assert result.status == "ok", f"pipeline failed: {result}"

    out = read_bits(op)
    assert len(out) == len(bits), f"length mismatch: {len(out)} != {len(bits)}"
    assert np.array_equal(out, bits), f"bits mismatch at {np.where(out != bits)}"


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
