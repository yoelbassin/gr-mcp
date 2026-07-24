"""Per-step frame counts, so "no messages" says which step lost them.

framing.py drops frames at seventeen separate `continue` statements. Counting
them per step -- rather than at each site -- is both cheaper and immune to
drift: a new op is measured the day it is written, with nothing to remember.
"""

from __future__ import annotations

import numpy as np
import pytest

from marconi.bits.carriers import RxCarrier, TxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.core.bitfile import write_bits
from marconi.core.models import Bitstream

_FLAG = [0, 1, 1, 1, 1, 1, 1, 0]


def _hdlc(payload: np.ndarray) -> np.ndarray:
    return np.concatenate([_FLAG, payload, _FLAG]).astype(np.uint8)


def _census_of(codec: CodecSpec, bits: np.ndarray) -> list:
    program = compile_codec(codec, registry(), "rx")
    census: list = []
    run_program(program, RxCarrier(bits=bits), census)
    return census


def test_rows_follow_codec_order_and_name_their_step() -> None:
    codec = CodecSpec(
        path=[
            CodecStep(conv="hdlc_deframe", params={}),
            CodecStep(conv="crc", params={"poly": 0x1021, "bits": 16, "init": 0xFFFF}),
        ]
    )
    census = _census_of(codec, _hdlc(np.zeros(64, np.uint8)))
    assert [c.stage for c in census] == ["hdlc_deframe[0]", "crc[1]"]


def test_the_step_that_dropped_every_frame_is_the_one_that_reads_zero_out() -> None:
    """crc over a payload shorter than the checksum drops the frame. The census
    localizes it without any per-site bookkeeping."""
    codec = CodecSpec(
        path=[
            CodecStep(conv="hdlc_deframe", params={}),
            CodecStep(conv="crc", params={"poly": 0x1021, "bits": 16, "init": 0xFFFF}),
        ]
    )
    census = _census_of(codec, _hdlc(np.zeros(8, np.uint8)))
    deframe, crc = census
    assert deframe.frames_out == 1, "the deframer should have found a frame"
    assert crc.frames_in == 1 and crc.frames_out == 0


def test_framing_surviving_with_no_check_passing_is_a_distinct_signature() -> None:
    """frames_out > 0 with crc_ok_out == 0 says the framing was right and the
    check parameters were not -- a different diagnosis from losing the frames."""
    codec = CodecSpec(
        path=[
            CodecStep(conv="hdlc_deframe", params={}),
            CodecStep(conv="crc", params={"poly": 0x8005, "bits": 16, "init": 0xFFFF}),
        ]
    )
    census = _census_of(codec, _hdlc(np.zeros(64, np.uint8)))
    crc = census[-1]
    assert crc.frames_out > 0
    assert crc.crc_ok_out == 0


def test_bit_counts_track_a_step_that_reshapes_the_stream() -> None:
    codec = CodecSpec(path=[CodecStep(conv="permute", params={"perm": [1, 0]})])
    (row,) = _census_of(codec, np.array([1, 0, 0, 1], np.uint8))
    assert row.bits_in == 4 and row.bits_out == 4


def test_an_empty_input_is_reported_not_hidden() -> None:
    codec = CodecSpec(path=[CodecStep(conv="hdlc_deframe", params={})])
    (row,) = _census_of(codec, np.zeros(0, np.uint8))
    assert row.bits_in == 0 and row.frames_out == 0


def test_a_tx_program_refuses_a_census_rather_than_silently_ignoring_it() -> None:
    codec = CodecSpec(
        path=[CodecStep(conv="crc", params={"poly": 0x1021, "bits": 16, "init": 0})]
    )
    program = compile_codec(codec, registry(), "tx")
    with pytest.raises(ValueError, match="rx"):
        run_program(program, TxCarrier(items=[b"\x00"]), [])


def test_the_census_reaches_the_seam(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """parse_bitstream is the product-facing entry point; the counts have to
    arrive there, not just exist inside the engine."""
    bits = _hdlc(np.zeros(64, np.uint8))
    path = tmp_path / "b.u8"
    write_bits(path, bits)
    codec = CodecSpec(
        path=[
            CodecStep(conv="hdlc_deframe", params={}),
            CodecStep(conv="crc", params={"poly": 0x1021, "bits": 16, "init": 0xFFFF}),
        ]
    )
    res = parse_bitstream(
        Bitstream(path=path, num_bits=int(bits.size)), codec, registry()
    )
    assert [c.stage for c in res.census] == ["hdlc_deframe[0]", "crc[1]"]
    assert res.census[0].bits_in == bits.size


def test_several_ops_from_one_stage_share_that_stage_name() -> None:
    """A stage that emits more than one op still reports under the name the
    operator wrote, so the rows map back onto the spec."""
    codec = CodecSpec(
        path=[CodecStep(conv="crc", params={"poly": 0x1021, "bits": 16, "init": 0})]
    )
    census = _census_of(codec, _hdlc(np.zeros(64, np.uint8)))
    assert {c.stage for c in census} == {"crc[0]"}
