"""Marks obey one invariant per op, never an accident of construction:
index-preserving ops keep them, realign shifts them, index-destroying decodes
drop them, and a transition op emits a clean carrier for its output domain.
Compile-time: marks recorded by a probe cannot cross a GR stage that changes
the item rate or level before the coding seam."""

from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.compiler import CompileError, compile_pipeline
from marconi.engine.descriptor import Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep
from marconi.engine.params import ParamValue
from marconi.engine.stages import stage_registry

MARKS = (3, 9)
IQ = Descriptor(Level.IQ, "c")
_DECHIRP: dict[str, ParamValue] = {
    "sf": 11,
    "oversample": 2,
    "zero_pad": 4,
    "preamble_len": 8,
    "sfd_symbols": 2.25,
    "sync_symbols": 2,
}


def _marked(windows: list[Window] | None = None) -> CodingCarrier:
    return CodingCarrier(
        bits=(np.arange(32) % 2).astype(np.uint8), windows=windows, marks=MARKS
    )


def test_index_preserving_ops_keep_marks() -> None:
    assert ops_bits.differential_rx(_marked()).marks == MARKS
    assert ops_bits.nibble_swap_rx(_marked()).marks == MARKS
    assert ops_bits.descramble_rx(_marked(), sequence="ff").marks == MARKS
    assert ops_bits.sync_word_rx(_marked(), sync="7e").marks == MARKS
    assert ops_bits.segment_rx(_marked(), frame_body_len=8).marks == MARKS


def test_realign_blind_shifts_marks_with_the_bits() -> None:
    out = ops_bits.realign_rx(
        CodingCarrier(bits=np.zeros(32, np.uint8), marks=(4, 10)), bit_offset=8
    )
    assert out.marks == (2,)


def test_realign_seeded_keeps_marks() -> None:
    out = ops_bits.realign_rx(_marked(windows=[Window(0, 0)]), bit_offset=3)
    assert out.marks == MARKS


def test_index_destroying_ops_drop_marks() -> None:
    decoded = ops_bits.block_code_rx(
        _marked(), code_bits=8, data_bits=4, parity_masks=[3, 5, 6, 7]
    )
    assert decoded.marks == ()
    assert ops_bits.permute_rx(_marked(), perm=[0, 2, 1]).marks == ()
    assert (
        ops_bits.codebook_rx(_marked(), code_bits=2, data_bits=1, table=[1, 2]).marks
        == ()
    )
    rs = ops_bits.rs_code_rx(_marked(), symbol_bits=4, n=7, k=3, prim_poly=0x13, fcr=1)
    assert rs.marks == ()


def test_probe_marks_cannot_cross_a_rate_changing_gr_stage() -> None:
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="dechirp", params=_DECHIRP),
            ModemStep(conv="burst_probe", params={}),
            ModemStep(conv="css_demap", params=dict(_DECHIRP)),
            ModemStep(conv="mark_frame", params={}),
        ],
    )
    with pytest.raises(CompileError, match="burst_probe"):
        compile_pipeline(
            modem,
            stage_registry(),
            direction="rx",
            sample_rate=4096.0,
            start=IQ,
            source_io={},
            sink_io={},
        )


def test_probe_at_the_end_of_the_gr_segment_compiles() -> None:
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="dechirp", params=_DECHIRP),
            ModemStep(conv="burst_probe", params={}),
            ModemStep(
                conv="symbol_map",
                params={
                    "code_bits": 11,
                    "data_bits": 11,
                    "table": list(range(1 << 11)),
                },
            ),
        ],
    )
    cp = compile_pipeline(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=4096.0,
        start=IQ,
        source_io={},
        sink_io={},
    )
    assert cp.coding is not None
