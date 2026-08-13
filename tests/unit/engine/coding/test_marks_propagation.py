"""Marks obey one invariant per op, never an accident of construction:
index-preserving ops keep them, realign shifts them, a fixed-width decode
RESCALES them onto the basis it produced, and a transition op emits a clean
carrier for its output domain. Compile-time: marks recorded by a probe cannot
cross a GR stage that changes the item rate or level before the coding seam."""

from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.coding.stages_bits import MarkFrameStep
from marconi.engine.coding.stages_symbols import SymbolMapStep
from marconi.engine.compile.compiler import compile_pipeline
from marconi.engine.compile.errors import CompileError
from marconi.engine.modulation.css.stages import CssDemapStep, DechirpStep
from marconi.engine.stages.probes import BurstProbeStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

MARKS = (3, 9)
IQ = Descriptor(Level.IQ, ItemType.C)
_SF = 11


def _dechirp_step() -> DechirpStep:
    return DechirpStep(sf=_SF, oversample=2, zero_pad=4)


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


def _assert_rescaled(out: CodingCarrier, name: str) -> None:
    """A fixed-width decode changes the bit basis, so a mark's offset scales by
    the ratio the decode applied. Dropping them instead was silent: normalize
    reads `if not c.marks: return c` and became a no-op with nothing in the
    census to say so, and mark_frame after any coding stage seeded no windows."""
    src = 32  # _marked() carries 32 bits
    ratio = int(out.bits.size) / src
    assert out.marks == tuple(
        sorted({int(m * ratio) for m in MARKS})
    ), f"{name}: {out.marks} at ratio {ratio}"
    assert all(0 <= m < out.bits.size for m in out.marks), name


def test_fixed_width_decodes_rescale_marks_onto_their_output() -> None:
    _assert_rescaled(
        ops_bits.block_code_rx(
            _marked(), code_bits=8, data_bits=4, parity_masks=[3, 5, 6, 7]
        ),
        "block_code",
    )
    _assert_rescaled(ops_bits.permute_rx(_marked(), perm=[0, 2, 1]), "permute")
    _assert_rescaled(
        ops_bits.codebook_rx(_marked(), code_bits=2, data_bits=1, table=[1, 2]),
        "codebook",
    )
    _assert_rescaled(
        ops_bits.rs_code_rx(_marked(), symbol_bits=4, n=7, k=3, prim_poly=0x13, fcr=1),
        "rs_code",
    )


def test_windowed_decode_rescales_marks_within_their_own_window() -> None:
    # two 16-bit windows halved by a (8,4) decode: a mark keeps its offset
    # within the window it belongs to, never the previous frame's tail
    c = CodingCarrier(
        bits=(np.arange(32) % 2).astype(np.uint8),
        windows=[Window(0, 0, 16), Window(16, 16, 32)],
        marks=(4, 20),
    )
    out = ops_bits.block_code_rx(c, code_bits=8, data_bits=4, parity_masks=[3, 5, 6, 7])
    assert out.marks == (2, 10)


def test_marks_outside_every_window_are_dropped() -> None:
    c = CodingCarrier(
        bits=np.zeros(32, np.uint8), windows=[Window(0, 0, 8)], marks=(2, 24)
    )
    out = ops_bits.permute_rx(c, perm=[1, 0])
    assert out.marks == (2,)


def test_probe_marks_cannot_cross_a_rate_changing_gr_stage() -> None:
    modem = Modem(
        symbol_rate=1.0,
        path=[
            _dechirp_step(),
            BurstProbeStep(),
            CssDemapStep(sf=_SF),
            MarkFrameStep(),
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
    modem = Modem(
        symbol_rate=1.0,
        path=[
            _dechirp_step(),
            BurstProbeStep(),
            SymbolMapStep(
                code_bits=11,
                data_bits=11,
                table=list(range(1 << 11)),
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
