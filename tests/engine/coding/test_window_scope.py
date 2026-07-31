"""Seeded-but-empty is not blind.

`windows is None` means no seeder ran — whole-stream (blind) ops are correct.
`windows == []` means a seeder ran and found nothing — decoding anything from
that scope fabricates data from noise. A run whose coding tail produced zero
items reports status "empty" (mirroring the GR half's empty-sink flag), never
"ok" with invented bits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.stages_bits import BlockCodeStep, SyncWordStep
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem

BITS = Descriptor(Level.BITS, ItemType.B)
HAMMING74 = [0b1011, 0b1101, 0b0111]


def _alternating(n: int) -> np.ndarray:
    return (np.arange(n) % 2).astype(np.uint8)


def test_seeded_empty_scope_decodes_nothing() -> None:
    empty = CodingCarrier(bits=_alternating(700), windows=[])
    decoded = ops_bits.block_code_rx(
        empty, code_bits=7, data_bits=4, parity_masks=HAMMING74
    )
    assert decoded.bits.size == 0
    assert ops_bits.permute_rx(empty, perm=[0, 2, 1]).bits.size == 0
    assert (
        ops_bits.codebook_rx(empty, code_bits=2, data_bits=1, table=[1, 2]).bits.size
        == 0
    )


def test_blind_scope_still_decodes_the_whole_stream() -> None:
    blind = CodingCarrier(bits=_alternating(700))
    assert blind.windows is None
    decoded = ops_bits.block_code_rx(
        blind, code_bits=7, data_bits=4, parity_masks=HAMMING74
    )
    assert decoded.bits.size == 400
    assert ops_bits.permute_rx(blind, perm=[0, 2, 1]).bits.size == 699


def test_seeded_empty_descramble_is_a_passthrough() -> None:
    bits = _alternating(64)
    out = ops_bits.descramble_rx(CodingCarrier(bits=bits, windows=[]), sequence="ff")
    assert np.array_equal(out.bits, bits)


def test_seeded_empty_realign_does_not_slice() -> None:
    bits = _alternating(64)
    out = ops_bits.realign_rx(CodingCarrier(bits=bits, windows=[]), bit_offset=8)
    assert np.array_equal(out.bits, bits)
    assert out.windows == []


def test_failed_acquisition_reports_empty_not_fabricated_bits(tmp_path: Path) -> None:
    # "deadbeef" has adjacent equal bits; an alternating stream can never match
    bits = _alternating(700)
    p = tmp_path / "in.u8"
    bits.tofile(p)
    modem = Modem(
        symbol_rate=1.0,
        path=[
            SyncWordStep(sync="deadbeef"),
            BlockCodeStep(code_bits=7, data_bits=4, parity_masks=HAMMING74),
        ],
    )
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=p, num_bits=int(bits.size)),
    )
    assert res.status == "empty"
    assert res.bitstream is None and res.symbolstream is None
    assert res.stalled_at == "sync_word[0]"
    assert res.error is not None and "0 items" in res.error
    sync_row, block_row = res.census
    assert sync_row.windows_in is None and sync_row.windows_out == 0
    assert block_row.windows_in == 0 and block_row.windows_out == 0
