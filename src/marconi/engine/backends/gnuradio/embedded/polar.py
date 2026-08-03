"""Polar SC / SC-list decode via stock gr-fec: fec.polar_decoder_sc (or
_sc_list) wrapped in fec.extended_decoder. Soft LLR in (bit 1 = positive; the
polar stage flips the engine's bit-1-negative convention upstream), K hard bits
out per N-bit frame. The frozen set (positions + values) and block size are
caller parameters — polar code construction is datasheet work, not the engine's.
"""

from __future__ import annotations

from typing import Any


def make_polar_decoder(
    ctx: Any,
    *,
    block_size: int,
    info_bits: int,
    frozen_positions: list[int],
    frozen_values: list[int],
    list_size: int,
) -> Any:
    if list_size > 1:
        decoder = ctx.fec.polar_decoder_sc_list.make(
            list_size, block_size, info_bits, frozen_positions, frozen_values
        )
    else:
        decoder = ctx.fec.polar_decoder_sc.make(
            block_size, info_bits, frozen_positions, frozen_values
        )
    return ctx.fec.extended_decoder(
        decoder, threading=None, ann=None, puncpat="11", integration_period=10000
    )
