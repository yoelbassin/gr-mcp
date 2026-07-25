"""Build a rate-1/n terminated convolutional Viterbi from gnuradio.trellis.
Generic over (rate_inv, polys, frame_bits, tail); the soft-metric table is
derived from the FSM output cardinality, so no protocol constant lives here."""

from __future__ import annotations

from typing import Any


def make_trellis_viterbi(
    ctx: Any, *, rate_inv: int, polys: list, frame_bits: int, tail: int
) -> Any:
    fsm = ctx.trellis.fsm(1, rate_inv, [int(p) for p in polys])
    o_card, d, k = fsm.O(), rate_inv, frame_bits + tail
    table = [
        1.0 - 2.0 * ((o >> (d - 1 - j)) & 1) for o in range(o_card) for j in range(d)
    ]
    return ctx.trellis.viterbi_combined_fb(
        fsm, k, 0, 0, d, table, ctx.digital.TRELLIS_EUCLIDEAN
    )
