"""Build a rate-1/n terminated convolutional Viterbi from gnuradio.trellis.
Generic over (rate_inv, polys, frame_bits, tail); the soft-metric table is
derived from the FSM output cardinality, so no protocol constant lives here."""

from __future__ import annotations

from typing import Any


def make_trellis_viterbi(
    ctx: Any, *, rate_inv: int, polys: list, frame_bits: int, tail: int, k: int
) -> Any:
    fsm = ctx.trellis.fsm(k, rate_inv, [int(p) for p in polys])
    o_card, d = fsm.O(), rate_inv
    steps = (frame_bits + tail) // k
    table = [
        1.0 - 2.0 * ((o >> (d - 1 - j)) & 1) for o in range(o_card) for j in range(d)
    ]
    end = 0 if tail else -1
    return ctx.trellis.viterbi_combined_fb(
        fsm, steps, end, end, d, table, ctx.digital.TRELLIS_EUCLIDEAN
    )
