from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from marconi.deadline import RunTimeout, set_deadline
from marconi.engine.coding import ops_bits, ops_symbols
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import _decode_scoped, rs_code_rx
from marconi.engine.coding.program import CodingProgram, CodingStep, run_coding


def test_run_coding_honors_deadline() -> None:
    prog = CodingProgram(
        steps=[CodingStep(name="noop", kind="noop", call=lambda c: c)],
    )
    carrier = CodingCarrier(bits=np.zeros(16, np.uint8))
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            run_coding(prog, carrier)


def test_rs_decode_honors_deadline() -> None:
    carrier = CodingCarrier(
        bits=np.random.default_rng(0).integers(0, 2, 8160).astype(np.uint8)
    )
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            rs_code_rx(carrier, symbol_bits=8, n=255, k=223, prim_poly=285)


def _bits(n: int) -> CodingCarrier:
    return CodingCarrier(
        bits=np.random.default_rng(0).integers(0, 2, n).astype(np.uint8)
    )


# Every coding op whose work grows with the INPUT rather than with a fixed
# constant. Each is entered with the deadline already spent, so a loop that
# only checks between steps runs to completion and fails here.
_UNBOUNDED_OPS: list[tuple[str, Callable[[], object]]] = [
    ("segment", lambda: ops_bits.segment_rx(_bits(400_000), frame_body_len=7)),
    (
        "sync_word",
        lambda: ops_bits.sync_word_rx(_bits(400_000), bits="1" * 64, max_errors=2),
    ),
    (
        "block_code",
        lambda: ops_bits.block_code_rx(
            _bits(400_000),
            code_bits=8,
            data_bits=4,
            parity_masks=[0b1011, 0b1101, 0b1110, 0b0111],
            correct_single=False,
        ),
    ),
    (
        "descramble_lfsr",
        lambda: ops_bits.descramble_rx(
            _bits(400_000), lfsr_mask=0b100101, lfsr_seed=1, lfsr_len=6
        ),
    ),
    (
        "normalize",
        lambda: ops_symbols.normalize_rx(
            CodingCarrier(
                bits=np.zeros(0, np.uint8),
                symbols=np.zeros(400_000, np.float32),
                marks=tuple(range(0, 200_000, 64)),
            ),
            span_symbols=4096,
        ),
    ),
]


@pytest.mark.parametrize(
    "name,call", _UNBOUNDED_OPS, ids=[n for n, _ in _UNBOUNDED_OPS]
)
def test_every_input_sized_coding_op_polls_the_deadline(
    name: str, call: Callable[[], object]
) -> None:
    with set_deadline(0.0), pytest.raises(RunTimeout):
        call()


def test_a_window_scoped_decode_polls_between_windows() -> None:
    seeded = ops_bits.segment_rx(_bits(400_000), frame_body_len=7)
    assert seeded.windows and len(seeded.windows) > 1000
    with set_deadline(0.0), pytest.raises(RunTimeout):
        _decode_scoped(seeded, lambda b: (b, None))


def test_a_coding_run_returns_within_the_timeout_it_was_given() -> None:
    """The guarantee is a WALL CLOCK one, so the test measures it. Asserting
    that check_deadline appears in a loop proves nothing about the loop that
    does not have it: this path measured 16.2s against a 2.0s timeout, and
    flat at 16.3s whether 2s or 5s was asked — the deadline was only seen
    between steps."""
    asked = 1.0
    carrier = _bits(4_000_000)
    prog = CodingProgram(
        steps=[
            CodingStep(
                name="segment[0]",
                kind="segment",
                call=lambda c: ops_bits.segment_rx(c, frame_body_len=7),
            ),
            CodingStep(
                name="block_code[1]",
                kind="block_code",
                call=lambda c: ops_bits.block_code_rx(
                    c,
                    code_bits=7,
                    data_bits=4,
                    parity_masks=[0b1011, 0b1101, 0b1110],
                    correct_single=True,
                ),
            ),
        ],
    )
    start = time.monotonic()
    with set_deadline(asked), pytest.raises(RunTimeout):
        run_coding(prog, carrier)
    elapsed = time.monotonic() - start
    assert (
        elapsed < asked * 3.0
    ), f"overran its {asked}s deadline by {elapsed / asked:.1f}x"
