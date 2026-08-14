from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest
from helpers._deadline import deadline_expiring_after

from marconi.deadline import RunTimeout, check_deadline, set_deadline
from marconi.engine.coding import ops_bits, ops_symbols
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import _correlate, _decode_scoped, rs_code_rx
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


def test_the_poll_clock_expires_on_the_poll_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two tests below are only worth anything if the deadline is still
    ALIVE when the op is entered — a clock expired from the start would let an
    entry poll answer them, which is the blind spot they exist to close."""
    with deadline_expiring_after(monkeypatch, polls=3) as clock:
        for _ in range(3):
            check_deadline()
        with pytest.raises(RunTimeout):
            check_deadline()
    assert clock.reads == 5  # set_deadline, three that survived, one that fired


def test_segment_polls_while_it_tiles_not_only_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spent-deadline gate above is answered by segment's ENTRY poll and
    can say nothing about the loop after it. Measured on the unfixed op:
    frame_body_len=1 (Field(ge=1) accepts it) over 2^22 bits built 4,194,304
    Window objects in 2.417 s and 708 MB after its only poll returned."""
    carrier = CodingCarrier(bits=np.zeros(1 << 22, np.uint8))
    start = time.monotonic()
    with deadline_expiring_after(monkeypatch, polls=1):
        with pytest.raises(RunTimeout):
            ops_bits.segment_rx(carrier, frame_body_len=1)
    assert time.monotonic() - start < 1.0


def test_the_correlator_polls_while_it_collects_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count phase polls once per pattern bit; the hit loop that walks
    every candidate position polled nowhere. Measured: a 16-bit pattern at
    max_errors=8 over 2^24 bits left 10,036,821 candidates and spent 0.617 s
    in pure Python collecting them — and _surrogate_chance runs the same
    correlation twice more."""
    rng = np.random.default_rng(0)
    stream = rng.integers(0, 2, 1 << 24).astype(np.uint8)
    pat = rng.integers(0, 2, 16).astype(np.uint8)
    start = time.monotonic()
    with deadline_expiring_after(monkeypatch, polls=int(pat.size)):
        with pytest.raises(RunTimeout):
            _correlate(stream, pat, 8)
    assert time.monotonic() - start < 1.0


def test_an_rs_decode_returns_within_the_timeout_it_was_given() -> None:
    """Same wall-clock guarantee as the coding run above, for the one stage
    whose PER-WORD cost is unbounded by anything the poll cadence knew about:
    reedsolo is pure Python here and a word costs n*(n-k)/4.5e6 seconds, so
    the fixed 256-word cadence left minutes between polls."""
    asked = 1.0
    words = 10
    n, nsym, symbol_bits = 8191, 256, 13
    bits = (
        np.random.default_rng(0)
        .integers(0, 2, words * n * symbol_bits)
        .astype(np.uint8)
    )
    start = time.monotonic()
    with set_deadline(asked), pytest.raises(RunTimeout):
        rs_code_rx(
            CodingCarrier(bits=bits),
            symbol_bits=symbol_bits,
            n=n,
            k=n - nsym,
            prim_poly=0x201B,
        )
    elapsed = time.monotonic() - start
    assert (
        elapsed < asked * 3.0
    ), f"overran its {asked}s deadline by {elapsed / asked:.1f}x"


def test_rs_work_per_word_is_bounded_at_the_spec() -> None:
    # a poll cadence bounds the deadline's granularity at ONE word, so the
    # spec has to bound what one word can cost: measured 4.5 Mwork/s over
    # n*(n-k) (flat within 7% from (255,32) to (65535,256)), and 2.7 Mwork/s
    # on the correctable path that runs Forney
    from pydantic import ValidationError

    from marconi.engine.coding.stages_bits import RsCodeStep

    with pytest.raises(ValidationError, match="symbol operations"):
        RsCodeStep(
            conv="rs_code", symbol_bits=16, n=65535, k=65535 - 256, prim_poly=0x1100B
        )
    wide = RsCodeStep(
        conv="rs_code", symbol_bits=13, n=8191, k=8191 - 256, prim_poly=0x201B
    )
    assert wide.n * (wide.n - wide.k) <= 1 << 21
    ok = RsCodeStep(conv="rs_code", symbol_bits=8, n=255, k=223, prim_poly=0x11D)
    assert ok.n - ok.k == 32


def test_rs_parity_width_is_bounded_at_the_spec() -> None:
    # the generator polynomial build is O((n-k)^2) pure Python with no poll
    # inside the library: n=65535,k=1 validated in 0.0 s and ran 752.8 s
    # past a 2.0 s deadline on EMPTY input
    import pytest
    from pydantic import ValidationError

    from marconi.engine.coding.stages_bits import RsCodeStep

    with pytest.raises(ValidationError, match="parity symbols"):
        RsCodeStep(conv="rs_code", symbol_bits=16, n=65535, k=1, prim_poly=0x1100B)
    ok = RsCodeStep(conv="rs_code", symbol_bits=8, n=255, k=223, prim_poly=0x11D)
    assert ok.n - ok.k == 32


def test_syndrome_loop_polls_the_deadline() -> None:
    # O(n_parity x data_bits) numpy passes with no poll: a legal (4096, 2048)
    # spec measured 3.85 s for ONE codeword under a 0.5 s deadline
    import numpy as np
    import pytest

    from marconi.deadline import RunTimeout, set_deadline
    from marconi.engine.coding.ops_bits import _syndrome_bits

    words = np.zeros((4, 4096), np.uint8)
    masks = [(1 << 2048) - 1] * 2048
    with pytest.raises(RunTimeout):
        with set_deadline(0.05):
            _syndrome_bits(words, masks, 2048)


def test_codebook_exact_reports_the_out_of_table_fraction() -> None:
    # an out-of-table codeword silently decodes to data value 0 (uniform
    # noise through a Manchester book emitted a plausible P(1)=0.25 stream
    # with stats=None); the in-table fraction is census-only honesty -
    # Codebook.validates_words stays False so it never touches the verdict
    import numpy as np

    from marconi.engine.coding.carrier import CodingCarrier
    from marconi.engine.coding.ops_bits import codebook_rx
    from marconi.engine.stages.registry import stage_registry

    # Manchester-shaped 2-entry book; chips 00 and 11 are out-of-table
    chips = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1], np.uint8)
    out = codebook_rx(CodingCarrier(bits=chips), code_bits=2, data_bits=1, table=[1, 2])
    assert out.stats is not None
    assert out.stats.words_total == 5
    assert out.stats.words_valid == 3  # 01, 10, 01 in-table; 00, 11 not
    assert out.stats.chance_word_rate is None  # census-only, no chance model
    assert stage_registry()["codebook"].validates_words is False
