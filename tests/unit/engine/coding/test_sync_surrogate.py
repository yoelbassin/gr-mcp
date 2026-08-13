from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding import ops_bits
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.program import CodingStep, _row
from marconi.engine.quality import Verdict, assess_quality
from marconi.engine.stages.registry import stage_registry

STREAM_BITS = 200_000


def _verdict(stream: np.ndarray, **search: object) -> tuple[Verdict, float, int]:
    before = CodingCarrier(bits=stream)
    after = ops_bits.sync_word_rx(before, **search)  # type: ignore[arg-type]
    row = _row(
        CodingStep(name="sync_word[0]", kind="sync_word", call=lambda c: c),
        before,
        after,
    )
    report = assess_quality(
        registry=stage_registry(),
        census=[row],
        diagnostics=[],
        marks=[],
        soft_stream=None,
    )
    assert row.chance_windows is not None and row.windows_out is not None
    return report.verdict, row.chance_windows, row.windows_out


def _planted(rng: np.random.Generator, pattern: str, every: int) -> np.ndarray:
    """Uniform payload with the sync word planted at a fixed cadence — what a
    real framed transmission looks like after a correct demod."""
    pat = np.array([int(ch) for ch in pattern], np.uint8)
    bits = rng.integers(0, 2, STREAM_BITS, dtype=np.uint8)
    for start in range(0, STREAM_BITS - pat.size, every):
        bits[start : start + pat.size] = pat
    return bits


def test_dead_all_zero_stream_is_not_decoded() -> None:
    # a demod stuck at zero (squelched, no antenna, wrong gain) searched for a
    # zero-heavy sync word: the uniform-bit expectation is 4.66e-05, so every
    # one of these 6250 matches used to clear both bars and read "decoded"
    verdict, chance, found = _verdict(np.zeros(STREAM_BITS, np.uint8), sync="00000000")
    assert found > 0
    assert chance == pytest.approx(found, rel=0.05)
    assert verdict is Verdict.UNCERTAIN


def test_idle_alternating_stream_is_not_decoded() -> None:
    idle = np.tile([1, 0], STREAM_BITS // 2).astype(np.uint8)
    verdict, chance, found = _verdict(idle, sync="aaaaaaaa")
    assert found > 0
    assert chance == pytest.approx(found, rel=0.05)
    assert verdict is Verdict.UNCERTAIN


def test_real_sync_in_uniform_payload_still_decodes() -> None:
    # the regression direction: the surrogate must not cost a genuine framed
    # signal its verdict. CCSDS-style 32-bit word, one frame per 2048 bits.
    pattern = "00011010110011111111110000011101"  # 0x1ACFFC1D
    bits = _planted(np.random.default_rng(0), pattern, every=2048)
    verdict, chance, found = _verdict(bits, bits=pattern)
    assert found >= 90
    assert chance < 1.0
    assert verdict is Verdict.DECODED


def test_surrogate_is_zero_on_uniform_bits() -> None:
    # a rotation of a real sync word matches uniform noise no more often than
    # the word itself, so the measured floor collapses to the uniform one
    rng = np.random.default_rng(1)
    pat = rng.integers(0, 2, 32, dtype=np.uint8)
    stream = rng.integers(0, 2, STREAM_BITS, dtype=np.uint8)
    assert ops_bits._surrogate_chance(stream, pat, 0) == 0.0


def test_surrogate_tracks_a_degenerate_stream() -> None:
    # every rotation of an all-zero word is an all-zero word, so the floor
    # rises to whatever the stream itself yields
    stream = np.zeros(4096, np.uint8)
    pat = np.zeros(16, np.uint8)
    assert ops_bits._surrogate_chance(stream, pat, 0) == len(
        ops_bits._correlate(stream, pat, 0)
    )
