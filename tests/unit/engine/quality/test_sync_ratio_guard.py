"""Sync counts need a chance-ratio, not only sigma: structured interference
(live 1090 MHz RFI) produced 332 matches vs 136 chance = 16 sigma with 100%
garbage frames. The 3x ratio kills that class; real gates run >>3x."""

import pytest

from marconi.engine.quality import (
    _SYNC_NEGATIVE_MIN_ITEMS,
    Assessment,
    _sync_assessment,
)

_SCANNED = 10 * _SYNC_NEGATIVE_MIN_ITEMS


def test_measured_live_false_positive_is_not_positive() -> None:
    # Pinned regression: the 2026-08-08 ADS-B blind dogfood false positive.
    assert _sync_assessment(332, 136.0, _SCANNED) is None


@pytest.mark.parametrize(
    ("found", "expected", "verdict"),
    [
        (1350, 136.0, "positive"),  # ~10x: real-gate class
        (408, 136.0, "positive"),  # exactly 3.0x and >5 sigma
        (407, 136.0, None),  # just under 3x
        # Was pinned "positive" as the zero-chance path UNCHANGED — a record of
        # what the ratio-guard commit did not touch, not a decision that a
        # missing null can certify. It cannot: both bars multiply the
        # expectation, so 0.0 clears them for any count. The GR-lane twin
        # already refused this exact reading (test_tag_sync_without_a_chance_
        # cannot_certify), so the tree asserted both answers at once.
        (3, 0.0, None),  # no null to judge against: untestable
        (0, 136.0, "negative"),  # absent, over a stream long enough to say so
        (140, 136.0, None),  # inside sigma band stays untestable
    ],
)
def test_ratio_and_sigma_are_both_required(
    found: int, expected: float, verdict: str | None
) -> None:
    assert _sync_assessment(found, expected, _SCANNED) == verdict


@pytest.mark.parametrize("scanned", [0, 1, 2, _SYNC_NEGATIVE_MIN_ITEMS - 1])
def test_zero_hits_over_too_little_stream_is_untestable(scanned: int) -> None:
    """The positive side got a chance model because a long pattern drives the
    expectation toward zero; the negative side kept answering without one. A
    64-bit pattern over a 65-bit stream is two search positions and an
    expectation of ~1e-9 — finding nothing there is what the NULL predicts, and
    it read as a definitive no_signal."""
    assert _sync_assessment(0, 1e-9, scanned) is None
    assert _sync_assessment(0, 136.0, scanned) is None


def test_zero_hits_over_a_long_enough_stream_is_still_negative() -> None:
    # the verdict itself, not `is not None`: a POSITIVE would have satisfied
    # the old assertion while the name promises negative
    assert _sync_assessment(0, 1e-9, _SYNC_NEGATIVE_MIN_ITEMS) is Assessment.NEGATIVE
