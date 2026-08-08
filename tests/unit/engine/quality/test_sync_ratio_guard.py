"""Sync counts need a chance-ratio, not only sigma: structured interference
(live 1090 MHz RFI) produced 332 matches vs 136 chance = 16 sigma with 100%
garbage frames. The 3x ratio kills that class; real gates run >>3x."""

import pytest

from marconi.engine.quality import _sync_assessment


def test_measured_live_false_positive_is_not_positive() -> None:
    # Pinned regression: the 2026-08-08 ADS-B blind dogfood false positive.
    assert _sync_assessment(332, 136.0) is None


@pytest.mark.parametrize(
    ("found", "expected", "verdict"),
    [
        (1350, 136.0, "positive"),  # ~10x: real-gate class
        (408, 136.0, "positive"),  # exactly 3.0x and >5 sigma
        (407, 136.0, None),  # just under 3x
        (3, 0.0, "positive"),  # zero-chance path unchanged
        (0, 136.0, "negative"),  # absent stays negative
        (140, 136.0, None),  # inside sigma band stays untestable
    ],
)
def test_ratio_and_sigma_are_both_required(
    found: int, expected: float, verdict: str | None
) -> None:
    assert _sync_assessment(found, expected) == verdict
