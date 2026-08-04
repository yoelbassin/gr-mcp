from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    start: int
    cursor: int


@dataclass(frozen=True)
class StepStats:
    """What a step measured about its own decode, for the quality extractors:
    how many windows pure chance would have seeded on this input, and how many
    codewords passed validation against the rate a random word passes. Census
    annotation, never data — run_coding strips it before the next step."""

    chance_windows: float | None = None
    words_valid: int | None = None
    words_total: int | None = None
    chance_word_rate: float | None = None


@dataclass(frozen=True)
class CodingCarrier:
    """windows is a scope, not a container: None means no seeder ran (blind,
    whole-stream ops are correct); [] means a seeder ran and found nothing
    (decoding anything from it would fabricate data)."""

    bits: np.ndarray
    windows: list[Window] | None = None
    symbols: np.ndarray | None = None
    marks: tuple[int, ...] = ()
    stats: StepStats | None = None
