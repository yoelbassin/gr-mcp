from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Window:
    start: int
    cursor: int
    end: int | None = None


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

    bits: npt.NDArray[np.uint8]
    windows: list[Window] | None = None
    symbols: (
        npt.NDArray[np.signedinteger[Any]] | npt.NDArray[np.floating[Any]] | None
    ) = None
    marks: tuple[int, ...] = ()
    stats: StepStats | None = None

    def window_spans(self) -> list[tuple[int, int]]:
        """One (lo, hi) bit span per window: from its cursor to its own end
        when the seeder fixed one (fixed-length framing), else to the next
        window's cursor, the last to end-of-stream. Equal cursors are legal —
        a window whose upstream decode emitted nothing scopes an empty span,
        keeping window counts aligned across stages; a cursor advanced past
        its end (realign of a short window) scopes an empty span too."""
        if self.windows is None:
            raise ValueError("window_spans needs a seeded carrier")
        cursors = [int(w.cursor) for w in self.windows]
        bad = [(a, b) for a, b in zip(cursors, cursors[1:]) if b < a]
        if bad:
            raise ValueError(f"windows need non-decreasing cursors: {bad[:5]}")
        his = [
            nxt if w.end is None else int(w.end)
            for w, nxt in zip(self.windows, cursors[1:] + [int(self.bits.size)])
        ]
        return [(lo, max(lo, hi)) for lo, hi in zip(cursors, his)]
