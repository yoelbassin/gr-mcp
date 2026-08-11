from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from marconi.engine.compile.ir import GrPipeline
from marconi.errors import register_error


class BackendError(Exception):
    pass


class BackendUnavailable(BackendError):
    pass


# Backend wiring/resolution failures (unknown block kind, bad connection): the
# spec the caller handed in is wrong, not an internal bug. A backend whose
# runtime is missing from the machine is an environment failure, not a spec
# error, and must not tell the agent to fix its spec.
register_error(BackendError, "invalid_argument")
register_error(BackendUnavailable, "failed_precondition")


class BlockCensus(BaseModel):
    """How many items a block consumed and produced. None where the port does
    not exist (a source has no input, a sink no output). The chance_* and
    words_* fields are set only by steps that measure them: chance_windows is
    how many windows pure chance would seed on this input, words_valid/total
    count codewords that passed validation, chance_word_rate is the
    probability a random word passes."""

    block: str
    kind: str
    items_in: int | None = None
    items_out: int | None = None
    windows_in: int | None = None
    windows_out: int | None = None
    chance_windows: float | None = None
    words_valid: int | None = None
    words_total: int | None = None
    chance_word_rate: float | None = None


class DiagnosticKey(StrEnum):
    """The whole vocabulary of the block -> verdict channel. Producers
    (embedded blocks) and consumers (quality extractors, run) both name keys
    from here: the channel is a bare dict[str, ...] across a process boundary,
    so a producer-side rename that no consumer follows cannot be caught by
    types. It degrades every affected verdict to "uncertain" while the run
    still reports ok — which is exactly how the equalizer's lock statistic
    went unread. Counters a block keeps for observability only (locks,
    frames_emitted, bursts_flushed) are deliberately not listed; nothing
    downstream judges on them."""

    SYNC_TAGS = "sync_tags"
    SYNC_CHANCE = "sync_chance"
    SYNC_ITEMS_SCANNED = "sync_items_scanned"
    LOCK_RATIO_BEST = "lock_ratio_best"
    LOCK_MIN = "lock_min"
    DOMINANT_SYMBOLS = "dominant_symbols"
    SYMBOLS_TOTAL = "symbols_total"
    DOMINANCE_CHANCE = "dominance_chance"
    BURSTS = "bursts"


class Diagnostic(BaseModel):
    block: str
    key: str
    count: int | None = None
    value: float | None = None
    marks: list[int] | None = None


def find_diagnostic(
    rows: Sequence[Diagnostic], block: str, key: str
) -> Diagnostic | None:
    return next((d for d in rows if d.block == block and d.key == key), None)


class DiagnosticRows:
    """Reads one harvest of diagnostics by key. Blocks emit from a per-block
    dict, so a (block, key) pair appears at most once and indexing by block is
    lossless."""

    def __init__(self, rows: Sequence[Diagnostic]) -> None:
        self._rows = rows

    def counts(self, key: str) -> dict[str, int]:
        return {
            d.block: d.count for d in self._rows if d.key == key and d.count is not None
        }

    def values(self, key: str) -> dict[str, float]:
        return {
            d.block: d.value for d in self._rows if d.key == key and d.value is not None
        }

    def latest_marks(self, key: str) -> list[int]:
        for d in reversed(self._rows):
            if d.key == key and d.marks is not None:
                return sorted({int(m) for m in d.marks})
        return []


class RunResult(BaseModel):
    # "empty": the graph ran clean but its terminal sink wrote nothing — a
    # decoded-nothing run is never reported as "ok".
    status: Literal["ok", "error", "timeout", "empty"]
    artifacts: list[str] = []
    error: str | None = None
    # the block id where the signal died on an "empty" run (first stage whose
    # items_out fell to zero), or None. The machine anchor for the census gradient.
    stalled_at: str | None = None
    # per-block counters/marks harvested after the run, e.g.
    # Diagnostic(block="b4", key="locks", count=2) or
    # Diagnostic(block="b7", key="bursts", marks=[0, 512]) — lets a caller
    # tell "no signal found" from silence, or recover per-burst symbol offsets
    diagnostics: list[Diagnostic] = []
    # in pipeline order, so the row where items_out falls to zero is the stage
    # that consumed the signal — the gradient a parameter search needs
    census: list[BlockCensus] = []


class Backend(ABC):
    """The swap seam between the compiler's IR and an execution engine. The
    GR-3.10 backend implements it today; GR-4.0 is the planned second. The IR is
    portable across GR versions above this seam — GR keeps most block names, so
    the port is largely a kind->factory remap, not an engine-neutral rewrite."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def instantiate(self, pipeline: GrPipeline) -> object:
        """Resolve every block kind and wire connections WITHOUT running.
        Raise BackendError on an unresolvable kind or an invalid connection."""

    @abstractmethod
    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        """Execute the wired pipeline to completion (file-to-file) and return its
        status + sink artifacts. Execution belongs to the seam so a second backend
        inherits the run model rather than re-authoring it."""
