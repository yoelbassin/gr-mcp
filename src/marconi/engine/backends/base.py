from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from marconi.engine.compile.ir import GrPipeline
from marconi.errors import register_error


class BackendError(Exception):
    pass


# Backend wiring/resolution failures (unknown block kind, bad connection): the
# spec the caller handed in is wrong, not an internal bug.
register_error(BackendError, "invalid_argument")


class BlockCensus(BaseModel):
    """How many items a block consumed and produced. None where the port does
    not exist (a source has no input, a sink no output)."""

    block: str
    kind: str
    items_in: int | None = None
    items_out: int | None = None
    windows_in: int | None = None
    windows_out: int | None = None


class Diagnostic(BaseModel):
    block: str
    key: str
    count: int | None = None
    marks: list[int] | None = None


def find_diagnostic(
    rows: Sequence[Diagnostic], block: str, key: str
) -> Diagnostic | None:
    return next((d for d in rows if d.block == block and d.key == key), None)


class RunResult(BaseModel):
    # "empty": the graph ran clean but its terminal sink wrote nothing — a
    # decoded-nothing run is NOT a success, so status stops saying "ok" for it.
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
