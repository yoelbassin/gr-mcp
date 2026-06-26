from __future__ import annotations

from abc import ABC, abstractmethod

from marconi.phy.ir import GrPipeline


class BackendError(Exception):
    pass


class Backend(ABC):
    """The swap seam between the compiler's IR and an execution engine. The GR-3.x
    backend (Plan 3) and a future GR-4.0 backend implement this; the compiler and
    GrPipeline are engine-portable above it."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def instantiate(self, pipeline: GrPipeline) -> object:
        """Resolve every block kind and wire connections WITHOUT running.
        Raise BackendError on an unresolvable kind or an invalid connection."""
