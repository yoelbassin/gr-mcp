from abc import ABC, abstractmethod

from marconi.models import DeviceInfo, PipelineSpec, RunResult


class BackendError(Exception):
    """A failure inside the sample engine, translated for the agent."""


class Backend(ABC):
    """The swap boundary: everything that moves samples or touches devices."""

    name: str

    @abstractmethod
    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        """Run a validated pipeline to completion (bounded by head blocks),
        supervised by `timeout` seconds."""

    @abstractmethod
    def enumerate_devices(self) -> list[DeviceInfo]:
        """Hardware devices this backend can drive (none in v1.0)."""
