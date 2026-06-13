from marconi.backends.base import Backend
from marconi.models import DeviceInfo, PipelineSpec, RunResult


class GnuRadioBackend(Backend):
    """GNU Radio sample engine (build/run implemented in the next tasks)."""

    name = "gnuradio"

    def run_pipeline(self, spec: PipelineSpec, timeout: float = 30.0) -> RunResult:
        raise NotImplementedError  # Task 5/6

    def enumerate_devices(self) -> list[DeviceInfo]:
        return []
