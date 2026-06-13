from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RunStatus = Literal["ok", "timeout", "error"]


class CaptureRef(BaseModel):
    """Reference to an IQ capture on disk (.sigmf-data + .sigmf-meta sidecar).

    Frequencies are in Hz, durations in seconds.
    """

    path: Path
    center_freq: float
    sample_rate: float = Field(gt=0)
    num_samples: int = Field(ge=0)
    datatype: Literal["cf32_le"] = "cf32_le"

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate


class SignalPeak(BaseModel):
    """A detected spectral peak. Frequencies in Hz, powers in dB."""

    freq: float
    power_db: float


class PSDResult(BaseModel):
    """Power spectral density result. Frequencies in Hz, powers in dB."""

    freqs: list[float]
    psd_db: list[float]
    noise_floor_db: float
    peaks: list[SignalPeak]

    @model_validator(mode="after")
    def _lengths_match(self) -> "PSDResult":
        if len(self.freqs) != len(self.psd_db):
            raise ValueError("freqs and psd_db must have the same length")
        return self


class DetectedSignal(BaseModel):
    """A signal detected in a capture.

    Frequencies and bandwidths in Hz, powers in dB.
    """

    center_freq: float
    bandwidth: float
    peak_power_db: float
    snr_db: float


class SignalMeasurement(BaseModel):
    """Measured properties of a signal. Frequencies in Hz, powers in dB.

    occupied_bw_99: 99% occupied bandwidth in Hz.
    """

    center_freq: float
    occupied_bw_99: float
    power_db: float
    snr_db: float


class Burst(BaseModel):
    """A detected burst event. Times and durations in seconds, powers in dB."""

    start_time: float
    duration: float
    mean_power_db: float


class RenderResult(BaseModel):
    """Result of a render operation (e.g. spectrogram image)."""

    path: Path
    kind: Literal["spectrogram", "psd", "constellation"]


class BlockSpec(BaseModel):
    """One block instance in a pipeline; type names come from marconi.vocabulary."""

    id: str
    type: str
    params: dict[str, float | int | str | bool] = {}


class ConnectionSpec(BaseModel):
    """Directed edge between block ports (port indices default to 0)."""

    src_block: str
    src_port: int = Field(0, ge=0)
    dst_block: str
    dst_port: int = Field(0, ge=0)


class PipelineSpec(BaseModel):
    """Engine-agnostic DSP graph. sample_rate in Hz is the default rate for
    rate-dependent blocks that don't set their own."""

    name: str = "pipeline"
    sample_rate: float = Field(gt=0)
    blocks: list[BlockSpec]
    connections: list[ConnectionSpec]


class SceneElement(BaseModel):
    """One emitter in a simulated RF environment. freq is the absolute Hz
    position (ignored for kind='noise'); extra knobs go in params."""

    kind: Literal["tone", "noise", "fm_tone", "iq_file"]
    freq: float = 0.0
    amplitude: float = 1.0
    params: dict[str, float | int | str] = {}


class SceneSpec(BaseModel):
    """What is 'on the air' for a simulated device."""

    name: str = "scene"
    elements: list[SceneElement] = []


class DeviceInfo(BaseModel):
    id: str
    kind: Literal["simulated"]  # hardware kinds arrive in v1.1
    can_tx: bool = False
    description: str = ""


class ValidationIssue(BaseModel):
    """One actionable pipeline validation error, addressed to the agent."""

    block_id: str | None = None
    field: str | None = None
    message: str


class RunResult(BaseModel):
    """Outcome of a pipeline run."""

    status: RunStatus
    elapsed_seconds: float
    artifacts: list[Path] = []
    error: str | None = None


class RunRecord(BaseModel):
    """A pipeline run's history entry: its run_id and pipeline name plus the
    flattened RunResult. This is the shape run_pipeline returns and list_runs
    lists (artifacts as workspace path strings)."""

    run_id: str
    pipeline: str
    status: RunStatus
    elapsed_seconds: float
    artifacts: list[str] = []
    error: str | None = None
