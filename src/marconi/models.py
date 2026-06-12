from pathlib import Path

from pydantic import BaseModel


class CaptureRef(BaseModel):
    """Reference to an IQ capture on disk (.sigmf-data + .sigmf-meta sidecar)."""

    path: Path
    center_freq: float
    sample_rate: float
    num_samples: int
    datatype: str = "cf32_le"

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate


class SignalPeak(BaseModel):
    freq: float
    power_db: float


class PSDResult(BaseModel):
    freqs: list[float]
    psd_db: list[float]
    noise_floor_db: float
    peaks: list[SignalPeak]


class DetectedSignal(BaseModel):
    center_freq: float
    bandwidth: float
    peak_power_db: float
    snr_db: float


class SignalMeasurement(BaseModel):
    center_freq: float
    occupied_bw_99: float
    power_db: float
    snr_db: float


class Burst(BaseModel):
    start_time: float
    duration: float
    mean_power_db: float


class RenderResult(BaseModel):
    path: Path
    kind: str
