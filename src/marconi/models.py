from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator


class CaptureRef(BaseModel):
    """Reference to an IQ capture on disk (.sigmf-data + .sigmf-meta sidecar).

    Frequencies are in Hz, durations in seconds.
    """

    path: Path
    center_freq: float
    sample_rate: float
    num_samples: int
    datatype: str = "cf32_le"

    @field_validator("sample_rate")
    @classmethod
    def _sample_rate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sample_rate must be > 0")
        return v

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
    kind: str
