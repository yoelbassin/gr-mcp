from pathlib import Path

import pytest
from pydantic import ValidationError

from marconi.models import (
    Burst,
    CaptureRef,
    DetectedSignal,
    PSDResult,
    RenderResult,
    SignalMeasurement,
    SignalPeak,
)


def test_capture_ref_duration() -> None:
    ref = CaptureRef(
        path=Path("captures/x.sigmf-data"),
        center_freq=100e6,
        sample_rate=1e6,
        num_samples=2_000_000,
    )
    assert ref.duration == 2.0
    assert ref.datatype == "cf32_le"


def test_capture_ref_roundtrip() -> None:
    ref = CaptureRef(
        path=Path("captures/x.sigmf-data"),
        center_freq=100e6,
        sample_rate=1e6,
        num_samples=10,
    )
    again = CaptureRef.model_validate_json(ref.model_dump_json())
    assert again == ref


def test_result_models_construct() -> None:
    psd = PSDResult(
        freqs=[1.0, 2.0],
        psd_db=[-90.0, -40.0],
        noise_floor_db=-90.0,
        peaks=[SignalPeak(freq=2.0, power_db=-40.0)],
    )
    assert psd.peaks[0].freq == 2.0
    sig = DetectedSignal(
        center_freq=100e6, bandwidth=12e3, peak_power_db=-40.0, snr_db=50.0
    )
    assert sig.bandwidth == 12e3
    burst = Burst(start_time=0.01, duration=0.005, mean_power_db=-30.0)
    assert burst.duration == 0.005


def test_signal_measurement_roundtrip() -> None:
    meas = SignalMeasurement(
        center_freq=433.92e6,
        occupied_bw_99=12_500.0,
        power_db=-55.3,
        snr_db=22.1,
    )
    again = SignalMeasurement.model_validate_json(meas.model_dump_json())
    assert again == meas
    assert again.occupied_bw_99 == 12_500.0


def test_render_result_roundtrip() -> None:
    result = RenderResult(path=Path("out/spectrogram.png"), kind="spectrogram")
    again = RenderResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert again.kind == "spectrogram"


def test_capture_ref_sample_rate_zero_raises() -> None:
    with pytest.raises(ValidationError):
        CaptureRef(
            path=Path("captures/x.sigmf-data"),
            center_freq=100e6,
            sample_rate=0,
            num_samples=1000,
        )


def test_psd_result_mismatched_lengths_raises() -> None:
    with pytest.raises(ValidationError):
        PSDResult(
            freqs=[1.0, 2.0, 3.0],
            psd_db=[-90.0, -40.0],
            noise_floor_db=-90.0,
            peaks=[],
        )
