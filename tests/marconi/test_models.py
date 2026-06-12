from pathlib import Path

from marconi.models import Burst, CaptureRef, DetectedSignal, PSDResult, SignalPeak


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
