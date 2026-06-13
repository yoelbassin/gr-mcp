from pathlib import Path

import numpy as np

import marconi


def test_analysis_api_surface() -> None:
    for name in (
        "Workspace",
        "CaptureRef",
        "load_capture",
        "psd",
        "find_signals",
        "measure",
        "detect_bursts",
        "spectrogram",
        "psd_plot",
        "constellation",
        "write_capture",
        "read_capture",
    ):
        assert hasattr(marconi, name), name


def test_end_to_end_analysis_flow(tmp_path: Path) -> None:
    """Synthesize IQ -> workspace capture -> find -> measure -> render."""
    ws = marconi.Workspace(tmp_path)
    fs = 1e6
    t = np.arange(int(fs * 0.05)) / fs
    rng = np.random.default_rng(0)
    x = (
        np.exp(2j * np.pi * 100e3 * t)
        + rng.normal(0, 0.01, len(t))
        + 1j * rng.normal(0, 0.01, len(t))
    ).astype(np.complex64)

    ref = marconi.write_capture(
        x, ws.new_capture_path("synth"), center_freq=433e6, sample_rate=fs
    )
    signals = marconi.find_signals(ref)
    assert len(signals) == 1

    m = marconi.measure(ref, center_freq=signals[0].center_freq)
    assert m.snr_db > 20

    render = marconi.spectrogram(ref, ws)
    assert render.path.exists()
    # the workspace now holds a reusable RF project
    assert (ws.root / "artifacts" / "captures").is_dir()
    assert (ws.root / "artifacts" / "renders").is_dir()
