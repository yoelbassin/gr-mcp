"""The killer demo as a test: three unknown signals — find them, identify
the FM one, demodulate it, verify the audio."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

import marconi


@pytest.fixture(autouse=True)
def _fresh_registry():
    marconi.clear_devices()
    yield
    marconi.clear_devices()


def test_public_api_surface() -> None:
    for name in (
        "PipelineSpec",
        "SceneSpec",
        "SceneElement",
        "BlockSpec",
        "ConnectionSpec",
        "RunResult",
        "DeviceInfo",
        "validate_pipeline",
        "run_pipeline",
        "save_pipeline",
        "load_pipeline",
        "save_scene",
        "load_scene",
        "scene_to_pipeline",
        "render_scene",
        "add_simulated_device",
        "list_devices",
        "get_device",
        "clear_devices",
        "capture",
        "transmit_capture",
        "export_grc",
    ):
        assert hasattr(marconi, name), name


def test_killer_demo_full_loop(tmp_path: Path) -> None:
    ws = marconi.Workspace(tmp_path)

    # 1. The world: three signals the agent doesn't know about
    scene = marconi.SceneSpec(
        name="three_signals",
        elements=[
            marconi.SceneElement(
                kind="fm_tone",
                freq=100.3e6,
                amplitude=1.0,
                params={"mod_freq": 1e3},
            ),
            marconi.SceneElement(kind="tone", freq=99.5e6, amplitude=0.4),
            marconi.SceneElement(kind="noise", amplitude=0.005),
        ],
    )
    marconi.add_simulated_device("sim0", scene)
    assert [d.id for d in marconi.list_devices()] == ["sim0"]

    # 2. Survey the band
    cap = marconi.capture(
        "sim0", center_freq=100e6, sample_rate=2e6, duration=0.25, workspace=ws
    )
    signals = marconi.find_signals(cap)
    assert len(signals) >= 2
    fm = max(signals, key=lambda s: s.bandwidth)  # FM is the wide one
    assert abs(fm.center_freq - 100.3e6) < 5e3

    # 3. Build, save, and run the receiver
    rx = marconi.PipelineSpec(
        name="fm_rx",
        sample_rate=2e6,
        blocks=[
            marconi.BlockSpec(
                id="src", type="file_source", params={"path": str(cap.path)}
            ),
            marconi.BlockSpec(
                id="chan",
                type="freq_xlating_lowpass",
                params={
                    "decimation": 20,
                    "center_offset": fm.center_freq - 100e6,
                    "cutoff": 8e3,
                    "transition": 4e3,
                },
            ),
            marconi.BlockSpec(
                id="demod",
                type="nbfm_rx",
                params={"audio_rate": 25000, "quad_rate": 100000},
            ),
            marconi.BlockSpec(
                id="audio",
                type="wav_sink",
                params={"path": str(tmp_path / "audio.wav"), "sample_rate": 25000},
            ),
        ],
        connections=[
            marconi.ConnectionSpec(src_block="src", dst_block="chan"),
            marconi.ConnectionSpec(src_block="chan", dst_block="demod"),
            marconi.ConnectionSpec(src_block="demod", dst_block="audio"),
        ],
    )
    assert marconi.validate_pipeline(rx) == []
    result = marconi.run_pipeline(rx)
    assert result.status == "ok"

    # 4. Verify: the demodulated audio is dominated by the 1 kHz tone
    rate, audio = wavfile.read(tmp_path / "audio.wav")
    assert rate == 25000
    audio = audio.astype(np.float32)
    audio = audio - audio.mean()
    n = len(audio)
    assert n > 1000
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(n)))
    peak_freq = np.fft.rfftfreq(n, 1 / rate)[int(np.argmax(spectrum))]
    assert abs(peak_freq - 1e3) < 50

    # 5. The durable artifacts
    from marconi.ops.pipeline import save_pipeline_to_workspace

    saved = save_pipeline_to_workspace(rx, ws)
    grc = marconi.export_grc(rx, ws.root / "pipelines" / "fm_rx.grc")
    assert saved.exists() and grc.exists()
    assert (ws.root / "captures").is_dir()
