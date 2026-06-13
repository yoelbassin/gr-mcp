import numpy as np
import pytest
from scipy.io import wavfile

from marconi.mcp import tools as T

pytestmark = pytest.mark.gnuradio


def test_fm_receive_loop_through_tools(server_state, tmp_path):
    # 1. The world
    T.simulate_scene(
        "sim0",
        [
            {
                "kind": "fm_tone",
                "freq": 100.3e6,
                "amplitude": 1.0,
                "params": {"mod_freq": 1e3},
            },
            {"kind": "tone", "freq": 99.5e6, "amplitude": 0.4},
            {"kind": "noise", "amplitude": 0.005},
        ],
    )
    # 2. Survey
    cap = T.capture("sim0", center_freq=100e6, sample_rate=2e6, duration=0.25)
    sigs = T.find_signals(cap["path"])
    assert len(sigs) >= 2
    fm = max(sigs, key=lambda s: s["bandwidth"])
    assert abs(fm["center_freq"] - 100.3e6) < 5e3
    # 3. Build + run a NBFM receiver via the tools
    wav = str(tmp_path / "audio.wav")
    pipeline = {
        "name": "fm_rx",
        "sample_rate": 2e6,
        "blocks": [
            {"id": "src", "type": "file_source", "params": {"path": cap["path"]}},
            {
                "id": "chan",
                "type": "freq_xlating_lowpass",
                "params": {
                    "decimation": 20,
                    "center_offset": fm["center_freq"] - 100e6,
                    "cutoff": 8e3,
                    "transition": 4e3,
                },
            },
            {
                "id": "demod",
                "type": "nbfm_rx",
                "params": {"audio_rate": 25000, "quad_rate": 100000},
            },
            {
                "id": "audio",
                "type": "wav_sink",
                "params": {"path": wav, "sample_rate": 25000},
            },
        ],
        "connections": [
            {"src_block": "src", "dst_block": "chan"},
            {"src_block": "chan", "dst_block": "demod"},
            {"src_block": "demod", "dst_block": "audio"},
        ],
    }
    assert T.validate_pipeline(pipeline) == []
    run = T.run_pipeline(pipeline)
    assert run["status"] == "ok"
    # 4. Verify recovered audio peaks at 1 kHz
    rate, audio = wavfile.read(wav)
    audio = audio.astype(np.float32)
    audio = audio - audio.mean()
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    peak = np.fft.rfftfreq(len(audio), 1 / rate)[int(np.argmax(spec))]
    assert abs(peak - 1e3) < 50


def test_tx_rx_loop_through_tools(server_state):
    # Receiver device: just a noise floor
    T.simulate_scene("rx0", [{"kind": "noise", "amplitude": 0.001}])
    # Payload: a tone 10 kHz above 100 MHz, recorded at 1 Msps
    payload = T.render_scene(
        [{"kind": "tone", "freq": 100.01e6, "amplitude": 0.5}],
        center_freq=100e6,
        sample_rate=1e6,
        duration=0.1,
    )
    # Transmit it into the receiver's scene (same rate), then receive it back
    el = T.transmit_capture("rx0", payload["path"], freq=100e6, confirmed=True)
    assert el["kind"] == "iq_file"
    cap = T.capture("rx0", center_freq=100e6, sample_rate=1e6, duration=0.1)
    sigs = T.find_signals(cap["path"])
    assert any(abs(s["center_freq"] - 100.01e6) < 5e3 for s in sigs)
