from pathlib import Path

from marconi import sigmf
from marconi.mcp import tools as T


def _capture(server_state, make_iq):
    x = make_iq([(50e3, 1.0)], sample_rate=1e6, duration=0.05)
    ref = sigmf.write_capture(
        x,
        server_state.workspace.new_capture_path("r"),
        center_freq=10e6,
        sample_rate=1e6,
    )
    return str(ref.path)


def test_spectrogram_writes_png(server_state, make_iq):
    out = T.spectrogram(_capture(server_state, make_iq))
    assert out["kind"] == "spectrogram"
    assert Path(out["path"]).exists() and out["path"].endswith(".png")


def test_psd_plot_writes_png(server_state, make_iq):
    out = T.psd_plot(_capture(server_state, make_iq))
    assert out["kind"] == "psd"
    assert Path(out["path"]).exists()


def test_constellation_writes_png(server_state, make_iq):
    out = T.constellation(_capture(server_state, make_iq))
    assert out["kind"] == "constellation"
    assert Path(out["path"]).exists()
