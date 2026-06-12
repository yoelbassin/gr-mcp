from pathlib import Path

from marconi.ops.render import constellation, psd_plot, spectrogram
from marconi.sigmf import write_capture
from marconi.workspace import Workspace

PNG_MAGIC = b"\x89PNG"


def _capture(tmp_path: Path, make_iq):
    return write_capture(
        make_iq([(100e3, 1.0)]),
        tmp_path / "cap",
        center_freq=100e6,
        sample_rate=1e6,
    )


def test_spectrogram_renders_png(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    result = spectrogram(_capture(tmp_path, make_iq), ws)
    assert result.kind == "spectrogram"
    assert result.path.read_bytes()[:4] == PNG_MAGIC
    assert result.path.stat().st_size > 5000


def test_psd_plot_renders_png(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    result = psd_plot(_capture(tmp_path, make_iq), ws)
    assert result.kind == "psd"
    assert result.path.read_bytes()[:4] == PNG_MAGIC


def test_constellation_renders_png(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    result = constellation(_capture(tmp_path, make_iq), ws)
    assert result.kind == "constellation"
    assert result.path.read_bytes()[:4] == PNG_MAGIC
