from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from marconi.ops.render import _figure, constellation, psd_plot, spectrogram
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


def test_figure_closed_on_error() -> None:
    """_figure must close the figure even when the body raises — no leak."""
    before = set(plt.get_fignums())
    with pytest.raises(RuntimeError):
        with _figure((4.0, 4.0)):
            raise RuntimeError("boom")
    assert set(plt.get_fignums()) == before


def test_spectrogram_rejects_empty_capture(tmp_path: Path, make_iq) -> None:
    ws = Workspace(tmp_path / "project")
    ref = write_capture(
        make_iq([], duration=1e-6),  # 1 sample
        tmp_path / "tiny",
        center_freq=100e6,
        sample_rate=1e6,
    )
    with pytest.raises(ValueError, match="too short"):
        spectrogram(ref, ws)
