import matplotlib

matplotlib.use("Agg")

from contextlib import contextmanager  # noqa: E402
from typing import Iterator  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from marconi.models import CaptureRef, RenderResult  # noqa: E402
from marconi.ops.analyze import psd  # noqa: E402
from marconi.sigmf import read_samples  # noqa: E402
from marconi.workspace import Workspace  # noqa: E402


@contextmanager
def _figure(figsize: tuple[float, float]) -> Iterator[tuple["plt.Figure", "plt.Axes"]]:
    """Yield (fig, ax) and ALWAYS close the figure on exit — including when the
    plotting body or savefig raises — so a render error never leaks a figure."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def _save(
    fig: "plt.Figure", workspace: Workspace, name: str, kind: str
) -> RenderResult:
    out = workspace.new_render_path(name)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    return RenderResult(path=out, kind=kind)


def spectrogram(
    capture: CaptureRef,
    workspace: Workspace,
    name: str = "spectrogram",
    nfft: int = 1024,
) -> RenderResult:
    x = read_samples(capture)
    if len(x) < 2:
        raise ValueError(f"capture too short to render: {len(x)} sample(s)")
    with _figure((10, 6)) as (fig, ax):
        spec, freqs, t, im = ax.specgram(
            x,
            NFFT=min(nfft, len(x)),
            Fs=capture.sample_rate,
            Fc=capture.center_freq,
            noverlap=min(nfft, len(x)) // 2,
        )
        fig.colorbar(im, ax=ax, label="Power (dB)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Spectrogram @ {capture.center_freq/1e6:.3f} MHz")
        return _save(fig, workspace, name, "spectrogram")


def psd_plot(
    capture: CaptureRef, workspace: Workspace, name: str = "psd"
) -> RenderResult:
    result = psd(capture)
    with _figure((10, 6)) as (fig, ax):
        freqs_mhz = np.array(result.freqs) / 1e6
        ax.plot(freqs_mhz, result.psd_db, linewidth=0.8)
        ax.axhline(
            result.noise_floor_db, linestyle="--", color="gray", label="noise floor"
        )
        for peak in result.peaks[:10]:
            ax.plot(peak.freq / 1e6, peak.power_db, "rv")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return _save(fig, workspace, name, "psd")


def constellation(
    capture: CaptureRef,
    workspace: Workspace,
    name: str = "constellation",
    max_points: int = 5000,
) -> RenderResult:
    x = read_samples(capture)
    if len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), max_points, replace=False)
        pts = x[idx]
    else:
        pts = x
    with _figure((6, 6)) as (fig, ax):
        ax.scatter(pts.real, pts.imag, s=2, alpha=0.4)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return _save(fig, workspace, name, "constellation")
