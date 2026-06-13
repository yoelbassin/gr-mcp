from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile

from marconi import sigmf
from marconi.models import CaptureRef
from marconi.workspace import Workspace

if TYPE_CHECKING:
    from marconi.devices import SimulatedDevice


def _wav_to_unit_float(data: np.ndarray) -> np.ndarray:
    """Normalize WAV PCM samples to float32 in [-1, 1].

    scipy returns signed ints for >=16-bit PCM (centered on 0), unsigned uint8
    for 8-bit PCM (offset-binary, centered on 2**(n-1)), and floats for float
    WAVs (already in range by convention, so left untouched)."""
    if np.issubdtype(data.dtype, np.unsignedinteger):
        midpoint = (int(np.iinfo(data.dtype).max) + 1) / 2  # 128 for uint8
        return (data.astype(np.float32) - midpoint) / midpoint
    if np.issubdtype(data.dtype, np.signedinteger):
        # full negative range maps to exactly -1.0 (e.g. -32768 for int16)
        return data.astype(np.float32) / -np.iinfo(data.dtype).min
    return data.astype(np.float32)


def load_capture(
    path: Path | str,
    workspace: Workspace,
    sample_rate: float | None = None,
    center_freq: float | None = None,
) -> CaptureRef:
    """Ingest an external IQ file.

    SigMF files are referenced in place. Raw .cf32 (complex64 interleaved)
    and .wav (stereo = I/Q) files are converted to SigMF in the workspace.
    """
    path = Path(path)
    name = path.name

    if name.endswith((".sigmf-data", ".sigmf-meta")):
        return sigmf.read_meta(path)

    if name.endswith(".cf32"):
        if sample_rate is None:
            raise ValueError("sample_rate is required for raw .cf32 files")
        samples = np.fromfile(path, dtype=np.complex64)
        return sigmf.write_capture(
            samples,
            workspace.new_capture_path(path.stem),
            center_freq=center_freq if center_freq is not None else 0.0,
            sample_rate=sample_rate,
        )

    if name.endswith(".wav"):
        rate, data = wavfile.read(path)
        data = _wav_to_unit_float(data)
        # Only the first two channels are used as I and Q; extras are ignored.
        if data.ndim == 2 and data.shape[1] >= 2:
            samples = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
        else:
            samples = data.reshape(-1).astype(np.complex64)
        return sigmf.write_capture(
            samples,
            workspace.new_capture_path(path.stem),
            center_freq=center_freq if center_freq is not None else 0.0,
            sample_rate=float(sample_rate or rate),
        )

    raise ValueError(f"unsupported capture format: {name}")


def capture(
    device: "SimulatedDevice | str",
    center_freq: float,
    sample_rate: float,
    duration: float,
    workspace: Workspace,
    name: str | None = None,
) -> CaptureRef:
    """Capture IQ from a device. v1.0: simulated devices only — renders the
    device's scene as seen at center_freq/sample_rate."""
    from marconi.devices import SimulatedDevice, get_device
    from marconi.ops.simulate import render_scene

    dev = get_device(device) if isinstance(device, str) else device
    if not isinstance(dev, SimulatedDevice):
        raise TypeError(f"unsupported device type: {type(dev).__name__}")
    return render_scene(
        dev.scene,
        center_freq=center_freq,
        sample_rate=sample_rate,
        duration=duration,
        workspace=workspace,
        name=name or f"{dev.id}_capture",
    )
