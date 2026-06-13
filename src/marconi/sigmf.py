import json
from pathlib import Path

import numpy as np

from marconi.models import CaptureRef

SIGMF_VERSION = "1.0.0"


def _base(path: Path) -> Path:
    name = path.name
    for suffix in (".sigmf-data", ".sigmf-meta"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path


def _meta_dict(center_freq: float, sample_rate: float) -> dict:
    return {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:version": SIGMF_VERSION,
        },
        "captures": [{"core:sample_start": 0, "core:frequency": center_freq}],
        "annotations": [],
    }


def write_capture(
    samples: np.ndarray, path: Path | str, center_freq: float, sample_rate: float
) -> CaptureRef:
    """Write complex64 samples as a SigMF pair; `path` may omit the extension."""
    path = Path(path)
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    samples = np.asarray(samples, dtype=np.complex64)
    samples.tofile(data_path)

    meta = _meta_dict(center_freq, sample_rate)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CaptureRef(
        path=data_path,
        center_freq=center_freq,
        sample_rate=sample_rate,
        num_samples=len(samples),
    )


def read_meta(path: Path | str) -> CaptureRef:
    """Read only the SigMF metadata; num_samples comes from the data file size."""
    base = _base(Path(path))
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    datatype = meta["global"]["core:datatype"]
    if datatype != "cf32_le":
        raise ValueError(f"unsupported SigMF datatype: {datatype}")
    if not meta.get("captures"):
        raise ValueError("SigMF meta has no captures")

    return CaptureRef(
        path=data_path,
        center_freq=float(meta["captures"][0].get("core:frequency", 0.0)),
        sample_rate=float(meta["global"]["core:sample_rate"]),
        num_samples=data_path.stat().st_size // 8,  # complex64 = 8 bytes
    )


def read_samples(capture: CaptureRef) -> np.ndarray:
    """The single reader for capture sample data."""
    return np.fromfile(capture.path, dtype=np.complex64)


def write_meta(
    data_path: Path | str, center_freq: float, sample_rate: float
) -> CaptureRef:
    """Create the .sigmf-meta sidecar for an existing raw cf32 data file."""
    data_path = Path(data_path)
    base = _base(data_path)
    meta_path = base.with_name(base.name + ".sigmf-meta")

    meta = _meta_dict(center_freq, sample_rate)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return CaptureRef(
        path=data_path,
        center_freq=center_freq,
        sample_rate=sample_rate,
        num_samples=data_path.stat().st_size // 8,
    )


def read_capture(path: Path | str) -> tuple[np.ndarray, CaptureRef]:
    """Read a SigMF pair; `path` may be the data file, meta file, or base."""
    ref = read_meta(path)
    samples = np.fromfile(ref.path, dtype=np.complex64)
    return samples, ref
