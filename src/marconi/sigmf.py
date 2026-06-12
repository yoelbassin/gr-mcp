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


def write_capture(
    samples: np.ndarray, path: Path, center_freq: float, sample_rate: float
) -> CaptureRef:
    """Write complex64 samples as a SigMF pair; `path` may omit the extension."""
    base = _base(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    samples = np.asarray(samples, dtype=np.complex64)
    samples.tofile(data_path)

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:version": SIGMF_VERSION,
        },
        "captures": [{"core:sample_start": 0, "core:frequency": center_freq}],
        "annotations": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return CaptureRef(
        path=data_path,
        center_freq=center_freq,
        sample_rate=sample_rate,
        num_samples=len(samples),
    )


def read_capture(path: Path) -> tuple[np.ndarray, CaptureRef]:
    """Read a SigMF pair; `path` may be the data file, meta file, or base."""
    base = _base(Path(path))
    data_path = base.with_name(base.name + ".sigmf-data")
    meta_path = base.with_name(base.name + ".sigmf-meta")

    meta = json.loads(meta_path.read_text())
    datatype = meta["global"]["core:datatype"]
    if datatype != "cf32_le":
        raise ValueError(f"unsupported SigMF datatype: {datatype}")

    samples = np.fromfile(data_path, dtype=np.complex64)
    ref = CaptureRef(
        path=data_path,
        center_freq=float(meta["captures"][0].get("core:frequency", 0.0)),
        sample_rate=float(meta["global"]["core:sample_rate"]),
        num_samples=len(samples),
    )
    return samples, ref
