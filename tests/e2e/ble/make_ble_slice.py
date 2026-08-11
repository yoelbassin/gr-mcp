"""Reproduce the asset-gated BLE advertising slice from its public Zenodo source.

Downloads the SDR4IoT BLE/Zigbee dataset (Zenodo 10.5281/zenodo.4639390, CC-BY),
extracts seven CRC-valid channel-37 advertising bursts (their exact sample windows
in the scenario-5 recordings, each a .sigmf tar of complex64 IQ), and concatenates
them (500-sample guards) into the slice the off-air gate
(tests/e2e/ble/test_ble_offair.py) decodes. 5 Msps.
Oracle self-validates via BLE CRC-24; ground truth cross-checked independently."""

from __future__ import annotations

import io
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://zenodo.org/records/4639390/files/dataset.zip?download=1"
OUT = Path(__file__).resolve().parents[3] / "artifacts" / "assets" / "BLE" / "ble.cf32"
PAD = 500
EXPECTED_SAMPLES = 17320
# (source .sigmf basename, packet sample lo, hi) — the seven CRC-valid ch37 bursts.
PACKETS = [
    ("S5_s31_2020-11-10_10-40_server11.sigmf", 27195, 28715),
    ("S5_s31_2020-11-10_10-40_server12.sigmf", 104154, 105674),
    ("S5_s31_2020-11-10_10-40_server15.sigmf", 237631, 238991),
    ("S5_s31_2020-11-10_11-21_server12.sigmf", 120759, 122119),
    ("S5_s32_2020-11-12_11-15_server11.sigmf", 12434, 13954),
    ("S5_s36_2020-11-09_10-18_server12.sigmf", 36692, 38212),
    ("S5_s36_2020-11-09_10-18_server12.sigmf", 83634, 85154),
]


def _load_sigmf(zf: zipfile.ZipFile, basename: str) -> np.ndarray:
    member = next((n for n in zf.namelist() if n.endswith(basename)), None)
    if member is None:
        raise SystemExit(f"{basename} not found in the dataset zip")
    with tarfile.open(fileobj=io.BytesIO(zf.read(member))) as tf:
        data = next(m for m in tf.getmembers() if m.name.endswith(".sigmf-data"))
        buf = tf.extractfile(data)
        assert buf is not None
        return np.frombuffer(buf.read(), dtype=np.complex64)


def _fetch() -> bytes:
    if len(sys.argv) > 1:  # a locally-downloaded dataset.zip
        return Path(sys.argv[1]).read_bytes()
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        payload: bytes = urllib.request.urlopen(req, timeout=300).read()  # noqa: S310
        return payload
    except urllib.error.URLError as e:
        code = getattr(e, "code", "?")
        raise SystemExit(
            f"Zenodo fetch failed (HTTP {code}; it rate-limits scripted downloads). "
            f"Fetch {URL} in a browser, then: python make_ble_slice.py <dataset.zip>"
        ) from e


def main() -> None:
    zf = zipfile.ZipFile(io.BytesIO(_fetch()))
    cache: dict[str, np.ndarray] = {}
    segments: list[np.ndarray] = []
    for basename, lo, hi in PACKETS:
        if basename not in cache:
            cache[basename] = _load_sigmf(zf, basename)
        iq = cache[basename]
        segments.append(iq[max(0, lo - PAD) : hi + PAD])
    sig = np.concatenate(segments).astype(np.complex64)
    assert sig.size == EXPECTED_SAMPLES, f"unexpected slice size {sig.size}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sig.tofile(OUT)
    print(f"wrote {OUT} ({sig.size} samples, {sig.size / 5_000_000:.4f}s)")


if __name__ == "__main__":
    main()
