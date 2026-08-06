from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.mcp.streams import render_page, stream_stats


def _write(tmp_path: Path, z: np.ndarray, name: str = "s.cf32") -> Path:
    p = tmp_path / name
    z.astype(np.complex64).tofile(p)
    return p


def test_read_stream_complex_pages_real_imag(tmp_path: Path) -> None:
    z = (np.arange(10) + 1j * np.arange(10)).astype(np.complex64)
    p = _write(tmp_path, z)
    page = render_page(p, offset=0, count=6, item_type="c")
    real, imag = page["real"], page["imag"]
    assert isinstance(real, list) and isinstance(imag, list)
    assert page["item_type"] == "c"
    assert len(real) == len(imag) == 6
    assert real[3] == 3.0 and imag[3] == 3.0
    assert page["total_items"] == 10


def test_stream_stats_complex_tight_qpsk(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    centers = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    z = centers[rng.integers(0, 4, 8000)] + 0.05 * (
        rng.standard_normal(8000) + 1j * rng.standard_normal(8000)
    )
    p = _write(tmp_path, z)
    s = stream_stats(p, item_type="c", clusters=4, bins=41)
    clusters = s["clusters"]
    assert isinstance(clusters, list) and len(clusters) == 4
    assert all(900 < c["count"] < 3100 for c in clusters)  # balanced
    assert s["evm"] < 0.2  # type: ignore[operator]  # tight
    assert s["constant_modulus_ratio"] < 0.15  # type: ignore[operator]  # single ring


def test_stream_stats_complex_false_lock_is_high_evm(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal(8000) + 1j * rng.standard_normal(8000)  # structureless blob
    p = _write(tmp_path, z)
    s = stream_stats(p, item_type="c", clusters=4)
    assert s["evm"] > 0.5  # type: ignore[operator]  # no real 4-cluster structure


def test_stream_stats_complex_infers_from_suffix(tmp_path: Path) -> None:
    z = np.ones(16, np.complex64)
    p = _write(tmp_path, z)
    s = stream_stats(p, item_type=None, clusters=0, bins=41)  # infer from .cf32
    assert s["item_type"] == "c"
