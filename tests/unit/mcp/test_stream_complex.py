from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

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


@pytest.mark.parametrize("seed", range(30))
def test_stream_stats_complex_tight_qpsk(tmp_path: Path, seed: int) -> None:
    # canonical QPSK lobes sit at 45/135/225/315 degrees - exactly the
    # unlucky rotation for a naive axis-aligned k-means seeding grid, so this
    # sweeps seeds rather than pinning one that happens to converge cleanly.
    rng = np.random.default_rng(seed)
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


def test_stream_stats_complex_all_zero_ratio_is_none_not_a_perfect_ring(
    tmp_path: Path,
) -> None:
    z = np.zeros(16, np.complex64)
    p = _write(tmp_path, z)
    s = stream_stats(p, item_type="c", clusters=0)
    assert s["mean_magnitude"] == 0.0
    assert s["constant_modulus_ratio"] is None


def test_the_constellation_measures_the_docstring_promises_are_real(
    tmp_path: Path,
) -> None:
    """std_magnitude, magnitude_histogram and phase_histogram are named in
    stream_stats' tool docstring — the agent's only instructions — and
    std_magnitude also ships in every trace row. Nothing executed any of them,
    so a renderer wired to the wrong measure would have shipped unnoticed.

    Two rings at radius 1 and 2, four phases each: every number below is
    read off that geometry rather than off the implementation."""
    rng = np.random.default_rng(0)
    phases = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2])
    radii = np.array([1.0, 2.0])
    z = (
        radii[rng.integers(0, 2, 8000)] * np.exp(1j * phases[rng.integers(0, 4, 8000)])
    ).astype(np.complex64)
    s = stream_stats(_write(tmp_path, z), item_type="c", clusters=8, bins=41)

    # |z| is 1 or 2 in equal measure: mean 1.5, population std 0.5
    assert s["mean_magnitude"] == pytest.approx(1.5, abs=0.05)
    assert s["std_magnitude"] == pytest.approx(0.5, abs=0.05)
    assert s["constant_modulus_ratio"] == pytest.approx(0.5 / 1.5, abs=0.05)

    mag_hist, phase_hist = s["magnitude_histogram"], s["phase_histogram"]
    assert isinstance(mag_hist, dict) and isinstance(phase_hist, dict)
    # the axis is derivable, never shipped: bin i sits at start + i*step
    assert set(mag_hist) == set(phase_hist) == {"start", "step", "counts"}
    assert sum(mag_hist["counts"]) == sum(phase_hist["counts"]) == s["sampled_items"]
    # two magnitude modes, four phase modes - the shape the geometry dictates
    assert sum(1 for n in mag_hist["counts"] if n > 100) == 2
    assert sum(1 for n in phase_hist["counts"] if n > 100) == 4

    # phase_deg is degrees, not radians, and sorted by angle
    lobes = cast(list[dict[str, float]], s["clusters"])
    degrees = [c["phase_deg"] for c in lobes]
    assert degrees == sorted(degrees)
    assert all(-180.0 <= d <= 180.0 for d in degrees)
    for want in (-90.0, 0.0, 90.0, 180.0):
        assert any(abs(d - want) < 5.0 for d in degrees), f"no lobe near {want} deg"


def test_magnitude_axis_holds_at_the_capture_lsb_scale(tmp_path: Path) -> None:
    # a weak capture quantized at the ci16 LSB (1/32768) lives four LSBs from
    # the origin: the fixed 6-decimal round shipped step=4e-06 against a true
    # 4.21057e-06 - 5% axis error, two whole bins of drift by bin 40
    rng = np.random.default_rng(3)
    lsb = 1.0 / 32768.0
    z = (rng.integers(-4, 5, 20_000) + 1j * rng.integers(-4, 5, 20_000)) * lsb
    s = stream_stats(_write(tmp_path, z), item_type="c", clusters=0, bins=41)
    hist = cast(dict[str, float], s["magnitude_histogram"])
    mag = np.abs(np.asarray(z, np.complex64).astype(np.complex128))
    lo, hi = (float(v) for v in np.percentile(mag, [0.5, 99.5]))
    assert hist["step"] == pytest.approx((hi - lo) / 41, rel=1e-4)
    assert hist["start"] == pytest.approx(lo + hist["step"] / 2, rel=1e-4)
