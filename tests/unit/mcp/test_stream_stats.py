from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from marconi.mcp.streams import stream_stats


def _f32(tmp_path: Path, x: np.ndarray, name: str = "s.f32") -> Path:
    p = tmp_path / name
    np.asarray(x, np.float32).tofile(p)
    return p


def test_four_level_clusters_recovered(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rails = rng.choice([-3.0, -1.0, 1.0, 3.0], size=8000)
    out = stream_stats(
        _f32(tmp_path, rails + rng.normal(0, 0.15, 8000)),
        item_type=None,
        clusters=4,
        bins=41,
    )
    centers = out["centers"]
    assert isinstance(centers, list) and len(centers) == 4
    assert centers == sorted(centers)
    assert centers[0] < -2.0 and centers[-1] > 2.0
    assert "levels" not in out, "centers must not ship twice"
    cluster_counts = out["cluster_counts"]
    assert isinstance(cluster_counts, list)
    assert sum(cluster_counts) == out["sampled_items"]


def test_histogram_length_and_totals(tmp_path: Path) -> None:
    out = stream_stats(
        _f32(tmp_path, np.random.default_rng(1).normal(0, 1, 5000)),
        item_type=None,
        clusters=0,
        bins=32,
    )
    hist = cast(dict[str, object], out["histogram"])
    counts = cast(list[int], hist["counts"])
    assert len(counts) == 32
    assert sum(counts) == out["sampled_items"]
    assert cast(float, hist["step"]) > 0
    assert "centers" not in out


def test_large_stream_is_strided(tmp_path: Path) -> None:
    out = stream_stats(
        _f32(tmp_path, np.random.default_rng(2).normal(0, 1, 200_000)),
        item_type=None,
        clusters=0,
        bins=16,
    )
    assert out["sampled"] is True
    assert out["sampled_items"] == 65536
    assert out["total_items"] == 200_000


def test_bits_stream_reports_ones_fraction(tmp_path: Path) -> None:
    p = tmp_path / "b.u8"
    np.array([1, 1, 1, 0, 1, 0, 1, 1], np.uint8).tofile(p)
    out = stream_stats(p, item_type=None, clusters=0, bins=8)
    assert out["item_type"] == "b"
    assert out["ones_fraction"] == pytest.approx(0.75)
    assert "histogram" not in out


def test_missing_path_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="resolves against"):
        stream_stats(tmp_path / "gone.f32", item_type=None, clusters=0, bins=8)


def test_tiny_stream_clamps_clusters_without_crash(tmp_path: Path) -> None:
    # 5 items with clusters=8: kmeans_1d clamps k to x.size and can return
    # fewer than `clusters` centers - the cluster_counts/keep masking must
    # follow the actual center count, not the requested one, or this raises.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    out = stream_stats(_f32(tmp_path, x), item_type=None, clusters=8, bins=8)
    centers = out["centers"]
    assert isinstance(centers, list) and 1 <= len(centers) <= x.size
    cluster_counts = out["cluster_counts"]
    assert isinstance(cluster_counts, list) and len(cluster_counts) == len(centers)
    assert sum(cluster_counts) == out["sampled_items"]
    assert "levels" not in out, "centers must not ship twice"


def test_over_requested_clusters_drops_phantoms(tmp_path: Path) -> None:
    # two well-separated clusters queried at clusters=3 must return two real
    # centers, not three with phantom zero-count centers in the empty gap.
    rng = np.random.default_rng(4)
    cluster1 = rng.normal(-3.0, 0.2, 3000)
    cluster2 = rng.normal(3.0, 0.2, 3000)
    data = np.concatenate([cluster1, cluster2])
    out = stream_stats(
        _f32(tmp_path, data),
        item_type=None,
        clusters=3,
        bins=41,
    )
    centers = out["centers"]
    assert isinstance(centers, list) and len(centers) == 2
    counts = out["cluster_counts"]
    assert isinstance(counts, list)
    assert all(c > 0 for c in counts)
    assert sum(counts) == out["sampled_items"]


def test_non_finite_items_are_counted_not_silently_dropped(tmp_path: Path) -> None:
    """Every statistic is computed over the survivors. Dropping the rest in
    silence let a stream the pipeline itself had poisoned (a zero-power span
    through the power agc) report a clean mean over the half that survived,
    with nothing in the response to say the other half existed."""
    x = np.concatenate(
        [np.full(500, np.nan, np.float32), np.full(1500, 2.0, np.float32)]
    )
    out = stream_stats(_f32(tmp_path, x), item_type=None, clusters=0, bins=8)
    assert out["total_items"] == 2000
    assert out["sampled_items"] == 1500
    assert out["non_finite_items"] == 500
    assert out["mean"] == 2.0


def test_a_clean_stream_carries_no_non_finite_key(tmp_path: Path) -> None:
    # absent, not zero: the omission rule means a reader never pays for a
    # measurement that had nothing to report
    x = np.full(1000, 1.5, np.float32)
    out = stream_stats(_f32(tmp_path, x), item_type=None, clusters=0, bins=8)
    assert "non_finite_items" not in out
