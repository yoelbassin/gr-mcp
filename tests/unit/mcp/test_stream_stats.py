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


def test_exact_match_keeps_the_earned_fit(tmp_path: Path) -> None:
    # five genuine equal-mass modes queried at clusters=5: kept == requested,
    # so the fit the data earned must ship untouched even though 5 is not a
    # power of two. The unfixed code re-fits anyway (5 is not a power of
    # two), discarding the correct 5-center fit for a fabricated 4-center one
    # whose centers do not sit on the true modes.
    rng = np.random.default_rng(5)
    modes = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    data = rng.choice(modes, size=10000) + rng.normal(0, 0.02, 10000)
    out = stream_stats(
        _f32(tmp_path, data),
        item_type=None,
        clusters=5,
        bins=41,
    )
    centers = out["centers"]
    assert isinstance(centers, list) and len(centers) == 5
    for true_mode, center in zip(modes, sorted(centers)):
        assert abs(center - true_mode) < 0.1
    counts = out["cluster_counts"]
    assert isinstance(counts, list) and len(counts) == 5
    expected = cast(int, out["sampled_items"]) / 5
    for c in counts:
        assert abs(c - expected) < 0.2 * expected


def test_trimodal_exact_request_matches_modes(tmp_path: Path) -> None:
    # three genuine modes queried at clusters=3: kept == requested, 3 is not
    # a power of two either, and the fit must still be reported as-is.
    rng = np.random.default_rng(6)
    modes = np.array([-2.0, 0.0, 2.0])
    data = rng.choice(modes, size=6000) + rng.normal(0, 0.02, 6000)
    out = stream_stats(
        _f32(tmp_path, data),
        item_type=None,
        clusters=3,
        bins=41,
    )
    centers = out["centers"]
    assert isinstance(centers, list) and len(centers) == 3
    for true_mode, center in zip(modes, sorted(centers)):
        assert abs(center - true_mode) < 0.1


def test_over_requested_clusters_still_snap_to_power_of_two(tmp_path: Path) -> None:
    # three real modes queried at clusters=8: the seeded fit naturally lands
    # on 6 non-empty centers (2 per true mode - the scenario the docstring's
    # power-of-two contract exists for), kept(6) < requested(8), non-pow2,
    # so the trigger fires and the fit snaps down to the largest power of
    # two the data supports (4).
    rng = np.random.default_rng(0)
    modes = [-2.0, 0.0, 2.0]
    data = np.concatenate([m + rng.normal(0, 0.02, 3000) for m in modes])
    out = stream_stats(
        _f32(tmp_path, data),
        item_type=None,
        clusters=8,
        bins=41,
    )
    centers = out["centers"]
    assert isinstance(centers, list)
    assert len(centers) & (len(centers) - 1) == 0, "count must be a power of two"
    assert len(centers) == 4


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


def test_small_magnitude_stats_and_axis_are_not_rounded_to_zero(
    tmp_path: Path,
) -> None:
    # a fixed 6-DECIMAL round is an absolute 1e-6 quantum: an N(0, 1e-8)
    # stream shipped min=max=mean=std=-0.0 with histogram start=-0.0 step=0.0,
    # so all 41 reconstructed bin centers were identical and the stream was
    # indistinguishable from a constant zero.
    x = (np.random.default_rng(11).standard_normal(20_000) * 1e-8).astype(np.float32)
    out = stream_stats(_f32(tmp_path, x), item_type=None, clusters=0, bins=41)
    assert cast(float, out["std"]) == pytest.approx(float(x.std()), rel=1e-4)
    assert cast(float, out["min"]) == pytest.approx(float(x.min()), rel=1e-4)
    assert cast(float, out["max"]) == pytest.approx(float(x.max()), rel=1e-4)
    hist = cast(dict[str, object], out["histogram"])
    step = cast(float, hist["step"])
    assert step > 0
    # the shipped axis must reconstruct the documented 0.5..99.5 percentile
    # span - bin i's center is start + i*step
    lo, hi = (float(v) for v in np.percentile(x.astype(np.float64), [0.5, 99.5]))
    assert step == pytest.approx((hi - lo) / 41, rel=1e-4)
    start = cast(float, hist["start"])
    assert start == pytest.approx(lo + step / 2, rel=1e-4)
    assert start + 40 * step == pytest.approx(hi - step / 2, rel=1e-4)


def test_a_clean_stream_carries_no_non_finite_key(tmp_path: Path) -> None:
    # absent, not zero: the omission rule means a reader never pays for a
    # measurement that had nothing to report
    x = np.full(1000, 1.5, np.float32)
    out = stream_stats(_f32(tmp_path, x), item_type=None, clusters=0, bins=8)
    assert "non_finite_items" not in out


def test_erasure_mass_is_reported_and_absent_when_none(tmp_path: Path) -> None:
    """An LLR stream that is 40% erasures reads as a clean bimodal fit unless
    the zero mass is on the wire: the decisions here split exactly 50/50, but
    the zeros give kmeans a third mode and the reported clusters go lopsided.
    zero_items is what lets the reader tell a level ladder from an erasure
    fraction. Modelled on a real depuncture output."""
    rng = np.random.default_rng(3)
    decisions = rng.choice([-1.0, 1.0], size=6000)
    x = np.concatenate([decisions, np.zeros(4000)])
    rng.shuffle(x)
    out = stream_stats(_f32(tmp_path, x), item_type=None, clusters=2, bins=41)
    assert out["zero_items"] == 4000
    # the honesty this buys: the fit is NOT a 50/50 read of the decisions
    counts = cast("list[int]", out["cluster_counts"])
    assert min(counts) / max(counts) < 0.9

    clean = stream_stats(
        _f32(tmp_path, decisions, "clean.f32"), item_type=None, clusters=2, bins=41
    )
    assert "zero_items" not in clean
