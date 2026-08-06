import numpy as np
import pytest

from marconi.levels import fit_levels, kmeans_1d


def _levels(centers: list[float], per: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(c, sigma, per) for c in centers])


def test_clean_two_level_is_order2_and_well_separated() -> None:
    fit = fit_levels(_levels([-1.0, 1.0], 500, 0.08, 0))
    assert fit.order == 2
    assert fit.separation > 6.0
    assert np.allclose(np.sort(fit.centers), [-1.0, 1.0], atol=0.05)
    assert fit.min_gap == pytest.approx(2.0, abs=0.1)
    assert fit.within == pytest.approx(0.08, abs=0.02)


def test_clean_four_level_is_order4_and_well_separated() -> None:
    fit = fit_levels(_levels([-3.0, -1.0, 1.0, 3.0], 500, 0.08, 1))
    assert fit.order == 4
    assert fit.separation > 6.0


def test_clean_eight_level_is_order8() -> None:
    centers = [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
    fit = fit_levels(_levels(centers, 300, 0.06, 2))
    assert fit.order == 8
    assert fit.separation > 6.0


def test_gaussian_blob_is_low_separation() -> None:
    # forced-K quantization of any smooth unimodal density (measured for
    # gaussian/uniform/exponential) converges to separation ~2.6-3.6, never
    # below 2.0; real multi-level signals measure 25+, so 4.0 is a
    # margin-safe boundary on both sides
    fit = fit_levels(np.random.default_rng(3).normal(0.0, 1.0, 4000))
    assert fit.separation < 4.0


def test_two_level_does_not_over_split_to_four() -> None:
    # max-over-K must prefer K=2 for a genuine 2-level stream
    fit = fit_levels(_levels([-1.0, 1.0], 800, 0.05, 4))
    assert fit.order == 2


def test_kmeans_1d_recovers_known_centers() -> None:
    centers = kmeans_1d(_levels([-2.0, 2.0], 400, 0.05, 5), 2)
    assert centers.shape == (2,)
    assert np.allclose(np.sort(centers), [-2.0, 2.0], atol=0.15)


def test_kmeans_1d_ignores_outlier_tail() -> None:
    # a handful of extreme points (well under the [1,99] clip's 1% each
    # side) must not drag the fitted centers off the true levels
    x = np.concatenate([_levels([-2.0, 2.0], 500, 0.05, 6), [500.0, -500.0, 800.0]])
    centers = kmeans_1d(x, 2)
    assert centers.shape == (2,)
    assert np.allclose(np.sort(centers), [-2.0, 2.0], atol=0.2)


def test_fit_levels_within_not_inflated_by_outlier_tail() -> None:
    # the same outlier tail must not leak into `within`/`separation` either
    # - centers and spread have to be computed over the same clipped support
    x = np.concatenate([_levels([-2.0, 2.0], 500, 0.05, 7), [500.0, -500.0, 800.0]])
    fit = fit_levels(x)
    assert fit.order == 2
    assert fit.separation > 6.0
    assert fit.within < 1.0
