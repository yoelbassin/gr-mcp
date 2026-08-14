"""Every constant that decides a verdict, pinned by a pair that straddles it.

These numbers are the product: they are what separates "decoded" from
"uncertain" for an agent that is told to trust only the first. Seven of them
carried a measurement in a comment and had no test at all, so the comment was
the only record of an experiment nothing re-ran. Each test below fails if its
constant moves in either direction — the lower assertion shows the value a
looser bar would have admitted, the upper one that the bar still passes real
evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.base import BlockCensus, Diagnostic
from marconi.engine.io.bitfile import write_llrs
from marconi.engine.quality import (
    _DOMINANCE_POSITIVE,
    _SOFT_MAX_SIGN_CORR,
    _SOFT_MULTILEVEL_SEPARATION,
    _SOFT_NEGATIVE,
    _SYNC_CHANCE_RATIO,
    _WORD_EXCESS_MIN_LOG_ODDS,
    _WORD_VALIDITY_POSITIVE,
    Assessment,
    QualityMetric,
    _sync_assessment,
    _word_excess_significant,
    dominance_evidence,
    soft_evidence,
    survival_evidence,
)
from marconi.engine.stages.registry import stage_registry
from marconi.levelfit import fit_levels


def _llrs(tmp_path: Path, values: np.ndarray) -> Path:
    p = tmp_path / "llrs.f32"
    write_llrs(p, values.astype(np.float32))
    return p


def _word_row(valid: int, total: int, chance: float) -> BlockCensus:
    return BlockCensus(
        block="block_code[0]",
        kind="block_code",
        words_valid=valid,
        words_total=total,
        chance_word_rate=chance,
    )


def _dominance_rows(dominant: int, total: int, chance: float) -> list[Diagnostic]:
    return [
        Diagnostic(block="b1", key="dominant_symbols", count=dominant),
        Diagnostic(block="b1", key="symbols_total", count=total),
        Diagnostic(block="b1", key="dominance_chance", value=chance),
    ]


def test_sync_chance_ratio_is_load_bearing() -> None:
    assert _SYNC_CHANCE_RATIO == 3.0
    # clears the 5-sigma bar (100 + 5*10 = 150) but not 3x chance
    assert 200 > 150 and 200 >= 2.0 * 100.0  # a 2x bar would have admitted it
    assert _sync_assessment(200, 100.0, 100_000) is None
    assert _sync_assessment(301, 100.0, 100_000) is Assessment.POSITIVE


def test_word_validity_positive_is_load_bearing() -> None:
    assert _WORD_VALIDITY_POSITIVE == 0.5
    registry = stage_registry()
    assert survival_evidence([_word_row(49, 100, 0.01)], registry) == []
    ev = survival_evidence([_word_row(51, 100, 0.01)], registry)
    assert [e.assessment for e in ev] == [Assessment.POSITIVE]
    assert ev[0].metric is QualityMetric.WORD_VALIDITY


def test_word_excess_log_odds_is_load_bearing() -> None:
    assert _WORD_EXCESS_MIN_LOG_ODDS == 15.0
    # every word valid, but too few of them to be significant at chance 0.125
    assert not _word_excess_significant(4, 4, 0.125)
    assert _word_excess_significant(8, 8, 0.125)
    registry = stage_registry()
    assert survival_evidence([_word_row(4, 4, 0.125)], registry) == []
    assert survival_evidence([_word_row(8, 8, 0.125)], registry) != []


def test_dominance_positive_is_load_bearing() -> None:
    assert _DOMINANCE_POSITIVE == 0.5
    assert dominance_evidence(_dominance_rows(49, 100, 0.05)) == []
    ev = dominance_evidence(_dominance_rows(51, 100, 0.05))
    assert [e.assessment for e in ev] == [Assessment.POSITIVE]
    assert ev[0].metric is QualityMetric.PEAK_DOMINANCE


@pytest.mark.parametrize(
    "sigma, above",
    [(0.30, True), (0.90, False)],
)
def test_soft_multilevel_separation_is_load_bearing(
    tmp_path: Path, sigma: float, above: bool
) -> None:
    assert _SOFT_MULTILEVEL_SEPARATION == 4.0
    rng = np.random.default_rng(4)
    levels = rng.choice([-3.0, -1.0, 1.0, 3.0], size=8000)
    x = levels + rng.normal(0.0, sigma, levels.size)
    # the measurement the constant is compared against, taken here so a moved
    # constant cannot be "fixed" by quietly re-tuning the fixture
    fit = fit_levels(x)
    assert (fit.separation >= _SOFT_MULTILEVEL_SEPARATION) is above
    ev = soft_evidence(_llrs(tmp_path, x))
    if above:
        assert [e.assessment for e in ev] == [Assessment.POSITIVE]
    else:
        # explicitly NO evidence: the old combined form (`bool(ev and
        # positive) is False`) was also satisfied by a NEGATIVE, so it could
        # not catch a below-bar M-ary eye being read as no_signal
        assert ev == []


def test_soft_negative_bar_is_load_bearing(tmp_path: Path) -> None:
    assert _SOFT_NEGATIVE == 1.45
    rng = np.random.default_rng(5)
    # gaussian noise sits at mean|x|/std|x| ~ 1.32: below the bar, negative
    noise = rng.normal(0.0, 1.0, 8000)
    mag = np.abs(noise)
    assert float(mag.mean() / mag.std()) < _SOFT_NEGATIVE
    ev = soft_evidence(_llrs(tmp_path, noise))
    assert [e.assessment for e in ev] == [Assessment.NEGATIVE]

    # a mild bimodal lift clears the negative bar without reaching positive:
    # the band between the two soft bars is deliberately "no evidence"
    mixed = rng.choice([-1.0, 1.0], size=8000) * 1.2 + rng.normal(0.0, 1.0, 8000)
    mag = np.abs(mixed)
    assert _SOFT_NEGATIVE < float(mag.mean() / mag.std()) < 2.0
    assert soft_evidence(_llrs(tmp_path, mixed)) == []


def test_soft_sign_correlation_bar_is_load_bearing(tmp_path: Path) -> None:
    assert _SOFT_MAX_SIGN_CORR == 0.15
    rng = np.random.default_rng(6)

    def stream(repeat_fraction: float) -> np.ndarray:
        signs = rng.choice([-2.0, 2.0], size=8000)
        held = rng.random(8000) < repeat_fraction
        for i in np.flatnonzero(held):
            if i:
                signs[i] = signs[i - 1]
        return signs + rng.normal(0.0, 0.3, 8000)

    def sign_corr(x: np.ndarray) -> float:
        s = np.sign(x)
        return float(np.mean(s[1:] * s[:-1]))

    whitened = stream(0.0)
    assert abs(sign_corr(whitened)) <= _SOFT_MAX_SIGN_CORR
    ev = soft_evidence(_llrs(tmp_path, whitened))
    assert [e.assessment for e in ev] == [Assessment.POSITIVE]

    # a wrong-rate decode repeats its decisions; confident-looking but unwhitened
    correlated = stream(0.5)
    assert abs(sign_corr(correlated)) > _SOFT_MAX_SIGN_CORR
    assert soft_evidence(_llrs(tmp_path, correlated)) == []


def test_dominance_negative_is_load_bearing() -> None:
    # the straddle this file's own header promises: _DOMINANCE_NEGATIVE was
    # claimed "straddled by test_verdict_constants" while appearing in no
    # test anywhere in the tree
    from marconi.engine.quality import _DOMINANCE_NEGATIVE

    assert _DOMINANCE_NEGATIVE == 0.05
    below = dominance_evidence(_dominance_rows(4, 100, 0.01))
    above = dominance_evidence(_dominance_rows(6, 100, 0.01))
    assert [e.assessment for e in below] == [Assessment.NEGATIVE]
    assert above == []  # past the bar: midband, not negative
