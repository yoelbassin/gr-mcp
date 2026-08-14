"""The soft negative arm's null must belong to the stream's own shape.

Two verdict inversions of the same class (a significance bar computed from
data containing the thing it judges): the binary noise bar applied to a
measurably multi-level stream read a 94%-correct 4-PAM decode as no_signal,
and the activity gate thresholded at a fraction of the same stream's p90
handed the verdict to an appended interferer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine import quality
from marconi.engine.io.bitfile import write_llrs
from marconi.engine.quality import (
    _SOFT_NEGATIVE,
    Assessment,
    Verdict,
    soft_evidence,
)
from marconi.engine.stages.registry import stage_registry
from marconi.levelfit import fit_levels

_PAM4_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])
_PAM4_SIGN_BIT = np.array([0, 0, 1, 1])
_PAM4_INNER_BIT = np.array([0, 1, 1, 0])


def _pam4_llrs(n_syms: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Textbook Gray 4-PAM max-log LLRs: per bit, the nearest-symbol distance
    of the bit-1 coset minus the bit-0 coset, so positive = bit 0 (the tree's
    convention: soft bit 1 is negative). Two LLR lanes per symbol."""
    idx = rng.integers(0, 4, size=n_syms)
    y = _PAM4_LEVELS[idx] + rng.normal(0.0, sigma, n_syms)
    d = (y[:, None] - _PAM4_LEVELS[None, :]) ** 2 / (2.0 * sigma**2)
    out = np.empty(2 * n_syms)
    for lane, bits in ((0, _PAM4_SIGN_BIT), (1, _PAM4_INNER_BIT)):
        m1 = np.min(np.where(bits[None, :] == 1, d, np.inf), axis=1)
        m0 = np.min(np.where(bits[None, :] == 0, d, np.inf), axis=1)
        out[lane::2] = m1 - m0
    return out


def _llr_file(tmp_path: Path, name: str, values: np.ndarray) -> Path:
    p = tmp_path / f"{name}.f32"
    write_llrs(p, values.astype(np.float32))
    return p


def _soft_only_report(path: Path) -> quality.QualityReport:
    return quality.assess_quality(
        registry=stage_registry(),
        census=[],
        diagnostics=[],
        marks=[],
        soft_stream=path,
    )


def _interfered(rng: np.random.Generator, n_garbage: int) -> np.ndarray:
    rails = rng.choice([-2.0, 2.0], size=58_000) + rng.normal(0.0, 0.3, 58_000)
    return np.concatenate([rails, rng.normal(0.0, 30.0, n_garbage)])


def test_moderate_noise_mary_llrs_are_not_negative(tmp_path: Path) -> None:
    # sigma 0.7 is BER ~0.057 — a decode worth keeping. The binary arm would
    # fire (ratio under the bar) on an order-4 shape; both pinned so the
    # fixture cannot drift out of the regime the rule judges.
    rng = np.random.default_rng(0)
    x = _pam4_llrs(8000, 0.7, rng)
    mag = np.abs(x)
    assert float(mag.mean() / mag.std()) <= _SOFT_NEGATIVE
    assert fit_levels(x).order > 2
    path = _llr_file(tmp_path, "pam4_mid", x)
    ev = soft_evidence(path)
    assert not any(e.assessment is Assessment.NEGATIVE for e in ev)
    report = _soft_only_report(path)
    assert report.verdict is not Verdict.NO_SIGNAL


def test_clean_mary_llrs_are_still_positive(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    ev = soft_evidence(_llr_file(tmp_path, "pam4_clean", _pam4_llrs(8000, 0.3, rng)))
    assert [e.assessment for e in ev] == [Assessment.POSITIVE]


def test_dead_mary_demod_noise_is_still_negative(tmp_path: Path) -> None:
    # pure noise through the same max-log formula — the demod's transform
    # imprints kinks on anything, so this is the honest-negative case a shape
    # exemption is most likely to kill
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 3.0, 8000)
    d = (y[:, None] - _PAM4_LEVELS[None, :]) ** 2
    x = np.empty(16000)
    for lane, bits in ((0, _PAM4_SIGN_BIT), (1, _PAM4_INNER_BIT)):
        m1 = np.min(np.where(bits[None, :] == 1, d, np.inf), axis=1)
        m0 = np.min(np.where(bits[None, :] == 0, d, np.inf), axis=1)
        x[lane::2] = m1 - m0
    ev = soft_evidence(_llr_file(tmp_path, "pam4_dead", x))
    assert [e.assessment for e in ev] == [Assessment.NEGATIVE]


def test_margin_ranks_moderate_mary_above_noise(tmp_path: Path) -> None:
    # the ranking half of the inversion: the 94%-correct stream's margin sat
    # BELOW pure noise (1.15 vs 1.32), so a parameter search walked away from
    # the signal. Fixed, the margin carries the fit separation (~3.0) once
    # the dip is significant; gated as a gap, not an exact value.
    rng = np.random.default_rng(0)

    def margin_of(values: np.ndarray, name: str) -> float:
        _, _, margin = quality._soft_reading(_llr_file(tmp_path, name, values))
        assert margin is not None
        return margin

    m_signal = margin_of(_pam4_llrs(8000, 0.7, rng), "pam4_margin")
    m_noise = margin_of(rng.normal(0.0, 1.0, 16000), "noise_margin")
    assert m_signal > m_noise + 1.0, (m_signal, m_noise)


def test_appended_interferer_does_not_invert_the_verdict(tmp_path: Path) -> None:
    # 58k decodable rail LLRs (control below) + 7k items of N(0, 30): 10.8%
    # duty puts p90 of windowed power inside the interferer, and the mask
    # keeps almost nothing but interference — pinned so the fixture keeps
    # exercising the inverted-selection path.
    rng = np.random.default_rng(0)
    x = _interfered(rng, 7_000)
    x32 = x.astype(np.float32)
    assert float(quality._active_mask(x32).mean()) < 0.2
    path = _llr_file(tmp_path, "interfered", x)
    ev = soft_evidence(path)
    assert not any(e.assessment is Assessment.NEGATIVE for e in ev)
    report = _soft_only_report(path)
    assert report.verdict is not Verdict.NO_SIGNAL
    assert "interferer" in report.rationale


def test_low_duty_interferer_does_not_invert_the_verdict(tmp_path: Path) -> None:
    # below ~10% duty the interferer no longer owns p90: the mask keeps
    # everything and the MIX reads ratio ~0.5. The rails are still a real
    # bimodal shape with a significant dip, so the negative is withheld by
    # the structure rule rather than the complement rule.
    rng = np.random.default_rng(0)
    x = _interfered(rng, 3_000)
    assert float(quality._active_mask(x.astype(np.float32)).mean()) > 0.9
    path = _llr_file(tmp_path, "low_duty", x)
    ev = soft_evidence(path)
    assert not any(e.assessment is Assessment.NEGATIVE for e in ev)
    assert _soft_only_report(path).verdict is not Verdict.NO_SIGNAL


def test_interferer_control_still_decodes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = _interfered(rng, 0)
    path = _llr_file(tmp_path, "control", x)
    ev = soft_evidence(path)
    assert [e.assessment for e in ev] == [Assessment.POSITIVE]
    assert _soft_only_report(path).verdict is Verdict.DECODED


def test_bursty_noise_is_still_negative(tmp_path: Path) -> None:
    # noise bursts over a faint idle floor: the mask keeps the bursts, the
    # discarded idle carries no confident decisions, and the negative stands
    rng = np.random.default_rng(7)
    blocks = []
    for _ in range(60):
        blocks.append(rng.normal(0.0, 1.0, 500))
        blocks.append(rng.normal(0.0, 0.02, 500))
    ev = soft_evidence(_llr_file(tmp_path, "bursty_noise", np.concatenate(blocks)))
    assert [e.assessment for e in ev] == [Assessment.NEGATIVE]


def test_dip_significance_bar_is_load_bearing(tmp_path: Path) -> None:
    # the straddle: unimodal noise fits order>2 with separation up to 3.13 —
    # overlapping a decodable 4-PAM LLR stream's 2.95-3.04 — but never dips
    # between its two heaviest levels beyond sampling luck (measured z max
    # +2.58 over 8 families x 300 seeds), while the genuine stream's dip
    # clears +5. The measurement is taken here so a moved constant cannot be
    # "fixed" by re-tuning the fixture.
    assert quality._SOFT_DIP_MIN_Z == 4.0
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 1.0, 8000).astype(np.float32)
    lane = quality._lane_stats(noise, quality._active_mask(noise))
    assert lane.fit.order > 2
    assert lane.dip_z < quality._SOFT_DIP_MIN_Z
    ev = soft_evidence(_llr_file(tmp_path, "noise_straddle", noise))
    assert [e.assessment for e in ev] == [Assessment.NEGATIVE]

    signal = _pam4_llrs(8000, 0.7, np.random.default_rng(0)).astype(np.float32)
    lane = quality._lane_stats(signal, quality._active_mask(signal))
    assert lane.fit.order > 2
    assert lane.dip_z >= quality._SOFT_DIP_MIN_Z
    ev = soft_evidence(_llr_file(tmp_path, "signal_straddle", signal))
    assert not any(e.assessment is Assessment.NEGATIVE for e in ev)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_interfered_mary_stream_is_not_reinverted(tmp_path: Path, seed: int) -> None:
    # the composition hole: the structure rule's regime (moderate-BER 4-PAM,
    # uncertain alone) plus the interferer scenario. The active lane is the
    # interferer (not structured) and the complement sits below the positive
    # bars (ratio ~1.15) but IS structured (dip z ~ +14) — each rule deferred
    # to the other and the false negative slipped between; margin re-inverted
    # under pure noise. A structured complement must rescue, and its
    # separation is the run's real decision margin.
    rng = np.random.default_rng(seed)
    x = np.concatenate([_pam4_llrs(29_000, 0.7, rng), rng.normal(0.0, 30.0, 7_000)])
    assert float(quality._active_mask(x.astype(np.float32)).mean()) < 0.2
    path = _llr_file(tmp_path, f"mary_interfered_{seed}", x)
    ev = soft_evidence(path)
    assert not any(e.assessment is Assessment.NEGATIVE for e in ev)
    report = _soft_only_report(path)
    assert report.verdict is not Verdict.NO_SIGNAL
    assert "interferer" in report.rationale
    # complement separation ~3.0 joins the margin; pure noise sits at ~1.32
    assert report.margin is not None and report.margin > 2.0


def test_doubled_mary_llrs_are_rescued_at_production_length(tmp_path: Path) -> None:
    # a symbol-doubled (oversampled) genuine 4-PAM stream pays the k_ess
    # deflation like repeated noise does; at the full 65536-item sample the
    # dip is still significant (measured z >= +10.1 over 5 seeds)
    rng = np.random.default_rng(0)
    base = _pam4_llrs(16_384, 0.7, rng)
    ev = soft_evidence(_llr_file(tmp_path, "doubled_long", np.repeat(base, 2)))
    assert not any(e.assessment is Assessment.NEGATIVE for e in ev)


def test_doubled_mary_short_stream_is_a_documented_blind_spot(
    tmp_path: Path,
) -> None:
    # measured edge: at 16k items the doubled stream's deflated dip z spans
    # +2.9..+5.7 across seeds — this seed sits at +2.9, under the bar, and
    # reads negative. k_ess cannot tell repeated items from repetitive
    # content, and undoing the deflation would let x4-repeated noise
    # (z +4.6 unscaled) through. Pinned so a future k_ess change gets a
    # signal, not because the behavior is desirable.
    rng = np.random.default_rng(3)
    base = _pam4_llrs(4_000, 0.7, rng)
    ev = soft_evidence(_llr_file(tmp_path, "doubled_short", np.repeat(base, 2)))
    assert [e.assessment for e in ev] == [Assessment.NEGATIVE]


@pytest.mark.parametrize("sigma", [0.5, 0.6, 0.7])
def test_below_positive_bar_mary_band_is_uncertain(
    tmp_path: Path, sigma: float
) -> None:
    # the genuine-but-below-positive band (BER 1.7% to 5.7%) must land as no
    # evidence — never negative, and not positive either (separation is
    # honestly under the positive bar there)
    rng = np.random.default_rng(1)
    ev = soft_evidence(
        _llr_file(tmp_path, f"band_{sigma}", _pam4_llrs(8000, sigma, rng))
    )
    assert ev == []
