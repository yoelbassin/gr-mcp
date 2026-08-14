from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine import quality
from marconi.engine.backends.base import BlockCensus
from marconi.engine.io.bitfile import write_llrs
from marconi.engine.quality import (
    _SOFT_POSITIVE,
    _SOFT_SAMPLE_ITEMS,
    _sample_soft,
    soft_evidence,
)
from marconi.engine.stages.registry import stage_registry


def _llr_file(tmp_path: Path, values: np.ndarray) -> Path:
    p = tmp_path / "llrs.f32"
    write_llrs(p, values.astype(np.float32))
    return p


def test_sample_soft_reads_contiguous_active_window_over_budget(tmp_path: Path) -> None:
    # A quiet lead longer than the sample budget, then a strong active block.
    # _sample_soft must return ONE contiguous highest-power window on the active
    # block -- not a head-only slice (would judge the quiet lead) and not
    # strided chunks stitched together.
    quiet = np.full(_SOFT_SAMPLE_ITEMS * 3, 0.02, np.float32)
    active = np.tile([2.0, -2.0], _SOFT_SAMPLE_ITEMS // 2).astype(np.float32)
    stream = np.concatenate([quiet, active, quiet])
    got = _sample_soft(_llr_file(tmp_path, stream))
    assert got.size == _SOFT_SAMPLE_ITEMS
    assert float(np.mean(got.astype(np.float64) ** 2)) > 1.0  # on the active block
    assert any(  # a genuine consecutive slice, not fragmented chunks
        np.array_equal(got, stream[k : k + _SOFT_SAMPLE_ITEMS])
        for k in range(0, stream.size - _SOFT_SAMPLE_ITEMS + 1, _SOFT_SAMPLE_ITEMS)
    )


def test_bimodal_llrs_are_positive(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    signs = rng.choice([-2.0, 2.0], size=4000)
    ev = soft_evidence(_llr_file(tmp_path, signs + rng.normal(0, 0.3, 4000)))
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "soft_confidence"


def test_mid_snr_bimodal_is_positive(tmp_path: Path) -> None:
    # the decodable envelope: +-1 rails at sigma 0.45 (~2.2 ratio) must be
    # positive - the old 6.0 bar starved everything below ~14 dB
    rng = np.random.default_rng(2)
    signs = rng.choice([-1.0, 1.0], size=4000)
    ev = soft_evidence(_llr_file(tmp_path, signs + rng.normal(0, 0.45, 4000)))
    assert [e.assessment for e in ev] == ["positive"]


def test_correlated_signs_are_not_positive(tmp_path: Path) -> None:
    # an oversampled wrong-rate decode repeats decisions (sign correlation
    # ~0.5); confident-looking but unwhitened streams must not read positive
    rng = np.random.default_rng(3)
    signs = np.repeat(rng.choice([-2.0, 2.0], size=2000), 2)
    ev = soft_evidence(_llr_file(tmp_path, signs + rng.normal(0, 0.3, 4000)))
    assert ev == []


def test_alternating_stream_is_not_positive(tmp_path: Path) -> None:
    x = np.tile([2.0, -2.0], 2000)
    assert soft_evidence(_llr_file(tmp_path, x)) == []


def test_structured_unscrambled_data_is_suppressed_by_design(
    tmp_path: Path,
) -> None:
    # measured edge: run-length-8 NRZ (unscrambled zero-heavy protocols) has
    # sign correlation +0.87 - indistinguishable from an oversampled decode
    # by this statistic, so the positive is (conservatively) withheld
    rng = np.random.default_rng(5)
    signs = np.repeat(rng.choice([-2.0, 2.0], size=500), 8)
    ev = soft_evidence(_llr_file(tmp_path, signs + rng.normal(0, 0.3, 4000)))
    assert ev == []


def test_mild_oversampling_is_a_documented_blind_spot(tmp_path: Path) -> None:
    # measured edge: a 1.1x wrong-rate decode repeats ~every 10th decision
    # (corr ~0.09, under the 0.15 bar) and reads positive - the guard only
    # protects from ~1.2x up; this pin exists so a future guard change gets a
    # signal, not because the behavior is desirable
    rng = np.random.default_rng(6)
    base = rng.choice([-2.0, 2.0], size=4000)
    x = np.repeat(base, np.where(np.arange(4000) % 10 == 0, 2, 1))
    ev = soft_evidence(_llr_file(tmp_path, x + rng.normal(0, 0.3, x.size)))
    assert [e.assessment for e in ev] == ["positive"]


def test_noise_lead_does_not_read_negative(tmp_path: Path) -> None:
    # a capture whose burst starts late: head-only sampling judged it on
    # noise it never needed to decode; strided sampling sees the signal too
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 1.0, 100_000)
    signal = rng.choice([-2.0, 2.0], size=100_000) + rng.normal(0, 0.3, 100_000)
    ev = soft_evidence(_llr_file(tmp_path, np.concatenate([noise, signal])))
    # ev != [] FIRST: a regression that stops finding the burst returns []
    # and "not negative" is vacuously true of nothing (mutated soft_evidence
    # to always return [] - this test still passed)
    assert [e.assessment for e in ev] == ["positive"]


def test_gaussian_noise_llrs_are_negative(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    ev = soft_evidence(_llr_file(tmp_path, rng.normal(0, 1.0, 4000)))
    assert [e.assessment for e in ev] == ["negative"]


def test_all_zero_stream_is_erasures_not_a_measurement(tmp_path: Path) -> None:
    # exact zeros are erasures (depuncture null_source, hard gates), so a
    # stream of nothing else was never measured at all: untestable, not
    # negative. A genuinely dead path emits small NOISE, which the ratio arm
    # still reads as negative (see test_gaussian_noise_llrs_are_negative).
    ev = soft_evidence(_llr_file(tmp_path, np.zeros(4000)))
    assert ev == []


def test_too_few_items_yield_nothing(tmp_path: Path) -> None:
    assert soft_evidence(_llr_file(tmp_path, np.ones(100))) == []


def test_none_and_missing_paths_yield_nothing(tmp_path: Path) -> None:
    assert soft_evidence(None) == []
    assert soft_evidence(tmp_path / "absent.f32") == []


def test_bursty_duty_cycle_is_not_negative(tmp_path: Path) -> None:
    # ~50% idle TDMA: bursts of +-2 rails interleaved with idle 0-runs. The
    # whole-stream mean|x|/std|x| reads as noise; the active-portion gate must
    # recover the decodable bursts.
    rng = np.random.default_rng(7)
    blocks = []
    for _ in range(60):
        burst = rng.choice([-2.0, 2.0], size=132) + rng.normal(0, 0.3, 132)
        blocks.append(np.concatenate([burst, np.zeros(132)]))
    ev = soft_evidence(_llr_file(tmp_path, np.concatenate(blocks)))
    assert ev != []
    assert ev[0].assessment != "negative"


def test_clean_four_level_soft_stream_is_positive(tmp_path: Path) -> None:
    # 4-level FM discriminator: four well-separated levels, not the binary
    # antipodal shape the ratio/polarity/whiteness logic below assumes - a
    # clean M-ary eye must still read as signal-present, not no_signal
    rng = np.random.default_rng(0)
    syms = rng.choice([-3.0, -1.0, 1.0, 3.0], size=6000).astype(np.float32)
    syms += rng.normal(0.0, 0.08, syms.size).astype(np.float32)
    ev = soft_evidence(_llr_file(tmp_path, syms))
    assert [e.assessment for e in ev] == ["positive"]
    assert ev[0].metric == "soft_confidence"


def test_four_level_noise_blob_is_not_positive(tmp_path: Path) -> None:
    # a forced 4-cluster fit on unimodal noise measures low separation (no
    # real levels present); must fall through to the existing binary path
    # rather than the new multi-level branch, which stays silent on it
    blob = np.random.default_rng(1).normal(0.0, 1.0, 6000).astype(np.float32)
    ev = soft_evidence(_llr_file(tmp_path, blob))
    # the binary path must SPEAK (noise ratio ~1.3 reads negative), not just
    # not-positive: "not any positive" was vacuously true of [] too
    assert [e.assessment for e in ev] == ["negative"]


def test_symbols_level_clean_eye_is_detection_grade(tmp_path: Path) -> None:
    # a bare demod's per-symbol eye (decode_grade=False): a clean stream reads
    # signal-PRESENT under the detection-tier metric soft_eye, never the
    # decode-tier soft_confidence -- a clean eye does not validate the bits (a
    # mistimed matched filter emits a clean eye of wrong symbols)
    rng = np.random.default_rng(0)
    signs = rng.choice([-2.0, 2.0], size=4000)
    ev = soft_evidence(
        _llr_file(tmp_path, signs + rng.normal(0, 0.3, 4000)), decode_grade=False
    )
    assert [(e.metric, e.assessment) for e in ev] == [("soft_eye", "positive")]


def test_symbols_level_multilevel_eye_is_detection_grade(tmp_path: Path) -> None:
    # the 4-FSK bare front end (DMR): a clean M-ary eye is signal-present, but
    # from a bare demod tap it is detection-grade, not a decoded certification
    rng = np.random.default_rng(0)
    syms = rng.choice([-3.0, -1.0, 1.0, 3.0], size=6000).astype(np.float32)
    syms += rng.normal(0.0, 0.08, syms.size).astype(np.float32)
    ev = soft_evidence(_llr_file(tmp_path, syms), decode_grade=False)
    assert [(e.metric, e.assessment) for e in ev] == [("soft_eye", "positive")]


def test_symbols_level_noise_is_suppressed_not_negative(tmp_path: Path) -> None:
    # a per-symbol demod eye cannot assert signal-ABSENT: a correct discriminator
    # decode of a bursty capture reads below the noise floor, so a low-ratio
    # symbols-level stream yields no evidence (falls to "uncertain"), never the
    # "no_signal"-driving negative that a bits-level LLR earns here
    rng = np.random.default_rng(1)
    ev = soft_evidence(
        _llr_file(tmp_path, rng.normal(0, 1.0, 4000)), decode_grade=False
    )
    assert ev == []


def test_symbols_level_all_zero_is_suppressed_not_negative(tmp_path: Path) -> None:
    ev = soft_evidence(_llr_file(tmp_path, np.zeros(4000)), decode_grade=False)
    assert ev == []


def test_active_gate_keeps_bursty_llrs_above_the_positive_bar(tmp_path: Path) -> None:
    # _SOFT_ACTIVE_FRACTION is calibrated so a ~50%-idle bursty stream is judged
    # on its bursts: windowed power leaks idle into each burst edge, and a lower
    # gate keeps more of that idle, widening the magnitude spread and pushing the
    # ratio down toward the noise bar. Pins the direction and the margin, so a
    # retune that re-admits idle fails here instead of silently degrading a
    # decode verdict.
    rng = np.random.default_rng(0)
    parts: list[np.ndarray] = []
    for _ in range(16):
        rails = rng.choice([-2.0, 2.0], 1024) + rng.normal(0.0, 0.25, 1024)
        parts.append(rails)
        parts.append(rng.normal(0.0, 0.02, 1024))  # idle between bursts
    path = _llr_file(tmp_path, np.concatenate(parts))

    def ratio_at(fraction: float) -> float:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(quality, "_SOFT_ACTIVE_FRACTION", fraction)
            (ev,) = [e for e in soft_evidence(path) if e.metric == "soft_confidence"]
            assert ev.assessment == "positive"
            return ev.value

    shipped = ratio_at(quality._SOFT_ACTIVE_FRACTION)
    assert shipped >= _SOFT_POSITIVE * 2.0, shipped  # measured 5.5, bar 2.0
    # every step down the gate admits more idle and costs ratio
    assert shipped > ratio_at(0.25) > ratio_at(0.10) > ratio_at(0.02)


def _poisoned(tmp_path: Path, nan_fraction: float, name: str) -> Path:
    """A clean +-2.0 decision stream with a leading NaN span — a poisoned
    PREFIX, not the all-NaN degenerate case. Written straight to the file on
    purpose: this scores whatever hands quality a non-finite stream, so it must
    not depend on any one producer still being able to make one."""
    rng = np.random.default_rng(0)
    n_bad = int(_SOFT_SAMPLE_ITEMS * nan_fraction)
    n_ok = _SOFT_SAMPLE_ITEMS - n_bad
    # noise on the rails: a bare sign ladder is the degenerate decision
    # stream the distinct-values gate refuses to score
    clean = np.sign(rng.standard_normal(n_ok)) * 2.0 + rng.normal(0, 0.3, n_ok)
    values = np.concatenate([np.full(n_bad, np.nan), clean])
    p = tmp_path / f"{name}.f32"
    write_llrs(p, values.astype(np.float32))
    return p


@pytest.mark.parametrize("nan_fraction", [0.02, 0.5, 0.9, 0.98])
def test_a_partly_non_finite_stream_scores_nothing(
    tmp_path: Path, nan_fraction: float
) -> None:
    """np.isfinite() DROPPED the poisoned items and scored the survivors as if
    they were the stream: at 98% NaN, 1311 real items of 65536 still certified
    "decoded". Partial poisoning is the case that matters — the all-NaN one was
    benign, falling under _SOFT_MIN_ITEMS and reading uncertain anyway.

    Found via agc mode='power' dividing by the rms of a zero span, which is now
    floored at its source; a soft stream a caller hands us via input_path is
    never re-scanned by _reject_nonfinite's capture path, and a diverging
    equalizer can still produce one, so the reading is guarded here regardless
    of who poisons it."""
    assert soft_evidence(_poisoned(tmp_path, nan_fraction, "p")) == []


def test_the_same_stream_without_the_poison_does_certify(tmp_path: Path) -> None:
    # the control: the surviving span is genuinely clean, so the refusal above
    # is the non-finites and nothing else
    clean = _poisoned(tmp_path, 0.0, "clean")
    assert [e.assessment for e in soft_evidence(clean)] == ["positive"]


def test_a_poisoned_stream_says_so_in_the_rationale(tmp_path: Path) -> None:
    """Returning [] alone would read as "no soft stream" — the operator needs
    the reason, since the fix is in their spec, not their capture."""
    report = quality.assess_quality(
        registry=stage_registry(),
        census=[],
        diagnostics=[],
        marks=[],
        soft_stream=_poisoned(tmp_path, 0.5, "half"),
    )
    assert report.verdict is quality.Verdict.UNCERTAIN
    assert report.evidence == []
    assert "non-finite" in report.rationale
    assert "50.0%" in report.rationale
    # and it must not also claim there was no soft stream to score
    assert "no soft stream" not in report.rationale


def test_a_poisoned_stream_does_not_erase_independent_evidence(
    tmp_path: Path,
) -> None:
    """A sync searcher that cleared its own chance model is untouched by a
    poisoned soft tap — but the caveat still has to travel."""
    report = quality.assess_quality(
        registry=stage_registry(),
        census=[
            BlockCensus(
                block="sync_word[0]",
                kind="sync_word",
                items_in=60_000,
                windows_out=1000,
                chance_windows=234.4,
            )
        ],
        diagnostics=[],
        marks=[],
        soft_stream=_poisoned(tmp_path, 0.5, "half2"),
    )
    assert report.verdict is quality.Verdict.DECODED
    assert [e.metric for e in report.evidence] == ["sync_matches"]
    assert "non-finite" in report.rationale


def test_saturated_coinflip_decisions_are_not_confidences(tmp_path: Path) -> None:
    # A demod saturating on noise emits constant-magnitude +-1 decisions:
    # spread == 0 made the ratio infinite, so MAXIMUM confidence was the
    # signature of the failure mode, and the whiteness guard cannot object
    # because random signs are white at every lag. Decisions carry no
    # per-item confidence; the reading is untestable, not certifiable.
    rng = np.random.default_rng(0)
    ev = soft_evidence(_llr_file(tmp_path, rng.choice([-1.0, 1.0], size=4000)))
    assert ev == []


def test_near_constant_magnitude_is_saturation_not_confidence(tmp_path: Path) -> None:
    # the continuous twin: a limiter's output jitters at float precision, so
    # an exact-zero-spread check alone would miss it
    rng = np.random.default_rng(7)
    signs = rng.choice([-1.0, 1.0], size=4000)
    x = signs * (1.0 + rng.normal(0, 1e-6, 4000))
    assert soft_evidence(_llr_file(tmp_path, x)) == []


def test_even_period_repeats_are_not_whitened(tmp_path: Path) -> None:
    # ++-- has lag-1 sign correlation exactly 0.0: a whiteness guard that
    # looks one lag deep certifies every even-period repeat
    rng = np.random.default_rng(3)
    x = np.tile([2.0, 2.0, -2.0, -2.0], 1000) + rng.normal(0, 0.2, 4000)
    assert soft_evidence(_llr_file(tmp_path, x)) == []


def test_three_level_zero_heavy_noise_is_not_positive(tmp_path: Path) -> None:
    # random {-1, 0, +1}: with the erasure zeros dropped this is a coinflip
    # decision stream, and the multilevel fit must not see the zero spike as
    # a third level
    rng = np.random.default_rng(5)
    ev = soft_evidence(_llr_file(tmp_path, rng.choice([-1.0, 0.0, 1.0], size=6000)))
    assert ev == []


def test_one_sided_multilevel_ladder_is_not_positive(tmp_path: Path) -> None:
    # an all-positive 4-level ladder has no polarity at all: the multilevel
    # branch must pass the same polarity gate the binary branch always had
    rng = np.random.default_rng(6)
    x = rng.choice([1.0, 2.0, 3.0, 4.0], size=6000) + rng.normal(0, 0.05, 6000)
    assert soft_evidence(_llr_file(tmp_path, x)) == []


def test_erasure_zeros_do_not_tighten_a_noise_fit(tmp_path: Path) -> None:
    # depuncture scatters exact-0.0 LLRs from its null_source; scoring them as
    # measurements let the erasure spike tighten a k=8 fit on pure noise past
    # the separation bar (measured: 7/8 pure-noise captures read "decoded"
    # at keep_mask=[1,0], while the same capture at 10 dB read "no_signal")
    rng = np.random.default_rng(8)
    holed = np.zeros(16000)
    holed[0::2] = rng.normal(0, 1.0, 8000)
    ev = soft_evidence(_llr_file(tmp_path, holed))
    assert not any(e.assessment == "positive" for e in ev)


def test_erasures_do_not_cost_a_genuine_decode_its_positive(tmp_path: Path) -> None:
    # the control for the erasure drop: a real punctured decode keeps its
    # positive from the items that were actually measured
    rng = np.random.default_rng(9)
    holed = np.zeros(16000)
    holed[0::2] = rng.choice([-2.0, 2.0], size=8000) + rng.normal(0, 0.3, 8000)
    ev = soft_evidence(_llr_file(tmp_path, holed))
    assert [e.assessment for e in ev] == ["positive"]
