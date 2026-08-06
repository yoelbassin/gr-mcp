from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.io.bitfile import write_llrs
from marconi.engine.quality import soft_evidence


def _llr_file(tmp_path: Path, values: np.ndarray) -> Path:
    p = tmp_path / "llrs.f32"
    write_llrs(p, values.astype(np.float32))
    return p


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
    assert all(e.assessment != "negative" for e in ev)


def test_gaussian_noise_llrs_are_negative(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    ev = soft_evidence(_llr_file(tmp_path, rng.normal(0, 1.0, 4000)))
    assert [e.assessment for e in ev] == ["negative"]


def test_all_zero_stream_is_negative(tmp_path: Path) -> None:
    ev = soft_evidence(_llr_file(tmp_path, np.zeros(4000)))
    assert [e.assessment for e in ev] == ["negative"]


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
    # C4FM-like discriminator: four well-separated levels, not the binary
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
    assert not any(e.assessment == "positive" for e in ev)
