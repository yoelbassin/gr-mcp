"""Survey is the agent's first tool on an unknown band and its answers become
the hypotheses; each test here pins a measured way the answers were wrong.
The common shape: a reference level computed from data that includes the
thing it was measuring (the burst floor contained the emitter, the centroid
contained the noise, the DC estimate contained the carrier)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import _synth

from marconi.survey.iqfile import CaptureNotRaw, sample_iq
from marconi.survey.measure import _symbol_rate, survey_iq

_FS = 1_000_000.0


def _write(tmp_path: Path, x: np.ndarray, name: str = "cap.cf32") -> Path:
    p = tmp_path / name
    x.astype(np.complex64).tofile(p)
    return p


def _tone(n: int, f_hz: float, fs: float, amp: float = 1.0) -> np.ndarray:
    return amp * np.exp(2j * np.pi * f_hz / fs * np.arange(n))


def test_continuous_emitter_does_not_delete_weak_bursts(tmp_path: Path) -> None:
    # a continuous carrier 20 dB over noise, nowhere near the weak bursts:
    # with the floor at the median of per-probe percentiles it BECAME the
    # floor at duty >= 50% and every weak burst fell under the 4x bar -
    # measured count=0 for 57 real bursts, "the band is dead"
    rng = np.random.default_rng(0)
    n = 1 << 21
    noise = (rng.normal(0, 0.05, n) + 1j * rng.normal(0, 0.05, n)) / np.sqrt(2)
    emitter = _tone(n, 200_000.0, _FS, amp=0.5)  # 20 dB over the noise, always on
    x = noise + emitter
    period, blen = 37117, 4096
    n_bursts = 0
    for s in range(0, n - blen, period):
        x[s : s + blen] += _tone(blen, 0.0, _FS, amp=1.5)
        n_bursts += 1
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.count >= int(0.8 * n_bursts), (res.bursts.count, n_bursts)


@pytest.mark.parametrize("duty", [0.76, 0.80, 0.90])
def test_high_duty_bursts_are_not_read_as_a_dead_band(
    tmp_path: Path, duty: float
) -> None:
    # once within-probe duty passed 0.75, the p25 of smoothed power was the
    # SIGNAL level in every probe (tone 26 dB over the noise here), the
    # aggregate floor rode it, and the 4x bar sat above the signal itself:
    # measured cliff — duty 0.74 -> count 128, duty 0.737; duty 0.76 ->
    # count 0, duty 0.0 ("the band is dead")
    rng = np.random.default_rng(int(duty * 100))
    n = 1 << 19
    noise = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    period = 4096
    mask = (np.arange(n) % period) < int(duty * period)
    x = (_tone(n, 100_000.0, _FS) * mask + noise).astype(np.complex64)
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.count > 0, res.bursts.count
    assert abs(res.bursts.duty_cycle - duty) <= 0.1, res.bursts.duty_cycle
    assert res.bursts.dominant_period_samples is not None


@pytest.mark.parametrize("snr_db", [26.0, 8.0])
def test_continuous_carrier_reports_duty_near_one(
    tmp_path: Path, snr_db: float
) -> None:
    # an emitter that never turns off leaves no observable noise floor, and
    # the bar built from its own level read it as a DEAD band: duty 0.0.
    # genuinely continuous means count may be 0 — but duty must say "always
    # on", not "never on". 8 dB gates the fallback's measured ~6 dB coverage
    # floor with margin (raw-shape 0.44 vs the 0.3 bar); an earlier bar
    # mismeasured against mislabeled SNR flipped to "dead" below exactly
    # 10 dB.
    rng = np.random.default_rng(11)
    n = 1 << 19
    npw = 10 ** (-snr_db / 10)
    noise = np.sqrt(npw / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x = (_tone(n, 100_000.0, _FS) + noise).astype(np.complex64)
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.duty_cycle >= 0.9, (snr_db, res.bursts.duty_cycle)


def test_always_on_ook_reads_duty_zero_the_documented_blind_spot(
    tmp_path: Path,
) -> None:
    # pins the explain("survey.bursts") claim: an always-on amplitude-
    # modulated emitter reads duty 0.0. Its raw-envelope shape (50%-ones OOK
    # measured 0.003-0.043; 16QAM 0.188) sits under the carrier bar because
    # nothing separates it honestly from exponential noise's 0.152 — the
    # envelope minima ARE sample-scale quiet.
    rng = np.random.default_rng(13)
    n = 1 << 19
    bits = rng.integers(0, 2, n // 8 + 1).astype(np.float64)
    env = np.repeat(bits, 8)[:n]
    noise = np.sqrt(0.01 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x = (env * _tone(n, 100_000.0, _FS) + noise).astype(np.complex64)
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.duty_cycle < 0.1, res.bursts.duty_cycle


def test_zero_noise_quiet_side_does_not_zero_the_bar(tmp_path: Path) -> None:
    # a squelched recorder writes EXACT zeros between bursts; those probes'
    # quiet-side p5 is 0.0 and a floor built from it collapses the activity
    # bar to zero, turning every noise sample in the unsquelched half
    # "active": measured with the zero guard deleted — count 1, duty 1.0
    # against a truth of ~0.4.
    half = (1 << 19) // 2
    mask = (np.arange(half) % 4096) < int(0.8 * 4096)
    squelched = _tone(half, 110_000.0, _FS) * mask
    rng = np.random.default_rng(9)
    noise = np.sqrt(0.0025 / 2) * (
        rng.standard_normal(half) + 1j * rng.standard_normal(half)
    )
    x = np.concatenate([squelched, noise]).astype(np.complex64)
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.count >= 50, res.bursts.count
    assert abs(res.bursts.duty_cycle - 0.4) <= 0.1, res.bursts.duty_cycle


def test_dead_noise_band_still_reads_duty_zero(tmp_path: Path) -> None:
    # the continuous-carrier fallback must not turn a dead band into an
    # always-on one: pure noise has the same probe-scale power profile as a
    # carrier, and only the raw-sample envelope shape tells them apart.
    rng = np.random.default_rng(12)
    n = 1 << 19
    x = (0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.duty_cycle < 0.1, res.bursts.duty_cycle


def test_burst_cadence_stays_out_of_the_default_rate_band(tmp_path: Path) -> None:
    # a bursty capture puts its burst-repetition comb inside the default
    # search band, and the true symbol rate never reached the candidate list
    # (measured: candidates [74.9 .. 861.2] for a 125 kBd signal); when the
    # burst block measures a dominant period, the default floor must clear
    # its low harmonics
    rng = np.random.default_rng(1)
    n = 1 << 20
    x = (rng.normal(0, 0.02, n) + 1j * rng.normal(0, 0.02, n)).astype(np.complex128)
    period, blen = 16384, 3277
    for s in range(0, n - blen, period):
        x[s : s + blen] += _tone(blen, 30_000.0, _FS)
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.bursts.dominant_period_samples is not None
    cadence = _FS / res.bursts.dominant_period_samples
    assert res.symbol_rate.search_lo_hz >= 4.0 * cadence * 0.9


def test_phase_branch_keeps_the_time_base(tmp_path: Path) -> None:
    # _gate DELETED the idle samples and _clock_spectrum then used the
    # uncompressed sample rate: a periodicity of P samples reported at f/rho
    # (measured +4.5..9.3% high, 20-104 clock bins off, while the docstring
    # promises candidates within one bin). CPFSK's discriminator has a hard
    # symbol line, so the phase branch dominates for it.
    rng = np.random.default_rng(2)
    sps, n_bits = 64, 4000
    bits = rng.integers(0, 2, n_bits)
    true_rate = _FS / sps
    x = _synth.cpfsk(bits, sps=sps, deviation=true_rate / 4.0, sample_rate=_FS)
    res = survey_iq(_write(tmp_path, np.asarray(x)), _FS)
    best = min(res.symbol_rate.candidates_hz, key=lambda c: abs(c - true_rate))
    assert abs(best - true_rate) <= 2.0 * res.symbol_rate.clock_resolution_hz, (
        best,
        true_rate,
        res.symbol_rate.clock_resolution_hz,
    )


def test_narrow_signal_offset_survives_low_snr(tmp_path: Path) -> None:
    # the spectral centroid included the noise floor, so for a narrow signal
    # it drifted toward zero as SNR fell and VETOED the correct mpsk line
    # (measured: reported 103 kHz at 12 dB for a +200 kHz QPSK); the moments
    # must run over the floor-subtracted PSD
    rng = np.random.default_rng(3)
    n = 1 << 19
    sps = 20
    bits = rng.integers(0, 2, 2 * (n // sps))
    sig = np.asarray(_synth.shaped_constellation(bits, scheme="psk", order=4, sps=sps))
    sig = sig[:n] * _tone(min(n, sig.size), 200_000.0, _FS)[: min(n, sig.size)]
    noise_rms = float(np.sqrt(np.mean(np.abs(sig) ** 2))) / 10 ** (12.0 / 20.0)
    noise = (rng.normal(0, 1, sig.size) + 1j * rng.normal(0, 1, sig.size)) * (
        noise_rms / np.sqrt(2)
    )
    res = survey_iq(_write(tmp_path, sig + noise), _FS)
    assert abs(res.carrier.offset_hz - 200_000.0) < 5_000.0, res.carrier
    assert res.carrier.off_center is True


def test_band_edge_carrier_is_flagged_ambiguous(tmp_path: Path) -> None:
    # a signal whose occupied band wraps +-fs/2 has a meaningless centroid;
    # dealiasing against it reported +499 kHz as -999.9 Hz with every
    # confidence flag clear
    rng = np.random.default_rng(4)
    n = 1 << 19
    sps = 8
    bits = rng.integers(0, 2, 2 * (n // sps))
    sig = np.asarray(_synth.shaped_constellation(bits, scheme="psk", order=4, sps=sps))
    sig = sig[:n] * _tone(min(n, sig.size), 499_000.0, _FS)[: min(n, sig.size)]
    res = survey_iq(_write(tmp_path, sig), _FS)
    wrapped_err = min(
        abs(res.carrier.offset_hz - 499_000.0),
        abs(res.carrier.offset_hz - 499_000.0 + _FS),
    )
    assert res.carrier.offset_ambiguous or wrapped_err < 5_000.0, res.carrier


def test_dc_carrier_keeps_its_phase_concentration(tmp_path: Path) -> None:
    # tuning the radio ONTO the signal is the common case, and the DC-blind
    # mean subtraction annihilated it: a clean CW at exactly 0 Hz measured
    # phase_concentration ~1.0 ("no phase-coherent carrier") while the same
    # tone 100 Hz off measured tens of thousands
    x = _tone(1 << 18, 0.0, _FS)
    res = survey_iq(_write(tmp_path, x), _FS)
    assert res.carrier.phase_concentration.order_2 > 100.0, res.carrier


def test_wav_container_is_refused_with_the_fix_named(tmp_path: Path) -> None:
    # a RIFF header reinterpreted as cf32 surveyed "confidently": peak at the
    # Nyquist bin, occupied_bw 0.0, carrier NaN - and NaN is not valid JSON
    p = tmp_path / "cap.wav"
    payload = np.zeros(40_000, np.int16)
    p.write_bytes(b"RIFF" + (payload.size * 2 + 36).to_bytes(4, "little") + b"WAVE")
    with p.open("ab") as f:
        payload.tofile(f)
    with pytest.raises(CaptureNotRaw, match="raw"):
        sample_iq(p)


def _noisy_2fsk(snr_db: float, seed: int, n: int = 1 << 18) -> np.ndarray:
    sps = 8
    dev = 0.35 * _FS / sps
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n // sps + 2)
    f = np.repeat(np.where(bits, dev, -dev), sps)[:n]
    x = np.exp(2j * np.pi * np.cumsum(f) / _FS)
    scale = 10 ** (-snr_db / 20)
    noise = scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    out: np.ndarray = (x + noise).astype(np.complex64)
    return out


def test_eye_rerank_does_not_crown_a_noise_line(tmp_path: Path) -> None:
    # On 2FSK + AWGN the eye at ANY rate samples the modulation's bimodal
    # instantaneous frequency, so noise-rate lines carry eyes 6.9-11.3 at
    # SNR 20-25 — clearing the eye bar and the 0.6x-of-max fraction — and
    # min-by-rate crowned a strength-0.04-0.05 noise line as candidates_hz[0]
    # with eye_confirmed True on 5/5 seeds. Strength is the discriminator:
    # the same sweep measured every real line >= 0.8 of max.
    true_rate = _FS / 8
    for seed in (5, 6, 7, 8, 9):
        s = _symbol_rate(_noisy_2fsk(20.0, seed), _FS, _FS / 1000, _FS / 2)
        near_true = abs(s.candidates_hz[0] - true_rate) < 0.03 * true_rate
        assert near_true or not s.eye_confirmed, (
            seed,
            s.candidates_hz,
            s.strengths,
            s.eye_openness,
            s.eye_confirmed,
        )


def test_segments_accumulation_is_bounded(tmp_path: Path) -> None:
    # 47,259 segments were accumulated to ship 512: a 300 s capture at
    # 20 Msps holds ~4.9 GB of tuples, all discarded but held at once
    rng = np.random.default_rng(5)
    n = 1 << 20
    x = (rng.normal(0, 0.02, n) + 1j * rng.normal(0, 0.02, n)).astype(np.complex128)
    for s in range(0, n - 64, 700):
        x[s : s + 64] += 1.0
    res = survey_iq(_write(tmp_path, x), _FS)
    assert len(res.bursts.segments) <= 512
    assert res.bursts.count > 512  # the count is still the truth
