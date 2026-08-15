from pathlib import Path

import numpy as np

from marconi.survey import survey_iq

# A NOISELESS 4-FSK made this test flake on CI, and the mechanism is worth
# keeping: eye_openness is a level separation divided by the within-level
# spread, and with no noise that spread is numerical dust — the measured eye
# came back 3.0e7 (pi/1e-8), five orders of magnitude outside the 8.x range the
# _CLEAR_EYE ladder was calibrated against. Ranking then compares 1/epsilon
# against 1/epsilon: a CI run measured the true line at 0.57x its local value,
# which dropped it under the _HARMONIC_FRAC admission bar and handed rank 0 to
# the 2x harmonic. At 30 dB the true line reads 14.4-15 and clears that bar by
# 57-67% instead of 16-20% (12 seeds each). Lowering this materially re-enters
# the fallback: at 20 dB the eye reads 4.8, under _CLEAR_EYE, and rank 0 is the
# true rate in only 5 of 12 seeds.
_SNR_DB = 30.0
# above this the eye is no longer a measurement, it is 1/(numerical noise)
_DEGENERATE_EYE = 1e3


def _fsk4(fs: float, rate: float, dev: float, n_sym: int) -> np.ndarray:
    sps = int(round(fs / rate))
    syms = np.array([-3.0, -1.0, 1.0, 3.0])[np.random.randint(0, 4, n_sym)]
    phase = 2 * np.pi * np.cumsum(np.repeat(syms * dev, sps)) / fs
    return np.exp(1j * phase).astype(np.complex64)


def _awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    noise = (np.random.randn(x.size) + 1j * np.random.randn(x.size)) / np.sqrt(2)
    scale = np.sqrt(np.mean(np.abs(x) ** 2) / (10 ** (snr_db / 10)))
    noisy: np.ndarray = (x + scale * noise).astype(np.complex64)
    return noisy


def test_survey_iq_recovers_fsk_ground_truth(tmp_path: Path) -> None:
    np.random.seed(0)
    fs, rate, dev = 48_000.0, 2_400.0, 1_200.0
    x = _awgn(_fsk4(fs, rate, dev, 8000), _SNR_DB)
    p = tmp_path / "fsk4.cf32"
    x.tofile(p)
    r = survey_iq(p, fs)
    assert r.span_samples == x.size
    # rank 0, not membership in an unbounded list: the product's documented
    # claim is that candidates_hz[0] is the rate, and the true rate measured
    # rank 4 of 5 here without this pin
    assert abs(r.symbol_rate.candidates_hz[0] - rate) < 0.05 * rate
    # the promotion path actually fired, and on a measurement rather than on
    # 1/epsilon — without this the rank-0 pin can be satisfied by the
    # degenerate regime that made this test flake
    assert r.symbol_rate.eye_confirmed
    assert max(r.symbol_rate.eye_openness) < _DEGENERATE_EYE
    assert len(r.symbol_rate.eye_openness) == len(r.symbol_rate.candidates_hz)
    assert 3 <= len(r.inst_freq.peaks_hz) <= 5
    assert r.envelope.const_envelope_ratio < 0.1
    assert r.spectrum.occupied_bw_hz > 0.0
