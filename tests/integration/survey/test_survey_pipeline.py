from pathlib import Path

import numpy as np

from marconi.survey import survey_iq


def _fsk4(fs: float, rate: float, dev: float, n_sym: int) -> np.ndarray:
    sps = int(round(fs / rate))
    syms = np.array([-3.0, -1.0, 1.0, 3.0])[np.random.randint(0, 4, n_sym)]
    phase = 2 * np.pi * np.cumsum(np.repeat(syms * dev, sps)) / fs
    return np.exp(1j * phase).astype(np.complex64)


def test_survey_iq_recovers_fsk_ground_truth(tmp_path: Path) -> None:
    np.random.seed(0)
    fs, rate, dev = 48_000.0, 2_400.0, 1_200.0
    x = _fsk4(fs, rate, dev, 8000)
    p = tmp_path / "fsk4.cf32"
    x.tofile(p)
    r = survey_iq(p, fs)
    assert r.span_samples == x.size
    # rank 0, not membership in an unbounded list: the product's documented
    # claim is that candidates_hz[0] is the rate, and the true rate measured
    # rank 4 of 5 here without this pin
    assert abs(r.symbol_rate.candidates_hz[0] - rate) < 0.05 * rate
    assert len(r.symbol_rate.eye_openness) == len(r.symbol_rate.candidates_hz)
    assert 3 <= len(r.inst_freq.peaks_hz) <= 5
    assert r.envelope.const_envelope_ratio < 0.1
    assert r.spectrum.occupied_bw_hz > 0.0
