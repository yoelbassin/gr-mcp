import numpy as np

from marconi.survey.measure import _eye_openness


def _fsk_instfreq(
    n_sym: int, sps: float, levels: list[float], sfo: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    syms = rng.choice(levels, size=n_sym)
    # fractional samples-per-symbol with mild clock drift (SFO) -> real impairment
    span = int(n_sym * sps)
    t = np.arange(span, dtype=np.float64)
    sym_index = np.floor(t / (sps * (1.0 + sfo))).astype(int)
    sym_index = np.clip(sym_index, 0, n_sym - 1)
    x = syms[sym_index].astype(np.float64)
    x += rng.normal(0.0, 0.06, x.size)  # discriminator noise
    x = np.roll(x, 3)  # fractional-ish STO
    return x


def test_eye_opens_at_true_rate_not_at_spurious_rates() -> None:
    fs, true_rate, sps = 39062.0, 4137.0, 9.44  # generic baud, not a protocol value
    instfreq = _fsk_instfreq(3000, sps, [-3.0, -1.0, 1.0, 3.0], sfo=5e-5, seed=0)
    active = np.ones(instfreq.size, dtype=bool)
    at_true = _eye_openness(instfreq, active, fs, true_rate)
    at_lo = _eye_openness(instfreq, active, fs, true_rate * 0.82)
    at_hi = _eye_openness(instfreq, active, fs, true_rate * 1.23)
    assert at_true > 8.0
    assert at_true > 2.0 * max(at_lo, at_hi)


def test_eye_openness_zero_when_no_active_run() -> None:
    instfreq = np.random.default_rng(1).normal(0.0, 1.0, 5000)
    active = np.zeros(instfreq.size, dtype=bool)
    assert _eye_openness(instfreq, active, 39062.0, 4137.0) == 0.0
