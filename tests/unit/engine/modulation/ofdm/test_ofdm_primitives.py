import numpy as np
import numpy.typing as npt

from marconi.engine.modulation.ofdm import primitives as p


def test_qpsk_lock_unit_on_clean_qpsk() -> None:
    pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    z = pts[np.random.default_rng(0).integers(0, 4, 4000)]
    assert p.qpsk_lock(z) > 0.99


def test_find_null_locates_low_energy_gap() -> None:
    rng = np.random.default_rng(1)
    sym_len, null_len, frame_len = 2552, 2656, 196608
    x = rng.standard_normal(frame_len * 2) + 1j * rng.standard_normal(frame_len * 2)
    x[5000 : 5000 + null_len] *= 0.01
    end = p.find_null(x, null_len=null_len, frame_len=frame_len, sym_len=sym_len)
    assert 5000 + null_len - 600 <= end <= 5000 + null_len + 600


def test_find_null_boundary_unbiased_in_noise() -> None:
    # A received null is noise-filled, not zero: at realistic SNR the returned
    # base must stay near the true frame start instead of crossing on the
    # first noise sample hundreds of samples inside the null.
    sym_len, null_len, frame_len = 2552, 2656, 196608
    start, n = 5000, 40000
    for snr_db in (6.0, 10.0):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            sig = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            sig[start : start + null_len] = 0
            scale = np.sqrt(10 ** (-snr_db / 10))
            x = sig + scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            end = p.find_null(
                x, null_len=null_len, frame_len=frame_len, sym_len=sym_len
            )
            err = end - (start + null_len)
            assert abs(err) <= 64, (snr_db, seed, err)


def _step_boundary_ref(
    pw: npt.NDArray[np.float64], lo: int, hi: int, *, floor: float, sig: float
) -> int | None:
    # scalar CUSUM argmin, kept as the exactness oracle for the vectorized
    # step_boundary
    seg = pw[lo:hi]
    n = seg.size
    if n < 2 or sig <= 0.0 or floor >= sig:
        return None
    floor = max(floor, 1e-12 * sig)
    best_k, best_cum, cum = 0, np.inf, 0.0
    for i, s in enumerate(seg):
        cum += np.log(floor / sig) + s * (1.0 / floor - 1.0 / sig)
        if cum < best_cum:
            best_k, best_cum = i, cum
    edge = best_k + 1
    return None if edge >= n else lo + edge


def _find_null_ref(
    x: npt.NDArray[np.complex128], *, null_len: int, win: int = 512
) -> int:
    # the per-sample scan mirroring find_null, kept as the exactness oracle
    # (integration fixtures pin find_null's +-1 boundary quirks)
    pw = np.abs(x) ** 2
    w = min(win, null_len)
    env = np.convolve(pw, np.ones(w) / w, mode="same")
    thresh = 0.25 * np.median(env)
    low = env < thresh
    i, n = 0, len(x)
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            if (j - i) > null_len * 0.5:
                run_sum = float(np.sum(pw[i:j]))
                floor = run_sum / (j - i)
                rest = (float(np.sum(pw)) - run_sum) / max(1, n - (j - i))
                edge = _step_boundary_ref(
                    pw, max(0, j - w), min(n, j + w), floor=floor, sig=rest
                )
                if edge is None or float(np.mean(pw[edge : j + w])) <= thresh:
                    raise ValueError("no null")
                return edge
            i = j
        else:
            i += 1
    raise ValueError("no null")


def test_find_null_matches_scalar_reference_exactly() -> None:
    null_len = 300
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4000, 12000))
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        if seed % 4:  # three planted-null shapes per no-null case
            start = int(rng.integers(0, n - 2 * null_len))
            depth = float(rng.uniform(0.001, 0.1))
            length = int(rng.integers(int(null_len * 0.6), 2 * null_len))
            x[start : start + length] *= depth
        try:
            expect: int | None = _find_null_ref(x, null_len=null_len)
        except ValueError:
            expect = None
        if expect is None:
            try:
                p.find_null(x, null_len=null_len, frame_len=2048, sym_len=256)
                raise AssertionError(f"seed {seed}: expected no-null raise")
            except ValueError:
                continue
        got = p.find_null(x, null_len=null_len, frame_len=2048, sym_len=256)
        assert got == expect, (seed, got, expect)
