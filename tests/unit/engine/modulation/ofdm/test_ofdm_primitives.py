import numpy as np

from marconi.engine.modulation.ofdm import primitives as p


def test_qpsk_lock_unit_on_clean_qpsk():
    pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    z = pts[np.random.default_rng(0).integers(0, 4, 4000)]
    assert p.qpsk_lock(z) > 0.99


def test_find_null_locates_low_energy_gap():
    rng = np.random.default_rng(1)
    sym_len, null_len, frame_len = 2552, 2656, 196608
    x = rng.standard_normal(frame_len * 2) + 1j * rng.standard_normal(frame_len * 2)
    x[5000 : 5000 + null_len] *= 0.01
    end = p.find_null(x, null_len=null_len, frame_len=frame_len, sym_len=sym_len)
    assert 5000 + null_len - 600 <= end <= 5000 + null_len + 600


def _find_null_ref(x, *, null_len, win=512):
    # the pre-vectorization per-sample scan, kept as the exactness oracle
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
                refined = max(0, j - w)
                while refined < n and pw[refined] <= thresh:
                    refined += 1
                if refined >= n:
                    raise ValueError("no null")
                return refined
            i = j
        else:
            i += 1
    raise ValueError("no null")


def test_find_null_matches_scalar_reference_exactly():
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
