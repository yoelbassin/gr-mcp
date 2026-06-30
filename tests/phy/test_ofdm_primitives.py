import numpy as np

from marconi.phy.modulation.ofdm import primitives as p


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
