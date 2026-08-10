from __future__ import annotations

import numpy as np
from helpers._fakegr import FAKE_GR

from marconi.engine.backends.gnuradio.embedded.decision import make_peak_decision


def _scalar_ref(vecs: np.ndarray, divisor: int, modulo: int) -> list[int]:
    return [round(int(np.argmax(v)) / divisor) % modulo for v in vecs]


def test_peak_decision_matches_scalar_reference_including_half_boundary() -> None:
    vlen, divisor, modulo = 8, 2, 5
    blk = make_peak_decision(FAKE_GR, vlen=vlen, divisor=divisor, modulo=modulo)
    vecs = np.random.default_rng(0).standard_normal((10, vlen)).astype(np.float32)
    # argmax=3 -> 3/2 = 1.5: pins round-half-to-even (1.5 -> 2, not 1)
    vecs[0] = 0.0
    vecs[0, 3] = 1.0
    out = np.zeros(10, np.int16)
    k = blk.work([vecs], [out])
    assert k == 10
    assert out.tolist() == _scalar_ref(vecs, divisor, modulo)


def test_peak_decision_full_precision_index_beyond_int16() -> None:
    # the raison d'etre: a peak past 32768 must not wrap (stock argmax_fs does).
    # divisor=zero_pad folds the big index down to a symbol < modulo=2**sf.
    vlen, divisor, modulo = 40000, 16, 4096
    blk = make_peak_decision(FAKE_GR, vlen=vlen, divisor=divisor, modulo=modulo)
    vecs = np.zeros((1, vlen), np.float32)
    vecs[0, 39000] = 1.0
    out = np.zeros(1, np.int16)
    blk.work([vecs], [out])
    assert int(np.argmax(vecs[0])) == 39000
    assert out.tolist() == [round(39000 / divisor) % modulo]  # 2438, no int16 wrap
