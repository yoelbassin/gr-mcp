import numpy as np

from marconi.bits.carriers import RxCarrier
from marconi.bits.framing import normalize_rx


def test_normalize_removes_dc_and_scales_per_span() -> None:
    span = np.array([10.0, 12.0, 8.0, 10.0], np.float32)  # dc≈10
    sym = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        span_symbols=4,
        dc="median",
        gain_percentile=50.0,
    )
    assert out.symbols is not None
    out_sym = out.symbols
    assert abs(float(np.median(out_sym[:4]))) < 1e-5
    assert out.marks == (0,)
    assert np.allclose(out_sym[4:], 0.0)  # outside the span, untouched
