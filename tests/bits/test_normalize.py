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


def test_normalize_mean_removes_mean() -> None:
    span = np.array([9.0, 11.0, 10.0, 30.0], np.float32)  # mean=15
    sym = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        span_symbols=4,
        dc="mean",
        gain_percentile=None,
    )
    assert out.symbols is not None
    assert abs(float(out.symbols[:4].mean())) < 1e-5


def test_normalize_dc_none_leaves_span_uncentered() -> None:
    span = np.array([10.0, 12.0, 8.0, 10.0], np.float32)  # mean=10, not 0
    sym = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        span_symbols=4,
        dc="none",
        gain_percentile=None,
    )
    assert out.symbols is not None
    assert np.allclose(out.symbols[:4], span)  # untouched: DC left in place


def test_normalize_gain_none_skips_rescale() -> None:
    span = np.array([10.0, 12.0, 8.0, 10.0], np.float32)  # dc≈10
    sym = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        span_symbols=4,
        dc="median",
        gain_percentile=None,
    )
    assert out.symbols is not None
    # DC-removed values only (median-subtracted), no rescale applied.
    assert np.allclose(out.symbols[:4], span - np.median(span))


def test_normalize_empty_gain_selection_produces_no_nan() -> None:
    zero_span = np.zeros(4, np.float32)
    sym = np.concatenate([zero_span, np.zeros(4, np.float32)]).astype(np.float32)
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        span_symbols=4,
        dc="median",
        gain_percentile=50.0,
    )
    assert out.symbols is not None
    assert not np.isnan(out.symbols).any()

    span = np.array([10.0, 12.0, 8.0, 10.0], np.float32)
    sym2 = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out2 = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym2, marks=(0,)),
        span_symbols=4,
        dc="median",
        gain_percentile=100.0,
    )
    assert out2.symbols is not None
    assert not np.isnan(out2.symbols).any()


def test_normalize_multiple_marks_independently_normalized() -> None:
    span_a = np.array([10.0, 12.0, 8.0, 10.0], np.float32)  # dc≈10
    gap = np.array([0.0, 0.0], np.float32)
    span_b = np.array([100.0, 104.0, 96.0, 100.0], np.float32)  # dc≈100
    sym = np.concatenate([span_a, gap, span_b]).astype(np.float32)
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0, 6)),
        span_symbols=4,
        dc="median",
    )
    assert out.symbols is not None
    out_sym = out.symbols
    assert abs(float(np.median(out_sym[0:4]))) < 1e-5
    assert abs(float(np.median(out_sym[6:10]))) < 1e-5
    assert np.allclose(out_sym[4:6], 0.0)  # gap between marks untouched


def test_normalize_end_of_array_clamp() -> None:
    span = np.array([10.0, 12.0, 8.0], np.float32)  # only 3 symbols left
    out = normalize_rx(
        RxCarrier(bits=np.zeros(0, np.uint8), symbols=span, marks=(0,)),
        span_symbols=4,  # mark + span_symbols runs past the array end
        dc="median",
    )
    assert out.symbols is not None
    assert out.symbols.size == 3
    assert abs(float(np.median(out.symbols))) < 1e-5


def test_normalize_stage_registered() -> None:
    from marconi.bits.registry import registry

    assert "normalize" in registry()
