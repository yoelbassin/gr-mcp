from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.phy.coding.carrier import CodingCarrier
from marconi.phy.coding.ops_symbols import m_slice_rx, normalize_rx, sync_symbols_rx
from marconi.phy.coding.stages_symbols import (
    MSlice,
    SymbolMap,
    _MSliceParams,
)
from marconi.phy.stages import stage_registry


def test_registry_holds_both_flavors() -> None:
    reg = stage_registry()
    assert reg["fsk"].engine == "gr"
    assert reg["sync_word"].engine == "coding"
    # demoted 2026-07: LoRa's header chapter is agent/test-side orchestration
    # (tests/helpers/css_explicit.py), not registered vocabulary
    assert "css_explicit_decode" not in reg


def test_m_slice_out_descriptor_hardens() -> None:
    d = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    out = MSlice().out_descriptor(d, {"thresholds": [0.0], "levels": [0, 1]})
    assert out.item_type == "s" and out.carrier is Carrier.HARD


def test_symbol_map_out_descriptor_lands_on_hard_bits() -> None:
    d = Descriptor(Level.SYMBOLS, "s", carrier=Carrier.HARD)
    out = SymbolMap().out_descriptor(
        d, {"code_bits": 1, "data_bits": 1, "table": [0, 1]}
    )
    assert out == Descriptor(Level.BITS, "b", carrier=Carrier.HARD)


# --- sync_symbols_rx (ported from tests/bits/test_sync_symbols.py) ---


def test_sync_symbols_marks_burst_start() -> None:
    # pattern of signs {+,+,-,-}; burst starts 3 symbols before the pattern.
    sym = np.array([0.0, 0.0, 0.0, 0.5, 0.4, -0.6, -0.7, 0.0], np.float32)
    out = sync_symbols_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=0,
        pre_symbols=3,
    )
    assert out.marks == (0,)  # pattern at index 3, minus pre_symbols 3


def test_sync_symbols_tolerates_one_error() -> None:
    sym = np.array([0.5, 0.4, 0.9, -0.7], np.float32)  # 3rd sign wrong vs pattern
    out = sync_symbols_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=1,
        pre_symbols=0,
    )
    assert out.marks == (0,)


def test_sync_symbols_drops_negative_offset_on_exact_match() -> None:
    sym = np.array([0.0, 0.0, 0.0, 0.5, 0.4, -0.6, -0.7, 0.0], np.float32)
    out = sync_symbols_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 1, -1, -1],
        max_errors=0,
        pre_symbols=4,  # match is at index 3; pre_symbols exceeds it
    )
    assert out.marks == ()


def test_sync_symbols_dont_care_excluded_from_agreement() -> None:
    sym = np.array(
        [0.5, 0.0, -0.7], np.float32
    )  # middle sign matches neither +1 nor -1
    out = sync_symbols_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym),
        pattern=[1, 0, -1],
        max_errors=0,
        pre_symbols=0,
    )
    assert out.marks == (0,)


def test_sync_symbols_stage_registered() -> None:
    assert "sync_symbols" in stage_registry()


# --- normalize_rx (ported from tests/bits/test_normalize.py) ---


def test_normalize_removes_dc_and_scales_per_span() -> None:
    span = np.array([10.0, 12.0, 8.0, 10.0], np.float32)  # dc≈10
    sym = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out = normalize_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
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
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
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
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
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
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
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
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        span_symbols=4,
        dc="median",
        gain_percentile=50.0,
    )
    assert out.symbols is not None
    assert not np.isnan(out.symbols).any()

    span = np.array([10.0, 12.0, 8.0, 10.0], np.float32)
    sym2 = np.concatenate([span, np.zeros(4, np.float32)]).astype(np.float32)
    out2 = normalize_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym2, marks=(0,)),
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
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0, 6)),
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
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=span, marks=(0,)),
        span_symbols=4,  # mark + span_symbols runs past the array end
        dc="median",
    )
    assert out.symbols is not None
    assert out.symbols.size == 3
    assert abs(float(np.median(out.symbols))) < 1e-5


def test_normalize_stage_registered() -> None:
    assert "normalize" in stage_registry()


# --- m_slice_rx (ported from tests/bits/test_m_slice.py) ---


def test_m_slice_maps_regions_to_levels() -> None:
    sym = np.array([-1.0, -0.5, 0.5, 1.0], np.float32)
    out = m_slice_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(2,)),
        thresholds=[-0.667, 0.0, 0.667],
        levels=[3, 2, 0, 1],
    )
    assert out.symbols is not None
    assert out.symbols.dtype == np.int16
    assert out.symbols.tolist() == [3, 2, 0, 1]
    assert out.marks == (2,)


def test_m_slice_symbol_exactly_on_threshold_lands_in_lower_region() -> None:
    sym = np.array([0.0], np.float32)  # exactly thresholds[1]
    out = m_slice_rx(
        CodingCarrier(bits=np.zeros(0, np.uint8), symbols=sym, marks=(0,)),
        thresholds=[-0.667, 0.0, 0.667],
        levels=[3, 2, 0, 1],
    )
    assert out.symbols is not None
    assert out.symbols.tolist() == [2]  # searchsorted(side='left') -> region 1


def test_m_slice_stage_registered() -> None:
    assert "m_slice" in stage_registry()


def test_m_slice_params_length_mismatch_raises() -> None:
    with pytest.raises(ValidationError):
        _MSliceParams.model_validate(
            {"thresholds": [-0.667, 0.0, 0.667], "levels": [3, 2, 0]}
        )


def test_m_slice_params_non_ascending_thresholds_raises() -> None:
    with pytest.raises(ValidationError):
        _MSliceParams.model_validate(
            {"thresholds": [0.0, 0.0, 1.0], "levels": [3, 2, 0, 1]}
        )
