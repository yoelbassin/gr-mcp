from __future__ import annotations

import numpy as np
import pytest
from phy._css_lora import HEADER as _HEADER
from phy.coding._css_synth import SF11_SYMBOLS as _SF11_SYMBOLS
from phy.coding._css_synth import assemble as _assemble
from phy.coding._css_synth import encode_header as _encode_header
from phy.coding._css_synth import encode_payload as _encode_payload
from phy.coding._css_synth import run as _run
from pydantic import ValidationError

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.stages import StageDirectionError
from marconi.phy.coding.builder import CodingBuilder
from marconi.phy.coding.carrier import CodingCarrier
from marconi.phy.coding.ops_symbols import m_slice_rx, normalize_rx, sync_symbols_rx
from marconi.phy.coding.stages_symbols import (
    CssExplicitDecode,
    MSlice,
    SymbolMap,
    _ExplicitParams,
    _MSliceParams,
)
from marconi.phy.stages import stage_registry

_EXPLICIT_PARAMS = dict(_HEADER)


def test_registry_holds_both_flavors() -> None:
    reg = stage_registry()
    assert reg["fsk"].engine == "gr"
    assert reg["sync_word"].engine == "coding"
    assert reg["css_explicit_decode"].engine == "coding"


def test_m_slice_out_descriptor_hardens() -> None:
    d = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
    out = MSlice().out_descriptor(d, {"thresholds": [0.0], "levels": [0, 1]})
    assert out.item_type == "s" and out.carrier is Carrier.HARD


def test_css_explicit_decode_out_descriptor_lands_on_hard_bits() -> None:
    d = Descriptor(Level.SYMBOLS, "s", carrier=Carrier.HARD)
    out = CssExplicitDecode().out_descriptor(d, _EXPLICIT_PARAMS)
    assert out == Descriptor(Level.BITS, "b", carrier=Carrier.HARD)


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


# --- css_explicit_decode_rx (ported from tests/bits/test_css_explicit_decode.py) ---


def test_explicit_decode_yields_rf_fingerpring_payload() -> None:
    bits = _run(_SF11_SYMBOLS)
    payload = _assemble(list(bits))
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]


def test_explicit_decode_loops_back_to_back_frames() -> None:
    """Two frames in one symbol stream decode as two frames — the pre-fix
    block set _done after frame 1 and consumed-and-discarded the rest
    (issue 03)."""
    one = _run(_SF11_SYMBOLS)
    two = _run(_SF11_SYMBOLS * 2)
    assert len(two) == 2 * len(one)
    assert _assemble(list(two[: len(one)])) == _assemble(list(one))
    assert _assemble(list(two[len(one) :])) == _assemble(list(one))


def test_explicit_decode_full_rate_payload_lane() -> None:
    """reduced=False routes the payload demap through the divisor=1/full-offset
    lane — the common non-LDRO config (SF7-9); the off-air fixtures are all
    LDRO, so only the reduced lane had coverage."""
    sf, cr, payload_len = _HEADER["sf"], 1, 16
    # 3 interleave blocks of sf nibbles: ceil((32-4)/11) — a count that differs
    # from the reduced-denominator ceil(28/9)=4, pinning the non-LDRO frame
    # algebra too
    nibbles = [(3 * i + 1) % 16 for i in range(3 * sf)]
    syms = _encode_header(payload_len, cr, 0) + _encode_payload(nibbles, cr, sf)
    bits = _run(syms, params={**_HEADER, "reduced": False})
    expected = [0] * 16  # the header block's carry nibbles are zeroed
    for nib in nibbles:
        expected.extend((nib >> (3 - j)) & 1 for j in range(4))
    assert list(bits) == expected[: 8 * payload_len]


def test_explicit_decode_oversize_length_does_not_swallow_later_marks() -> None:
    """A parity-valid header whose declared frame overruns the symbol array
    must drop only its own mark — the pre-fix carve broke out of the marks
    loop there, swallowing every later real burst (the module docstring's
    'a corrupt length can never swallow a real burst')."""
    one = _run(_SF11_SYMBOLS)
    # fixture guard: the synthetic header must be parity-valid and decode,
    # else the oversize case below would pass via the corrupt-header skip
    # even without the fix (first 2 bytes differ: carry nibbles are zeroed)
    swapped = _run(_encode_header(255, 1, 1) + _SF11_SYMBOLS[8:], marks=(0,))
    assert _assemble(list(swapped))[2:] == _assemble(list(one))[2:]

    oversize = _encode_header(255, 4, 1)
    bits = _run(oversize + [0] * 60 + _SF11_SYMBOLS, marks=(0, len(oversize) + 60))
    assert list(bits) == list(one)


def test_explicit_decode_corrupt_header_skips_to_next_mark() -> None:
    """A corrupt header is skipped and the next marked burst still decodes —
    the pre-fix block treated a header-parity failure as end-of-stream and
    reported nothing (issue 03)."""
    rng = np.random.default_rng(5)
    corrupt = list(_SF11_SYMBOLS)
    corrupt[:8] = [int(v) for v in rng.integers(1, 2048, 8)]
    frame_len = len(_SF11_SYMBOLS)
    one = _run(_SF11_SYMBOLS)
    bits = _run(corrupt + _SF11_SYMBOLS, marks=(0, frame_len))
    payload = _assemble(list(bits))
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]
    assert list(bits) == list(one)


# --- CssExplicitDecode stage (ported from tests/bits/test_css_explicit_stage.py) ---


def test_css_explicit_decode_stage_levels_and_emit() -> None:
    s = CssExplicitDecode()
    assert (s.from_level, s.to_level) == (Level.SYMBOLS, Level.BITS)
    b = CodingBuilder()
    s.emit_rx(b, _EXPLICIT_PARAMS)
    assert len(b.steps) == 1


def test_css_explicit_decode_stage_is_rx_only() -> None:
    assert CssExplicitDecode().directions == frozenset({"rx"})
    with pytest.raises(StageDirectionError):
        CssExplicitDecode().emit_tx(CodingBuilder(), _EXPLICIT_PARAMS)


def test_css_explicit_decode_stage_in_registry() -> None:
    assert "css_explicit_decode" in stage_registry()


def test_explicit_params_valid_sf11_cr4() -> None:
    p = _ExplicitParams.model_validate(_EXPLICIT_PARAMS)
    assert p.sf == 11 and p.header_cr == 4


def test_explicit_params_sf_too_low_raises() -> None:
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate({**_EXPLICIT_PARAMS, "sf": 4})


def test_explicit_params_sf_too_high_raises() -> None:
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate({**_EXPLICIT_PARAMS, "sf": 15})


def test_explicit_params_header_cr_absent_from_parity_table_raises() -> None:
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate({**_EXPLICIT_PARAMS, "header_cr": 5})


def test_explicit_params_reject_sf_reduction_ge_sf_when_reduced() -> None:
    with pytest.raises(ValidationError):
        _ExplicitParams.model_validate(
            {**_EXPLICIT_PARAMS, "sf": 7, "reduced": True, "sf_reduction": 7}
        )
