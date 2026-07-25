"""Behavior of the test-side CSS explicit-header orchestrator (demoted from a
registered stage 2026-07; tests ported with it). The synthesis funnel lives in
tests/phy/coding/_css_synth.py; the SF11 fixtures are real off-air symbols."""

from __future__ import annotations

import numpy as np
from engine._css_lora import HEADER as _HEADER
from engine.coding._css_synth import SF11_SYMBOLS as _SF11_SYMBOLS
from engine.coding._css_synth import assemble as _assemble
from engine.coding._css_synth import encode_header as _encode_header
from engine.coding._css_synth import encode_payload as _encode_payload
from engine.coding._css_synth import run as _run


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
