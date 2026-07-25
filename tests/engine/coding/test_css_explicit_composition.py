"""The composition proof for the css_explicit orchestrator (2026-07 review;
demoted from a registered stage to tests/helpers/css_explicit.py on its
strength): a fast-path over the generic vocabulary, not a protocol smuggled
behind 18 parameters. An agent-style two-pass decode -- a static
header codec, then a payload codec constructed from the decoded header
fields -- reproduces the orchestrator bit-exactly using only generic pieces:

  demap_symbols            gray demap over a bin window     core primitive
  diag_deinterleave_perm   one static block permutation     core primitive
  block_code_rx            block-FEC decode                 the generic stage
  parse_message            header fields                    the helpers oracle
  header_parity_ok         mask-parity gate                 core primitive
  css_explicit_frame_len   frame-extent algebra             core primitive

The header->payload dependency (the payload's code rate and extent live in
the header) is exactly the agent loop, so all the orchestrator adds is
doing both passes in one call. Gaps if this were ever to run in-engine:
a SYMBOLS->BITS demap stage and a block-permutation stage.
"""

from __future__ import annotations

import numpy as np
from engine._css_lora import HEADER as _HEADER
from engine.coding._css_synth import SF11_SYMBOLS as _SF11_SYMBOLS
from engine.coding._css_synth import assemble as _assemble
from engine.coding._css_synth import encode_header as _encode_header
from engine.coding._css_synth import encode_payload as _encode_payload
from engine.coding._css_synth import run as _run
from helpers import bitops, parse

from marconi.engine.coding import ops_bits
from marconi.engine.coding import primitives as coding
from marconi.engine.coding.carrier import CodingCarrier


def _demap(syms, sf_app, p):
    n = 1 << p["sf"]
    if sf_app < p["sf"]:
        return coding.demap_symbols(
            [int(s) for s in syms],
            sf_app,
            n=n,
            divisor=1 << p["sf_reduction"],
            offset=p["reduced_offset"],
        )
    return coding.demap_symbols(
        [int(s) for s in syms], sf_app, n=n, divisor=1, offset=p["full_offset"]
    )


def _block_code_stage(bits, sf_app, cr, p):
    """Deinterleave and the codeword-basis bridge fold into ONE static gather
    (the diagonal perm emits codewords MSB-first, the block_code op reads
    LSB-first strides), then the generic op decodes; nibble bits return
    MSB-first to match the emitted stream order."""
    d = p["data_bits"]
    cw_len = cr + d
    perm = coding.diag_deinterleave_perm(sf_app, cw_len)
    lsb_perm = [
        perm[(i // cw_len) * cw_len + (cw_len - 1 - i % cw_len)]
        for i in range(len(perm))
    ]
    block = sf_app * cw_len
    stream: list[int] = []
    for b in range(len(bits) // block):
        chunk = bits[b * block : (b + 1) * block]
        stream.extend(chunk[q] for q in lsb_perm)
    decoded = ops_bits.block_code_rx(
        CodingCarrier(bits=np.asarray(stream, np.uint8)),
        code_bits=cw_len,
        data_bits=d,
        parity_masks=coding.parity_for_cr(p["parity_masks"], cr),
    ).bits
    return decoded.reshape(-1, d)[:, ::-1].reshape(-1)


def _parse_stage_header(hbits, p):
    """helpers.parse.parse_message over the packed header bits; the field
    schema derives from the same (start, len) spans the orchestrator takes.
    A '_pad'-prefixed field name is dropped from the result, exactly as the
    old bits-engine parse stage dropped underscore-prefixed construct keys."""
    spans = {
        "payload_len": p["field_payload_len"],
        "cr": p["field_cr"],
        "has_crc": p["field_has_crc"],
        "parity": p["field_parity"],
    }
    raw: list[dict] = []
    pos = 0
    for name, (start, length) in sorted(spans.items(), key=lambda kv: kv[1][0]):
        if start > pos:
            raw.append({"name": f"_pad{pos}", "bits": start - pos})
        raw.append({"name": name, "bits": length})
        pos = start + length
    packed = bitops.bits_to_bytes(
        np.asarray(list(hbits) + [0] * ((-len(hbits)) % 8), np.uint8)
    )
    msg = parse.parse_message(packed, raw, bit_order="msb")
    assert msg is not None
    return msg


def _two_pass_decode(syms, p):
    d, hn, hs = p["data_bits"], p["header_nibbles"], p["header_symbols"]
    sf_app_hdr = p["sf"] - p["sf_reduction"]
    hdr_stream = _block_code_stage(
        _demap(syms[:hs], sf_app_hdr, p), sf_app_hdr, p["header_cr"], p
    )
    hbits = hdr_stream[: hn * d]
    carry = hdr_stream[hn * d :]  # the header block's spare nibbles lead the payload
    msg = _parse_stage_header(hbits, p)
    data_int = coding.bits_to_uint([int(b) for b in hbits], 0, p["header_data_bits"])
    assert coding.header_parity_ok(data_int, int(msg["parity"]), p["header_parity"])
    payload_len, cr, has_crc = (
        int(msg["payload_len"]),
        int(msg["cr"]),
        int(msg["has_crc"]),
    )
    assert coding.supported_cr(p["parity_masks"], cr)
    frame_len = coding.css_explicit_frame_len(
        payload_len,
        has_crc,
        cr,
        p["sf"],
        int(p["reduced"]),
        p["sf_reduction"],
        data_bits=d,
        header_nibbles=hn,
        crc_bytes=p["crc_bytes"],
    )
    sf_app_pl = p["sf"] - p["sf_reduction"] if p["reduced"] else p["sf"]
    body = _block_code_stage(
        _demap(syms[hs : hs + frame_len], sf_app_pl, p), sf_app_pl, cr, p
    )
    declared = (payload_len + has_crc * p["crc_bytes"]) * 8
    return np.concatenate([carry, body])[:declared]


def test_composition_matches_orchestrator_on_offair_frame():
    composed = _two_pass_decode(_SF11_SYMBOLS, _HEADER)
    assert np.array_equal(composed, _run(_SF11_SYMBOLS))
    payload = _assemble(list(composed))
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]


def test_composition_matches_orchestrator_non_ldro():
    p = {**_HEADER, "reduced": False}
    nibbles = [(3 * i + 1) % 16 for i in range(3 * _HEADER["sf"])]
    syms = _encode_header(16, 1, 0) + _encode_payload(nibbles, 1, _HEADER["sf"])
    composed = _two_pass_decode(syms, p)
    assert np.array_equal(composed, _run(syms, params=p))
    assert composed.size == 16 * 8


def test_composition_matches_orchestrator_reduced_correcting_cr():
    """cr=4 payload: the correcting-FEC path (can_correct holds) through the
    reduced payload lane."""
    sf_app = _HEADER["sf"] - _HEADER["sf_reduction"]
    nibbles = [(5 * i + 2) % 16 for i in range(4 * sf_app)]
    syms = _encode_header(16, 4, 0) + _encode_payload(nibbles, 4, sf_app)
    composed = _two_pass_decode(syms, _HEADER)
    assert np.array_equal(composed, _run(syms))
    expected = [0] * 16
    for nib in nibbles:
        expected.extend((nib >> (3 - j)) & 1 for j in range(4))
    assert list(composed) == expected[: 16 * 8]
