"""The composition proof for css_explicit_decode (2026-07 review): the stage
is an orchestration fast-path over the generic vocabulary, not a protocol
smuggled behind 18 parameters. An agent-style two-pass decode — a static
header codec, then a payload codec constructed from the decoded header
fields — reproduces the orchestrator bit-exactly using only generic pieces:

  demap_symbols            gray demap over a bin window     core primitive
  diag_deinterleave_perm   one static block permutation     core primitive
  block_code_rx            block-FEC decode                 the generic stage
  parse_rx                 header fields                    the generic stage
  header_parity_ok         mask-parity gate                 core primitive
  css_explicit_frame_len   frame-extent algebra             core primitive

The header->payload dependency (the payload's code rate and extent live in
the header) is exactly the agent loop, so all css_explicit_decode adds is
doing both passes in one call. Stage-level gaps if this were to run
in-engine: a SYMBOLS->BITS demap stage and a block-permutation stage."""

from __future__ import annotations

import numpy as np
from bits._css_synth import (
    HEADER,
    SF11_SYMBOLS,
    assemble,
    encode_header,
    encode_payload,
    run,
)

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, _Frame
from marconi.core import coding


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
    (the diagonal perm emits codewords MSB-first, the block_code stage reads
    LSB-first strides), then the generic stage decodes; nibble bits return
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
    decoded = framing.block_code_rx(
        RxCarrier(bits=np.asarray(stream, np.uint8)),
        code_bits=cw_len,
        data_bits=d,
        parity_masks=coding.parity_for_cr(p["parity_masks"], cr),
    ).bits
    return decoded.reshape(-1, d)[:, ::-1].reshape(-1)


def _parse_stage_header(hbits, p):
    """The generic parse stage over the packed header bits; the field schema
    derives from the same (start, len) spans the orchestrator takes."""
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
    fields = framing.parse_fields(raw)
    packed = framing.bits_to_bytes(
        np.asarray(list(hbits) + [0] * ((-len(hbits)) % 8), np.uint8)
    )
    out = framing.parse_rx(
        RxCarrier(
            bits=np.zeros(0, np.uint8),
            frames=[_Frame(start=0, cursor=0, payload=packed, crc_ok=True)],
        ),
        struct=framing.build_struct(fields),
        fields=fields,
        bit_order="msb",
    )
    msg = out.frames[0].message
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
    composed = _two_pass_decode(SF11_SYMBOLS, HEADER)
    assert np.array_equal(composed, run(SF11_SYMBOLS))
    payload = assemble(list(composed))
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]


def test_composition_matches_orchestrator_non_ldro():
    p = {**HEADER, "reduced": False}
    nibbles = [(3 * i + 1) % 16 for i in range(3 * HEADER["sf"])]
    syms = encode_header(16, 1, 0) + encode_payload(nibbles, 1, HEADER["sf"])
    composed = _two_pass_decode(syms, p)
    assert np.array_equal(composed, run(syms, params=p))
    assert composed.size == 16 * 8


def test_composition_matches_orchestrator_reduced_correcting_cr():
    """cr=4 payload: the correcting-FEC path (can_correct holds) through the
    reduced payload lane."""
    sf_app = HEADER["sf"] - HEADER["sf_reduction"]
    nibbles = [(5 * i + 2) % 16 for i in range(4 * sf_app)]
    syms = encode_header(16, 4, 0) + encode_payload(nibbles, 4, sf_app)
    composed = _two_pass_decode(syms, HEADER)
    assert np.array_equal(composed, run(syms))
    expected = [0] * 16
    for nib in nibbles:
        expected.extend((nib >> (3 - j)) & 1 for j in range(4))
    assert list(composed) == expected[: 16 * 8]
