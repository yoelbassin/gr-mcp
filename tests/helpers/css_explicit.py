"""CSS explicit-header orchestration — datasheet work, deliberately outside
the product. Parses the protocol-declared header at each burst mark, then
decodes the payload at its declared code rate through helpers.blockmath's
scalar oracles (demap / diagonal deinterleave / block FEC / frame-extent
algebra). Marked mode parses headers only at burst-start marks so inter-burst
junk is never mistaken for a header, a corrupt header skips to the next mark,
and a frame whose declared extent crosses the next mark is cut there so a
corrupt length can never swallow a real burst. Blind mode assumes back-to-back
frames from 0 and stops at the first header failure.

Demoted from a registered stage 2026-07: the composition proof
(tests/unit/engine/coding/test_css_explicit_composition.py) shows this adds no
vocabulary the product lane doesn't already have."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from helpers import blockmath as coding


def css_explicit_decode(
    symbols: Sequence[int] | np.ndarray,
    marks: Sequence[int],
    params: Mapping[str, Any],
) -> np.ndarray:
    sf = int(params["sf"])
    header_cr = int(params["header_cr"])
    reduced = bool(params["reduced"])
    header_symbols = int(params["header_symbols"])
    header_nibbles = int(params["header_nibbles"])
    sf_reduction = int(params["sf_reduction"])
    header_data_bits = int(params["header_data_bits"])
    data_bits = int(params["data_bits"])
    crc_bytes = int(params["crc_bytes"])
    reduced_offset = int(params.get("reduced_offset", 0))
    full_offset = int(params.get("full_offset", 0))
    masks = [int(x) for x in params["header_parity"]]
    fec_masks = [int(x) for x in params["parity_masks"]]
    pl_start, pl_len = (int(x) for x in params["field_payload_len"])
    cr_start, cr_len = (int(x) for x in params["field_cr"])
    crc_start, crc_len = (int(x) for x in params["field_has_crc"])
    par_start, par_len = (int(x) for x in params["field_parity"])

    syms = [int(s) for s in symbols]
    n = 1 << sf
    header_sf_app = sf - sf_reduction
    payload_sf_app = sf - sf_reduction if reduced else sf

    def _demap(sym_slice: list[int], sf_app: int) -> list[int]:
        if sf_app < sf:
            return coding.demap_symbols(
                sym_slice, sf_app, n=n, divisor=1 << sf_reduction, offset=reduced_offset
            )
        return coding.demap_symbols(
            sym_slice, sf_app, n=n, divisor=1, offset=full_offset
        )

    def _fec_nibbles(sym_slice: list[int], sf_app: int, cr: int) -> list[int]:
        cw_len = cr + data_bits
        correct = coding.can_correct(cr, data_bits)
        parity = coding.parity_for_cr(fec_masks, cr)
        bits = _demap(sym_slice, sf_app)
        perm = coding.diag_deinterleave_perm(sf_app, cw_len)
        block = sf_app * cw_len
        nibbles: list[int] = []
        for b in range(len(bits) // block):
            chunk = bits[b * block : (b + 1) * block]
            deint = [chunk[p] for p in perm]
            for i in range(len(deint) // cw_len):
                word = 0
                for bit in deint[i * cw_len : (i + 1) * cw_len]:
                    word = (word << 1) | bit
                nibbles.append(
                    coding.block_fec_decode(word, parity, data_bits, correct)
                )
        return nibbles

    def _nibbles_to_bits(nibbles: list[int]) -> list[int]:
        out: list[int] = []
        for nib in nibbles:
            out.extend((nib >> (data_bits - 1 - j)) & 1 for j in range(data_bits))
        return out

    def _parse_header(start: int) -> tuple[int, int, int, list[int]] | None:
        nibbles = _fec_nibbles(
            syms[start : start + header_symbols], header_sf_app, header_cr
        )
        hbits = _nibbles_to_bits(nibbles[:header_nibbles])
        data_int = coding.bits_to_uint(hbits, 0, header_data_bits)
        received = coding.bits_to_uint(hbits, par_start, par_len)
        payload_cr = coding.bits_to_uint(hbits, cr_start, cr_len)
        if not coding.header_parity_ok(data_int, received, masks) or not (
            coding.supported_cr(fec_masks, payload_cr)
        ):
            return None
        payload_len = coding.bits_to_uint(hbits, pl_start, pl_len)
        has_crc = coding.bits_to_uint(hbits, crc_start, crc_len)
        frame_len = coding.css_explicit_frame_len(
            payload_len,
            has_crc,
            payload_cr,
            sf,
            int(reduced),
            sf_reduction,
            data_bits=data_bits,
            header_nibbles=header_nibbles,
            crc_bytes=crc_bytes,
        )
        declared = (payload_len + has_crc * crc_bytes) * 8
        carry = _nibbles_to_bits(nibbles[header_nibbles:])
        return frame_len, payload_cr, declared, carry

    def _frame_bits(
        start: int, frame_len: int, payload_cr: int, declared: int, carry: list[int]
    ) -> list[int]:
        payload = syms[start + header_symbols : start + header_symbols + frame_len]
        bits = carry + _nibbles_to_bits(
            _fec_nibbles(payload, payload_sf_app, payload_cr)
        )
        # emit exactly the header-declared frame content; the last FEC block's
        # rounding pad would misalign every following frame in the bit stream
        return bits[:declared]

    out: list[int] = []
    mark_list = [int(m) for m in marks]
    if mark_list:
        pos = 0
        for i, start in enumerate(mark_list):
            if start < pos or start + header_symbols > len(syms):
                continue
            hdr = _parse_header(start)
            if hdr is None:
                continue
            frame_len, payload_cr, declared, carry = hdr
            need = start + header_symbols + frame_len
            if need > len(syms):
                continue
            out.extend(_frame_bits(start, frame_len, payload_cr, declared, carry))
            nxt = next((m for m in mark_list[i + 1 :] if m > start), None)
            pos = need if nxt is None or nxt >= need else nxt
    else:
        pos = 0
        while pos + header_symbols <= len(syms):
            hdr = _parse_header(pos)
            if hdr is None:
                break
            frame_len, payload_cr, declared, carry = hdr
            need = pos + header_symbols + frame_len
            if need > len(syms):
                break
            out.extend(_frame_bits(pos, frame_len, payload_cr, declared, carry))
            pos = need
    return np.asarray(out, dtype=np.uint8)
