"""Symbol-level explicit-header CSS decoder (embedded GR block). Reads dechirped
int16 symbols, parses the explicit header from the first 8 symbols, decodes the
payload at its own code rate, splices the SF>7 carry nibbles, and emits the de-FEC'd
payload bits (uint8, one per byte). Generic over CSS explicit-header frames — the
header field layout and code rates are parameters; LoRa is one parameter set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.modulation.css import coding


def make_css_explicit_decode(
    gr: Any,
    *,
    sf: int,
    header_cr: int,
    ldro: bool,
    header_symbols: int,
    header_nibbles: int,
    sf_reduction: int,
    header_data_bits: int,
    header_parity: list,
    field_payload_len: list,
    field_cr: list,
    field_has_crc: list,
    field_parity: list,
) -> Any:
    n = 1 << sf
    header_sf_app = sf - sf_reduction
    payload_sf_app = sf - sf_reduction if ldro else sf
    masks = [int(x) for x in header_parity]
    pl_start, pl_len = (int(x) for x in field_payload_len)
    cr_start, cr_len = (int(x) for x in field_cr)
    crc_start, crc_len = (int(x) for x in field_has_crc)
    par_start, par_len = (int(x) for x in field_parity)

    def _demap(symbols: list[int], sf_app: int) -> list[int]:
        if sf_app < sf:  # reduced-rate (header / LDRO payload): value in s // 4
            return coding.demap_symbols(symbols, sf_app, n=n, divisor=4, offset=0)
        return coding.demap_symbols(symbols, sf_app, n=n, divisor=1, offset=1)

    def _fec_nibbles(symbols: list[int], sf_app: int, cr: int) -> list[int]:
        cw_len = cr + 4
        correct = cr >= 3
        parity = coding.HAMMING_PARITY[cr]
        bits = _demap(symbols, sf_app)
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
                nibbles.append(coding.block_fec_decode(word, parity, correct))
        return nibbles

    def _nibbles_to_bits(nibbles: list[int]) -> list[int]:
        out: list[int] = []
        for nib in nibbles:
            out.extend((nib >> (3 - j)) & 1 for j in range(4))
        return out

    class _CssExplicitDecode(gr.basic_block):
        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="css_explicit_decode", in_sig=[np.int16], out_sig=[np.uint8]
            )
            self._symbols: list[int] = []
            self._frame_len: int | None = None
            self._payload_cr: int = 0
            self._carry: list[int] = []
            self._out: list[int] | None = None
            self._done = False

        def forecast(self, noutput_items: int, ninputs: int) -> list:
            return [1] * ninputs

        def _parse_header(self) -> None:
            nibbles = _fec_nibbles(
                self._symbols[:header_symbols], header_sf_app, header_cr
            )
            hbits = _nibbles_to_bits(nibbles[:header_nibbles])
            data_int = coding.bits_to_uint(hbits, 0, header_data_bits)
            received = coding.bits_to_uint(hbits, par_start, par_len)
            if not coding.header_parity_ok(data_int, received, masks):
                self._done = True
                return
            payload_len = coding.bits_to_uint(hbits, pl_start, pl_len)
            payload_cr = coding.bits_to_uint(hbits, cr_start, cr_len)
            has_crc = coding.bits_to_uint(hbits, crc_start, crc_len)
            self._frame_len = coding.css_explicit_frame_len(
                payload_len, has_crc, payload_cr, sf, int(ldro), sf_reduction
            )
            self._payload_cr = payload_cr
            self._carry = _nibbles_to_bits(nibbles[header_nibbles:])

        def _decode_frame(self) -> None:
            assert self._frame_len is not None
            payload = self._symbols[header_symbols : header_symbols + self._frame_len]
            nibbles = _fec_nibbles(payload, payload_sf_app, self._payload_cr)
            self._out = self._carry + _nibbles_to_bits(nibbles)
            self._done = True

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            out = output_items[0]
            if not self._done and self._out is None:
                self._symbols.extend(int(s) for s in inp)
                self.consume(0, len(inp))
                if self._frame_len is None and len(self._symbols) >= header_symbols:
                    self._parse_header()
                if (
                    self._frame_len is not None
                    and self._out is None
                    and len(self._symbols) >= header_symbols + self._frame_len
                ):
                    self._decode_frame()
                if not self._out:
                    return 0
            if self._out:
                emit = min(len(self._out), len(out))
                out[:emit] = self._out[:emit]
                self._out = self._out[emit:]
                return emit
            self.consume(0, len(inp))
            return 0

    return _CssExplicitDecode()
