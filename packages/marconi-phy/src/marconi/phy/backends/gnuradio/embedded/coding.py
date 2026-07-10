from __future__ import annotations

from typing import Any

import numpy as np

from marconi.phy.backends.gnuradio.embedded.lifecycle import OutQueue, forecast_drain
from marconi.phy.modulation.css import coding


def make_css_explicit_decode(
    gr: Any,
    *,
    sf: int,
    header_cr: int,
    reduced: bool,
    header_symbols: int,
    header_nibbles: int,
    sf_reduction: int,
    header_data_bits: int,
    header_parity: list,
    field_payload_len: list,
    field_cr: list,
    field_has_crc: list,
    field_parity: list,
    data_bits: int,
    crc_bytes: int,
    parity_masks: list,
    reduced_offset: int,
    full_offset: int,
) -> Any:
    pmt = gr.pmt
    n = 1 << sf
    header_sf_app = sf - sf_reduction
    payload_sf_app = sf - sf_reduction if reduced else sf
    masks = [int(x) for x in header_parity]
    fec_masks = [int(x) for x in parity_masks]
    pl_start, pl_len = (int(x) for x in field_payload_len)
    cr_start, cr_len = (int(x) for x in field_cr)
    crc_start, crc_len = (int(x) for x in field_has_crc)
    par_start, par_len = (int(x) for x in field_parity)

    def _demap(symbols: list[int], sf_app: int) -> list[int]:
        if sf_app < sf:  # reduced-rate window: symbol value carries sf_app units
            return coding.demap_symbols(
                symbols, sf_app, n=n, divisor=1 << sf_reduction, offset=reduced_offset
            )
        return coding.demap_symbols(symbols, sf_app, n=n, divisor=1, offset=full_offset)

    def _fec_nibbles(symbols: list[int], sf_app: int, cr: int) -> list[int]:
        cw_len = cr + data_bits
        correct = coding.can_correct(cr, data_bits)
        parity = coding.parity_for_cr(fec_masks, cr)
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
                nibbles.append(
                    coding.block_fec_decode(word, parity, data_bits, correct)
                )
        return nibbles

    def _nibbles_to_bits(nibbles: list[int]) -> list[int]:
        out: list[int] = []
        for nib in nibbles:
            out.extend((nib >> (data_bits - 1 - j)) & 1 for j in range(data_bits))
        return out

    class _CssExplicitDecode(gr.basic_block):
        """Decodes every frame in the stream. Without burst tags, frames are
        assumed back-to-back (a header parse follows the previous frame's end;
        a header failure stalls until a tag arrives). With upstream "burst"
        tags (chirp_sync -> dechirp), headers parse only at tagged symbol
        offsets, so inter-burst junk is never mistaken for a header and a
        corrupt header just skips to the next burst. Per-run counters are
        exposed as `diagnostics` and surfaced through RunResult."""

        def __init__(self) -> None:
            gr.basic_block.__init__(
                self, name="css_explicit_decode", in_sig=[np.int16], out_sig=[np.uint8]
            )
            self.set_tag_propagation_policy(gr.TPP_DONT)
            self._symbols: list[int] = []
            self._abs0 = 0  # absolute symbol index of _symbols[0]
            self._start = 0  # absolute symbol index of the current frame candidate
            self._tags: list[int] = []
            self._tagged = False
            self._stalled = False
            self._frame_len: int | None = None
            self._payload_cr: int = 0
            self._declared_bits = 0
            self._carry: list[int] = []
            self._out = OutQueue(np.uint8)
            self.diagnostics = {
                "frames_seen": 0,
                "header_ok": 0,
                "header_fail": 0,
                "frames_decoded": 0,
            }

        def forecast(self, noutput_items: int, ninputs: int) -> list:
            return forecast_drain(self._out.pending, ninputs)

        def _sym(self, lo: int, hi: int) -> list[int]:
            return self._symbols[lo - self._abs0 : hi - self._abs0]

        def _avail(self) -> int:
            return self._abs0 + len(self._symbols)

        def _next_start(self) -> int | None:
            if not self._tagged:
                return None if self._stalled else self._start
            while self._tags and self._tags[0] < self._start:
                self._tags.pop(0)
            return self._tags.pop(0) if self._tags else None

        def _parse_header(self, start: int) -> bool:
            nibbles = _fec_nibbles(
                self._sym(start, start + header_symbols), header_sf_app, header_cr
            )
            hbits = _nibbles_to_bits(nibbles[:header_nibbles])
            data_int = coding.bits_to_uint(hbits, 0, header_data_bits)
            received = coding.bits_to_uint(hbits, par_start, par_len)
            self.diagnostics["frames_seen"] += 1
            payload_cr = coding.bits_to_uint(hbits, cr_start, cr_len)
            if not coding.header_parity_ok(data_int, received, masks) or (
                not coding.supported_cr(fec_masks, payload_cr)
            ):
                self.diagnostics["header_fail"] += 1
                return False
            self.diagnostics["header_ok"] += 1
            payload_len = coding.bits_to_uint(hbits, pl_start, pl_len)
            has_crc = coding.bits_to_uint(hbits, crc_start, crc_len)
            self._frame_len = coding.css_explicit_frame_len(
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
            self._payload_cr = payload_cr
            self._declared_bits = (payload_len + has_crc * crc_bytes) * 8
            self._carry = _nibbles_to_bits(nibbles[header_nibbles:])
            return True

        def _decode_frame(self, start: int, frame_len: int) -> None:
            payload = self._sym(
                start + header_symbols, start + header_symbols + frame_len
            )
            nibbles = _fec_nibbles(payload, payload_sf_app, self._payload_cr)
            bits = self._carry + _nibbles_to_bits(nibbles)
            # emit exactly the header-declared frame content; the last FEC
            # block's rounding pad would misalign every following frame in
            # the bit stream (a single-frame stream never shows this)
            self._out.push(np.asarray(bits[: self._declared_bits], dtype=np.uint8))
            self.diagnostics["frames_decoded"] += 1

        def _advance(self, to: int) -> None:
            self._start = to
            self._frame_len = None
            drop = min(to, self._avail()) - self._abs0
            if drop > 0:
                self._symbols = self._symbols[drop:]
                self._abs0 += drop

        def _process(self) -> None:
            while True:
                if self._frame_len is None:
                    start = self._next_start()
                    if start is None or self._avail() < start + header_symbols:
                        if start is not None and self._tagged:
                            self._tags.insert(0, start)  # not enough symbols yet
                        return
                    self._start = start
                    if not self._parse_header(start):
                        if self._tagged:
                            continue  # hunt the next tagged burst
                        self._stalled = True  # blind stream: no way to resync
                        return
                frame_len = self._frame_len
                if frame_len is None:  # narrow: parse either set it or returned
                    return
                need = self._start + header_symbols + frame_len
                if self._avail() < need:
                    return
                self._decode_frame(self._start, frame_len)
                # a frame whose declared extent crosses the next burst tag is
                # cut there so a corrupt length can never swallow a real burst
                nxt = self._tags[0] if self._tags else None
                self._advance(need if nxt is None or nxt >= need else nxt)

        def general_work(self, input_items: Any, output_items: Any) -> int:
            inp = input_items[0]
            if len(inp):
                base = self.nitems_read(0)
                for t in self.get_tags_in_window(0, 0, len(inp)):
                    if pmt.symbol_to_string(t.key) == "burst":
                        self._tags.append(int(t.offset))
                        self._tagged = True
                        self._stalled = False
                if not self._symbols:
                    self._abs0 = base
                self._symbols.extend(int(s) for s in inp)
                self.consume(0, len(inp))
                self._process()
            return self._out.drain(output_items[0])

    return _CssExplicitDecode()
