"""LoRa's CSS explicit-header parameters — the protocol constants that used to
live in production before the agnosticism purge. Every value here is
a datasheet/gr-lora fact; production takes them as caller-supplied parameters.

- Hamming(4+cr, 4): data_bits=4, so a byte is 2 nibbles and the CRC is 2 bytes.
- Parity masks per code rate 1..4 (each mask = a 4-bit parity row; cr rows for
  the code with cr parity bits), laid out low-cr-first for coding.parity_for_cr.
- Reduced-rate windows divide the symbol by 2^sf_reduction with no bin offset;
  full-rate windows subtract one bin (Gray offset).
- Header: 8 symbols, 5 header nibbles, reduced by 2, 12 data bits, checksum
  masks. Fields carry (start, length) bit-positions inside the header nibbles.
- SFD is 2.25 down-chirps; 2 sync (netid) symbols sit between preamble and SFD.
"""

from __future__ import annotations

from typing import Any

# Hamming parity masks: cr -> rows, flattened low-cr-first.
_PARITY = {
    1: [15],
    2: [7, 14],
    3: [11, 13, 14],
    4: [7, 14, 11, 13],
}
PARITY_MASKS = [m for cr in sorted(_PARITY) for m in _PARITY[cr]]

# css_explicit_decode / CssExplicitDecode geometry.
HEADER: dict[str, Any] = {
    "sf": 11,
    "header_cr": 4,
    "reduced": True,
    "header_symbols": 8,
    "header_nibbles": 5,
    "sf_reduction": 2,
    "header_data_bits": 12,
    "header_parity": [3840, 2273, 1178, 599, 303],
    "field_payload_len": [0, 8],
    "field_cr": [8, 3],
    "field_has_crc": [11, 1],
    "field_parity": [15, 5],
    "data_bits": 4,
    "crc_bytes": 2,
    "parity_masks": PARITY_MASKS,
    "reduced_offset": 0,
    "full_offset": 1,
}

# chirp_sync / ChirpSync acquisition shape.
SFD_SYMBOLS = 2.25
SYNC_SYMBOLS = 2
