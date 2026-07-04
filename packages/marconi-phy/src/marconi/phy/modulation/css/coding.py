"""Generic channel-coding primitives for chirp-spread-spectrum symbol streams —
pure numpy/Python, no GNU Radio. Gray mapping, the diagonal interleave
permutation, and a systematic single-error block-FEC decoder. Every value that
belongs to a specific protocol (parity rows, data width, frame geometry) is
caller-supplied; nothing here encodes one."""

from __future__ import annotations

from math import ceil

from marconi.core.coding import (
    bits_to_uint,
    block_fec_decode,
    can_correct,
    gray_decode,
    gray_encode,
    header_parity_ok,
    parity_for_cr,
    parity_rows,
    supported_cr,
)

__all__ = [
    "bits_to_uint",
    "block_fec_decode",
    "can_correct",
    "gray_decode",
    "gray_encode",
    "header_parity_ok",
    "parity_for_cr",
    "parity_rows",
    "supported_cr",
    "demap_symbols",
    "diag_deinterleave_perm",
    "css_explicit_frame_len",
]


def demap_symbols(
    symbols: list[int], sf_app: int, *, n: int, divisor: int, offset: int
) -> list[int]:
    bits: list[int] = []
    for s in symbols:
        g = gray_encode(((s % n) // divisor - offset) % n)
        bits.extend((g >> (sf_app - 1 - j)) & 1 for j in range(sf_app))
    return bits


def diag_deinterleave_perm(sf_app: int, cw_len: int) -> list[int]:
    """Flat gather permutation P with output[i] = input[P[i]] for symbols packed
    MSB-first -> codewords packed MSB-first (a diagonal de-interleaver)."""
    perm: list[int] = []
    for oc in range(sf_app):
        for k in range(cw_len):
            perm.append(
                (cw_len - 1 - k) * sf_app + ((cw_len - 1 - k) - oc - 1) % sf_app
            )
    return perm


def css_explicit_frame_len(
    payload_len: int,
    has_crc: int,
    cr: int,
    sf: int,
    reduced: int,
    sf_reduction: int,
    *,
    data_bits: int,
    header_nibbles: int,
    crc_bytes: int,
) -> int:
    """Payload-symbol count of a CSS explicit-header frame. A frame carries
    ``payload_len`` bytes plus an optional ``crc_bytes``-byte checksum as
    ``data_bits``-wide units; the header block already emits its spare units
    into the payload stream (header_carried); each payload symbol carries
    ``sf - sf_reduction*reduced`` units, and each interleave block spans
    ``cr + data_bits`` symbols. No value here is protocol-specific — the caller
    supplies the code's data width, checksum size, and header geometry."""
    units_per_byte = 8 // data_bits
    payload_units = units_per_byte * (payload_len + has_crc * crc_bytes)
    header_carried = (sf - sf_reduction) - header_nibbles
    denom = sf - sf_reduction * reduced
    blocks = max(ceil((payload_units - header_carried) / denom), 0)
    return (cr + data_bits) * blocks
