"""External truth vectors (issue 04 step-0): expected values sourced OUTSIDE
this codebase, so an internally-consistent-but-wrong table cannot pass."""

from __future__ import annotations

from itertools import permutations

from phy._css_lora import PARITY_MASKS

from marconi.core import coding

# Binary-reflected Gray code, first 16 values as published (any standards text,
# e.g. Knuth TAOCP 4A §7.2.1.1 / Wikipedia "Gray code" table).
_BRGC = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]

# LoRa Hamming(8,4) encode table as published by gr-lora (rpp0) /
# Robyns et al., "A Multi-Channel Software Decoder for the LoRa PHY".
_GRLORA_H84 = [
    0x00, 0xD2, 0x55, 0x87, 0x99, 0x4B, 0xCC, 0x1E,
    0xE1, 0x33, 0xB4, 0x66, 0x78, 0xAA, 0x2D, 0xFF,
]  # fmt: skip

# LoRa cr=4 parity masks — the four XOR-of-data check rows. Every value is
# forced by the published _GRLORA_H84 table alone (proved by
# test_cr4_parity_masks_match_published_table), so this is external truth, not a
# restatement of production's own constant.
_CR4_MASKS = [7, 14, 11, 13]


def test_gray_encode_matches_published_table() -> None:
    assert [coding.gray_encode(n) for n in range(16)] == _BRGC


def test_gray_decode_matches_published_table() -> None:
    assert [coding.gray_decode(g) for g in _BRGC] == list(range(16))


def _gather(cw: int, data_pos: tuple[int, ...], par_pos: tuple[int, ...]) -> int:
    data = sum(((cw >> p) & 1) << k for k, p in enumerate(data_pos))
    par = sum(((cw >> p) & 1) << k for k, p in enumerate(par_pos))
    return data | (par << 4)


def test_block_fec_decode_agrees_with_grlora_table() -> None:
    # Anchors the codeword -> data-nibble extraction convention (LSB-first from
    # the low nibble), NOT the parity masks. A plain nibble swap/reversal bridge
    # does not exist: LoRa's published byte interleaves data and parity bits
    # rather than keeping them in separate halves, so the bridge is a permutation
    # of all 8 bit positions into (4 data, 4 parity). At least one such
    # permutation maps ALL 16 published codewords onto the decoder's data output;
    # the data-column assignment is forced (only cols (1,2,3,5) reproduce every
    # nibble). This says nothing about mask correctness — block_fec_decode
    # returns only the data nibble, so any masks that leave a clean codeword's
    # data intact pass here (garbage masks included). The masks are anchored
    # separately by test_cr4_parity_masks_match_published_table.
    bridges = [
        perm
        for perm in permutations(range(8))
        if all(
            coding.block_fec_decode(
                _gather(cw, perm[:4], perm[4:]), _CR4_MASKS, 4, True
            )
            == nib
            for nib, cw in enumerate(_GRLORA_H84)
        )
    ]
    assert bridges, "no bit-position bridge maps the published table onto the decoder"


def _cr4_masks_from_published_table() -> set[int]:
    # Solve the parity-check masks from the published gr-lora codewords alone:
    # every codeword bit is a fixed XOR of the data value's bits (data value =
    # table index), so each of the 8 columns has exactly one mask in 1..15 that
    # reproduces it across all 16 rows. Single-bit solutions are the systematic
    # data columns; the multi-bit solutions are the four parity checks. No
    # Marconi constant enters — this is the external anchor the bridge test above
    # cannot be.
    parity: set[int] = set()
    for pos in range(8):
        (mask,) = [
            m
            for m in range(1, 16)
            if all(
                ((cw >> pos) & 1) == (bin(m & nib).count("1") & 1)
                for nib, cw in enumerate(_GRLORA_H84)
            )
        ]
        if bin(mask).count("1") > 1:
            parity.add(mask)
    return parity


def test_cr4_parity_masks_match_published_table() -> None:
    external = _cr4_masks_from_published_table()
    assert external == {7, 11, 13, 14}
    assert set(coding.parity_for_cr(PARITY_MASKS, 4)) == external


# The bridge that carries a published codeword into production's layout: data
# bits from cols (1,2,3,5) (forced above), parity cols (4,0,6,7) placed to match
# production's parity-row order [7,14,11,13]. Both tuples come from the published
# table's structure, not from any Marconi constant.
_PUB_DATA_POS = (1, 2, 3, 5)
_PUB_PAR_POS = (4, 0, 6, 7)


def test_block_fec_corrects_single_error_on_published_codeword() -> None:
    # Reference codewords are read straight from the published gr-lora table (via
    # the external bridge), so this is a true anchor: a wrong parity mask makes
    # the clean codeword's syndrome nonzero and mis-corrects, failing here. With
    # the correct masks every single-bit error over all 16 nibbles is corrected.
    for nib in range(16):
        cw = _gather(_GRLORA_H84[nib], _PUB_DATA_POS, _PUB_PAR_POS)
        assert coding.block_fec_decode(cw, _CR4_MASKS, 4, correct=True) == nib
        for bit in range(8):
            assert coding.block_fec_decode(cw ^ (1 << bit), _CR4_MASKS, 4, True) == nib


# IQ_2 capture, CRC-valid frame, extracted 2026-07-04: real off-air LoRa SF11
# header from artifacts/assets/LoRa/iq2_frame.cf32, pulled by instrumenting
# css_explicit_decode's _parse_header to surface (data_int, received) through
# RunResult.diagnostics for one run of the production rx path (chirp_sync ->
# dechirp -> css_explicit_decode). Both CRC-valid frames in the slice carry
# identical header fields (payload_len=255, cr=1, has_crc=1 -> data_int=4083),
# confirmed against the independently-decoded parity value received=10.
_IQ2_HEADER_VECTOR = (4083, 10)


def test_header_parity_vector_from_offair_frame() -> None:
    data_int, received = _IQ2_HEADER_VECTOR
    masks = [3840, 2273, 1178, 599, 303]
    assert coding.header_parity_ok(data_int, received, masks)
    for bit in range(12):
        assert not coding.header_parity_ok(data_int ^ (1 << bit), received, masks)
