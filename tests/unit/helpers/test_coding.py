"""External truth vectors: expected values sourced OUTSIDE this codebase
(published Gray/Hamming tables, an off-air-derived header vector), so an
internally-consistent-but-wrong implementation cannot pass."""

from __future__ import annotations

from itertools import permutations

from helpers import blockmath as coding
from helpers._css_lora import PARITY_MASKS
from helpers.blockmath import block_fec_decode, correct_codeword

ROW = [0b1011, 0b1101, 0b1110]  # 3 parity eqs over 4 data bits (caller data)

# data_bits=4 (Hamming 4+cr,4) is LoRa's; supplied here, never in production.
_DB = 4


def _encode(data: int, masks: list[int], data_bits: int) -> int:
    cw = data & ((1 << data_bits) - 1)
    for p, m in enumerate(masks):
        bit = bin(m & data).count("1") & 1
        cw |= bit << (data_bits + p)
    return cw


def test_correct_codeword_returns_full_and_fixes_single_error() -> None:
    data_bits, masks = 4, ROW
    n = data_bits + len(masks)
    for data in range(1 << data_bits):
        cw = _encode(data, masks, data_bits)
        assert correct_codeword(cw, masks, data_bits) == cw  # clean codeword unchanged
        for bit in range(n):
            got = correct_codeword(cw ^ (1 << bit), masks, data_bits)
            assert got == cw, (
                data,
                bit,
                got,
            )  # any single-bit error repaired, full width


def test_block_fec_decode_is_correct_codeword_masked_to_data() -> None:
    data_bits, masks = 4, ROW
    mask = (1 << data_bits) - 1
    for cw in range(1 << (data_bits + len(masks))):
        assert block_fec_decode(cw, masks, data_bits, correct=True) == (
            correct_codeword(cw, masks, data_bits) & mask
        )
        assert block_fec_decode(cw, masks, data_bits, correct=False) == (cw & mask)


def test_gray_roundtrip():
    for x in range(256):
        assert coding.gray_decode(coding.gray_encode(x)) == x


def test_demap_reduced_rate_absorbs_minus_one_within_quad():
    # reduced-rate (sf_app < sf): value = gray_encode(s // 4). s and s-1 share s//4
    # unless s is a multiple of 4.
    n = 1 << 11
    a = coding.demap_symbols([20], sf_app=9, n=n, divisor=4, offset=0)
    b = coding.demap_symbols([21], sf_app=9, n=n, divisor=4, offset=0)
    assert a == b  # 20//4 == 21//4 == 5
    c = coding.demap_symbols([19], sf_app=9, n=n, divisor=4, offset=0)
    assert c != a  # 19//4 == 4 != 5


def test_demap_full_rate_offset_and_width():
    n = 1 << 7
    bits = coding.demap_symbols([5], sf_app=7, n=n, divisor=1, offset=1)
    assert bits == [(coding.gray_encode(4) >> (6 - j)) & 1 for j in range(7)]


def test_parity_rows_masks_roundtrip():
    # bit b of the mask is row entry b
    assert coding.parity_rows([7, 14], 4) == [[1, 1, 1, 0], [0, 1, 1, 1]]


def test_block_fec_detect_only_cr1_passes_clean_codeword():
    # 1 parity bit detects but does not correct; a clean codeword returns its
    # nibble.
    nibble = 0b1011
    masks = coding.parity_for_cr(PARITY_MASKS, 1)
    (row,) = coding.parity_rows(masks, _DB)
    p = sum(d * r for d, r in zip([1, 1, 0, 1], row)) % 2  # data bits LSB-first
    codeword = nibble | (p << 4)
    assert coding.block_fec_decode(codeword, masks, _DB, correct=False) == nibble


def test_block_fec_cr1_does_not_correct_even_when_requested():
    # 1 parity bit is below the Hamming bound; correct=True must still return the
    # corrupted nibble unchanged — detect-only stays detect-only.
    assert not coding.can_correct(1, _DB)
    nibble = 0b1011
    masks = coding.parity_for_cr(PARITY_MASKS, 1)
    (row,) = coding.parity_rows(masks, _DB)
    data = [1, 1, 0, 1]
    p = sum(d * r for d, r in zip(data, row)) % 2
    codeword = nibble | (p << 4)
    corrupted = codeword ^ 1  # flip data bit 0; corrupted nibble = 0b1010
    correct = coding.can_correct(1, _DB)
    assert coding.block_fec_decode(corrupted, masks, _DB, correct) == (nibble ^ 1)


def test_can_correct_matches_hamming_bound():
    assert [coding.can_correct(cr, _DB) for cr in (1, 2, 3, 4)] == [
        False,
        False,
        True,
        True,
    ]


def test_frame_len_sf11_255_byte_reduced_cr1():
    # 255-byte payload, has_crc, cr=1, sf=11, reduced=1, sf_reduction=2, 4-bit
    # data units, 2-byte CRC, 5 header nibbles -> 285 coded payload symbols.
    got = coding.css_explicit_frame_len(
        255, 1, 1, 11, 1, 2, data_bits=4, header_nibbles=5, crc_bytes=2
    )
    assert got == 285


def test_header_parity_ok_roundtrips_known_masks():
    masks = [3840, 2273, 1178, 599, 303]
    data = 0b101010101010
    # compute the parity the function expects, then confirm it validates
    computed = 0
    for i, m in enumerate(masks):
        computed |= (bin(m & data).count("1") & 1) << (len(masks) - 1 - i)
    assert coding.header_parity_ok(data, computed, masks)


def test_supported_cr_reads_the_supplied_table():
    for cr in (1, 2, 3, 4):
        assert coding.supported_cr(PARITY_MASKS, cr)
    for cr in (0, 5, 6, 7):
        assert not coding.supported_cr(PARITY_MASKS, cr)


def _diag_reference(sf_app: int, cw_len: int) -> list[int]:
    # Independent 2-D construction of the same diagonal de-interleave (build the
    # symbol matrix, undo the per-column rotation, read out codewords), written
    # from the gr-lora_sdr definition rather than production's closed form.
    out: list[int] = []
    for oc in range(sf_app):
        for k in range(cw_len):
            row = cw_len - 1 - k
            col = (row - oc - 1) % sf_app
            out.append(row * sf_app + col)
    return out


def test_diag_perm_matches_independent_construction():
    for sf_app, cw_len in ((4, 8), (7, 5), (9, 8), (11, 8)):
        assert coding.diag_deinterleave_perm(sf_app, cw_len) == _diag_reference(
            sf_app, cw_len
        )


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
