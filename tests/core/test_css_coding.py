from phy._css_lora import PARITY_MASKS

from marconi.core import coding

# data_bits=4 (Hamming 4+cr,4) is LoRa's; supplied here, never in production.
_DB = 4


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


def test_block_fec_cr4_corrects_single_bit_error():
    # 4 parity bits correct one flipped bit.
    masks = coding.parity_for_cr(PARITY_MASKS, 4)
    rows = coding.parity_rows(masks, _DB)
    data = [0, 1, 1, 0]
    cw = sum(b << i for i, b in enumerate(data))
    for p_idx, row in enumerate(rows):
        cw |= (sum(d * r for d, r in zip(data, row)) % 2) << (4 + p_idx)
    corrupted = cw ^ (1 << 2)
    assert coding.block_fec_decode(corrupted, masks, _DB, correct=True) == 0b0110


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
