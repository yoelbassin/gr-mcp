from marconi.phy.modulation.css import coding


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


def test_block_fec_detect_only_cr1_passes_clean_codeword():
    # CR4/5 (1 parity bit) detects but does not correct; a clean codeword
    # returns its nibble.
    nibble = 0b1011
    parity = coding.HAMMING_PARITY[1]
    p = sum(d * r for d, r in zip([1, 1, 0, 1], parity[0])) % 2  # data bits LSB-first
    codeword = nibble | (p << 4)
    assert coding.block_fec_decode(codeword, parity, correct=False) == nibble


def test_block_fec_cr4_corrects_single_bit_error():
    # CR4/8 (4 parity bits) corrects one flipped bit.
    parity = coding.HAMMING_PARITY[4]
    # build a valid codeword for nibble 0b0110, then flip bit 2
    data = [0, 1, 1, 0]
    cw = sum(b << i for i, b in enumerate(data))
    for p_idx, row in enumerate(parity):
        cw |= (sum(d * r for d, r in zip(data, row)) % 2) << (4 + p_idx)
    corrupted = cw ^ (1 << 2)
    assert coding.block_fec_decode(corrupted, parity, correct=True) == 0b0110


def test_block_fec_cr1_does_not_correct_even_when_requested():
    # CR4/5 (1 parity bit) is below the correction threshold; passing correct=True
    # must still return the corrupted nibble unchanged — detect-only stays detect-only.
    nibble = 0b1011
    parity = coding.HAMMING_PARITY[1]
    data = [1, 1, 0, 1]
    p = sum(d * r for d, r in zip(data, parity[0])) % 2
    codeword = nibble | (p << 4)
    corrupted = codeword ^ 1  # flip data bit 0; corrupted nibble = 0b1010
    assert coding.block_fec_decode(corrupted, parity, correct=True) == (nibble ^ 1)


def test_frame_len_sf11_255_byte_ldro_cr1():
    # IQ_2: 255-byte payload, has_crc, cr=1, sf=11, ldro=1 → 285 coded payload symbols.
    assert coding.css_explicit_frame_len(255, 1, 1, 11, 1) == 285


def test_header_parity_ok_roundtrips_known_masks():
    masks = [3840, 2273, 1178, 599, 303]
    data = 0b101010101010
    # compute the parity the function expects, then confirm it validates
    computed = 0
    for i, m in enumerate(masks):
        computed |= (bin(m & data).count("1") & 1) << (len(masks) - 1 - i)
    assert coding.header_parity_ok(data, computed, masks)
