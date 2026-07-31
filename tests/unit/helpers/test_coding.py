from __future__ import annotations

from helpers.blockmath import block_fec_decode, correct_codeword

ROW = [0b1011, 0b1101, 0b1110]  # 3 parity eqs over 4 data bits (caller data)


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
