import numpy as np
from phy._css_lora import HEADER as _HEADER
from phy._css_lora import PARITY_MASKS as _PARITY_MASKS

from marconi.bits.carriers import RxCarrier
from marconi.bits.symbols import css_explicit_decode_rx
from marconi.core import coding

# IQ_2 SF11/BW125/CR4-5/LDRO explicit-header frame — raw dechirp argmax bins
# (8 header + 285 payload), known-good (V1 tests/phy/test_header_demux_stage.py).
_SF11_SYMBOLS = [
    1813, 1225, 1085, 685, 281, 353, 81, 1445, 1713, 241, 1413, 1449, 573, 1213,
    1869, 345, 1657, 969, 1353, 1077, 785, 117, 1233, 1509, 1837, 1633, 1549, 1317,
    2041, 1309, 389, 281, 817, 1005, 1493, 645, 481, 1237, 1925, 1717, 1837, 189,
    1081, 133, 1529, 1029, 837, 1301, 1605, 1573, 1097, 1265, 1737, 1657, 1629, 837,
    445, 1117, 385, 1617, 857, 1321, 657, 445, 1577, 937, 97, 1541, 1593, 1753, 585,
    1269, 173, 1337, 1825, 581, 577, 517, 833, 977, 93, 217, 1077, 897, 597, 1873,
    1641, 1901, 1505, 361, 1325, 1473, 549, 1757, 237, 545, 881, 321, 325, 2013, 69,
    233, 1141, 581, 1573, 72, 1629, 1756, 965, 729, 589, 1273, 905, 821, 825, 1420,
    552, 1824, 692, 56, 1932, 128, 368, 692, 56, 112, 1580, 984, 1324, 244, 492, 952,
    1556, 1772, 140, 284, 1292, 832, 140, 1100, 152, 340, 1548, 892, 1108, 1876, 1944,
    1188, 864, 916, 1836, 572, 364, 860, 1000, 2000, 1896, 1012, 1336, 216, 1612, 768,
    1700, 1524, 320, 380, 1484, 972, 1032, 700, 640, 1688, 1284, 1848, 800, 1468, 948,
    1952, 312, 1824, 1604, 236, 528, 260, 1880, 328, 1440, 132, 260, 1880, 328, 264,
    1324, 264, 1860, 372, 1056, 1608, 364, 1928, 232, 436, 1016, 336, 2036, 20, 1820,
    156, 1832, 1272, 500, 904, 1620, 1832, 1276, 1540, 1336, 752, 1704, 2044, 2040,
    108, 1952, 1684, 1920, 248, 1728, 708, 144, 1136, 228, 596, 1128, 940, 1524, 2024,
    1816, 1776, 976, 756, 1516, 1088, 824, 44, 756, 1516, 744, 1964, 1584, 1736, 1428,
    1868, 632, 1776, 1868, 664, 512, 220, 1908, 1088, 132, 360, 1384, 1204, 1084, 1924,
    1988, 48, 1200, 972, 1948, 860, 1412, 1920, 592, 1884, 1652, 288, 1540, 1696, 1348,
    1296, 1936, 1332, 1852, 632, 960, 1224, 1640, 716, 56, 1976, 1772,
]  # fmt: skip

# whitening sequence (LoRa, off-air); from V1 examples/lora/sfo_search.py
_WHITEN = bytes.fromhex(
    "fffefcf8f0e1c2850b172f5ebc78f1e3c68d1a3468d0a04080010204081123478e1c3871e2c489"
    "12254b972e5cb870e0c08103060c193264c992244993264d9b376edcb972e4c890204182050a15"
    "2b56ad5bb66ddab56bd6ac59b265cb962c58b061c3870f1f3e7dfbf6eddbb76fdebd7af5ebd7ae"
    "5dba74e8d1a24488102143860d1b366cd8b163c78f1e3c79f3e7ce9c3973e6cc983162c58b162d"
    "5ab469d2a4489122458a142952a54a952a54a953a74e9d3b77eeddbb76ecd9b367cf9e3d7bf7ef"
    "dfbf7efdfaf4e9d3a64c993366cd9a356ad4a851a3468c183060c183070e1d3a75ead5aa55ab57"
    "af5fbe7cf9f2e5ca942850a142840913274f9f3f7f"
)


def _assemble(bits, payload_len=255):
    nibbles = [
        (bits[4 * i] << 3)
        | (bits[4 * i + 1] << 2)
        | (bits[4 * i + 2] << 1)
        | bits[4 * i + 3]
        for i in range(len(bits) // 4)
    ]
    raw = bytearray()
    for i in range(len(nibbles) // 2):
        low, high = nibbles[2 * i], nibbles[2 * i + 1]
        if i < payload_len:
            low ^= _WHITEN[i] & 0x0F
            high ^= _WHITEN[i] >> 4
        raw.append(((high << 4) | low) & 0xFF)
    return bytes(raw[:payload_len])


def _run(symbols, marks=(), params=_HEADER):
    carrier = RxCarrier(
        bits=np.zeros(0, np.uint8),
        symbols=np.asarray(symbols, dtype=np.int16),
        marks=tuple(marks),
    )
    return css_explicit_decode_rx(carrier, **params).bits


def test_explicit_decode_yields_rf_fingerpring_payload():
    bits = _run(_SF11_SYMBOLS)
    payload = _assemble(list(bits))
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]


def test_explicit_decode_loops_back_to_back_frames():
    """Two frames in one symbol stream decode as two frames — the pre-fix
    block set _done after frame 1 and consumed-and-discarded the rest
    (issue 03)."""
    one = _run(_SF11_SYMBOLS)
    two = _run(_SF11_SYMBOLS * 2)
    assert len(two) == 2 * len(one)
    assert _assemble(list(two[: len(one)])) == _assemble(list(one))
    assert _assemble(list(two[len(one) :])) == _assemble(list(one))


def _encode_header(payload_len, cr, has_crc):
    """Synthesize a parity-valid explicit header (carry nibbles zeroed) by
    running the decoder's coding primitives in reverse."""
    sf, sf_red = _HEADER["sf"], _HEADER["sf_reduction"]
    sf_app = sf - sf_red
    header_cr = _HEADER["header_cr"]
    cw_len = header_cr + _HEADER["data_bits"]
    data_int = (payload_len << 4) | (cr << 1) | has_crc
    masks = _HEADER["header_parity"]
    parity = 0
    for i, m in enumerate(masks):
        parity |= (bin(m & data_int).count("1") & 1) << (len(masks) - 1 - i)
    hbits = [(data_int >> (11 - i)) & 1 for i in range(12)] + [0, 0, 0]
    hbits += [(parity >> (4 - i)) & 1 for i in range(5)]
    nibbles = [
        (hbits[4 * i] << 3)
        | (hbits[4 * i + 1] << 2)
        | (hbits[4 * i + 2] << 1)
        | hbits[4 * i + 3]
        for i in range(5)
    ] + [0] * (sf_app - 5)
    fec = coding.parity_for_cr(_PARITY_MASKS, header_cr)
    perm = coding.diag_deinterleave_perm(sf_app, cw_len)
    deint = []
    for nib in nibbles:
        cw = nib
        for p, m in enumerate(fec):
            cw |= (bin(m & nib).count("1") & 1) << (4 + p)
        deint.extend(((cw >> (cw_len - 1 - k)) & 1) for k in range(cw_len))
    chunk = [0] * (sf_app * cw_len)
    for i, src in enumerate(perm):
        chunk[src] = deint[i]
    syms = []
    for s in range(_HEADER["header_symbols"]):
        gv = 0
        for j in range(sf_app):
            gv = (gv << 1) | chunk[s * sf_app + j]
        syms.append((coding.gray_decode(gv) * (1 << sf_red)) % (1 << sf))
    return syms


def _encode_payload(nibbles, cr, sf_app):
    """Synthesize payload symbols by running the decoder's coding primitives
    in reverse; full-rate (sf_app == sf) uses the divisor=1 bin-offset lane."""
    sf, data_bits = _HEADER["sf"], _HEADER["data_bits"]
    cw_len = cr + data_bits
    fec = coding.parity_for_cr(_PARITY_MASKS, cr)
    perm = coding.diag_deinterleave_perm(sf_app, cw_len)
    syms = []
    for b in range(len(nibbles) // sf_app):
        deint = []
        for nib in nibbles[b * sf_app : (b + 1) * sf_app]:
            cw = nib
            for p, m in enumerate(fec):
                cw |= (bin(m & nib).count("1") & 1) << (data_bits + p)
            deint.extend(((cw >> (cw_len - 1 - k)) & 1) for k in range(cw_len))
        chunk = [0] * (sf_app * cw_len)
        for i, src in enumerate(perm):
            chunk[src] = deint[i]
        for s in range(cw_len):
            gv = 0
            for j in range(sf_app):
                gv = (gv << 1) | chunk[s * sf_app + j]
            if sf_app == sf:
                syms.append(
                    (coding.gray_decode(gv) + _HEADER["full_offset"]) % (1 << sf)
                )
            else:
                syms.append((coding.gray_decode(gv) * (1 << (sf - sf_app))) % (1 << sf))
    return syms


def test_explicit_decode_full_rate_payload_lane():
    """reduced=False routes the payload demap through the divisor=1/full-offset
    lane — the common non-LDRO config (SF7-9); the off-air fixtures are all
    LDRO, so only the reduced lane had coverage."""
    sf, cr, payload_len = _HEADER["sf"], 1, 16
    # 3 interleave blocks of sf nibbles: ceil((32-4)/11) — a count that differs
    # from the reduced-denominator ceil(28/9)=4, pinning the non-LDRO frame
    # algebra too
    nibbles = [(3 * i + 1) % 16 for i in range(3 * sf)]
    syms = _encode_header(payload_len, cr, 0) + _encode_payload(nibbles, cr, sf)
    bits = _run(syms, params={**_HEADER, "reduced": False})
    expected = [0] * 16  # the header block's carry nibbles are zeroed
    for nib in nibbles:
        expected.extend((nib >> (3 - j)) & 1 for j in range(4))
    assert list(bits) == expected[: 8 * payload_len]


def test_explicit_decode_oversize_length_does_not_swallow_later_marks():
    """A parity-valid header whose declared frame overruns the symbol array
    must drop only its own mark — the pre-fix carve broke out of the marks
    loop there, swallowing every later real burst (the module docstring's
    'a corrupt length can never swallow a real burst')."""
    one = _run(_SF11_SYMBOLS)
    # fixture guard: the synthetic header must be parity-valid and decode,
    # else the oversize case below would pass via the corrupt-header skip
    # even without the fix (first 2 bytes differ: carry nibbles are zeroed)
    swapped = _run(_encode_header(255, 1, 1) + _SF11_SYMBOLS[8:], marks=(0,))
    assert _assemble(list(swapped))[2:] == _assemble(list(one))[2:]

    oversize = _encode_header(255, 4, 1)
    bits = _run(oversize + [0] * 60 + _SF11_SYMBOLS, marks=(0, len(oversize) + 60))
    assert list(bits) == list(one)


def test_explicit_decode_corrupt_header_skips_to_next_mark():
    """A corrupt header is skipped and the next marked burst still decodes —
    the pre-fix block treated a header-parity failure as end-of-stream and
    reported nothing (issue 03)."""
    rng = np.random.default_rng(5)
    corrupt = list(_SF11_SYMBOLS)
    corrupt[:8] = [int(v) for v in rng.integers(1, 2048, 8)]
    frame_len = len(_SF11_SYMBOLS)
    one = _run(_SF11_SYMBOLS)
    bits = _run(corrupt + _SF11_SYMBOLS, marks=(0, frame_len))
    payload = _assemble(list(bits))
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]
    assert list(bits) == list(one)
