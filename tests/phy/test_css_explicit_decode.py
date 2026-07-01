from marconi.phy.backends.gnuradio.embedded.coding import make_css_explicit_decode

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

_HEADER = dict(
    header_cr=4,
    header_symbols=8,
    header_nibbles=5,
    sf_reduction=2,
    header_data_bits=12,
    header_parity=[3840, 2273, 1178, 599, 303],
    field_payload_len=[0, 8],
    field_cr=[8, 3],
    field_has_crc=[11, 1],
    field_parity=[15, 5],
)

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


def _run_block(symbols):
    from gnuradio import blocks, gr

    blk = make_css_explicit_decode(gr, sf=11, ldro=True, **_HEADER)
    src = blocks.vector_source_s([int(s) for s in symbols], False, 1, [])
    snk = blocks.vector_sink_b()
    tb = gr.top_block("css_explicit_decode_test")
    tb._py_instances = {"blk": blk}  # GC anchor
    tb.connect(src, blk, snk)
    # uint8-output embedded blocks segfault on the main thread; run off-thread.
    import threading

    t = threading.Thread(target=tb.run)
    t.start()
    t.join(30)
    return list(snk.data())


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


def test_explicit_decode_yields_rf_fingerpring_payload():
    bits = _run_block(_SF11_SYMBOLS)
    payload = _assemble(bits)
    assert payload.startswith(b"RF fingerpring Project for Lora"), payload[:31]
