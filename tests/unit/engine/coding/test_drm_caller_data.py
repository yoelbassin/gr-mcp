import numpy as np
from e2e.drm import _drm


def test_scattered_pilot_matches_golden():
    # golden phase indices from Dream/scratch (drm_phy_ref.md worked check):
    # s=0,k=1 -> phase 0 ; s=0,k=7 -> phase 524
    assert (
        abs(_drm.scat_ref(0, 1) - np.sqrt(2) * np.exp(1j * 2 * np.pi * 0 / 1024)) < 1e-9
    )
    assert (
        abs(_drm.scat_ref(0, 7) - np.sqrt(2) * np.exp(1j * 2 * np.pi * 524 / 1024))
        < 1e-9
    )
    # boosted band-edge pilot amplitude == 2
    assert abs(abs(_drm.scat_ref(2, -103)) - 2.0) < 1e-9


def test_pilot_positions_period_and_count():
    for fs in range(15):
        ks = _drm.scat_carriers(fs)
        assert all(k % 2 == 1 for k in ks)  # scattered pilots on odd carriers
        assert all((k - 1 - 2 * (fs % 3)) % 6 == 0 for k in ks)


def test_fac_cells_count_and_parity():
    assert len(_drm.FAC_CELLS) == 65
    assert all(c % 2 == 1 for _, c in _drm.FAC_CELLS)  # FAC carriers all odd


def test_crc8_and_crc16_are_real():
    z = np.zeros(64, np.int64)
    assert _drm.crc8(z) != _drm.crc8(np.r_[z[:-1], [1]])  # sensitive to input
    z16 = np.zeros(300, np.int64)
    assert _drm.crc16(z16) != _drm.crc16(np.r_[z16[:-1], [1]])


def test_pilot_tables_matches_scat_functions():
    carriers, values = _drm.pilot_tables()
    assert len(carriers) == len(values) == _drm.NS
    for fs in range(_drm.NS):
        assert carriers[fs] == _drm.scat_carriers(fs)
        assert values[fs] == [_drm.scat_ref(fs, k) for k in carriers[fs]]


def test_frequency_pilots():
    assert _drm.FP_CARRIERS == [16, 48, 64]
    assert len(_drm.FP_VALUES) == 3
    assert all(abs(abs(v) - np.sqrt(2)) < 1e-9 for v in _drm.FP_VALUES)


def test_sdc_cells_count_and_symbol_split():
    assert len(_drm.SDC_CELLS) == 322
    sym0 = [c for fs, c in _drm.SDC_CELLS if fs == 0]
    sym1 = [c for fs, c in _drm.SDC_CELLS if fs == 1]
    assert len(sym0) == 153 and len(sym1) == 169
    assert all(fs in (0, 1) for fs, _ in _drm.SDC_CELLS)


def test_select_perms_are_full_permutations_with_wanted_cells_fronted():
    # blockinterleaver_cc needs perm to span the whole repeating block it
    # gathers over; keep_m_in_n_c(65|322, block) then trims to the front.
    fac_perm = _drm.fac_select_perm()
    assert _drm.FAC_SELECT_BLOCK == _drm.NS * _drm.N_CARRIERS == len(fac_perm)
    assert sorted(fac_perm) == list(range(_drm.FAC_SELECT_BLOCK))
    assert fac_perm[:65] == [
        fs * _drm.N_CARRIERS + _drm._CARRIER_INDEX[k] for fs, k in _drm.FAC_CELLS
    ]

    sdc_perm = _drm.sdc_select_perm()
    assert _drm.SDC_SELECT_BLOCK == 3 * _drm.NS * _drm.N_CARRIERS == len(sdc_perm)
    assert sorted(sdc_perm) == list(range(_drm.SDC_SELECT_BLOCK))
    assert sdc_perm[:322] == [
        fs * _drm.N_CARRIERS + _drm._CARRIER_INDEX[k] for fs, k in _drm.SDC_CELLS
    ]


def test_bit_perms_are_permutations_of_their_length():
    assert sorted(_drm.fac_bit_perm()) == list(range(130))
    assert sorted(_drm.sdc_bit_perm()) == list(range(644))


def test_bit_perm_inverts_the_reference_tx_interleave():
    # fac_bit_perm/sdc_bit_perm must be the functional inverse of the raw
    # Dream interleave table so a gather-only permute (out[t] = in[perm[t]])
    # reproduces Dream's proven scatter-style RX deinterleave exactly.
    rng = np.random.default_rng(0)
    for coded_bits, bit_perm in (
        (_drm.FAC_CODED_BITS, _drm.fac_bit_perm()),
        (_drm.SDC_CODED_BITS, _drm.sdc_bit_perm()),
    ):
        coded = rng.integers(0, 2, coded_bits)
        table = _drm._interleave_table(coded_bits, _drm.INTERLEAVER_T0)
        sent = coded[table]  # Dream TX gather: sent[i] = coded[table[i]]
        deint = sent[np.asarray(bit_perm, dtype=np.int64)]  # gather via perm
        assert np.array_equal(deint, coded)


def test_puncture_keep_masks_lengths_and_weight():
    assert len(_drm.FAC_PUNCTURE_KEEP) == _drm.FAC_STEPS * _drm.CONV_RATE_INV
    assert sum(_drm.FAC_PUNCTURE_KEEP) == _drm.FAC_CODED_BITS
    assert len(_drm.SDC_PUNCTURE_KEEP) == _drm.SDC_STEPS * _drm.CONV_RATE_INV
    assert sum(_drm.SDC_PUNCTURE_KEEP) == _drm.SDC_CODED_BITS


def test_conv_code_matches_shared_dab_drm_mother_code():
    # Same K=7 rate-1/4 mother code already proven end-to-end for real
    # off-air DAB via the generic "fec" scheme="cc" stage.
    assert _drm.CONV_POLYS == [0o133, 0o171, 0o145, 0o133]
    assert _drm.CONV_RATE_INV == 4
    assert _drm.TAIL == 6


def test_energy_prbs_is_deterministic_and_nontrivial():
    seq = _drm.energy_prbs(72)
    assert seq.shape == (72,)
    assert set(np.unique(seq)) <= {0, 1}
    assert 0 < int(seq.sum()) < 72  # not all-zero, not all-one
    assert np.array_equal(seq, _drm.energy_prbs(72))  # deterministic


def _pack_msb(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="big").tobytes()


def test_parse_fac_roundtrips_channel_parameter_fields():
    # parse_fac(bytes.fromhex(...)) reads msb-first packed bytes end to end
    # (helpers.bitops.bits_to_bytes convention) -- see tests/e2e/drm/test_drm_offair.py.
    bits = np.zeros(72, np.int64)
    bits[0] = 1  # base_enh
    bits[1:3] = [1, 1]  # identity = 11
    bits[3:7] = [0, 0, 1, 1]  # occupancy = 0011 (occ 3)
    bits[8:10] = [1, 0]  # msc_mode = 10
    bits[10] = 1  # sdc_mode
    fields = _drm.parse_fac(_pack_msb(bits))
    assert fields["base_enh"] == 1
    assert fields["identity"] == "11"
    assert fields["occupancy"] == "0011"
    assert fields["msc_mode"] == "10"
    assert fields["sdc_mode"] == 1


def test_parse_sdc_label_extracts_utf8_type1_entity():
    block = np.zeros(316, np.int64)
    label = b"DW DRM"
    entity_bits = np.zeros(7 + 1 + 4 + 4 + 8 * len(label), np.int64)
    pos = 0
    entity_bits[pos : pos + 7] = [int(b) for b in format(len(label), "07b")]
    pos += 7 + 1  # length field, then version flag left at 0
    entity_bits[pos : pos + 4] = [int(b) for b in format(1, "04b")]  # type 1
    pos += 4 + 4  # etype field, then 2-bit short_id + 2-bit rfu left at 0
    for i, byte in enumerate(label):
        entity_bits[pos + 8 * i : pos + 8 * i + 8] = [
            int(b) for b in format(byte, "08b")
        ]
    block[4 : 4 + entity_bits.size] = entity_bits
    assert _drm.parse_sdc_label(_pack_msb(block)) == "DW DRM"


def test_parse_sdc_label_empty_when_no_type1_entity():
    block = np.zeros(316, np.int64)
    assert _drm.parse_sdc_label(_pack_msb(block)) == ""
