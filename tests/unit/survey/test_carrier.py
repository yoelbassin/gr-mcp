from __future__ import annotations

import numpy as np
import numpy.typing as npt

from marconi.survey.measure import _MPSK_JUMP, _carrier

_FS = 1_000_000.0
_SPS = 8
_NSYM = 4000
_FOFF = 37_000.0


def _rng() -> np.random.Generator:
    return np.random.default_rng(1)


def _upsample(sym: np.ndarray) -> np.ndarray:
    x = np.zeros(sym.size * _SPS, dtype=complex)
    x[::_SPS] = sym
    return np.convolve(x, np.ones(_SPS), "same")


def _noisy(
    x: npt.NDArray[np.complex64], foff: float, snr_db: float
) -> npt.NDArray[np.complex64]:
    r = _rng()
    x = x * np.exp(1j * 2 * np.pi * foff / _FS * np.arange(x.size))
    p = float(np.mean(np.abs(x) ** 2))
    npow = p / (10 ** (snr_db / 10))
    noise = np.sqrt(npow / 2) * (
        r.standard_normal(x.size) + 1j * r.standard_normal(x.size)
    )
    out: npt.NDArray[np.complex64] = (x + noise).astype(np.complex64)
    return out


def _psk(
    order: int, snr_db: float = 15.0, foff: float = _FOFF
) -> npt.NDArray[np.complex64]:
    k = _rng().integers(0, order, _NSYM)
    return _noisy(_upsample(np.exp(1j * 2 * np.pi * k / order)), foff, snr_db)


def _narrowband_fsk(snr_db: float = 15.0) -> np.ndarray:
    # small deviation relative to fs: the two tones sit close to a common
    # carrier, so x^2 concentrates strongly at 2x that carrier — the exact
    # confound that makes a naive "order-2 line => BPSK" classifier wrong.
    bits = _rng().integers(0, 2, _NSYM) * 2 - 1
    inst = np.repeat(bits * (_FS / 200.0), _SPS)
    phase = 2 * np.pi * np.cumsum(inst) / _FS
    return _noisy(np.exp(1j * phase), _FOFF, snr_db)


def _psk_seeded(order: int, snr_db: float, foff: float, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    k = g.integers(0, order, _NSYM)
    x = _upsample(np.exp(1j * 2 * np.pi * k / order))
    x = x * np.exp(1j * 2 * np.pi * foff / _FS * np.arange(x.size))
    p = float(np.mean(np.abs(x) ** 2))
    npow = p / (10 ** (snr_db / 10))
    n = np.sqrt(npow / 2) * (g.standard_normal(x.size) + 1j * g.standard_normal(x.size))
    out: npt.NDArray[np.complex64] = (x + n).astype(np.complex64)
    return out


def _qam(
    levels: npt.NDArray[np.float64], snr_db: float, foff: float, seed: int
) -> npt.NDArray[np.complex64]:
    g = np.random.default_rng(seed)
    k = levels.size
    sym = levels[g.integers(0, k, _NSYM)] + 1j * levels[g.integers(0, k, _NSYM)]
    sym = sym / np.sqrt(np.mean(np.abs(sym) ** 2))
    x = _upsample(sym) * np.exp(
        1j * 2 * np.pi * foff / _FS * np.arange(sym.size * _SPS)
    )
    p = float(np.mean(np.abs(x) ** 2))
    npow = p / (10 ** (snr_db / 10))
    n = np.sqrt(npow / 2) * (g.standard_normal(x.size) + 1j * g.standard_normal(x.size))
    out: npt.NDArray[np.complex64] = (x + n).astype(np.complex64)
    return out


def _bursty_psk(
    order: int, duty: float, foff: float, snr_db: float = 18.0
) -> np.ndarray:
    period = 4000
    g = np.random.default_rng(5)
    k = g.integers(0, order, _NSYM)
    x = _upsample(np.exp(1j * 2 * np.pi * k / order))
    x = x * np.exp(1j * 2 * np.pi * foff / _FS * np.arange(x.size))
    on = (np.arange(x.size) % period) < int(duty * period)
    x = x * on
    p = float(np.mean(np.abs(x[on]) ** 2))
    npow = p / (10 ** (snr_db / 10))
    n = np.sqrt(npow / 2) * (g.standard_normal(x.size) + 1j * g.standard_normal(x.size))
    out: npt.NDArray[np.complex64] = (x + n).astype(np.complex64)
    return out


def test_qpsk_reads_order_4_with_precise_offset() -> None:
    c = _carrier(_psk(4), _FS, 30_000.0, _FOFF)
    assert c.psk_order == 4
    assert c.method == "mpsk"
    assert abs(c.offset_hz - _FOFF) < 500.0


def test_8psk_reads_order_8() -> None:
    c = _carrier(_psk(8), _FS, 30_000.0, _FOFF)
    assert c.psk_order == 8
    assert abs(c.offset_hz - _FOFF) < 500.0


def test_bpsk_abstains_but_surfaces_the_order_2_line() -> None:
    # squaring cannot separate BPSK from a bare carrier, so order is null; the
    # strong order-2 concentration is the evidence the caller reads with the
    # envelope block (constant envelope => BPSK, amplitude-modulated => OOK).
    c = _carrier(_psk(2), _FS, 30_000.0, _FOFF)
    assert c.psk_order is None
    assert c.phase_concentration.order_2 > c.phase_concentration.order_4
    assert c.phase_concentration.order_2 >= 25.0


def test_narrowband_fsk_is_not_mislabeled_bpsk() -> None:
    c = _carrier(_narrowband_fsk(), _FS, 30_000.0, 0.0)
    assert c.psk_order is None  # order-2 line present, but no jump at 4/8
    assert c.phase_concentration.order_2 >= c.phase_concentration.order_4


def test_low_snr_8psk_reads_none_not_a_false_lock() -> None:
    c = _carrier(_psk(8, snr_db=3.0), _FS, 30_000.0, _FOFF)
    assert c.psk_order is None


def test_noise_has_no_carrier() -> None:
    r = _rng()
    noise = (r.standard_normal(32_000) + 1j * r.standard_normal(32_000)).astype(
        np.complex64
    )
    c = _carrier(noise, _FS, 30_000.0, 0.0)
    assert c.psk_order is None
    pc = c.phase_concentration
    assert pc.order_2 < 5.0 and pc.order_4 < 5.0 and pc.order_8 < 5.0


def test_off_center_flags_a_large_offset() -> None:
    on = _carrier(_psk(4, foff=1000.0), _FS, 40_000.0, 1000.0)
    off = _carrier(_psk(4, foff=37_000.0), _FS, 40_000.0, 37_000.0)
    assert on.off_center is False
    assert off.off_center is True


def test_dc_spike_does_not_defeat_order_detection() -> None:
    # LO leakage (universal on RTL-SDR) squares into a strong order-2 line;
    # without DC removal it blanks the order-4 claim.
    x = _psk(4)
    dc = 3.0 * np.sqrt(np.mean(np.abs(x) ** 2))
    c = _carrier((x + dc).astype(np.complex64), _FS, 30_000.0, _FOFF)
    assert c.psk_order == 4


def test_large_offset_8psk_reports_true_offset_not_alias() -> None:
    for foff in (70_000.0, 90_000.0):
        c = _carrier(_psk(8, foff=foff), _FS, 30_000.0, foff)
        assert c.psk_order == 8
        assert abs(c.offset_hz - foff) < 500.0, (foff, c.offset_hz)


def test_offset_at_fs_over_8_is_not_reported_as_centered() -> None:
    foff = _FS / 8  # 8*foff == fs -> old code aliased the line to DC
    c = _carrier(_psk(8, foff=foff), _FS, 40_000.0, foff)
    assert abs(c.offset_hz - foff) < 1000.0, c.offset_hz
    assert c.off_center is True


def test_ambiguous_offset_falls_back_to_centroid() -> None:
    # centroid sits halfway between two M-th-power candidates -> can't
    # disambiguate; report the centroid and flag it rather than guess.
    step = _FS / 8
    c = _carrier(_psk(8, foff=37_000.0), _FS, 200_000.0, 37_000.0 + step / 2)
    assert c.offset_ambiguous is True
    assert c.method == "spectral_centroid"


def test_bounded_truncates_to_cap() -> None:
    from marconi.survey.measure import _CARRIER_MAX_SAMPLES, _bounded

    big = np.ones(_CARRIER_MAX_SAMPLES + 123, dtype=np.complex64)
    assert _bounded(big).size == _CARRIER_MAX_SAMPLES
    small = np.ones(1000, dtype=np.complex64)
    assert _bounded(small).size == 1000


def test_qpsk_detected_across_snr_and_seeds() -> None:
    for seed in (1, 2, 3):
        for snr in (8.0, 12.0):
            c = _carrier(_psk_seeded(4, snr, _FOFF, seed), _FS, 30_000.0, _FOFF)
            assert c.psk_order == 4, (seed, snr, c.psk_order)


def test_square_qam_shares_the_order_4_line() -> None:
    # 16-QAM has QPSK's 4-fold phase symmetry; the M-th-power cannot separate
    # them, so a strong order-4 line is expected. The envelope block (amplitude
    # kurtosis) is what disambiguates PSK from QAM — NOT this block.
    c = _carrier(
        _qam(np.array([-3.0, -1.0, 1.0, 3.0]), 20.0, _FOFF, 1),
        _FS,
        30_000.0,
        _FOFF,
    )
    assert c.phase_concentration.order_4 >= 25.0


def test_64qam_shares_the_order_4_line() -> None:
    # 64-QAM has the same 4-fold phase symmetry as 16-QAM and QPSK; with a
    # correct (masked) time base its order-4 line is genuine, not a splice
    # artifact. The envelope block disambiguates PSK from QAM, not this block.
    c = _carrier(_qam(np.arange(-7.0, 8.0, 2.0), 25.0, _FOFF, 1), _FS, 30_000.0, _FOFF)
    assert c.phase_concentration.order_4 >= 25.0


def test_bursty_offset_precision() -> None:
    c = _carrier(_bursty_psk(4, 0.05, 37_000.0), _FS, 30_000.0, 37_000.0)
    assert c.psk_order == 4
    assert abs(c.offset_hz - 37_000.0) < 150.0, c.offset_hz


def _pi4_dqpsk(snr_db: float = 18.0, foff: float = _FOFF) -> np.ndarray:
    g = np.random.default_rng(7)
    incr = (2 * g.integers(0, 4, _NSYM) + 1) * (np.pi / 4)  # {±pi/4, ±3pi/4}
    return _noisy(_upsample(np.exp(1j * np.cumsum(incr))), foff, snr_db)


def test_pi4_staggered_quaternary_reads_order_8_with_true_offset() -> None:
    # z^4 of a staggered (pi/4-shifted) quaternary alphabet alternates between
    # its two component grids: the 4-fold line is displaced by Rs/2, so a naive
    # order-4 claim would report an offset biased by Rs/8 (~15.6 kHz here) with
    # "mpsk" confidence. The 8-fold line is clean and at the true 8*offset.
    c = _carrier(_pi4_dqpsk(), _FS, 30_000.0, _FOFF)
    assert c.psk_order == 8, c.psk_order
    assert c.method == "mpsk"
    assert abs(c.offset_hz - _FOFF) < 500.0, c.offset_hz


def test_hot_interferer_burst_does_not_gate_out_the_victim() -> None:
    # a short burst 30 dB above the victim must not become the activity mask's
    # reference level (an |x|^2 max would): the victim stays in the mask and
    # its order/offset stay readable.
    x = _psk(4).astype(np.complex64)
    n = x.size
    hit = slice(n // 2, n // 2 + int(0.02 * n))
    tone = np.sqrt(1000.0) * np.exp(
        1j * 2 * np.pi * 0.11 * np.arange(hit.stop - hit.start)
    )
    x[hit] += tone.astype(np.complex64)
    c = _carrier(x, _FS, 30_000.0, _FOFF)
    assert c.psk_order == 4, c.psk_order
    assert abs(c.offset_hz - _FOFF) < 1000.0, c.offset_hz


def _fsk(dev_hz: float, snr_db: float = 20.0) -> npt.NDArray[np.complex64]:
    r = _rng()
    bits = r.integers(0, 2, _NSYM) * 2 - 1
    inst = np.repeat(bits.astype(float), _SPS) * dev_hz
    phase = 2 * np.pi * np.cumsum(inst) / _FS
    return _noisy(np.exp(1j * phase).astype(np.complex64), 0.0, snr_db)


def test_binary_fsk_is_not_claimed_as_a_four_phase_alphabet() -> None:
    # 2-FSK folds measurably harder at order 4 than order 2, landing in the
    # band (jump ~5) where only _MPSK_JUMP holds the claim back. Without it
    # survey reports QPSK/square-QAM phase symmetry for plain frequency
    # shift keying and the agent builds a coherent PSK chain for it.
    c = _carrier(_fsk(_FS / 60.0), _FS, 60_000.0, 0.0)
    pc = c.phase_concentration
    assert pc.order_4 > pc.order_2, (pc.order_2, pc.order_4)
    assert pc.order_4 < _MPSK_JUMP * pc.order_2, (pc.order_2, pc.order_4)
    assert c.psk_order is None


def test_signal_half_its_bandwidth_off_channel_is_flagged() -> None:
    # The only off_center coverage sat at ratio 0.025 (False) and 0.925
    # (True); a signal squarely between them decides whether the agent is
    # told to re-centre before a DC-assuming demod misses it.
    occupied = 30_000.0
    c = _carrier(_psk(4, foff=occupied / 2.0), _FS, occupied, occupied / 2.0)
    assert c.off_center
