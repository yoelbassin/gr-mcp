from __future__ import annotations

import numpy as np

from marconi.survey.measure import _carrier

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


def _noisy(x: np.ndarray, foff: float, snr_db: float) -> np.ndarray:
    r = _rng()
    x = x * np.exp(1j * 2 * np.pi * foff / _FS * np.arange(x.size))
    p = float(np.mean(np.abs(x) ** 2))
    npow = p / (10 ** (snr_db / 10))
    noise = np.sqrt(npow / 2) * (
        r.standard_normal(x.size) + 1j * r.standard_normal(x.size)
    )
    return (x + noise).astype(np.complex64)


def _psk(order: int, snr_db: float = 15.0, foff: float = _FOFF) -> np.ndarray:
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
