"""Each generator held to the property that DEFINES its modulation.

These synths replaced the engine's tx path as the source of round-trip test
signals. That swap is only safe if the generators are checked against something
other than the demods that consume them — a generator validated by "the decode
worked" is the circularity the swap exists to remove.

So: no decode appears here. Each test measures the waveform directly for the
property its modulation is defined by (instantaneous frequency, envelope,
dechirped bin, matched-filter symbol recovery). Every one was also verified
sample-for-sample against the engine's tx path before that path was deleted;
these are what survive to keep them honest.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from helpers import _synth as synth

_RNG = np.random.default_rng(20260814)


def _inst_freq(x: np.ndarray, sample_rate: float) -> npt.NDArray[np.float64]:
    phase: npt.NDArray[np.float64] = np.angle(x[1:] * np.conj(x[:-1]))
    return phase / (2 * np.pi) * sample_rate


def test_cpfsk_holds_each_bit_at_its_own_tone() -> None:
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    sps, dev, rate = 16, 1000.0, 16000.0
    x = synth.cpfsk(bits, sps=sps, deviation=dev, sample_rate=rate)
    assert x.size == bits.size * sps
    f = _inst_freq(x, rate)
    # sample each symbol at its centre, away from the transition
    centres = f[sps // 2 :: sps][: bits.size]
    expected = np.where(bits.astype(bool), dev, -dev)
    assert np.allclose(centres, expected, rtol=1e-6), (centres, expected)


def test_cpfsk_envelope_is_constant_and_phase_continuous() -> None:
    bits = _RNG.integers(0, 2, 256).astype(np.uint8)
    x = synth.cpfsk(bits, sps=8, deviation=500.0, sample_rate=8000.0)
    assert np.allclose(np.abs(x), 1.0, atol=1e-6)
    # continuous phase: no sample-to-sample jump beyond the per-sample advance
    step = np.abs(np.angle(x[1:] * np.conj(x[:-1])))
    assert step.max() < np.pi / 2, step.max()


def test_msk_is_cpfsk_at_modulation_index_one_half() -> None:
    bits = _RNG.integers(0, 2, 64).astype(np.uint8)
    sps, rs, rate = 8, 1200.0, 9600.0
    got = synth.msk(bits, sps=sps, symbol_rate=rs, sample_rate=rate)
    want = synth.cpfsk(bits, sps=sps, deviation=rs / 4.0, sample_rate=rate)
    assert np.allclose(got, want)
    # h = 2*dev/Rs = 0.5
    f = _inst_freq(got, rate)[sps // 2 :: sps][: bits.size]
    assert np.allclose(np.abs(f), rs / 4.0, rtol=1e-6)


def test_ook_envelope_reproduces_the_bits() -> None:
    bits = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)
    sps = 5
    x = synth.ook(bits, sps=sps)
    assert x.size == bits.size * sps
    assert np.allclose(np.abs(x).reshape(-1, sps).mean(axis=1), bits)


def test_ook_bursts_keep_their_per_burst_amplitude_and_gaps() -> None:
    a = np.ones(4, dtype=np.uint8)
    b = np.ones(4, dtype=np.uint8)
    x = synth.ook_bursts(
        [a, b], sps=2, gap_samples=6, lead_samples=3, scales=[1.0, 0.25]
    )
    assert x.size == 3 + (8 + 6) * 2
    first = np.abs(x[3:11])
    second = np.abs(x[17:25])
    assert np.allclose(first, 1.0) and np.allclose(second, 0.25)
    assert np.allclose(np.abs(x[11:17]), 0.0), "gap must be silent"


@pytest.mark.parametrize("sf,oversample", [(7, 1), (7, 2), (9, 2)])
def test_a_chirp_symbol_dechirps_to_its_own_bin(sf: int, oversample: int) -> None:
    """The defining property of CSS: multiplying symbol s by the conjugate base
    sweep leaves a tone whose FFT peak sits at bin s."""
    n = 1 << sf
    symbols = [0, 1, 5, n // 2, n - 1]
    x = synth.chirp_symbols(symbols, sf=sf, oversample=oversample)
    base = synth.chirp_symbols([0], sf=sf, oversample=oversample)
    sn = oversample * n
    for i, s in enumerate(symbols):
        seg = x[i * sn : (i + 1) * sn] * np.conj(base)
        spec = np.abs(np.fft.fft(seg, n=sn))
        # oversampled: energy folds to bin s modulo n
        peak = int(np.argmax(spec)) % n
        assert peak == s, f"symbol {s} dechirped to bin {peak}"


def test_chirp_preamble_is_up_chirps_then_a_fractional_down_sweep() -> None:
    sf, os_, up_n, sfd = 7, 2, 4, 2.25
    sn = os_ * (1 << sf)
    pre = synth.chirp_preamble(
        sf=sf, oversample=os_, preamble_len=up_n, sfd_symbols=sfd
    )
    assert pre.size == sn * up_n + int(round(sfd * sn))
    up = synth.chirp_symbols([0], sf=sf, oversample=os_)
    for i in range(up_n):
        assert np.allclose(pre[i * sn : (i + 1) * sn], up, atol=1e-6)
    down = np.conj(up)
    assert np.allclose(pre[up_n * sn : (up_n + 1) * sn], down, atol=1e-6)


@pytest.mark.parametrize(
    "scheme,order", [("psk", 2), ("psk", 4), ("psk", 8), ("qam", 16), ("qam", 64)]
)
def test_map_bits_lands_on_the_scheme_point_table(scheme: str, order: int) -> None:
    k = int(np.log2(order))
    # enumerate every index rather than sampling: 200 random draws over 64
    # points leaves one uncollected about as often as not, and a coverage
    # assertion that flakes is worse than no coverage assertion
    indices = np.arange(order, dtype=np.uint8)
    bits = ((indices[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    syms = synth.map_bits(bits, scheme=scheme, order=order)
    points = synth.constellation_points(scheme, order)
    assert syms.size == order
    # index i must land on table entry i, and the table is fully exercised
    assert np.allclose(syms, points)
    assert np.unique(syms).size == order


@pytest.mark.parametrize("scheme,order", [("psk", 4), ("qam", 16)])
def test_matched_filtering_a_shaped_waveform_recovers_its_symbols(
    scheme: str, order: int
) -> None:
    """RRC is a root filter: shaping then matched-filtering gives a raised
    cosine, which is Nyquist — zero ISI at the symbol instants. Decimating at
    sps must return the symbols that went in."""
    k, sps = int(np.log2(order)), 4
    bits = _RNG.integers(0, 2, k * 300).astype(np.uint8)
    symbols = synth.map_bits(bits, scheme=scheme, order=order)
    x = synth.shaped_constellation(bits, scheme=scheme, order=order, sps=sps)
    taps = synth.rrc_taps(sps=float(sps), gain=1.0)
    matched = np.convolve(x.astype(np.complex128), taps)
    delay = len(taps) - 1  # two convolutions, each (len(taps)-1)//2
    # the shaped waveform is trimmed to n*sps (what an interpolating FIR
    # emits), so the final `span` symbols lose part of their pulse tail and
    # are legitimately not recoverable — compare the fully-supported ones
    span = 11
    keep = symbols.size - span
    recovered = matched[delay::sps][:keep]
    want = symbols[:keep]
    scale = np.vdot(recovered, want) / np.vdot(recovered, recovered)
    err = np.abs(recovered * scale - want).max() / np.abs(want).max()
    assert err < 0.02, f"ISI at the symbol instants: {err:.3f}"


def test_bits_must_be_binary() -> None:
    with pytest.raises(ValueError, match="0/1"):
        synth.cpfsk([0, 1, 7], sps=2, deviation=1.0, sample_rate=4.0)
