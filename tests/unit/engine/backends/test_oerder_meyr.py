from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from helpers._fakegr import FAKE_GR, drive

from marconi.engine.backends.gnuradio.embedded.oerder_meyr import (
    _MAX_REGION_SYMS,
    _cubic,
    estimate_tau,
    make_oerder_meyr,
)


def _rrc(sps: int, span: int = 11, beta: float = 0.35) -> np.ndarray:
    t = (np.arange(span * sps + 1) - span * sps / 2) / sps
    with np.errstate(all="ignore"):
        num = np.sin(np.pi * t * (1 - beta)) + 4 * beta * t * np.cos(
            np.pi * t * (1 + beta)
        )
        h = num / (np.pi * t * (1 - (4 * beta * t) ** 2))
    h[np.abs(t) < 1e-8] = 1 - beta + 4 * beta / np.pi
    return (h / np.sqrt(np.sum(h**2))).astype(float)


def _shaped_qpsk(nsym: int, sps: int, tau: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    syms = np.exp(1j * (np.pi / 2) * rng.integers(0, 4, nsym))
    up = np.zeros(nsym * sps, complex)
    up[::sps] = syms
    x = np.convolve(up, _rrc(sps), "same")
    n = np.arange(x.size)
    return (
        np.interp(n - tau * sps, n, x.real) + 1j * np.interp(n - tau * sps, n, x.imag)
    ).astype(np.complex64)


def _resample_symbols(samples: np.ndarray, sps: int, tau: float) -> np.ndarray:
    nsym = samples.size // sps
    instants = np.arange(nsym) * float(sps) + tau * sps
    return _cubic(samples.astype(np.complex128), instants).astype(np.complex64)


def test_estimate_tau_recovers_known_offset() -> None:
    for tau in (-0.3, 0.0, 0.2, 0.45):
        x = _shaped_qpsk(400, 8, tau)
        est = estimate_tau(x, 8)
        # wrap-aware error in [-0.5, 0.5)
        err = (est - tau + 0.5) % 1.0 - 0.5
        assert abs(err) < 0.05, (tau, est, err)


def test_resample_hits_the_symbols() -> None:
    x = _shaped_qpsk(200, 8, 0.0, seed=3)
    tau = estimate_tau(x, 8)
    sym = _resample_symbols(x, 8, tau)
    assert sym.size == x.size // 8
    r = abs(
        np.mean(np.exp(1j * 4 * np.angle(sym[np.abs(sym) > np.median(np.abs(sym))])))
    )
    assert r > 0.9, r  # clean 4-point QPSK constellation


def test_rejects_fractional_and_low_sps() -> None:
    with pytest.raises(ValueError, match="integer sps"):
        make_oerder_meyr(FAKE_GR, sps=4.5)
    with pytest.raises(ValueError, match="sps >= 4"):
        make_oerder_meyr(FAKE_GR, sps=2)


def _finality_probe(total: int) -> SimpleNamespace:
    return SimpleNamespace(expected_items=total, exhausted=lambda: True)


def _drive(x: np.ndarray, sps: float, chunk: int) -> np.ndarray:
    blk = make_oerder_meyr(FAKE_GR, sps=sps)
    blk.eof_probe = _finality_probe(x.size)
    return drive(blk, x.astype(np.complex64), chunk=chunk, out_dtype=np.complex64)


def test_block_decodes_a_burst_at_a_fractional_offset() -> None:
    x = _shaped_qpsk(600, 8, tau=0.3, seed=5)
    out = _drive(x, 8, chunk=x.size)
    assert abs(out.size - x.size // 8) <= 1
    z = out[np.abs(out) > np.median(np.abs(out))]
    r = abs(np.mean(np.exp(1j * 4 * np.angle(z))))
    assert r > 0.9, r


def test_output_is_chunk_independent() -> None:
    x = _shaped_qpsk(600, 8, tau=0.2, seed=6)
    whole = _drive(x, 8, chunk=x.size)
    pieced = _drive(x, 8, chunk=137)  # odd chunk, crosses blocks
    n = min(whole.size, pieced.size)
    assert np.allclose(whole[:n], pieced[:n], atol=1e-4), "chunking changed output"


def test_output_is_chunk_independent_across_gaps() -> None:
    # the hairy state machine paths — idle emission, rise/fall straddling a
    # block boundary, the quiet-counter carry — must all be chunk-blind too,
    # not just the single-burst steady state
    iq = _bursty(nbursts=8, L=80, gap=120, sto=0.37, sfo_ppm=25.0, snr=25)
    mf = np.convolve(iq.astype(complex), _RRC, "same").astype(np.complex64)
    whole = _drive(mf, _BURST_SPS, chunk=mf.size)
    pieced = _drive(mf, _BURST_SPS, chunk=137)
    n = min(whole.size, pieced.size)
    assert np.allclose(whole[:n], pieced[:n], atol=1e-4), "chunking changed output"


def test_withholds_the_uncapped_tail_without_a_finality_probe() -> None:
    x = _shaped_qpsk(600, 8, tau=0.0, seed=7)
    blk = make_oerder_meyr(FAKE_GR, sps=8)  # no eof_probe set
    out = drive(blk, x.astype(np.complex64), chunk=x.size, out_dtype=np.complex64)
    # without finality, only cap-committed chunks may emit: the open region's
    # residue past the last cap stays withheld (sim-pad rule). The exact count
    # pins that a flush-at-EOF-without-proof regression cannot pass.
    assert out.size == _MAX_REGION_SYMS, out.size


_BURST_SPS = 8
_RRC = _rrc(_BURST_SPS)


def _r4(z: np.ndarray) -> float:
    z = z[np.abs(z) > np.median(np.abs(z))]
    return 0.0 if z.size < 8 else float(abs(np.mean(np.exp(1j * 4 * np.angle(z)))))


def _bursty(
    nbursts: int,
    L: int,
    gap: int,
    sto: float,
    sfo_ppm: float,
    snr: float,
    seed: int = 11,
    lead: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seq: list[complex] = [0j] * lead
    for _ in range(nbursts):
        k = rng.integers(0, 4, L)
        seq.extend(np.exp(1j * np.cumsum(k) * (np.pi / 2)).tolist())
        seq.extend([0j] * gap)
    up = np.zeros(len(seq) * _BURST_SPS, complex)
    up[::_BURST_SPS] = np.asarray(seq)
    x = np.convolve(up, _RRC, "same")
    n = np.arange(x.size)
    warped = n * (1 + sfo_ppm * 1e-6) + sto
    xw = np.interp(warped, n, x.real) + 1j * np.interp(warped, n, x.imag)
    pw = np.mean(np.abs(xw[np.abs(xw) > 1e-6]) ** 2)
    noise = np.sqrt(pw / 10 ** (snr / 10) / 2) * (
        rng.standard_normal(xw.size) + 1j * rng.standard_normal(xw.size)
    )
    return (xw + noise).astype(np.complex64)


def _matched(iq: np.ndarray) -> np.ndarray:
    return np.convolve(iq.astype(complex), _RRC, "same").astype(np.complex64)


def _burst_recovery_r4(nbursts: int, length: int, gap: int) -> float:
    mf = _matched(_bursty(nbursts, length, gap, sto=0.37, sfo_ppm=25.0, snr=25))
    sym = _drive(mf, _BURST_SPS, chunk=mf.size)
    z = sym[1:] * np.conj(sym[:-1])
    return _r4(z)


def test_recovers_short_bursts() -> None:
    r4 = _burst_recovery_r4(nbursts=60, length=80, gap=120)
    assert r4 > 0.9, r4


def test_recovers_shorter_bursts() -> None:
    r4 = _burst_recovery_r4(nbursts=60, length=40, gap=80)
    assert r4 > 0.9, r4


def test_very_short_bursts_use_graded_edge_exclusion() -> None:
    # 20-symbol bursts sit below a full 2*span+interior exclusion; the graded
    # fallback must still drop as much RRC edge transient as the interior
    # floor allows instead of estimating tau over the whole (biased) burst
    mf = _matched(_bursty(60, 20, 60, sto=0.37, sfo_ppm=25.0, snr=25))
    blk = make_oerder_meyr(FAKE_GR, sps=_BURST_SPS)
    blk.eof_probe = _finality_probe(mf.size)
    sym = drive(blk, mf, chunk=mf.size, out_dtype=np.complex64)
    assert blk.diagnostics["bursts_partial_edge_exclusion"] > 0
    z = sym[1:] * np.conj(sym[:-1])
    assert _r4(z) > 0.85, _r4(z)


def test_flushes_a_long_continuous_region_without_finality() -> None:
    # a continuous signal is ONE active region that never falls; in a ratio-1
    # engine chain no finality probe arrives either, so it must flush on
    # LENGTH (the max_region cap), not only on fall-or-finality
    mf = _matched(_bursty(1, 2000, 0, sto=0.37, sfo_ppm=25.0, snr=25))
    blk = make_oerder_meyr(FAKE_GR, sps=_BURST_SPS)
    blk.eof_probe = None  # no finality ever arrives
    sym = drive(blk, mf, chunk=677, out_dtype=np.complex64)  # odd, multi-chunk

    nsym = mf.size // _BURST_SPS
    assert sym.size >= nsym - _MAX_REGION_SYMS, sym.size
    assert sym.size > 0

    z = sym[1:] * np.conj(sym[:-1])
    r4 = _r4(z)
    assert r4 > 0.9, r4


def test_gates_a_gap_larger_than_one_detection_block() -> None:
    # a gap spanning multiple whole _FLOOR_BLOCKs has chunks that are pure
    # noise; the persistent peak must keep reading them as idle (a per-block
    # peak reads a noise chunk's own top decile as "active")
    gap = 320  # symbols; 320*8=2560 samples, over 2 full _FLOOR_BLOCK chunks
    length = 80
    mf = _matched(_bursty(2, length, gap, sto=0.37, sfo_ppm=25.0, snr=25))
    sym = _drive(mf, _BURST_SPS, chunk=mf.size)

    # well inside the gap, clear of both burst edges' RRC spillover
    gap_region = sym[length + 50 : length + gap - 50]
    assert gap_region.size > 0
    max_gap_amp = float(np.max(np.abs(gap_region)))
    assert max_gap_amp == 0.0, max_gap_amp  # idle emits a literal zero


def test_burst_tail_grid_slots_are_zero() -> None:
    # the confirmed-quiet fall tail is buffered (it is how the fall is
    # detected) but its grid slots hold no burst energy: they must emit the
    # idle zero, not resampled noise with a random phase — no median gating
    # here, the assertion is on the raw emitted values
    length, gap = 100, 200
    mf = _matched(_bursty(1, length, gap, sto=0.2, sfo_ppm=0.0, snr=25))
    sym = _drive(mf, _BURST_SPS, chunk=mf.size)
    tail = sym[length + 4 :]
    assert tail.size > 0
    assert float(np.max(np.abs(tail))) == 0.0


def test_leading_noise_emits_zeros_not_symbols() -> None:
    # cold start: the peak seeds from the first block — pure noise on a
    # capture that starts before the transmitter keys up. The clock-line gate
    # must zero the noise promoted to "active", and later bursts must still
    # decode once the first real burst retrains the peak.
    rng = np.random.default_rng(21)
    lead_n = 9 * 1024
    bursts = _matched(_bursty(5, 80, 120, sto=0.37, sfo_ppm=25.0, snr=25, seed=13))
    scale = float(np.sqrt(np.mean(np.abs(bursts) ** 2))) / 18.0
    lead = scale * (
        rng.standard_normal(lead_n) + 1j * rng.standard_normal(lead_n)
    ).astype(np.complex64)
    x = np.concatenate([lead, bursts]).astype(np.complex64)
    sym = _drive(x, _BURST_SPS, chunk=x.size)

    # the leading-noise region, clear of the first-burst transition
    head = sym[10 : (lead_n // _BURST_SPS) - 64]
    assert head.size > 0
    assert float(np.max(np.abs(head))) == 0.0, "leading noise leaked symbols"

    # bursts after the first (the peak-retraining one) decode cleanly
    first_after = (lead_n + (80 + 120) * _BURST_SPS) // _BURST_SPS
    z = sym[first_after + 1 :] * np.conj(sym[first_after:-1])
    assert _r4(z) > 0.9, _r4(z)


def test_long_gap_decayed_peak_noise_is_zeroed() -> None:
    # a multi-second gap decays the tracked peak into the noise floor; the
    # noise then reads "active" again, but the clock-line gate keeps those
    # capped regions zeroed instead of resampling them into fake symbols
    rng = np.random.default_rng(31)
    burst = _matched(_bursty(1, 80, 20, sto=0.3, sfo_ppm=0.0, snr=15, seed=17))
    scale = float(np.sqrt(np.mean(np.abs(burst) ** 2))) / np.sqrt(10 ** (15 / 10))
    gap_n = 260_000
    gap = (
        scale
        / np.sqrt(2)
        * (rng.standard_normal(gap_n) + 1j * rng.standard_normal(gap_n))
    ).astype(np.complex64)
    x = np.concatenate([burst, gap, burst]).astype(np.complex64)
    sym = _drive(x, _BURST_SPS, chunk=x.size)

    gap_lo = (burst.size + 20 * 1024) // _BURST_SPS
    gap_hi = (burst.size + gap_n - 20 * 1024) // _BURST_SPS
    gap_region = sym[gap_lo:gap_hi]
    assert gap_region.size > 0
    assert float(np.max(np.abs(gap_region))) == 0.0, "decayed-peak noise leaked"

    # the second burst still decodes
    b2 = sym[(burst.size + gap_n) // _BURST_SPS :]
    z = b2[1:] * np.conj(b2[:-1])
    assert _r4(z) > 0.9, _r4(z)
