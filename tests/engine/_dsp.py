from __future__ import annotations

from pathlib import Path

import numpy as np


def channel(
    iq_path: Path,
    out_path: Path,
    *,
    snr_db: float | None = None,
    cfo_hz: float = 0.0,
    sto: float = 0.0,
    sfo_ppm: float = 0.0,
    multipath: list[complex] | None = None,
    sample_rate: float = 1.0,
    seed: int = 0,
) -> Path:
    """Apply realistic impairments to an IQ file: a multipath/frequency-selective
    channel (FIR convolution with `multipath` taps), then fractional
    sample-timing offset (circular FFT delay), carrier frequency offset,
    sample-clock offset (resample), then power-calibrated AWGN. Order is
    MULTIPATH -> STO -> CFO -> SFO -> AWGN."""
    x = np.fromfile(iq_path, dtype=np.complex64).astype(np.complex128)
    rng = np.random.default_rng(seed)
    if multipath is not None:
        x = np.convolve(x, np.asarray(multipath, dtype=np.complex128))
    if sto:
        fr = np.fft.fftfreq(len(x))
        x = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * fr * sto))
    if cfo_hz:
        n = np.arange(len(x))
        x = x * np.exp(2j * np.pi * cfo_hz * n / sample_rate)
    if sfo_ppm:
        from scipy.signal import resample

        n_out = int(round(len(x) * (1.0 + sfo_ppm * 1e-6)))
        if n_out == len(x):
            raise ValueError(
                f"sfo_ppm={sfo_ppm} on {len(x)} samples is a silent no-op: "
                "same-length resample is identity. Drift needs "
                "|ppm| * n_samples >= ~1e6; use an exaggerated ppm for "
                "short bursts."
            )
        x = resample(x, n_out)
    if snr_db is not None:
        p = float(np.mean(np.abs(x) ** 2)) or 1.0
        amp = np.sqrt(p / (2 * 10 ** (snr_db / 10)))
        x = x + amp * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    x.astype(np.complex64).tofile(out_path)
    return out_path


def upconvert_widen(
    in_path: Path,
    out_path: Path,
    *,
    up: int,
    offset_hz: float,
    rate_wide: float,
    snr_db: float | None = None,
    seed: int = 0,
) -> Path:
    """Model a wideband capture: interpolate baseband IQ by `up` (to rate_wide)
    and place the signal at +offset_hz, optionally adding power-calibrated AWGN.
    This is the impairment `channelize` corrects — a signal off-centre in a wider
    capture."""
    from scipy.signal import resample_poly

    x = np.fromfile(in_path, dtype=np.complex64).astype(np.complex128)
    xw = resample_poly(x, up, 1)
    n = np.arange(len(xw))
    xw = xw * np.exp(2j * np.pi * offset_hz / rate_wide * n)
    if snr_db is not None:
        rng = np.random.default_rng(seed)
        p = float(np.mean(np.abs(xw) ** 2)) or 1.0
        amp = np.sqrt(p / (2 * 10 ** (snr_db / 10)))
        xw = xw + amp * (
            rng.standard_normal(len(xw)) + 1j * rng.standard_normal(len(xw))
        )
    xw.astype(np.complex64).tofile(out_path)
    return out_path


def write_bits(path: Path, bits: np.ndarray) -> Path:
    np.asarray(bits, dtype=np.uint8).tofile(path)
    return path


def read_bits(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint8)


def read_complex(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.complex64)


def nearest_syms(z: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    return np.argmin(np.abs(z[:, None] - pts[None, :]), axis=1)


def tx_sym_indices(bits: np.ndarray, k: int) -> np.ndarray:
    """Group bits into k-bit symbol indices, MSB first (matches pack_k_bits_bb)."""
    b = np.asarray(bits, dtype=int)[: (len(bits) // k) * k].reshape(-1, k)
    return b.dot(1 << np.arange(k)[::-1])


def resolved_ser(
    rx_sym: np.ndarray,
    tx_sym_idx: np.ndarray,
    points: np.ndarray,
    M: int,
    max_shift: int = 320,
    settle: int = 64,
) -> float:
    """Minimum symbol-error rate over the M carrier-phase rotations x timing
    shifts, after discarding a `settle` acquisition prefix. Coherent carrier
    recovery leaves an M-fold phase ambiguity (no preamble — acquisition is
    deferred); this resolves it the way aligned_ber resolves timing. SER 0
    <=> BER 0 for a Gray constellation."""
    if max_shift <= settle:
        raise ValueError(
            f"max_shift={max_shift} <= settle={settle}: rx is trimmed by "
            "settle, so realignment needs a shift past it - a smaller "
            "max_shift reads a clean decode as random"
        )
    best = 1.0
    roots = np.exp(2j * np.pi * np.arange(M) / M)
    rx_sym = rx_sym[settle:]
    for r in roots:
        rs = nearest_syms(rx_sym * r, points)
        for shift in range(min(max_shift, len(tx_sym_idx)) + 1):
            n = min(len(rs), len(tx_sym_idx) - shift)
            if n < len(rs) // 2:
                break
            ser = float(np.mean(rs[:n] != tx_sym_idx[shift : shift + n]))
            best = min(best, ser)
            if best == 0.0:
                return best
    return best


def resolved_ser_hard(
    rx_idx: np.ndarray,
    tx_sym_idx: np.ndarray,
    points: np.ndarray,
    settle: int = 1500,
    max_shift: int | None = None,
) -> float:
    """Minimum symbol-error rate for a HARD symbol-index stream (e.g. from
    constellation_receiver_cb) over QAM's 4-fold (90 deg) rotational symmetry
    x timing shifts, after discarding a `settle` acquisition prefix. A 90 deg
    rotation relabels each index via nearest-point on the rotated constellation;
    the true labelling is one of the four roots. The decision-directed receiver
    leaves this discrete ambiguity (no preamble — acquisition is deferred),
    resolved here the way resolved_ser resolves PSK's M-fold ambiguity."""
    pts = np.asarray(points)
    rx = np.asarray(rx_idx, dtype=int)[settle:]
    tx = np.asarray(tx_sym_idx, dtype=int)
    if max_shift is not None and max_shift <= settle:
        raise ValueError(
            f"max_shift={max_shift} <= settle={settle}: rx is trimmed by "
            "settle, so realignment needs a shift past it - a smaller "
            "max_shift reads a clean decode as random"
        )
    ms = settle + 700 if max_shift is None else max_shift
    best = 1.0
    for r in (1, 1j, -1, -1j):
        relabel = nearest_syms(pts * r, pts)  # index i under a rotation by r
        ri = relabel[rx]
        for shift in range(min(ms, len(tx)) + 1):
            n = min(len(ri), len(tx) - shift)
            if n < len(ri) // 2:
                break
            best = min(best, float(np.mean(ri[:n] != tx[shift : shift + n])))
            if best == 0.0:
                return best
    return best


def make_preamble(
    points: np.ndarray, n: int = 64, seed: int = 7
) -> tuple[list[float], list[float]]:
    """A random sync sequence drawn from the modulation constellation `points`
    so the carrier loop tracks it cleanly (an off-constellation preamble is
    mistracked by higher-order PSK -> a wrong phase lock). Returns (i_list,
    q_list)."""
    pts = np.asarray(points)
    s = pts[np.random.default_rng(seed).integers(0, len(pts), n)]
    return s.real.tolist(), s.imag.tolist()


def aligned_ber(rx: np.ndarray, tx: np.ndarray, max_shift: int = 256) -> float:
    """Minimum BER of tx against rx over integer shifts 0..max_shift in EITHER
    direction, requiring at least half of tx to overlap. Returns 0.0 on an exact
    (shifted) match.

    Two-sided because a resampler's group delay can place rx a few samples EARLY
    relative to tx (negative lag): rational_resampler_ccf reads a bit-perfect
    decode but lands rx ~2 samples ahead, which a forward-only search scores as
    ~0.46 random. Searching both directions is a superset of the forward-only
    search, so it never raises the score of an already-aligned path.

    WARNING (issue 05): the half-overlap floor launders systematic tail
    truncation — measured, not hypothetical. Requiring full coverage instead
    fails 19/30 phy round-trips: clock_correct emits 1260/1350 bits (90-bit
    tail lost, scored 0.0 today), and even the clean FSK/OOK paths truncate.
    Fixing this honestly needs a per-path tail-loss investigation (inherent
    streaming group-delay vs real bug) plus explicit bounded-tail assertions;
    tracked in the issue, deliberately not curve-fitted here."""
    rx = np.asarray(rx, dtype=np.uint8)
    tx = np.asarray(tx, dtype=np.uint8)
    floor = len(tx) // 2
    best = 1.0
    # (lead, lag) = (rx, tx): rx delayed; (tx, rx): rx advanced (negative lag).
    for lead, lag in ((rx, tx), (tx, rx)):
        for shift in range(min(max_shift, len(lead)) + 1):
            n = min(len(lead) - shift, len(lag))
            if n < floor:
                break
            best = min(best, float(np.mean(lead[shift : shift + n] != lag[:n])))
            if best == 0.0:
                return best
    return best
