"""Numpy signal synthesis for round-trip tests.

Marconi is RX-only, so a test that needs a signal has to make one. These are
written from the textbook definition of each modulation, NOT lifted from the
engine — that is the point. When the generator was the engine's own tx path, a
demod bug and an encoder bug that agreed with each other produced BER 0 and the
sweep proved nothing. An independent generator cannot agree with a demod bug it
has never seen.

Two things are taken from GNU Radio rather than reimplemented (CLAUDE.md: don't
re-implement the wheel): constellation POINT TABLES and RRC filter TAPS. Both
are static reference data — a lookup table and a filter design — not modulation
logic, and hand-rolling either would be reproducing a standard badly. Every
modulation step itself is numpy here.

tests/unit/helpers/test_synth.py holds each generator to the property that
defines it, so a generator that drifts fails on its own terms rather than by
silently making some decode look good.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

Complex = npt.NDArray[np.complex64]


def _as_bits(bits: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if b.size and b.max() > 1:
        raise ValueError("bits must be 0/1")
    return b


def cpfsk(
    bits: npt.ArrayLike, *, sps: int, deviation: float, sample_rate: float
) -> Complex:
    """Continuous-phase binary FSK: antipodal symbols held for `sps` samples,
    integrated into phase. bit 1 => +deviation Hz, bit 0 => -deviation Hz.

    exp(j * 2*pi*deviation/rate * cumsum(+-1)) is the definition of CPFSK; the
    phase accumulates BEFORE the sample is emitted, which is what makes the
    first symbol land on the same grid a discriminator recovers it from.
    """
    if sps < 1:
        raise ValueError(f"sps must be >= 1, got {sps}")
    symbols = 2.0 * _as_bits(bits).astype(np.float64) - 1.0
    held = np.repeat(symbols, sps)
    sensitivity = 2.0 * np.pi * deviation / sample_rate
    return np.exp(1j * np.cumsum(sensitivity * held)).astype(np.complex64)


def fm_levels(
    levels: npt.ArrayLike, *, sps: int, deviation: float, sample_rate: float
) -> Complex:
    """Frequency-modulate an arbitrary level sequence: hold each level for `sps`
    samples and integrate into phase. The M-ary generalisation of cpfsk — a
    4-level sequence gives C4FM, a +/-1 sequence gives binary CPFSK."""
    if sps < 1:
        raise ValueError(f"sps must be >= 1, got {sps}")
    held = np.repeat(np.asarray(levels, dtype=np.float64).ravel(), sps)
    sensitivity = 2.0 * np.pi * deviation / sample_rate
    return np.exp(1j * np.cumsum(sensitivity * held)).astype(np.complex64)


def msk(
    bits: npt.ArrayLike, *, sps: int, symbol_rate: float, sample_rate: float
) -> Complex:
    """MSK is CPFSK at modulation index h = 0.5, i.e. deviation = Rs/4."""
    return cpfsk(bits, sps=sps, deviation=symbol_rate / 4.0, sample_rate=sample_rate)


def ook(bits: npt.ArrayLike, *, sps: int, amplitude: float = 1.0) -> Complex:
    """On-off keying: the bit itself is the envelope, held for `sps` samples,
    on the real axis. No pulse shaping — a rectangular chip is what an envelope
    detector and a fixed-threshold slicer are specified against."""
    if sps < 1:
        raise ValueError(f"sps must be >= 1, got {sps}")
    return (amplitude * np.repeat(_as_bits(bits).astype(np.float64), sps)).astype(
        np.complex64
    )


def ook_bursts(
    bursts: list[npt.ArrayLike],
    *,
    sps: int,
    gap_samples: int,
    lead_samples: int = 0,
    scales: list[float] | None = None,
) -> Complex:
    """Several OOK bursts separated by silence, each at its own amplitude.

    The per-burst scale is the reason this exists: a burst detector that
    references one global threshold passes a loud burst and drops a quiet one,
    and only a capture holding both can show it.
    """
    gains = scales if scales is not None else [1.0] * len(bursts)
    if len(gains) != len(bursts):
        raise ValueError("scales must be one per burst")
    gap = np.zeros(gap_samples, dtype=np.complex64)
    parts: list[Complex] = [np.zeros(lead_samples, dtype=np.complex64)]
    for payload, gain in zip(bursts, gains):
        parts.append(ook(payload, sps=sps, amplitude=gain))
        parts.append(gap)
    return np.concatenate(parts).astype(np.complex64)


def chirp_symbols(symbols: npt.ArrayLike, *, sf: int, oversample: int) -> Complex:
    """CSS: symbol s is the base up-chirp cyclically shifted by s of its 2^sf
    bins, which is exp(j*2*pi*s*t/N) applied to the sweep.

    base sweep: exp(j*2*pi*(t^2/(2N) - t/2)), t in samples/oversample. The -t/2
    centres the sweep on DC so the dechirped tone lands at bin s rather than
    s + N/2.
    """
    n = 1 << sf
    t = np.arange(oversample * n) / oversample
    base = np.exp(1j * 2 * np.pi * (t * t / (2 * n) - t / 2))
    syms = np.asarray(symbols, dtype=np.int64).ravel() % n
    return np.concatenate(
        [np.exp(1j * 2 * np.pi * int(s) * t / n) * base for s in syms]
    ).astype(np.complex64)


def chirp_preamble(
    *, sf: int, oversample: int, preamble_len: int, sfd_symbols: float
) -> Complex:
    """`preamble_len` up-chirps then `sfd_symbols` down-chirps (fractional
    allowed, truncated mid-symbol) — the geometry a chirp sync correlates for."""
    up = chirp_symbols([0], sf=sf, oversample=oversample)
    down = np.conj(up)
    sn = oversample * (1 << sf)
    whole = int(sfd_symbols)
    frac = int(round((sfd_symbols - whole) * sn))
    return np.concatenate(
        [np.tile(up, preamble_len), np.tile(down, whole), down[:frac]]
    ).astype(np.complex64)


def _gr_modules() -> tuple[Any, Any]:
    from gnuradio import digital
    from gnuradio import filter as gr_filter

    return digital, gr_filter


def constellation_points(scheme: str, order: int) -> npt.NDArray[np.complex128]:
    """The scheme's point table, MSB-first by index, taken from GNU Radio.

    A lookup table, not modulation: reproducing GR's Gray mapping by hand is
    exactly the wheel CLAUDE.md says not to rebuild, and a demod compared
    against a hand-rolled ordering would fail for the ordering rather than for
    the DSP.
    """
    digital, _ = _gr_modules()
    if scheme == "psk":
        builders = {
            2: digital.constellation_bpsk,
            4: digital.constellation_qpsk,
            8: digital.constellation_8psk,
        }
        if order not in builders:
            raise ValueError(f"unsupported psk order {order}")
        con = builders[order]()
    elif scheme == "qam":
        if order not in (16, 64):
            raise ValueError(f"unsupported qam order {order}")
        con = digital.qam.qam_constellation(
            constellation_points=order,
            differential=False,
            mod_code=digital.mod_codes.GRAY_CODE,
            large_ampls_to_corners=False,
        )
        con.normalize(digital.constellation.POWER_NORMALIZATION)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return np.asarray(con.points(), dtype=np.complex128)


def rrc_taps(
    *,
    sps: float,
    alpha: float = 0.35,
    span: int = 11,
    gain: float = 1.0,
    sample_rate: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Root-raised-cosine taps from firdes — a filter DESIGN, taken rather than
    rebuilt. Same argument shape the psk/qam demods use, so a synthesized
    waveform is matched-filtered by the same pulse it was shaped with."""
    _, gr_filter = _gr_modules()
    return np.asarray(
        gr_filter.firdes.root_raised_cosine(
            gain, sample_rate, sample_rate / sps, alpha, round(sps * span) + 1
        ),
        dtype=np.float64,
    )


def map_bits(
    bits: npt.ArrayLike, *, scheme: str, order: int
) -> npt.NDArray[np.complex128]:
    """Pack bits MSB-first into symbol indices and look each up in the point
    table — GR's pack_k_bits_bb + chunks_to_symbols_bc, in two numpy lines."""
    points = constellation_points(scheme, order)
    k = int(np.log2(order))
    b = _as_bits(bits)
    usable = (b.size // k) * k
    idx = b[:usable].reshape(-1, k) @ (1 << np.arange(k - 1, -1, -1))
    return points[idx]


def shaped_constellation(
    bits: npt.ArrayLike,
    *,
    scheme: str,
    order: int,
    sps: int,
    alpha: float = 0.35,
    span: int = 11,
) -> Complex:
    """Linear modulation: bits -> symbols -> upsample by sps -> RRC pulse shape.

    Upsampling by zero-insertion then filtering is what an interpolating FIR
    does; writing it out keeps the pulse shape visible in the test rather than
    buried behind an interpolation factor.
    """
    symbols = map_bits(bits, scheme=scheme, order=order)
    return shape_symbols(symbols, sps=sps, alpha=alpha, span=span)


def shape_symbols(
    symbols: npt.NDArray[np.complex128],
    *,
    sps: int,
    alpha: float = 0.35,
    span: int = 11,
) -> Complex:
    """RRC pulse-shape an existing symbol sequence at `sps`."""
    upsampled = np.zeros(symbols.size * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    taps = rrc_taps(sps=float(sps), alpha=alpha, span=span, gain=float(sps))
    # TRIMMED to n*sps, which is what an interpolating FIR emits: the extra
    # len(taps)-1 tail of a full linear convolution is a decaying transient
    # with no symbols under it, and feeding it to a timing loop measurably
    # broke lock (24044-sample input decoded 4971 of 6000 symbols at SER 0.92;
    # the same waveform trimmed decoded all 6000 at SER 0).
    shaped = np.convolve(upsampled, taps)[: symbols.size * sps]
    return shaped.astype(np.complex64)


def write(path: Path, x: Complex) -> Path:
    np.asarray(x, dtype=np.complex64).tofile(path)
    return path


# --- step-spec dispatch -------------------------------------------------
# Most round-trip tests describe their signal as the modem path that used to
# generate it. Mapping that path onto the generators above keeps every test's
# fixture declarative without re-deriving the same if-ladder in a dozen files.
# The SIGNAL still comes from the textbook generators; only the lookup lives
# here, so the oracle is still independent of the demod under test.


def _steps_key(steps: Sequence[Any]) -> tuple[str, ...]:
    return tuple(getattr(s, "conv", "") for s in steps)


def _find(steps: Sequence[Any], conv: str) -> Any:
    return next(s for s in steps if getattr(s, "conv", "") == conv)


def settle_ramp(n: int) -> npt.NDArray[np.complex128]:
    """Constant-modulus alternating +/-1: transition-rich for a Gardner TED and
    decorrelated from a random preamble, so it never false-peaks a correlator."""
    r = np.ones(int(n), dtype=np.complex128)
    r[1::2] = -1.0
    return r


def symbol_indices(bits: npt.ArrayLike, *, order: int) -> npt.NDArray[np.uint8]:
    """Bits packed MSB-first into k-bit symbol-index bytes — the u8 wire a
    demod's hard-decision output rides, not a waveform."""
    k = int(np.log2(order))
    b = _as_bits(bits)
    usable = (b.size // k) * k
    idx = b[:usable].reshape(-1, k) @ (1 << np.arange(k - 1, -1, -1))
    return idx.astype(np.uint8)


def from_steps(
    steps: Sequence[Any],
    bits: npt.ArrayLike,
    *,
    sample_rate: float,
    symbol_rate: float,
) -> Complex:
    """The waveform a modem path describes, synthesized in numpy.

    Raises on an unmapped path rather than guessing: a silently-wrong fixture
    is the failure mode this whole module exists to prevent.
    """
    convs = _steps_key(steps)
    sps = int(round(sample_rate / symbol_rate))
    b = _as_bits(bits)

    if "fsk" in convs:
        return cpfsk(
            b, sps=sps, deviation=_find(steps, "fsk").deviation, sample_rate=sample_rate
        )
    if "ook_envelope" in convs:
        return ook(b, sps=sps)
    for scheme, demod, demap in (
        ("psk", "psk_demod", "psk_demap"),
        ("qam", "qam_demod", "qam_demap"),
    ):
        if demod not in convs and demap not in convs:
            continue
        src = _find(steps, demap) if demap in convs else _find(steps, demod)
        order = int(src.order)
        # no demod stage (or a 1-sps sampler) means the path IS the symbol
        # sequence: constellation points, unshaped and unrepeated
        if demod not in convs or "sample_symbols" in convs:
            return np.asarray(
                map_bits(b, scheme=scheme, order=order), dtype=np.complex64
            )
        d = _find(steps, demod)
        symbols = map_bits(b, scheme=scheme, order=order)
        if "preamble_sync" in convs:
            # the acquisition stage prepends a settle ramp and a known training
            # sequence to the SYMBOL stream, before pulse shaping
            a = _find(steps, "preamble_sync")
            preamble = np.asarray(a.preamble_i, dtype=np.float64) + 1j * np.asarray(
                a.preamble_q, dtype=np.float64
            )
            symbols = np.concatenate(
                [settle_ramp(int(a.pad_symbols)), preamble, symbols]
            )
        return shape_symbols(
            symbols,
            sps=sps,
            alpha=getattr(d, "alpha", 0.35),
            span=getattr(d, "span", 11),
        )
    if "dechirp" in convs:
        d = _find(steps, "dechirp")
        sf, os_ = int(d.sf), int(d.oversample)
        usable = (b.size // sf) * sf
        raw = b[:usable].reshape(-1, sf) @ (1 << np.arange(sf - 1, -1, -1))
        payload = chirp_symbols([_gray(int(v)) for v in raw], sf=sf, oversample=os_)
        if "chirp_sync" not in convs:
            return payload
        c = _find(steps, "chirp_sync")
        return np.concatenate(
            [
                chirp_preamble(
                    sf=sf,
                    oversample=os_,
                    preamble_len=int(c.preamble_len),
                    sfd_symbols=float(c.sfd_symbols),
                ),
                payload,
            ]
        ).astype(np.complex64)
    raise NotImplementedError(
        f"no numpy generator for the path {convs}; add one to tests/helpers/"
        "synth.py rather than letting a test invent its own signal"
    )


def _gray(v: int) -> int:
    """CSS symbol mapping packs sf bits then Gray-encodes; the mapping is a
    convention of the caller's framing, which is why it lives in tests."""
    return v ^ (v >> 1)
