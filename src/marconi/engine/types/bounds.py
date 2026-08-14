"""Upper bounds for step params whose COST scales with the value.

The rule, and the only reason these exist: a param that sizes an allocation, a
filter, or an iteration count is work the caller can ask for before anything
checks it. `ge=` alone made every one of them a spec that validates, compiles,
and then either allocates until the machine dies or grinds past the deadline
with nothing to show — measured across the tree at 2^40, twenty-odd fields
accepted it. Physical quantities (Hz, dB, ppm, ratios) are deliberately NOT
bounded here: a wrong one is cheap and the rate model already checks it.

Each cap sits orders of magnitude above the largest value any real spec in the
suite uses, so the bound catches a typo'd or fabricated number, never a working
recipe. test_param_bounds holds every cost-sizing field to one of them.
"""

from __future__ import annotations

import math

from pydantic_core import PydanticCustomError


def sample_rate_problem(rate: float, field: str = "sample_rate") -> str | None:
    """What is wrong with a caller-supplied sample rate, or None.

    Modem.symbol_rate is Field(gt=0); the quantity every rate check pairs it
    with had nothing, at any tool entry but capture. Non-finite is the one that
    bit: inf reached round() inside the compiler and surfaced as an
    OverflowError, which classifies internal_error — "stop and report a bug" —
    for a number the caller could have retyped."""
    if not math.isfinite(rate):
        return f"{field} must be a finite number, got {rate!r}"
    if rate <= 0:
        return f"{field} must be > 0, got {rate:g}"
    return None


def check_sample_rate(rate: float, field: str = "sample_rate") -> None:
    problem = sample_rate_problem(rate, field)
    if problem is not None:
        raise ValueError(problem)


def check_match_tolerance(max_errors: int, compared: int, *, field: str) -> None:
    """A correlator's error tolerance cannot reach the number of positions it
    actually COMPARES: a Hamming distance over m compared positions lives in
    [0, m].

    `compared` is not always the pattern's length. sync_symbols' 0 entries are
    wildcards its correlator never compares, and passing the whole length here
    let a 20-long pattern with two +-1 entries take max_errors=3 — the score bar
    became `agree >= -1`, so every offset matched (measured 4981 hits in 4981
    positions of pure noise, versus 1251 at max_errors=0). Callers pass the
    compared count, never len(pattern).

    Unbounded, the value also reached _chance_valid_rate, whose sphere-volume
    sum iterates range(t+1) — at 2^40 that is a pure-Python loop with no
    deadline check in it, and the first term past the pattern length raises out
    of math.lgamma, so the agent got "expected a noninteger or positive integer,
    got 0.0" from inside the coding lane instead of a word about its own spec."""
    if max_errors >= compared:
        raise PydanticCustomError(
            "value_error",
            "{field}={max_errors} is not below the {compared} pattern "
            "position(s) it is compared against; a match within that many "
            "flips is every position in the stream",
            {
                "field": field,
                "max_errors": max_errors,
                "compared": compared,
            },
        )


def nyquist_problem(center_hz: float, rate: float) -> str | None:
    """What is wrong with a caller's mixer offset, or None. ONE home for the
    rule: `translate` (a rotator) and `channelize` (a freq_xlating filter) both
    tune by mixing, so both wrap mod the sample rate, and each spelled this
    check out with its own byte-identical copy of the message."""
    if abs(center_hz) > 0.5 * rate:
        return (
            f"center_hz {center_hz:g} lies outside the +-{0.5 * rate:g} Hz "
            f"Nyquist span of the {rate:g} Hz input; a mixer wraps mod the "
            f"sample rate and would silently tune an aliased sub-band"
        )
    return None


def channelization_problem(
    *, rate: float, decim: int, bandwidth_hz: float, center_hz: float
) -> str | None:
    """What is wrong with a sub-band request, or None. ONE rule for the two
    channelizers: the `channelize` stage (a GR freq_xlating_fir_filter) and
    survey's own streaming numpy channelizer, which are different DSP and
    cannot share an implementation but must not disagree about what a caller
    may ask for. survey's used to validate `decim >= 1` and nothing else while
    the stage refused the same request three ways, under a docstring telling
    the agent both describe that channel alone."""
    if not 1 <= decim <= MAX_DECIM:
        return f"decim must be in [1, {MAX_DECIM}], got {decim}"
    off_band = nyquist_problem(center_hz, rate)
    if off_band is not None:
        return off_band
    if bandwidth_hz <= 0:
        return f"bandwidth_hz must be > 0, got {bandwidth_hz:g}"
    if bandwidth_hz < MIN_TRANSITION_FRAC * rate:
        return (
            f"bandwidth_hz {bandwidth_hz:g} is narrower than "
            f"{MIN_TRANSITION_FRAC:g} of the {rate:g} Hz input rate; the "
            f"anti-alias filter's tap count scales as rate/transition, so "
            f"this asks for a filter too long to run (the flowgraph would "
            f"reach its deadline with nothing to show). Decimate first, "
            f"then channelize the narrow band at the lower rate"
        )
    if bandwidth_hz > rate / decim:
        return (
            f"bandwidth_hz {bandwidth_hz:g} exceeds the decimated "
            f"output rate {rate / decim:g}; the passband folds after "
            f"decimation — reduce bandwidth_hz or decim"
        )
    return None


# firdes sizes the FIR as roughly rate/transition taps, so a transition tied
# 1:1 to a narrow passband is unbounded: 20 Hz on a 2.048 Msps capture asks for
# ~493k taps (~1e12 MACs over a 2 M-sample run) and compiles clean, then dies
# on the wall-clock deadline with no indication which parameter did it.
MIN_TRANSITION_FRAC = 1.0e-3

# FIR lengths a stage hands a backend to design. firdes/firwin cost scales with
# the tap count, and the taps are held in memory for the run's whole life.
MAX_FILTER_TAPS = 8192

# Rational-resample interpolation/decimation. The anti-imaging filter is sized
# by the interpolation factor, so a large one designs a huge filter inside the
# worker; the widest ratio any suite spec uses is 125/256.
MAX_RESAMPLE_FACTOR = 4096

# Integer decimation for channelize. The bandwidth checks already bound this
# transitively at compile (a passband must clear _MIN_TRANSITION_FRAC of the
# input rate AND fit the decimated rate, which caps decim near 1000), but a
# param that only fails two checks later cannot be read off describe_stages.
MAX_DECIM = 1024

# Polyphase branch counts: sub-sample timing resolution finer than this buys
# nothing, and the matched-filter bank is (flen x oversample) wide.
MAX_OVERSAMPLE = 64

# Sliding-window and framing lengths measured in symbols or items. Each becomes
# a buffer or a per-item stride in a block.
MAX_WINDOW_SYMBOLS = 1 << 20
MAX_FRAME_ITEMS = 1 << 20

# Delay-line depth in items: delay_cc allocates the whole line up front.
MAX_DELAY_ITEMS = 1 << 20

# RS parity width n-k: the reedsolo generator-polynomial build is O((n-k)^2)
# pure Python with no poll inside the library - measured 0.65 s at 2000 and
# 9.9 s at 8000 (clean 4x per doubling), and RsCodeStep(n=65535, k=1)
# validated instantly then ran 12.5 MINUTES past a 2 s deadline on empty
# input. The widest standardized RS codes carry ~64 parity symbols; 256
# builds in ~12 ms and leaves generous margin.
MAX_RS_PARITY_SYMBOLS = 256

# A sync correlation costs one full stream pass PER PATTERN BIT (and the
# diversity null costs up to another 64): measured 14.7 s for a 65,536-bit
# pattern over a 200k-bit stream, from a spec validate_modem called valid.
# The longest real-world sync words are a few hundred bits; 1024 is generous.
# This was invisible to the bounds sweep because its arms covered int/float
# and list fields only - a str-typed cost-sizing param had no arm at all.
MAX_SYNC_PATTERN_BITS = 1 << 10

# Iterative decoders: time scales linearly and the loop lives in the worker.
MAX_DECODER_ITERATIONS = 1000

# A convolutional generator polynomial's BIT LENGTH, not its value, is the cost:
# gnuradio.trellis.fsm builds a 2**(K-1)-state table from the widest poly, so
# one extra bit doubles the states and roughly quadruples the build. MEASURED
# through trellis.fsm(1, 2, [p, p|1]):
#   K=11    1024 states   0.04 s     62 MB
#   K=13    4096 states   1.5  s    543 MB
#   K=14    8192 states   6.9  s   2083 MB
#   K=15   16384 states  32    s   6197 MB
#   K=16   32768 states  163   s  11053 MB      (K=18 was OS-killed)
# The cap sits at 15 because that is the widest DEPLOYED convolutional code
# (the K=15 deep-space codes); everything else in use is K<=9. It cannot sit
# "orders of magnitude" clear of real usage the way the caps above do — the
# cost is exponential, so the only honest place for it is the edge of what
# real codes need. Note the top of the range is already expensive: K=15 spends
# 32 s and 6 GB before a single symbol is decoded.
#
# This one has to be caught at the spec boundary, not by the run deadline: the
# allocation happens inside the GR worker, and check_deadline() reads a
# contextvar that belongs to the parent process, so the parent can only reap a
# worker that has already taken the machine's memory.
MAX_POLY_BITS = 15

# Successive-cancellation list width: the decoder holds `list_size` full paths.
MAX_LIST_SIZE = 32

# Dechirp's FFT is oversample * 2**sf * zero_pad long and one is allocated per
# symbol window, so the PRODUCT is the thing to bound — no single field's cap
# expresses it (sf 14 with oversample 8 and zero_pad 8 already lands here).
MAX_DECHIRP_FFT = 1 << 20

# The chirp preamble span, (preamble_len + sfd) * oversample * 2**sf complex
# samples. chirp_sync scans and holds references across it, so the PRODUCT is
# the cost — no single field's cap expresses it.
MAX_CHIRP_PREFIX_SAMPLES = 1 << 22
