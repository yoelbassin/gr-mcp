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

from pydantic_core import PydanticCustomError


def check_match_tolerance(max_errors: int, pattern_bits: int, *, field: str) -> None:
    """A correlator's error tolerance cannot exceed the pattern it matches: a
    Hamming distance over an m-bit word lives in [0, m]. Unbounded, the value
    reached _chance_valid_rate, whose sphere-volume sum iterates range(t+1) —
    at 2^40 that is a pure-Python loop with no deadline check in it, and the
    first term past the pattern length raises out of math.lgamma, so the agent
    got "expected a noninteger or positive integer, got 0.0" from inside the
    coding lane instead of a word about its own spec."""
    if max_errors >= pattern_bits:
        raise PydanticCustomError(
            "value_error",
            "{field}={max_errors} is not below the {pattern_bits}-bit pattern "
            "it tolerates errors in; a match within that many flips is every "
            "position in the stream",
            {
                "field": field,
                "max_errors": max_errors,
                "pattern_bits": pattern_bits,
            },
        )


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

# Iterative decoders: time scales linearly and the loop lives in the worker.
MAX_DECODER_ITERATIONS = 1000

# Successive-cancellation list width: the decoder holds `list_size` full paths.
MAX_LIST_SIZE = 32

# Dechirp's FFT is oversample * 2**sf * zero_pad long and one is allocated per
# symbol window, so the PRODUCT is the thing to bound — no single field's cap
# expresses it (sf 14 with oversample 8 and zero_pad 8 already lands here).
MAX_DECHIRP_FFT = 1 << 20

# A chirp TX prefix materializes (preamble_len + sfd) * oversample * 2**sf
# complex samples as one Python list on the way into vector_insert_c.
MAX_CHIRP_PREFIX_SAMPLES = 1 << 22
