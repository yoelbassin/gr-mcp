"""Retrievable interpretation manuals for the measurement tools.

The @tool docstrings are the only instructions the driving agent gets, and they
are loaded on EVERY session whether or not the tool is called — measured at 5.5k
tokens before the agent did anything, survey alone 2.2k. Most of that was not
needed to CALL survey; it was needed to READ a survey result, and only for the
blocks whose reading is ambiguous.

So the same split describe_stages already makes for the stage vocabulary: the
tool docstring carries what you need to call it and the one-line warning that
stops a misreading, and the reasoning behind that warning lives here, one topic
per measurement block, fetched when a result actually needs it.

Nothing here is optional detail. Every paragraph is a trap someone hit on real
off-air signals; a tool docstring that names a topic must resolve to one, which
tests/unit/mcp/test_explain.py enforces in both directions.
"""

from __future__ import annotations

from enum import StrEnum

from marconi.wire import Payload


class Topic(StrEnum):
    SURVEY_WINDOWING = "survey.windowing"
    SURVEY_SPECTRUM = "survey.spectrum"
    SURVEY_CARRIER = "survey.carrier"
    SURVEY_ENVELOPE = "survey.envelope"
    SURVEY_SYMBOL_RATE = "survey.symbol_rate"
    SURVEY_INST_FREQ = "survey.inst_freq"
    SURVEY_MULTICARRIER = "survey.multicarrier"
    SURVEY_BURSTS = "survey.bursts"
    RUN_RX_LEVELS = "run_rx.levels"
    RUN_RX_RESULT = "run_rx.result"
    RUN_RX_TRACE = "run_rx.trace"
    RUN_RX_QUALITY = "run_rx.quality"


_SUMMARY: dict[Topic, str] = {
    Topic.SURVEY_WINDOWING: (
        "which samples each block measured, and when the window moved itself"
    ),
    Topic.SURVEY_SPECTRUM: "reading psd_db and the occupied-bandwidth figures",
    Topic.SURVEY_CARRIER: (
        "what psk_order does and does NOT claim; offset method, ambiguity, ppm"
    ),
    Topic.SURVEY_ENVELOPE: "why the ratio is active-gated, and FSK vs PSK vs QAM",
    Topic.SURVEY_SYMBOL_RATE: (
        "strengths are relative, eye_confirmed, and what to do when the eye "
        "never clears"
    ),
    Topic.SURVEY_INST_FREQ: (
        "peaks_hz tone resolution, why spread_hz is ONE-SIDED, mirrored IQ"
    ),
    Topic.SURVEY_MULTICARRIER: (
        "blind OFDM geometry: what fft_len pins exactly and what it does not"
    ),
    Topic.SURVEY_BURSTS: "mapping a segment back onto the capture to re-decode it",
    Topic.RUN_RX_LEVELS: (
        "level tells two same-wire streams apart; the soft sign conventions"
    ),
    Topic.RUN_RX_RESULT: "every field the result carries, and the omission rule",
    Topic.RUN_RX_TRACE: "what trace=True taps, and finding where a path breaks",
    Topic.RUN_RX_QUALITY: "what decoded / uncertain / no_signal are worth",
}


_TEXT: dict[Topic, str] = {
    Topic.SURVEY_WINDOWING: """
"bursts" streams the whole slice; every other block measures ONE contiguous
window of at most 2^20 samples, relocated to the most active megasample when
the slice is larger — so those blocks can describe a different part of the
capture than you asked for.

analyzed_start (with analyzed_samples and span_samples) reports where it
landed, indexed from the slice origin, which is the same frame as
bursts.segments, so the two line up. analyzed_start > 0 means a strong emitter
pulled the window; re-window with capture_offset/capture_samples to measure a
span deliberately.
""",
    Topic.SURVEY_SPECTRUM: """
Coarse two-sided PSD in psd_db. Bin i sits at freq_start_hz + i*freq_step_hz —
the axis is derivable and is never shipped as hundreds of floats.

Also: center_offset_hz (the energy centroid), peak_offset_hz (the strongest
bin), and the 99%-power occupied band as occupied_lo_hz / occupied_hi_hz /
occupied_bw_hz. Extra carriers, sub-bands and spectral asymmetry are yours to
eyeball; this block never labels anything.
""",
    Topic.SURVEY_CARRIER: """
offset_hz is the carrier's offset from the slice centre.

method is "mpsk" (a clean order-M line, de-aliased against the spectrum
centroid and interpolated) or "spectral_centroid" (the robust fallback — but
with more than one emitter in the slice the centroid BLENDS them, so check the
spectrum block before trusting it as a single carrier's offset).
offset_ambiguous is true for either of two reasons: the M-fold offset
candidates could not be told apart and the centroid was reported instead, or
the centroid disagreed with the occupied band it came from (below).

off_center is true when the offset is a large fraction of the occupied
bandwidth: the signal sits off-channel, so a demod that assumes a carrier at DC
will miss. Re-run with center_hz=offset_hz. A known tuning error becomes
ppm = offset_hz / tuned_hz * 1e6.

The centroid is POWER-WEIGHTED, so a wide flat emitter carrying a gain tilt
across its band pulls it toward the loud side — far from where the emitter
actually sits. When method is "spectral_centroid" and the centroid lands more
than ~15% of the occupied bandwidth from the midpoint of occupied_lo_hz and
occupied_hi_hz, that midpoint is reported as offset_hz instead and
offset_ambiguous goes true: the two estimators disagree, and the tilt-immune
one wins. Measured on an off-air multicarrier ensemble (1.5 MHz wide, tuned
dead centre, ~7 dB of in-band tilt): the centroid read +458 kHz against a
+60 kHz band midpoint, and off_center claimed the capture was mistuned when it
was not. The spectrum block still publishes the raw centroid as
center_offset_hz, so both
numbers remain readable — an offset_ambiguous carrier block is telling you to
prefer the spectrum block's band edges over any single carrier figure.

psk_order (4 or 8, else null) is an M-fold PHASE-SYMMETRY line, claimed when
its score jumps clear of the order below it. It is NOT a modulation label:
QPSK/8-PSK AND a square QAM/APSK of that order share it, and the M-th-power
cannot separate them — read it WITH the envelope block (unimodal amplitude /
low kurtosis => PSK; multimodal => QAM).

Order 8 also covers a staggered quaternary alphabet (pi/4-shifted QPSK): its
4-fold line is displaced by half the symbol rate, so when the 4- and 8-fold
lines imply different offsets the claim goes to 8 and the true offset. 8-PSK vs
pi/4-quaternary is yours to separate (a differential decode at order 4 vs 8).

Order 2 is never claimed alone: squaring turns ANY carrier-bearing signal (OOK,
FSK, GMSK) into an order-2 line. A strong order-2 with a constant envelope is
BPSK; amplitude-modulated is OOK/ASK.

phase_concentration is the raw ln(N)-normalized order_2 / order_4 / order_8
strengths. ~1 at every order means no phase-coherent carrier — noise, or too
weak to lock.
""",
    Topic.SURVEY_ENVELOPE: """
const_envelope_ratio and amplitude_kurtosis; you decide FSK vs PSK vs QAM.

Computed on a power-smoothed ACTIVE portion, gated at burst timescale (windowed
power above a fraction of its own peak), so idle gaps between bursts do not
dilute it into looking amplitude-modulated, while genuine within-burst OOK/ASK
keying — faster than a burst boundary — still registers. A continuous signal
has nothing to gate, so the reading is consistent whether you hand it a whole
bursty capture or a single burst.
""",
    Topic.SURVEY_SYMBOL_RATE: """
Cyclostationary candidates_hz with strengths and eye_openness (same index).

strengths are RELATIVE — normalized to the single strongest line across the
clock spectrum's amplitude and phase branches together, not the PSD and not
each branch's own peak — so pure noise still tops out near 1.0. They RANK
candidates; they are not a signal-present score. A candidate that looks like a
low-order harmonic of a weaker periodicity (TDMA/burst repetition, a
capture-chain artifact) is down-weighted, so strengths[0] need not be the
largest raw line and can read well under 1.0 on a confident detection. The true
rate itself can never be mistaken for such a comb's fundamental.

eye_openness is how cleanly a multi-level symbol eye opens when the capture is
resampled at that rate. It is NOT near zero for every wrong guess: on a
constant-envelope signal a wrong rate still samples the modulation's own
frequency levels, so even a noise-frequency line can carry a sizable eye at
high SNR. candidates_hz therefore re-heads only among candidates that ALSO
carry real cyclostationary strength, toward whichever such candidate's eye
clearly opens, preferring the lower rate when several do (a harmonic's eye can
look clean too), else falls back to strength order — so candidates_hz order
need not match strengths order. eye_confirmed says which regime you are in:
true => candidates_hz[0] was promoted on its eye AND its strength together,
false => strength-ranked only.

WHEN THE EYE NEVER CLEARS: on noisy or pulse-shaped off-air signals the eye can
fail to clear even at the true rate (usual for constant-envelope signals, and
for pulse-shaped multi-level continuous-phase FM such as 4-level C4FM), and
strength order is NOT trustworthy there — it can seat a burst harmonic, a TDMA
slot-cadence line, or a capture-chain periodicity first. Do not trust
candidates_hz[0] alone: demodulate your top few candidates through run_rx and
compare decoded-symbol cluster tightness with stream_stats.

Treat candidates as ranked hypotheses and expect harmonics of the true rate
among them. The estimate is least reliable for amplitude-null signals (OOK/ASK
with a carrier offset) and for short windows spanning few burst/TDMA cycles —
widen the window for bursty signals, narrow with min/max_symbol_rate.

search_lo_hz / search_hi_hz report the searched band. The default floor is what
the analyzed span can resolve (a few clock-spectrum bins), so a longer slice
searches lower, and a rate below search_lo_hz needs an explicit
min_symbol_rate. clock_resolution_hz is that spectrum's bin width —
candidates_hz is quantized to it, so a candidate within one bin of your
hypothesis is a match.
""",
    Topic.SURVEY_INST_FREQ: """
Instantaneous-frequency histogram (hist_counts; bin i center = hist_start_hz +
i*hist_step_hz), tone peaks_hz, and spread_hz — active-gated like envelope.

peaks_hz reads tone order only when tones fully resolve: at low
samples-per-symbol, closely-spaced M-ary FSK tones smear into one lobe, so a
LOW peak count does not mean few-ary or non-FSK.

spread_hz (std of the active instantaneous frequency) is ONE-SIDED. Narrow
relative to symbol_rate reliably rules OUT wideband frequency modulation even
when peaks_hz has collapsed. Wide is AMBIGUOUS — sharp-edged PSK/QAM (abrupt
phase steps) or a noisy tone read just as wide as genuine FSK, so never
conclude "frequency-modulated" from a wide spread_hz alone.

Wide spread plus a constant envelope: run both an fsk and a psk/qam demod
through run_rx and let stream_stats cluster tightness decide.

Some capture chains mirror IQ spectrally. If a constant-envelope signal's
fsk/msk demod reads no_signal, prepend the invert stage and retry before ruling
out frequency modulation.
""",
    Topic.SURVEY_MULTICARRIER: """
ABSENT unless a cyclic prefix was found, and that absence is itself the
answer: a single-carrier signal has no CP structure to measure. Do not read a
missing block as "not measured".

An OFDM symbol repeats the tail of its useful part as its prefix, so the sample
autocorrelation carries a line at lag = fft_len that nothing in a
single-carrier signal produces. That line is sharp, so fft_len and
subcarrier_spacing_hz are EXACT.

cp_len, symbol_len and symbol_rate_hz are not. They come from the period of the
CP-product envelope, whose autocorrelation peak is as broad as the prefix that
makes it — measured off-air on a 504/2552 geometry, eleven consecutive lags sat
within 0.0007 of each other and the argmax landed a sample out. Treat them as
accurate to a few samples, and settle the remainder with
ofdm_frame_sync_probe: its cp_corr is sharp exactly where this is flat (0.99 at
the right cp_len against 0.20 one guess away), so a +-4 sweep converges in a
handful of calls.

cp_correlation is cp_len / symbol_len, the height of that line, and is the
independent check on the pair: if the reported cp_len over symbol_len is far
from it, the period estimate is the part that slipped. peak_ratio is the line
over the median of its own search band with the line's neighbourhood excised —
a null that does not contain the thing it judges.

The symbol_rate block is single-carrier cyclostationary and has nothing to say
here: it reports eye_confirmed false on OFDM, with candidates that are
artifacts of the frame cadence. symbol_rate_hz in THIS block is the OFDM symbol
rate (1 / symbol_len), which is not a subcarrier symbol rate and not what a
single-carrier demod means by the term.
""",
    Topic.SURVEY_BURSTS: """
Activity segments, duty_cycle, and dominant_period_samples from burst spacing
(the TDMA cadence).

The activity bar is referenced to the slice's own noise floor, taken from its
quietest stretches — between probes and, within a probe, from a genuine quiet
side under a loud emitter — so an emitter with ANY observable off-time does
not hide weaker transmissions elsewhere in the capture. An emitter that never
turns off masks weaker total-power activity physically: the bar then rides its
level, and when the raw envelope says constant carrier rather than noise, the
result reports duty_cycle near 1.0 with count 0 (always on, genuinely
continuous) — measured reliable for a constant-envelope carrier >= ~6 dB over
the noise floor; a weaker always-on carrier still reads duty 0.0, so treat a
low-SNR "dead" verdict with the spectrum block in hand. An always-on
amplitude-modulated emitter (QAM/OOK envelope) can
still read duty 0.0 — its envelope minima are indistinguishable from a quiet
band at sample scale — so read duty with the spectrum block, not alone.

segments is capped at 512 with a segments_total count and NO sidecar —
re-window with capture_offset/capture_samples to inspect a busier span, which
is the answer a truncated list should push toward anyway.

Each [start, length] is in SLICE samples, not the analyzed window (see
survey.windowing). "capture_scale" {offset_samples, decim} maps back onto the
original capture to re-decode one burst:
    capture_offset  = offset_samples + start * decim
    capture_samples = length * decim
""",
    Topic.RUN_RX_LEVELS: """
level is the rung the items sit on, and it is what tells two same-wire streams
apart. Read it before parsing anything.

  item_type "f" at level "symbols" — a per-symbol demod value; slice POSITIVE
                                     to bit 1.
  item_type "f" at level "bits"    — an LLR of the OPPOSITE convention;
                                     bit 1 is NEGATIVE.
  item_type "c" at level "symbols" — a complex constellation symbol.
  item_type "c" at level "iq"      — conditioned IQ, not symbols at all.

A path may therefore end at complex constellation symbols (psk_demod,
sample_symbols, and the ofdm cell stages land there). Inspect one with
stream_stats(item_type="c") and read its constant_modulus_ratio exactly as that
tool documents — it, not EVM, is the false-lock check.

A path may also end BEFORE any demod, at conditioned IQ: the same "c" .cf32
wire, distinguished only by level "iq" — e.g. channelize->agc with no demod
stage. The result "stream" is then the conditioned sub-band itself, to feed
back into survey (characterize the cleaned channel), into stream_stats, or into
your own soft work. This is how you extract a specific multi-stage conditioning
chain that survey's plain center_hz/decim cannot reproduce.
""",
    Topic.RUN_RX_RESULT: """
The result omits null-valued fields throughout (a census row lists only the
ports and measures its block has); "stream" and "soft_stream" alone stay
explicit when null, because their absence would read as "this run has no such
stream" rather than "this field was omitted".

  status        — ok / empty / error / timeout.
  stream        — {path, item_type, level, items}; page it with read_stream.
                  See run_rx.levels before parsing it.
  soft_stream   — {path, item_type "f", items, level, bit1_sign}: the demod tap
                  or soft seam the quality layer scored, for your own
                  soft-decision work (PPM/Manchester pair comparisons, soft
                  correlation, confidence-weighted CRC repair). Pass
                  item_type "f" to read_stream for a suffix-less path;
                  bit1_sign restates the convention for its level. When the
                  path itself ends soft this names the same file as "stream";
                  it is null when the run produced no soft stream (a hard-bit
                  coding-only run).
  windows/marks — an exact arithmetic tiling collapses to windows_ramp /
                  marks_ramp = {start, stride, count} with an empty inline
                  list; other lists over 512 entries truncate inline with a
                  windows_total / marks_total count and a windows_path /
                  marks_path int64 sidecar holding the full list, which you
                  page with read_stream.
  census        — per block, in pipeline order; a row's "kind" appears only
                  when its block id does not already start with it.
  diagnostics   — per-block counters and mark lists.
  hints         — free-text, actionable, specific to THIS path (e.g. a
                  coherent-demod bit-polarity retry).
  quality       — see run_rx.quality.
""",
    Topic.RUN_RX_TRACE: """
trace=True taps EVERY GR-segment stage's output to its own sidecar and returns
a "trace" list — one row per stage:

  {after "conv[i]", level, item_type, sample_rate, path, items, stats}

where "stats" is compact and per item type: c => mean_magnitude,
std_magnitude, constant_modulus_ratio; f/s => min, max, mean, std;
b => ones_fraction.

So one run shows WHERE a path breaks — a constellation gone from ring to points
at the demod, an eye closed after a resample — instead of rebuilding truncated
specs one stage at a time. Each row's path pages with read_stream/stream_stats
like any other stream.

Coding-tail stages (post-demod descramble/deinterleave/FEC) are not tapped: the
per-block census already reports their item counts.

The taps are full-length. Trace a windowed burst (capture_offset/
capture_samples), not a whole multi-second capture, or the sidecars grow large.
""",
    Topic.RUN_RX_QUALITY: """
A conservative verdict with the evidence behind it. Treat only "decoded" as
trustworthy.

  decoded    — at least one DECODE-grade positive and nothing negative.
  uncertain  — evidence absent, at chance, or conflicting. Read
               quality.rationale; it names which.
  no_signal  — a decode-grade negative that nothing of the same grade rebuts.

Evidence is tiered. DETECTION evidence (a burst mark, a bare demod's per-symbol
soft eye) proves a signal is PRESENT, which is not a claim about the decode: it
can never reach "decoded" on its own, and it does not rebut a decode-grade
negative either — it is reported alongside one, not against it.

Each producer is measured against its own chance model, and a producer with no
model stays silent rather than guessing. A zero-hit sync search over a stream
too short to have contained a hit reads untestable, not absent.

quality.margin is the parameter search's OBJECTIVE FUNCTION: the soft
stream's raw decision-margin statistic, reported even when it earns no
evidence, so two runs whose verdicts are both "uncertain" are still
rankable — sweep a parameter and keep the run with the higher margin.
Higher = cleaner decisions. A degenerate stream that fails the
polarity/whiteness gates ranks at zero, below any run that passed them.
It is NOT a verdict: a wrong rate that aliases into clean decisions ranks
high (no stream statistic can see that), and it is null when the stream
carries decisions rather than confidences.

A hard-decision demod that measures its own decision dominance (dechirp's
argmax over the dechirped spectrum) feeds quality like a soft stream, so a bare
css path can earn "decoded" — and reads no_signal when the spec's chirp
geometry does not match the capture.
""",
}


_untexted = sorted(
    t.value
    for t in Topic
    if t not in _TEXT or t not in _SUMMARY or not _TEXT[t].strip()
)
if _untexted:
    raise RuntimeError(f"explain topics with no summary/text: {_untexted}")


class TopicText(Payload):
    topic: str
    text: str


class ExplainPayload(Payload):
    """The index (topic -> one-line summary) or the requested sections, never
    both: an index call is choosing what to read, a topic call is reading it."""

    topics: dict[str, str] | None = None
    sections: list[TopicText] | None = None


def topic_index() -> dict[str, str]:
    return {t.value: _SUMMARY[t] for t in Topic}


def topic_sections(topic: str) -> list[TopicText]:
    """Exact topic, or every topic under a prefix ("survey" -> all survey.*),
    so an agent that wants the whole manual for a tool asks once."""
    exact = [t for t in Topic if t.value == topic]
    matched = exact or [t for t in Topic if t.value.startswith(f"{topic}.")]
    if not matched:
        raise ValueError(
            f"unknown topic {topic!r}; known: {sorted(t.value for t in Topic)} "
            "(a bare prefix like 'survey' returns all of its sections)"
        )
    return [TopicText(topic=t.value, text=_TEXT[t].strip()) for t in matched]
