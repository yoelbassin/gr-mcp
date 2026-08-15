from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from marconi.capture import capture_budget_s, capture_iq
from marconi.deadline import set_deadline
from marconi.engine.compile.compiler import compile_pipeline
from marconi.engine.compile.errors import CompileError
from marconi.engine.run import composition_warnings
from marconi.engine.run import run_rx as engine_run_rx
from marconi.engine.stages.base import SpecValidationError
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem, Symbolstream
from marconi.errors import classify_error
from marconi.mcp.boundary import tool_error_boundary
from marconi.mcp.explain import ExplainPayload, topic_index, topic_sections
from marconi.mcp.payload import (
    ErrorRow,
    SpecTraceRow,
    pipeline_payload,
    spec_trace_rows,
    survey_payload,
)
from marconi.mcp.streams import ensure_cf32, render_page, require_file
from marconi.mcp.streams import stream_stats as _compute_stats
from marconi.mcp.vocab import (
    ENVELOPE,
    Envelope,
    StageEntry,
    family_names,
    stage_details,
    stage_index,
)
from marconi.mcp.workspace import discarded_if_unused, new_run_dir
from marconi.survey import channelize_to_file, survey_iq
from marconi.wire import Payload

# A pipeline is only ever entered at a stream the caller holds; AUDIO is a
# terminal rung nothing produces on disk, so it is excluded by name rather
# than by a hand-copied list that a new Level would silently fall out of.
_ENTRY_LEVELS: dict[str, Level] = {
    lv.value: lv for lv in Level if lv is not Level.AUDIO
}

# The rung an entry item type sits on when the caller does not say. Keyed by
# the enum so a new ItemType fails the completeness check below, not at
# runtime inside a tool call.
_DEFAULT_ENTRY_LEVEL: dict[ItemType, Level] = {
    ItemType.C: Level.IQ,
    ItemType.B: Level.BITS,
    ItemType.S: Level.SYMBOLS,
    ItemType.F: Level.SYMBOLS,
}

# Entry item types that name an existing stream file; "c" enters as a capture
# and "l" is a paging-only sidecar type, neither of which is a decode input.
_STREAM_ENTRY_TYPES: tuple[ItemType, ...] = (ItemType.B, ItemType.S, ItemType.F)


class StageVocabulary(Payload):
    """describe_stages' wire shape: the stage rows, plus the spec envelope and
    level/item-type primer on the index call only. An index call groups the
    rows by family, a detail call returns them flat.

    `omitted` names the stages a detail call dropped to stay inside its byte
    budget — absent when it returned everything asked for."""

    stages: dict[str, list[StageEntry]] | list[StageEntry]
    envelope: Envelope | None = None
    omitted: list[str] | None = None


class ValidateModemPayload(Payload):
    """validate_modem's wire shape. A failing spec is a normal result, so the
    errors ride the payload instead of raising."""

    valid: bool
    trace: list[SpecTraceRow] | None = None
    warnings: list[str] | None = None
    errors: list[ErrorRow] | None = None


_missing_default = sorted(t.value for t in ItemType if t not in _DEFAULT_ENTRY_LEVEL)
if _missing_default:
    raise RuntimeError(
        f"ItemType members with no default entry level: " f"{_missing_default}"
    )


def _start_descriptor(item_type: str, level: str | None) -> Descriptor:
    try:
        item = ItemType(item_type)
    except ValueError:
        raise ValueError(
            f"input_item_type must be one of {sorted(t.value for t in ItemType)}"
        ) from None
    if level is None:
        start_level = _DEFAULT_ENTRY_LEVEL[item]
    elif level in _ENTRY_LEVELS:
        start_level = _ENTRY_LEVELS[level]
    else:
        raise ValueError(f"input_level must be one of {sorted(_ENTRY_LEVELS)}")
    carrier = Carrier.SOFT if item is ItemType.F else Carrier.HARD
    return Descriptor(start_level, item, carrier)


def _input_stream(path: Path, item_type: str) -> Bitstream | Symbolstream:
    accepted = [t.value for t in _STREAM_ENTRY_TYPES]
    if item_type not in accepted:
        raise ValueError(f"input_item_type must be one of {accepted} with input_path")
    item = ItemType(item_type)
    require_file(path, "input stream")
    items = path.stat().st_size // item.item_bytes
    if item is ItemType.B:
        return Bitstream(path=path, num_bits=items)
    return Symbolstream(path=path, num_symbols=items, item_type=item.require_symbol())


def explain(topic: str | None = None) -> dict[str, object]:
    """How to READ a survey or run_rx result — the interpretation manual for
    each measurement block, fetched when a result needs it.

    No argument: the topic index, {name: one-line summary}. Pass a topic name
    for its section, or a bare tool prefix ("survey", "run_rx") for all of that
    tool's sections at once.

    Each section is the measured reasoning behind one block: what a figure does
    and does NOT claim, which readings are ambiguous, and what to do next when
    one is. The tool docstrings carry what you need to CALL a tool and the
    warning that stops a misreading; the reasoning behind that warning is
    here."""
    if topic is None:
        return ExplainPayload.build(topics=topic_index()).as_payload()
    return ExplainPayload.build(sections=topic_sections(topic)).as_payload()


def describe_stages(
    family: str | None = None, stage: str | None = None
) -> dict[str, object]:
    """Marconi's stage vocabulary, generated live from the engine registry.

    With no arguments: a compact index of every stage grouped by family —
    each row {name, levels: "<from>><to>", description?} — plus the
    modem-spec envelope and the level/item-type primer (index call only).
    Pass stage=<name> or family=<name> for full
    per-stage detail: input contracts (item type, carrier, amplitude,
    minimum samples-per-symbol) and the JSON schema of the stage's spec
    parameters. A contract that varies with the stage's own parameters is
    reported as null and listed in "step_conditional" — the stage's
    description states the rule (e.g. symbol_sync needs 4 samples per symbol
    open-loop and 2 closed-loop), and the compiler enforces the value for the
    step you actually write. A detail call is capped for size:
    "omitted" names any stages it dropped, which you fetch with stage=<name>.

    Compose a modem spec as {"symbol_rate": <float>, "path":
    [{"conv": <stage name>, ...params}]}; check it with validate_modem
    before running."""
    if stage is not None and family is not None:
        raise ValueError(
            "pass stage= or family=, not both; stage= names one stage, "
            "family= names a group of them"
        )
    if stage is not None:
        rows, omitted = stage_details([stage])
        return StageVocabulary.build(stages=rows, omitted=omitted or None).as_payload()
    if family is not None:
        rows, omitted = stage_details(family_names(family))
        return StageVocabulary.build(stages=rows, omitted=omitted or None).as_payload()
    return StageVocabulary.build(stages=stage_index(), envelope=ENVELOPE).as_payload()


def validate_modem(
    spec: dict[str, Any],
    sample_rate: float,
    input_item_type: str = "c",
    input_level: str | None = None,
    timeout: float = 60.0,
) -> dict[str, object]:
    """Compile a modem spec without running it — the fast iteration loop.

    Returns {"valid": true, "trace": [...]} where each trace row shows the
    stream after each stage (level, item type, carrier hardness, amplitude
    statistic, symbol-alphabet order, frame length, sample rate; null
    fields are omitted), or
    {"valid": false, "errors": [...]} with structured compile errors: a
    failing spec is a normal result here, not an exception.
    input_item_type/input_level describe the entry stream (defaults: complex
    IQ).

    A "warnings" list flags compositions that compile but silently misbehave
    (e.g. a window-seeding stage discarding an earlier seeder's windows) -
    fix or knowingly ignore. timeout caps the compile: a step model can do
    real work in its validator (a syndrome table, a codebook sweep)."""
    start = _start_descriptor(input_item_type, input_level)
    try:
        with set_deadline(timeout):
            modem = Modem.from_spec(spec, step_models())
            cp = compile_pipeline(
                modem,
                stage_registry(),
                sample_rate=sample_rate,
                start=start,
                source_io={"path": "unused"},
                sink_io={"path": "unused"},
            )
    except SpecValidationError as exc:
        code, _ = classify_error(exc)
        rows = [
            ErrorRow.build(code=code, message=i.message, at=i.block_id)
            for i in exc.issues
        ]
        return ValidateModemPayload.build(valid=False, errors=rows).as_payload()
    except (CompileError, ValidationError, ValueError) as exc:
        code, message = classify_error(exc)
        return ValidateModemPayload.build(
            valid=False, errors=[ErrorRow.build(code=code, message=message)]
        ).as_payload()
    return ValidateModemPayload.build(
        valid=True,
        trace=spec_trace_rows(modem, cp),
        warnings=composition_warnings(modem, stage_registry()),
    ).as_payload()


def run_rx_tool(
    spec: dict[str, Any],
    sample_rate: float,
    capture_path: str | None = None,
    capture_dtype: str = "cf32",
    capture_offset: int = 0,
    capture_samples: int = 0,
    input_path: str | None = None,
    input_item_type: str | None = None,
    input_level: str | None = None,
    trace: bool = False,
    timeout: float = 180.0,
) -> dict[str, object]:
    """Decode: run a modem spec over an IQ capture (or an existing bit/symbol
    stream for coding-only paths) and return the full pipeline result.

    Pass exactly one of capture_path (raw IQ; capture_dtype cf32/ci16/ci8/cu8;
    a non-cf32 capture converts into a shared cache keyed by capture_path AND
    the requested slice, so re-running one slice is free while each new
    capture_offset/capture_samples pair converts once; the cache is evicted
    least-recently-used against a disk budget, so sweeping many slices costs
    conversion time again, never unbounded disk) or input_path (+
    input_item_type b/s/f, optional input_level). capture_offset/
    capture_samples (complex samples; 0 samples = to EOF) decode a bounded
    slice while streaming — the answer to a capture too large for one run.

    trace=True taps every GR-segment stage's output to its own sidecar and adds
    a "trace" list, so one run shows WHERE a path breaks. Trace a windowed
    burst, not a whole multi-second capture — the taps are full-length.

    The result carries: status; "stream" {path, item_type, level, items} and
    "soft_stream"; windows/marks; per-block census; diagnostics; free-text
    "hints" actionable for THIS path; and "quality", a conservative verdict
    (decoded / uncertain / no_signal) with its evidence. Null-valued fields are
    omitted throughout, except "stream"/"soft_stream" which stay explicit.

    READ THIS BEFORE PARSING A STREAM: item_type alone does not identify it.
    "f" is a per-symbol demod value at level "symbols" and an LLR of the
    OPPOSITE sign convention at level "bits"; "c" is a constellation symbol at
    level "symbols" and conditioned IQ at level "iq". Check "level".
    TREAT ONLY quality.verdict "decoded" AS TRUSTWORTHY; "uncertain" means
    evidence absent or conflicting — read quality.rationale.
    Details: explain("run_rx") — sections run_rx.levels, run_rx.result,
    run_rx.trace, run_rx.quality.

    timeout is a hard wall-clock cap on the entire call — conversion, pre-scan,
    decode, coding and quality-scoring all count against it — so an oversized
    decode raises [deadline_exceeded] instead of running unbounded; window a
    large capture rather than only raising timeout. Exception: once the GR
    pipeline has started the deadline is reported, not raised — landing mid-run
    OR in the coding/quality tail after it delivered, the call returns
    status="timeout" with the evidence the run already had: census, diagnostics
    and any stream it finished decoding. Output streams live under
    $MARCONI_WORKSPACE (default ~/.cache/marconi) in marconi-runs/ and are not
    auto-cleaned; page with read_stream (or summarize with stream_stats) until
    you remove them."""
    if (capture_path is None) == (input_path is None):
        raise ValueError("pass exactly one of capture_path or input_path")
    if capture_offset < 0 or capture_samples < 0:
        raise ValueError("capture_offset and capture_samples must be >= 0")
    if capture_path is None and (capture_offset or capture_samples):
        raise ValueError("capture_offset/capture_samples apply to capture_path only")
    entry_args_passed = input_item_type is not None or input_level is not None
    if capture_path is not None and entry_args_passed:
        # the mirror: a capture enters as complex IQ by construction (its
        # on-disk format is capture_dtype), so these were not merely
        # inapplicable - they were DISCARDED, and a bits file passed as
        # capture_path decoded as IQ to status ok
        raise ValueError("input_item_type/input_level apply to input_path only")
    # the whole call, spec parse included: a step model can do real work in its
    # validator, and "hard cap on the entire call" has to mean the entire call
    with set_deadline(timeout), discarded_if_unused(new_run_dir("rx")) as run_dir:
        modem = Modem.from_spec(spec, step_models())
        if capture_path is not None:
            require_file(Path(capture_path), "capture")
            src_slice = ensure_cf32(
                Path(capture_path),
                capture_dtype,
                offset=capture_offset,
                samples=capture_samples,
            )
            result = engine_run_rx(
                modem,
                stage_registry(),
                sample_rate=sample_rate,
                start=_start_descriptor("c", None),
                workdir=run_dir,
                source=src_slice,
                trace=trace,
                timeout=timeout,
            )
        elif input_path is not None:
            if input_item_type is None:
                raise ValueError("input_item_type is required with input_path")
            result = engine_run_rx(
                modem,
                stage_registry(),
                sample_rate=sample_rate,
                start=_start_descriptor(input_item_type, input_level),
                workdir=run_dir,
                input_stream=_input_stream(Path(input_path), input_item_type),
                trace=trace,
                timeout=timeout,
            )
        else:
            raise ValueError("pass exactly one of capture_path or input_path")
        return pipeline_payload(result, run_dir)


def read_stream(
    path: str,
    offset: int = 0,
    count: int | None = None,
    item_type: str | None = None,
) -> dict[str, object]:
    """Page a decoded stream back as data you can parse directly.

    Bits (.u8) return as a '0'/'1' string — the files are unpacked and
    frames are rarely byte-aligned, so slice the string at any bit offset
    for your framing/CRC/field work. Hard symbols (.i16) return as ints;
    window/mark sidecars (.i64, from a truncated run_rx list) as ints under
    "values". Soft floats (.f32) return as "values" whose sign convention is
    per level (the final validate_modem trace row states it): bits-level =
    LLRs, bit 1 NEGATIVE; symbols-level (a path ending at a demod stage,
    e.g. bare fsk) slices POSITIVE to bit 1. Complex constellation symbols
    (.cf32, a path ending at a bare demod, e.g. psk_demod) return
    "real"/"imag" lists. Every float ships at 6 significant figures, and a
    NaN or infinite item ships as the string "nan"/"inf"/"-inf" (JSON has no
    number for either) with the page reporting non_finite_items, present
    only when some were found — they are the run's own product, so read its
    quality verdict rather than paging around them. item_type b/s/f/l/c
    overrides suffix inference (required for suffix-less paths). count
    defaults per item type to keep a page a few KB (b 4096, s 2048, f/l
    1024, c 256) and is capped per type (b 16384, s 6144, f 2304, l 2048,
    c 1152) — the ceilings differ because an item's WORST JSON cost does: a
    bit is 1 byte, an int16 up to 7, a float up to 20, an int64 up to 21, a
    complex pair up to 40, so a full page of any type stays under 48 KB even
    at extreme values. A count past the cap is clamped and the page reports
    capped_at ONLY when items were actually withheld — a page short of your
    count with no capped_at is the end of the stream; use offset to walk
    longer streams — total_items reports the full length. Stream files live
    under $MARCONI_WORKSPACE (default ~/.cache/marconi) in marconi-runs/
    until externally removed; a missing path returns a [not_found] error
    asking you to re-run the spec."""
    return render_page(Path(path), offset=offset, count=count, item_type=item_type)


def stream_stats(
    path: str,
    item_type: str | None = None,
    clusters: int = 0,
    bins: int = 41,
) -> dict[str, object]:
    """Summarize a decoded stream: distribution shape and, on request, its
    modulation levels — the fast way to size an mfsk_soft_demap or spot a
    closed eye without hand-rolling numpy.

    Returns total_items, the sampled count (streams over 65536 items are
    read as evenly-strided chunks spanning the file), min/max/mean/std, and
    — only when some were found — non_finite_items, the NaN/inf samples
    EXCLUDED from every statistic below it; a non-zero count means the
    numbers describe a subset, so check the run that wrote the stream, and
    a "histogram" {start, step, counts} over the 0.5..99.5 percentile range
    — bin i's center is start + i*step. clusters=K (1..16) also fits up to K
    levels: sorted "centers" with matching "cluster_counts". "centers" is
    paste-ready for mfsk_soft_demap only when K itself is a power of two (2,
    4, 8, 16): if fewer real modes exist than K, the fit re-runs at the
    largest supported power of two so the count still lands on one
    mfsk_soft_demap accepts; if every requested cluster carries real mass,
    it ships exactly K instead. A non-power-of-two K matched exactly by the
    data (asked 5, found 5 real modes) ships that count as-is, which
    mfsk_soft_demap's power-of-two-only levels validator REJECTS — request a
    power-of-two K when the target is mfsk_soft_demap. A single center
    means no multilevel structure to demap at all — read the histogram to
    judge modality. Bits (.u8) report only ones_fraction.

    item_type "c" (complex constellation symbols, e.g. a path ending at a
    bare demod stage) returns mean_magnitude, std_magnitude,
    constant_modulus_ratio (std/mean of |z|), and magnitude_histogram +
    phase_histogram in place of the single histogram. Read
    constant_modulus_ratio FIRST — it is the robust false-lock check: a
    genuine PSK/CPFSK lock sits on one ring (near 0), a locked QAM
    constellation on a few fixed rings (moderate, structured), and a false
    lock — an FSK signal forced through a PSK demod, a loop that never
    settled — drifts across many radii into a high, erratic ratio even when
    a K-cluster fit still returns K points. clusters=K adds a 2-D fit:
    sorted "clusters" ({real, imag, magnitude, phase_deg, count}) and "evm"
    (RMS error to the nearest center over RMS cluster magnitude). EVM and
    cluster balance corroborate, they are not primary: once
    constant_modulus_ratio says the lock is real, tight balanced clusters at
    low EVM confirm the order — but EVM alone is unsafe (a mid-capture cycle
    slip can read misleadingly low at the wrong K, or misleadingly high on a
    genuinely locked signal, while constant_modulus_ratio — blind to
    phase/rotation — does not move). K is a request, not a measurement:
    over-asking splits real lobes into extra low-EVM sub-clusters, so a
    K-length "clusters" list alone is not confirmation of order K — and EVM
    across K values is NOT the discriminator (measured: textbook BPSK at
    25 dB reads EVM 0.146 at K=4 vs 0.657 at K=2, 4.5x BETTER at the wrong
    order — a free K-point fit always lands on a lower-K cloud). The order
    test is CLUSTER BALANCE: the true order is the largest K whose clusters
    all carry comparable "count" mass; an over-asked fit starves or empties
    the surplus clusters. item_type b/s/f/l/c overrides suffix
    inference."""
    return _compute_stats(Path(path), item_type=item_type, clusters=clusters, bins=bins)


def survey(
    capture_path: str,
    sample_rate: float,
    capture_dtype: str = "cf32",
    capture_offset: int = 0,
    capture_samples: int = 0,
    min_symbol_rate: float | None = None,
    max_symbol_rate: float | None = None,
    center_hz: float = 0.0,
    decim: int = 1,
    bandwidth_hz: float | None = None,
    timeout: float = 180.0,
) -> dict[str, object]:
    """Characterize a raw-IQ capture — pure DSP measurements, no interpretation.
    Never labels the modulation and never runs a spec. Runs before you know the
    modem: the pre-demod counterpart to stream_stats.

    Reads a bounded slice (capture_offset/capture_samples in complex samples,
    0 = to EOF; capture_dtype cf32/ci16/ci8/cu8, as run_rx). Characterizes the
    DOMINANT signal in the slice — window in time for a multi-signal capture.

    SUB-BAND: center_hz (an OFFSET from the capture centre, never an absolute
    RF frequency) and/or decim>1 characterize ONE channel: the slice is shifted
    so center_hz lands at DC, low-passed (bandwidth_hz sets the passband,
    default most of the decimated band), and decimated BEFORE measurement, so
    every block below describes that channel alone. The reported sample_rate is
    the decimated one and min/max_symbol_rate scale to it. For a conditioning
    chain richer than one center/decim (channelize->invert->agc, say), end a
    run_rx path at conditioned IQ and survey the "c" stream it returns.

    Six measurement blocks, all raw numbers for you to judge:

    "spectrum"    — coarse two-sided PSD (psd_db; bin i at freq_start_hz +
                    i*freq_step_hz), centroid offset, 99% occupied bandwidth,
                    peak.
    "carrier"     — offset_hz, method, off_center, offset_ambiguous, psk_order,
                    phase_concentration.
    "envelope"    — const_envelope_ratio + amplitude_kurtosis, active-gated.
    "symbol_rate" — candidates_hz with strengths and eye_openness (same index),
                    eye_confirmed, search band, clock_resolution_hz.
    "inst_freq"   — frequency histogram, tone peaks_hz, spread_hz.
    "bursts"      — segments, duty_cycle, dominant_period_samples,
                    capture_scale.

    FOUR READINGS THAT MISLEAD IF TAKEN AT FACE VALUE — read the section named
    before concluding anything from them:
      psk_order is a phase-symmetry line, NOT a modulation label: QAM/APSK of
        the same order share it            -> explain("survey.carrier")
      strengths are RELATIVE, so noise tops out near 1.0; when eye_confirmed is
        false, candidate order is untrustworthy -> explain("survey.symbol_rate")
      spread_hz is ONE-SIDED: narrow rules FM out, wide proves nothing
                                           -> explain("survey.inst_freq")
      every block but "bursts" measures ONE window of at most 2^20 samples,
        relocated to the most active megasample; analyzed_start says where
                                           -> explain("survey.windowing")
    explain("survey") returns all sections, including how to map a burst
    segment back onto the capture (survey.bursts).

    timeout is a hard wall-clock cap on the whole call — conversion,
    channelization and every block count against it — so an oversized capture
    raises [deadline_exceeded]; window it rather than only raising timeout."""
    if capture_offset < 0 or capture_samples < 0:
        raise ValueError("capture_offset and capture_samples must be >= 0")
    if decim < 1:
        raise ValueError("decim must be >= 1")
    require_file(Path(capture_path), "capture")
    with set_deadline(timeout):
        src_slice = ensure_cf32(
            Path(capture_path),
            capture_dtype,
            offset=capture_offset,
            samples=capture_samples,
        )
        # bandwidth_hz alone must also route through the channelizer: on the
        # default path it was silently ignored (byte-identical payload with
        # and without it - even bandwidth_hz=1e-6, which the decim>1 path
        # rejects as too narrow)
        if center_hz != 0.0 or decim > 1 or bandwidth_hz is not None:
            run_dir = new_run_dir("survey")
            try:
                channel = run_dir / "channel.cf32"
                _, out_rate = channelize_to_file(
                    src_slice.path,
                    channel,
                    sample_rate,
                    center_hz=center_hz,
                    decim=decim,
                    offset=src_slice.offset,
                    length=src_slice.length,
                    bandwidth_hz=bandwidth_hz,
                )
                result = survey_iq(
                    channel,
                    out_rate,
                    min_symbol_rate=min_symbol_rate,
                    max_symbol_rate=max_symbol_rate,
                )
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)
        else:
            result = survey_iq(
                src_slice.path,
                sample_rate,
                offset=src_slice.offset,
                length=src_slice.length,
                min_symbol_rate=min_symbol_rate,
                max_symbol_rate=max_symbol_rate,
            )
    # NOT src_slice.offset: converted dtypes bake the slice into the cached
    # cf32 (offset 0) — the analyzed stream starts at capture_offset in the
    # original capture regardless of conversion
    return survey_payload(result, offset_samples=capture_offset, decim=decim)


def capture_tool(
    center_hz: float,
    sample_rate: float = 2.048e6,
    duration_s: float = 5.0,
    gain_db: float | None = None,
    ppm: float = 0.0,
    device: str = "",
    timeout: float | None = None,
) -> dict[str, object]:
    """Record a bounded raw-IQ capture from an attached SDR into a .cf32
    file — the live-hardware entry to the survey → validate_modem → run_rx
    loop.

    Tunes the first SoapySDR-enumerated device
    (device="driver=rtlsdr,serial=..." picks among several), applies gain_db
    (None = hardware AGC) and ppm correction, discards a short settle
    transient, and records duration_s seconds (max 300) at sample_rate under
    $MARCONI_WORKSPACE (default ~/.cache/marconi) in marconi-runs/. The
    returned sample_rate and center_hz are DEVICE READBACK — what the hardware
    delivered, not what was requested — and a
    warnings entry flags any divergence. Carry the returned SAMPLE_RATE into
    survey/run_rx; it is the same quantity there. The returned center_hz is
    the ABSOLUTE RF frequency the radio tuned, which is NOT what survey and
    the channelize stage mean by center_hz — theirs is an offset from the
    capture's own centre, so passing this value there is always out of range.
    To re-centre, survey the capture and use its carrier.offset_hz.

    "levels" is capture-chain health, not signal
    analysis: rms near zero means gain too low or no antenna; clip_fraction
    > 0 (samples at >= 0.99 full scale) means lower gain_db (a clearly
    railing or dead capture also warns); dc_offset is the usual SDR center
    spike — survey/channelize a sub-band away from DC rather than on it. The
    capture is raw — no DC removal, filtering, or resampling; conditioning
    belongs in the modem spec, and characterization (modulation, symbol
    rate, bursts) is survey's job on the returned path. status
    "error"/"timeout" with a path present is a partial capture whose samples
    are real and usable. "overflow_events" counts the buffer drops the SDR
    driver reported: > 0 — even with status "ok" — means the host could not
    keep up and the recording is TIME-SPLICED, complete in sample count but
    discontinuous, so duration, burst spacing and any timing measured across
    a gap are unreliable; lower sample_rate or shorten duration_s and
    re-capture (a warnings entry repeats the count). 0 is the driver
    reporting no drops. Iterate on ONE capture while forming hypotheses
    (same bits every run); re-capture only when you want fresh RF. Captures
    persist there until you remove them.

    timeout is a hard wall-clock cap on the whole call — probing the radio,
    settling and recording. It only TIGHTENS: the default is derived from
    duration_s, a value above it changes nothing, and one that cannot cover
    duration_s and the settle is refused. A radio that never answers
    enumerate/open/tune is killed and raises [deadline_exceeded] rather than
    hanging the call — capture waits at most 20 s for a radio to answer. A
    sample_rate no host could write, or a recording that would not fit the
    free disk, is refused before recording starts."""
    budget = capture_budget_s(duration_s, timeout)
    with set_deadline(budget), discarded_if_unused(new_run_dir("capture")) as run_dir:
        result = capture_iq(
            run_dir / "iq.cf32",
            center_hz=center_hz,
            sample_rate=sample_rate,
            duration_s=duration_s,
            gain_db=gain_db,
            ppm=ppm,
            device=device,
        )
        return result.as_payload()


TOOLS: dict[str, Callable[..., object]] = {
    "explain": tool_error_boundary(explain),
    "describe_stages": tool_error_boundary(describe_stages),
    "validate_modem": tool_error_boundary(validate_modem),
    "run_rx": tool_error_boundary(run_rx_tool),
    "read_stream": tool_error_boundary(read_stream),
    "stream_stats": tool_error_boundary(stream_stats),
    "survey": tool_error_boundary(survey),
    "capture": tool_error_boundary(capture_tool),
}
