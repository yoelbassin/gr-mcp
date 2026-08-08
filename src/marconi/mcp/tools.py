from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend
from marconi.engine.compile.compiler import (
    CompiledPipeline,
    CompileError,
    compile_modem,
    compile_pipeline,
)
from marconi.engine.deadline import set_deadline
from marconi.engine.run import PipelineResult
from marconi.engine.run import run_rx as engine_run_rx
from marconi.engine.stages.base import SpecValidationError
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem, Symbolstream
from marconi.engine.types.params import ParamValue
from marconi.errors import classify_error
from marconi.mcp.boundary import tool_error_boundary
from marconi.mcp.streams import (
    _ITEM_DTYPES,
    _require_file,
    ensure_cf32,
    parse_bits,
    render_page,
)
from marconi.mcp.streams import stream_stats as _compute_stats
from marconi.mcp.vocab import ENVELOPE, stage_details, stage_index
from marconi.mcp.workspace import new_run_dir
from marconi.survey import channelize_to_file, survey_iq

_START_LEVELS = {"iq": Level.IQ, "symbols": Level.SYMBOLS, "bits": Level.BITS}
_DEFAULT_LEVEL = {"c": "iq", "b": "bits", "s": "symbols", "f": "symbols"}
_ITEM = {"c": ItemType.C, "b": ItemType.B, "s": ItemType.S, "f": ItemType.F}
_ITEM_BYTES: dict[str, int] = {k: np.dtype(v).itemsize for k, v in _ITEM_DTYPES.items()}


def _start_descriptor(item_type: str, level: str | None) -> Descriptor:
    if item_type not in _ITEM:
        raise ValueError(f"input_item_type must be one of {sorted(_ITEM)}")
    level_key = level if level is not None else _DEFAULT_LEVEL[item_type]
    if level_key not in _START_LEVELS:
        raise ValueError(f"input_level must be one of {sorted(_START_LEVELS)}")
    carrier = Carrier.SOFT if item_type == "f" else Carrier.HARD
    return Descriptor(_START_LEVELS[level_key], _ITEM[item_type], carrier)


def _trace_rows(modem: Modem, cp: CompiledPipeline) -> list[dict[str, object]]:
    labels = ["<start>"] + [f"{s.conv}[{i}]" for i, s in enumerate(modem.path)]
    return [
        {
            "after": label,
            "level": desc.level.value,
            "item_type": desc.item_type.value,
            "carrier": desc.carrier.value,
            "amplitude": desc.amplitude.value,
            "order": desc.order,
            "frame_len": desc.frame_len,
            "sample_rate": rate,
        }
        for label, desc, rate in zip(labels, cp.boundaries, cp.rates)
    ]


def _input_stream(path: Path, item_type: str) -> Bitstream | Symbolstream:
    if item_type not in ("b", "f", "s"):  # "l" is a paging-only sidecar type
        raise ValueError(
            "input_item_type must be one of ['b', 'f', 's'] with input_path"
        )
    _require_file(path)
    items = path.stat().st_size // _ITEM_BYTES[item_type]
    if item_type == "b":
        return Bitstream(path=path, num_bits=items)
    return Symbolstream(
        path=path, num_symbols=items, item_type=cast(Literal["s", "f"], item_type)
    )


def _stream_summary(result: PipelineResult) -> dict[str, object] | None:
    if result.bitstream is not None:
        return {
            "path": str(result.bitstream.path),
            "item_type": "b",
            "items": result.bitstream.num_bits,
        }
    if result.symbolstream is not None:
        return {
            "path": str(result.symbolstream.path),
            "item_type": result.symbolstream.item_type,
            "items": result.symbolstream.num_symbols,
        }
    return None


def describe_stages(
    family: str | None = None, stage: str | None = None
) -> dict[str, object]:
    """Marconi's stage vocabulary, generated live from the engine registry.

    With no arguments: a compact index of every stage (name, family, levels,
    directions) plus the modem-spec envelope and the level/item-type primer.
    Pass stage=<name> or family=<name> for full per-stage detail: input
    contracts (item type, carrier, amplitude, minimum samples-per-symbol) and
    the JSON schema of the stage's spec parameters. Compose a modem spec as
    {"symbol_rate": <float>, "path": [{"conv": <stage name>, ...params}]};
    check it with validate_modem before running."""
    if stage is not None:
        return {"stages": stage_details([stage]), "envelope": ENVELOPE}
    if family is not None:
        names = [str(e["name"]) for e in stage_index() if e["family"] == family]
        if not names:
            fams = sorted({str(e["family"]) for e in stage_index()})
            raise ValueError(f"unknown family {family!r}; known: {fams}")
        return {"stages": stage_details(names), "envelope": ENVELOPE}
    return {"stages": stage_index(), "envelope": ENVELOPE}


def validate_modem(
    spec: dict[str, Any],
    sample_rate: float,
    direction: str = "rx",
    input_item_type: str = "c",
    input_level: str | None = None,
) -> dict[str, object]:
    """Compile a modem spec without running it — the fast iteration loop.

    Returns {"valid": true, "trace": [...]} where each trace row shows the
    stream after each stage (level, item type, carrier hardness, amplitude
    statistic, symbol-alphabet order, frame length, sample rate), or
    {"valid": false, "errors": [...]} with structured compile errors: a
    failing spec is a normal result here, not an exception. direction is
    "rx" (decode) or "tx" (generate); input_item_type/input_level describe
    the entry stream (defaults: complex IQ)."""
    start = _start_descriptor(input_item_type, input_level)
    try:
        modem = Modem.from_spec(spec, step_models())
        cp = compile_pipeline(
            modem,
            stage_registry(),
            direction=direction,
            sample_rate=sample_rate,
            start=start,
            source_io={"path": "unused"},
            sink_io={"path": "unused"},
        )
    except SpecValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {"code": "invalid_argument", "message": i.message, "at": i.block_id}
                for i in exc.issues
            ],
        }
    except (CompileError, ValidationError, ValueError) as exc:
        code, message = classify_error(exc)
        return {"valid": False, "errors": [{"code": code, "message": message}]}
    return {"valid": True, "trace": _trace_rows(modem, cp)}


_MAX_INLINE_LIST = 512


def _cap_list(container: dict[str, Any], key: str, sidecar: Path | None = None) -> None:
    seq = container.get(key)
    if isinstance(seq, list) and len(seq) > _MAX_INLINE_LIST:
        container[f"{key}_total"] = len(seq)
        if sidecar is not None:
            # the full list must stay reachable: entries past the cap have no
            # other home, and a >512-frame carve needs every window offset
            np.asarray(seq, np.int64).tofile(sidecar)
            container[f"{key}_path"] = str(sidecar)
        container[key] = seq[:_MAX_INLINE_LIST]


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
    timeout: float = 180.0,
) -> dict[str, object]:
    """Decode: run a modem spec over an IQ capture (or an existing bit/symbol
    stream for coding-only paths) and return the full pipeline result.

    Pass exactly one of capture_path (raw IQ; capture_dtype one of
    cf32/ci16/ci8/cu8, converted once per capture into a shared cache if not
    cf32) or input_path (+ input_item_type b/s/f and optionally input_level).
    capture_offset/capture_samples (complex samples; 0 samples = to EOF)
    decode a bounded slice while streaming - the answer to a capture too
    large for one run. A soft 'f' stream's sign convention depends on its
    level: bits-level soft floats are LLRs where bit 1 = NEGATIVE;
    symbols-level soft floats (the default level for 'f' input, and the
    output of a path ending at a demod stage, e.g. bare fsk) are demod
    outputs where POSITIVE slices to bit 1. A path may instead end at a bare
    demod stage that lands on complex constellation symbols (item_type "c",
    a .cf32 file) rather than soft floats — psk_demod, sample_symbols, and
    the ofdm cell stages all land there; the final validate_modem trace
    row's item_type says which a given spec produces. Inspect a "c" stream
    with stream_stats(item_type="c") — its constant_modulus_ratio is the
    robust false-lock check, read exactly as that tool documents (a single
    ring means PSK/CPFSK; a few structured rings can mean a genuine QAM
    lock, not a false one; only an erratic/drifting ratio means the demod
    you picked doesn't match the signal). clusters=K's EVM/cluster fit
    corroborates once modulus says the lock is real. The result carries
    status, a "stream" summary {path, item_type, items} to page with read_stream,
    windows/marks (truncated to 512 entries when longer, with a *_total
    count and a *_path int64 sidecar holding the full list - page it with
    read_stream), per-block census, diagnostics, free-text "hints" (actionable
    tips for THIS path, e.g. a coherent-demod bit-polarity-ambiguity retry
    suggestion), and "quality": a conservative verdict (decoded / uncertain /
    no_signal) with the evidence behind it. Treat only verdict "decoded" as
    trustworthy output; "uncertain" means the evidence was absent or
    conflicting — read
    quality.rationale. A hard-decision demod that measures its own decision
    dominance (dechirp's argmax over the dechirped spectrum) feeds quality
    the same way a soft stream does, so a bare css path can earn "decoded"
    — and reads no_signal when the spec's chirp geometry does not match the
    capture. timeout is a hard wall-clock cap on the entire call —
    pre-scan, decode, coding, and quality-scoring all count against it, not
    just the GR pipeline — so a decode too large or slow to finish in time
    raises a [deadline_exceeded] error instead of running unbounded; window a
    large capture with capture_offset/capture_samples rather than only
    raising timeout (an exception: a deadline landing specifically while the
    GR pipeline itself is mid-run instead returns status="timeout" normally,
    not a raised error). Output streams live under ./marconi-runs/ and are
    not auto-cleaned; page with read_stream (or summarize with stream_stats)
    until you remove them."""
    if (capture_path is None) == (input_path is None):
        raise ValueError("pass exactly one of capture_path or input_path")
    if capture_offset < 0 or capture_samples < 0:
        raise ValueError("capture_offset and capture_samples must be >= 0")
    if capture_path is None and (capture_offset or capture_samples):
        raise ValueError("capture_offset/capture_samples apply to capture_path only")
    modem = Modem.from_spec(spec, step_models())
    run_dir = new_run_dir("rx")
    if capture_path is not None:
        # a first-time ci16/ci8/cu8 conversion is real work with no bound of
        # its own; set_deadline here too so it counts against timeout - the
        # min-nesting means engine_run_rx's own inner deadline can't extend it
        with set_deadline(timeout):
            src, offset, length = ensure_cf32(
                Path(capture_path),
                capture_dtype,
                offset=capture_offset,
                samples=capture_samples,
            )
            source_io: dict[str, ParamValue] = {"path": str(src)}
            if offset:
                source_io["offset"] = offset
            if length:
                source_io["length"] = length
            result = engine_run_rx(
                modem,
                stage_registry(),
                sample_rate=sample_rate,
                start=_start_descriptor("c", None),
                workdir=run_dir,
                source_io=source_io,
                timeout=timeout,
            )
    else:
        if input_item_type is None:
            raise ValueError("input_item_type is required with input_path")
        result = engine_run_rx(
            modem,
            stage_registry(),
            sample_rate=sample_rate,
            start=_start_descriptor(input_item_type, input_level),
            workdir=run_dir,
            input_stream=_input_stream(Path(cast(str, input_path)), input_item_type),
            timeout=timeout,
        )
    payload: dict[str, Any] = result.model_dump(mode="json")
    _cap_list(payload, "windows", run_dir / "windows.i64")
    _cap_list(payload, "marks", run_dir / "marks.i64")
    for diag in payload.get("diagnostics", []):
        if isinstance(diag, dict):
            _cap_list(diag, "marks")
    payload["stream"] = _stream_summary(result)
    return payload


def run_tx_tool(
    spec: dict[str, Any],
    sample_rate: float,
    bits: str | None = None,
    bits_path: str | None = None,
    out_path: str | None = None,
    timeout: float = 180.0,
) -> dict[str, object]:
    """Generate: render input bits through a modem spec (direction tx) into a
    complex64 IQ file — simulation output only, nothing is transmitted.

    Pass exactly one of bits (a '0'/'1' string) or bits_path (a file of one
    uint8 bit per byte). The path must compile in the tx direction; coding
    stages do not (the engine executes no tx-side coding), so supply bits
    that already carry any framing/encoding you need. Returns the IQ file
    path, sample count, and per-block census."""
    if (bits is None) == (bits_path is None):
        raise ValueError("pass exactly one of bits or bits_path")
    modem = Modem.from_spec(spec, step_models())
    run_dir = new_run_dir("tx")
    if bits is not None:
        bits_file = run_dir / "tx_bits.u8"
        parse_bits(bits).tofile(bits_file)
    else:
        bits_file = Path(cast(str, bits_path))
    out = Path(out_path) if out_path is not None else run_dir / "out.cf32"
    gr = compile_modem(
        modem,
        stage_registry(),
        direction="tx",
        sample_rate=sample_rate,
        start=Descriptor(Level.IQ, ItemType.C),
        source_io={"path": str(bits_file)},
        sink_io={"path": str(out)},
    )
    r = GnuRadioBackend().run_pipeline(gr, timeout=timeout)
    return {
        "status": r.status,
        "error": r.error,
        "iq_path": str(out),
        "num_samples": out.stat().st_size // 8 if out.is_file() else 0,
        "sample_rate": sample_rate,
        "census": [c.model_dump() for c in r.census],
    }


def read_stream(
    path: str, offset: int = 0, count: int = 4096, item_type: str | None = None
) -> dict[str, object]:
    """Page a decoded stream back as data you can parse directly.

    Bits (.u8) return as a '0'/'1' string — the files are unpacked and frames
    are rarely byte-aligned, so slice the string at any bit offset for your
    framing/CRC/field work. Hard symbols (.i16) return as ints. Soft floats
    (.f32) return as "values", whose sign convention depends on the stream's
    level (the final validate_modem trace row states it): bits-level soft
    streams are LLRs where bit 1 = NEGATIVE; symbols-level soft streams (a
    path ending at a demod stage, e.g. bare fsk) are demod outputs where
    POSITIVE slices to bit 1. Complex constellation symbols (.cf32, item_type
    "c" — the output of a path ending at a bare demod stage, e.g. psk_demod)
    return "real"/"imag" lists. Window/mark sidecars (.i64, from a truncated
    run_rx result) return as ints under "values". item_type b/s/f/l/c overrides
    suffix inference (required for suffix-less paths). Pages are capped at
    65536 items; use offset to walk longer streams. total_items reports the
    full stream length. Stream files live under ./marconi-runs/ (or
    $MARCONI_WORKSPACE) and persist until externally removed; a missing path
    returns a [not_found] error asking you to re-run the spec."""
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

    Returns total_items, the sampled item count (streams over 65536 items are
    read as evenly-strided chunks across the whole file), min/max/mean/std, and
    a histogram of {center, count} over the 0.5..99.5 percentile range. Pass
    clusters=K (1..16) to also fit up to K levels: sorted "centers", their
    "cluster_counts", and a paste-ready "levels" list for mfsk_soft_demap (ask
    for a power-of-two K to feed it). Levels with no support are dropped, so
    "levels" may be shorter than K — a short list means the stream has fewer
    real modes than you asked for. Read the histogram to judge modality. Bits
    (.u8) report only ones_fraction.

    item_type "c" (complex constellation symbols, e.g. the output of a path
    ending at a demod stage) returns a different shape: mean_magnitude,
    std_magnitude, and constant_modulus_ratio (std/mean of |z|), plus
    magnitude_histogram and phase_histogram in place of the single
    histogram. Read constant_modulus_ratio first: it is the robust
    false-lock check. A genuine PSK/CPFSK lock sits on one ring (ratio near
    0); a locked QAM constellation sits on a few fixed rings (moderate,
    structured); a false lock — e.g. an FSK/frequency-modulated signal
    forced through a PSK demod, or a demod that never fully settled — drifts
    across many radii and reads a high, erratic ratio, even when a K-cluster
    fit still happily returns K points. clusters=K (>=1) additionally fits a
    2-D constellation: sorted "clusters" (each {real, imag, magnitude,
    phase_deg, count}) and an overall "evm" (RMS error from each symbol to
    its nearest cluster center, normalized by the RMS cluster magnitude).
    Treat EVM and cluster balance as corroborating, not primary: once
    constant_modulus_ratio says the lock is real, tight balanced clusters at
    low EVM confirm which order K — but EVM by itself is not a safe
    false-lock check (a demod that partially loses lock mid-capture, e.g. a
    cycle slip, can read a misleadingly low EVM at the wrong K, or a
    misleadingly high one on a signal that IS genuinely locked, while
    constant_modulus_ratio — blind to phase/rotation — does not move). K is
    a request, not a measurement of the signal: asking for more clusters
    than the signal truly has over-splits real lobes into extra low-EVM
    sub-clusters, so a returned K-length "clusters" list alone is not
    confirmation of order K — compare EVM across a few K values (it drops
    sharply at the true order and plateaus past it) together with
    cluster-count balance. item_type b/s/f/l/c overrides suffix inference."""
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
) -> dict[str, object]:
    """Characterize a raw-IQ capture — pure DSP measurements, no interpretation.

    Runs before you know the modem: the pre-demod counterpart to stream_stats.
    Reads a bounded slice (capture_offset/capture_samples in complex samples,
    0 = to EOF; capture_dtype one of cf32/ci16/ci8/cu8, matching run_rx).

    SUB-BAND: pass center_hz (sub-band offset from the capture centre) and/or
    decim (>1) to characterize ONE channel of a wideband multi-signal capture
    instead of the whole span. The slice is frequency-shifted so center_hz lands
    at DC, low-passed, and decimated by decim BEFORE measurement, so every block
    below then describes that channel alone — the pre-demod way to confirm a
    channel is what you think before building a modem (there is no need to
    channelize through run_rx first; run_rx has no conditioned-IQ output). The
    reported sample_rate is the decimated rate (sample_rate/decim), and
    min/max_symbol_rate scale to it. bandwidth_hz is the channel filter passband
    width (default passes most of the decimated band); the aggregate whole-span
    survey (the default, center_hz=0 decim=1) is unchanged. Returns five
    measurement blocks, all raw numbers for you to judge:
    "spectrum" (a coarse two-sided PSD in psd_db — a shape to eyeball for
    extra carriers/sub-bands, kept small on purpose; bin i sits at
    freq_start_hz + i*freq_step_hz, so the axis is reconstructed, never
    shipped as a redundant ramp — plus energy-centroid offset, 99%-power
    occupied bandwidth, and peak; read spectral asymmetry off these
    yourself), "envelope"
    (constant-envelope ratio + kurtosis — you decide FSK vs PSK vs QAM;
    computed on a power-smoothed active portion of the window, gated at
    burst timescale (a windowed power envelope above a fraction of its own
    peak) rather than per-sample, so a sustained idle gap between bursts
    doesn't dilute it into looking amplitude-modulated, while genuine
    within-burst amplitude keying (OOK/ASK, which varies faster than a
    burst boundary) still registers as amplitude-modulated. A continuous
    signal has nothing to gate, so this is a no-op there, and the reading
    stays consistent whether you hand survey a whole bursty capture or a
    single isolated burst), "symbol_rate" (cyclostationary rate candidates_hz
    ranked by strengths that are RELATIVE — normalized to the single
    strongest line across the clock spectrum's amplitude and phase branches
    together (not the PSD "spectrum" block above, and not each branch's own
    peak), so even pure noise produces a top strength near 1.0: strengths
    rank the candidates, they are not a signal-present score. A candidate
    that looks like a low-order harmonic of some other, weaker-frequency
    periodicity (TDMA/burst repetition, a capture-chain artifact) is
    down-weighted, so strengths[0] is not always the largest raw line and can
    read well under 1.0 even for a confidently-detected signal; the true
    rate itself can never be mistaken for this comb's own fundamental. Each
    candidate also carries eye_openness (same index as candidates_hz and
    strengths): how cleanly a multi-level symbol eye opens when the capture
    is resampled at that rate — near zero for a smeared or wrong-rate guess,
    sharply higher when a clean symbol eye is genuinely present. candidates_hz
    already re-heads toward whichever candidate's eye CLEARLY opens
    (preferring the lower rate when more than one does, since a harmonic's
    eye can look clean too), falling back to the strength order when no eye
    clearly opens — so strengths[0] is not guaranteed to be the largest
    strengths value. eye_openness is a clean-eye diagnostic, not an
    authoritative rate finder: on noisy or pulse-shaped real off-air signals
    the eye can fail to clear its floor even at the true rate, in which case
    every candidate stays in cyclostationary-strength order, and that order
    is NOT trustworthy for constant-envelope/FSK-family signals — strengths
    ranks spectral lines and can seat a burst-harmonic, a TDMA slot-cadence
    line, or another capture-chain periodicity first. When the eye never
    clears, don't trust candidates_hz[0] alone; confirm the rate by
    demodulating your top few candidates through run_rx and comparing
    decoded-symbol cluster tightness with stream_stats. Treat candidates as
    ranked hypotheses, expect harmonics of the true rate among them, and note
    the estimate is least reliable for amplitude-null signals such as
    OOK/ASK with a carrier offset, and for short windows spanning only a few
    burst/TDMA cycles — prefer widening the window for bursty
    signals; narrow with min_symbol_rate/max_symbol_rate. The search band is
    reported as search_lo_hz/search_hi_hz: the default floor is what the
    analyzed span can actually resolve (a few clock-spectrum bins), so a
    longer slice searches lower by default, and clock_resolution_hz is that
    spectrum's bin width — candidates_hz values are quantized to it, so a
    candidate within one resolution bin of your hypothesis is a match, and a
    rate you expect below search_lo_hz needs an explicit min_symbol_rate),
    "inst_freq"
    (instantaneous-frequency histogram counts (bin i center =
    hist_start_hz + i*hist_step_hz), tone peaks_hz, and spread_hz —
    active-gated like envelope. peaks_hz only tells you the tone order when
    tones fully resolve: at low samples-per-symbol, closely-spaced M-ary FSK
    tones smear into one blurred lobe, so a LOW peak count does not mean
    few-ary or non-FSK — don't trust tone count alone. spread_hz (the
    standard deviation of the active instantaneous frequency) is ONE-SIDED
    evidence, not a detector: a NARROW spread relative to symbol_rate
    reliably rules out wideband frequency modulation, even when peaks_hz
    has collapsed to a single lobe — but a WIDE spread is AMBIGUOUS, since
    unshaped/sharp-edged PSK or QAM (abrupt phase steps, not true frequency
    change) and a plain noisy tone can read just as wide as genuine FSK. Do
    NOT conclude "frequency-modulated" from a wide spread_hz alone. When
    spread_hz is wide and envelope reads constant-envelope, don't guess the
    modulation family — run both an fsk demod and a psk/qam demod through
    run_rx and compare symbol-cluster tightness with stream_stats; the
    demod that produces cleaner clusters is the better hypothesis. Some
    capture chains mirror IQ spectrally, so if a constant-envelope signal's
    fsk/msk demod comes back no_signal, prepend the invert stage and retry
    before ruling out frequency modulation), and
    "bursts" (activity segments, duty_cycle, and dominant_period_samples
    from burst spacing — the TDMA cadence; segments is capped at 512
    entries, with a segments_total count when longer — survey writes no
    sidecar, so re-window with capture_offset/capture_samples to inspect a
    busier span. Each segment's [start, length] is in analyzed-stream samples;
    "capture_scale" {offset_samples, decim} maps them back to the original
    capture so you can re-decode one burst with a targeted slice —
    capture_offset = offset_samples + start*decim, capture_samples =
    length*decim). It never
    labels the modulation, never assembles or runs a spec, and characterizes the
    dominant signal in the slice; window in time for a multi-signal capture."""
    if capture_offset < 0 or capture_samples < 0:
        raise ValueError("capture_offset and capture_samples must be >= 0")
    if decim < 1:
        raise ValueError("decim must be >= 1")
    _require_file(Path(capture_path))
    src, offset, length = ensure_cf32(
        Path(capture_path),
        capture_dtype,
        offset=capture_offset,
        samples=capture_samples,
    )
    if center_hz != 0.0 or decim > 1:
        run_dir = new_run_dir("survey")
        try:
            channel = run_dir / "channel.cf32"
            _, out_rate = channelize_to_file(
                src,
                channel,
                sample_rate,
                center_hz=center_hz,
                decim=decim,
                offset=offset,
                length=length,
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
            src,
            sample_rate,
            offset=offset,
            length=length,
            min_symbol_rate=min_symbol_rate,
            max_symbol_rate=max_symbol_rate,
        )
    payload: dict[str, Any] = result.model_dump(mode="json")
    bursts = cast(dict, payload["bursts"])
    _cap_list(bursts, "segments")
    # map segment [start, length] back to original capture samples so a burst can
    # be re-decoded with a targeted slice: capture_offset = offset_samples +
    # start*decim, capture_samples = length*decim.
    bursts["capture_scale"] = {"offset_samples": offset, "decim": decim}
    return payload


TOOLS: dict[str, Callable[..., object]] = {
    "describe_stages": tool_error_boundary(describe_stages),
    "validate_modem": tool_error_boundary(validate_modem),
    "run_rx": tool_error_boundary(run_rx_tool),
    "run_tx": tool_error_boundary(run_tx_tool),
    "read_stream": tool_error_boundary(read_stream),
    "stream_stats": tool_error_boundary(stream_stats),
    "survey": tool_error_boundary(survey),
}
