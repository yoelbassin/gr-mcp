from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import ValidationError

from marconi.capture import capture_iq
from marconi.engine.backends.gnuradio.runner import GnuRadioBackend
from marconi.engine.compile.compiler import (
    CompiledPipeline,
    CompileError,
    compile_modem,
    compile_pipeline,
)
from marconi.engine.deadline import set_deadline
from marconi.engine.run import PipelineResult, composition_warnings
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
from marconi.mcp.vocab import ENVELOPE, family_names, stage_details, stage_index
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
    rows = [
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
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


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


_SOFT_BIT1_SIGN = {"symbols": "positive", "bits": "negative"}


def _slim_census(rows: list[Any]) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        kind = row.get("kind")
        if isinstance(kind, str) and str(row.get("block", "")).startswith(kind):
            del row["kind"]


def _soft_summary(result: PipelineResult) -> dict[str, object] | None:
    if result.softstream is None:
        return None
    return {
        "path": str(result.softstream.path),
        "item_type": "f",
        "items": result.softstream.num_items,
        "level": result.softstream.level,
        "bit1_sign": _SOFT_BIT1_SIGN[result.softstream.level],
    }


_TRACE_STAT_KEYS: dict[str, tuple[str, ...]] = {
    "c": ("mean_magnitude", "std_magnitude", "constant_modulus_ratio"),
    "f": ("min", "max", "mean", "std"),
    "s": ("min", "max", "mean", "std"),
    "b": ("ones_fraction",),
}


def _trace_payload(result: PipelineResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for st in result.trace:
        row: dict[str, object] = {
            "after": st.after,
            "level": st.level,
            "item_type": st.item_type,
            "sample_rate": round(st.sample_rate, 3),
            "path": st.path,
            "items": st.items,
        }
        keys = _TRACE_STAT_KEYS.get(st.item_type)
        if st.items and keys:
            try:
                full = _compute_stats(Path(st.path), item_type=st.item_type, clusters=0)
            except (ValueError, FileNotFoundError):
                full = {}
            stat = {k: full[k] for k in keys if full.get(k) is not None}
            if stat:
                row["stats"] = stat
        rows.append(row)
    return rows


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

    With no arguments: a compact index of every stage grouped by family —
    each row {name, levels: "<from>><to>", dir: "rx"/"tx"/"rx,tx",
    description?} — plus the modem-spec envelope and the level/item-type
    primer (index call only). Pass stage=<name> or family=<name> for full
    per-stage detail: input contracts (item type, carrier, amplitude,
    minimum samples-per-symbol) and the JSON schema of the stage's spec
    parameters. Compose a modem spec as {"symbol_rate": <float>, "path":
    [{"conv": <stage name>, ...params}]}; check it with validate_modem
    before running."""
    if stage is not None:
        return {"stages": stage_details([stage])}
    if family is not None:
        return {"stages": stage_details(family_names(family))}
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
    statistic, symbol-alphabet order, frame length, sample rate; null
    fields are omitted), or
    {"valid": false, "errors": [...]} with structured compile errors: a
    failing spec is a normal result here, not an exception. direction is
    "rx" (decode) or "tx" (generate); input_item_type/input_level describe
    the entry stream (defaults: complex IQ).

    A "warnings" list flags compositions that compile but silently misbehave
    (e.g. a window-seeding stage discarding an earlier seeder's windows) -
    fix or knowingly ignore."""
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
    return {
        "valid": True,
        "trace": _trace_rows(modem, cp),
        "warnings": composition_warnings(modem, stage_registry()),
    }


_MAX_INLINE_LIST = 512
_RAMP_MIN_LEN = 64


def _as_ramp(seq: list[Any]) -> dict[str, int] | None:
    if len(seq) <= _RAMP_MIN_LEN or not all(isinstance(v, int) for v in seq):
        return None
    stride = seq[1] - seq[0]
    arr = np.asarray(seq, np.int64)
    if not bool(np.all(np.diff(arr) == stride)):
        return None
    return {"start": int(seq[0]), "stride": int(stride), "count": len(seq)}


def _cap_list(container: dict[str, Any], key: str, sidecar: Path | None = None) -> None:
    seq = container.get(key)
    if not isinstance(seq, list):
        return
    ramp = _as_ramp(seq)
    if ramp is not None:
        # a uniform tiling is fully derivable: ship the formula, not the list
        container[f"{key}_total"] = ramp["count"]
        container[f"{key}_ramp"] = ramp
        container[key] = []
        return
    if len(seq) > _MAX_INLINE_LIST:
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
    trace: bool = False,
    timeout: float = 180.0,
) -> dict[str, object]:
    """Decode: run a modem spec over an IQ capture (or an existing bit/symbol
    stream for coding-only paths) and return the full pipeline result.

    Pass exactly one of capture_path (raw IQ; capture_dtype one of
    cf32/ci16/ci8/cu8, non-cf32 converted once per capture into a shared
    cache) or input_path (+ input_item_type b/s/f, optional input_level).
    capture_offset/capture_samples (complex samples; 0 samples = to EOF)
    decode a bounded slice while streaming — the answer to a capture too
    large for one run.

    A soft 'f' stream's sign convention is per level: bits-level floats are
    LLRs where bit 1 = NEGATIVE; symbols-level floats (the default level for
    'f' input, and the output of a path ending at a demod stage, e.g. bare
    fsk) slice POSITIVE to bit 1. A path may instead end at complex
    constellation symbols (item_type "c", a .cf32 file — psk_demod,
    sample_symbols, and the ofdm cell stages land there); the final
    validate_modem trace row's item_type says which a spec produces. Inspect
    a "c" stream with stream_stats(item_type="c") and read its
    constant_modulus_ratio exactly as that tool documents — it, not EVM, is
    the false-lock check.

    A path may also end BEFORE any demod, at conditioned IQ (level "iq", the
    same "c" .cf32 wire) — e.g. channelize->agc with no demod stage. The
    result "stream" is then the conditioned sub-band itself, to feed back into
    survey (characterize the cleaned channel), stream_stats, or your own soft
    work; this is how you extract a specific multi-stage conditioning chain
    that survey's plain center_hz/decim cannot reproduce.

    trace=True taps EVERY GR-segment stage's output to its own sidecar and
    returns a "trace" list — one row per stage {after "conv[i]", level,
    item_type, sample_rate, path, items, and a compact "stats" (c:
    mean_magnitude/constant_modulus_ratio, f/s: min/max/mean/std, b:
    ones_fraction)} — so one run shows WHERE a path breaks (a constellation
    gone from ring to points at the demod, an eye closed after a resample)
    instead of rebuilding truncated specs. Each row's path pages with
    read_stream/stream_stats like any stream. Coding-tail stages (post-demod
    descramble/deinterleave/FEC) are not tapped — the per-block census already
    reports their item counts. The taps are full-length: trace a windowed
    burst (capture_offset/capture_samples), not a whole multi-second capture,
    or the sidecars grow large.

    The result omits null-valued fields throughout (a census row lists only
    the ports/measures its block has); "stream" and "soft_stream" alone stay
    explicit when null. It carries: status; "stream" {path, item_type,
    items} — page it with read_stream; "soft_stream" {path, item_type "f",
    items, level, bit1_sign} — the demod tap or soft seam the quality layer
    scored, for your own soft-decision work (PPM/Manchester pair
    comparisons, soft correlation, confidence-weighted CRC repair; pass
    item_type "f" to read_stream for a suffix-less path; bit1_sign restates
    the sign convention for its level). When the path itself ends soft,
    soft_stream names the same file as "stream"; it is null when the run
    produced no soft stream (a hard-bit coding-only run). windows/marks: an
    exact arithmetic tiling collapses to {key}_ramp = {start, stride, count}
    with an empty inline list; other lists over 512 entries truncate inline
    with a *_total count and a *_path int64 sidecar holding the full list —
    page it with read_stream. Also: per-block census (a row's kind appears
    only when its block id doesn't already start with it), diagnostics,
    free-text "hints" (actionable tips for THIS path, e.g. a coherent-demod
    bit-polarity retry), and "quality": a conservative verdict (decoded /
    uncertain / no_signal) with the evidence behind it. Treat only verdict
    "decoded" as trustworthy; "uncertain" means evidence absent or
    conflicting — read quality.rationale. A hard-decision demod that
    measures its own decision dominance (dechirp's argmax over the dechirped
    spectrum) feeds quality like a soft stream, so a bare css path can earn
    "decoded" — and reads no_signal when the spec's chirp geometry does not
    match the capture.

    timeout is a hard wall-clock cap on the entire call — conversion,
    pre-scan, decode, coding, and quality-scoring all count against it — so
    an oversized decode raises [deadline_exceeded] instead of running
    unbounded; window a large capture with capture_offset/capture_samples
    rather than only raising timeout. Exception: a deadline landing while
    the GR pipeline itself is mid-run returns status="timeout" normally, not
    a raised error. Output streams live under ./marconi-runs/ and are not
    auto-cleaned; page with read_stream (or summarize with stream_stats)
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
                trace=trace,
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
            trace=trace,
            timeout=timeout,
        )
    # null-valued fields are omitted product-wide (a census row lists only the
    # ports/measures its block has); stream/soft_stream stay explicit below
    payload: dict[str, Any] = result.model_dump(mode="json", exclude_none=True)
    for dup in ("bitstream", "symbolstream", "softstream"):
        payload.pop(dup, None)
    _slim_census(payload.get("census", []))
    _cap_list(payload, "windows", run_dir / "windows.i64")
    _cap_list(payload, "marks", run_dir / "marks.i64")
    for diag in payload.get("diagnostics", []):
        if isinstance(diag, dict):
            _cap_list(diag, "marks")
    payload.pop("trace", None)  # replace the model's raw trace with the enriched one
    if result.trace:
        payload["trace"] = _trace_payload(result)
    payload["stream"] = _stream_summary(result)
    payload["soft_stream"] = _soft_summary(result)
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
    payload: dict[str, object] = {
        "status": r.status,
        "iq_path": str(out),
        "num_samples": out.stat().st_size // 8 if out.is_file() else 0,
        "sample_rate": sample_rate,
        "census": [c.model_dump(exclude_none=True) for c in r.census],
    }
    _slim_census(cast(list[Any], payload["census"]))
    if r.error is not None:
        payload["error"] = r.error
    return payload


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
    "real"/"imag" lists. item_type b/s/f/l/c overrides suffix inference
    (required for suffix-less paths). count defaults per item type to keep a
    page a few KB (b 4096, s 2048, f/l 1024, c 256) and caps at 65536; use
    offset to walk longer streams — total_items reports the full length.
    Stream files live under ./marconi-runs/ (or $MARCONI_WORKSPACE) until
    externally removed; a missing path returns a [not_found] error asking
    you to re-run the spec."""
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
    a "histogram" {start, step, counts} over the 0.5..99.5 percentile range
    — bin i's center is start + i*step. clusters=K (1..16) also fits up to K
    levels: sorted "centers", "cluster_counts", and a paste-ready "levels"
    list for mfsk_soft_demap (ask for a power-of-two K to feed it). Levels
    with no support are dropped — a short list means fewer real modes than
    you asked for; read the histogram to judge modality. Bits (.u8) report
    only ones_fraction.

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
    K-length "clusters" list alone is not confirmation of order K — compare
    EVM across a few K values (it drops sharply at the true order and
    plateaus past it) together with cluster balance. item_type b/s/f/l/c
    overrides suffix inference."""
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

    Runs before you know the modem: the pre-demod counterpart to
    stream_stats. Reads a bounded slice (capture_offset/capture_samples in
    complex samples, 0 = to EOF; capture_dtype cf32/ci16/ci8/cu8, matching
    run_rx).

    SUB-BAND: center_hz (offset from the capture centre) and/or decim (>1)
    characterize ONE channel of a wideband capture: the slice is shifted so
    center_hz lands at DC, low-passed (bandwidth_hz sets the passband,
    default most of the decimated band), and decimated BEFORE measurement,
    so every block below describes that channel alone — the pre-demod way to
    confirm a channel before building a modem. (For a conditioning chain
    richer than one center/decim — channelize->invert->agc, say — end a
    run_rx path at conditioned IQ and survey the "c" stream it returns.) The
    reported sample_rate is the decimated rate; min/max_symbol_rate scale to
    it. The whole-span default (center_hz=0, decim=1) is unchanged.

    Returns five measurement blocks, all raw numbers for you to judge:

    "spectrum" — coarse two-sided PSD (psd_db; bin i sits at freq_start_hz +
    i*freq_step_hz, the axis is derivable, never shipped), energy-centroid
    offset, 99%-power occupied bandwidth, and peak; eyeball extra
    carriers/sub-bands and spectral asymmetry yourself.

    "envelope" — constant-envelope ratio + kurtosis; you decide FSK vs PSK
    vs QAM. Computed on a power-smoothed active portion gated at burst
    timescale (windowed power above a fraction of its own peak), so idle
    gaps between bursts don't dilute it into looking amplitude-modulated
    while genuine within-burst OOK/ASK keying (faster than a burst boundary)
    still registers; a continuous signal has nothing to gate, and the
    reading is consistent for a whole bursty capture or a single burst.

    "symbol_rate" — cyclostationary candidates_hz with strengths and
    eye_openness (same index). strengths are RELATIVE — normalized to the
    single strongest line across the clock spectrum's amplitude and phase
    branches together (not the PSD above, not each branch's own peak), so
    pure noise still tops out near 1.0: they rank candidates, they are not a
    signal-present score. A candidate that looks like a low-order harmonic
    of a weaker periodicity (TDMA/burst repetition, a capture-chain
    artifact) is down-weighted, so strengths[0] need not be the largest raw
    line and can read well under 1.0 on a confident detection (the true rate
    itself can never be mistaken for such a comb's fundamental).
    eye_openness is how cleanly a multi-level symbol eye opens when the
    capture is resampled at that rate — near zero for a smeared/wrong guess,
    sharply higher for a genuine eye. candidates_hz re-heads toward
    whichever candidate's eye CLEARLY opens (preferring the lower rate when
    several do — a harmonic's eye can look clean too), else falls back to
    strength order, so candidates_hz order need not match strengths order;
    the boolean eye_confirmed says which regime you are in: true →
    candidates_hz[0] is eye-corroborated, false → strength-ranked only. On
    noisy or pulse-shaped off-air signals the eye can fail to clear even at
    the true rate (usual for constant-envelope GMSK/C4FM), and strength
    order is NOT trustworthy there — it can seat a burst harmonic, a TDMA
    slot-cadence line, or a capture-chain periodicity first. When the eye
    never clears, don't trust candidates_hz[0] alone: demodulate your top
    few candidates through run_rx and compare decoded-symbol cluster
    tightness with stream_stats. Treat candidates as ranked hypotheses and
    expect harmonics of the true rate among them; the estimate is least
    reliable for amplitude-null signals (OOK/ASK with a carrier offset) and
    short windows spanning few burst/TDMA cycles — widen the window for
    bursty signals, narrow with min/max_symbol_rate. search_lo_hz/
    search_hi_hz report the searched band; the default floor is what the
    analyzed span can resolve (a few clock-spectrum bins), so a longer slice
    searches lower, and a rate below search_lo_hz needs an explicit
    min_symbol_rate. clock_resolution_hz is that spectrum's bin width —
    candidates_hz is quantized to it, so a candidate within one bin of your
    hypothesis is a match.

    "inst_freq" — instantaneous-frequency histogram (hist_counts; bin i
    center = hist_start_hz + i*hist_step_hz), tone peaks_hz, and spread_hz —
    active-gated like envelope. peaks_hz reads tone order only when tones
    fully resolve: at low samples-per-symbol, closely-spaced M-ary FSK tones
    smear into one lobe, so a LOW peak count does not mean few-ary or
    non-FSK. spread_hz (std of the active instantaneous frequency) is
    ONE-SIDED: narrow relative to symbol_rate reliably rules OUT wideband
    frequency modulation even when peaks_hz has collapsed, but wide is
    AMBIGUOUS — sharp-edged PSK/QAM (abrupt phase steps) or a noisy tone
    read just as wide as genuine FSK, so never conclude
    "frequency-modulated" from a wide spread_hz alone. Wide spread + a
    constant envelope: run both an fsk and a psk/qam demod through run_rx
    and let stream_stats cluster tightness decide. Some capture chains
    mirror IQ spectrally — if a constant-envelope signal's fsk/msk demod
    reads no_signal, prepend the invert stage and retry before ruling out
    frequency modulation.

    "bursts" — activity segments, duty_cycle, and dominant_period_samples
    from burst spacing (the TDMA cadence). segments is capped at 512 with a
    segments_total count and no sidecar — re-window with capture_offset/
    capture_samples to inspect a busier span. Each [start, length] is in
    analyzed-stream samples; "capture_scale" {offset_samples, decim} maps
    back to the original capture (capture_offset = offset_samples +
    start*decim, capture_samples = length*decim) to re-decode one burst.

    Never labels the modulation, never assembles or runs a spec;
    characterizes the dominant signal in the slice — window in time for a
    multi-signal capture."""
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


def capture_tool(
    center_hz: float,
    sample_rate: float = 2.048e6,
    duration_s: float = 5.0,
    gain_db: float | None = None,
    ppm: float = 0.0,
    device: str = "",
) -> dict[str, object]:
    """Record a bounded raw-IQ capture from an attached SDR into a .cf32
    file — the live-hardware entry to the survey → validate_modem → run_rx
    loop.

    Tunes the first SoapySDR-enumerated device
    (device="driver=rtlsdr,serial=..." picks among several), applies gain_db
    (None = hardware AGC) and ppm correction, discards a short settle
    transient, and records duration_s seconds (max 300) at sample_rate under
    ./marconi-runs/. The returned sample_rate and center_hz are DEVICE
    READBACK — what the hardware delivered, not what was requested — always
    carry the RETURNED values into survey/run_rx; a warnings entry flags any
    divergence, and a frequency the device cannot place inside the captured
    span raises instead. "levels" is capture-chain health, not signal
    analysis: rms near zero means gain too low or no antenna; clip_fraction
    > 0 (samples at >= 0.99 full scale) means lower gain_db (a clearly
    railing or dead capture also warns); dc_offset is the usual SDR center
    spike — survey/channelize a sub-band away from DC rather than on it. The
    capture is raw — no DC removal, filtering, or resampling; conditioning
    belongs in the modem spec, and characterization (modulation, symbol
    rate, bursts) is survey's job on the returned path. status
    "error"/"timeout" with a path present is a partial capture whose samples
    are real and usable. Iterate on ONE capture while forming hypotheses
    (same bits every run); re-capture only when you want fresh RF. Captures
    persist under ./marconi-runs/ until you remove them."""
    run_dir = new_run_dir("capture")
    try:
        result = capture_iq(
            run_dir / "iq.cf32",
            center_hz=center_hz,
            sample_rate=sample_rate,
            duration_s=duration_s,
            gain_db=gain_db,
            ppm=ppm,
            device=device,
        )
    except Exception:
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise
    return result.model_dump(mode="json")


TOOLS: dict[str, Callable[..., object]] = {
    "describe_stages": tool_error_boundary(describe_stages),
    "validate_modem": tool_error_boundary(validate_modem),
    "run_rx": tool_error_boundary(run_rx_tool),
    "run_tx": tool_error_boundary(run_tx_tool),
    "read_stream": tool_error_boundary(read_stream),
    "stream_stats": tool_error_boundary(stream_stats),
    "survey": tool_error_boundary(survey),
    "capture": tool_error_boundary(capture_tool),
}
