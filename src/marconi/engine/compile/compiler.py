from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.program import CodingProgram
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.stages.base import (
    CodingStage,
    SpecValidationError,
    Stage,
    validate_path,
)
from marconi.engine.stages.conditioning import agc_modes_for
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.models import Modem, ValidationIssue
from marconi.engine.types.params import ParamValue
from marconi.engine.types.step import Step, stage_label

# item_type -> (source_kind, sink_kind). The GR wire type alone selects IO;
# carrier is decision-hardness (a seam invariant), never a routing knob. Descriptor
# data, never stage names: replaces v1's rate-by-name scan and sink-by-type lookup.
_IO_BLOCKS: dict[ItemType, tuple[str | None, str]] = {
    ItemType.C: ("iq_file_source", "iq_file_sink"),
    ItemType.S: (None, "symbols_file_sink"),
    ItemType.B: ("bits_file_source", "bits_file_sink"),
    ItemType.F: ("soft_bits_file_source", "soft_bits_file_sink"),
}


def _io_kinds(desc: Descriptor) -> tuple[str | None, str]:
    if desc.item_type not in _IO_BLOCKS:
        raise CompileError(
            f"no source/sink block for item_type {desc.item_type.value!r}"
        )
    return _IO_BLOCKS[desc.item_type]


_TRACE_SUFFIX: dict[ItemType, str] = {
    ItemType.C: ".cf32",
    ItemType.F: ".f32",
    ItemType.S: ".i16",
    ItemType.B: ".u8",
}


def trace_sink_path(
    trace_dir: Path, index: int, conv: str, item_type: ItemType
) -> Path:
    """Where the trace tap for GR stage `index` (named `conv`, item_type per its
    output boundary) writes. Shared by the compiler that emits the tap and the
    runner that harvests it, so the two never disagree on the filename."""
    return trace_dir / f"stage_{index}_{conv}{_TRACE_SUFFIX[item_type]}"


def _source_kind(desc: Descriptor) -> str:
    src, _ = _io_kinds(desc)
    if src is None:
        raise CompileError(f"item_type {desc.item_type.value!r} has no source block")
    return src


def _sink_kind(desc: Descriptor) -> str:
    return _io_kinds(desc)[1]


def _resolve(step: Step, registry: Mapping[str, Stage[Any, Any]]) -> Stage[Any, Any]:
    stage = registry.get(step.conv)
    if stage is None:
        raise CompileError(f"unknown stage '{step.conv}'; known: {sorted(registry)}")
    return stage


def _forward_pass(
    steps: Sequence[Step],
    registry: Mapping[str, Stage[Any, Any]],
    start: Descriptor,
    sample_rate: float,
) -> tuple[list[Descriptor], list[float]]:
    boundaries = [start]
    rates = [sample_rate]
    for step in steps:
        stage = _resolve(step, registry)
        boundaries.append(stage.out_descriptor(boundaries[-1], step))
        rates.append(rates[-1] * stage.rate_factor(step))
    return boundaries, rates


# Relative tolerance on a stage's required input rate. Comfortably admits
# clock_correct's ppm-scale rate shift (~5e-5 at 50 ppm) while catching the gross
# resample-ratio mistakes issue 06 is about (forgotten/swapped/off-by-one ratios,
# all >=2%) — the silent-garbage class the fixed-window dechirp cannot self-detect.
_RATE_TOL = 0.02


def _validate_descriptors(
    steps: Sequence[Step],
    registry: Mapping[str, Stage[Any, Any]],
    boundaries: Sequence[Descriptor],
    rates: Sequence[float],
    symbol_rate: float,
    direction: str,
) -> None:
    for i, step in enumerate(steps):
        stage = _resolve(step, registry)
        in_desc = boundaries[i]
        producer = steps[i - 1].conv if i > 0 else "<source>"
        if (
            stage.accepts_item_type is not None
            and in_desc.item_type != stage.accepts_item_type
        ):
            raise CompileError(
                f"stage '{step.conv}' accepts item_type "
                f"{stage.accepts_item_type.value!r} but '{producer}' produces "
                f"{in_desc.item_type.value!r}"
            )
        if (
            stage.accepts_carrier is not None
            and in_desc.carrier != stage.accepts_carrier
        ):
            raise CompileError(
                f"stage '{step.conv}' accepts {stage.accepts_carrier.value} "
                f"carrier but '{producer}' produces {in_desc.carrier.value}"
            )
        required_amplitude = stage.accepts_amplitude_for(step)
        if (
            direction == "rx"
            and required_amplitude is not None
            and in_desc.amplitude not in required_amplitude
        ):
            wanted = ", ".join(sorted(a.value for a in required_amplitude))
            modes = agc_modes_for(required_amplitude)
            mode_hint = " or ".join(f"mode='{m}'" for m in modes)
            fix = (
                f"insert an 'agc' stage ({mode_hint}) after any "
                "channelization or resampling"
                if in_desc.amplitude is Amplitude.UNKNOWN
                else f"set the upstream 'agc' to {mode_hint}"
            )
            raise CompileError(
                f"stage '{step.conv}' requires input amplitude normalized to "
                f"{wanted} but '{producer}' produces {in_desc.amplitude.value}; "
                f"{fix}"
            )
        required_order = stage.required_input_order(step)
        if (
            required_order is not None
            and in_desc.order is not None
            and in_desc.order != required_order
        ):
            raise CompileError(
                f"stage '{step.conv}' decodes an order-{required_order} "
                f"alphabet but '{producer}' produces order-{in_desc.order} "
                f"symbols; align the two stages' order/sf params"
            )
        required = stage.required_input_rate(step, symbol_rate)
        if required is not None and required > 0:
            if abs(rates[i] - required) > _RATE_TOL * required:
                raise CompileError(
                    f"stage '{step.conv}' requires input sample rate ~{required:g} "
                    f"but the pipeline delivers {rates[i]:g} at that boundary; "
                    f"check resample ratios, oversample, and symbol_rate"
                )
        min_sps = stage.min_input_sps_for(step)
        if direction == "rx" and min_sps is not None:
            sps = rates[i] / symbol_rate
            if sps < min_sps * (1.0 - _RATE_TOL):
                raise CompileError(
                    f"stage '{step.conv}' recovers symbol timing and needs >= "
                    f"{min_sps:g} samples per symbol but the "
                    f"pipeline delivers {sps:g} ({rates[i]:g} into symbol_rate "
                    f"{symbol_rate:g}); resample the input or fix symbol_rate"
                )
        problem = stage.validate_input(in_desc, step)
        if problem is not None:
            raise CompileError(f"stage '{step.conv}': {problem}")


def _validate_probe_marks(
    steps: Sequence[Step],
    registry: Mapping[str, Stage[Any, Any]],
    boundaries: Sequence[Descriptor],
    rates: Sequence[float],
    k: int,
    direction: str,
) -> None:
    """A probe records burst marks in item units at its own graph position; the
    engine hands them to the coding entry at the seam. If a GR stage between
    the two changes the item rate or level, the marks reach the seam in the
    wrong units and silently seed wrong windows."""
    if direction != "rx" or k >= len(steps):
        return
    for j, step in enumerate(steps[:k]):
        if _resolve(step, registry).family != "probe":
            continue
        if boundaries[k].level is not boundaries[j + 1].level or (
            rates[k] != rates[j + 1]
        ):
            raise CompileError(
                f"'{step.conv}' records burst marks in item units at its own "
                f"position, but a later GR stage changes the item rate or "
                f"level before the coding seam, so the marks would be misread "
                f"there; move the probe to the end of the GR segment"
            )


def _validate(
    modem: Modem,
    registry: Mapping[str, Stage[Any, Any]],
    start: Descriptor,
    direction: str,
) -> None:
    issues: list[ValidationIssue] = []
    validate_path(
        modem.path, registry, start.level, "modem", issues, direction=direction
    )
    if issues:
        raise SpecValidationError(issues, "modem")


def _emit_gr_segment(
    steps: Sequence[Step],
    registry: Mapping[str, Stage[Any, Any]],
    boundaries: Sequence[Descriptor],
    rates: Sequence[float],
    symbol_rate: float,
    direction: str,
    sample_rate: float,
    start: Descriptor,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str,
    soft_tap_path: Path | None = None,
    trace_dir: Path | None = None,
) -> GrPipeline:
    n = len(steps)
    ctx = CompileContext(start, sample_rate, symbol_rate)

    if direction == "rx":
        ctx.chain(_source_kind(boundaries[0]), **dict(source_io))
        soft_tail: str | None = None
        for i, step in enumerate(steps):
            stage = _resolve(step, registry)
            ctx.descriptor = boundaries[i]
            ctx.rate = rates[i]
            stage.emit_rx(ctx, step)
            if boundaries[i + 1].item_type is ItemType.F:
                soft_tail = ctx.tail
            if trace_dir is not None and ctx.tail is not None:
                out = boundaries[i + 1]
                tap = ctx.add(
                    _sink_kind(out),
                    path=str(trace_sink_path(trace_dir, i, step.conv, out.item_type)),
                )
                ctx.connect(ctx.tail, tap)
        ctx.descriptor = boundaries[n]
        ctx.rate = rates[n]
        ctx.chain(_sink_kind(boundaries[n]), **dict(sink_io))
        # Quality tap: a bits-final path hard-slices a soft demod, so its final
        # sink carries no decision confidence. Tap the last soft-symbol wire to a
        # sidecar so the run can score the demod it actually produced.
        if soft_tap_path is not None and soft_tail is not None:
            tap = ctx.add("soft_bits_file_sink", path=str(soft_tap_path))
            ctx.connect(soft_tail, tap)
    else:  # tx
        ctx.chain(_source_kind(boundaries[n]), **dict(source_io))
        for i in range(n - 1, -1, -1):
            step = steps[i]
            stage = _resolve(step, registry)
            ctx.descriptor = boundaries[i + 1]
            ctx.rate = rates[i + 1]
            stage.emit_tx(ctx, step)
        ctx.descriptor = boundaries[0]
        ctx.rate = rates[0]
        ctx.chain(_sink_kind(boundaries[0]), **dict(sink_io))

    return ctx.build(name, sample_rate)


def _known_engine(step: Step, registry: Mapping[str, Stage[Any, Any]]) -> str | None:
    stage = registry.get(step.conv)
    if stage is None:
        return None
    return "coding" if isinstance(stage, CodingStage) else "gr"


def _split_index(
    steps: Sequence[Step], registry: Mapping[str, Stage[Any, Any]], direction: str
) -> int:
    # Unknown names map to None so _validate stays the sole (aggregating)
    # reporter of unknown-stage issues; they neither start the coding segment
    # nor count as a GR re-entry.
    engines = [_known_engine(s, registry) for s in steps]
    k = next((i for i, e in enumerate(engines) if e == "coding"), len(steps))
    if direction == "tx" and k < len(steps):
        raise CompileError(
            f"stage '{steps[k].conv}' is a coding stage; coded tx is not supported yet"
        )
    for i in range(k, len(steps)):
        if engines[i] not in (None, "coding"):
            raise CompileError(
                f"stage '{steps[i].conv}' would re-enter the GR engine after "
                f"'{steps[k].conv}'; a pipeline never re-enters GR"
            )
    return k


@dataclass(frozen=True)
class CompiledPipeline:
    gr: GrPipeline | None
    coding: CodingProgram | None
    boundary: Descriptor
    final: Descriptor
    boundaries: list[Descriptor]
    rates: list[float]
    # sidecar the GR segment taps its last soft-symbol wire to, when an all-GR
    # path hard-slices a soft demod to bits; feeds the quality soft evidence.
    soft_seam: Path | None = None
    # count of GR-segment stages (the coding split index); the trace taps cover
    # exactly modem.path[:gr_steps], so the runner harvests them by this bound.
    gr_steps: int = 0


def compile_pipeline(
    modem: Modem,
    registry: Mapping[str, Stage[Any, Any]],
    *,
    direction: str,
    sample_rate: float,
    start: Descriptor,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str = "pipeline",
    quality_tap: bool = False,
    trace_dir: Path | None = None,
) -> CompiledPipeline:
    if direction not in ("rx", "tx"):
        raise CompileError(f"direction must be 'rx' or 'tx', got {direction!r}")
    steps = modem.path
    k = _split_index(steps, registry, direction)
    _validate(modem, registry, start, direction)
    boundaries, rates = _forward_pass(steps, registry, start, sample_rate)
    _validate_descriptors(
        steps, registry, boundaries, rates, modem.symbol_rate, direction
    )
    _validate_probe_marks(steps, registry, boundaries, rates, k, direction)
    # An all-GR path that hard-slices a soft demod to bits (no coding tail) leaves
    # the demod's decision confidence unmeasurable at the final sink; tap the last
    # soft-symbol wire to a sidecar the run scores instead of reading "uncertain".
    soft_seam: Path | None = None
    if (
        quality_tap
        and direction == "rx"
        and k == len(steps)
        and boundaries[-1].item_type is ItemType.B
        and any(b.item_type is ItemType.F for b in boundaries[1:])
    ):
        soft_seam = Path(str(sink_io["path"])).with_name("soft_tap.f32")
    gr: GrPipeline | None = None
    if k or direction == "tx" or not steps:
        gr = _emit_gr_segment(
            steps[:k],
            registry,
            boundaries,
            rates,
            modem.symbol_rate,
            direction,
            sample_rate,
            start,
            source_io,
            sink_io,
            name,
            soft_seam,
            trace_dir,
        )
    coding: CodingProgram | None = None
    if k < len(steps):
        b = CodingBuilder()
        for i in range(k, len(steps)):
            stage = _resolve(steps[i], registry)
            b.label = stage_label(i, steps[i].conv)
            b.kind = steps[i].conv
            stage.emit_rx(b, steps[i])
        coding = CodingProgram(
            steps=b.steps,
            entry_level=boundaries[k].level,
            entry_item_type=boundaries[k].item_type,
        )
    return CompiledPipeline(
        gr=gr,
        coding=coding,
        boundary=boundaries[k],
        final=boundaries[len(steps)],
        boundaries=boundaries,
        rates=rates,
        soft_seam=soft_seam,
        gr_steps=k,
    )


def compile_modem(
    modem: Modem,
    registry: Mapping[str, Stage[Any, Any]],
    *,
    direction: str,
    sample_rate: float,
    start: Descriptor,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str = "pipeline",
) -> GrPipeline:
    cp = compile_pipeline(
        modem,
        registry,
        direction=direction,
        sample_rate=sample_rate,
        start=start,
        source_io=source_io,
        sink_io=sink_io,
        name=name,
    )
    if cp.coding is not None:
        raise CompileError(
            "path contains coding stages; compile_modem is the pure-GR entry — "
            "use compile_pipeline/run_rx"
        )
    assert cp.gr is not None
    return cp.gr
