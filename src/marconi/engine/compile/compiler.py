from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.program import CodingProgram
from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import _IO_BLOCKS, GrPipeline
from marconi.engine.stages.base import (
    CodingStage,
    SpecValidationError,
    Stage,
    validate_path,
)
from marconi.engine.stages.conditioning import agc_modes_for
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem, ValidationIssue
from marconi.engine.types.params import ParamValue
from marconi.engine.types.step import Step, stage_label

Direction = Literal["rx", "tx"]


def _io_kinds(desc: Descriptor) -> tuple[str | None, str]:
    if desc.item_type not in _IO_BLOCKS:
        raise CompileError(
            f"no source/sink block for item_type {desc.item_type.value!r}"
        )
    return _IO_BLOCKS[desc.item_type]


def trace_item_type(desc: Descriptor) -> ItemType:
    """The item type a boundary's trace sidecar carries. Hard symbol indices
    ride the u8 wire (a qam-class demod boundary): their tap converts to i16
    symbols — a 'b' sidecar would invite bitwise parsing of symbol indices,
    the trap the terminal (SYMBOLS, 'b') rewrite exists to close."""
    if desc.item_type is ItemType.B and desc.level is Level.SYMBOLS:
        return ItemType.S
    return desc.item_type


def trace_sink_path(
    trace_dir: Path, index: int, conv: str, item_type: ItemType
) -> Path:
    """Where the trace tap for GR stage `index` (named `conv`, item_type per
    trace_item_type of its output boundary) writes. Shared by the compiler that
    emits the tap and the runner that harvests it, so the two never disagree on
    the filename."""
    return trace_dir / f"stage_{index}_{conv}{item_type.suffix}"


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
# resample-ratio mistakes (forgotten/swapped/off-by-one ratios,
# all >=2%) — the silent-garbage class the fixed-window dechirp cannot self-detect.
_RATE_TOL = 0.02


@dataclass(frozen=True)
class _CompilePlan:
    """The forward pass's outputs plus what produced them — the unit every
    validation and emission pass consumes."""

    steps: Sequence[Step]
    registry: Mapping[str, Stage[Any, Any]]
    boundaries: Sequence[Descriptor]
    rates: Sequence[float]
    symbol_rate: float
    sample_rate: float
    direction: Direction


@dataclass(frozen=True)
class _StepInput:
    """One step and the stream arriving at it — everything an input check reads."""

    step: Step
    stage: Stage[Any, Any]
    desc: Descriptor
    rate: float
    producer: str
    symbol_rate: float
    direction: Direction


def _check_item_type(s: _StepInput) -> str | None:
    want = s.stage.accepts_item_type
    if want is None or s.desc.item_type == want:
        return None
    return (
        f"stage '{s.step.conv}' accepts item_type "
        f"{want.value!r} but '{s.producer}' produces "
        f"{s.desc.item_type.value!r}"
    )


def _check_carrier(s: _StepInput) -> str | None:
    want = s.stage.accepts_carrier
    if want is None or s.desc.carrier == want:
        return None
    return (
        f"stage '{s.step.conv}' accepts {want.value} "
        f"carrier but '{s.producer}' produces {s.desc.carrier.value}"
    )


def _check_amplitude(s: _StepInput) -> str | None:
    required = s.stage.accepts_amplitude_for(s.step)
    if s.direction != "rx" or required is None or s.desc.amplitude in required:
        return None
    wanted = ", ".join(sorted(a.value for a in required))
    mode_hint = " or ".join(f"mode='{m}'" for m in agc_modes_for(required))
    fix = (
        f"insert an 'agc' stage ({mode_hint}) after any " "channelization or resampling"
        if s.desc.amplitude is Amplitude.UNKNOWN
        else f"set the upstream 'agc' to {mode_hint}"
    )
    return (
        f"stage '{s.step.conv}' requires input amplitude normalized to "
        f"{wanted} but '{s.producer}' produces {s.desc.amplitude.value}; "
        f"{fix}"
    )


def _check_alphabet_order(s: _StepInput) -> str | None:
    required = s.stage.required_input_order(s.step)
    if required is None or s.desc.order is None or s.desc.order == required:
        return None
    return (
        f"stage '{s.step.conv}' decodes an order-{required} "
        f"alphabet but '{s.producer}' produces order-{s.desc.order} "
        f"symbols; align the two stages' order/sf params"
    )


def _check_required_rate(s: _StepInput) -> str | None:
    required = s.stage.required_input_rate(s.step, s.symbol_rate)
    if required is None or required <= 0:
        return None
    if abs(s.rate - required) <= _RATE_TOL * required:
        return None
    return (
        f"stage '{s.step.conv}' requires input sample rate ~{required:g} "
        f"but the pipeline delivers {s.rate:g} at that boundary; "
        f"check resample ratios, oversample, and symbol_rate"
    )


def _check_min_sps(s: _StepInput) -> str | None:
    min_sps = s.stage.min_input_sps_for(s.step)
    if s.direction != "rx" or min_sps is None:
        return None
    sps = s.rate / s.symbol_rate
    if sps >= min_sps * (1.0 - _RATE_TOL):
        return None
    return (
        f"stage '{s.step.conv}' recovers symbol timing and needs >= "
        f"{min_sps:g} samples per symbol but the "
        f"pipeline delivers {sps:g} ({s.rate:g} into symbol_rate "
        f"{s.symbol_rate:g}); resample the input or fix symbol_rate"
    )


def _check_stage_input(s: _StepInput) -> str | None:
    problem = s.stage.validate_input(s.desc, s.step)
    return None if problem is None else f"stage '{s.step.conv}': {problem}"


def _check_stage_input_rate(s: _StepInput) -> str | None:
    problem = s.stage.validate_input_rate(s.step, s.rate)
    return None if problem is None else f"stage '{s.step.conv}': {problem}"


# Every contract a stage's input must satisfy, each independently answerable
# from a _StepInput. Order is the order a reader meets them: what the stream is,
# then how it is scaled, then what the stage says for itself.
_INPUT_CHECKS: tuple[Callable[[_StepInput], str | None], ...] = (
    _check_item_type,
    _check_carrier,
    _check_amplitude,
    _check_alphabet_order,
    _check_required_rate,
    _check_min_sps,
    _check_stage_input,
    _check_stage_input_rate,
)


def _validate_descriptors(plan: _CompilePlan) -> None:
    for i, step in enumerate(plan.steps):
        arriving = _StepInput(
            step=step,
            stage=_resolve(step, plan.registry),
            desc=plan.boundaries[i],
            rate=plan.rates[i],
            producer=plan.steps[i - 1].conv if i > 0 else "<source>",
            symbol_rate=plan.symbol_rate,
            direction=plan.direction,
        )
        for check in _INPUT_CHECKS:
            problem = check(arriving)
            if problem is not None:
                raise CompileError(problem)


def _validate_probe_marks(plan: _CompilePlan, k: int) -> None:
    """A probe records burst marks in item units at its own graph position; the
    engine hands them to the coding entry at the seam. If a GR stage between
    the two changes the item rate or level, the marks reach the seam in the
    wrong units and silently seed wrong windows."""
    steps, boundaries, rates = plan.steps, plan.boundaries, plan.rates
    if plan.direction != "rx" or k >= len(steps):
        return
    for j, step in enumerate(steps[:k]):
        if _resolve(step, plan.registry).family != "probe":
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
    direction: Direction,
) -> None:
    issues: list[ValidationIssue] = []
    validate_path(
        modem.path, registry, start.level, "modem", issues, direction=direction
    )
    if issues:
        raise SpecValidationError(issues, "modem")


def _emit_gr_segment(
    plan: _CompilePlan,
    k: int,
    *,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str,
    soft_tap_path: Path | None = None,
    trace_dir: Path | None = None,
) -> GrPipeline:
    steps = plan.steps[:k]
    registry, boundaries, rates = plan.registry, plan.boundaries, plan.rates
    n = len(steps)
    ctx = CompileContext(boundaries[0], plan.sample_rate, plan.symbol_rate)

    if plan.direction == "rx":
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
                trace_it = trace_item_type(out)
                tail = ctx.tail
                if trace_it is not out.item_type:
                    # exact uchar->i16 (char_to_short scales by 256)
                    to_f = ctx.add("uchar_to_float")
                    ctx.connect(tail, to_f)
                    to_s = ctx.add("float_to_short", scale=1.0)
                    ctx.connect(to_f, to_s)
                    tail = to_s
                tap = ctx.add(
                    _IO_BLOCKS[trace_it][1],
                    path=str(trace_sink_path(trace_dir, i, step.conv, trace_it)),
                )
                ctx.connect(tail, tap)
        ctx.descriptor = boundaries[n]
        ctx.rate = rates[n]
        terminal = ctx.chain(_sink_kind(boundaries[n]), **dict(sink_io))
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
        terminal = ctx.chain(_sink_kind(boundaries[0]), **dict(sink_io))

    return ctx.build(name, plan.sample_rate, terminal_sink=terminal)


def _emit_coding_segment(plan: _CompilePlan, k: int) -> CodingProgram:
    b = CodingBuilder()
    for i in range(k, len(plan.steps)):
        stage = _resolve(plan.steps[i], plan.registry)
        b.begin_step(stage_label(i, plan.steps[i].conv), plan.steps[i].conv)
        stage.emit_rx(b, plan.steps[i])
    return CodingProgram(
        steps=b.steps,
    )


def _known_engine(step: Step, registry: Mapping[str, Stage[Any, Any]]) -> str | None:
    stage = registry.get(step.conv)
    if stage is None:
        return None
    return "coding" if isinstance(stage, CodingStage) else "gr"


def _split_index(
    steps: Sequence[Step], registry: Mapping[str, Stage[Any, Any]], direction: Direction
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
    direction: Direction,
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
    plan = _CompilePlan(
        steps=steps,
        registry=registry,
        boundaries=boundaries,
        rates=rates,
        symbol_rate=modem.symbol_rate,
        sample_rate=sample_rate,
        direction=direction,
    )
    _validate_descriptors(plan)
    _validate_probe_marks(plan, k)
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
            plan,
            k,
            source_io=source_io,
            sink_io=sink_io,
            name=name,
            soft_tap_path=soft_seam,
            trace_dir=trace_dir,
        )
    coding: CodingProgram | None = None
    if k < len(steps):
        coding = _emit_coding_segment(plan, k)
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
    direction: Direction,
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
