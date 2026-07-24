from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marconi.core.descriptor import Amplitude, Descriptor
from marconi.core.errors import register_error
from marconi.core.models import ValidationIssue
from marconi.core.params import ParamValue
from marconi.core.stages import SpecStep, Stage, validate_path
from marconi.phy.coding.builder import CodingBuilder
from marconi.phy.coding.program import CodingProgram
from marconi.phy.compile_context import CompileContext
from marconi.phy.ir import GrPipeline
from marconi.phy.models import ModemSpec


class CompileError(Exception):
    pass


register_error(CompileError, "invalid_argument")  # an uncompilable spec


# item_type -> (source_kind, sink_kind). The GR wire type alone selects IO;
# carrier is decision-hardness (a seam invariant), never a routing knob. Descriptor
# data, never stage names: replaces v1's rate-by-name scan and sink-by-type lookup.
_IO_BLOCKS: dict[str, tuple[str | None, str]] = {
    "c": ("iq_file_source", "iq_file_sink"),
    "s": (None, "symbols_file_sink"),
    "b": ("bits_file_source", "bits_file_sink"),
    "f": ("soft_bits_file_source", "soft_bits_file_sink"),
}


def _io_kinds(desc: Descriptor) -> tuple[str | None, str]:
    if desc.item_type not in _IO_BLOCKS:
        raise CompileError(f"no source/sink block for item_type {desc.item_type!r}")
    return _IO_BLOCKS[desc.item_type]


def _source_kind(desc: Descriptor) -> str:
    src, _ = _io_kinds(desc)
    if src is None:
        raise CompileError(f"item_type {desc.item_type!r} has no source block")
    return src


def _sink_kind(desc: Descriptor) -> str:
    return _io_kinds(desc)[1]


def _resolve(step: SpecStep, registry: Mapping[str, Stage[Any]]) -> Stage[Any]:
    stage = registry.get(step.conv)
    if stage is None:
        raise CompileError(f"unknown stage '{step.conv}'; known: {sorted(registry)}")
    return stage


def _forward_pass(
    steps: Sequence[SpecStep],
    registry: Mapping[str, Stage[Any]],
    start: Descriptor,
    sample_rate: float,
) -> tuple[list[Descriptor], list[float]]:
    boundaries = [start]
    rates = [sample_rate]
    for step in steps:
        stage = _resolve(step, registry)
        boundaries.append(stage.out_descriptor(boundaries[-1], step.params))
        rates.append(rates[-1] * stage.rate_factor(step.params))
    return boundaries, rates


# Relative tolerance on a stage's required input rate. Comfortably admits
# clock_correct's ppm-scale rate shift (~5e-5 at 50 ppm) while catching the gross
# resample-ratio mistakes issue 06 is about (forgotten/swapped/off-by-one ratios,
# all >=2%) — the silent-garbage class the fixed-window dechirp cannot self-detect.
_RATE_TOL = 0.02


def _validate_descriptors(
    steps: Sequence[SpecStep],
    registry: Mapping[str, Stage[Any]],
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
                f"{stage.accepts_item_type!r} but '{producer}' produces "
                f"{in_desc.item_type!r}"
            )
        if (
            stage.accepts_carrier is not None
            and in_desc.carrier != stage.accepts_carrier
        ):
            raise CompileError(
                f"stage '{step.conv}' accepts {stage.accepts_carrier.value} "
                f"carrier but '{producer}' produces {in_desc.carrier.value}"
            )
        if (
            direction == "rx"
            and stage.accepts_amplitude is not None
            and in_desc.amplitude not in stage.accepts_amplitude
        ):
            wanted = ", ".join(sorted(a.value for a in stage.accepts_amplitude))
            fix = (
                "insert an 'agc' stage after any channelization or resampling"
                if in_desc.amplitude is Amplitude.UNKNOWN
                else f"'{producer}' normalizes the wrong statistic — change its mode"
            )
            raise CompileError(
                f"stage '{step.conv}' requires input amplitude normalized to "
                f"{wanted} but '{producer}' produces {in_desc.amplitude.value}; "
                f"{fix} (the agc stage's `mode` selects the statistic)"
            )
        required = stage.required_input_rate(step.params, symbol_rate)
        if required is not None and required > 0:
            if abs(rates[i] - required) > _RATE_TOL * required:
                raise CompileError(
                    f"stage '{step.conv}' requires input sample rate ~{required:g} "
                    f"but the pipeline delivers {rates[i]:g} at that boundary; "
                    f"check resample ratios, oversample, and symbol_rate"
                )


def _validate(
    modem: ModemSpec,
    registry: Mapping[str, Stage[Any]],
    start: Descriptor,
    direction: str,
) -> None:
    issues: list[ValidationIssue] = []
    validate_path(
        modem.path, registry, start.level, "modem", issues, direction=direction
    )
    if issues:
        raise CompileError(
            "modem path invalid:\n"
            + "\n".join(
                f"  {i.block_id or '<modem>'}"
                f"{'.' + i.field if i.field else ''}: {i.message}"
                for i in issues
            )
        )


def _emit_gr_segment(
    steps: Sequence[SpecStep],
    registry: Mapping[str, Stage[Any]],
    boundaries: Sequence[Descriptor],
    rates: Sequence[float],
    symbol_rate: float,
    direction: str,
    sample_rate: float,
    start: Descriptor,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str,
) -> GrPipeline:
    n = len(steps)
    ctx = CompileContext(start, sample_rate, symbol_rate)

    if direction == "rx":
        ctx.chain(_source_kind(boundaries[0]), **dict(source_io))
        for i, step in enumerate(steps):
            stage = _resolve(step, registry)
            ctx.descriptor = boundaries[i]
            ctx.rate = rates[i]
            stage.emit_rx(ctx, step.params)
        ctx.descriptor = boundaries[n]
        ctx.rate = rates[n]
        ctx.chain(_sink_kind(boundaries[n]), **dict(sink_io))
    else:  # tx
        ctx.chain(_source_kind(boundaries[n]), **dict(source_io))
        for i in range(n - 1, -1, -1):
            step = steps[i]
            stage = _resolve(step, registry)
            ctx.descriptor = boundaries[i + 1]
            ctx.rate = rates[i + 1]
            stage.emit_tx(ctx, step.params)
        ctx.descriptor = boundaries[0]
        ctx.rate = rates[0]
        ctx.chain(_sink_kind(boundaries[0]), **dict(sink_io))

    return ctx.build(name, sample_rate)


def _split_index(
    steps: Sequence[SpecStep], registry: Mapping[str, Stage[Any]], direction: str
) -> int:
    k = next(
        (i for i, s in enumerate(steps) if _resolve(s, registry).engine == "coding"),
        len(steps),
    )
    if direction == "tx" and k < len(steps):
        raise CompileError(
            f"stage '{steps[k].conv}' is a coding stage; coded tx is not supported yet"
        )
    for i in range(k, len(steps)):
        if _resolve(steps[i], registry).engine != "coding":
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


def compile_pipeline(
    modem: ModemSpec,
    registry: Mapping[str, Stage[Any]],
    *,
    direction: str,
    sample_rate: float,
    start: Descriptor,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str = "pipeline",
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
    gr: GrPipeline | None = None
    if k or direction == "tx":
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
        )
    coding: CodingProgram | None = None
    if k < len(steps):
        b = CodingBuilder()
        for i in range(k, len(steps)):
            stage = _resolve(steps[i], registry)
            b.label = f"{steps[i].conv}[{i}]"
            b.kind = steps[i].conv
            stage.emit_rx(b, steps[i].params)
        coding = CodingProgram(
            steps=b.steps,
            entry_level=boundaries[k].level,
            entry_item_type=boundaries[k].item_type,
        )
    return CompiledPipeline(
        gr=gr, coding=coding, boundary=boundaries[k], final=boundaries[len(steps)]
    )


def compile_modem(
    modem: ModemSpec,
    registry: Mapping[str, Stage[Any]],
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
