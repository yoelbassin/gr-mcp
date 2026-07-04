from __future__ import annotations

from collections.abc import Mapping, Sequence

from marconi.core.descriptor import Descriptor
from marconi.core.errors import register_error
from marconi.core.models import ValidationIssue
from marconi.core.params import ParamValue
from marconi.core.stages import SpecStep, Stage, validate_path
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


def _resolve(
    step: SpecStep, registry: Mapping[str, Stage[CompileContext]]
) -> Stage[CompileContext]:
    stage = registry.get(step.conv)
    if stage is None:
        raise CompileError(f"unknown stage '{step.conv}'; known: {sorted(registry)}")
    return stage


def _forward_pass(
    steps: Sequence[SpecStep],
    registry: Mapping[str, Stage[CompileContext]],
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
    registry: Mapping[str, Stage[CompileContext]],
    boundaries: Sequence[Descriptor],
    rates: Sequence[float],
    symbol_rate: float,
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
    registry: Mapping[str, Stage[CompileContext]],
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


def compile_modem(
    modem: ModemSpec,
    registry: Mapping[str, Stage[CompileContext]],
    *,
    direction: str,
    sample_rate: float,
    start: Descriptor,
    source_io: Mapping[str, ParamValue],
    sink_io: Mapping[str, ParamValue],
    name: str = "pipeline",
) -> GrPipeline:
    if direction not in ("rx", "tx"):
        raise CompileError(f"direction must be 'rx' or 'tx', got {direction!r}")
    _validate(modem, registry, start, direction)
    steps = modem.path
    n = len(steps)
    boundaries, rates = _forward_pass(steps, registry, start, sample_rate)
    _validate_descriptors(steps, registry, boundaries, rates, modem.symbol_rate)
    ctx = CompileContext(start, sample_rate, modem.symbol_rate)

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
