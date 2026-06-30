from __future__ import annotations

from collections.abc import Mapping, Sequence

from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.models import ValidationIssue
from marconi.core.params import ParamValue
from marconi.core.stages import SpecStep, Stage, validate_path
from marconi.phy.compile_context import CompileContext
from marconi.phy.ir import GrPipeline
from marconi.phy.models import ModemSpec


class CompileError(Exception):
    pass


# (item_type, carrier) -> (source_kind, sink_kind). Descriptor data, never stage
# names: this is what replaces v1's rate-by-name scan and sink-by-block-type lookup.
_IO_BLOCKS: dict[tuple[str, Carrier], tuple[str | None, str]] = {
    ("c", Carrier.HARD): ("iq_file_source", "iq_file_sink"),
    ("c", Carrier.SOFT): ("iq_file_source", "iq_file_sink"),
    ("s", Carrier.HARD): (None, "symbols_file_sink"),
    ("b", Carrier.HARD): ("bits_file_source", "bits_file_sink"),
    ("f", Carrier.SOFT): ("soft_bits_file_source", "soft_bits_file_sink"),
}


def _io_kinds(desc: Descriptor) -> tuple[str | None, str]:
    key = (desc.item_type, desc.carrier)
    if key not in _IO_BLOCKS:
        raise CompileError(
            f"no source/sink block for item_type {desc.item_type!r} "
            f"carrier {desc.carrier.value}"
        )
    return _IO_BLOCKS[key]


def _source_kind(desc: Descriptor) -> str:
    src, _ = _io_kinds(desc)
    if src is None:
        raise CompileError(
            f"item_type {desc.item_type!r} carrier {desc.carrier.value} "
            "has no source block"
        )
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
            + "\n".join(f"  {i.block_id or '<modem>'}: {i.message}" for i in issues)
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
