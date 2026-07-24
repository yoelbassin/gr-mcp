from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from marconi.bits.builder import ProgramBuilder
from marconi.bits.models import CodecSpec
from marconi.bits.program import FrameProgram
from marconi.core.errors import register_error
from marconi.core.levels import Level
from marconi.core.models import ValidationIssue
from marconi.core.stages import (
    SpecValidationError,
    Stage,
    StageDirectionError,
    validate_params,
)


class CodecCompileError(Exception):
    pass


register_error(CodecCompileError, "invalid_argument")  # an uncompilable codec


def compile_codec(
    codec: CodecSpec,
    registry: Mapping[str, Stage[ProgramBuilder]],
    direction: Literal["rx", "tx"],
) -> FrameProgram:
    issues: list[ValidationIssue] = []
    for idx, step in enumerate(codec.path):
        stage = registry.get(step.conv)
        if stage is None:
            raise CodecCompileError(
                f"unknown stage '{step.conv}'; known: {sorted(registry)}"
            )
        validate_params(f"{step.conv}[{idx}]", stage.params_model, step.params, issues)
    if issues:
        raise SpecValidationError(issues, "codec")

    b = ProgramBuilder()
    indexed = list(enumerate(codec.path))
    for idx, step in indexed if direction == "rx" else list(reversed(indexed)):
        stage = registry[step.conv]
        if direction not in stage.directions:
            raise StageDirectionError(stage.name, direction, stage.directions)
        b.label = f"{step.conv}[{idx}]"
        emit = stage.emit_rx if direction == "rx" else stage.emit_tx
        emit(b, step.params)
    first = registry[codec.path[0].conv] if codec.path else None
    return FrameProgram(
        direction=direction,
        steps=b.steps,
        requires_symbol_input=first is not None and first.from_level is Level.SYMBOLS,
    )
