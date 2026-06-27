from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level, adjacent
from marconi.core.models import ValidationIssue
from marconi.core.params import StageParams

B = TypeVar("B")


class SpecStep(Protocol):
    @property
    def conv(self) -> str: ...
    @property
    def params(self) -> Mapping[str, object]: ...


class SpecValidationError(Exception):
    def __init__(self, issues: list[ValidationIssue], kind: str) -> None:
        self.issues = issues
        lines = [
            f"{i.block_id or f'<{kind}>'}"
            f"{'.' + i.field if i.field else ''}: {i.message}"
            for i in issues
        ]
        super().__init__(f"{kind} validation failed:\n" + "\n".join(lines))


class StageDirectionError(Exception):
    def __init__(self, stage: str, direction: str, supported: frozenset[str]) -> None:
        super().__init__(
            f"stage '{stage}' does not support direction '{direction}'; "
            f"supported: {sorted(supported)}"
        )


class Stage(ABC, Generic[B]):
    name: str
    from_level: Level
    to_level: Level
    family: str
    directions: frozenset[str] = frozenset({"rx", "tx"})
    params_model: type[StageParams]

    @abstractmethod
    def emit_rx(self, b: B, params: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def emit_tx(self, b: B, params: Mapping[str, Any]) -> None: ...

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(
            self.to_level, in_desc.item_type, in_desc.layout, in_desc.carrier
        )

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return 1.0


class RxStage(Stage[B]):
    directions: frozenset[str] = frozenset({"rx"})

    def emit_tx(self, b: B, params: Mapping[str, Any]) -> None:
        raise StageDirectionError(self.name, "tx", self.directions)


class TxStage(Stage[B]):
    directions: frozenset[str] = frozenset({"tx"})

    def emit_rx(self, b: B, params: Mapping[str, Any]) -> None:
        raise StageDirectionError(self.name, "rx", self.directions)


class DuplexStage(Stage[B]):
    directions: frozenset[str] = frozenset({"rx", "tx"})


def validate_params(
    block_id: str,
    params_model: type[BaseModel],
    supplied: Mapping[str, object],
    out: list[ValidationIssue],
) -> None:
    try:
        params_model.model_validate(dict(supplied))
    except ValidationError as exc:
        for err in exc.errors():
            field = ".".join(str(p) for p in err["loc"]) or None
            out.append(
                ValidationIssue(block_id=block_id, field=field, message=err["msg"])
            )


def validate_path(
    steps: Sequence[SpecStep],
    registry: Mapping[str, Stage[Any]],
    start_level: Level,
    entity_name: str,
    issues: list[ValidationIssue],
    direction: str | None = None,
) -> None:
    prev_level = start_level
    for idx, step in enumerate(steps):
        sid = f"{step.conv}[{idx}]"
        conv = registry.get(step.conv)
        if conv is None:
            known = ", ".join(sorted(registry))
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"unknown converter '{step.conv}'; known: {known}",
                )
            )
            continue
        validate_params(sid, conv.params_model, step.params, issues)
        if direction is not None and direction not in conv.directions:
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{conv.name} does not support direction "
                    f"'{direction}'; supports {sorted(conv.directions)}",
                )
            )
        if idx == 0 and conv.from_level != start_level:
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{entity_name} must start at {start_level.value} "
                    f"but {conv.name} starts at {conv.from_level.value}",
                )
            )
        elif idx > 0 and not adjacent(prev_level, conv.from_level):
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{conv.name} input level {conv.from_level.value} is not "
                    f"adjacent to the previous level {prev_level.value}",
                )
            )
        prev_level = conv.to_level
