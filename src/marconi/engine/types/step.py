from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from marconi.errors import register_error


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv: str


S = TypeVar("S", bound=Step)


class StepSpecError(ValueError):
    def __init__(self, index: int, conv: object, detail: str) -> None:
        self.index = index
        self.conv = conv
        super().__init__(f"path[{index}] conv={conv!r}: {detail}")


register_error(StepSpecError, "invalid_argument")


def steps_from_spec(
    items: Sequence[Mapping[str, object]],
    step_models: Mapping[str, type[Step]],
) -> list[Step]:
    out: list[Step] = []
    for index, obj in enumerate(items):
        conv = obj.get("conv")
        if not isinstance(conv, str) or conv not in step_models:
            raise StepSpecError(
                index, conv, f"unknown conv; known: {sorted(step_models)}"
            )
        try:
            out.append(step_models[conv].model_validate(dict(obj)))
        except ValidationError as exc:
            raise StepSpecError(index, conv, str(exc)) from exc
    return out
