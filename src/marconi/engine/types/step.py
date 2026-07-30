from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv: str


S = TypeVar("S", bound=Step)


def steps_from_spec(
    items: Sequence[Mapping[str, object]],
    step_models: Mapping[str, type[Step]],
) -> list[Step]:
    out: list[Step] = []
    for obj in items:
        conv = obj["conv"]
        if not isinstance(conv, str) or conv not in step_models:
            raise KeyError(f"unknown conv {conv!r}; known: {sorted(step_models)}")
        out.append(step_models[conv].model_validate(dict(obj)))
    return out
