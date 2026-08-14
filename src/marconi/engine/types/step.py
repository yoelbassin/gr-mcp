from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from marconi.errors import register_error


def stage_label(index: int, conv: str) -> str:
    """One home for the step-instance label — census, trace, and coding-builder
    rows cross-reference by it and must agree."""
    return f"{conv}[{index}]"


def _non_finite_problem(name: str, value: object) -> str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return f"{name} must be a finite number, got {value!r}"
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, float) and not math.isfinite(item):
                return f"{name}[{index}] must be a finite number, got {item!r}"
    return None


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conv: str

    @model_validator(mode="after")
    def _every_float_is_finite(self) -> "Step":
        """NaN is False for every `x <= 0`-style hand-rolled guard, and +inf
        clears every one-sided Field(gt=/ge=) constraint (inf > 0 is True);
        unblocked, both validate, compile, and pass validate_modem, then die
        inside the GR worker naming the IR field they corrupted, not the spec
        field the caller actually got wrong."""
        for name in type(self).model_fields:
            problem = _non_finite_problem(name, getattr(self, name))
            if problem is not None:
                raise ValueError(problem)
        return self


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
        if not isinstance(obj, Mapping):
            # dict(s) on a non-mapping raised TypeError past the tool's
            # (CompileError, ValidationError, ValueError) net, so path=[null]
            # surfaced as "'NoneType' object is not iterable" - naming no
            # index, no parameter and no expected shape
            raise StepSpecError(
                index,
                None,
                f"each path entry must be an object like "
                f'{{"conv": "<stage>", ...}}, got {type(obj).__name__}',
            )
        conv = obj.get("conv")
        if not isinstance(conv, str) or conv not in step_models:
            raise StepSpecError(
                index, conv, f"unknown conv; known: {sorted(step_models)}"
            )
        model = step_models[conv]
        try:
            out.append(model.model_validate(dict(obj)))
        except ValidationError as exc:
            raise StepSpecError(index, conv, _detail(model, exc)) from exc
    return out


def _detail(model: type[Step], exc: ValidationError) -> str:
    """Extra keys get the accepted field list appended. Steps forbid extras so
    a typo fails loudly, but the bare pydantic text ("Extra inputs are not
    permitted") reads identically whether the caller misspelled a field or is
    replaying a recipe whose field this stage no longer takes — and a sibling
    stage may still accept that very name."""
    extras = sorted(
        str(e["loc"][0])
        for e in exc.errors()
        if e["type"] == "extra_forbidden" and e["loc"]
    )
    if not extras:
        return str(exc)
    return (
        f"{exc}\nrejected field(s) {extras}; {model.__name__} accepts "
        f"{sorted(model.model_fields)}"
    )
