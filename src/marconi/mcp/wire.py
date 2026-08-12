from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


def _restore_set_nulls(
    model: BaseModel, dumped: dict[str, object]
) -> dict[str, object]:
    for name in model.model_fields_set:
        value = getattr(model, name, None)
        if value is None:
            dumped[name] = None
            continue
        nested = dumped.get(name)
        if isinstance(value, BaseModel) and isinstance(nested, dict):
            _restore_set_nulls(value, cast("dict[str, object]", nested))
        elif isinstance(value, list) and isinstance(nested, list):
            for item, entry in zip(value, nested):
                if isinstance(item, BaseModel) and isinstance(entry, dict):
                    _restore_set_nulls(item, cast("dict[str, object]", entry))
    return dumped


class Payload(BaseModel):
    """Base for every MCP response shape. The tools return JSON, so something
    must produce a dict — but only the model produces its own, from declared
    and validated fields, instead of each caller assembling one key at a time.

    Omission rule, applied at every nesting depth: a field the producer never
    set is absent (a census row lists only the ports its block has), and a
    field the producer explicitly set to null stays on the wire — "measured,
    and undefined" is not the same answer as "not applicable to this stream",
    and an agent reading a missing key cannot tell them apart. So a producer
    that wants a key omitted must leave it out of the construction buffer, not
    pass None for it."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def build(cls, **fields: object) -> "Self":
        """Construct from keyword fields, DROPPING the None ones. This is the
        default a producer wants: None means "no value here". Keeping a null on
        the wire is the deliberate act — pass the key through model_validate.

        Plain `cls(...)` marks every keyword as set, so a `None` handed to it
        SHIPS as null. That is right for a producer choosing to publish an
        undefined measurement and wrong for one that simply had nothing; three
        payloads regressed that way before this existed."""
        return cls.model_validate({k: v for k, v in fields.items() if v is not None})

    def as_payload(self) -> dict[str, object]:
        dumped: dict[str, object] = self.model_dump(mode="json", exclude_none=True)
        return _restore_set_nulls(self, dumped)


class Histogram(Payload):
    """Bin i's center is start + i*step: the axis is derivable, never shipped."""

    start: float
    step: float
    counts: list[int]


class Ramp(Payload):
    """A uniform arithmetic tiling, shipped as its formula rather than its
    terms: entry i is start + i*stride."""

    start: int
    stride: int
    count: int
