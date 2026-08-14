from __future__ import annotations

from typing import ClassVar, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict

_M = TypeVar("_M", bound=BaseModel)


def replace(model: _M, **changes: object) -> _M:
    """A model with some fields replaced, RE-VALIDATED. pydantic's own
    model_copy(update=...) validates nothing: it stores a value the field's
    type forbids and silently attaches a misspelled field name as a stray
    attribute — measured on PipelineResult.status, which is the field the whole
    MCP surface reads to decide whether a decode happened. Every field
    replacement in the tree goes through here instead, so a typo is a raise and
    a wrong type is a ValidationError. Nested models are passed through by
    identity (pydantic revalidates instances only when asked), so this stays
    cheap on a result carrying thousands of census rows.

    Only the fields the original SET are carried over, so a replacement cannot
    quietly promote an omitted field to an explicit null — rebuilding from
    __dict__ marks every field as set, which put `"description": null` on every
    describe_stages detail row."""
    unknown = sorted(set(changes) - set(type(model).model_fields))
    if unknown:
        raise ValueError(f"{type(model).__name__} has no field(s) {unknown}")
    kept = {name: getattr(model, name) for name in model.model_fields_set}
    return type(model).model_validate({**kept, **changes})


def _restore_nested(value: object, dumped: object) -> None:
    """Walk a field's value alongside its dump, restoring deliberate nulls at
    every depth a payload nests: a model, a list of them, or a dict of either
    (describe_stages groups its rows by family, and a dict-shaped field used to
    drop the nulls the rows below deliberately published)."""
    if isinstance(value, BaseModel) and isinstance(dumped, dict):
        _restore_set_nulls(value, cast("dict[str, object]", dumped))
    elif isinstance(value, list) and isinstance(dumped, list):
        for item, entry in zip(value, dumped):
            _restore_nested(item, entry)
    elif isinstance(value, dict) and isinstance(dumped, dict):
        for key, item in value.items():
            _restore_nested(item, dumped.get(key))


def _restore_set_nulls(
    model: BaseModel, dumped: dict[str, object]
) -> dict[str, object]:
    for name in model.model_fields_set:
        value = getattr(model, name, None)
        if value is None:
            dumped[name] = None
            continue
        _restore_nested(value, dumped.get(name))
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

    # The fields this shape publishes even when null, because absent and null
    # are different answers for them. Declared on the model that has the
    # exception rather than re-derived by each producer as an if-ladder.
    always_present: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def build(cls, **fields: object) -> "Self":
        """Construct from keyword fields, DROPPING the None ones except those
        the model declares `always_present`. This is the default a producer
        wants: None means "no value here".

        Plain `cls(...)` marks every keyword as set, so a `None` handed to it
        SHIPS as null. That is right for a producer choosing to publish an
        undefined measurement and wrong for one that simply had nothing; three
        payloads regressed that way before this existed."""
        return cls.model_validate(
            {
                k: v
                for k, v in fields.items()
                if v is not None or k in cls.always_present
            }
        )

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
