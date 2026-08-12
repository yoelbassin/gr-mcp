from __future__ import annotations

from collections.abc import Callable
from typing import Any

from marconi.engine.stages.base import Stage
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.mcp.wire import Payload

__all__ = ["ENVELOPE", "family_names", "stage_index", "stage_details"]

# The agent-facing gloss for each rung and wire type. Keyed by the enums and
# checked for completeness at import: this primer is the only description of
# the vocabulary the agent gets, and a member missing from it is a member the
# agent never learns exists.
_LEVEL_GLOSS: dict[Level, str] = {
    Level.IQ: "complex baseband samples",
    Level.SYMBOLS: "one item per symbol (hard index or soft value)",
    Level.BITS: "one item per bit (hard u8, or soft float32 LLR)",
    Level.AUDIO: "demodulated audio floats",
}

_ITEM_GLOSS: dict[ItemType, str] = {
    ItemType.C: "complex64",
    ItemType.S: "int16 hard symbol index",
    ItemType.F: (
        "float32 soft value; sign is per level: at bits it is an LLR "
        "where bit 1 = NEGATIVE, at symbols it is a demod output where "
        "POSITIVE slices to bit 1"
    ),
    ItemType.B: "uint8 hard bit, one byte per bit",
}

_ungloss = sorted(
    [lv.value for lv in Level if lv not in _LEVEL_GLOSS]
    + [t.value for t in ItemType if t not in _ITEM_GLOSS]
)
if _ungloss:
    raise RuntimeError(f"vocabulary members with no agent-facing gloss: {_ungloss}")

ENVELOPE: dict[str, object] = {
    "spec_envelope": {
        "name": "optional str",
        "symbol_rate": ("float > 0 (symbols/second at the demod boundary)"),
        "path": "list of steps, each {'conv': <stage name>, ...stage params}",
    },
    "levels": {lv.value: gloss for lv, gloss in _LEVEL_GLOSS.items()},
    "item_types": {t.value: gloss for t, gloss in _ITEM_GLOSS.items()},
    "direction": (
        "the same path compiles rx (decode) and tx (generate); "
        "stages declare which directions they support"
    ),
}


class StageEntry(Payload):
    """One stage as describe_stages publishes it: the index rows carry the
    first three (plus a description where the name alone does not carry the
    use), a per-stage detail call fills in the input contracts and the spec
    schema. A contract this call cannot evaluate without a step is reported
    null and named in step_conditional — see _overrides."""

    name: str
    levels: str
    dir: str
    description: str | None = None
    family: str | None = None
    step_conditional: list[str] | None = None
    accepts_item_type: str | None = None
    accepts_carrier: str | None = None
    accepts_amplitude: list[str] | None = None
    min_input_sps: float | None = None
    seeds_windows: bool | None = None
    params_schema: dict[str, Any] | None = None


def _index_fields(stage: Stage[Any, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "name": stage.name,
        "levels": f"{stage.from_level.value}>{stage.to_level.value}",
        "dir": ",".join(sorted(stage.directions)),
    }
    if stage.description:
        fields["description"] = stage.description
    return fields


def stage_index() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, stage in sorted(stage_registry().items()):
        row = StageEntry.model_validate(_index_fields(stage)).as_payload()
        grouped.setdefault(stage.family, []).append(row)
    return dict(sorted(grouped.items()))


def family_names(family: str) -> list[str]:
    names = sorted(n for n, s in stage_registry().items() if s.family == family)
    if not names:
        fams = sorted({s.family for s in stage_registry().values()})
        raise ValueError(f"unknown family {family!r}; known: {fams}")
    return names


def stage_details(names: list[str]) -> list[dict[str, Any]]:
    registry = stage_registry()
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {sorted(registry)}")
    out: list[dict[str, Any]] = []
    for n in names:
        s = registry[n]
        fields = _index_fields(s)
        fields["family"] = s.family
        conditional_amp = _overrides(s, Stage.accepts_amplitude_for)
        conditional_sps = _overrides(s, Stage.min_input_sps_for)
        step_conditional = sorted(
            name
            for name, cond in (
                ("accepts_amplitude", conditional_amp),
                ("min_input_sps", conditional_sps),
            )
            if cond
        )
        if step_conditional:
            fields["step_conditional"] = step_conditional
        fields.update(
            accepts_item_type=(
                None if s.accepts_item_type is None else s.accepts_item_type.value
            ),
            accepts_carrier=(
                None if s.accepts_carrier is None else s.accepts_carrier.value
            ),
            accepts_amplitude=(
                None
                if conditional_amp or s.accepts_amplitude is None
                else sorted(a.value for a in s.accepts_amplitude)
            ),
            min_input_sps=None if conditional_sps else s.min_input_sps,
            seeds_windows=s.seeds_windows,
            params_schema=s.step_model.model_json_schema(),
        )
        out.append(StageEntry.model_validate(fields).as_payload())
    return out


def _overrides(stage: Stage[Any, Any], base_hook: Callable[..., Any]) -> bool:
    """Whether this stage decides the contract per STEP. Stage.base states the
    rule: "the compiler consults this method, never the bare attribute". A
    step-free call like describe_stages cannot evaluate it, so publishing the
    class attribute here is guaranteed to be the wrong number wherever the
    hook is overridden — symbol_sync advertised a 2.0 floor while enforcing
    4.0 open-loop and none closed-loop, and ook_envelope advertised an
    amplitude contract that its open-loop path drops. The key is withheld and
    named in step_conditional instead; the stage description carries the rule.

    The base hook arrives as the function object, not its name: a renamed hook
    must then fail to resolve at import instead of quietly reporting False and
    re-publishing the very numbers this guard exists to withhold."""
    return getattr(type(stage), base_hook.__name__) is not base_hook
