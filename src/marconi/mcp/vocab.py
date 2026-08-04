from __future__ import annotations

from typing import Any

from marconi.engine.stages.base import Stage
from marconi.engine.stages.registry import stage_registry

__all__ = ["ENVELOPE", "stage_index", "stage_details"]

ENVELOPE: dict[str, object] = {
    "spec_envelope": {
        "name": "optional str",
        "symbol_rate": ("float > 0 (symbols/second at the demod boundary)"),
        "path": "list of steps, each {'conv': <stage name>, ...stage params}",
    },
    "levels": {
        "iq": "complex baseband samples",
        "symbols": "one item per symbol (hard index or soft value)",
        "bits": "one item per bit (hard u8, or soft float32 LLR)",
        "audio": "demodulated audio floats",
    },
    "item_types": {
        "c": "complex64",
        "s": "int16 hard symbol index",
        "f": (
            "float32 soft value; sign is per level: at bits it is an LLR "
            "where bit 1 = NEGATIVE, at symbols it is a demod output where "
            "POSITIVE slices to bit 1"
        ),
        "b": "uint8 hard bit, one byte per bit",
    },
    "direction": (
        "the same path compiles rx (decode) and tx (generate); "
        "stages declare which directions they support"
    ),
}


def _index_entry(stage: Stage[Any, Any]) -> dict[str, Any]:
    return {
        "name": stage.name,
        "family": stage.family,
        "from_level": stage.from_level.value,
        "to_level": stage.to_level.value,
        "directions": sorted(stage.directions),
    }


def stage_index() -> list[dict[str, Any]]:
    return [_index_entry(s) for _, s in sorted(stage_registry().items())]


def stage_details(names: list[str]) -> list[dict[str, Any]]:
    registry = stage_registry()
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {sorted(registry)}")
    out: list[dict[str, Any]] = []
    for n in names:
        s = registry[n]
        entry = _index_entry(s)
        entry.update(
            {
                "accepts_item_type": (
                    None if s.accepts_item_type is None else s.accepts_item_type.value
                ),
                "accepts_carrier": (
                    None if s.accepts_carrier is None else s.accepts_carrier.value
                ),
                "accepts_amplitude": (
                    None
                    if s.accepts_amplitude is None
                    else sorted(a.value for a in s.accepts_amplitude)
                ),
                "min_input_sps": s.min_input_sps,
                "seeds_windows": s.seeds_windows,
                "params_schema": s.step_model.model_json_schema(),
            }
        )
        out.append(entry)
    return out
