from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage


class Parse(DuplexStage[ProgramBuilder]):
    name = "parse"
    from_level = Level.FRAMES
    to_level = Level.MESSAGES
    family = "message"

    class _Params(StageParams):
        fields: list
        bit_order: str = "msb"
        discriminator: str | None = None
        cases: list = []

    params_model = _Params

    def _kw(self, params: Mapping[str, Any]) -> dict[str, Any]:
        fields = framing.parse_fields(params["fields"])
        discriminator = params.get("discriminator")
        cases: dict[int, tuple[Any, list[Any]]] = {}
        for case in params.get("cases") or []:
            body = fields + framing.parse_fields(case["fields"])
            cases[int(case["when"])] = (framing.build_struct(body), body)
        return {
            "struct": framing.build_struct(fields),
            "fields": fields,
            "bit_order": str(params.get("bit_order", "msb")),
            "discriminator": str(discriminator) if discriminator else None,
            "cases": cases or None,
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.parse_rx, **self._kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.parse_tx, **self._kw(params))


MESSAGE_STAGES: tuple[type[DuplexStage[ProgramBuilder]], ...] = (Parse,)
