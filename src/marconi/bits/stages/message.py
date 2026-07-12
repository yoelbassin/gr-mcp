from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.bits import framing
from marconi.bits.builder import ProgramBuilder
from marconi.bits.models import ParseField
from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import DuplexStage


class Parse(DuplexStage[ProgramBuilder]):
    name = "parse"
    from_level = Level.FRAMES
    to_level = Level.MESSAGES
    family = "message"

    class _Params(StageParams):
        fields: list[ParseField]
        bit_order: str = "msb"
        discriminator: str | None = None
        cases: list = []

    params_model = _Params

    @staticmethod
    def _reject_misplaced_rest(fields: list[ParseField]) -> None:
        positions = [i for i, f in enumerate(fields) if f.rest]
        if positions and positions != [len(fields) - 1]:
            raise ValueError("only one rest field is allowed and it must be last")

    def _kw(self, params: Mapping[str, Any]) -> dict[str, Any]:
        p = self._Params.model_validate(dict(params))
        fields = framing.parse_fields(p.fields)
        self._reject_misplaced_rest(fields)
        cases: dict[int, tuple[Any, list[Any]]] = {}
        for case in p.cases:
            case_fields = framing.parse_fields(case["fields"])
            if any(f.rest for f in case_fields):
                raise ValueError("rest fields are not allowed inside cases")
            body = fields + case_fields
            self._reject_misplaced_rest(body)
            cases[int(case["when"])] = (framing.build_struct(body), body)
        return {
            "struct": framing.build_struct(fields),
            "fields": fields,
            "bit_order": p.bit_order,
            "discriminator": p.discriminator or None,
            "cases": cases or None,
        }

    def emit_rx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.parse_rx, **self._kw(params))

    def emit_tx(self, b: ProgramBuilder, params: Mapping[str, Any]) -> None:
        b.add(framing.parse_tx, **self._kw(params))


MESSAGE_STAGES: tuple[type[DuplexStage[ProgramBuilder]], ...] = (Parse,)
