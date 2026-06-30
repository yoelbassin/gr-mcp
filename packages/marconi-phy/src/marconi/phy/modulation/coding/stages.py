from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage
from marconi.phy.compile_context import CompileContext


class _DeinterleaveParams(StageParams):
    perm: list[int]


class Deinterleave(RxStage[CompileContext]):
    """Generic block permute, BITS->BITS (carrier passes through). Stock
    blockinterleaver_ff: out[t]=in[perm[t]] per block of len(perm)."""

    name = "deinterleave"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _DeinterleaveParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("blockinterleaver_ff", perm=[int(x) for x in params["perm"]], mode=True)


class _DepunctureParams(StageParams):
    keep_mask: list[int]


class Depuncture(RxStage[CompileContext]):
    """Generic depuncture, BITS->BITS soft. Scatters soft into a wider codeword per
    a keep-mask (0 = erasure). Protocol puncture tables are params."""

    name = "depuncture"
    from_level = Level.BITS
    to_level = Level.BITS
    family = "coding"
    params_model = _DepunctureParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain("depuncture", keep_mask=[int(x) for x in params["keep_mask"]])


CODING_STAGES: tuple[type[Stage[CompileContext]], ...] = (Deinterleave, Depuncture)
