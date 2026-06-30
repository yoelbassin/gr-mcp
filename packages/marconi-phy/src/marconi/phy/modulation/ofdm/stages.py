from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt

from marconi.core.levels import Level
from marconi.core.params import StageParams
from marconi.core.stages import RxStage, Stage
from marconi.phy.compile_context import CompileContext


class _SyncParams(StageParams):
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    null_len: StrictInt
    frame_len: StrictInt
    data_syms: StrictInt


class OfdmFrameSyncProbe(RxStage[CompileContext]):
    """IQ->IQ: null-sync + CP-strip only (test/diagnostic isolation of the block)."""

    name = "ofdm_frame_sync_probe"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "ofdm"
    params_model = _SyncParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        b.chain(
            "ofdm_frame_sync",
            fft_len=int(params["fft_len"]),
            cp_len=int(params["cp_len"]),
            sym_len=int(params["sym_len"]),
            null_len=int(params["null_len"]),
            frame_len=int(params["frame_len"]),
            data_syms=int(params["data_syms"]),
        )


OFDM_STAGES: tuple[type[Stage[CompileContext]], ...] = (OfdmFrameSyncProbe,)
