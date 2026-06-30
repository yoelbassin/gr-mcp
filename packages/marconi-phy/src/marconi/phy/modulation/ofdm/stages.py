from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import StrictInt

from marconi.core.descriptor import Carrier, Descriptor
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


class _OfdmParams(StageParams):
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    null_len: StrictInt
    frame_len: StrictInt
    n_frame_syms: StrictInt
    data_syms: StrictInt
    n_carriers: StrictInt
    bin_perm: list[int]


class OfdmDemod(RxStage[CompileContext]):
    """OFDM demod, IQ->SYMBOLS (RX-only). Custom null-sync + CP-strip, stock FFTW
    fft_vcc, stock carrier permute + select -> symbol-major active carriers. Generic
    over OFDM frames; the carrier permutation is a parameter."""

    name = "ofdm_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "ofdm"
    params_model = _OfdmParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        fft_len = int(params["fft_len"])
        b.chain(
            "ofdm_frame_sync",
            fft_len=fft_len,
            cp_len=int(params["cp_len"]),
            sym_len=int(params["sym_len"]),
            null_len=int(params["null_len"]),
            frame_len=int(params["frame_len"]),
            data_syms=int(params["data_syms"]),
        )
        b.chain("stream_to_vector", vlen=fft_len)
        b.chain("fft_vcc", fft_len=fft_len, forward=True)
        b.chain("vector_to_stream", vlen=fft_len)
        b.chain(
            "blockinterleaver_cc", perm=[int(x) for x in params["bin_perm"]], mode=True
        )
        b.chain("keep_m_in_n_c", m=int(params["n_carriers"]), n=fft_len, offset=0)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "c", in_desc.layout, Carrier.HARD)


OFDM_STAGES: tuple[type[Stage[CompileContext]], ...] = (OfdmDemod, OfdmFrameSyncProbe)
