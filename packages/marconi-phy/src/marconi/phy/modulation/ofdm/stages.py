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


class _DqpskParams(StageParams):
    data_syms: StrictInt
    n_carriers: StrictInt
    scheme: str = "psk"
    order: StrictInt = 4


class DqpskSoftDemap(RxStage[CompileContext]):
    """Differential-QPSK soft demap, SYMBOLS->BITS soft, from STOCK GR blocks. The
    per-carrier differential is delay + multiply_conjugate; the PRS reference is
    dropped by keep_m_in_n; the soft decision is constellation_soft_decoder.
    Generic over differential PSK over symbol-major framed carriers."""

    name = "dqpsk_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "ofdm"
    params_model = _DqpskParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _DqpskParams.model_validate(dict(params))
        nc = p.n_carriers
        ds = p.data_syms
        src = b.tail  # incoming carrier stream (never None after IO source)
        assert src is not None
        dly = b.chain("delay_cc", samples=nc)  # src -> delay, tail=delay
        mc = b.add("multiply_conjugate_cc")
        b.connect(src, mc, dst_port=0)  # carriers  -> mc.0
        b.connect(dly, mc, dst_port=1)  # delayed   -> mc.1  (c[i]*conj(c[i-nc]))
        b.set_tail(mc)
        b.chain("keep_m_in_n_c", m=ds * nc, n=(ds + 1) * nc, offset=nc)  # drop PRS diff
        b.chain("constellation_soft_decoder", scheme=p.scheme, order=p.order)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.BITS, "f", in_desc.layout, Carrier.SOFT)


OFDM_STAGES: tuple[type[Stage[CompileContext]], ...] = (
    OfdmDemod,
    DqpskSoftDemap,
    OfdmFrameSyncProbe,
)
