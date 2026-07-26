from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.params import StageParams


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
        return Descriptor(Level.SYMBOLS, "c", Carrier.SOFT)


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
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

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
        return Descriptor(Level.BITS, "f", Carrier.SOFT)

    def required_input_order(self, params: Mapping[str, Any]) -> int | None:
        return int(_DqpskParams.model_validate(dict(params)).order)


class _CoherentParams(StageParams):
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    n_frame_syms: StrictInt
    n_carriers: StrictInt
    kmin: StrictInt
    dc_search: StrictInt
    warmup_syms: StrictInt
    pilot_lens: list[int]
    pilot_carriers: list[int]
    pilot_i: list[float]
    pilot_q: list[float]
    fp_carriers: list[int]
    fp_i: list[float]
    fp_q: list[float]

    @model_validator(mode="after")
    def _geometry(self) -> "_CoherentParams":
        n = sum(self.pilot_lens)
        checks = {
            "sym_len must equal fft_len + cp_len": self.sym_len
            == self.fft_len + self.cp_len,
            "pilot_lens needs one entry per frame symbol": len(self.pilot_lens)
            == self.n_frame_syms,
            "pilot arrays must match sum(pilot_lens)": len(self.pilot_carriers)
            == len(self.pilot_i)
            == len(self.pilot_q)
            == n,
            "fp arrays must be equal length": len(self.fp_carriers)
            == len(self.fp_i)
            == len(self.fp_q),
            "n_carriers must be positive": self.n_carriers > 0,
            "carrier span must straddle DC (kmin <= 0 <= kmin + n_carriers)": self.kmin
            <= 0
            <= self.kmin + self.n_carriers,
            "warmup_syms must exceed one frame": self.warmup_syms > self.n_frame_syms,
        }
        bad = [msg for msg, ok in checks.items() if not ok]
        if bad:
            raise ValueError("; ".join(bad))
        return self


class OfdmCoherentSync(RxStage[CompileContext]):
    """Coherent scattered-pilot OFDM demod, IQ->SYMBOLS (RX-only). Streaming
    CP-correlation symbol tracker, stock vectorize + FFT, and a scattered-pilot
    lattice equalizer: fine-CFO derotation off the frequency pilots, 2-D
    channel estimation, and equalization to symbol-major active carriers.
    Generic over the OFDM geometry and pilot lattice; the geometry, the
    scattered/frequency pilot carriers, and their reference values are all
    parameters."""

    name = "ofdm_coherent_sync"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "ofdm"
    params_model = _CoherentParams

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CoherentParams.model_validate(dict(params))
        b.chain(
            "cp_symbol_sync",
            fft_len=p.fft_len,
            cp_len=p.cp_len,
            warmup_syms=p.warmup_syms,
        )
        b.chain("stream_to_vector", vlen=p.fft_len)
        b.chain("fft_vcc", fft_len=p.fft_len, shift=True)
        eq = p.model_dump()
        for k in ("cp_len", "sym_len"):
            del eq[k]
        b.chain("pilot_lattice_equalizer", **eq)

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "c", Carrier.SOFT)


class _CellSelectParams(StageParams):
    select_perm: list[int]
    keep: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def _shaped(self) -> "_CellSelectParams":
        if sorted(self.select_perm) != list(range(len(self.select_perm))):
            raise PydanticCustomError(
                "value_error",
                "select_perm must be a permutation of 0..len-1; the gather "
                "walks one whole block per output block",
            )
        if self.keep > len(self.select_perm):
            raise PydanticCustomError(
                "value_error",
                "keep must be <= len(select_perm); cannot keep more cells "
                "than a block holds",
            )
        return self


class CellSelect(RxStage[CompileContext]):
    """SYMBOLS(c)->SYMBOLS(c) cell gather: reorder each fixed-size symbol
    block by a whole-block permutation (stock blockinterleaver_cc) with the
    wanted cells at the front, then keep that front (keep_m_in_n). Which
    cells matter is the caller's anatomy — the perm and keep count are
    params. Pins frame_len=keep so downstream frame geometry is checked."""

    name = "cell_select"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "ofdm"
    params_model = _CellSelectParams
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, params: Mapping[str, Any]) -> None:
        p = _CellSelectParams.model_validate(dict(params))
        b.chain(
            "blockinterleaver_cc",
            perm=[int(x) for x in p.select_perm],
            mode=True,
        )
        b.chain("keep_m_in_n_c", m=int(p.keep), n=len(p.select_perm))

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        return replace(in_desc, frame_len=int(params["keep"]))

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        p = _CellSelectParams.model_validate(dict(params))
        return p.keep / len(p.select_perm)

    def validate_input(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> str | None:
        span = len(list(params["select_perm"]))
        if in_desc.frame_len is not None and span % in_desc.frame_len:
            return (
                f"gathers blocks of {span} cells but the input is framed at "
                f"{in_desc.frame_len}, which does not divide {span}; gather "
                f"blocks would straddle frame boundaries"
            )
        return None


OFDM_STAGES: tuple[type[Stage[CompileContext]], ...] = (
    OfdmDemod,
    DqpskSoftDemap,
    OfdmFrameSyncProbe,
    OfdmCoherentSync,
    CellSelect,
)
