from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class OfdmFrameSyncProbeStep(Step):
    conv: Literal["ofdm_frame_sync_probe"] = "ofdm_frame_sync_probe"
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    null_len: StrictInt
    frame_len: StrictInt
    data_syms: StrictInt


class OfdmFrameSyncProbe(RxStage[CompileContext, OfdmFrameSyncProbeStep]):
    """IQ->IQ: null-sync + CP-strip only (test/diagnostic isolation of the block)."""

    name = "ofdm_frame_sync_probe"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "ofdm"
    step_model = OfdmFrameSyncProbeStep

    def emit_rx(self, b: CompileContext, step: OfdmFrameSyncProbeStep) -> None:
        b.chain(
            "ofdm_frame_sync",
            fft_len=step.fft_len,
            cp_len=step.cp_len,
            sym_len=step.sym_len,
            null_len=step.null_len,
            frame_len=step.frame_len,
            data_syms=step.data_syms,
        )


class OfdmDemodStep(Step):
    conv: Literal["ofdm_demod"] = "ofdm_demod"
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    null_len: StrictInt
    frame_len: StrictInt
    n_frame_syms: StrictInt
    data_syms: StrictInt
    n_carriers: StrictInt
    bin_perm: list[int]


class OfdmDemod(RxStage[CompileContext, OfdmDemodStep]):
    """OFDM demod, IQ->SYMBOLS (RX-only). Custom null-sync + CP-strip, stock FFTW
    fft_vcc, stock carrier permute + select -> symbol-major active carriers. Generic
    over OFDM frames; the carrier permutation is a parameter."""

    name = "ofdm_demod"
    from_level = Level.IQ
    to_level = Level.SYMBOLS
    family = "ofdm"
    step_model = OfdmDemodStep

    def emit_rx(self, b: CompileContext, step: OfdmDemodStep) -> None:
        fft_len = step.fft_len
        b.chain(
            "ofdm_frame_sync",
            fft_len=fft_len,
            cp_len=step.cp_len,
            sym_len=step.sym_len,
            null_len=step.null_len,
            frame_len=step.frame_len,
            data_syms=step.data_syms,
        )
        b.chain("stream_to_vector", vlen=fft_len)
        b.chain("fft_vcc", fft_len=fft_len, forward=True)
        b.chain("vector_to_stream", vlen=fft_len)
        b.chain(
            "blockinterleaver_cc",
            perm=[int(x) for x in step.bin_perm],
            mode=True,
        )
        b.chain("keep_m_in_n_c", m=step.n_carriers, n=fft_len, offset=0)

    def out_descriptor(self, in_desc: Descriptor, step: OfdmDemodStep) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "c", Carrier.SOFT)


class DqpskSoftDemapStep(Step):
    conv: Literal["dqpsk_soft_demap"] = "dqpsk_soft_demap"
    data_syms: StrictInt
    n_carriers: StrictInt
    scheme: str = "psk"
    order: StrictInt = 4


class DqpskSoftDemap(RxStage[CompileContext, DqpskSoftDemapStep]):
    """Differential-QPSK soft demap, SYMBOLS->BITS soft, from STOCK GR blocks. The
    per-carrier differential is delay + multiply_conjugate; the PRS reference is
    dropped by keep_m_in_n; the soft decision is constellation_soft_decoder.
    Generic over differential PSK over symbol-major framed carriers."""

    name = "dqpsk_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "ofdm"
    step_model = DqpskSoftDemapStep
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: DqpskSoftDemapStep) -> None:
        nc = step.n_carriers
        ds = step.data_syms
        src = b.tail  # incoming carrier stream (never None after IO source)
        assert src is not None
        dly = b.chain("delay_cc", samples=nc)  # src -> delay, tail=delay
        mc = b.add("multiply_conjugate_cc")
        b.connect(src, mc, dst_port=0)  # carriers  -> mc.0
        b.connect(dly, mc, dst_port=1)  # delayed   -> mc.1  (c[i]*conj(c[i-nc]))
        b.set_tail(mc)
        b.chain("keep_m_in_n_c", m=ds * nc, n=(ds + 1) * nc, offset=nc)  # drop PRS diff
        b.chain("constellation_soft_decoder", scheme=step.scheme, order=step.order)

    def out_descriptor(
        self, in_desc: Descriptor, step: DqpskSoftDemapStep
    ) -> Descriptor:
        return Descriptor(Level.BITS, "f", Carrier.SOFT)

    def required_input_order(self, step: DqpskSoftDemapStep) -> int | None:
        return int(step.order)


class OfdmCoherentSyncStep(Step):
    conv: Literal["ofdm_coherent_sync"] = "ofdm_coherent_sync"
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
    def _geometry(self) -> "OfdmCoherentSyncStep":
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


class OfdmCoherentSync(RxStage[CompileContext, OfdmCoherentSyncStep]):
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
    step_model = OfdmCoherentSyncStep

    def emit_rx(self, b: CompileContext, step: OfdmCoherentSyncStep) -> None:
        b.chain(
            "cp_symbol_sync",
            fft_len=step.fft_len,
            cp_len=step.cp_len,
            warmup_syms=step.warmup_syms,
        )
        b.chain("stream_to_vector", vlen=step.fft_len)
        b.chain("fft_vcc", fft_len=step.fft_len, shift=True)
        eq = step.model_dump(exclude={"conv", "cp_len", "sym_len"})
        b.chain("pilot_lattice_equalizer", **eq)

    def out_descriptor(
        self, in_desc: Descriptor, step: OfdmCoherentSyncStep
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, "c", Carrier.SOFT)


class SoftDemapStep(Step):
    conv: Literal["soft_demap"] = "soft_demap"
    scheme: str = "explicit"
    order: StrictInt | None = None
    points_i: list[float] | None = None
    points_q: list[float] | None = None

    @model_validator(mode="after")
    def _shaped(self) -> "SoftDemapStep":
        if self.scheme in ("psk", "qam"):
            if (
                self.order is None
                or self.points_i is not None
                or self.points_q is not None
            ):
                raise PydanticCustomError(
                    "value_error",
                    "named schemes take order, not points",
                )
            if self.scheme == "qam" and self.order != 16:
                raise PydanticCustomError(
                    "value_error",
                    "qam soft decode supports order 16 only: GR's 64-QAM "
                    "constellations emit collapsed soft decisions (measured: 64 "
                    "clean points decode to 16 distinct values)",
                )
        elif self.scheme == "explicit":
            if self.order is not None or self.points_i is None or self.points_q is None:
                raise PydanticCustomError(
                    "value_error",
                    "explicit takes points_i/points_q, not order",
                )
            n = len(self.points_i)
            if len(self.points_q) != n:
                raise PydanticCustomError(
                    "value_error", "points_i and points_q must be equal length"
                )
            if n < 2 or n & (n - 1):
                raise PydanticCustomError(
                    "value_error",
                    "explicit needs a power-of-two point count >= 2; a "
                    "point's bit pattern is its index",
                )
        else:
            raise PydanticCustomError("value_error", "scheme must be psk|qam|explicit")
        return self

    def alphabet(self) -> int:
        return int(self.order) if self.order is not None else len(self.points_i or [])


class SoftDemap(RxStage[CompileContext, SoftDemapStep]):
    """Constellation soft demap, SYMBOLS->BITS soft, for any constellation:
    named psk/qam schemes or explicit caller points (a point's bit pattern is
    its index, MSB-first; points are power-normalized so input must be unit
    RMS). Negates constellation_soft_decoder's LLR to this engine's
    bit-1-negative convention. The generic door from a cell stream into the
    soft coding lane (deinterleave/depuncture/fec)."""

    name = "soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "ofdm"
    step_model = SoftDemapStep
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: SoftDemapStep) -> None:
        if step.scheme == "explicit":
            b.chain(
                "constellation_soft_decoder",
                scheme=step.scheme,
                points_i=[float(x) for x in step.points_i or []],
                points_q=[float(x) for x in step.points_q or []],
            )
        else:
            b.chain(
                "constellation_soft_decoder", scheme=step.scheme, order=step.alphabet()
            )
        b.chain("multiply_const_ff", value=-1.0)

    def out_descriptor(self, in_desc: Descriptor, step: SoftDemapStep) -> Descriptor:
        k = step.alphabet().bit_length() - 1
        frame = None if in_desc.frame_len is None else in_desc.frame_len * k
        return Descriptor(Level.BITS, "f", Carrier.SOFT, frame_len=frame)

    def required_input_order(self, step: SoftDemapStep) -> int | None:
        return step.alphabet()


class CellSelectStep(Step):
    conv: Literal["cell_select"] = "cell_select"
    select_perm: list[int]
    keep: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def _shaped(self) -> "CellSelectStep":
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


class CellSelect(RxStage[CompileContext, CellSelectStep]):
    """SYMBOLS(c)->SYMBOLS(c) cell gather: reorder each fixed-size symbol
    block by a whole-block permutation (stock blockinterleaver_cc) with the
    wanted cells at the front, then keep that front (keep_m_in_n). Which
    cells matter is the caller's anatomy — the perm and keep count are
    params. Pins frame_len=keep so downstream frame geometry is checked."""

    name = "cell_select"
    from_level = Level.SYMBOLS
    to_level = Level.SYMBOLS
    family = "ofdm"
    step_model = CellSelectStep
    accepts_item_type = "c"
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: CellSelectStep) -> None:
        b.chain(
            "blockinterleaver_cc",
            perm=[int(x) for x in step.select_perm],
            mode=True,
        )
        b.chain("keep_m_in_n_c", m=int(step.keep), n=len(step.select_perm))

    def out_descriptor(self, in_desc: Descriptor, step: CellSelectStep) -> Descriptor:
        return replace(in_desc, frame_len=int(step.keep))

    def rate_factor(self, step: CellSelectStep) -> float:
        return step.keep / len(step.select_perm)

    def validate_input(self, in_desc: Descriptor, step: CellSelectStep) -> str | None:
        span = len(step.select_perm)
        if in_desc.frame_len is not None:
            # Accept if either divides the other; reject only if both straddle
            if span % in_desc.frame_len != 0 and in_desc.frame_len % span != 0:
                return (
                    f"gathers blocks of {span} cells but the input is framed at "
                    f"{in_desc.frame_len}; neither divides the other, so gather "
                    f"blocks would straddle frame boundaries"
                )
        return None


OFDM_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    OfdmDemod,
    DqpskSoftDemap,
    OfdmFrameSyncProbe,
    OfdmCoherentSync,
    CellSelect,
    SoftDemap,
)
