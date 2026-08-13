from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.ofdm.primitives import LOCK_MIN_RATIO_DEFAULT
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.perm import check_block_permutation
from marconi.engine.types.step import Step


def _frame_geometry_errors(
    *,
    fft_len: int,
    cp_len: int,
    sym_len: int,
    null_len: int,
    frame_len: int,
    data_syms: int,
) -> list[str]:
    """The null-sync + CP-strip contract both RX-only OFDM steps compile to.
    Each of these reached the embedded block as a live parameter: a zero
    frame_len leaves its resync branch a no-op, so the buffer never trims and
    general_work spins until the run's deadline; a zero or negative null_len
    convolves against an empty kernel; a sym_len that is not fft_len + cp_len
    was accepted in silence and then stripped the CP at the wrong stride,
    emitting garbage under status ok."""
    checks = {
        "fft_len must be positive": fft_len > 0,
        "cp_len must be non-negative": cp_len >= 0,
        "null_len must be positive": null_len > 0,
        "frame_len must be positive": frame_len > 0,
        "data_syms must be positive": data_syms > 0,
        "sym_len must equal fft_len + cp_len": sym_len == fft_len + cp_len,
        "frame_len must hold the null and its data symbols": frame_len
        >= null_len + (data_syms + 1) * sym_len,
    }
    return [msg for msg, ok in checks.items() if not ok]


class OfdmFrameSyncProbeStep(Step):
    conv: Literal["ofdm_frame_sync_probe"] = "ofdm_frame_sync_probe"
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    null_len: StrictInt
    frame_len: StrictInt
    data_syms: StrictInt

    @model_validator(mode="after")
    def _geometry(self) -> "OfdmFrameSyncProbeStep":
        bad = _frame_geometry_errors(
            fft_len=self.fft_len,
            cp_len=self.cp_len,
            sym_len=self.sym_len,
            null_len=self.null_len,
            frame_len=self.frame_len,
            data_syms=self.data_syms,
        )
        if bad:
            raise ValueError("; ".join(bad))
        return self


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

    def output_item_rate(
        self, step: OfdmFrameSyncProbeStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        usefuls = (step.data_syms + 1) * step.fft_len
        return in_rate * usefuls / step.frame_len


class OfdmDemodStep(Step):
    conv: Literal["ofdm_demod"] = "ofdm_demod"
    fft_len: StrictInt
    cp_len: StrictInt
    sym_len: StrictInt
    null_len: StrictInt
    frame_len: StrictInt
    data_syms: StrictInt
    n_carriers: StrictInt
    bin_perm: list[int]

    @model_validator(mode="after")
    def _geometry(self) -> "OfdmDemodStep":
        bad = _frame_geometry_errors(
            fft_len=self.fft_len,
            cp_len=self.cp_len,
            sym_len=self.sym_len,
            null_len=self.null_len,
            frame_len=self.frame_len,
            data_syms=self.data_syms,
        )
        if not 0 < self.n_carriers <= self.fft_len:
            bad.append("n_carriers must be positive and no wider than fft_len")
        if len(self.bin_perm) != self.fft_len:
            bad.append("bin_perm needs one entry per FFT bin")
        if bad:
            raise ValueError("; ".join(bad))
        check_block_permutation(self.bin_perm, field="bin_perm")
        return self


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
        return Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT)

    def output_item_rate(
        self, step: OfdmDemodStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        cells = (step.data_syms + 1) * step.n_carriers
        return in_rate * cells / step.frame_len


def _check_explicit_points(
    points_i: list[float] | None, points_q: list[float] | None
) -> None:
    if points_i is None or points_q is None:
        raise PydanticCustomError(
            "value_error", "explicit takes points_i/points_q, not order"
        )
    if len(points_q) != len(points_i):
        raise PydanticCustomError(
            "value_error", "points_i and points_q must be equal length"
        )
    n = len(points_i)
    if n < 2 or n & (n - 1):
        raise PydanticCustomError(
            "value_error",
            "explicit needs a power-of-two point count >= 2; a "
            "point's bit pattern is its index",
        )


class DqpskSoftDemapStep(Step):
    conv: Literal["dqpsk_soft_demap"] = "dqpsk_soft_demap"
    data_syms: StrictInt
    n_carriers: StrictInt
    scheme: Literal["psk", "explicit"] = "psk"
    order: StrictInt = 4
    points_i: list[float] | None = None
    points_q: list[float] | None = None

    @model_validator(mode="after")
    def _shaped(self) -> "DqpskSoftDemapStep":
        if self.scheme == "psk":
            if self.points_i is not None or self.points_q is not None:
                raise PydanticCustomError(
                    "value_error", "named schemes take order, not points"
                )
        elif self.scheme == "explicit":
            _check_explicit_points(self.points_i, self.points_q)
        else:
            raise PydanticCustomError("value_error", "scheme must be psk|explicit")
        return self

    def alphabet(self) -> int:
        return len(self.points_i) if self.points_i is not None else int(self.order)


class DqpskSoftDemap(RxStage[CompileContext, DqpskSoftDemapStep]):
    """Differential-QPSK soft demap, SYMBOLS->BITS soft, from STOCK GR blocks. The
    per-carrier differential is delay + multiply_conjugate; the leading reference
    dropped by keep_m_in_n; the soft decision is constellation_soft_decoder over
    the named psk scheme or explicit caller points (a point's bit pattern is its
    index, MSB-first — the door for protocols whose differential mapping differs
    from GR's stock constellation). Negates the LLR to this engine's
    bit-1-negative convention. Generic over differential PSK over symbol-major
    framed carriers."""

    name = "dqpsk_soft_demap"
    from_level = Level.SYMBOLS
    to_level = Level.BITS
    family = "ofdm"
    step_model = DqpskSoftDemapStep
    accepts_item_type = ItemType.C
    accepts_carrier = Carrier.SOFT

    def emit_rx(self, b: CompileContext, step: DqpskSoftDemapStep) -> None:
        nc = step.n_carriers
        ds = step.data_syms
        src = b.require_tail()
        dly = b.chain("delay_cc", samples=nc)
        mc = b.add("multiply_conjugate_cc")
        # c[i] * conj(c[i-nc]): differential across one carrier spacing
        b.connect(src, mc, dst_port=0)
        b.connect(dly, mc, dst_port=1)
        b.set_tail(mc)
        b.chain("keep_m_in_n_c", m=ds * nc, n=(ds + 1) * nc, offset=nc)
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
        b.chain_llr_flip()

    def out_descriptor(
        self, in_desc: Descriptor, step: DqpskSoftDemapStep
    ) -> Descriptor:
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT)

    def required_input_order(self, step: DqpskSoftDemapStep) -> int | None:
        return step.alphabet()

    def output_item_rate(
        self, step: DqpskSoftDemapStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        # reference symbol dropped, then log2(alphabet) LLRs per kept carrier
        kept = step.data_syms / (step.data_syms + 1)
        return in_rate * kept * math.log2(step.alphabet())


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
    lock_min_ratio: float = Field(default=LOCK_MIN_RATIO_DEFAULT, ge=0.0)
    lock_min_score: float = Field(default=0.35, ge=0.0)

    @model_validator(mode="after")
    def _geometry(self) -> "OfdmCoherentSyncStep":
        n = sum(self.pilot_lens)
        dc0 = self.fft_len // 2
        span = [self.kmin, self.kmin + self.n_carriers]
        lo = min(span + self.pilot_carriers + self.fp_carriers)
        hi = max(span + self.pilot_carriers + self.fp_carriers)
        checks = {
            "carrier bins must stay inside the FFT across the DC search "
            "(an out-of-FFT pilot bin starves its channel node)": (
                lo + dc0 - self.dc_search >= 0
                and hi + dc0 + self.dc_search < self.fft_len
            ),
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
            lock_min_ratio=step.lock_min_ratio,
        )
        b.chain("stream_to_vector", vlen=step.fft_len)
        b.chain("fft_vcc", fft_len=step.fft_len, shift=True)
        b.chain(
            "pilot_lattice_equalizer",
            fft_len=step.fft_len,
            n_frame_syms=step.n_frame_syms,
            n_carriers=step.n_carriers,
            kmin=step.kmin,
            dc_search=step.dc_search,
            warmup_syms=step.warmup_syms,
            pilot_lens=[int(x) for x in step.pilot_lens],
            pilot_carriers=[int(x) for x in step.pilot_carriers],
            pilot_i=step.pilot_i,
            pilot_q=step.pilot_q,
            fp_carriers=[int(x) for x in step.fp_carriers],
            fp_i=step.fp_i,
            fp_q=step.fp_q,
            lock_min_score=step.lock_min_score,
        )

    def out_descriptor(
        self, in_desc: Descriptor, step: OfdmCoherentSyncStep
    ) -> Descriptor:
        return Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT)

    def output_item_rate(
        self, step: OfdmCoherentSyncStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return in_rate * step.n_carriers / step.sym_len


class SoftDemapStep(Step):
    conv: Literal["soft_demap"] = "soft_demap"
    scheme: Literal["psk", "qam", "explicit"] = "explicit"
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
            if self.order is not None:
                raise PydanticCustomError(
                    "value_error",
                    "explicit takes points_i/points_q, not order",
                )
            _check_explicit_points(self.points_i, self.points_q)
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
    accepts_item_type = ItemType.C
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
        b.chain_llr_flip()

    def out_descriptor(self, in_desc: Descriptor, step: SoftDemapStep) -> Descriptor:
        k = step.alphabet().bit_length() - 1
        frame = None if in_desc.frame_len is None else in_desc.frame_len * k
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT, frame_len=frame)

    def required_input_order(self, step: SoftDemapStep) -> int | None:
        return step.alphabet()

    def output_item_rate(
        self, step: SoftDemapStep, in_rate: float, symbol_rate: float
    ) -> float | None:
        return in_rate * (step.alphabet().bit_length() - 1)


class CellSelectStep(Step):
    conv: Literal["cell_select"] = "cell_select"
    select_perm: list[int]
    keep: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def _shaped(self) -> "CellSelectStep":
        check_block_permutation(self.select_perm, field="select_perm")
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
    accepts_item_type = ItemType.C
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
