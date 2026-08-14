from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.modulation.ofdm.primitives import (
    LOCK_MIN_RATIO_DEFAULT,
    carrier_bin_problem,
)
from marconi.engine.stages.base import Stage
from marconi.engine.types.bounds import MAX_DELAY_ITEMS, MAX_FRAME_ITEMS
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
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
    frame_len: StrictInt = Field(le=MAX_FRAME_ITEMS)
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


class OfdmFrameSyncProbe(Stage[CompileContext, OfdmFrameSyncProbeStep]):
    """IQ->IQ: null-sync + CP-strip only (test/diagnostic isolation of the block)."""

    name = "ofdm_frame_sync_probe"
    description = (
        "Observability probe: reports OFDM frame-sync candidates (null/CP "
        "correlation) without demodulating - for measuring geometry before "
        "committing to a demod spec."
    )
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
    frame_len: StrictInt = Field(le=MAX_FRAME_ITEMS)
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


class OfdmDemod(Stage[CompileContext, OfdmDemodStep]):
    """OFDM demod, IQ->SYMBOLS (RX-only). Custom null-sync + CP-strip, stock FFTW
    fft_vcc, stock carrier permute + select -> symbol-major active carriers. Generic
    over OFDM frames; the carrier permutation is a parameter."""

    name = "ofdm_demod"
    description = (
        "Non-coherent OFDM demod: CP-correlation symbol sync, FFT, and bin_perm "
        "carrier extraction to soft symbols. For pilot-based coherent "
        "equalization use ofdm_coherent_sync."
    )
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


_NAMED_SCHEME_ORDERS: dict[str, frozenset[int]] = {
    "psk": frozenset(int(o) for o in PskOrder),
    "qam": frozenset({16}),
}


def _check_named_order(scheme: str, order: int) -> None:
    """A named scheme resolves to a stock GR constellation, and only a few
    orders have one. Unchecked, the order reached _const_psk in the worker and
    came back as a BackendError naming a block id, from a spec validate_modem
    had called valid."""
    supported = _NAMED_SCHEME_ORDERS[scheme]
    if order not in supported:
        raise PydanticCustomError(
            "value_error",
            "scheme {scheme!r} supports order {supported}, got {order}",
            {"scheme": scheme, "supported": sorted(supported), "order": order},
        )


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
    data_syms: StrictInt = Field(ge=1, le=MAX_FRAME_ITEMS)
    # n_carriers sizes a delay_cc line, which is allocated up front
    n_carriers: StrictInt = Field(ge=1, le=MAX_DELAY_ITEMS)
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
            _check_named_order(self.scheme, self.order)
        elif self.scheme == "explicit":
            _check_explicit_points(self.points_i, self.points_q)
        else:
            raise PydanticCustomError("value_error", "scheme must be psk|explicit")
        return self

    def alphabet(self) -> int:
        return len(self.points_i) if self.points_i is not None else int(self.order)


class DqpskSoftDemap(Stage[CompileContext, DqpskSoftDemapStep]):
    """Differential-QPSK soft demap, SYMBOLS->BITS soft, from STOCK GR blocks. The
    per-carrier differential is delay + multiply_conjugate; the leading reference
    dropped by keep_m_in_n; the soft decision is constellation_soft_decoder over
    the named psk scheme or explicit caller points (a point's bit pattern is its
    index, MSB-first — the door for protocols whose differential mapping differs
    from GR's stock constellation). Negates the LLR to this engine's
    bit-1-negative convention. Generic over differential PSK over symbol-major
    framed carriers."""

    name = "dqpsk_soft_demap"
    description = (
        "Differential QPSK soft demap across OFDM symbols (reference symbol "
        "dropped): equalizer-free DQPSK-OFDM to LLRs."
    )
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
        # the reference is a whole dropped FRAME, not items within one, so
        # the per-frame geometry scales by bits-per-symbol alone
        k = step.alphabet().bit_length() - 1
        frame = None if in_desc.frame_len is None else in_desc.frame_len * k
        return Descriptor(Level.BITS, ItemType.F, Carrier.SOFT, frame_len=frame)

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
    fft_len: StrictInt = Field(ge=1)
    # ge=1, NOT the sibling's ge=0: this step compiles to cp_symbol_sync, whose
    # acquisition correlates over np.ones(cp_len), and numpy refuses an empty
    # kernel ("v cannot be empty") - so a zero CP validated here and then killed
    # the worker, and a negative one died a line earlier on a broadcast
    # mismatch. A CP-correlation synchronizer with no CP is meaningless, not
    # merely degenerate. The sibling's floor does not transfer: ofdm_frame_sync
    # uses cp_len as a slice offset and tol=max(1, cp_len // 2), and survives 0.
    cp_len: StrictInt = Field(ge=1)
    sym_len: StrictInt
    # ge=1: at zero the equalizer is structurally dead, not merely wrong - the
    # phase search loops over range(0) so its score stays -1.0, no
    # lock_min_score can be cleared, and the block emits nothing under status ok
    n_frame_syms: StrictInt = Field(ge=1)
    n_carriers: StrictInt = Field(ge=1)
    kmin: StrictInt
    # ge=0: range(-dc_search, dc_search+1) is EMPTY for a negative value, so
    # the out-of-FFT guard inverted and _try_lock crashed on an empty argmin
    dc_search: StrictInt = Field(ge=0)
    # ge=8: LOCK_MIN_RATIO_DEFAULT's noise calibration only holds from 8
    # warmup symbols up (see primitives.py) - below it the ratio's noise
    # distribution rises over the bar and pure AWGN locks
    warmup_syms: StrictInt = Field(ge=8, le=MAX_FRAME_ITEMS)
    pilot_lens: list[int]
    pilot_carriers: list[int]
    pilot_i: list[float]
    pilot_q: list[float]
    fp_carriers: list[int]
    fp_i: list[float]
    fp_q: list[float]
    # gt=0.0: zero clears every significance bar (ratio < 0.0 is never true,
    # so a zero-bar block locks unconditionally - on pure AWGN, on anything)
    lock_min_ratio: float = Field(default=LOCK_MIN_RATIO_DEFAULT, gt=0.0)
    lock_min_score: float = Field(default=0.35, gt=0.0)

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
            "carrier span must straddle DC (kmin <= 0 <= kmin + n_carriers)": self.kmin
            <= 0
            <= self.kmin + self.n_carriers,
            "warmup_syms must exceed one frame": self.warmup_syms > self.n_frame_syms,
            "the acquisition buffer warmup_syms * sym_len + fft_len + cp_len "
            f"must stay within {MAX_FRAME_ITEMS} items: cp_symbol_sync fills "
            "it before it can attempt a lock, and no single field's ceiling "
            "expresses the product": (
                self.warmup_syms * self.sym_len + self.fft_len + self.cp_len
                <= MAX_FRAME_ITEMS
            ),
        }
        bad = [msg for msg, ok in checks.items() if not ok]
        bins = carrier_bin_problem(
            fft_len=self.fft_len,
            n_carriers=self.n_carriers,
            kmin=self.kmin,
            dc_search=self.dc_search,
            pilot_carriers=self.pilot_carriers,
            fp_carriers=self.fp_carriers,
        )
        if bins is not None:
            bad.append(bins)
        if bad:
            raise ValueError("; ".join(bad))
        return self


class OfdmCoherentSync(Stage[CompileContext, OfdmCoherentSyncStep]):
    """Coherent scattered-pilot OFDM demod, IQ->SYMBOLS (RX-only). Streaming
    CP-correlation symbol tracker, stock vectorize + FFT, and a scattered-pilot
    lattice equalizer: fine-CFO derotation off the frequency pilots, 2-D
    channel estimation, and equalization to symbol-major active carriers.
    Generic over the OFDM geometry and pilot lattice; the geometry, the
    scattered/frequency pilot carriers, and their reference values are all
    parameters."""

    name = "ofdm_coherent_sync"
    description = (
        "Coherent scattered-pilot OFDM front end: CP timing, fine CFO off the "
        "frequency pilots, 2-D channel estimate, equalized symbol-major carriers "
        "out. Geometry and pilot lattice are all parameters."
    )
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
            _check_named_order(self.scheme, self.order)
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


class SoftDemap(Stage[CompileContext, SoftDemapStep]):
    """Constellation soft demap, SYMBOLS->BITS soft, for any constellation:
    named psk/qam schemes or explicit caller points (a point's bit pattern is
    its index, MSB-first; points are power-normalized so input must be unit
    RMS). Negates constellation_soft_decoder's LLR to this engine's
    bit-1-negative convention. The generic door from a cell stream into the
    soft coding lane (deinterleave/depuncture/fec)."""

    name = "soft_demap"
    description = (
        "Constellation soft demap: equalized complex cells to per-bit LLRs (named "
        "scheme+order, or explicit points). Input must be unit-RMS."
    )
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


class CellSelect(Stage[CompileContext, CellSelectStep]):
    """SYMBOLS(c)->SYMBOLS(c) cell gather: reorder each fixed-size symbol
    block by a whole-block permutation (stock blockinterleaver_cc) with the
    wanted cells at the front, then keep that front (keep_m_in_n). Which
    cells matter is the caller's anatomy — the perm and keep count are
    params. Pins frame_len=keep so downstream frame geometry is checked."""

    name = "cell_select"
    description = (
        "Keep a subset of equalized OFDM cells per symbol (keep= carrier indices) "
        "and pin the frame geometry for downstream FEC."
    )
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
