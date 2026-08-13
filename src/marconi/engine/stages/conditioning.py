from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import Stage
from marconi.engine.types.bounds import (
    MAX_DECIM,
    MAX_FILTER_TAPS,
    MAX_FRAME_ITEMS,
    MAX_RESAMPLE_FACTOR,
    MAX_WINDOW_SYMBOLS,
    channelization_problem,
)
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class ChannelizeStep(Step):
    conv: Literal["channelize"] = "channelize"
    decim: StrictInt = Field(
        ge=1,
        le=MAX_DECIM,
        description="Integer decimation factor; output rate = input rate / decim.",
    )
    bandwidth_hz: float = Field(
        description=(
            "Filter PASSBAND width in Hz (cutoff = bandwidth_hz/2 each side of "
            "center_hz), not the signal's occupied bandwidth. A signal that "
            "fills its band edge-to-edge (e.g. a chirp sweeping +/-B/2) needs "
            "bandwidth_hz >= ~2x its occupied bandwidth, or the band edges are "
            "clipped — a silent, progressive decode degradation, not an error."
        )
    )
    center_hz: float = Field(
        default=0.0,
        description="Offset of the wanted sub-band from the capture centre, in Hz.",
    )

    @model_validator(mode="after")
    def _ok(self) -> "ChannelizeStep":
        if self.bandwidth_hz <= 0:
            raise PydanticCustomError("value_error", "bandwidth_hz must be > 0")
        return self


class TranslateStep(Step):
    conv: Literal["translate"] = "translate"
    center_hz: float = Field(
        description="the offset frequency shifted to DC; signed, in Hz"
    )


class InvertStep(Step):
    conv: Literal["invert"] = "invert"


# The transition is allowed to widen past the passband once the passband is
# narrow — a channel this much finer than the input rate wants decimation
# first — and channelization_problem's floor refuses the rest.
_MAX_TRANSITION_TAPS = 4096


def _transition_hz(bandwidth_hz: float, rate: float) -> float:
    return max(bandwidth_hz / 2.0, rate / _MAX_TRANSITION_TAPS)


def _nyquist_problem(center_hz: float, rate: float) -> str | None:
    if abs(center_hz) > 0.5 * rate:
        return (
            f"center_hz {center_hz:g} lies outside the +-{0.5 * rate:g} Hz "
            f"Nyquist span of the {rate:g} Hz input; a mixer wraps mod the "
            f"sample rate and would silently tune an aliased sub-band"
        )
    return None


class Channelize(Stage[CompileContext, ChannelizeStep]):
    name = "channelize"
    description = (
        "Extract a sub-band: frequency-shift center_hz to baseband, low-pass "
        "to bandwidth_hz, and decimate by decim (output rate = input rate / "
        "decim). RX-only conditioning."
    )
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = ChannelizeStep
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, step: ChannelizeStep) -> None:
        b.chain(
            "freq_xlating_fir_filter_ccf",
            decim=step.decim,
            center=step.center_hz,
            cutoff=step.bandwidth_hz / 2.0,
            transition=_transition_hz(step.bandwidth_hz, b.rate),
            rate=b.rate,
        )

    def rate_factor(self, step: ChannelizeStep) -> float:
        return 1.0 / int(step.decim)

    def validate_input_rate(self, step: ChannelizeStep, rate: float) -> str | None:
        return channelization_problem(
            rate=rate,
            decim=step.decim,
            bandwidth_hz=step.bandwidth_hz,
            center_hz=step.center_hz,
        )


class Translate(Stage[CompileContext, TranslateStep]):
    """Frequency-translate a sub-band to baseband: multiply by a complex rotator so
    center_hz lands at DC (stock rotator_cc). PURE shift — no filtering and no
    decimation, so |z| (amplitude) is preserved. For channel isolation (shift +
    low-pass + decimate) use channelize instead; reach for translate when you only
    need to MOVE the spectrum, e.g. to slide an AM/FM audio subcarrier down to DC
    before an IQ demod. RX-only, IQ->IQ, rate unchanged."""

    name = "translate"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = TranslateStep

    def emit_rx(self, b: CompileContext, step: TranslateStep) -> None:
        b.chain("rotator_cc", phase_inc=-2.0 * math.pi * step.center_hz / b.rate)

    def validate_input_rate(self, step: TranslateStep, rate: float) -> str | None:
        return _nyquist_problem(step.center_hz, rate)


class Invert(Stage[CompileContext, InvertStep]):
    name = "invert"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = InvertStep

    def emit_rx(self, b: CompileContext, step: InvertStep) -> None:
        b.chain("conjugate_cc")


class ResampleStep(Step):
    conv: Literal["resample"] = "resample"
    # the anti-imaging filter is sized by the interpolation factor, so an
    # unbounded ratio designs an unbounded filter inside the worker
    interpolation: StrictInt = Field(ge=1, le=MAX_RESAMPLE_FACTOR)
    decimation: StrictInt = Field(ge=1, le=MAX_RESAMPLE_FACTOR)


class Resample(Stage[CompileContext, ResampleStep]):
    """Rational resample by interpolation/decimation via rational_resampler_ccf
    (integer ratio, auto-designed anti-imaging taps). The first non-unity
    rate_factor that is not 1/decim: lands an arbitrary capture rate exactly on a
    target (CSS needs oversample*bandwidth). RX-only conditioning."""

    name = "resample"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = ResampleStep
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, step: ResampleStep) -> None:
        b.chain(
            "rational_resampler_ccf",
            interpolation=int(step.interpolation),
            decimation=int(step.decimation),
        )

    def rate_factor(self, step: ResampleStep) -> float:
        return int(step.interpolation) / int(step.decimation)


class ClockCorrectStep(Step):
    conv: Literal["clock_correct"] = "clock_correct"
    ppm: float = Field(
        description=(
            "transmitter clock offset in parts-per-million; signed, and the "
            "sample rate is corrected by 1/(1+ppm*1e-6)"
        )
    )

    @model_validator(mode="after")
    def _ok(self) -> "ClockCorrectStep":
        if self.ppm <= -1e6:
            raise PydanticCustomError("value_error", "ppm must be > -1e6")
        return self


class ClockCorrect(Stage[CompileContext, ClockCorrectStep]):
    """Cancel sampling-frequency offset (transmitter clock drift) by resampling
    by 1/(1 + ppm*1e-6) via the polyphase arbitrary resampler. RX-only. SFO is a
    silent no-op below ~1 chip of cumulative drift (ppm * n_samples), so it only
    matters on long (multi-second) frames."""

    name = "clock_correct"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = ClockCorrectStep
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, step: ClockCorrectStep) -> None:
        b.chain("pfb_arb_resampler_ccf", rate=1.0 / (1.0 + float(step.ppm) * 1e-6))

    def rate_factor(self, step: ClockCorrectStep) -> float:
        return 1.0 / (1.0 + float(step.ppm) * 1e-6)


class AgcStep(Step):
    conv: Literal["agc"] = "agc"
    mode: AgcMode = AgcMode.FEEDFORWARD
    reference: float = 1.0
    window_symbols: float = Field(
        default=16.0,
        le=MAX_WINDOW_SYMBOLS,
        description=(
            "Feedforward/power normalization window, in symbols. For bursty "
            "signals it must exceed the burst-repetition scale (e.g. 1024): "
            "small windows normalize noise-only stretches to full scale, "
            "erasing the on/off contrast a downstream slicer needs (measured "
            "on a live impulsive-RFI capture: ones-fraction 0.335 vs 0.018). "
            "For bursty/pulsed "
            "OOK specifically, prefer ook_envelope(loop_bw=0), which detects and "
            "normalizes each burst internally and needs no agc; the large-window "
            "guidance here applies when an agc IS used (continuous signals / "
            "closed-loop)."
        ),
    )
    attack_symbols: float = 1.0
    decay_symbols: float = 16.0
    max_gain: float = 0.0

    @model_validator(mode="after")
    def _ok(self) -> "AgcStep":
        if self.reference <= 0.0:
            raise PydanticCustomError("value_error", "reference must be > 0")
        if self.window_symbols <= 0.0:
            raise PydanticCustomError("value_error", "window_symbols must be > 0")
        if self.attack_symbols <= 0.0 or self.decay_symbols <= 0.0:
            raise PydanticCustomError(
                "value_error", "attack_symbols and decay_symbols must be > 0"
            )
        if self.max_gain < 0.0:
            raise PydanticCustomError("value_error", "max_gain must be >= 0")
        spec = _AGC_MODES[self.mode]
        unused = (
            frozenset().union(*(s.fields for s in _AGC_MODES.values())) - spec.fields
        )
        fields = type(self).model_fields
        supplied_unused = {
            f for f in unused if getattr(self, f) != fields[f].get_default()
        }
        if supplied_unused:
            raise PydanticCustomError(
                "value_error",
                f"mode '{self.mode.value}' does not use {sorted(supplied_unused)}",
            )
        return self


def _agc_feedforward(b: CompileContext, p: AgcStep) -> None:
    b.chain(
        "feedforward_agc_cc",
        nsamples=max(1, round(p.window_symbols * b.sps)),
        reference=p.reference,
    )


def _agc_feedback(b: CompileContext, p: AgcStep) -> None:
    b.chain(
        "agc2_cc",
        attack_rate=1.0 / (p.attack_symbols * b.sps),
        decay_rate=1.0 / (p.decay_symbols * b.sps),
        reference=p.reference,
        max_gain=p.max_gain,
    )


def _agc_power(b: CompileContext, p: AgcStep) -> None:
    src = b.require_tail()
    alpha = 1.0 / (p.window_symbols * b.sps)
    c2f = b.add("complex_to_float")
    rms = b.add("rms_cf", alpha=alpha)
    b.connect(src, c2f)
    b.connect(src, rms)
    dre = b.add("divide_ff")
    b.connect(c2f, dre, src_port=0, dst_port=0)
    b.connect(rms, dre, src_port=0, dst_port=1)
    dim = b.add("divide_ff")
    b.connect(c2f, dim, src_port=1, dst_port=0)
    b.connect(rms, dim, src_port=0, dst_port=1)
    f2c = b.add("float_to_complex")
    b.connect(dre, f2c, src_port=0, dst_port=0)
    b.connect(dim, f2c, src_port=0, dst_port=1)
    b.set_tail(f2c)


@dataclass(frozen=True)
class AgcModeSpec:
    """One AGC mode: the step fields it reads, how it emits, and which
    amplitude statistic it drives to `reference`. A consumer declares the
    statistic it needs, so a mode that normalizes the wrong one is a compile
    error instead of a silent mis-decode. One row per mode, so a mode cannot
    gain an emitter without an amplitude contract or a field list."""

    fields: frozenset[str]
    emit: Callable[[CompileContext, AgcStep], None]
    amplitude: Amplitude


_AGC_MODES: dict[AgcMode, AgcModeSpec] = {
    AgcMode.FEEDFORWARD: AgcModeSpec(
        frozenset({"window_symbols", "reference"}),
        _agc_feedforward,
        Amplitude.PEAK_UNITY,
    ),
    AgcMode.FEEDBACK: AgcModeSpec(
        frozenset({"attack_symbols", "decay_symbols", "max_gain", "reference"}),
        _agc_feedback,
        Amplitude.MEAN_MAG_UNITY,
    ),
    AgcMode.POWER: AgcModeSpec(
        frozenset({"window_symbols"}), _agc_power, Amplitude.RMS_UNITY
    ),
}

_missing_modes = sorted(m.value for m in AgcMode if m not in _AGC_MODES)
if _missing_modes:
    raise RuntimeError(f"AgcMode members with no AgcModeSpec: {_missing_modes}")


def agc_modes_for(amplitudes: Iterable[Amplitude]) -> list[str]:
    wanted = frozenset(amplitudes)
    return sorted(m.value for m, s in _AGC_MODES.items() if s.amplitude in wanted)


class Agc(Stage[CompileContext, AgcStep]):
    """Normalize IQ amplitude. `mode` selects WHICH statistic is driven to
    `reference`, and that choice is part of the output contract:
    feedforward -> peak_unity, feedback -> mean_mag_unity, power -> rms_unity."""

    name = "agc"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = AgcStep

    def emit_rx(self, b: CompileContext, step: AgcStep) -> None:
        _AGC_MODES[step.mode].emit(b, step)

    def out_descriptor(self, in_desc: Descriptor, step: AgcStep) -> Descriptor:
        return Descriptor(
            Level.IQ,
            in_desc.item_type,
            in_desc.carrier,
            _AGC_MODES[step.mode].amplitude,
        )


class SquelchStep(Step):
    conv: Literal["squelch"] = "squelch"
    threshold_db: float = Field(
        description=(
            "Absolute gate threshold in dB relative to unity-RMS input power "
            "(requires rms_unity amplitude), applied to an IIR-averaged power "
            "estimate; samples below it are zeroed, not dropped."
        )
    )
    alpha_symbols: float = Field(
        default=1.0,
        le=MAX_WINDOW_SYMBOLS,
        description=(
            "Averaging time constant of the power estimate, in symbols. Too "
            "fast gates OOK/PPM frame bodies off mid-burst; marginal-SNR PPM "
            "(~50% duty, pulses 2-4x noise) cannot be squelch-framed at ANY "
            "alpha (measured 0-6/400 full-body holds) - prefer ook_envelope "
            "loop_bw=0, which detects bursts itself."
        ),
    )
    ramp_symbols: float = Field(
        default=0.0,
        le=MAX_WINDOW_SYMBOLS,
        description=(
            "cosine edge length in symbols applied at each gate opening and "
            "closing; suppresses the switching transient"
        ),
    )

    @model_validator(mode="after")
    def _ok(self) -> "SquelchStep":
        if self.alpha_symbols <= 0.0:
            raise PydanticCustomError("value_error", "alpha_symbols must be > 0")
        if self.ramp_symbols < 0.0:
            raise PydanticCustomError("value_error", "ramp_symbols must be >= 0")
        return self


class Squelch(Stage[CompileContext, SquelchStep]):
    """Mute the stream below a power threshold, IQ->IQ (stock pwr_squelch_cc).

    A clock-recovery loop that free-runs through the gaps between bursts
    integrates noise and arrives at the next burst mistimed; on a real bursty
    capture that turns float-level nondeterminism into whole-symbol drift.
    Muting the gaps zeroes the timing error there, so the loop holds its state
    instead of wandering.

    Samples are ZEROED, never dropped: gating would make the output length
    signal-dependent, which no rate_factor can express, so every downstream
    sps/rate computation would silently disagree with reality.

    `threshold_db` is power relative to the input's scale, so this must run on a
    stream whose scale is known -- hence the amplitude precondition. Put it
    after the agc, and give that agc a window long enough to span the gaps
    (otherwise it lifts the noise floor between bursts to meet the threshold)."""

    name = "squelch"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    accepts_amplitude = frozenset({Amplitude.RMS_UNITY})
    step_model = SquelchStep

    def emit_rx(self, b: CompileContext, step: SquelchStep) -> None:
        b.chain(
            "pwr_squelch_cc",
            threshold_db=step.threshold_db,
            alpha=1.0 / (step.alpha_symbols * b.sps),
            ramp=round(step.ramp_symbols * b.sps),
            gate=False,
        )


class EqualizerStep(Step):
    conv: Literal["equalizer"] = "equalizer"
    num_taps: StrictInt = Field(
        default=15,
        ge=1,
        le=MAX_FILTER_TAPS,
        description="FIR length in taps; must span the channel's delay spread",
    )
    step_size: float = Field(
        default=0.01,
        description="CMA adaptation rate (mu); larger converges "
        "faster and tracks worse",
    )
    modulus: float = Field(
        default=1.0, description="target constant modulus the taps drive |y| toward"
    )

    @model_validator(mode="after")
    def _ok(self) -> "EqualizerStep":
        if self.step_size <= 0.0:
            raise PydanticCustomError("value_error", "step_size must be > 0")
        if self.modulus <= 0.0:
            raise PydanticCustomError("value_error", "modulus must be > 0")
        return self


class Equalizer(Stage[CompileContext, EqualizerStep]):
    """Blind adaptive channel equalizer: a symbol-spaced FIR whose taps adapt by
    the Constant Modulus Algorithm (stock linear_equalizer at sps=1 +
    adaptive_algorithm_cma). It inverts the inter-symbol interference a
    multipath/frequency-selective channel imposes, with NO training sequence --
    it exploits only that the modulation is (near) constant modulus (PSK, FSK,
    GMSK), so no protocol datasheet enters the product. Run it on a 1-sps stream
    after timing recovery; put an agc before it when the scale is unknown. CMA
    leaves a global sign/phase ambiguity, so a downstream costas / constellation
    receiver still resolves carrier phase. RX-only, IQ->IQ, rate-preserving; the
    output amplitude is left UNKNOWN because CMA's steady state is approximate."""

    name = "equalizer"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = EqualizerStep
    alters_amplitude = True

    def emit_rx(self, b: CompileContext, step: EqualizerStep) -> None:
        b.chain(
            "cma_equalizer",
            num_taps=step.num_taps,
            step_size=step.step_size,
            modulus=step.modulus,
        )


class AmStep(Step):
    conv: Literal["am"] = "am"
    dc_block_len: StrictInt = Field(default=1024, ge=2, le=MAX_FRAME_ITEMS)


class Am(Stage[CompileContext, AmStep]):
    name = "am"
    from_level = Level.IQ
    to_level = Level.AUDIO
    family = "conditioning"
    step_model = AmStep

    def emit_rx(self, b: CompileContext, step: AmStep) -> None:
        b.chain("complex_to_mag")
        b.chain("dc_blocker_ff", d=step.dc_block_len)

    def out_descriptor(self, in_desc: Descriptor, step: AmStep) -> Descriptor:
        return Descriptor(Level.AUDIO, ItemType.F, in_desc.carrier)


class FmDemodStep(Step):
    conv: Literal["fm_demod"] = "fm_demod"
    deviation: float = Field(gt=0)
    dc_block_len: StrictInt = Field(default=1024, ge=2, le=MAX_FRAME_ITEMS)


class FmDemod(Stage[CompileContext, FmDemodStep]):
    name = "fm_demod"
    from_level = Level.IQ
    to_level = Level.AUDIO
    family = "conditioning"
    step_model = FmDemodStep

    def emit_rx(self, b: CompileContext, step: FmDemodStep) -> None:
        b.chain("quadrature_demod", gain=b.rate / (2.0 * math.pi * step.deviation))
        b.chain("dc_blocker_ff", d=step.dc_block_len)

    def out_descriptor(self, in_desc: Descriptor, step: FmDemodStep) -> Descriptor:
        return Descriptor(Level.AUDIO, ItemType.F, in_desc.carrier)


class AnalyticStep(Step):
    conv: Literal["analytic"] = "analytic"
    ntaps: StrictInt = Field(default=65, le=MAX_FILTER_TAPS)

    @model_validator(mode="after")
    def _ok(self) -> "AnalyticStep":
        if self.ntaps < 3 or self.ntaps % 2 == 0:
            raise PydanticCustomError("value_error", "ntaps must be odd and >= 3")
        return self


class Analytic(Stage[CompileContext, AnalyticStep]):
    name = "analytic"
    from_level = Level.AUDIO
    to_level = Level.IQ
    family = "conditioning"
    step_model = AnalyticStep

    def emit_rx(self, b: CompileContext, step: AnalyticStep) -> None:
        b.chain("hilbert_fc", ntaps=step.ntaps)

    def out_descriptor(self, in_desc: Descriptor, step: AnalyticStep) -> Descriptor:
        return Descriptor(Level.IQ, ItemType.C, in_desc.carrier)


CONDITIONING_STAGES: tuple[type[Stage[CompileContext, Any]], ...] = (
    Channelize,
    Translate,
    Invert,
    Resample,
    ClockCorrect,
    Agc,
    Squelch,
    Equalizer,
    Am,
    FmDemod,
    Analytic,
)
