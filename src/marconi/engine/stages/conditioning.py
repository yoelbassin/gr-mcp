from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.stages.base import RxStage, Stage
from marconi.engine.types.descriptor import Amplitude, Descriptor
from marconi.engine.types.enums import AgcMode, ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.step import Step


class ChannelizeStep(Step):
    conv: Literal["channelize"] = "channelize"
    decim: (
        StrictInt  # required & explicit: rate_factor (1/decim) must be params-derivable
    )
    bandwidth_hz: float  # required: signal bandwidth -> LPF cutoff/transition
    center_hz: float = 0.0  # offset of the sub-band from the capture centre

    @model_validator(mode="after")
    def _ok(self) -> "ChannelizeStep":
        if self.decim < 1:
            raise PydanticCustomError("value_error", "decim must be >= 1")
        if self.bandwidth_hz <= 0:
            raise PydanticCustomError("value_error", "bandwidth_hz must be > 0")
        return self


class InvertStep(Step):
    conv: Literal["invert"] = "invert"


class Channelize(RxStage[CompileContext, ChannelizeStep]):
    """Extract a sub-band: frequency-shift center_hz to baseband, low-pass, and
    decimate by `decim` (one freq_xlating_fir_filter). RX-only conditioning. The
    cutoff/transition are derived from bandwidth_hz; `decim` is explicit because
    rate_factor (1/decim) is computed from params alone. The freq_xlating reads
    the INPUT rate (b.rate); the compiler hands the decimated rate to downstream
    stages via rate_factor.

    NOTE: bandwidth_hz is the filter PASSBAND width (cutoff = bandwidth_hz/2), not
    the signal's occupied bandwidth. A signal that fills its band edge-to-edge --
    notably a CSS chirp sweeping +/-B/2 -- must be given bandwidth_hz >= ~2x its
    bandwidth, or the band edges are clipped (for chirps: a silent, progressive
    decode drift, not an error)."""

    name = "channelize"
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
            transition=step.bandwidth_hz / 2.0,
            rate=b.rate,
        )

    def rate_factor(self, step: ChannelizeStep) -> float:
        return 1.0 / int(step.decim)


class Invert(RxStage[CompileContext, InvertStep]):
    """Spectral conjugate (un-mirror a flipped capture, e.g. SDR# .wav). RX-only,
    IQ->IQ, rate unchanged."""

    name = "invert"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = InvertStep

    def emit_rx(self, b: CompileContext, step: InvertStep) -> None:
        b.chain("conjugate_cc")


class ResampleStep(Step):
    conv: Literal["resample"] = "resample"
    interpolation: StrictInt
    decimation: StrictInt

    @model_validator(mode="after")
    def _ok(self) -> "ResampleStep":
        if self.interpolation < 1:
            raise PydanticCustomError("value_error", "interpolation must be >= 1")
        if self.decimation < 1:
            raise PydanticCustomError("value_error", "decimation must be >= 1")
        return self


class Resample(RxStage[CompileContext, ResampleStep]):
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
    ppm: float  # transmitter clock offset in parts-per-million (signed)

    @model_validator(mode="after")
    def _ok(self) -> "ClockCorrectStep":
        if self.ppm <= -1e6:
            raise PydanticCustomError("value_error", "ppm must be > -1e6")
        return self


class ClockCorrect(RxStage[CompileContext, ClockCorrectStep]):
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
    window_symbols: float = 16.0
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
        allowed = _MODE_FIELDS[self.mode]
        unused = set().union(*_MODE_FIELDS.values()) - allowed
        fields = type(self).model_fields
        supplied_unused = {
            f for f in unused if getattr(self, f) != fields[f].get_default()
        }
        if supplied_unused:
            raise PydanticCustomError(
                "value_error",
                f"mode '{self.mode}' does not use {sorted(supplied_unused)}",
            )
        return self


_MODE_FIELDS: dict[str, set[str]] = {
    "feedforward": {"window_symbols", "reference"},
    "feedback": {"attack_symbols", "decay_symbols", "max_gain", "reference"},
    "power": {"window_symbols"},
}


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
    src = b.tail
    assert src is not None
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


_AGC_MODES: dict[str, Callable[[CompileContext, AgcStep], None]] = {
    "feedforward": _agc_feedforward,
    "feedback": _agc_feedback,
    "power": _agc_power,
}

# Which statistic each mode drives to `reference`. A consumer declares the
# statistic it needs, so a mode that normalizes the wrong one is a compile
# error instead of a silent mis-decode.
_MODE_AMPLITUDE: dict[str, Amplitude] = {
    "feedforward": Amplitude.PEAK_UNITY,
    "feedback": Amplitude.MEAN_MAG_UNITY,
    "power": Amplitude.RMS_UNITY,
}


class Agc(RxStage[CompileContext, AgcStep]):
    """Normalize IQ amplitude. `mode` selects WHICH statistic is driven to
    `reference`, and that choice is part of the output contract:
    feedforward -> peak_unity, feedback -> mean_mag_unity, power -> rms_unity."""

    name = "agc"
    from_level = Level.IQ
    to_level = Level.IQ
    family = "conditioning"
    step_model = AgcStep

    def emit_rx(self, b: CompileContext, step: AgcStep) -> None:
        _AGC_MODES[step.mode](b, step)

    def out_descriptor(self, in_desc: Descriptor, step: AgcStep) -> Descriptor:
        return Descriptor(
            Level.IQ,
            in_desc.item_type,
            in_desc.carrier,
            _MODE_AMPLITUDE[step.mode],
        )


class SquelchStep(Step):
    conv: Literal["squelch"] = "squelch"
    threshold_db: float
    alpha_symbols: float = 1.0  # power-estimate time constant
    ramp_symbols: float = 0.0  # cosine edge, suppresses the switching transient

    @model_validator(mode="after")
    def _ok(self) -> "SquelchStep":
        if self.alpha_symbols <= 0.0:
            raise PydanticCustomError("value_error", "alpha_symbols must be > 0")
        if self.ramp_symbols < 0.0:
            raise PydanticCustomError("value_error", "ramp_symbols must be >= 0")
        return self


class Squelch(RxStage[CompileContext, SquelchStep]):
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
    num_taps: StrictInt = 15  # FIR length; must span the channel's delay spread
    step_size: float = 0.01  # CMA adaptation rate (mu)
    modulus: float = 1.0  # target constant modulus the taps drive |y| toward

    @model_validator(mode="after")
    def _ok(self) -> "EqualizerStep":
        if self.num_taps < 1:
            raise PydanticCustomError("value_error", "num_taps must be >= 1")
        if self.step_size <= 0.0:
            raise PydanticCustomError("value_error", "step_size must be > 0")
        if self.modulus <= 0.0:
            raise PydanticCustomError("value_error", "modulus must be > 0")
        return self


class Equalizer(RxStage[CompileContext, EqualizerStep]):
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
    dc_block_len: StrictInt = 1024

    @model_validator(mode="after")
    def _ok(self) -> "AmStep":
        if self.dc_block_len < 2:
            raise PydanticCustomError("value_error", "dc_block_len must be >= 2")
        return self


class Am(RxStage[CompileContext, AmStep]):
    """AM envelope to audio: stock complex_to_mag + dc_blocker. RX-only, IQ->AUDIO,
    rate unchanged."""

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
    dc_block_len: StrictInt = 1024

    @model_validator(mode="after")
    def _ok(self) -> "FmDemodStep":
        if self.dc_block_len < 2:
            raise PydanticCustomError("value_error", "dc_block_len must be >= 2")
        return self


class FmDemod(RxStage[CompileContext, FmDemodStep]):
    """FM discriminator to audio: stock quadrature_demod scaled so +-deviation
    maps to +-1.0, then dc_blocker (carrier frequency error rides the
    discriminator output as DC). RX-only, IQ->AUDIO, rate unchanged."""

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
    ntaps: StrictInt = 65

    @model_validator(mode="after")
    def _ok(self) -> "AnalyticStep":
        if self.ntaps < 3 or self.ntaps % 2 == 0:
            raise PydanticCustomError("value_error", "ntaps must be odd and >= 3")
        return self


class Analytic(RxStage[CompileContext, AnalyticStep]):
    """Real audio to its analytic signal (stock hilbert_fc) so IQ demods compose
    downstream. RX-only, AUDIO->IQ, rate unchanged."""

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
