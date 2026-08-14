from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.compile.errors import CompileError
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ValidationIssue
from marconi.engine.types.step import Step, stage_label
from marconi.errors import register_error

B = TypeVar("B")
S = TypeVar("S", bound=Step)


class SpecValidationError(CompileError):
    """Carries the individual issues so the MCP surface can return them as an
    addressable list; subclasses CompileError so callers catching the general
    compile failure keep working."""

    def __init__(self, issues: list[ValidationIssue], kind: str) -> None:
        self.issues = issues
        lines = [
            f"{i.block_id or f'<{kind}>'}"
            f"{'.' + i.field if i.field else ''}: {i.message}"
            for i in issues
        ]
        super().__init__(f"{kind} validation failed:\n" + "\n".join(lines))


# A bad spec is the user's to fix, not a bug to report.
register_error(SpecValidationError, "invalid_argument")


class Stage(ABC, Generic[B, S]):
    name: str
    from_level: Level
    to_level: Level
    family: str
    step_model: type[S]

    # One agent-facing line naming what the stage is FOR, surfaced by
    # describe_stages. Populate it only when the name alone does not carry the
    # use — a generic mechanism (line code, interleaver, bit re-phaser) an
    # operator must map their datasheet concept onto. Empty = self-explanatory.
    description: str = ""

    # Establishes burst windows on the coding carrier (PHY sync, not framing).
    seeds_windows: bool = False

    # A window-seeding stage that SEARCHES for its pattern (vs cutting
    # unconditionally by geometry): its found-window count is signal evidence.
    sync_search: bool = False

    # A stage that REPLACES the run's marks with its own pattern-search hits.
    # Burst marks are a detector's physical reading and earn detection-grade
    # evidence; search hits judged with no chance model are not evidence of
    # anything, so quality skips marks_evidence when such a stage ran.
    marks_are_search_hits: bool = False

    # A decoder that checks every codeword against its code structure and
    # tallies how many passed (words_valid/words_total in its census row): the
    # valid fraction measured against chance is signal evidence. Output is
    # still passed through unchanged - validation counts, it never drops.
    validates_words: bool = False

    # A coherent demod that recovers carrier phase only up to a rotational
    # ambiguity (e.g. MSK's 180 deg), so its sliced bit polarity may be
    # inverted. run_rx surfaces a hint to retry flipped when such a stage feeds
    # a bit/symbol output. A frequency discriminator (fsk) has a fixed sense and
    # is NOT tagged (its only flip is a spectrally mirrored capture -> invert).
    polarity_ambiguous: bool = False

    # Seam invariant: the wire item_type / decision-carrier a stage
    # accepts on input. The phy compiler checks them against the upstream
    # descriptor, so an ill-typed composition (e.g. hard bits into a soft-LLR
    # consumer) fails at compile, not in the worker. None = polymorphic. The rule
    # they encode: SYMBOLS is SOFT except QAM/CSS (hard@SYMBOLS); BITS is HARD
    # except the soft-LLR lane ("f"/SOFT).
    accepts_item_type: ItemType | None = None
    accepts_carrier: Carrier | None = None
    # The amplitude statistics this stage can decode from, or None if it is
    # scale-invariant. A set, not one value: a constant-modulus demod tolerates
    # several, a fixed-radius one tolerates exactly its own. Each member must be
    # measured across a gain sweep, not assumed.
    accepts_amplitude: frozenset[Amplitude] | None = None
    alters_amplitude: bool = False
    # The minimum samples-per-symbol an internal timing-recovery loop needs on
    # RX, or None for stages that do not recover symbol timing. Gardner-class
    # TEDs need 2: below that the loop either fails to construct in the backend
    # or, worse, runs and emits confident garbage (measured BER 0.504 with
    # status "ok" at 1.5 sps), so the compiler rejects the rate pair up front.
    min_input_sps: float | None = None

    @abstractmethod
    def emit_rx(self, b: B, step: S) -> None: ...

    def out_descriptor(self, in_desc: Descriptor, step: S) -> Descriptor:
        amplitude = (
            in_desc.amplitude
            if self.to_level is Level.IQ and not self.alters_amplitude
            else Amplitude.UNKNOWN
        )
        return Descriptor(
            self.to_level,
            in_desc.item_type,
            in_desc.carrier,
            amplitude,
            in_desc.order if self.to_level is in_desc.level else None,
        )

    def rate_factor(self, step: S) -> float:
        return 1.0

    def required_input_rate(self, step: S, symbol_rate: float) -> float | None:
        """The input sample rate this stage needs, or None if rate-agnostic. The
        compiler checks it against the rate it computed for this stage's input
        boundary (within a tolerance that admits ppm-scale clock correction), so
        a wrong resample ratio fails at compile instead of decoding garbage with
        no diagnostics."""
        return None

    def required_input_order(self, step: S) -> int | None:
        """The symbol-alphabet size this stage decodes, or None if any. The
        compiler compares it with the producer's pinned Descriptor.order when
        both sides declare, so a demod/demap order mismatch fails at compile
        instead of emitting wrong-width garbage bits."""
        return None

    def accepts_amplitude_for(self, step: S) -> frozenset[Amplitude] | None:
        """The amplitude statistics this stage requires for this specific
        step config, or None if scale-invariant for it - defaults to the
        class-level accepts_amplitude. Override when a stage's requirement
        varies by its own step (e.g. an alternate RX path that normalizes
        internally and needs no upstream amplitude convention at all). The
        compiler consults this method, never the bare attribute, so a
        step-conditional requirement is expressed in one place."""
        return self.accepts_amplitude

    def min_input_sps_for(self, step: S) -> float | None:
        """Like accepts_amplitude_for, for the samples-per-symbol floor:
        defaults to the class-level min_input_sps; override when the floor
        varies by step (e.g. a path that re-acquires timing per burst instead
        of running one continuous loop)."""
        return self.min_input_sps

    def validate_input(self, in_desc: Descriptor, step: S) -> str | None:
        """A stage-specific input check the declarative attributes cannot
        express; a returned message fails the compile."""
        return None

    def validate_input_rate(self, step: S, rate: float) -> str | None:
        """A stage-specific check against the compiled input sample rate (a
        Nyquist bound the step model alone cannot know); a returned message
        fails the compile."""
        return None

    def retry_hints(self, step: S, path_convs: frozenset[str]) -> list[str]:
        """Actionable retry guidance when a decode's verdict fell short of
        'decoded', owned by the stage whose parameters the retry would change
        (a conv-name lookup table in the runner drifts the moment a stage
        gains a new mode). path_convs names the other stages in the spec for
        cross-stage advice ('remove the agc')."""
        return []

    def output_item_rate(
        self, step: S, in_rate: float, symbol_rate: float
    ) -> float | None:
        """True items-per-second of this stage's OUTPUT stream where it
        departs from the compiled rate model: an RX demod that decimates by
        sps internally (rate_factor stays 1.0 there) emits at the
        symbol rate; a demap that unpacks k bits per symbol multiplies by k.
        None means the rate model is already the truth. Observability only —
        trace metadata, never compile checks."""
        return None


class CodingStage(Stage[CodingBuilder, S]):
    """Coding-lane stage: emits into the numpy CodingProgram via a
    CodingBuilder, never the GR graph. The compiler partitions the path on
    CodingStage membership (isinstance), so a stage's execution flavor and the
    context its emit body is written against are one fact — it cannot claim one
    flavor and emit for the other."""


def validate_path(
    steps: Sequence[Step],
    registry: Mapping[str, Stage[Any, Any]],
    start_level: Level,
    entity_name: str,
    issues: list[ValidationIssue],
) -> None:
    prev_level = start_level
    for idx, step in enumerate(steps):
        sid = stage_label(idx, step.conv)
        conv = registry.get(step.conv)
        if conv is None:
            known = ", ".join(sorted(registry))
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"unknown converter '{step.conv}'; known: {known}",
                )
            )
            continue
        if idx == 0 and conv.from_level != start_level:
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{entity_name} must start at {start_level.value} "
                    f"but {conv.name} starts at {conv.from_level.value}",
                )
            )
        elif idx > 0 and conv.from_level != prev_level:
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{conv.name} input level {conv.from_level.value} does "
                    f"not match the previous stage's output level "
                    f"{prev_level.value}; rung moves happen inside a stage, so "
                    f"boundary levels must be equal",
                )
            )
        prev_level = conv.to_level
