from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ValidationIssue
from marconi.engine.types.step import Step
from marconi.errors import register_error

B = TypeVar("B")
S = TypeVar("S", bound=Step)


class SpecValidationError(Exception):
    def __init__(self, issues: list[ValidationIssue], kind: str) -> None:
        self.issues = issues
        lines = [
            f"{i.block_id or f'<{kind}>'}"
            f"{'.' + i.field if i.field else ''}: {i.message}"
            for i in issues
        ]
        super().__init__(f"{kind} validation failed:\n" + "\n".join(lines))


class StageDirectionError(Exception):
    def __init__(self, stage: str, direction: str, supported: frozenset[str]) -> None:
        super().__init__(
            f"stage '{stage}' does not support direction '{direction}'; "
            f"supported: {sorted(supported)}"
        )


# A bad spec/direction is the user's to fix, not a bug to report.
register_error(SpecValidationError, "invalid_argument")
register_error(StageDirectionError, "invalid_argument")


class Stage(ABC, Generic[B, S]):
    name: str
    from_level: Level
    to_level: Level
    family: str
    directions: frozenset[str] = frozenset({"rx", "tx"})
    step_model: type[S]

    # Execution flavor: "gr" emits into the GR graph, "coding" into the numpy
    # coding program. The compiler partitions the path on this, never on names.
    engine: str = "gr"
    # Establishes burst windows on the coding carrier (PHY sync, not framing).
    seeds_windows: bool = False

    # Seam invariant (issue 06): the wire item_type / decision-carrier a stage
    # accepts on input. The phy compiler checks them against the upstream
    # descriptor, so an ill-typed composition (e.g. hard bits into a soft-LLR
    # consumer) fails at compile, not in the worker. None = polymorphic. The rule
    # they encode: SYMBOLS is SOFT except QAM/CSS (hard@SYMBOLS); BITS is HARD
    # except the soft-LLR lane ("f"/SOFT).
    accepts_item_type: str | None = None
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

    @abstractmethod
    def emit_tx(self, b: B, step: S) -> None: ...

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
        no diagnostics (issue 06)."""
        return None

    def required_input_order(self, step: S) -> int | None:
        """The symbol-alphabet size this stage decodes, or None if any. The
        compiler compares it with the producer's pinned Descriptor.order when
        both sides declare, so a demod/demap order mismatch fails at compile
        instead of emitting wrong-width garbage bits."""
        return None

    def validate_input(self, in_desc: Descriptor, step: S) -> str | None:
        """A stage-specific input check the declarative attributes cannot
        express; a returned message fails the compile."""
        return None


class RxStage(Stage[B, S]):
    directions: frozenset[str] = frozenset({"rx"})

    def emit_tx(self, b: B, step: S) -> None:
        raise StageDirectionError(self.name, "tx", self.directions)


class TxStage(Stage[B, S]):
    directions: frozenset[str] = frozenset({"tx"})

    def emit_rx(self, b: B, step: S) -> None:
        raise StageDirectionError(self.name, "rx", self.directions)


class DuplexStage(Stage[B, S]):
    directions: frozenset[str] = frozenset({"rx", "tx"})


def validate_path(
    steps: Sequence[Step],
    registry: Mapping[str, Stage[Any, Any]],
    start_level: Level,
    entity_name: str,
    issues: list[ValidationIssue],
    direction: str | None = None,
) -> None:
    prev_level = start_level
    for idx, step in enumerate(steps):
        sid = f"{step.conv}[{idx}]"
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
        if direction is not None and direction not in conv.directions:
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{conv.name} does not support direction "
                    f"'{direction}'; supports {sorted(conv.directions)}",
                )
            )
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
