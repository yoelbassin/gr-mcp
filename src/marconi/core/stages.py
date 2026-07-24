from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from marconi.core.descriptor import Amplitude, Carrier, Descriptor
from marconi.core.errors import register_error
from marconi.core.levels import Level
from marconi.core.models import ValidationIssue
from marconi.core.params import StageParams

B = TypeVar("B")


class SpecStep(Protocol):
    @property
    def conv(self) -> str: ...
    @property
    def params(self) -> Mapping[str, object]: ...


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


class Stage(ABC, Generic[B]):
    name: str
    from_level: Level
    to_level: Level
    family: str
    directions: frozenset[str] = frozenset({"rx", "tx"})
    params_model: type[StageParams]

    # Framing capabilities a codec validator consumes (declared by the stage,
    # not enumerated by name in the validator — so a user-registered stage
    # advertises its own capability). A seeder establishes frame boundaries
    # (BITS->FRAMES); it is self_slicing if it also produces complete frames,
    # else a slices_body stage must carve the seeded region into frame bodies.
    seeds_frames: bool = False
    self_slicing: bool = False
    slices_body: bool = False
    # A level-preserving stage accepts whatever level the previous stage emits
    # and passes it through unchanged (its runtime dispatches on carrier state).
    # Used by transforms that work in both stream (BITS) and seeded (FRAMES) scope.
    level_preserving: bool = False

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

    @abstractmethod
    def emit_rx(self, b: B, params: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def emit_tx(self, b: B, params: Mapping[str, Any]) -> None: ...

    def out_descriptor(
        self, in_desc: Descriptor, params: Mapping[str, Any]
    ) -> Descriptor:
        amplitude = (
            in_desc.amplitude
            if self.to_level is Level.IQ and not self.alters_amplitude
            else Amplitude.UNKNOWN
        )
        return Descriptor(
            self.to_level,
            in_desc.item_type,
            in_desc.layout,
            in_desc.carrier,
            amplitude,
        )

    def rate_factor(self, params: Mapping[str, Any]) -> float:
        return 1.0

    def required_input_rate(
        self, params: Mapping[str, Any], symbol_rate: float
    ) -> float | None:
        """The input sample rate this stage needs, or None if rate-agnostic. The
        compiler checks it against the rate it computed for this stage's input
        boundary (within a tolerance that admits ppm-scale clock correction), so
        a wrong resample ratio fails at compile instead of decoding garbage with
        no diagnostics (issue 06)."""
        return None


class RxStage(Stage[B]):
    directions: frozenset[str] = frozenset({"rx"})

    def emit_tx(self, b: B, params: Mapping[str, Any]) -> None:
        raise StageDirectionError(self.name, "tx", self.directions)


class TxStage(Stage[B]):
    directions: frozenset[str] = frozenset({"tx"})

    def emit_rx(self, b: B, params: Mapping[str, Any]) -> None:
        raise StageDirectionError(self.name, "rx", self.directions)


class DuplexStage(Stage[B]):
    directions: frozenset[str] = frozenset({"rx", "tx"})


def validate_params(
    block_id: str,
    params_model: type[BaseModel],
    supplied: Mapping[str, object],
    out: list[ValidationIssue],
) -> None:
    try:
        params_model.model_validate(dict(supplied))
    except ValidationError as exc:
        for err in exc.errors():
            field = ".".join(str(p) for p in err["loc"]) or None
            out.append(
                ValidationIssue(block_id=block_id, field=field, message=err["msg"])
            )


def validate_path(
    steps: Sequence[SpecStep],
    registry: Mapping[str, Stage[Any]],
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
        validate_params(sid, conv.params_model, step.params, issues)
        if direction is not None and direction not in conv.directions:
            issues.append(
                ValidationIssue(
                    block_id=sid,
                    message=f"{conv.name} does not support direction "
                    f"'{direction}'; supports {sorted(conv.directions)}",
                )
            )
        transparent = conv.level_preserving and prev_level in (
            conv.from_level,
            Level.FRAMES,
        )
        if transparent:
            pass  # transparent: emits whatever level it received
        elif idx == 0 and conv.from_level != start_level:
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
        if not transparent:
            prev_level = conv.to_level
