from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Union, cast, overload

import numpy as np

from marconi.bits.carriers import RxCarrier, TxCarrier
from marconi.bits.models import StageCensus

Carrier = Union[RxCarrier, TxCarrier]


@dataclass(frozen=True)
class ProgramStep:
    name: str
    call: Callable[..., Any]

    def __call__(self, carrier: Carrier) -> Any:
        return self.call(carrier)


@dataclass
class FrameProgram:
    direction: Literal["rx", "tx"]
    steps: list[ProgramStep]
    requires_symbol_input: bool = False


def _step_census(name: str, before: RxCarrier, after: RxCarrier) -> StageCensus:
    return StageCensus(
        stage=name,
        frames_in=len(before.frames),
        frames_out=len(after.frames),
        bits_in=int(np.asarray(before.bits).size),
        bits_out=int(np.asarray(after.bits).size),
        crc_ok_out=sum(1 for f in after.frames if f.crc_ok is True),
    )


@overload
def run_program(
    program: FrameProgram,
    carrier: RxCarrier,
    census: list[StageCensus] | None = None,
) -> RxCarrier: ...


@overload
def run_program(
    program: FrameProgram,
    carrier: TxCarrier,
    census: list[StageCensus] | None = None,
) -> TxCarrier: ...


def run_program(
    program: FrameProgram,
    carrier: Carrier,
    census: list[StageCensus] | None = None,
) -> Carrier:
    if census is None:
        for step in program.steps:
            carrier = step(carrier)
        return carrier
    if program.direction != "rx":
        raise ValueError("a census counts recovered frames; only an rx program has any")
    rx = cast(RxCarrier, carrier)
    for step in program.steps:
        after = cast(RxCarrier, step(rx))
        census.append(_step_census(step.name, rx, after))
        rx = after
    return rx
