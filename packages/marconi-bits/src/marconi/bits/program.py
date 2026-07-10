from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Union, overload

from marconi.bits.carriers import RxCarrier, TxCarrier

Carrier = Union[RxCarrier, TxCarrier]


@dataclass
class FrameProgram:
    direction: Literal["rx", "tx"]
    steps: list[Callable[..., Any]]
    requires_soft_input: bool = False
    requires_symbol_input: bool = False


@overload
def run_program(program: FrameProgram, carrier: RxCarrier) -> RxCarrier: ...


@overload
def run_program(program: FrameProgram, carrier: TxCarrier) -> TxCarrier: ...


def run_program(program: FrameProgram, carrier: Carrier) -> Carrier:
    for step in program.steps:
        carrier = step(carrier)
    return carrier
