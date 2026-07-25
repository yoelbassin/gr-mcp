from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from marconi.core.levels import Level
from marconi.phy.backends.base import BlockCensus
from marconi.phy.coding.carrier import CodingCarrier


@dataclass(frozen=True)
class CodingStep:
    name: str
    kind: str
    call: Callable[[CodingCarrier], CodingCarrier]


@dataclass(frozen=True)
class CodingProgram:
    steps: list[CodingStep]
    entry_level: Level
    entry_item_type: str


def _items(c: CodingCarrier) -> int:
    if c.bits.size:
        return int(c.bits.size)
    if c.symbols is not None:
        return int(np.asarray(c.symbols).size)
    return 0


def _windows(c: CodingCarrier) -> int | None:
    return None if c.windows is None else len(c.windows)


def _row(step: CodingStep, before: CodingCarrier, after: CodingCarrier) -> BlockCensus:
    return BlockCensus(
        block=step.name,
        kind=step.kind,
        items_in=_items(before),
        items_out=_items(after),
        windows_in=_windows(before),
        windows_out=_windows(after),
    )


def run_coding(
    program: CodingProgram,
    carrier: CodingCarrier,
    census: list[BlockCensus] | None = None,
) -> CodingCarrier:
    for step in program.steps:
        after = step.call(carrier)
        if census is not None:
            census.append(_row(step, carrier, after))
        carrier = after
    return carrier
