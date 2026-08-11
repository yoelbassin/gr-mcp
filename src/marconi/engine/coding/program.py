from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from marconi.engine.backends.base import BlockCensus
from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.deadline import check_deadline


@dataclass(frozen=True)
class CodingStep:
    name: str
    kind: str
    call: Callable[[CodingCarrier], CodingCarrier]


@dataclass(frozen=True)
class CodingProgram:
    steps: list[CodingStep]


def _items(c: CodingCarrier) -> int:
    if c.bits.size:
        return int(c.bits.size)
    if c.symbols is not None:
        return int(np.asarray(c.symbols).size)
    return 0


def _windows(c: CodingCarrier) -> int | None:
    return None if c.windows is None else len(c.windows)


def _row(step: CodingStep, before: CodingCarrier, after: CodingCarrier) -> BlockCensus:
    stats = after.stats
    return BlockCensus(
        block=step.name,
        kind=step.kind,
        items_in=_items(before),
        items_out=_items(after),
        windows_in=_windows(before),
        windows_out=_windows(after),
        chance_windows=stats.chance_windows if stats else None,
        words_valid=stats.words_valid if stats else None,
        words_total=stats.words_total if stats else None,
        chance_word_rate=stats.chance_word_rate if stats else None,
    )


def run_coding(
    program: CodingProgram,
    carrier: CodingCarrier,
    census: list[BlockCensus] | None = None,
) -> CodingCarrier:
    for step in program.steps:
        check_deadline()
        # stats describe the step that computed them; ops built on replace()
        # would otherwise carry the previous step's stats into their own row
        src = replace(carrier, stats=None) if carrier.stats is not None else carrier
        after = step.call(src)
        if census is not None:
            census.append(_row(step, src, after))
        carrier = after
    return carrier
