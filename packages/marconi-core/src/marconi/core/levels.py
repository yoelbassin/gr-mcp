from __future__ import annotations

from enum import Enum


class Level(str, Enum):
    IQ = "iq"
    SYMBOLS = "symbols"
    BITS = "bits"
    FRAMES = "frames"
    MESSAGES = "messages"
    AUDIO = "audio"


# The lattice is a small DAG, not a flat ladder: IQ branches to SYMBOLS (digital)
# or AUDIO (analog), and AUDIO is a through-station back to IQ (an envelope/FM
# carrier can itself carry a sub-carrier). Adjacency is "same rung, or a direct
# RX successor."
_RX_SUCCESSORS: dict[Level, frozenset[Level]] = {
    Level.IQ: frozenset({Level.SYMBOLS, Level.AUDIO}),
    Level.SYMBOLS: frozenset({Level.BITS}),
    Level.BITS: frozenset({Level.FRAMES}),
    Level.FRAMES: frozenset({Level.MESSAGES}),
    Level.MESSAGES: frozenset(),
    Level.AUDIO: frozenset({Level.IQ}),
}


def adjacent(src: Level, dst: Level) -> bool:
    return src == dst or dst in _RX_SUCCESSORS[src]
