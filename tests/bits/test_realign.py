from __future__ import annotations

import numpy as np
import pytest

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier, _Frame
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.core.stages import StageDirectionError


def test_realign_drops_leading_bits() -> None:
    c = RxCarrier(bits=np.array([1, 1, 0, 1, 0], np.uint8))
    out = framing.realign_rx(c, bit_offset=2)
    assert out.bits.tolist() == [0, 1, 0]
    assert out.frames == []


def test_realign_rejects_seeded_carrier() -> None:
    c = RxCarrier(bits=np.zeros(8, np.uint8), frames=[_Frame(start=0, cursor=0)])
    with pytest.raises(ValueError, match="before any frame seeder"):
        framing.realign_rx(c, bit_offset=1)


def test_realign_registered_rx_only() -> None:
    spec = CodecSpec(path=[CodecStep(conv="realign", params={"bit_offset": 3})])
    compile_codec(spec, registry(), "rx")
    with pytest.raises(StageDirectionError):
        compile_codec(spec, registry(), "tx")
