from __future__ import annotations

import numpy as np
import pytest

from marconi.engine.coding.carrier import CodingCarrier
from marconi.engine.coding.ops_bits import rs_code_rx
from marconi.engine.coding.program import CodingProgram, CodingStep, run_coding
from marconi.engine.deadline import RunTimeout, set_deadline
from marconi.engine.types.levels import Level


def test_run_coding_honors_deadline() -> None:
    prog = CodingProgram(
        steps=[CodingStep(name="noop", kind="noop", call=lambda c: c)],
        entry_level=Level.BITS,
        entry_item_type="b",
    )
    carrier = CodingCarrier(bits=np.zeros(16, np.uint8))
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            run_coding(prog, carrier)


def test_rs_decode_honors_deadline() -> None:
    carrier = CodingCarrier(
        bits=np.random.default_rng(0).integers(0, 2, 8160).astype(np.uint8)
    )
    with set_deadline(0.0):
        with pytest.raises(RunTimeout):
            rs_code_rx(carrier, symbol_bits=8, n=255, k=223, prim_poly=285)
