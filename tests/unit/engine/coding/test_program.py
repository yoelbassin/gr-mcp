import numpy as np

from marconi.engine.backends.base import BlockCensus
from marconi.engine.coding.builder import CodingBuilder
from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.coding.program import CodingProgram, run_coding


def _drop_half(c: CodingCarrier) -> CodingCarrier:
    return CodingCarrier(bits=c.bits[: c.bits.size // 2], windows=c.windows)


def _seed_one(c: CodingCarrier) -> CodingCarrier:
    return CodingCarrier(bits=c.bits, windows=[Window(start=0, cursor=0)])


def _program() -> CodingProgram:
    b = CodingBuilder()
    b.label, b.kind = "seed[0]", "seed"
    b.add(_seed_one)
    b.label, b.kind = "half[1]", "half"
    b.add(_drop_half)
    return CodingProgram(steps=b.steps)


def test_run_coding_applies_steps_in_order() -> None:
    out = run_coding(_program(), CodingCarrier(bits=np.zeros(8, np.uint8)))
    assert out.bits.size == 4
    assert [w.start for w in out.windows or []] == [0]


def test_run_coding_census_rows_count_items_and_windows() -> None:
    census: list[BlockCensus] = []
    run_coding(_program(), CodingCarrier(bits=np.zeros(8, np.uint8)), census)
    assert [r.block for r in census] == ["seed[0]", "half[1]"]
    assert census[0].windows_in is None and census[0].windows_out == 1
    assert census[1].items_in == 8 and census[1].items_out == 4


def test_census_counts_symbols_when_bits_empty() -> None:
    b = CodingBuilder()
    b.label, b.kind = "id[0]", "id"
    b.add(lambda c: c)
    prog = CodingProgram(steps=b.steps)
    census: list[BlockCensus] = []
    carrier = CodingCarrier(bits=np.zeros(0, np.uint8), symbols=np.zeros(7, np.float32))
    run_coding(prog, carrier, census)
    assert census[0].items_in == 7
