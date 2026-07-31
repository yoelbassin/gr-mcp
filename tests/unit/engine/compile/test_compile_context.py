import pytest

from marconi.engine.compile.compile_context import CompileContext, SampleRateError
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.errors import classify_error

IQ = Descriptor(Level.IQ, ItemType.C)


def test_sps_int_accepts_integral_rate_pair() -> None:
    assert CompileContext(IQ, rate=16.0, symbol_rate=2.0).sps_int() == 8


def test_sps_int_rejects_fractional_rate_pair() -> None:
    # sample_rate/symbol_rate = 36.75: TX must not silently transmit at 36 or
    # 37 sps while the rate model says otherwise (issue 10)
    ctx = CompileContext(IQ, rate=36.75, symbol_rate=1.0)
    with pytest.raises(SampleRateError) as ei:
        ctx.sps_int()
    assert "36.75" in str(ei.value) and "1.0" in str(ei.value)
    assert classify_error(ei.value)[0] == "invalid_argument"


def test_chain_auto_connects_and_advances_tail() -> None:
    ctx = CompileContext(IQ, rate=10.0, symbol_rate=2.0)
    a = ctx.chain("src")
    b = ctx.chain("mid")
    c = ctx.chain("snk")
    pipe = ctx.build("p", 10.0)
    assert [bl.id for bl in pipe.blocks] == [a, b, c]
    assert [(cn.src_block, cn.dst_block) for cn in pipe.connections] == [(a, b), (b, c)]


def test_add_does_not_connect() -> None:
    ctx = CompileContext(IQ, 10.0, 2.0)
    ctx.add("a")
    ctx.add("b")
    pipe = ctx.build("p", 10.0)
    assert pipe.connections == []


def test_explicit_connect_with_ports() -> None:
    ctx = CompileContext(IQ, 10.0, 2.0)
    a = ctx.add("a")
    b = ctx.add("b")
    ctx.connect(a, b, src_port=1, dst_port=0)
    pipe = ctx.build("p", 10.0)
    assert pipe.connections[0].src_port == 1
    assert (pipe.connections[0].src_block, pipe.connections[0].dst_block) == (a, b)


def test_sps_is_rate_over_symbol_rate_and_may_be_fractional() -> None:
    ctx = CompileContext(IQ, rate=10.0, symbol_rate=4.0)
    assert ctx.sps == 2.5


def test_auto_ids_are_unique_per_kind() -> None:
    ctx = CompileContext(IQ, 10.0, 2.0)
    ids = [ctx.add("k") for _ in range(3)]
    assert ids == ["k_0", "k_1", "k_2"]


def test_set_tail_redirects_chaining() -> None:
    ctx = CompileContext(IQ, 10.0, 2.0)
    a = ctx.add("a")
    b = ctx.add("b")
    ctx.set_tail(a)
    c = ctx.chain("c")
    pipe = ctx.build("p", 10.0)
    assert (pipe.connections[0].src_block, pipe.connections[0].dst_block) == (a, c)
    assert b not in [cn.dst_block for cn in pipe.connections]
