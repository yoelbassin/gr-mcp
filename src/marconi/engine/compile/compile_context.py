from __future__ import annotations

from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.params import ParamValue


class CompileContext:
    """Mutable build target threaded through a stage path. Stages add blocks via
    chain()/add()/connect() and read the current rate/sps/descriptor; the compiler
    owns advancing `descriptor` and `rate` between stages."""

    def __init__(self, descriptor: Descriptor, rate: float, symbol_rate: float) -> None:
        self.descriptor = descriptor
        self.rate = rate
        self.symbol_rate = symbol_rate
        self._blocks: list[GrBlock] = []
        self._connections: list[GrConnection] = []
        self._tail: str | None = None
        self._counter: dict[str, int] = {}

    @property
    def sps(self) -> float:
        return self.rate / self.symbol_rate

    def _new_id(self, kind: str) -> str:
        n = self._counter.get(kind, 0)
        self._counter[kind] = n + 1
        return f"{kind}_{n}"

    def add(self, kind: str, **params: ParamValue) -> str:
        block_id = self._new_id(kind)
        self._blocks.append(
            GrBlock(id=block_id, kind=kind, params=dict(params), sample_rate=self.rate)
        )
        return block_id

    def connect(self, src: str, dst: str, src_port: int = 0, dst_port: int = 0) -> None:
        self._connections.append(
            GrConnection(
                src_block=src, src_port=src_port, dst_block=dst, dst_port=dst_port
            )
        )

    def chain_llr_flip(self) -> str:
        # the engine-wide soft convention is LLR bit-1-NEGATIVE; stock
        # correlator/decoder blocks read bit-1-positive. One home for the
        # sign bracket every soft boundary crosses.
        return self.chain("multiply_const_ff", value=-1.0)

    def chain(self, kind: str, **params: ParamValue) -> str:
        block_id = self.add(kind, **params)
        if self._tail is not None:
            self.connect(self._tail, block_id)
        self._tail = block_id
        return block_id

    @property
    def tail(self) -> str | None:
        return self._tail

    def require_tail(self) -> str:
        """The block a fan-out stage branches from. Every RX stage runs after
        the source has been chained, so this is a structural invariant — but it
        is enforced rather than asserted: an assert is stripped under -O, and
        the None would then reach connect() and surface as an unresolvable
        block id instead of naming the real problem."""
        if self._tail is None:
            raise CompileError(
                "stage needs an upstream block to branch from but the graph is "
                "empty; a fan-out stage cannot be the first block in a path"
            )
        return self._tail

    def set_tail(self, block_id: str) -> None:
        self._tail = block_id

    def build(
        self, name: str, sample_rate: float, *, terminal_sink: str | None = None
    ) -> GrPipeline:
        return GrPipeline(
            name=name,
            sample_rate=sample_rate,
            blocks=list(self._blocks),
            connections=list(self._connections),
            terminal_sink=terminal_sink,
        )
