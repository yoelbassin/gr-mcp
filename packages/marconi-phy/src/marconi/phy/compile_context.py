from __future__ import annotations

from marconi.core.descriptor import Descriptor
from marconi.core.params import ParamValue
from marconi.phy.ir import GrBlock, GrConnection, GrPipeline


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
        self._blocks.append(GrBlock(id=block_id, kind=kind, params=dict(params)))
        return block_id

    def connect(self, src: str, dst: str, src_port: int = 0, dst_port: int = 0) -> None:
        self._connections.append(
            GrConnection(
                src_block=src, src_port=src_port, dst_block=dst, dst_port=dst_port
            )
        )

    def chain(self, kind: str, **params: ParamValue) -> str:
        block_id = self.add(kind, **params)
        if self._tail is not None:
            self.connect(self._tail, block_id)
        self._tail = block_id
        return block_id

    @property
    def tail(self) -> str | None:
        return self._tail

    def set_tail(self, block_id: str) -> None:
        self._tail = block_id

    def build(self, name: str, sample_rate: float) -> GrPipeline:
        return GrPipeline(
            name=name,
            sample_rate=sample_rate,
            blocks=list(self._blocks),
            connections=list(self._connections),
        )
