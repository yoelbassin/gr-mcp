from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from marconi.engine.backends.base import Backend, BackendError, RunResult
from marconi.engine.compile.ir import GrPipeline


@dataclass(frozen=True)
class BlockArity:
    n_in: int
    n_out: int


@dataclass
class StubGraph:
    blocks: dict[str, BlockArity]
    edges: list[tuple[str, int, str, int]] = field(default_factory=list)


class StubBackend(Backend):
    """Pure-Python Backend that proves a GrPipeline is well-formed and
    backend-resolvable without GNU Radio: every kind resolves in the arity map,
    every connection references real blocks with in-range ports."""

    def __init__(self, factories: Mapping[str, BlockArity]) -> None:
        self._factories = dict(factories)

    @property
    def name(self) -> str:
        return "stub"

    def instantiate(self, pipeline: GrPipeline) -> StubGraph:
        blocks: dict[str, BlockArity] = {}
        for b in pipeline.blocks:
            arity = self._factories.get(b.kind)
            if arity is None:
                raise BackendError(f"block '{b.id}': kind '{b.kind}' has no factory")
            blocks[b.id] = arity
        edges: list[tuple[str, int, str, int]] = []
        for c in pipeline.connections:
            if c.src_block not in blocks:
                raise BackendError(f"connection from unknown block '{c.src_block}'")
            if c.dst_block not in blocks:
                raise BackendError(f"connection to unknown block '{c.dst_block}'")
            if not 0 <= c.src_port < blocks[c.src_block].n_out:
                raise BackendError(
                    f"block '{c.src_block}' has no output port {c.src_port}"
                )
            if not 0 <= c.dst_port < blocks[c.dst_block].n_in:
                raise BackendError(
                    f"block '{c.dst_block}' has no input port {c.dst_port}"
                )
            edges.append((c.src_block, c.src_port, c.dst_block, c.dst_port))
        return StubGraph(blocks=blocks, edges=edges)

    def run_pipeline(self, pipeline: GrPipeline, timeout: float = 30.0) -> RunResult:
        raise BackendError("stub backend validates wiring only; it does not execute")
