"""LDPC decode via stock gr-fec: soft belief-propagation fec.ldpc_decoder
wrapped in fec.extended_decoder. Soft LLR in (bit 1 = positive; the ldpc stage
flips the engine's bit-1-negative convention upstream), K = N - M hard bits out
per N-bit codeword. gr-fec's decoder reads its parity-check matrix from an alist
file, so the caller-supplied sparse matrix (the variable indices each check
connects) is serialized to a throwaway alist here -- the code is datasheet work,
not the engine's. The decoder parses the file at construction, so it is unlinked
immediately, before the flowgraph runs."""

from __future__ import annotations

import os
import tempfile
from typing import Any


def _alist(block_size: int, check_nodes: list[list[int]]) -> str:
    """Serialize a sparse parity-check matrix to MacKay alist text. check_nodes[r]
    is the 0-based variable indices of parity check r; the column lists are its
    transpose."""
    rows = [sorted(v + 1 for v in row) for row in check_nodes]
    cols: list[list[int]] = [[] for _ in range(block_size)]
    for check_index, row in enumerate(rows, start=1):
        for variable in row:
            cols[variable - 1].append(check_index)
    col_w = [len(c) for c in cols]
    row_w = [len(r) for r in rows]
    dmax_c, dmax_r = max(col_w), max(row_w)
    lines = [
        f"{block_size} {len(rows)}",
        f"{dmax_c} {dmax_r}",
        " ".join(map(str, col_w)),
        " ".join(map(str, row_w)),
    ]
    lines += [" ".join(map(str, c + [0] * (dmax_c - len(c)))) for c in cols]
    lines += [" ".join(map(str, r + [0] * (dmax_r - len(r)))) for r in rows]
    return "\n".join(lines) + "\n"


def make_ldpc_decoder(
    ctx: Any,
    *,
    block_size: int,
    check_nodes: list[list[int]],
    max_iterations: int,
) -> Any:
    fd, path = tempfile.mkstemp(prefix="marconi-ldpc-", suffix=".alist")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(_alist(block_size, check_nodes))
        decoder = ctx.fec.ldpc_decoder.make(path, max_iterations)
    finally:
        os.unlink(path)
    return ctx.fec.extended_decoder(
        decoder, threading=None, ann=None, puncpat="11", integration_period=10000
    )
