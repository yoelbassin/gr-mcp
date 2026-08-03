"""Locate the LDPC codes GNU Radio ships as test fixtures. A concrete
parity-check matrix is datasheet data, so it lives in tests, never in the
product; the alist files gr-fec installs give real, encodable codes without
vendoring one into the repo. gap-form files carry (N, K, gap) in the filename:
gap is the encoder's structured-elimination parameter (decode never needs it)."""

from __future__ import annotations

import re
from pathlib import Path


def ldpc_codes() -> list[tuple[Path, int, int, int]]:
    """(alist path, N, K, gap) for every shipped gap-form code, smallest N first."""
    from gnuradio import gr

    root = Path(gr.prefix()) / "share" / "gnuradio" / "fec" / "ldpc"
    if not root.is_dir():
        return []
    out: list[tuple[Path, int, int, int]] = []
    for path in root.glob("*gap*.alist"):
        m = re.match(r"n_(\d+)_k_(\d+)_gap_(\d+)", path.name)
        if m:
            out.append((path, int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return sorted(out, key=lambda t: (t[1], t[2]))


def parse_check_nodes(path: Path, block_size: int) -> list[list[int]]:
    """The M parity rows of an alist as 0-based variable-index lists (the sparse
    parity-check matrix the ldpc stage takes as a caller param)."""
    nums = [ln.split() for ln in path.read_text().splitlines() if ln.strip()]
    m = int(nums[0][1])
    row_lines = nums[4 + block_size : 4 + block_size + m]
    return [[int(x) - 1 for x in ln if int(x) != 0] for ln in row_lines]
