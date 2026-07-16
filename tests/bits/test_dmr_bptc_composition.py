from __future__ import annotations

import numpy as np
from bits import _dmr


def _dibits(s: str) -> np.ndarray:
    return np.array([int(ch) for ch in s], np.uint8)


GOLDEN = {
    "preamble_csbk": (
        "101011320033231102132300010100101203222120222103002031313333111331"
        "131131331131301203221203132120113131200320300001131102120120000011",
        3169855,
        2247700,
    ),
    "udt_header": (
        "020302120111130211203300032201200110010233232200302121313333111331"
        "131131331131112100123001221100210312031231201201312010011320301202",
        3169855,
        2247700,
    ),
}


def test_bptc_composes_from_generic_stages_on_golden_vectors() -> None:
    for name, (dibits, src, tgt) in GOLDEN.items():
        d = _dibits(dibits)
        assert d.size == 132
        r = _dmr.decode_burst(d)
        assert r is not None, name
        assert r["source"] == src and r["target"] == tgt, (name, r)
