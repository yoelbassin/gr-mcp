"""DMR BPTC(196,96) as caller data for the composition proof (Task 4), now
driven through the product coding engine (a coding-only ModemSpec over
run_rx) instead of calling the plain-function bptc_generic in _dmr.py
directly. Every DMR value here -- deinterleave/transpose/extract perms,
Hamming-derived parity masks, CRC xorouts, field offsets -- is caller data
(tests/e2e/_dmr.py); the generic vocabulary (permute + block_code
emit="codeword", iterated row/column) carries the whole product-code decode
in one pipeline. Tables verified vs OK-DMRlib + dsd-neo.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from e2e import _dmr

from marconi.core.bitfile import read_bits, write_bits
from marconi.core.descriptor import Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.phy.engine import run_rx
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

BITS = Descriptor(Level.BITS, "b")

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


def _dibits(s: str) -> np.ndarray:
    return np.array([int(ch) for ch in s], np.uint8)


def _row_step() -> ModemStep:
    return ModemStep(
        conv="block_code",
        params={
            "code_bits": 15,
            "data_bits": 11,
            "parity_masks": cast("list[float | int]", _dmr.ROW_MASKS),
            "correct": True,
            "emit": "codeword",
        },
    )


def _col_step() -> ModemStep:
    return ModemStep(
        conv="block_code",
        params={
            "code_bits": 13,
            "data_bits": 9,
            "parity_masks": cast("list[float | int]", _dmr.COL_MASKS),
            "correct": True,
            "emit": "codeword",
        },
    )


def _transpose_step() -> ModemStep:
    return ModemStep(
        conv="permute", params={"perm": cast("list[float | int]", _dmr.TRANSPOSE)}
    )


def _transpose_inv_step() -> ModemStep:
    return ModemStep(
        conv="permute", params={"perm": cast("list[float | int]", _dmr.TRANSPOSE_INV)}
    )


def _bptc_modem() -> ModemSpec:
    round_trip = [_row_step(), _transpose_step(), _col_step(), _transpose_inv_step()]
    return ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(
                conv="permute",
                params={"perm": cast("list[float | int]", _dmr.SCATTER_INV)},
            ),
            ModemStep(
                conv="permute",
                params={"perm": cast("list[float | int]", list(range(1, 196)))},
            ),
            *round_trip,
            *round_trip,
            ModemStep(
                conv="permute",
                params={"perm": cast("list[float | int]", _dmr.EXTRACT)},
            ),
        ],
    )


def _decode_payload(dibits: np.ndarray, tmp_path: Path) -> dict[str, int | str] | None:
    burst264 = _dmr._dibits_to_bits(dibits)
    info196 = np.concatenate([burst264[0:98], burst264[166:264]])
    src = tmp_path / "info.u8"
    write_bits(src, info196)
    res = run_rx(
        _bptc_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=src, num_bits=int(info196.size)),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    payload = read_bits(res.bitstream.path)
    assert payload.size == 96
    for kind, xorout in (
        ("csbk", _dmr.CSBK_XOROUT),
        ("data_header", _dmr.HEADER_XOROUT),
    ):
        if not _dmr._crc_ok(payload, xorout):
            continue
        if kind == "csbk":
            return {
                "kind": kind,
                "csbko": _dmr._u(payload, *_dmr.CSBK_CSBKO),
                "target": _dmr._u(payload, *_dmr.CSBK_TARGET),
                "source": _dmr._u(payload, *_dmr.CSBK_SOURCE),
            }
        return {
            "kind": kind,
            "target": _dmr._u(payload, *_dmr.HEADER_TARGET),
            "source": _dmr._u(payload, *_dmr.HEADER_SOURCE),
        }
    return None


def test_bptc_composes_from_generic_stages_on_golden_vectors(tmp_path: Path) -> None:
    for name, (dibits, src, tgt) in GOLDEN.items():
        d = _dibits(dibits)
        assert d.size == 132
        r = _decode_payload(d, tmp_path)
        assert r is not None, name
        assert r["source"] == src and r["target"] == tgt, (name, r)
