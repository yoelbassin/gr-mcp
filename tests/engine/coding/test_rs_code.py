"""rs_code: generic Reed-Solomon over GF(2^m) in the coding lane. Field
polynomial, code lengths, and root parameters are caller data — no protocol
constants in src. Encoding here runs the reedsolo library in reverse, so the
decode path is checked against an independent implementation of the math."""

from __future__ import annotations

import numpy as np
import pytest
import reedsolo
from pydantic import ValidationError

from marconi.engine.coding.carrier import CodingCarrier, Window
from marconi.engine.coding.ops_bits import rs_code_rx
from marconi.engine.coding.stages_bits import RsCode
from marconi.engine.stages import stage_registry

RS15: dict[str, int] = {
    "symbol_bits": 4,
    "n": 15,
    "k": 9,
    "prim_poly": 0x13,
    "fcr": 1,
}
DATA15 = list(range(1, 10))


def _encode(
    data: list[int],
    *,
    symbol_bits: int,
    n: int,
    k: int,
    prim_poly: int,
    fcr: int,
    generator: int = 2,
) -> list[int]:
    codec = reedsolo.RSCodec(
        nsym=n - k,
        nsize=n,
        fcr=fcr,
        prim=prim_poly,
        generator=generator,
        c_exp=symbol_bits,
    )
    return list(codec.encode(list(data)))


def _sym_bits(symbols: list[int], width: int) -> np.ndarray:
    out = [(s >> j) & 1 for s in symbols for j in range(width - 1, -1, -1)]
    return np.array(out, np.uint8)


def _decode(c: CodingCarrier, p: dict[str, int], emit: str = "data") -> CodingCarrier:
    return rs_code_rx(
        c,
        symbol_bits=p["symbol_bits"],
        n=p["n"],
        k=p["k"],
        prim_poly=p["prim_poly"],
        fcr=p["fcr"],
        emit=emit,
    )


def test_rs_code_corrects_symbol_errors_in_the_blind_scope() -> None:
    word = _encode(DATA15, **RS15)
    word[0] ^= 5
    word[7] ^= 3
    word[14] ^= 1
    out = _decode(CodingCarrier(bits=_sym_bits(word, 4)), RS15)
    assert np.array_equal(out.bits, _sym_bits(DATA15, 4))


def test_rs_code_decodes_consecutive_codewords() -> None:
    d2 = list(range(9, 0, -1))
    stream = _encode(DATA15, **RS15) + _encode(d2, **RS15)
    stream[3] ^= 7
    stream[20] ^= 2
    out = _decode(CodingCarrier(bits=_sym_bits(stream, 4)), RS15)
    assert np.array_equal(out.bits, _sym_bits(DATA15 + d2, 4))


def test_rs_code_windowed_scope_decodes_per_window() -> None:
    d2 = list(range(9, 0, -1))
    stream = _sym_bits(_encode(DATA15, **RS15) + _encode(d2, **RS15), 4)
    seeded = CodingCarrier(
        bits=stream, windows=[Window(start=0, cursor=0), Window(start=60, cursor=60)]
    )
    out = _decode(seeded, RS15)
    assert np.array_equal(out.bits, _sym_bits(DATA15 + d2, 4))
    assert [w.start for w in out.windows or []] == [0, 36]


def test_rs_code_seeded_empty_scope_decodes_nothing() -> None:
    word = _encode(DATA15, **RS15)
    out = _decode(CodingCarrier(bits=_sym_bits(word, 4), windows=[]), RS15)
    assert out.bits.size == 0


def test_rs_code_uncorrectable_block_passes_data_through_unchanged() -> None:
    word = _encode(DATA15, **RS15)
    for i in (0, 2, 4, 6, 8, 10, 12):
        word[i] ^= 7  # beyond t=3: reedsolo raises, verified deterministic
    out = _decode(CodingCarrier(bits=_sym_bits(word, 4)), RS15)
    assert np.array_equal(out.bits, _sym_bits(word[:9], 4))


def test_rs_code_emit_codeword_returns_the_corrected_word() -> None:
    clean = _encode(DATA15, **RS15)
    word = list(clean)
    word[2] ^= 9
    out = _decode(CodingCarrier(bits=_sym_bits(word, 4)), RS15, emit="codeword")
    assert np.array_equal(out.bits, _sym_bits(clean, 4))


def test_rs_code_shortened_gf64_code_corrects() -> None:
    params: dict[str, int] = {
        "symbol_bits": 6,
        "n": 24,
        "k": 12,
        "prim_poly": 0x43,
        "fcr": 1,
    }
    data = list(range(1, 13))
    word = _encode(data, **params)
    word[3] ^= 0x2A
    word[20] ^= 1
    out = _decode(CodingCarrier(bits=_sym_bits(word, 6)), params)
    assert np.array_equal(out.bits, _sym_bits(data, 6))


def test_rs_code_stage_is_registered_and_validates_shape() -> None:
    assert "rs_code" in stage_registry()
    ok = dict(RS15)
    RsCode._Params.model_validate(ok)
    for bad in [
        {**ok, "k": 15},
        {**ok, "n": 16},
        {**ok, "prim_poly": 0x7},
        {**ok, "emit": "junk"},
    ]:
        with pytest.raises(ValidationError):
            RsCode._Params.model_validate(bad)
