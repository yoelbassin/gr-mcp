"""No signal present is the NORMAL case for a survey tool.

The correct answer to silence, to a stuck rail, and to noise is zero frames --
never an exception. Every shipped codec is run against all three: a decode that
crashes when the band is empty cannot be pointed at a band.

Regression: POCSAG's codec raised IndexError on all three inputs. permute_rx's
unseeded branch assumed perm was a permutation of range(len(perm)), but POCSAG's
is a DROPPING gather (496 indices spanning 511 bits), and that branch is exactly
where execution lands when sync_word seeds no frames.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.bits.carriers import RxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
from marconi.bits.registry import registry
from marconi.bits.seam import parse_bitstream
from marconi.bits.stages.framing_ops import Permute
from marconi.core.models import Bitstream, Symbolstream

sys.path.insert(0, str(Path(__file__).resolve().parent))

_N = 60_000


def _codecs() -> dict[str, Callable[[], CodecSpec]]:
    from test_ais_offair import _ais_codec
    from test_ble_offair import _ble_codec
    from test_dab_codec import _codec as _dab_codec
    from test_dmr_offair import _dmr_codec
    from test_lora_codec import _codec as _lora_bits_codec
    from test_pocsag_offair import _pocsag_codec

    return {
        "ais": _ais_codec,
        "ble": _ble_codec,
        "dab": _dab_codec,
        "dmr": _dmr_codec,
        "lora": _lora_bits_codec,
        "pocsag": _pocsag_codec,
    }


def _no_signal(kind: str, rng: np.random.Generator, soft: bool) -> np.ndarray:
    if soft:
        # what an fsk discriminator emits with no carrier: DC, a stuck rail, noise
        return {
            "zeros": np.zeros(_N, np.float32),
            "ones": np.ones(_N, np.float32),
            "noise": rng.normal(0, 1, _N).astype(np.float32),
        }[kind]
    return {
        "zeros": np.zeros(_N, np.uint8),
        "ones": np.ones(_N, np.uint8),
        "noise": rng.integers(0, 2, _N).astype(np.uint8),
    }[kind]


@pytest.mark.parametrize("name", sorted(_codecs()))
@pytest.mark.parametrize("kind", ["zeros", "ones", "noise"])
def test_shipped_codec_survives_no_signal(name: str, kind: str, tmp_path: Path) -> None:
    codec = _codecs()[name]()
    reg = registry()
    wants_symbols = compile_codec(codec, reg, "rx").requires_symbol_input
    data = _no_signal(kind, np.random.default_rng(0), soft=wants_symbols)
    path = tmp_path / f"{name}_{kind}.bin"
    data.tofile(path)
    if wants_symbols:
        stream: Bitstream | Symbolstream = Symbolstream(
            path=path, num_symbols=data.size, item_type="f"
        )
    else:
        stream = Bitstream(path=path, num_bits=data.size)
    result = parse_bitstream(stream, codec, reg)
    assert result.num_crc_ok == 0, f"{name} found a CRC-valid frame in {kind}"


def test_dropping_gather_works_in_the_unseeded_scope() -> None:
    """The POCSAG shape: read 3 of every 4 bits, i.e. span 4 but length 3."""
    perm = [0, 1, 2, 4, 5, 6]
    bits = np.arange(16, dtype=np.uint8) % 2
    out = run_program(
        compile_codec(
            CodecSpec(path=[CodecStep(conv="permute", params={"perm": perm})]),
            registry(),
            "rx",
        ),
        RxCarrier(bits=bits),
    )
    span = max(perm) + 1
    blocks = bits.size // span
    want = np.concatenate([bits[b * span + np.asarray(perm)] for b in range(blocks)])
    assert np.array_equal(out.bits, want)
    assert out.bits.size == blocks * len(perm)


def test_unseeded_and_seeded_scopes_agree_on_stride() -> None:
    """A gather must consume the same input span whichever scope it runs in."""
    perm = [0, 2, 5]
    bits = np.random.default_rng(1).integers(0, 2, 60).astype(np.uint8)
    unseeded = run_program(
        compile_codec(
            CodecSpec(path=[CodecStep(conv="permute", params={"perm": perm})]),
            registry(),
            "rx",
        ),
        RxCarrier(bits=bits),
    )
    from marconi.bits.carriers import _Frame
    from marconi.bits.framing import permute_rx

    span = max(perm) + 1
    seeded = permute_rx(
        RxCarrier(
            bits=bits,
            frames=[_Frame(start=b * span, cursor=b * span) for b in range(10)],
        ),
        perm=perm,
    )
    assert np.array_equal(unseeded.bits, seeded.bits)


def test_negative_perm_index_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Permute._Params.model_validate({"perm": [0, -1, 2]})
    with pytest.raises(ValidationError):
        Permute._Params.model_validate({"perm": []})
