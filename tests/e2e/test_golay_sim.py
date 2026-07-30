"""Golay(23,12) t=3 through the GENERIC vocabulary: sync_word seeds windows,
block_code(correct=3) repairs three errors per codeword. Generator
polynomial, masks, sync word and payload are caller data, only here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import bitops
from helpers.golay import golay_codeword, golay_masks

from marconi.engine.coding.stages_bits import BlockCodeStep, SyncWordStep
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import EmitMode
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem

BITS = Descriptor(Level.BITS, "b")
SYNC = "b433"


def _encode(words: list[int]) -> np.ndarray:
    return np.concatenate([golay_codeword(w) for w in words]).astype(np.uint8)


_WORDS = [0b101100111010, 0b000011111100, 0b110101010101, 0b011110000111]


def test_golay_frames_survive_three_errors_each(tmp_path: Path) -> None:
    coded = _encode(_WORDS)
    rng = np.random.default_rng(11)
    for w in range(len(_WORDS)):
        pos = rng.choice(23, size=3, replace=False)
        coded[w * 23 + pos] ^= 1
    wire = np.concatenate(
        [
            np.ones(10, np.uint8),
            bitops.bytes_to_bits(bytes.fromhex(SYNC)),
            coded,
            np.zeros(7, np.uint8),
        ]
    ).astype(np.uint8)
    modem = Modem(
        symbol_rate=1.0,
        path=[
            SyncWordStep(sync=SYNC),
            BlockCodeStep(
                code_bits=23,
                data_bits=12,
                parity_masks=golay_masks(),
                correct=3,
                emit=EmitMode.DATA,
            ),
        ],
    )
    p = tmp_path / "golay.u8"
    write_bits(p, wire)
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=p, num_bits=int(wire.size)),
    )
    assert res.status == "ok", res
    assert res.bitstream is not None
    bits = read_bits(res.bitstream.path)
    w0 = res.windows[0]
    got = [
        sum(int(bits[w0 + w * 12 + j]) << j for j in range(12))
        for w in range(len(_WORDS))
    ]
    assert got == _WORDS
