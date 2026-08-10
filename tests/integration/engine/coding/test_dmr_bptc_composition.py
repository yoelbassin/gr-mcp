"""DMR BPTC(196,96) as caller data for the composition proof, now
driven through the product coding engine (a coding-only Modem over
run_rx) instead of calling the plain-function bptc_generic in _dmr.py
directly. Every DMR value here -- deinterleave/transpose/extract perms,
Hamming-derived parity masks, CRC xorouts, field offsets -- is caller data
(tests/e2e/dmr/_dmr.py); the generic vocabulary (permute + block_code
emit="codeword", iterated row/column) carries the whole product-code decode
in one pipeline. Tables verified vs OK-DMRlib + dsd-neo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from e2e.dmr import _dmr

from marconi.engine.coding.stages_bits import SegmentStep
from marconi.engine.io.bitfile import read_bits, write_bits
from marconi.engine.run import run_rx
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Bitstream, Modem

BITS = Descriptor(Level.BITS, ItemType.B)

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


def _bptc_modem() -> Modem:
    return Modem(symbol_rate=1.0, path=_dmr.bptc_steps())


def _decode_payload(dibits: np.ndarray, tmp_path: Path) -> dict[str, int | str] | None:
    burst264 = _dmr._dibits_to_bits(dibits)
    info196 = burst264[np.asarray(_dmr.INFO_SPLICE, np.int64)]
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
    return _dmr.parse_payload(payload)


def test_bptc_composes_from_generic_stages_on_golden_vectors(tmp_path: Path) -> None:
    for name, (dibits, src, tgt) in GOLDEN.items():
        d = _dibits(dibits)
        assert d.size == 132
        r = _decode_payload(d, tmp_path)
        assert r is not None, name
        assert r["source"] == src and r["target"] == tgt, (name, r)


def _spliced_bptc_modem() -> Modem:
    """Full burst -> payload in one windowed pipeline: segment seeds a window
    per 264-bit burst, the interior splice gathers the info bits around the
    mid-burst sync, then the BPTC decode runs per window."""
    return Modem(
        symbol_rate=1.0,
        path=[
            SegmentStep(frame_body_len=264),
            _dmr.splice_step(),
            *_dmr.bptc_steps(),
        ],
    )


def test_interior_splice_and_bptc_decode_multiple_bursts(tmp_path: Path) -> None:
    """The seam-collapse fix: DMR's info bits wrap AROUND the mid-burst sync,
    and the whole splice + BPTC now runs in-product, windowed, over a stream of
    back-to-back bursts -- no test-side numpy splice before the pipeline."""
    bursts = [_dmr._dibits_to_bits(_dibits(g[0])) for g in GOLDEN.values()]
    for b in bursts:
        assert b.size == 264
    stream = np.concatenate(bursts).astype(np.uint8)
    src = tmp_path / "bursts.u8"
    write_bits(src, stream)

    res = run_rx(
        _spliced_bptc_modem(),
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=Bitstream(path=src, num_bits=int(stream.size)),
    )
    assert res.status == "ok", res
    assert res.windows == [0, 96], res.windows
    assert res.bitstream is not None
    payloads = read_bits(res.bitstream.path)
    assert payloads.size == 96 * len(bursts)

    for i, (name, (dibits, want_src, want_tgt)) in enumerate(GOLDEN.items()):
        payload = payloads[i * 96 : (i + 1) * 96]
        reference = _dmr.bptc_generic(
            np.concatenate(
                [bursts[i][0:98], bursts[i][166:264]]  # the numpy splice it replaces
            )
        )
        assert np.array_equal(payload, reference), name
        parsed = _dmr.parse_payload(payload)
        assert parsed is not None and parsed["source"] == want_src, (name, parsed)
        assert parsed["target"] == want_tgt, (name, parsed)
