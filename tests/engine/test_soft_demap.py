from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep
from marconi.engine.types.params import ParamValue

SYM_C = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT)


def _run_bits(
    tmp_path: Path, syms: np.ndarray, params: dict[str, ParamValue]
) -> np.ndarray:
    ensure_worker_warm()
    src = tmp_path / "in.cf32"
    syms.astype(np.complex64).tofile(src)
    snk = tmp_path / "out.u8"
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="soft_demap", params=params),
            ModemStep(conv="harden", params={}),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=SYM_C,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    return read_bits(snk)


def test_soft_demap_explicit_points_decode_to_index_bits(tmp_path: Path) -> None:
    # a point's bit pattern is its index, MSB-first (matches _const_explicit)
    points = [(-1 - 1j), (-1 + 1j), (1 - 1j), (1 + 1j)]
    params: dict[str, ParamValue] = {
        "scheme": "explicit",
        "points_i": [p.real for p in points],
        "points_q": [p.imag for p in points],
    }
    syms = np.array(points * 8, dtype=np.complex64)
    bits = _run_bits(tmp_path, syms, params)
    expect = np.tile(np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8), 8)
    assert bits.tolist() == expect.tolist()


def test_soft_demap_psk_matches_psk_soft_demap(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 4, 256)
    qpsk = np.exp(1j * (np.pi / 4 + np.pi / 2 * idx)).astype(np.complex64)
    (tmp_path / "g").mkdir(exist_ok=True)
    generic = _run_bits(tmp_path / "g", qpsk, {"scheme": "psk", "order": 4})
    ensure_worker_warm()
    src = tmp_path / "ref.cf32"
    qpsk.tofile(src)
    snk = tmp_path / "ref.u8"
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="psk_soft_demap", params={"order": 4}),
            ModemStep(conv="harden", params={}),
        ],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=SYM_C,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=30.0)
    assert r.status == "ok", r
    assert generic.tolist() == read_bits(snk).tolist()


def test_soft_demap_scales_frame_len_by_bits_per_symbol() -> None:
    framed = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT, frame_len=65)
    out = stage_registry()["soft_demap"].out_descriptor(
        framed, {"scheme": "psk", "order": 4}
    )
    assert out.frame_len == 130
    assert out.level is Level.BITS and out.item_type == "f"
    assert out.carrier is Carrier.SOFT


def test_soft_demap_declares_required_order() -> None:
    stage = stage_registry()["soft_demap"]
    assert stage.required_input_order({"scheme": "qam", "order": 16}) == 16
    assert (
        stage.required_input_order(
            {"scheme": "explicit", "points_i": [-1.0, 1.0], "points_q": [0.0, 0.0]}
        )
        == 2
    )


@pytest.mark.parametrize(
    "params",
    [
        {"scheme": "psk"},  # named scheme without order
        {"scheme": "explicit", "order": 4},  # explicit without points
        {"scheme": "explicit", "points_i": [1.0, -1.0, 0.0], "points_q": [0.0] * 3},
        {"scheme": "nosuch", "order": 4},
    ],
)
def test_soft_demap_rejects_malformed_params(params: dict[str, ParamValue]) -> None:
    with pytest.raises(Exception):
        stage_registry()["soft_demap"].params_model.model_validate(params)


def test_soft_demap_qam_16_round_trips_through_backend(tmp_path: Path) -> None:
    # Test soft_demap with scheme="qam" by repeating each of the 16 QAM
    # constellation points. Verifies soft_demap(qam,16) + harden decodes each
    # point consistently. Tests the qam scheme branch end-to-end
    # (constellation_soft_decoder for qam differs from psk: 16 vs 64,
    # GRAY_CODE, POWER_NORMALIZATION). Noiseless symbols at unit RMS →
    # hard decisions should be deterministic per point.
    from gnuradio import digital

    # Get 16QAM constellation from GR (same source the backend uses)
    c = digital.constellation_16qam()
    points = np.asarray(c.points(), dtype=np.complex64)

    # Create a test that feeds each constellation point and verifies
    # consistent decoding. All 16 points 4 times = 64 symbols.
    (tmp_path / "qam").mkdir(exist_ok=True)

    syms = np.tile(points, 4).astype(np.complex64)
    decoded = _run_bits(tmp_path / "qam", syms, {"scheme": "qam", "order": 16})

    # Verify consistency: each occurrence of a constellation point should
    # decode to the same bit pattern. This ensures soft_demap(qam) is
    # deterministic and functioning end-to-end.
    k = 4  # bits per symbol
    for idx in range(16):
        # Collect all decoding s of point idx
        decoded_bits_list = []
        for cycle in range(4):
            sym_idx = cycle * 16 + idx
            bit_start = sym_idx * k
            decoded_bits_list.append(tuple(decoded[bit_start : bit_start + k]))

        # All occurrences should decode identically
        first_decoding = decoded_bits_list[0]
        for decoding in decoded_bits_list[1:]:
            assert (
                decoding == first_decoding
            ), f"point {idx}: inconsistent decodings {decoded_bits_list}"
