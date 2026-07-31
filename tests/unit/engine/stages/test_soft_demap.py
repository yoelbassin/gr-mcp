from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import HardenStep
from marconi.engine.modulation.ofdm.stages import SoftDemapStep
from marconi.engine.modulation.psk.stages import PskSoftDemapStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.params import ParamValue

SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT)


def _run_bits(
    tmp_path: Path, syms: np.ndarray, params: dict[str, ParamValue]
) -> np.ndarray:
    ensure_worker_warm()
    src = tmp_path / "in.cf32"
    syms.astype(np.complex64).tofile(src)
    snk = tmp_path / "out.u8"
    modem = Modem(
        symbol_rate=1.0,
        path=[SoftDemapStep.model_validate(params), HardenStep()],
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
    modem = Modem(
        symbol_rate=1.0,
        path=[PskSoftDemapStep(order=PskOrder(4)), HardenStep()],
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
    framed = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, frame_len=65)
    out = stage_registry()["soft_demap"].out_descriptor(
        framed, SoftDemapStep(scheme="psk", order=4)
    )
    assert out.frame_len == 130
    assert out.level is Level.BITS and out.item_type == "f"
    assert out.carrier is Carrier.SOFT


def test_soft_demap_declares_required_order() -> None:
    stage = stage_registry()["soft_demap"]
    assert stage.required_input_order(SoftDemapStep(scheme="qam", order=16)) == 16
    assert (
        stage.required_input_order(
            SoftDemapStep(scheme="explicit", points_i=[-1.0, 1.0], points_q=[0.0, 0.0])
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
        {"scheme": "psk", "order": 4, "points_q": [0.0, 1.0]},  # stray points_q
    ],
)
def test_soft_demap_rejects_malformed_params(params: dict[str, ParamValue]) -> None:
    with pytest.raises(Exception):
        stage_registry()["soft_demap"].step_model.model_validate(params)


def test_soft_demap_qam16_decodes_every_point_to_its_index(tmp_path: Path) -> None:
    from gnuradio import digital

    con = digital.qam.qam_constellation(
        constellation_points=16,
        differential=False,
        mod_code=digital.mod_codes.GRAY_CODE,
        large_ampls_to_corners=False,
    )
    con.normalize(digital.constellation.POWER_NORMALIZATION)
    pts = np.asarray(con.points(), dtype=np.complex64)
    reps = np.tile(pts, 8)
    bits = _run_bits(tmp_path, reps, {"scheme": "qam", "order": 16})
    expect = np.tile(
        np.array(
            [[(v >> (3 - j)) & 1 for j in range(4)] for v in range(16)], np.uint8
        ).reshape(-1),
        8,
    )
    assert bits.tolist() == expect.tolist()


def test_soft_demap_qam64_is_rejected() -> None:
    with pytest.raises(Exception, match="order 16 only"):
        stage_registry()["soft_demap"].step_model.model_validate(
            {"scheme": "qam", "order": 64}
        )
