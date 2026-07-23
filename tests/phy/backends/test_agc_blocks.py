from pathlib import Path

import numpy as np
import pytest

from marconi.phy.backends.gnuradio.blocks import GR_BLOCKS, _make_ctx
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.ir import GrBlock, GrConnection, GrPipeline

_KINDS = [
    ("feedforward_agc_cc", {"nsamples": 64, "reference": 1.0}),
    (
        "agc2_cc",
        {
            "attack_rate": 0.01,
            "decay_rate": 0.001,
            "reference": 1.0,
            "max_gain": 0.0,
        },
    ),
]


@pytest.mark.parametrize("kind,params", _KINDS)
def test_agc_factory_constructs(kind: str, params: dict) -> None:
    assert GR_BLOCKS[kind](_make_ctx(4.0), params) is not None


def test_agc2_applies_max_gain() -> None:
    blk = GR_BLOCKS["agc2_cc"](
        _make_ctx(4.0),
        {
            "attack_rate": 0.01,
            "decay_rate": 0.001,
            "reference": 1.0,
            "max_gain": 7.5,
        },
    )
    assert blk.max_gain() == pytest.approx(7.5)


_N = 200_000
_SETTLE = 100_000


def _two_level_signal() -> np.ndarray:
    mags = np.where(np.arange(_N) % 2 == 0, 1.0, 3.0)
    phase = np.exp(2j * np.pi * np.arange(_N) / 7.0)
    return (mags * phase).astype(np.complex64)


def _run_block(tmp_path: Path, kind: str, params: dict) -> np.ndarray:
    src = tmp_path / "in.cf32"
    snk = tmp_path / "out.cf32"
    _two_level_signal().tofile(src)
    pipe = GrPipeline(
        name="agc_char",
        sample_rate=1.0e6,
        blocks=[
            GrBlock(id="src", kind="iq_file_source", params={"path": str(src)}),
            GrBlock(id="agc", kind=kind, params=params),
            GrBlock(id="snk", kind="iq_file_sink", params={"path": str(snk)}),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="agc"),
            GrConnection(src_block="agc", dst_block="snk"),
        ],
    )
    assert GnuRadioBackend().run_pipeline(pipe, timeout=180.0).status == "ok"
    return np.fromfile(snk, dtype=np.complex64)[_SETTLE:]


def _statistics(z: np.ndarray) -> dict[str, float]:
    mag = np.abs(z)
    return {
        "mean_mag": float(mag.mean()),
        "rms": float(np.sqrt((mag**2).mean())),
        "peak": float(mag.max()),
    }


def test_v1_characterize_feedforward(tmp_path: Path) -> None:
    ensure_worker_warm()
    z = _run_block(tmp_path, "feedforward_agc_cc", {"nsamples": 1024, "reference": 1.0})
    stats = _statistics(z)
    print(f"\nV1 feedforward -> {stats}")
    # measured: feedforward_agc_cc's alpha-max-beta-min envelope (coeffs 1,
    # 0.4) overestimates true magnitude by up to ~7.7% off-axis, biasing
    # peak-normalized output ~7% below the reference rather than settling on it
    assert stats["peak"] == pytest.approx(0.9306, abs=0.01), stats


def test_v1_characterize_feedback(tmp_path: Path) -> None:
    ensure_worker_warm()
    z = _run_block(
        tmp_path,
        "agc2_cc",
        {
            "attack_rate": 0.01,
            "decay_rate": 0.001,
            "reference": 1.0,
            "max_gain": 0.0,
        },
    )
    stats = _statistics(z)
    print(f"\nV1 feedback -> {stats}")
    assert stats["mean_mag"] == pytest.approx(1.0, abs=0.05), stats
