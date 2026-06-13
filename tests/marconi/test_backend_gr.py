import pytest

from marconi.backends import get_backend
from marconi.backends.base import Backend
from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("imaginary")


def test_gnuradio_backend_resolves() -> None:
    b = get_backend("gnuradio")
    assert isinstance(b, Backend)
    assert b.name == "gnuradio"
    assert b.enumerate_devices() == []  # no hardware support in v1.0


def _tone_pipeline(out_path: str, n: int = 50000) -> PipelineSpec:
    return PipelineSpec(
        name="tone_to_file",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": n}),
            BlockSpec(id="snk", type="file_sink", params={"path": out_path}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_build_top_block(tmp_path) -> None:
    from marconi.backends.gnuradio_backend import build_top_block

    spec = _tone_pipeline(str(tmp_path / "o.cf32"))
    tb, artifacts = build_top_block(spec)
    assert artifacts == [tmp_path / "o.cf32"]
    assert hasattr(tb, "run")  # it is a gr.top_block


def test_build_error_carries_block_id(tmp_path) -> None:
    from marconi.backends.base import BackendError
    from marconi.backends.gnuradio_backend import build_top_block

    spec = _tone_pipeline(str(tmp_path / "o.cf32"))
    # nbfm_tx with the verified-broken 20x ratio -> GR raises at construction
    spec.blocks.append(
        BlockSpec(
            id="bad_tx",
            type="nbfm_tx",
            params={"audio_rate": 50000, "quad_rate": 1000000},
        )
    )
    spec.blocks.append(
        BlockSpec(id="audio", type="audio_tone_source", params={"freq": 1e3})
    )
    spec.connections.append(ConnectionSpec(src_block="audio", dst_block="bad_tx"))

    with pytest.raises(BackendError, match="bad_tx"):
        build_top_block(spec)
