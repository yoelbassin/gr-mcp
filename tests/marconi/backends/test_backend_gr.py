import numpy as np
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


def test_run_failure_is_concise_not_traceback() -> None:
    # a runtime flowgraph failure must surface a one-line, path-free message to
    # the agent; the full traceback is logged server-side, never returned.
    from marconi.backends.gnuradio_backend import _run_with_timeout

    class _Boom:
        def run(self) -> None:
            raise RuntimeError("bad block parameter")

        def stop(self) -> None: ...

        def wait(self) -> None: ...

    timed_out, failure = _run_with_timeout(_Boom(), timeout=5.0, name="t")
    assert timed_out is False
    assert failure == "RuntimeError: bad block parameter"
    assert failure is not None and "Traceback" not in failure


def _tone_pipeline(
    out_path: str, n: int = 50000, noise_amplitude: float = 0.0
) -> PipelineSpec:
    if noise_amplitude == 0.0:
        # Pure three-block spec: tone_source → head → file_sink.
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
    # A pure synthetic tone has no noise floor so find_signals() sees
    # phase-accumulator sidebands from GR's NCO and may detect >1 signal.
    # The closed-loop test needs a realistic noise floor: mix in noise before head.
    return PipelineSpec(
        name="tone_to_file",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(
                id="noise",
                type="noise_source",
                params={"amplitude": noise_amplitude},
            ),
            BlockSpec(id="mix", type="add", params={}),
            BlockSpec(id="hd", type="head", params={"num_samples": n}),
            BlockSpec(id="snk", type="file_sink", params={"path": out_path}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="mix", dst_port=0),
            ConnectionSpec(src_block="noise", dst_block="mix", dst_port=1),
            ConnectionSpec(src_block="mix", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


@pytest.mark.gnuradio
def test_build_top_block(tmp_path) -> None:
    from marconi.backends.gnuradio_backend import build_top_block

    spec = _tone_pipeline(str(tmp_path / "o.cf32"))
    tb, artifacts = build_top_block(spec)
    assert artifacts == [tmp_path / "o.cf32"]
    assert hasattr(tb, "run")  # it is a gr.top_block


@pytest.mark.gnuradio
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


@pytest.mark.gnuradio
def test_run_pipeline_produces_analyzable_capture(tmp_path) -> None:
    """The loop closes: backend-generated samples are found by Plan-1 analysis."""
    import marconi
    from marconi.backends import get_backend

    out = tmp_path / "tone.cf32"
    result = get_backend("gnuradio").run_pipeline(
        _tone_pipeline(str(out), noise_amplitude=0.001)
    )
    assert result.status == "ok"
    assert result.artifacts == [out]
    assert result.error is None
    assert result.elapsed_seconds < 30

    ws = marconi.Workspace(tmp_path / "project")
    ref = marconi.load_capture(out, ws, sample_rate=1e6, center_freq=433e6)
    signals = marconi.find_signals(ref)
    assert len(signals) == 1
    assert abs(signals[0].center_freq - 433.1e6) < 2e3


@pytest.mark.gnuradio
def test_run_pipeline_timeout(tmp_path) -> None:
    """A never-ending flowgraph is stopped by the watchdog."""
    raw = tmp_path / "loop.cf32"
    np.zeros(1024, dtype=np.complex64).tofile(raw)
    spec = PipelineSpec(
        name="endless",
        sample_rate=1e6,
        blocks=[
            BlockSpec(
                id="src",
                type="file_source",
                params={"path": str(raw), "repeat": True},
            ),
            BlockSpec(
                id="snk", type="file_sink", params={"path": str(tmp_path / "o.cf32")}
            ),
        ],
        connections=[ConnectionSpec(src_block="src", dst_block="snk")],
    )
    from marconi.backends import get_backend

    result = get_backend("gnuradio").run_pipeline(spec, timeout=1.0)
    assert result.status == "timeout"
    assert result.elapsed_seconds >= 1.0
