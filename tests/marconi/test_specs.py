from pathlib import Path

from marconi.models import (
    BlockSpec,
    ConnectionSpec,
    DeviceInfo,
    PipelineSpec,
    RunResult,
    SceneElement,
    SceneSpec,
)
from marconi.specs import load_pipeline, load_scene, save_pipeline, save_scene


def _pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="tone_to_file",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": 50000}),
            BlockSpec(id="snk", type="file_sink", params={"path": "out.cf32"}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_pipeline_yaml_roundtrip(tmp_path: Path) -> None:
    spec = _pipeline()
    path = save_pipeline(spec, tmp_path / "p.yaml")
    assert path.exists()
    assert load_pipeline(path) == spec


def test_scene_yaml_roundtrip(tmp_path: Path) -> None:
    scene = SceneSpec(
        name="three_signals",
        elements=[
            SceneElement(kind="tone", freq=100.1e6, amplitude=0.5),
            SceneElement(kind="noise", amplitude=0.01),
            SceneElement(kind="fm_tone", freq=100.3e6, params={"mod_freq": 1e3}),
        ],
    )
    path = save_scene(scene, tmp_path / "s.yaml")
    assert load_scene(path) == scene


def test_run_result_and_device_info_construct() -> None:
    r = RunResult(status="ok", elapsed_seconds=1.5, artifacts=[Path("a.cf32")])
    assert r.status == "ok" and r.error is None
    d = DeviceInfo(id="sim0", kind="simulated", can_tx=True)
    assert d.can_tx


def test_connection_defaults_ports_to_zero() -> None:
    c = ConnectionSpec(src_block="a", dst_block="b")
    assert c.src_port == 0 and c.dst_port == 0
