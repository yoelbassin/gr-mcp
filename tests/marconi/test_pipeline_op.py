from pathlib import Path

import pytest

from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.ops.pipeline import run_pipeline, save_pipeline_to_workspace
from marconi.vocabulary import PipelineValidationError
from marconi.workspace import Workspace


def _spec(out: str) -> PipelineSpec:
    return PipelineSpec(
        name="tone",
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 50e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": 10000}),
            BlockSpec(id="snk", type="file_sink", params={"path": out}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_run_validates_first(tmp_path: Path) -> None:
    bad = _spec(str(tmp_path / "o.cf32"))
    bad.blocks[0] = BlockSpec(id="src", type="tone_source", params={})
    with pytest.raises(PipelineValidationError, match="freq"):
        run_pipeline(bad)


def test_run_executes_valid_pipeline(tmp_path: Path) -> None:
    result = run_pipeline(_spec(str(tmp_path / "o.cf32")))
    assert result.status == "ok"
    assert (tmp_path / "o.cf32").exists()


def test_save_pipeline_to_workspace(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    p = save_pipeline_to_workspace(_spec("o.cf32"), ws)
    assert p.parent == ws.root / "pipelines"
    assert p.suffix == ".yaml"
