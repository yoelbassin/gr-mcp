import pytest
from pydantic import ValidationError

from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline


def test_pipeline_round_trips_through_json() -> None:
    pipe = GrPipeline(
        name="p",
        sample_rate=1e6,
        blocks=[
            GrBlock(id="a", kind="src"),
            GrBlock(id="b", kind="snk", params={"path": "x"}),
        ],
        connections=[GrConnection(src_block="a", dst_block="b")],
    )
    again = GrPipeline.model_validate_json(pipe.model_dump_json())
    assert again == pipe


def test_block_lookup_and_ids() -> None:
    pipe = GrPipeline(sample_rate=1.0, blocks=[GrBlock(id="a", kind="x")])
    assert pipe.block_ids == ["a"]
    assert pipe.block("a").kind == "x"
    with pytest.raises(KeyError):
        pipe.block("missing")


def test_duplicate_block_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        GrPipeline(
            sample_rate=1.0,
            blocks=[GrBlock(id="a", kind="x"), GrBlock(id="a", kind="y")],
        )


def test_sample_rate_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        GrPipeline(sample_rate=0.0, blocks=[])


def test_connection_defaults_to_port_zero() -> None:
    c = GrConnection(src_block="a", dst_block="b")
    assert c.src_port == 0 and c.dst_port == 0
