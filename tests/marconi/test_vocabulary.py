from marconi.models import BlockSpec, ConnectionSpec, PipelineSpec
from marconi.vocabulary import (
    VOCABULARY,
    PipelineValidationError,
    validate_pipeline,
)


def _valid() -> PipelineSpec:
    return PipelineSpec(
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 100e3}),
            BlockSpec(id="hd", type="head", params={"num_samples": 1000}),
            BlockSpec(id="snk", type="file_sink", params={"path": "o.cf32"}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="hd"),
            ConnectionSpec(src_block="hd", dst_block="snk"),
        ],
    )


def test_valid_pipeline_has_no_issues() -> None:
    assert validate_pipeline(_valid()) == []


def test_unknown_block_type() -> None:
    p = _valid()
    p.blocks[0] = BlockSpec(id="src", type="warp_drive", params={})
    issues = validate_pipeline(p)
    assert any("warp_drive" in i.message and i.block_id == "src" for i in issues)


def test_missing_required_param() -> None:
    p = _valid()
    p.blocks[0] = BlockSpec(id="src", type="tone_source", params={})
    issues = validate_pipeline(p)
    assert any(i.field == "freq" and i.block_id == "src" for i in issues)


def test_unknown_param_rejected() -> None:
    p = _valid()
    p.blocks[0].params["warp"] = 9
    issues = validate_pipeline(p)
    assert any(i.field == "warp" for i in issues)


def test_dangling_connection_endpoint() -> None:
    p = _valid()
    p.connections.append(ConnectionSpec(src_block="ghost", dst_block="snk"))
    issues = validate_pipeline(p)
    assert any("ghost" in i.message for i in issues)


def test_dtype_mismatch() -> None:
    # quadrature_demod outputs float; file_sink expects complex
    p = PipelineSpec(
        sample_rate=1e6,
        blocks=[
            BlockSpec(id="src", type="tone_source", params={"freq": 1e3}),
            BlockSpec(id="qd", type="quadrature_demod", params={}),
            BlockSpec(id="snk", type="file_sink", params={"path": "o.cf32"}),
        ],
        connections=[
            ConnectionSpec(src_block="src", dst_block="qd"),
            ConnectionSpec(src_block="qd", dst_block="snk"),
        ],
    )
    issues = validate_pipeline(p)
    assert any("complex" in i.message and "float" in i.message for i in issues)


def test_unconnected_input_port() -> None:
    p = _valid()
    p.connections = p.connections[1:]  # head's input now dangles
    issues = validate_pipeline(p)
    assert any(i.block_id == "hd" for i in issues)


def test_duplicate_block_id() -> None:
    p = _valid()
    p.blocks.append(BlockSpec(id="src", type="tone_source", params={"freq": 1.0}))
    issues = validate_pipeline(p)
    assert any("duplicate" in i.message.lower() for i in issues)


def test_validation_error_formats_issues() -> None:
    p = _valid()
    p.blocks[0] = BlockSpec(id="src", type="tone_source", params={})
    err = PipelineValidationError(validate_pipeline(p))
    assert "src" in str(err) and "freq" in str(err)


def test_vocabulary_covers_spec_minimum() -> None:
    for t in (
        "tone_source",
        "audio_tone_source",
        "noise_source",
        "file_source",
        "head",
        "add",
        "multiply_const",
        "freq_shift",
        "freq_xlating_lowpass",
        "quadrature_demod",
        "rational_resampler_f",
        "rational_resampler_c",
        "fm_deemphasis",
        "nbfm_rx",
        "nbfm_tx",
        "file_sink",
        "wav_sink",
    ):
        assert t in VOCABULARY, t
