from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.stages.framing_ops import (
    Descramble,
    DescrambleBits,
    FixedFrame,
    Segment,
)
from marconi.bits.validate import validate_codec
from marconi.core.stages import validate_params


def test_segment_fixed_frame_codec_validates():
    codec = CodecSpec(
        name="c",
        path=[
            CodecStep(conv="nibble_swap", params={}),
            CodecStep(conv="segment", params={"frame_body_len": 2040}),
            CodecStep(conv="fixed_frame", params={"payload_bits": 2040}),
            CodecStep(
                conv="parse",
                params={"bit_order": "msb", "fields": [{"name": "p", "bits": 2040}]},
            ),
        ],
    )
    issues = validate_codec(codec, registry())
    assert not issues, issues


def test_segment_rejects_zero_and_negative_frame_body_len():
    for bad in (0, -8):
        issues: list = []
        validate_params(
            "segment[0]", Segment.params_model, {"frame_body_len": bad}, issues
        )
        assert issues, f"frame_body_len={bad} must be rejected"


def test_fixed_frame_rejects_nonpositive_payload_bits():
    for bad in (0, -16):
        issues: list = []
        validate_params(
            "fixed_frame[0]", FixedFrame.params_model, {"payload_bits": bad}, issues
        )
        assert issues, f"payload_bits={bad} must be rejected"


def test_descramble_rejects_invalid_hex_sequence():
    for bad in ("zz", "abc"):  # non-hex chars; odd length
        for stage in (Descramble, DescrambleBits):
            issues: list = []
            validate_params(
                f"{stage.name}[0]", stage.params_model, {"sequence": bad}, issues
            )
            assert issues, f"sequence={bad!r} must be rejected for {stage.name}"


def test_descramble_bits_accepts_empty_sequence():
    issues: list = []
    validate_params(
        "descramble_bits[0]", DescrambleBits.params_model, {"sequence": ""}, issues
    )
    assert not issues, issues


def test_segment_zero_frame_body_len_fails_codec_validation():
    codec = CodecSpec(
        name="c",
        path=[
            CodecStep(conv="segment", params={"frame_body_len": 0}),
            CodecStep(conv="fixed_frame", params={"payload_bits": 8}),
            CodecStep(
                conv="parse",
                params={"bit_order": "msb", "fields": [{"name": "p", "bits": 8}]},
            ),
        ],
    )
    assert validate_codec(codec, registry()), "0-len segment must not validate"
