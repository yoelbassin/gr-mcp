from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.registry import registry
from marconi.bits.validate import validate_codec


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
