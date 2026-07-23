import numpy as np

from marconi.bits import framing
from marconi.bits.carriers import RxCarrier
from marconi.bits.compiler import compile_codec
from marconi.bits.models import CodecSpec, CodecStep
from marconi.bits.program import run_program
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


def test_body_slicer_after_self_slicing_seeder_is_rejected():
    """hdlc_deframe emits complete frames whose cursors index its destuffed
    payload space, not the raw bit stream — a body slicer after it reads
    noise as frame content. Must be rejected at validation, not decoded."""
    for slicer in (
        CodecStep(conv="fixed_frame", params={"payload_bits": 16}),
        CodecStep(conv="length_frame", params={"length_bits": 8, "base_bytes": 0}),
    ):
        codec = CodecSpec(
            name="c", path=[CodecStep(conv="hdlc_deframe", params={}), slicer]
        )
        issues = validate_codec(codec, registry())
        assert any("complete frames" in i.message for i in issues), (
            slicer.conv,
            issues,
        )


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


def test_seeded_transforms_validate_after_a_seeder():
    codec = CodecSpec(
        name="c",
        path=[
            CodecStep(conv="sync_word", params={"sync": "2d"}),
            CodecStep(conv="permute", params={"perm": [0, 1, 2]}),
            CodecStep(
                conv="block_code",
                params={
                    "code_bits": 3,
                    "data_bits": 2,
                    "parity_masks": [0b11],
                    "correct": False,
                    "emit": "data",
                },
            ),
            CodecStep(conv="fixed_frame", params={"payload_bits": 2}),
        ],
    )
    issues = validate_codec(codec, registry())
    assert issues == [], issues


def test_seeded_transforms_run_end_to_end():
    sync = framing.bytes_to_bits(bytes.fromhex("2d"))
    codeword_a = np.array([1, 0, 1], np.uint8)  # data=10, parity 1^0 matches 0b11
    codeword_b = np.array([0, 1, 1], np.uint8)  # data=01, parity 0^1 matches 0b11
    bits = np.concatenate([sync, codeword_a, sync, codeword_b]).astype(np.uint8)

    spec = CodecSpec(
        name="c",
        path=[
            CodecStep(conv="sync_word", params={"sync": "2d"}),
            CodecStep(conv="permute", params={"perm": [0, 1, 2]}),
            CodecStep(
                conv="block_code",
                params={
                    "code_bits": 3,
                    "data_bits": 2,
                    "parity_masks": [0b11],
                    "correct": False,
                    "emit": "data",
                },
            ),
            CodecStep(conv="fixed_frame", params={"payload_bits": 2}),
        ],
    )
    assert validate_codec(spec, registry()) == []
    program = compile_codec(spec, registry(), "rx")
    out = run_program(program, RxCarrier(bits=bits))
    assert [f.payload for f in out.frames] == [b"\x80", b"@"]


def test_malformed_parse_field_is_rejected():
    # A parse field missing its required 'bits' must produce a validation issue.
    codec = CodecSpec(
        name="c",
        path=[
            CodecStep(conv="fixed_frame", params={"payload_bits": 8}),
            CodecStep(conv="parse", params={"fields": [{"name": "p"}]}),
        ],
    )
    issues = validate_codec(codec, registry())
    assert any("fields" in (i.field or "") for i in issues), issues
