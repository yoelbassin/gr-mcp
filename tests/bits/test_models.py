from __future__ import annotations

from marconi.bits.models import CodecSpec, CodecStep


def test_codec_params_for() -> None:
    spec = CodecSpec(
        name="x", path=[CodecStep(conv="crc", params={"poly": 0x1021, "bits": 16})]
    )
    assert spec.params_for("crc") == {"poly": 0x1021, "bits": 16}
    assert spec.params_for("absent") == {}
