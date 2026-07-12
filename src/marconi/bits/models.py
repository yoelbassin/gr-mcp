from __future__ import annotations

from pydantic import BaseModel, model_validator


class ParseField(BaseModel):
    name: str
    bits: int
    signed: bool = False
    charset: str | None = None
    enum: dict[int, str] | None = None
    char_bits: int = 6
    rest: bool = False

    @model_validator(mode="after")
    def _rest_shape(self) -> "ParseField":
        if self.char_bits < 1:
            raise ValueError("char_bits must be >= 1")
        if self.rest and (self.charset is None or self.bits != 0):
            raise ValueError("a rest field needs charset set and bits == 0")
        return self


class CodecStep(BaseModel):
    conv: str
    # bare `list` arm: the `parse` stage's `fields` param is a list[dict].
    params: dict[str, float | int | str | bool | list] = {}


class CodecSpec(BaseModel):
    name: str = "codec"
    path: list[CodecStep]

    def params_for(self, conv: str) -> dict:
        for step in self.path:
            if step.conv == conv:
                return dict(step.params)
        return {}


class FrameResult(BaseModel):
    bit_offset: int
    payload_hex: str
    crc_ok: bool | None
    message: dict[str, int | str] | None = None


class DecodeResult(BaseModel):
    messages: list[dict[str, int | str]]
    frames: list[FrameResult]
    num_frames: int
    num_crc_ok: int
    num_unchecked: int = 0
