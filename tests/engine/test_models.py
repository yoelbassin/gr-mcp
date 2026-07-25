from pathlib import Path

import pytest
from pydantic import ValidationError

from marconi.engine.models import Bitstream, CaptureRef, SoftBitstream


def test_capture_duration() -> None:
    c = CaptureRef(
        path=Path("x.iq"), center_freq=915e6, sample_rate=1e6, num_samples=2_000_000
    )
    assert c.duration == 2.0


def test_capture_supports_multiple_input_dtypes() -> None:
    for dt in ("cf32_le", "ci16_le", "cf64_le"):
        CaptureRef(
            path=Path("x"), center_freq=0.0, sample_rate=1e6, num_samples=0, datatype=dt
        )


def test_capture_rejects_nonpositive_sample_rate() -> None:
    with pytest.raises(ValidationError):
        CaptureRef(path=Path("x"), center_freq=0.0, sample_rate=0.0, num_samples=0)


def test_soft_bitstream_is_a_sibling_not_a_flag() -> None:
    s = SoftBitstream(path=Path("s.f32"), num_bits=80)
    assert isinstance(s, SoftBitstream) and not isinstance(s, Bitstream)
