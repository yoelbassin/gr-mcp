import pytest

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.modulation.coding.stages import (
    DeinterleaveStep,
    DepunctureStep,
    FecStep,
    HardenStep,
    SyncAlignStep,
)
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

BITS_F = Descriptor(Level.BITS, ItemType.F, carrier=Carrier.SOFT)
_CODE = "10110111"
_POLYS2: list[int] = [0o133, 0o171]
_POLYS3: list[int] = [0o133, 0o171, 0o165]


def _sync(frame_len: int) -> SyncAlignStep:
    return SyncAlignStep(access_code=_CODE, frame_len=frame_len)


def _fec(frame_bits: int, rate_inv: int, tail: int = 0) -> FecStep:
    polys = _POLYS2 if rate_inv == 2 else _POLYS3
    return FecStep(
        scheme="cc",
        rate_inv=rate_inv,
        polys=polys,
        frame_bits=frame_bits,
        tail=tail,
    )


def _compile(path: list[Step]) -> None:
    compile_modem(
        Modem(symbol_rate=1.0, path=path),
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=BITS_F,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_sync_align_pins_frame_len() -> None:
    out = stage_registry()["sync_align"].out_descriptor(
        BITS_F, SyncAlignStep(access_code=_CODE, frame_len=100)
    )
    assert out.frame_len == 100


def test_frame_fec_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([_sync(100), _fec(frame_bits=80, rate_inv=2)])
    assert "160" in str(exc.value) and "100" in str(exc.value)


def test_frame_fec_matched_compiles() -> None:
    _compile([_sync(160), _fec(frame_bits=80, rate_inv=2)])


def test_frame_fec_with_tail_matched_compiles() -> None:
    _compile([_sync(172), _fec(frame_bits=80, rate_inv=2, tail=6)])


def test_deinterleave_nondivisor_perm_fails() -> None:
    perm = list(reversed(range(8)))
    with pytest.raises(CompileError) as exc:
        _compile([_sync(100), DeinterleaveStep(perm=perm)])
    assert "8" in str(exc.value) and "100" in str(exc.value)


def test_deinterleave_divisor_perm_preserves_frame() -> None:
    perm = list(reversed(range(16)))
    _compile([_sync(160), DeinterleaveStep(perm=perm), _fec(80, 2)])


def test_depuncture_scales_frame() -> None:
    _compile([_sync(90), DepunctureStep(keep_mask=[1, 1, 0]), _fec(45, 3)])


def test_depuncture_wrong_downstream_fec_fails() -> None:
    with pytest.raises(CompileError):
        _compile([_sync(90), DepunctureStep(keep_mask=[1, 1, 0]), _fec(44, 3)])


def test_depuncture_nondivisible_kept_fails() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([_sync(91), DepunctureStep(keep_mask=[1, 1, 0])])
    assert "91" in str(exc.value)


def test_harden_preserves_frame_len() -> None:
    framed = Descriptor(Level.BITS, ItemType.F, Carrier.SOFT, frame_len=100)
    out = stage_registry()["harden"].out_descriptor(framed, HardenStep())
    assert out.frame_len == 100


def test_fec_pins_output_frame() -> None:
    out = stage_registry()["fec"].out_descriptor(
        BITS_F,
        FecStep(scheme="cc", rate_inv=2, polys=_POLYS2, frame_bits=80),
    )
    assert out.frame_len == 80


def test_unframed_stream_skips_fec_geometry() -> None:
    _compile([_fec(frame_bits=80, rate_inv=2)])


def test_frame_len_below_one_rejected() -> None:
    with pytest.raises(ValueError):
        Descriptor(Level.BITS, ItemType.F, Carrier.SOFT, frame_len=0)
