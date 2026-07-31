from __future__ import annotations

from marconi.engine.types.enums import (
    AgcMode,
    DcMode,
    DecodeMode,
    EmitMode,
    PskOrder,
    QamOrder,
)


def test_psk_order_members() -> None:
    assert {m.value for m in PskOrder} == {2, 4, 8}
    assert PskOrder(4) is PskOrder.QPSK


def test_qam_order_members() -> None:
    assert {m.value for m in QamOrder} == {16, 64}
    assert QamOrder(16) is QamOrder.QAM16


def test_agc_mode_members() -> None:
    assert {m.value for m in AgcMode} == {"feedforward", "feedback", "power"}
    assert AgcMode("power") is AgcMode.POWER


def test_decode_mode_members() -> None:
    assert {m.value for m in DecodeMode} == {"exact", "nearest"}
    assert DecodeMode("nearest") is DecodeMode.NEAREST


def test_emit_mode_members() -> None:
    assert {m.value for m in EmitMode} == {"data", "codeword"}
    assert EmitMode("codeword") is EmitMode.CODEWORD


def test_dc_mode_members() -> None:
    assert {m.value for m in DcMode} == {"median", "mean", "none"}
    assert DcMode("mean") is DcMode.MEAN
