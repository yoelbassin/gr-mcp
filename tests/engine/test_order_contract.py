import pytest

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep
from marconi.engine.types.params import ParamValue

IQ = Descriptor(Level.IQ, "c")
SYM_C = Descriptor(Level.SYMBOLS, "c", carrier=Carrier.SOFT)
_AGC = ModemStep(conv="agc", params={"mode": "power"})
_DECHIRP_PARAMS: dict[str, ParamValue] = {"sf": 7, "oversample": 2, "zero_pad": 4}


def _step(conv: str, **params: ParamValue) -> ModemStep:
    return ModemStep(conv=conv, params=params)


def _compile(
    path: list[ModemStep],
    direction: str = "rx",
    start: Descriptor = IQ,
    sample_rate: float = 4.0,
):
    return compile_modem(
        ModemSpec(symbol_rate=1.0, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=start,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_psk_demod_pins_order() -> None:
    out = stage_registry()["psk_demod"].out_descriptor(IQ, {"order": 4})
    assert out.order == 4


def test_qam_demod_pins_order() -> None:
    out = stage_registry()["qam_demod"].out_descriptor(IQ, {"order": 16})
    assert out.order == 16


def test_dechirp_pins_order() -> None:
    out = stage_registry()["dechirp"].out_descriptor(IQ, _DECHIRP_PARAMS)
    assert out.order == 128


def test_level_preserving_default_propagates_order() -> None:
    pinned = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT, order=4)
    pre = {"preamble_i": [1.0, -1.0], "preamble_q": [0.0, 0.0]}
    out = stage_registry()["preamble_sync"].out_descriptor(pinned, pre)
    assert out.order == 4


def test_level_change_resets_order() -> None:
    pinned = Descriptor(Level.SYMBOLS, "c", Carrier.SOFT, order=4)
    out = stage_registry()["psk_demap"].out_descriptor(pinned, {"order": 4})
    assert out.order is None


def test_psk_order_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([_AGC, _step("psk_demod", order=2), _step("psk_demap", order=8)])
    assert "order-8" in str(exc.value) and "order-2" in str(exc.value)


def test_psk_soft_order_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([_AGC, _step("psk_demod", order=2), _step("psk_soft_demap", order=8)])


def test_qam_order_mismatch_fails_rx() -> None:
    with pytest.raises(CompileError) as exc:
        _compile([_AGC, _step("qam_demod", order=16), _step("qam_demap", order=64)])
    assert "order-64" in str(exc.value) and "order-16" in str(exc.value)


def test_qam_order_mismatch_fails_tx() -> None:
    with pytest.raises(CompileError):
        _compile(
            [_step("qam_demod", order=16), _step("qam_demap", order=64)],
            direction="tx",
        )


def test_css_sf_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile(
            [_step("dechirp", **_DECHIRP_PARAMS), _step("css_demap", sf=9)],
            sample_rate=256.0,
        )
    assert "order-512" in str(exc.value) and "order-128" in str(exc.value)


def test_dqpsk_fixed_order_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile(
            [
                _AGC,
                _step("psk_demod", order=8),
                _step("dqpsk_soft_demap", data_syms=4, n_carriers=4),
            ]
        )


def test_matched_orders_compile() -> None:
    _compile([_AGC, _step("psk_demod", order=4), _step("psk_demap", order=4)])


def test_unpinned_symbols_source_compiles() -> None:
    _compile([_step("psk_demap", order=8)], start=SYM_C)


def test_f_lane_stays_unpinned() -> None:
    _compile(
        [_step("fsk", deviation=1.0), _step("mfsk_soft_demap", levels=[-1.0, 1.0])]
    )
