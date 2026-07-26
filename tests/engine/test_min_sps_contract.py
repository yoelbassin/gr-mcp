import pytest

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import ModemSpec, ModemStep
from marconi.engine.types.params import ParamValue

IQ = Descriptor(Level.IQ, "c")
_AGC = ModemStep(conv="agc", params={"mode": "power"})


def _step(conv: str, **params: ParamValue) -> ModemStep:
    return ModemStep(conv=conv, params=params)


def _compile(path: list[ModemStep], sample_rate: float, direction: str = "rx"):
    return compile_modem(
        ModemSpec(symbol_rate=1.0, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


_PSK = [_AGC, _step("psk_demod", order=4), _step("psk_demap", order=4)]


def test_psk_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile(_PSK, sample_rate=1.5)
    assert "samples per symbol" in str(exc.value) and "1.5" in str(exc.value)


def test_psk_at_symbol_rate_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile(_PSK, sample_rate=1.0)


def test_qam_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile(
            [_AGC, _step("qam_demod", order=16), _step("qam_demap", order=16)],
            sample_rate=1.5,
        )


def test_fsk_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([_step("fsk", deviation=0.25), _step("slice")], sample_rate=1.5)


def test_msk_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([_step("msk"), _step("slice")], sample_rate=1.5)


def test_ook_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([_AGC, _step("ook_envelope"), _step("slice")], sample_rate=1.5)


def test_two_sps_compiles() -> None:
    _compile(_PSK, sample_rate=2.0)


def test_fractional_sps_above_two_compiles() -> None:
    _compile(_PSK, sample_rate=2.5)


def test_ppm_scale_slack_below_two_compiles() -> None:
    _compile(_PSK, sample_rate=1.999)


def test_tx_is_not_gated_on_rx_timing_floor() -> None:
    _compile(
        [_step("psk_demod", order=4), _step("psk_demap", order=4)],
        sample_rate=1.0,
        direction="tx",
    )
