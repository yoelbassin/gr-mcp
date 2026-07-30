import pytest

from marconi.engine.compile.compiler import CompileError, compile_modem
from marconi.engine.modulation.fsk.stages import FskStep, MskStep
from marconi.engine.modulation.ook.stages import OokEnvelopeStep
from marconi.engine.modulation.psk.stages import PskDemapStep, PskDemodStep
from marconi.engine.modulation.qam.stages import QamDemapStep, QamDemodStep
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.enums import AgcMode, PskOrder, QamOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, "c")
_AGC = AgcStep(mode=AgcMode.POWER)


def _compile(path: list[Step], sample_rate: float, direction: str = "rx"):
    return compile_modem(
        Modem(symbol_rate=1.0, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=IQ,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


_PSK: list[Step] = [
    _AGC,
    PskDemodStep(order=PskOrder.QPSK),
    PskDemapStep(order=PskOrder.QPSK),
]


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
            [
                _AGC,
                QamDemodStep(order=QamOrder.QAM16),
                QamDemapStep(order=QamOrder.QAM16),
            ],
            sample_rate=1.5,
        )


def test_fsk_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([FskStep(deviation=0.25), SliceStep()], sample_rate=1.5)


def test_msk_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([MskStep(), SliceStep()], sample_rate=1.5)


def test_ook_below_two_sps_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile([_AGC, OokEnvelopeStep(), SliceStep()], sample_rate=1.5)


def test_two_sps_compiles() -> None:
    _compile(_PSK, sample_rate=2.0)


def test_fractional_sps_above_two_compiles() -> None:
    _compile(_PSK, sample_rate=2.5)


def test_ppm_scale_slack_below_two_compiles() -> None:
    _compile(_PSK, sample_rate=1.999)


def test_tx_is_not_gated_on_rx_timing_floor() -> None:
    _compile(
        [PskDemodStep(order=PskOrder.QPSK), PskDemapStep(order=PskOrder.QPSK)],
        sample_rate=1.0,
        direction="tx",
    )
