import pytest

from marconi.engine.coding.stages_symbols import MSliceStep
from marconi.engine.compile.compiler import (
    CompileError,
    compile_modem,
    compile_pipeline,
)
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.css.stages import CssDemapStep, DechirpStep
from marconi.engine.modulation.fsk.stages import FskStep, MfskSoftDemapStep
from marconi.engine.modulation.ofdm.stages import DqpskSoftDemapStep
from marconi.engine.modulation.psk.stages import (
    PskDemapStep,
    PskDemodStep,
    PskSoftDemapStep,
)
from marconi.engine.modulation.qam.stages import QamDemapStep, QamDemodStep
from marconi.engine.stages.acquisition import PreambleSyncStep
from marconi.engine.stages.conditioning import AgcStep
from marconi.engine.stages.registry import stage_registry, step_models
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import AgcMode, ItemType, PskOrder, QamOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.params import ParamValue
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)
_AGC = AgcStep(mode=AgcMode.POWER)
_DECHIRP_PARAMS: dict[str, ParamValue] = {"sf": 7, "oversample": 2, "zero_pad": 4}


def _compile(
    path: list[Step],
    direction: str = "rx",
    start: Descriptor = IQ,
    sample_rate: float = 4.0,
) -> GrPipeline:
    return compile_modem(
        Modem(symbol_rate=1.0, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=sample_rate,
        start=start,
        source_io={"path": "/dev/null"},
        sink_io={"path": "/dev/null"},
    )


def test_psk_demod_pins_order() -> None:
    out = stage_registry()["psk_demod"].out_descriptor(
        IQ, PskDemodStep(order=PskOrder.QPSK)
    )
    assert out.order == 4


def test_qam_demod_pins_order() -> None:
    out = stage_registry()["qam_demod"].out_descriptor(
        IQ, QamDemodStep(order=QamOrder.QAM16)
    )
    assert out.order == 16


def test_dechirp_pins_order() -> None:
    out = stage_registry()["dechirp"].out_descriptor(
        IQ, DechirpStep(sf=7, oversample=2, zero_pad=4)
    )
    assert out.order == 128


def test_level_preserving_default_propagates_order() -> None:
    pinned = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, order=4)
    step = PreambleSyncStep(preamble_i=[1.0, -1.0], preamble_q=[0.0, 0.0])
    out = stage_registry()["preamble_sync"].out_descriptor(pinned, step)
    assert out.order == 4


def test_level_change_resets_order() -> None:
    pinned = Descriptor(Level.SYMBOLS, ItemType.C, Carrier.SOFT, order=4)
    out = stage_registry()["psk_demap"].out_descriptor(
        pinned, PskDemapStep(order=PskOrder.QPSK)
    )
    assert out.order is None


def test_psk_order_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile(
            [_AGC, PskDemodStep(order=PskOrder.BPSK), PskDemapStep(order=PskOrder.PSK8)]
        )
    assert "order-8" in str(exc.value) and "order-2" in str(exc.value)


def test_psk_soft_order_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile(
            [
                _AGC,
                PskDemodStep(order=PskOrder.BPSK),
                PskSoftDemapStep(order=PskOrder.PSK8),
            ]
        )


def test_qam_order_mismatch_fails_rx() -> None:
    with pytest.raises(CompileError) as exc:
        _compile(
            [
                _AGC,
                QamDemodStep(order=QamOrder.QAM16),
                QamDemapStep(order=QamOrder.QAM64),
            ]
        )
    assert "order-64" in str(exc.value) and "order-16" in str(exc.value)


def test_qam_order_mismatch_fails_tx() -> None:
    with pytest.raises(CompileError):
        _compile(
            [QamDemodStep(order=QamOrder.QAM16), QamDemapStep(order=QamOrder.QAM64)],
            direction="tx",
        )


def test_css_sf_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError) as exc:
        _compile(
            [DechirpStep(sf=7, oversample=2, zero_pad=4), CssDemapStep(sf=9)],
            sample_rate=256.0,
        )
    assert "order-512" in str(exc.value) and "order-128" in str(exc.value)


def test_dqpsk_fixed_order_mismatch_fails_at_compile() -> None:
    with pytest.raises(CompileError):
        _compile(
            [
                _AGC,
                PskDemodStep(order=PskOrder.PSK8),
                DqpskSoftDemapStep(data_syms=4, n_carriers=4),
            ]
        )


def test_matched_orders_compile() -> None:
    _compile(
        [_AGC, PskDemodStep(order=PskOrder.QPSK), PskDemapStep(order=PskOrder.QPSK)]
    )


def test_unpinned_symbols_source_compiles() -> None:
    _compile([PskDemapStep(order=PskOrder.PSK8)], start=SYM_C)


def test_f_lane_stays_unpinned() -> None:
    _compile([FskStep(deviation=1.0), MfskSoftDemapStep(levels=[-1.0, 1.0])])


_QPSK_PRE: dict[str, list[float]] = {
    "preamble_i": [0.7071, -0.7071, 0.7071, -0.7071],
    "preamble_q": [0.7071, 0.7071, -0.7071, -0.7071],
}
_8PSK_PRE: dict[str, list[float]] = {
    "preamble_i": [1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071, 0.0, 0.7071],
    "preamble_q": [0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071],
}


def _psk_path(order: int, pre: dict[str, list[float]]) -> list[Step]:
    o = PskOrder(order)
    return [
        _AGC,
        PskDemodStep(order=o),
        PreambleSyncStep(preamble_i=pre["preamble_i"], preamble_q=pre["preamble_q"]),
        PskDemapStep(order=o),
    ]


def test_on_grid_preamble_compiles() -> None:
    _compile(_psk_path(4, _QPSK_PRE))


def test_rounded_8psk_preamble_compiles() -> None:
    _compile(_psk_path(8, _8PSK_PRE))


def test_order_propagates_through_preamble_sync() -> None:
    path = _psk_path(4, _QPSK_PRE)
    path[3] = PskDemapStep(order=PskOrder.PSK8)
    with pytest.raises(CompileError) as exc:
        _compile(path)
    assert "order-8" in str(exc.value) and "order-4" in str(exc.value)


def test_off_grid_preamble_fails() -> None:
    bad: dict[str, list[float]] = {
        "preamble_i": [0.7071, 0.5736],
        "preamble_q": [0.7071, 0.8192],
    }
    with pytest.raises(CompileError) as exc:
        _compile(_psk_path(4, bad))
    assert "phase grid" in str(exc.value)


def test_off_circle_preamble_fails() -> None:
    bad: dict[str, list[float]] = {
        "preamble_i": [0.7071, 1.5],
        "preamble_q": [0.7071, 1.5],
    }
    with pytest.raises(CompileError) as exc:
        _compile(_psk_path(4, bad))
    assert "radius" in str(exc.value)


def test_zero_preamble_point_fails() -> None:
    bad: dict[str, list[float]] = {
        "preamble_i": [1.0, 0.0],
        "preamble_q": [0.0, 0.0],
    }
    with pytest.raises(CompileError) as exc:
        _compile(_psk_path(2, bad))
    assert "zero" in str(exc.value)


def test_freeform_preamble_skips_grid_check() -> None:
    # a CAZAC-class training sequence is deliberately off the data
    # constellation; declaring it freeform keeps the order check alive on the
    # rest of the path instead of forcing an unpinned producer
    off: dict[str, list[float]] = {
        "preamble_i": [0.7071, 0.5736],
        "preamble_q": [0.7071, 0.8192],
    }
    path = _psk_path(4, off)
    idx = next(i for i, s in enumerate(path) if isinstance(s, PreambleSyncStep))
    path[idx] = PreambleSyncStep(
        preamble_i=off["preamble_i"],
        preamble_q=off["preamble_q"],
        constellation_preamble=False,
    )
    _compile(path)


def test_unpinned_boundary_skips_preamble_check() -> None:
    off_grid: dict[str, list[float]] = {
        "preamble_i": [0.7071, 0.5736],
        "preamble_q": [0.7071, 0.8192],
    }
    _compile(
        [
            PreambleSyncStep(
                preamble_i=off_grid["preamble_i"], preamble_q=off_grid["preamble_q"]
            ),
            PskDemapStep(order=PskOrder.PSK8),
        ],
        start=SYM_C,
    )


SYM_F = Descriptor(Level.SYMBOLS, ItemType.F, carrier=Carrier.SOFT)


def _m_slice() -> MSliceStep:
    return MSliceStep(thresholds=[0.0], levels=[0, 1])


def test_m_slice_pins_order() -> None:
    out = stage_registry()["m_slice"].out_descriptor(SYM_F, _m_slice())
    assert out.order == 2


def test_m_slice_declares_required_order() -> None:
    assert stage_registry()["m_slice"].required_input_order(_m_slice()) == 2


def test_mfsk_soft_demap_declares_required_order() -> None:
    step = MfskSoftDemapStep(levels=[-3.0, -1.0, 1.0, 3.0])
    assert stage_registry()["mfsk_soft_demap"].required_input_order(step) == 4


def test_m_slice_conflicting_upstream_pin_fails_at_compile() -> None:
    pinned = Descriptor(Level.SYMBOLS, ItemType.F, Carrier.SOFT, order=4)
    with pytest.raises(CompileError) as exc:
        compile_pipeline(
            Modem(symbol_rate=1.0, path=[_m_slice()]),
            stage_registry(),
            direction="rx",
            sample_rate=1.0,
            start=pinned,
            source_io={"path": "/dev/null"},
            sink_io={"path": "/dev/null"},
        )
    assert "order-2" in str(exc.value) and "order-4" in str(exc.value)


_ORDER_MATRIX: dict[str, tuple[dict[str, ParamValue], int]] = {
    "psk_demod": ({"order": 4}, 4),
    "psk_demap": ({"order": 4}, 4),
    "psk_soft_demap": ({"order": 4}, 4),
    "soft_demap": ({"scheme": "psk", "order": 4}, 4),
    "qam_demod": ({"order": 16}, 16),
    "qam_demap": ({"order": 16}, 16),
    "dechirp": (_DECHIRP_PARAMS, 128),
    "css_demap": ({"sf": 7}, 128),
    "dqpsk_soft_demap": ({"data_syms": 4, "n_carriers": 4}, 4),
    "mfsk_soft_demap": ({"levels": [-1.0, 1.0]}, 2),
}


def test_order_bearing_stages_are_in_the_contract() -> None:
    models = step_models()
    for name, stage in stage_registry().items():
        fields = set(stage.step_model.model_fields)
        if not {"order", "sf", "levels"} & fields:
            continue
        produces = (
            stage.to_level is Level.SYMBOLS and stage.from_level is not Level.SYMBOLS
        )
        consumes = (
            stage.from_level is Level.SYMBOLS and stage.to_level is not Level.SYMBOLS
        )
        if not (produces or consumes):
            continue
        assert name in _ORDER_MATRIX, (
            f"stage '{name}' carries an order/sf param at a SYMBOLS boundary; "
            "add it to _ORDER_MATRIX and give it a pin or requirement"
        )
        params, expected = _ORDER_MATRIX[name]
        step = models[name](**params)  # type: ignore[arg-type]
        if produces:
            out = stage.out_descriptor(Descriptor(stage.from_level, ItemType.C), step)
            assert out.order == expected, name
        if consumes:
            assert stage.required_input_order(step) == expected, name
