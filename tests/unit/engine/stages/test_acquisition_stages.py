import numpy as np
import pytest
from pydantic import ValidationError

from marconi.engine.compile.compile_context import CompileContext
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.compile.errors import CompileError
from marconi.engine.compile.ir import GrPipeline
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.modulation.psk.stages import PskDemodStep
from marconi.engine.stages.acquisition import PreambleSync, PreambleSyncStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Amplitude, Carrier, Descriptor
from marconi.engine.types.enums import ItemType, PskOrder
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

_PRE = np.exp(1j * np.linspace(-3.0, 3.0, 64))
_P = PreambleSyncStep(
    preamble_i=_PRE.real.tolist(),
    preamble_q=_PRE.imag.tolist(),
    pad_symbols=192,
    threshold=0.9,
)
SYM_C = Descriptor(Level.SYMBOLS, ItemType.C, carrier=Carrier.SOFT)


def test_registered_as_acquisition() -> None:
    assert stage_registry()["preamble_sync"].family == "acquisition"


def test_symbols_to_symbols_soft_unity_rate() -> None:
    s = PreambleSync()
    assert s.out_descriptor(SYM_C, _P) == SYM_C
    assert s.rate_factor(_P) == 1.0


def test_emit_rx_chains_corr_est_then_strip() -> None:
    b = CompileContext(SYM_C, rate=1.0, symbol_rate=1.0)
    PreambleSync().emit_rx(b, _P)
    blocks = b.build("t", 1.0).blocks
    assert [x.kind for x in blocks] == ["corr_est_cc", "sym_strip"]
    ce, strip = blocks
    assert ce.params["sps"] == 1 and ce.params["mark_delay"] == 0
    assert ce.params["threshold"] == 0.9
    pi, pq = ce.params["preamble_i"], ce.params["preamble_q"]
    assert isinstance(pi, list) and isinstance(pq, list)
    assert len(pi) == 64 and len(pq) == 64
    assert strip.params["n_pre"] == 64


def _compile_rx(*steps: Step) -> GrPipeline:
    return compile_modem(
        Modem(symbol_rate=1.0, path=list(steps)),
        stage_registry(),
        sample_rate=8.0,
        start=Descriptor(Level.IQ, ItemType.C, amplitude=Amplitude.PEAK_UNITY),
        source_io={"path": "in.iq"},
        sink_io={"path": "out.cf32"},
    )


def test_float_symbols_into_preamble_sync_rejected_at_compile() -> None:
    """corr_est_cc and sym_strip are complex-only; fsk emits real-float symbols.
    The seam check must reject fsk->preamble_sync at compile, not let it die on
    an itemsize mismatch in the backend."""
    with pytest.raises(CompileError, match="item_type"):
        _compile_rx(FskStep(deviation=0.5), _P)


def test_soft_complex_symbols_into_preamble_sync_compiles() -> None:
    """The valid composition still compiles: psk_demod emits complex soft
    symbols, exactly what preamble_sync accepts."""
    qpsk_pre = PreambleSyncStep(
        preamble_i=[0.7071, -0.7071, 0.7071, -0.7071],
        preamble_q=[0.7071, 0.7071, -0.7071, -0.7071],
        pad_symbols=192,
        threshold=0.9,
    )
    pipe = _compile_rx(PskDemodStep(order=PskOrder(4)), qpsk_pre)
    assert any(b.kind == "corr_est_cc" for b in pipe.blocks)


def test_param_validation() -> None:
    with pytest.raises(ValidationError):
        PreambleSyncStep(preamble_i=[1.0, 0.0], preamble_q=[0.0])  # unequal length
    with pytest.raises(ValidationError):
        PreambleSyncStep(preamble_i=[], preamble_q=[])  # empty preamble
