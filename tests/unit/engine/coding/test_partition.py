from collections.abc import Callable

import pytest
from pydantic import ValidationError

from marconi.engine.coding.stages_bits import SyncWordStep
from marconi.engine.compile.compiler import (
    compile_modem,
    compile_pipeline,
)
from marconi.engine.compile.errors import CompileError
from marconi.engine.modulation.coding.stages import HardenStep
from marconi.engine.modulation.fsk.stages import FskStep
from marconi.engine.stages.conditioning import AgcStep, ChannelizeStep
from marconi.engine.stages.general import SliceStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, ItemType.C)
SOFT_BITS = Descriptor(Level.BITS, ItemType.B, carrier=Carrier.HARD)

# the e2e POCSAG decode's exact GR prefix (tests/e2e/pocsag) so the
# rate/amplitude contract checks see a known-good chain
_BUILDERS: dict[str, Callable[[], Step]] = {
    "channelize": lambda: ChannelizeStep(
        decim=4, bandwidth_hz=14000.0, center_hz=-10250.0
    ),
    "agc": lambda: AgcStep(window_symbols=64.0),
    "fsk": lambda: FskStep(deviation=4500.0),
    "slice": SliceStep,
    "sync_word": lambda: SyncWordStep(sync="7e"),
    # a BITS->BITS GR stage, so a coding->GR re-entry can be exercised without
    # also tripping the path-level rung check that now reports first
    "harden": HardenStep,
}


def _spec(*convs: str) -> Modem:
    return Modem(symbol_rate=1200.0, path=[_BUILDERS[c]() for c in convs])


def test_mixed_path_partitions_at_first_coding_stage() -> None:
    cp = compile_pipeline(
        _spec("channelize", "agc", "fsk", "slice", "sync_word"),
        stage_registry(),
        sample_rate=128000.0,
        start=IQ,
        source_io={"path": "in.cf32"},
        sink_io={"path": "seam.u8"},
    )
    assert cp.gr is not None and cp.coding is not None
    assert cp.boundary.level is Level.BITS
    assert [s.kind for s in cp.coding.steps] == ["sync_word"]
    assert not any(b.kind == "sync_word" for b in cp.gr.blocks)


def test_pure_coding_path_has_no_gr_segment() -> None:
    cp = compile_pipeline(
        _spec("sync_word"),
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_BITS,
        source_io={},
        sink_io={},
    )
    assert cp.gr is None and cp.coding is not None


def test_gr_after_coding_is_a_compile_error() -> None:
    with pytest.raises(CompileError, match="re-enter"):
        compile_pipeline(
            _spec("channelize", "agc", "fsk", "slice", "sync_word", "harden"),
            stage_registry(),
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )


def test_path_validation_reports_before_partitioning() -> None:
    """_validate is the AGGREGATING reporter — it names the offending stage and
    can list several issues; _split_index raises one bare string. Partitioning
    ran first, so a spec that was wrong in both ways only ever surfaced the
    coarser message."""
    with pytest.raises(CompileError) as exc:
        compile_pipeline(
            _spec("channelize", "agc", "fsk", "slice", "sync_word", "agc"),
            stage_registry(),
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )
    assert "agc[5]" in str(exc.value), str(exc.value)


def test_unknown_stage_is_a_compile_error() -> None:
    spec = Modem(
        symbol_rate=1200.0,
        path=[Step(conv="totally_bogus_stage"), FskStep(deviation=4500.0)],
    )
    with pytest.raises(CompileError) as e:
        compile_pipeline(
            spec,
            stage_registry(),
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )
    msg = str(e.value)
    assert "totally_bogus_stage[0]" in msg and "unknown converter" in msg


def test_missing_required_field_raises_at_construction() -> None:
    # Step construction validates eagerly now (the compiler never sees an
    # incomplete step): fsk requires `deviation`, so building one without it
    # raises while constructing the step, naming the field, not a later
    # aggregated CompileError (mirrors test_validate_before_compile.py's
    # test_missing_param_names_the_field).
    with pytest.raises(ValidationError) as e:
        FskStep()  # type: ignore[call-arg]
    assert "deviation" in str(e.value)


def test_empty_rx_path_builds_passthrough_gr_pipeline() -> None:
    pipe = compile_modem(
        Modem(symbol_rate=1200.0, path=[]),
        stage_registry(),
        sample_rate=128000.0,
        start=IQ,
        source_io={"path": "in.cf32"},
        sink_io={"path": "out.cf32"},
    )
    assert [b.kind for b in pipe.blocks] == ["iq_file_source", "iq_file_sink"]


def test_compile_modem_rejects_coding_stages() -> None:
    with pytest.raises(CompileError):
        compile_modem(
            _spec("channelize", "agc", "fsk", "slice", "sync_word"),
            stage_registry(),
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )
