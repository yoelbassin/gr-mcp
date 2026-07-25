import pytest

from marconi.engine.compiler import CompileError, compile_modem, compile_pipeline
from marconi.engine.descriptor import Carrier, Descriptor
from marconi.engine.levels import Level
from marconi.engine.models import ModemSpec, ModemStep
from marconi.engine.params import ParamValue
from marconi.engine.stages import stage_registry

IQ = Descriptor(Level.IQ, "c")
SOFT_BITS = Descriptor(Level.BITS, "b", carrier=Carrier.HARD)


def _spec(*convs: str) -> ModemSpec:
    # exact POCSAG GR prefix (tests/bits/test_pocsag_offair.py:89-101) so the
    # rate/amplitude contract checks see a known-good chain
    params: dict[str, dict[str, ParamValue]] = {
        "channelize": {"decim": 4, "bandwidth_hz": 14000.0, "center_hz": -10250.0},
        "agc": {"window_symbols": 64.0},
        "fsk": {"deviation": 4500.0},
        "sync_word": {"sync": "7e"},
    }
    return ModemSpec(
        symbol_rate=1200.0,
        path=[ModemStep(conv=c, params=params.get(c, {})) for c in convs],
    )


def test_mixed_path_partitions_at_first_coding_stage() -> None:
    cp = compile_pipeline(
        _spec("channelize", "agc", "fsk", "slice", "sync_word"),
        stage_registry(),
        direction="rx",
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
        direction="rx",
        sample_rate=1.0,
        start=SOFT_BITS,
        source_io={},
        sink_io={},
    )
    assert cp.gr is None and cp.coding is not None


def test_gr_after_coding_is_a_compile_error() -> None:
    with pytest.raises(CompileError, match="re-enter"):
        compile_pipeline(
            _spec("channelize", "agc", "fsk", "slice", "sync_word", "agc"),
            stage_registry(),
            direction="rx",
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )


def test_tx_with_coding_stage_is_a_compile_error() -> None:
    with pytest.raises(CompileError, match="coded tx"):
        compile_pipeline(
            _spec("sync_word"),
            stage_registry(),
            direction="tx",
            sample_rate=1.0,
            start=SOFT_BITS,
            source_io={},
            sink_io={},
        )


def test_unknown_stage_aggregates_with_other_spec_issues() -> None:
    spec = ModemSpec(
        symbol_rate=1200.0,
        path=[
            ModemStep(conv="totally_bogus_stage", params={}),
            ModemStep(conv="fsk", params={}),
        ],
    )
    with pytest.raises(CompileError) as e:
        compile_pipeline(
            spec,
            stage_registry(),
            direction="rx",
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )
    msg = str(e.value)
    assert "totally_bogus_stage[0]" in msg and "unknown converter" in msg
    assert "fsk[1].deviation" in msg


def test_empty_rx_path_builds_passthrough_gr_pipeline() -> None:
    pipe = compile_modem(
        ModemSpec(symbol_rate=1200.0, path=[]),
        stage_registry(),
        direction="rx",
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
            direction="rx",
            sample_rate=128000.0,
            start=IQ,
            source_io={},
            sink_io={},
        )
