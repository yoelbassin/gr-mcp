from engine._fixtures import (
    FakeDemapStep,
    FakeDemodStep,
    FakeResamplerStep,
    fixture_registry,
)

from marconi.engine.compile.compiler import compile_modem
from marconi.engine.types.descriptor import Descriptor
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem
from marconi.engine.types.step import Step

IQ = Descriptor(Level.IQ, "c")


def _modem(*steps: Step, symbol_rate: float) -> Modem:
    return Modem(symbol_rate=symbol_rate, path=list(steps))


def _compile_tx(modem: Modem, sample_rate: float):
    return compile_modem(
        modem,
        fixture_registry(),
        direction="tx",
        sample_rate=sample_rate,
        start=IQ,
        source_io={"path": "in.bits"},
        sink_io={"path": "out.iq"},
    )


def test_tx_reverses_path_and_swaps_io() -> None:
    pipe = _compile_tx(_modem(FakeDemodStep(), FakeDemapStep(), symbol_rate=2.0), 8.0)
    assert [b.kind for b in pipe.blocks] == [
        "bits_file_source",
        "fake_map",
        "fake_mod",
        "iq_file_sink",
    ]
    assert pipe.blocks[0].params["path"] == "in.bits"
    assert pipe.blocks[-1].params["path"] == "out.iq"


def test_tx_chain_is_linearly_connected() -> None:
    pipe = _compile_tx(_modem(FakeDemodStep(), FakeDemapStep(), symbol_rate=2.0), 8.0)
    ids = pipe.block_ids
    edges = [(c.src_block, c.dst_block) for c in pipe.connections]
    assert edges == list(zip(ids, ids[1:]))


def test_tx_modulator_sees_iq_domain_sps() -> None:
    pipe = _compile_tx(_modem(FakeDemodStep(), FakeDemapStep(), symbol_rate=2.0), 8.0)
    mod = next(b for b in pipe.blocks if b.kind == "fake_mod")
    assert mod.params["sps"] == 8.0 / 2.0  # 4.0


def test_tx_resampler_round_trips_rate() -> None:
    # A TX-valid path must end (bits side) in BITS; ending in SYMBOLS would have no
    # _IO_BLOCKS source entry. rates = [6, 6*2/3=4, 4, 4]; the tx modulator (demod at
    # index 1) reads its input boundary rates[2]=4, so sps = 4/2 = 2.
    pipe = _compile_tx(
        _modem(
            FakeResamplerStep(interp=2, decim=3),
            FakeDemodStep(),
            FakeDemapStep(),
            symbol_rate=2.0,
        ),
        6.0,
    )
    mod = next(b for b in pipe.blocks if b.kind == "fake_mod")
    assert mod.params["sps"] == 4.0 / 2.0  # 2.0
    assert [b.kind for b in pipe.blocks] == [
        "bits_file_source",
        "fake_map",
        "fake_mod",
        "fake_resampler",
        "iq_file_sink",
    ]
