from marconi.engine.backends.base import BlockCensus
from marconi.engine.stages.base import CodingStage, Stage
from marconi.engine.stages.registry import stage_registry


def test_coding_flavor_is_the_type_not_a_settable_field() -> None:
    # The GR-vs-coding split is CodingStage membership, decided by the type the
    # emit body is written against — not a stringly-typed field the body could
    # silently contradict. A plain Stage is not a coding stage.
    assert not issubclass(Stage, CodingStage)
    assert Stage.seeds_windows is False


def test_registry_partitions_into_coding_stages_and_the_rest() -> None:
    reg = stage_registry()
    assert any(isinstance(s, CodingStage) for s in reg.values())


def test_block_census_window_fields_default_none() -> None:
    row = BlockCensus(block="b0", kind="agc")
    assert row.windows_in is None and row.windows_out is None
