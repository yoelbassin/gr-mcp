from marconi.engine.backends.base import BlockCensus
from marconi.engine.stage import Stage
from marconi.engine.stages import stage_registry


def test_stage_defaults_gr_engine_no_window_seeding() -> None:
    assert Stage.engine == "gr"
    assert Stage.seeds_windows is False


def test_registry_engine_is_gr_unless_a_stage_opts_into_coding() -> None:
    reg = stage_registry()
    assert any(s.engine == "coding" for s in reg.values())
    assert all(
        s.engine in ("gr", "coding") for s in reg.values()
    ), "a stage declared an engine other than the two the compiler partitions on"


def test_block_census_window_fields_default_none() -> None:
    row = BlockCensus(block="b0", kind="agc")
    assert row.windows_in is None and row.windows_out is None
