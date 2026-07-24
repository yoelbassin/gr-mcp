from marconi.core.stages import Stage
from marconi.phy.backends.base import BlockCensus
from marconi.phy.stages import stage_registry


def test_stage_defaults_gr_engine_no_window_seeding() -> None:
    assert Stage.engine == "gr"
    assert Stage.seeds_windows is False
    assert all(s.engine == "gr" for s in stage_registry().values())


def test_block_census_window_fields_default_none() -> None:
    row = BlockCensus(block="b0", kind="agc")
    assert row.windows_in is None and row.windows_out is None
