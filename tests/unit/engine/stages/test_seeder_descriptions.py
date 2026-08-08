"""The blind ADS-B dogfood composed sync_word -> segment -> codebook and got a
silently wrong decode: segment is a SEEDER that re-tiles from 0, discarding
sync_word's windows. These descriptions are the discoverability fix."""

from marconi.engine.stages.registry import stage_registry


def test_seeder_stages_state_their_window_semantics() -> None:
    reg = stage_registry()
    for name, must_contain in {
        "sync_word": "MARK",
        "sync_align": "GATE",
        "segment": "DISCARD",
        "mark_frame": "window",
    }.items():
        desc = reg[name].description
        assert must_contain.lower() in desc.lower(), (name, desc)
        assert len(desc) > 40, name
