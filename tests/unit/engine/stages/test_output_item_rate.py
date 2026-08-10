"""Every stage that crosses IQ->SYMBOLS either models its item rate honestly
(rate_factor) or declares the true rate via output_item_rate — a demod that
decimates internally while reporting rate_factor 1.0 would otherwise make
every trace row at and after it overstate the sidecar's rate by sps."""

from unit.engine.types.test_modem_facade import VALID_STEPS

from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.levels import Level


def test_every_iq_to_symbols_stage_owns_its_item_rate() -> None:
    for name, stage in stage_registry().items():
        if stage.from_level is not Level.IQ or stage.to_level is not Level.SYMBOLS:
            continue
        step = VALID_STEPS[name]
        honest_model = stage.rate_factor(step) != 1.0
        declared = stage.output_item_rate(step, 8_000.0, 1_000.0) is not None
        assert honest_model or declared, (
            f"stage '{name}' crosses IQ->SYMBOLS with rate_factor 1.0 and no "
            "output_item_rate declaration — its trace rows would lie by sps"
        )
