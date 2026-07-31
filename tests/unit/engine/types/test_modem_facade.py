"""Regression guard for the Modem string facade: serialize/parse must
round-trip every registered stage. `to_spec`/`from_spec` are the MCP/agent
seam (a Modem crosses the wire as JSON, not as live Step objects), so a stage
whose validator only tolerates direct construction -- and rejects its own
re-dumped defaults -- silently breaks every agent that saves and reloads a
modem built with that stage."""

from __future__ import annotations

import pytest
from helpers import _lattice

from marconi.engine.coding.stages_bits import (  # noqa: E402
    BlockCodeStep,
    CodebookStep,
    DescrambleStep,
    DifferentialStep,
    MarkFrameStep,
    NibbleSwapStep,
    PermuteStep,
    RealignStep,
    RsCodeStep,
    SegmentStep,
    SyncWordStep,
)
from marconi.engine.coding.stages_symbols import (  # noqa: E402
    MSliceStep,
    NormalizeStep,
    SymbolMapStep,
    SyncSymbolsStep,
)
from marconi.engine.modulation.coding.stages import (  # noqa: E402
    DeinterleaveStep,
    DepunctureStep,
    FecStep,
    HardenStep,
    SyncAlignStep,
)
from marconi.engine.modulation.css.stages import (  # noqa: E402
    ChirpSyncStep,
    CssDemapStep,
    DechirpStep,
)
from marconi.engine.modulation.fsk.stages import (  # noqa: E402
    FskStep,
    MfskSoftDemapStep,
    MskStep,
)
from marconi.engine.modulation.ofdm.stages import (  # noqa: E402
    CellSelectStep,
    DqpskSoftDemapStep,
    OfdmCoherentSyncStep,
    OfdmDemodStep,
    OfdmFrameSyncProbeStep,
    SoftDemapStep,
)
from marconi.engine.modulation.ook.stages import OokEnvelopeStep  # noqa: E402
from marconi.engine.modulation.psk.stages import (  # noqa: E402
    DifferentialDemodStep,
    PskDemapStep,
    PskDemodStep,
    PskSoftDemapStep,
    SampleSymbolsStep,
    SymbolSyncStep,
)
from marconi.engine.modulation.qam.stages import (  # noqa: E402
    QamDemapStep,
    QamDemodStep,
)
from marconi.engine.stages.acquisition import FllStep, PreambleSyncStep  # noqa: E402
from marconi.engine.stages.conditioning import (  # noqa: E402
    AgcStep,
    AmStep,
    AnalyticStep,
    ChannelizeStep,
    ClockCorrectStep,
    EqualizerStep,
    FmDemodStep,
    InvertStep,
    ResampleStep,
    SquelchStep,
)
from marconi.engine.stages.general import SliceStep  # noqa: E402
from marconi.engine.stages.probes import BurstProbeStep  # noqa: E402
from marconi.engine.stages.registry import stage_registry, step_models  # noqa: E402
from marconi.engine.types.enums import AgcMode, PskOrder, QamOrder  # noqa: E402
from marconi.engine.types.models import Modem  # noqa: E402
from marconi.engine.types.step import Step, StepSpecError  # noqa: E402

# OFDM bin_perm has no pydantic-level shape constraint (the DAG-level geometry
# check lives at compile time), but a plausible carrier-select permutation
# mirrors tests/engine/test_amplitude_invariance.py's _OFDM_BIN_PERM fixture.
_OFDM_FFT = 16
_OFDM_ACTIVE = [1, 2, 14, 15]
_OFDM_BIN_PERM = _OFDM_ACTIVE + [b for b in range(_OFDM_FFT) if b not in _OFDM_ACTIVE]

# One valid instance per registered `conv`, keyed by conv. Every stage in
# stage_registry() must have an entry (test_valid_steps_cover_every_stage
# enforces this), so a new stage forces a new entry here instead of a silent
# gap in round-trip coverage.
VALID_STEPS: dict[str, Step] = {
    "agc": AgcStep(),
    "am": AmStep(),
    "analytic": AnalyticStep(),
    "block_code": BlockCodeStep(
        code_bits=8, data_bits=4, parity_masks=[1, 2, 4, 8], correct_single=False
    ),
    "burst_probe": BurstProbeStep(),
    "cell_select": CellSelectStep(select_perm=list(range(12)), keep=3),
    "channelize": ChannelizeStep(decim=2, bandwidth_hz=1000.0),
    "chirp_sync": ChirpSyncStep(
        sf=7,
        oversample=2,
        zero_pad=4,
        preamble_len=8,
        sfd_symbols=2.25,
        sync_symbols=2,
    ),
    "clock_correct": ClockCorrectStep(ppm=5.0),
    "codebook": CodebookStep(code_bits=2, data_bits=1, table=[1, 2]),
    "css_demap": CssDemapStep(sf=7),
    "dechirp": DechirpStep(sf=7, oversample=2, zero_pad=4),
    "deinterleave": DeinterleaveStep(perm=[0, 1, 2, 3]),
    "depuncture": DepunctureStep(keep_mask=[1, 0, 1, 0]),
    "descramble": DescrambleStep(sequence="ff"),
    "differential": DifferentialStep(),
    "differential_demod": DifferentialDemodStep(),
    "dqpsk_soft_demap": DqpskSoftDemapStep(data_syms=4, n_carriers=4),
    "equalizer": EqualizerStep(),
    "fec": FecStep(scheme="cc", rate_inv=2, polys=[0o133, 0o171], frame_bits=80),
    "fll": FllStep(),
    "fm_demod": FmDemodStep(deviation=1000.0),
    "fsk": FskStep(deviation=1000.0),
    "harden": HardenStep(),
    "invert": InvertStep(),
    "m_slice": MSliceStep(thresholds=[0.0], levels=[0, 1]),
    "mark_frame": MarkFrameStep(),
    "mfsk_soft_demap": MfskSoftDemapStep(levels=[-1.0, 1.0]),
    "msk": MskStep(),
    "nibble_swap": NibbleSwapStep(),
    "normalize": NormalizeStep(span_symbols=16),
    "ofdm_coherent_sync": OfdmCoherentSyncStep(**_lattice.sync_params()),
    "ofdm_demod": OfdmDemodStep(
        fft_len=16,
        cp_len=4,
        sym_len=20,
        null_len=24,
        frame_len=100,
        n_frame_syms=4,
        data_syms=4,
        n_carriers=4,
        bin_perm=_OFDM_BIN_PERM,
    ),
    "ofdm_frame_sync_probe": OfdmFrameSyncProbeStep(
        fft_len=16, cp_len=4, sym_len=20, null_len=24, frame_len=100, data_syms=4
    ),
    "ook_envelope": OokEnvelopeStep(),
    "permute": PermuteStep(perm=[0, 1, 2, 3]),
    "preamble_sync": PreambleSyncStep(preamble_i=[1.0, -1.0], preamble_q=[0.0, 0.0]),
    "psk_demap": PskDemapStep(order=PskOrder.QPSK),
    "psk_demod": PskDemodStep(order=PskOrder.QPSK),
    "psk_soft_demap": PskSoftDemapStep(order=PskOrder.QPSK),
    "qam_demap": QamDemapStep(order=QamOrder.QAM16),
    "qam_demod": QamDemodStep(order=QamOrder.QAM16),
    "realign": RealignStep(bit_offset=0),
    "resample": ResampleStep(interpolation=2, decimation=3),
    "rs_code": RsCodeStep(symbol_bits=4, n=15, k=9, prim_poly=0x13, fcr=1),
    "sample_symbols": SampleSymbolsStep(),
    "segment": SegmentStep(frame_body_len=8),
    "slice": SliceStep(),
    "soft_demap": SoftDemapStep(scheme="psk", order=4),
    "squelch": SquelchStep(threshold_db=-20.0),
    "symbol_map": SymbolMapStep(code_bits=2, data_bits=2, table=[0, 1, 2, 3]),
    "symbol_sync": SymbolSyncStep(sps=4),
    "sync_align": SyncAlignStep(access_code="10110111", frame_len=80),
    "sync_symbols": SyncSymbolsStep(pattern=[0, 1, 0, 1]),
    "sync_word": SyncWordStep(sync="ab12"),
}


def test_valid_steps_cover_every_registered_stage() -> None:
    assert set(VALID_STEPS) == set(stage_registry())


@pytest.mark.parametrize("conv", sorted(VALID_STEPS))
def test_round_trip_every_registered_stage(conv: str) -> None:
    step = VALID_STEPS[conv]
    modem = Modem(symbol_rate=1e5, path=[step])
    restored = Modem.from_spec(modem.to_spec(), step_models())
    assert restored == modem
    assert isinstance(restored.path[0], type(step))


def test_agc_default_round_trips() -> None:
    modem = Modem(symbol_rate=1e5, path=[AgcStep()])
    restored = Modem.from_spec(modem.to_spec(), step_models())
    assert restored == modem


@pytest.mark.parametrize(
    "step",
    [
        AgcStep(mode=AgcMode.FEEDFORWARD),
        AgcStep(mode=AgcMode.FEEDBACK),
        AgcStep(mode=AgcMode.POWER),
    ],
    ids=["feedforward", "feedback", "power"],
)
def test_agc_each_mode_round_trips(step: AgcStep) -> None:
    modem = Modem(symbol_rate=1e5, path=[step])
    restored = Modem.from_spec(modem.to_spec(), step_models())
    assert restored == modem


def test_to_spec_includes_subclass_fields() -> None:
    modem = Modem(symbol_rate=1e5, path=[PskDemodStep(order=PskOrder(4))])
    data = modem.to_spec()
    path = data["path"]
    assert isinstance(path, list)
    step_dict = path[0]
    assert isinstance(step_dict, dict)
    assert step_dict["conv"] == "psk_demod"
    assert "order" in step_dict


def test_unknown_conv_fails_at_parse() -> None:
    with pytest.raises(StepSpecError) as exc:
        Modem.from_spec(
            {
                "symbol_rate": 1e5,
                "path": [{"conv": "psk_demod", "order": 4}, {"conv": "nope"}],
            },
            step_models(),
        )
    assert exc.value.index == 1 and exc.value.conv == "nope"


def test_bad_param_fails_at_parse() -> None:
    with pytest.raises(StepSpecError) as exc:
        Modem.from_spec(
            {"symbol_rate": 1e5, "path": [{"conv": "psk_demod", "order": 3}]},
            step_models(),
        )
    assert exc.value.index == 0 and exc.value.conv == "psk_demod"


def test_missing_symbol_rate_fails_cleanly() -> None:
    with pytest.raises(ValueError, match="symbol_rate"):
        Modem.from_spec({"path": []}, step_models())
