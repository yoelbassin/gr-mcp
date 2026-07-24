"""mfsk_soft_demap: the M-ary FSK entry to the soft coding lane.

Before this stage the only SYMBOLS->BITS converter accepting "f" was the binary
`slice`, so a multi-level FSK signal could not produce soft bits at all and the
coding lane was unreachable for the whole M-ary FSK family.

`levels` here is P25 C4FM's dibit->deviation map (01:+1800, 00:+600, 10:-600,
11:-1800 Hz) normalized to the max deviation -- a parameter, not vocabulary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from marconi.core.bitfile import harden, read_bits, read_llrs
from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.params import ParamValue
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.compiler import compile_modem
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

SYM_F = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
IQ = Descriptor(Level.IQ, "c")
_POLYS = [0o171, 0o133]
_D, _TAIL, _FRAME_BITS = 2, 6, 512
_SR, _SYM_RATE, _DEV = 48000.0, 4800.0, 1800.0
_C4FM = [600.0 / 1800.0, 1.0, -600.0 / 1800.0, -1.0]
# symbol_sync_ff withholds 3 symbols at EOF (measured); a terminated block code
# needs its exact frame length, so the stream must outlast the frame.
_TAIL_PAD = 8


def _run(
    path: list[ModemStep],
    src: Path,
    snk: Path,
    start: Descriptor,
    direction: str = "rx",
) -> Path:
    pipe = compile_modem(
        ModemSpec(symbol_rate=_SYM_RATE, path=path),
        stage_registry(),
        direction=direction,
        sample_rate=_SR,
        start=start,
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=300.0)
    assert r.status == "ok", r
    return snk


def _demap(levels: list[float]) -> ModemStep:
    params: dict[str, ParamValue] = {"levels": list(levels)}
    return ModemStep(conv="mfsk_soft_demap", params=params)


def _fec() -> ModemStep:
    params: dict[str, ParamValue] = {
        "scheme": "cc",
        "rate_inv": _D,
        "polys": list(_POLYS),
        "frame_bits": _FRAME_BITS,
        "tail": _TAIL,
    }
    return ModemStep(conv="fec", params=params)


def _ccsds_encode(info: np.ndarray) -> np.ndarray:
    from gnuradio import blocks, gr, trellis

    padded = np.concatenate([info, np.zeros(_TAIL, np.uint8)]).astype(np.uint8)
    tb = gr.top_block()
    snk = blocks.vector_sink_b()
    tb.connect(
        blocks.vector_source_b(padded.tolist(), False),
        trellis.encoder_bb(trellis.fsm(1, _D, _POLYS), 0),
        snk,
    )
    tb.run()
    o = np.asarray(snk.data(), np.uint8)
    return np.stack([(o >> (_D - 1 - j)) & 1 for j in range(_D)], axis=1).reshape(-1)


def _levels_for(bits: np.ndarray, levels: list[float], k: int) -> np.ndarray:
    g = bits.reshape(-1, k)
    idx = np.zeros(g.shape[0], int)
    for j in range(k):
        idx = idx * 2 + g[:, j]
    return np.asarray(levels, np.float32)[idx]


@pytest.mark.parametrize(
    "levels", [[-1.0, 1.0], _C4FM, [-1.0, -1 / 3, 1 / 3, 1.0]], ids=["m2", "c4fm", "m4"]
)
def test_soft_demap_hardens_to_the_transmitted_symbols(
    levels: list[float], tmp_path: Path
) -> None:
    ensure_worker_warm()
    k = len(levels).bit_length() - 1
    bits = np.random.default_rng(len(levels)).integers(0, 2, 4000 * k).astype(np.uint8)
    src = tmp_path / "lv.f32"
    _levels_for(bits, levels, k).tofile(src)
    llr = read_llrs(_run([_demap(levels)], src, tmp_path / "o.f32", SYM_F))
    assert llr.size == bits.size
    assert np.array_equal(harden(llr), bits)


def test_levels_must_be_a_distinct_power_of_two_set() -> None:
    reg = stage_registry()
    model = reg["mfsk_soft_demap"].params_model
    for bad in ([1.0, 2.0, 3.0], [1.0, 1.0], [0.0, 0.0], [1.0]):
        with pytest.raises(ValidationError):
            model.model_validate({"levels": bad})
    model.model_validate({"levels": _C4FM})


def test_soft_lane_corrects_errors_the_hard_decision_makes(tmp_path: Path) -> None:
    """Coding gain on 4-level symbols, with the hard-decision arm as control.

    At sigma 0.25 the nearest-level hard decision errs on ~7% of coded bits;
    the soft lane recovers the frame exactly.
    """
    ensure_worker_warm()
    rng = np.random.default_rng(3)
    info = rng.integers(0, 2, _FRAME_BITS).astype(np.uint8)
    coded = _ccsds_encode(info)
    clean = _levels_for(coded, _C4FM, 2)
    noisy = (clean + rng.normal(0, 0.25, clean.size)).astype(np.float32)

    lv = np.asarray(_C4FM)
    hard = np.argmin(np.abs(noisy[:, None] - lv[None, :]), axis=1)
    uncoded_ber = float(
        np.mean(np.stack([hard >> 1, hard & 1], axis=1).reshape(-1) != coded)
    )
    assert uncoded_ber > 0.02, f"control arm too clean to prove anything: {uncoded_ber}"

    src = tmp_path / "n.f32"
    noisy.tofile(src)
    out = read_bits(_run([_demap(_C4FM), _fec()], src, tmp_path / "o.u8", SYM_F))
    assert out.size == _FRAME_BITS
    assert np.array_equal(out, info), f"BER {float(np.mean(out != info))}"


def test_c4fm_chain_decodes_through_the_real_discriminator(tmp_path: Path) -> None:
    """Full IQ: fsk TX -> fsk discriminator + Gardner timing -> soft demap -> fec."""
    ensure_worker_warm()
    info = np.random.default_rng(3).integers(0, 2, _FRAME_BITS).astype(np.uint8)
    coded = _ccsds_encode(info)
    levels = np.concatenate(
        [_levels_for(coded, _C4FM, 2), np.full(_TAIL_PAD, _C4FM[0], np.float32)]
    )
    src = tmp_path / "lv.f32"
    levels.tofile(src)
    fsk = ModemStep(conv="fsk", params={"deviation": _DEV})
    iq = _run([fsk], src, tmp_path / "tx.cf32", IQ, direction="tx")
    out = read_bits(_run([fsk, _demap(_C4FM), _fec()], iq, tmp_path / "rx.u8", IQ))
    assert out.size == _FRAME_BITS
    assert np.array_equal(out, info), f"BER {float(np.mean(out != info))}"
