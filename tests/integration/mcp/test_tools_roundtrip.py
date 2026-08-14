from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from helpers import _synth as synth

from marconi.mcp.tools import read_stream, run_rx_tool

_TXRX = {
    "symbol_rate": 1.0,
    "path": [{"conv": "fsk", "deviation": 1.0}, {"conv": "slice"}],
}


def _fsk_capture(tmp_path: Path, bits: np.ndarray, name: str = "fsk.cf32") -> str:
    """The capture these RX tests decode. Synthesized, not round-tripped
    through a modulator: what is under test is run_rx's reporting — the bit
    page, the soft-stream sign convention, the symbol-stream suffix — and a
    generator that shares the demod's assumptions cannot challenge any of it."""
    return str(
        synth.write(
            tmp_path / name,
            synth.cpfsk(bits, sps=4, deviation=1.0, sample_rate=4.0),
        )
    )


def test_tx_rx_roundtrip_ber0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    rng = np.random.default_rng(11)
    bit_array = rng.integers(0, 2, 2048).astype(np.uint8)
    bits = "".join("01"[b] for b in bit_array)
    capture = _fsk_capture(tmp_path, bit_array)
    rx = run_rx_tool(_TXRX, sample_rate=4.0, capture_path=capture)
    assert rx["status"] == "ok", rx
    stream = cast(dict[str, object], rx["stream"])
    assert stream["item_type"] == "b"
    page = read_stream(cast(str, stream["path"]), offset=0, count=4096)
    decoded = cast(str, page["bits"])
    # the reverse-substring arm (`decoded in bits`) accepted ANY decode that
    # happened to be a substring of the 2048 transmitted bits - a 10-bit one
    # included; a decode must carry nearly all of the payload before any
    # alignment-shape acceptance applies
    assert len(decoded) >= int(0.9 * len(bits)), (len(decoded), len(bits))
    assert bits in decoded or _aligned_equal(bits, decoded)
    # the demod tap feeding the slicer is surfaced for soft-decision work
    soft = cast(dict[str, object], rx["soft_stream"])
    assert soft["item_type"] == "f"
    assert soft["level"] == "symbols"
    assert soft["bit1_sign"] == "positive"
    assert cast(int, soft["items"]) > 0
    soft_path = Path(cast(str, soft["path"]))
    assert soft_path.is_file()
    values = np.fromfile(soft_path, np.float32)
    assert values.size == soft["items"]
    # symbols-level convention: positive slices to bit 1
    assert (
        np.mean(
            (values > 0)
            == (
                np.frombuffer(cast(str, page["bits"])[: values.size].encode(), np.uint8)
                == ord("1")
            )
        )
        > 0.9
    )


def _aligned_equal(tx_bits: str, rx_bits: str, max_shift: int = 64) -> bool:
    tx_arr = np.frombuffer(tx_bits.encode(), np.uint8)
    rx_arr = np.frombuffer(rx_bits.encode(), np.uint8)
    n = min(tx_arr.size, rx_arr.size) - max_shift
    if n <= 0:
        return False
    return any(
        np.array_equal(rx_arr[s : s + n], tx_arr[:n])
        or np.array_equal(rx_arr[:n], tx_arr[s : s + n])
        for s in range(max_shift)
    )


def test_soft_terminal_path_is_its_own_soft_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    rng = np.random.default_rng(7)
    capture = _fsk_capture(tmp_path, rng.integers(0, 2, 512).astype(np.uint8))
    rx = run_rx_tool(
        {
            "symbol_rate": 1.0,
            "path": [
                {"conv": "fsk", "deviation": 1.0},
                {"conv": "mfsk_soft_demap", "levels": [-1.0, 1.0]},
            ],
        },
        sample_rate=4.0,
        capture_path=capture,
    )
    assert rx["status"] == "ok", rx
    stream = cast(dict[str, object], rx["stream"])
    soft = cast(dict[str, object], rx["soft_stream"])
    assert soft["path"] == stream["path"]
    assert soft["items"] == stream["items"]
    assert soft["level"] == "bits"
    assert soft["bit1_sign"] == "negative"


def test_css_symbols_stream_pages_with_i16_suffix_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    sf, oversample, zero_pad, symbol_rate = 7, 2, 4, 1.0
    rate = oversample * (1 << sf) * symbol_rate
    css_sync = {
        "conv": "chirp_sync",
        "sf": sf,
        "oversample": oversample,
        "zero_pad": zero_pad,
        "preamble_len": 8,
        "sfd_symbols": 2.25,
        "sync_symbols": 2,
    }
    dechirp = {
        "conv": "dechirp",
        "sf": sf,
        "oversample": oversample,
        "zero_pad": zero_pad,
    }
    rx_spec = {"symbol_rate": symbol_rate, "path": [css_sync, dechirp]}
    symbols = np.random.default_rng(5).integers(0, 1 << sf, 20)
    capture = synth.write(
        tmp_path / "css.cf32",
        np.concatenate(
            [
                synth.chirp_preamble(
                    sf=sf, oversample=oversample, preamble_len=8, sfd_symbols=2.25
                ),
                synth.chirp_symbols(symbols, sf=sf, oversample=oversample),
            ]
        ),
    )
    rx = run_rx_tool(rx_spec, sample_rate=rate, capture_path=str(capture))
    assert rx["status"] == "ok", rx
    stream = cast(dict[str, object], rx["stream"])
    assert stream["item_type"] == "s"
    path = cast(str, stream["path"])
    assert Path(path).suffix == ".i16"
    page = read_stream(path, count=cast(int, stream["items"]))
    assert page["item_type"] == "s"
    assert page["total_items"] == stream["items"]


def test_run_rx_rejects_both_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARCONI_WORKSPACE", str(tmp_path))
    from fastmcp.exceptions import ToolError

    from marconi.mcp.tools import TOOLS

    with pytest.raises(ToolError, match=r"\[invalid_argument\]"):
        TOOLS["run_rx"](
            _TXRX, sample_rate=4.0, capture_path="a.cf32", input_path="b.u8"
        )
