"""_wire_eof_probe: sizing a consenting block's finality from the source
slice and the compiler rate. Stubbed instances (a source exposing item size,
a consenting block exposing eof_probe) keep most of this off GR - the
arithmetic is the unit under test, not the scheduler. The resampler cases are
the exception: their whole claim is that the wired count matches what the live
scheduler hands the block, so they run a real graph."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from marconi.engine.backends.gnuradio.build import _wire_eof_probe
from marconi.engine.backends.gnuradio.runner import GnuRadioBackend
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline
from marconi.engine.types.params import ParamValue

_SR = 2_048_000.0
_DECIM = 32


def _source(tmp_path: Path, n_samples: int) -> tuple[Path, SimpleNamespace]:
    p = tmp_path / "iq.cf32"
    p.write_bytes(b"\x00" * (n_samples * 8))  # complex64 = 8 bytes/item
    src = SimpleNamespace(
        output_signature=lambda: SimpleNamespace(sizeof_stream_item=lambda port: 8)
    )
    return p, src


def _channelized_pipeline(path: Path, **src_params: int) -> GrPipeline:
    # src -> channelize(/decim) -> complex_to_mag(1:1) -> burst_sampler
    params: dict[str, ParamValue] = {"path": str(path), **src_params}
    return GrPipeline(
        name="p",
        sample_rate=_SR,
        blocks=[
            GrBlock(
                id="src",
                kind="iq_file_source",
                params=params,
                sample_rate=_SR,
            ),
            GrBlock(id="ch", kind="freq_xlating_fir_filter", sample_rate=_SR),
            GrBlock(id="mag", kind="complex_to_mag", sample_rate=_SR / _DECIM),
            GrBlock(id="bs", kind="burst_sampler", sample_rate=_SR / _DECIM),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="ch"),
            GrConnection(src_block="ch", dst_block="mag"),
            GrConnection(src_block="mag", dst_block="bs"),
        ],
    )


def _wire(tmp_path: Path, pipe: GrPipeline, src: SimpleNamespace) -> Any:
    bs = SimpleNamespace(eof_probe=None)
    _wire_eof_probe(pipe, {"src": src, "bs": bs})
    return bs.eof_probe


def test_channelized_block_gets_emitted_over_decim(tmp_path: Path) -> None:
    path, src = _source(tmp_path, 3200)
    probe = _wire(tmp_path, _channelized_pipeline(path), src)
    assert probe.expected_items == 3200 // _DECIM  # emitted (whole file) / decim


def test_sliced_source_clips_the_emitted_count(tmp_path: Path) -> None:
    path, src = _source(tmp_path, 3200)
    pipe = _channelized_pipeline(path, offset=800, length=1600)
    probe = _wire(tmp_path, pipe, src)
    assert probe.expected_items == 1600 // _DECIM  # length wins over file size


def test_offset_only_slice_reads_to_eof(tmp_path: Path) -> None:
    path, src = _source(tmp_path, 3200)
    pipe = _channelized_pipeline(path, offset=1200)  # length 0 => to EOF
    probe = _wire(tmp_path, pipe, src)
    assert probe.expected_items == (3200 - 1200) // _DECIM


def test_undecimated_non_direct_block_is_not_sized(tmp_path: Path) -> None:
    # a hop one edge back from the source but at the source rate (ratio 1) can
    # be a hand-built graph mislabelling an interpolated wire: withhold, not size
    path, src = _source(tmp_path, 3200)
    pipe = _channelized_pipeline(path)
    pipe = pipe.model_copy(
        update={
            "blocks": [
                (
                    b.model_copy(update={"sample_rate": _SR})
                    if b.id in {"mag", "bs"}
                    else b
                )
                for b in pipe.blocks
            ]
        }
    )
    probe = _wire(tmp_path, pipe, src)
    assert probe.expected_items is None


def _marked(pipe: GrPipeline) -> GrPipeline:
    return pipe.model_copy(update={"terminal_sink": "any_sink"})


def test_compiler_marked_ratio_one_chain_is_sized(tmp_path: Path) -> None:
    # the shipped live shape: src -> complex_to_mag(1:1) -> burst_sampler, all
    # truthfully tagged at the source rate; a compiler-built pipeline (terminal
    # mark present) grants finality through ratio-1 hops so the final burst
    # can flush at EOF
    path, src = _source(tmp_path, 3200)
    pipe = GrPipeline(
        name="p",
        sample_rate=_SR,
        blocks=[
            GrBlock(
                id="src",
                kind="iq_file_source",
                params={"path": str(path)},
                sample_rate=_SR,
            ),
            GrBlock(id="mag", kind="complex_to_mag", sample_rate=_SR),
            GrBlock(id="bs", kind="burst_sampler", sample_rate=_SR),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="mag"),
            GrConnection(src_block="mag", dst_block="bs"),
        ],
    )
    probe = _wire(tmp_path, _marked(pipe), src)
    assert probe.expected_items == 3200


def test_marked_channelized_chain_keeps_emitted_over_decim(tmp_path: Path) -> None:
    path, src = _source(tmp_path, 3200)
    probe = _wire(tmp_path, _marked(_channelized_pipeline(path)), src)
    assert probe.expected_items == 3200 // _DECIM


def test_marked_interpolated_path_stays_unsized(tmp_path: Path) -> None:
    # resample(x2) then decimate(/4): the net last-edge ratio reads as an
    # integer decimation of 2, but floor composition makes the true delivered
    # count drift off emitted//2 — the path walk sees the rate INCREASE at the
    # resampler hop and refuses to size, withholding instead of flushing early
    path, src = _source(tmp_path, 3200)
    pipe = GrPipeline(
        name="p",
        sample_rate=_SR,
        blocks=[
            GrBlock(
                id="src",
                kind="iq_file_source",
                params={"path": str(path)},
                sample_rate=_SR,
            ),
            GrBlock(id="rs", kind="rational_resampler", sample_rate=_SR),
            GrBlock(id="ch", kind="freq_xlating_fir_filter", sample_rate=2 * _SR),
            GrBlock(id="bs", kind="burst_sampler", sample_rate=_SR / 2),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="rs"),
            GrConnection(src_block="rs", dst_block="ch"),
            GrConnection(src_block="ch", dst_block="bs"),
        ],
    )
    probe = _wire(tmp_path, _marked(pipe), src)
    assert probe.expected_items is None


_RESAMPLED_ITEMS = 10_000


def _run_off_thread(tb: Any, timeout: float = 30.0) -> None:
    done = threading.Event()
    err: list[BaseException] = []

    def _t() -> None:
        try:
            tb.run()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    threading.Thread(target=_t, daemon=True).start()
    if not done.wait(timeout):
        tb.stop()
        tb.wait()
        raise TimeoutError("flowgraph timed out")
    if err:
        raise err[0]


def _resampled_pipeline(cap: Path, out: Path, decim: int) -> GrPipeline:
    rate = _SR / decim
    return GrPipeline(
        name="resampled",
        sample_rate=_SR,
        terminal_sink="snk",
        blocks=[
            GrBlock(
                id="src",
                kind="iq_file_source",
                params={"path": str(cap)},
                sample_rate=_SR,
            ),
            GrBlock(
                id="rs",
                kind="rational_resampler_ccf",
                params={"interpolation": 1, "decimation": decim},
                sample_rate=_SR,
            ),
            GrBlock(id="mag", kind="complex_to_mag", sample_rate=rate),
            GrBlock(
                id="bs", kind="burst_sampler", params={"sps": 2.0}, sample_rate=rate
            ),
            GrBlock(
                id="snk",
                kind="soft_bits_file_sink",
                params={"path": str(out)},
                sample_rate=rate,
            ),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="rs"),
            GrConnection(src_block="rs", dst_block="mag"),
            GrConnection(src_block="mag", dst_block="bs"),
            GrConnection(src_block="bs", dst_block="snk"),
        ],
    )


@pytest.mark.parametrize("decim", [4, 10])
def test_resampled_path_is_sized_to_what_the_resampler_delivers(
    tmp_path: Path, decim: int
) -> None:
    # rational_resampler_ccf keeps a polyphase tail back, so it hands the
    # downstream chain FEWER than emitted//decim items; sizing the probe by the
    # rate ratio alone leaves expected_items forever out of reach and eof_final
    # never fires. Truth here is the live scheduler's own count.
    rng = np.random.default_rng(0)
    iq = (
        rng.standard_normal(_RESAMPLED_ITEMS)
        + 1j * rng.standard_normal(_RESAMPLED_ITEMS)
    ).astype(np.complex64)
    cap = tmp_path / "in.cf32"
    iq.tofile(cap)
    pipe = _resampled_pipeline(cap, tmp_path / "out.f32", decim)
    tb = GnuRadioBackend().instantiate(pipe)
    bs = tb._py_instances["bs"]
    _run_off_thread(tb)
    read = bs.nitems_read(0)
    assert read > 0  # a graph that never ran would make the equality vacuous
    assert bs.eof_probe.expected_items == read


def test_source_adjacent_block_keeps_direct_behavior(tmp_path: Path) -> None:
    path, src = _source(tmp_path, 3200)
    pipe = GrPipeline(
        name="p",
        sample_rate=_SR,
        blocks=[
            GrBlock(
                id="src",
                kind="iq_file_source",
                params={"path": str(path)},
                sample_rate=_SR,
            ),
            GrBlock(id="bs", kind="chirp_sync", sample_rate=_SR),
        ],
        connections=[GrConnection(src_block="src", dst_block="bs")],
    )
    probe = _wire(tmp_path, pipe, src)
    assert probe.expected_items == 3200  # whole emitted count, decim 1
