from pathlib import Path

import numpy as np
import pytest

from marconi.engine.backends.base import BackendError
from marconi.engine.backends.gnuradio.build import build_top_block
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline

SF = 7


def _css_demap_pipeline(tmp_path: Path) -> GrPipeline:
    """css_demap is the one embedded block that OUTPUTS uint8 — the shape the
    main-thread guard exists for. It is fed through a real dechirp chain from
    IQ, since no symbol-file source exists to drive it directly."""
    from marconi.engine.compile.compile_context import CompileContext
    from marconi.engine.modulation.css.stages import Dechirp, DechirpStep
    from marconi.engine.types.descriptor import Descriptor
    from marconi.engine.types.enums import ItemType
    from marconi.engine.types.levels import Level

    iq_in = tmp_path / "in.iq"
    bits_out = tmp_path / "out.bits"
    np.zeros(4 * (1 << SF), dtype=np.complex64).tofile(iq_in)
    rate = float(2 * (1 << SF))
    ctx = CompileContext(Descriptor(Level.IQ, ItemType.C), rate, 1.0)
    ctx.chain("iq_file_source", path=str(iq_in))
    Dechirp().emit_rx(ctx, DechirpStep(sf=SF, oversample=2, zero_pad=1))
    ctx.chain("css_demap", sf=SF)
    terminal = ctx.chain("bits_file_sink", path=str(bits_out))
    return ctx.build("css_demap_guard", rate, terminal_sink=terminal)


def _complex_only_pipeline(tmp_path: Path) -> tuple[GrPipeline, Path]:
    iq_in = tmp_path / "in.iq"
    iq_out = tmp_path / "out.iq"
    np.ones(16, dtype=np.complex64).tofile(iq_in)
    return _complex_only_graph(iq_in, iq_out), iq_out


def _complex_only_graph(iq_in: Path, iq_out: Path) -> GrPipeline:
    return GrPipeline(
        name="complex_only_guard",
        sample_rate=1.0,
        blocks=[
            GrBlock(id="src", kind="iq_file_source", params={"path": str(iq_in)}),
            GrBlock(id="snk", kind="iq_file_sink", params={"path": str(iq_out)}),
        ],
        connections=[GrConnection(src_block="src", dst_block="snk")],
    )


def test_complex_only_pipeline_runs_on_main_thread(tmp_path: Path) -> None:
    # the guard admits a GR-only graph to the main thread; "admitted" is only
    # meaningful if the graph actually moved its samples through
    pipe, sink = _complex_only_pipeline(tmp_path)
    tb = build_top_block(pipe)
    tb.run()
    assert np.array_equal(
        np.fromfile(sink, np.complex64), np.ones(16, dtype=np.complex64)
    )


def test_uint8_python_block_rejects_main_thread_run(tmp_path: Path) -> None:
    tb = build_top_block(_css_demap_pipeline(tmp_path))
    with pytest.raises(BackendError, match="run_pipeline"):
        tb.run()
