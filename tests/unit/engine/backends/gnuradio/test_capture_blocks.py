from __future__ import annotations

from pathlib import Path

import numpy as np

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.ir import GrBlock, GrConnection, GrPipeline


def test_skiphead_and_head_bound_a_stream(tmp_path: Path) -> None:
    ensure_worker_warm()
    src_path = tmp_path / "in.cf32"
    out_path = tmp_path / "out.cf32"
    samples = (np.arange(1000) + 1j * np.arange(1000)).astype(np.complex64)
    samples.tofile(src_path)
    pipeline = GrPipeline(
        name="skip-head-probe",
        sample_rate=1e6,
        blocks=[
            GrBlock(id="src", kind="iq_file_source", params={"path": str(src_path)}),
            GrBlock(id="skip", kind="iq_skiphead", params={"num_items": 100}),
            GrBlock(id="head", kind="iq_head", params={"num_items": 300}),
            GrBlock(id="sink", kind="iq_file_sink", params={"path": str(out_path)}),
        ],
        connections=[
            GrConnection(src_block="src", dst_block="skip"),
            GrConnection(src_block="skip", dst_block="head"),
            GrConnection(src_block="head", dst_block="sink"),
        ],
    )
    result = GnuRadioBackend().run_pipeline(pipeline, timeout=30.0)
    assert result.status == "ok", result
    out = np.fromfile(out_path, dtype=np.complex64)
    assert out.size == 300
    np.testing.assert_array_equal(out, samples[100:400])
