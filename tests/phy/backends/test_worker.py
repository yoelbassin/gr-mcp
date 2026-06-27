import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from marconi.phy.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.phy.ir import GrBlock, GrConnection, GrPipeline


def _passthrough(src: Path, dst: Path) -> GrPipeline:
    return GrPipeline(
        name="pt",
        sample_rate=1.0,
        blocks=[
            GrBlock(id="s", kind="iq_file_source", params={"path": str(src)}),
            GrBlock(id="k", kind="iq_file_sink", params={"path": str(dst)}),
        ],
        connections=[GrConnection(src_block="s", dst_block="k")],
    )


def test_run_pipeline_passthrough_byte_exact(tmp_path: Path) -> None:
    ensure_worker_warm()
    rng = np.random.default_rng(1)
    data = (rng.standard_normal(800) + 1j * rng.standard_normal(800)).astype(
        np.complex64
    )
    src = tmp_path / "in.iq"
    dst = tmp_path / "out.iq"
    data.tofile(src)
    res = GnuRadioBackend().run_pipeline(_passthrough(src, dst))
    assert res.status == "ok"
    assert np.array_equal(np.fromfile(dst, dtype=np.complex64), data)


def test_bad_kind_returns_error_without_killing_parent() -> None:
    pipe = GrPipeline(
        name="bad",
        sample_rate=1.0,
        blocks=[GrBlock(id="z", kind="nonesuch", params={})],
        connections=[],
    )
    res = GnuRadioBackend().run_pipeline(pipe)
    assert res.status == "error" and res.error  # surfaced, parent alive


def test_parent_process_stays_gnuradio_free() -> None:
    code = textwrap.dedent(
        """
        import sys, importlib.abc

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "gnuradio" or name.startswith("gnuradio."):
                    raise ImportError("gnuradio banned in parent")
                return None

        sys.meta_path.insert(0, _Block())
        from marconi.phy.backends.gnuradio.runner import (
            GnuRadioBackend, ensure_worker_warm,
        )
        GnuRadioBackend()
        ensure_worker_warm()
        assert "gnuradio" not in sys.modules, "parent imported gnuradio"
        print("OK")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "OK" in out.stdout, out.stderr
