import threading
from pathlib import Path

import numpy as np
import pytest

from marconi.phy.backends.base import BackendError
from marconi.phy.backends.gnuradio.runner import GnuRadioBackend
from marconi.phy.ir import GrBlock, GrConnection, GrPipeline


def _run_off_thread(tb: object, timeout: float = 15.0) -> None:
    done = threading.Event()
    err: list[BaseException] = []

    def _t() -> None:
        try:
            tb.run()  # type: ignore[attr-defined]
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    threading.Thread(target=_t, daemon=True).start()
    if not done.wait(timeout):
        tb.stop()  # type: ignore[attr-defined]
        tb.wait()  # type: ignore[attr-defined]
        raise TimeoutError("flowgraph timed out")
    if err:
        raise err[0]


def test_iq_passthrough_is_byte_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(
        np.complex64
    )
    src = tmp_path / "in.iq"
    dst = tmp_path / "out.iq"
    data.tofile(src)
    pipe = GrPipeline(
        name="passthrough",
        sample_rate=1.0,
        blocks=[
            GrBlock(id="s", kind="iq_file_source", params={"path": str(src)}),
            GrBlock(id="k", kind="iq_file_sink", params={"path": str(dst)}),
        ],
        connections=[GrConnection(src_block="s", dst_block="k")],
    )
    tb = GnuRadioBackend().instantiate(pipe)
    _run_off_thread(tb)
    assert np.array_equal(np.fromfile(dst, dtype=np.complex64), data)


def test_unknown_kind_raises_backend_error() -> None:
    pipe = GrPipeline(
        name="x",
        sample_rate=1.0,
        blocks=[GrBlock(id="z", kind="nonesuch", params={})],
        connections=[],
    )
    with pytest.raises(BackendError):
        GnuRadioBackend().instantiate(pipe)


def test_dangling_connection_raises_backend_error(tmp_path: Path) -> None:
    (tmp_path / "i.iq").write_bytes(b"\x00" * 8)
    pipe = GrPipeline(
        name="x",
        sample_rate=1.0,
        blocks=[
            GrBlock(
                id="s",
                kind="iq_file_source",
                params={"path": str(tmp_path / "i.iq")},
            )
        ],
        connections=[GrConnection(src_block="s", dst_block="ghost")],
    )
    with pytest.raises(BackendError):
        GnuRadioBackend().instantiate(pipe)
