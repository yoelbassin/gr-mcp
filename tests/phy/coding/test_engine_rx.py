from pathlib import Path

import numpy as np

from marconi.core.bitfile import read_bits, write_bits
from marconi.core.descriptor import Carrier, Descriptor
from marconi.core.levels import Level
from marconi.core.models import Bitstream
from marconi.phy.backends.gnuradio.runner import ensure_worker_warm
from marconi.phy.engine import run_rx
from marconi.phy.models import ModemSpec, ModemStep
from marconi.phy.stages import stage_registry

BITS = Descriptor(Level.BITS, "b")
SOFT_SYMBOLS = Descriptor(Level.SYMBOLS, "f", carrier=Carrier.SOFT)
_FLAG_BITS = np.concatenate(
    [
        np.zeros(3, np.uint8),
        np.array([0, 1, 1, 1, 1, 1, 1, 0], np.uint8),
        np.ones(24, np.uint8),
    ]
)


def _flag_stream(tmp_path: Path) -> Bitstream:
    p = tmp_path / "in.u8"
    write_bits(p, _FLAG_BITS)
    return Bitstream(path=p, num_bits=int(_FLAG_BITS.size))


def _flag_soft_symbols(tmp_path: Path) -> Path:
    soft = np.where(_FLAG_BITS == 1, 1.0, -1.0).astype(np.float32)
    p = tmp_path / "flag.f32"
    soft.tofile(p)
    return p


def test_pure_coding_run_seeds_windows_and_reports_census(tmp_path: Path) -> None:
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[ModemStep(conv="sync_word", params={"sync": "7e"})],
    )
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=BITS,
        workdir=tmp_path,
        input_stream=_flag_stream(tmp_path),
    )
    assert res.status == "ok"
    assert res.windows == [11]
    assert res.bitstream is not None
    assert read_bits(res.bitstream.path).size == 35
    assert [r.kind for r in res.census] == ["sync_word"]
    assert res.census[0].windows_out == 1


def test_gr_plus_coding_run_shares_windows_and_census(tmp_path: Path) -> None:
    ensure_worker_warm()
    modem = ModemSpec(
        symbol_rate=1.0,
        path=[
            ModemStep(conv="slice"),
            ModemStep(conv="sync_word", params={"sync": "7e"}),
        ],
    )
    res = run_rx(
        modem,
        stage_registry(),
        sample_rate=1.0,
        start=SOFT_SYMBOLS,
        workdir=tmp_path,
        source_io={"path": str(_flag_soft_symbols(tmp_path))},
    )
    assert res.status == "ok", res.error
    assert res.windows == [11]
    assert len(res.census) >= 3
    assert res.census[-1].kind == "sync_word"
