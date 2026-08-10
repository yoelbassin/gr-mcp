import numpy as np
import pytest
from helpers._ldpc import ldpc_codes, parse_check_nodes
from pydantic import ValidationError

from marconi.engine.backends.gnuradio.runner import GnuRadioBackend, ensure_worker_warm
from marconi.engine.compile.compiler import compile_modem
from marconi.engine.io.bitfile import read_bits
from marconi.engine.modulation.coding.stages import Ldpc, LdpcStep
from marconi.engine.stages.registry import stage_registry
from marconi.engine.types.descriptor import Carrier, Descriptor
from marconi.engine.types.enums import ItemType
from marconi.engine.types.levels import Level
from marconi.engine.types.models import Modem

_CODES = ldpc_codes()
_needs_codes = pytest.mark.skipif(
    not _CODES, reason="no gr-fec LDPC alist codes installed"
)


def _encode(path, gap, info):
    from gnuradio import blocks as gb
    from gnuradio import fec, gr

    enc = fec.ldpc_par_mtrx_encoder.make(str(path), gap)
    tb = gr.top_block()
    snk = gb.vector_sink_b()
    tb.connect(
        gb.vector_source_b(info.tolist(), False),
        fec.extended_encoder(enc, threading=None, puncpat="11"),
        snk,
    )
    tb.run()
    return np.asarray(snk.data(), np.uint8)


@_needs_codes
def test_ldpc_roundtrip(tmp_path):
    ensure_worker_warm()
    path, n, k, gap = _CODES[0]
    checks = parse_check_nodes(path, n)
    info = np.random.default_rng(1).integers(0, 2, k * 30).astype(np.uint8)
    coded = _encode(path, gap, info)
    # engine convention: bit 1 -> negative LLR (the ldpc stage flips it internally)
    soft = (1.0 - 2.0 * coded).astype(np.float32)
    src = tmp_path / "s.f32"
    soft.tofile(src)
    snk = tmp_path / "b.u8"
    modem = Modem(
        name="ldpc",
        symbol_rate=1.0,
        path=[LdpcStep(block_size=n, check_nodes=checks, max_iterations=50)],
    )
    pipe = compile_modem(
        modem,
        stage_registry(),
        direction="rx",
        sample_rate=1.0,
        start=Descriptor(Level.BITS, ItemType.F, Carrier.SOFT),
        source_io={"path": str(src)},
        sink_io={"path": str(snk)},
    )
    r = GnuRadioBackend().run_pipeline(pipe, timeout=90.0)
    assert r.status == "ok", r
    assert np.array_equal(read_bits(snk)[: info.size], info)


@_needs_codes
def test_ldpc_info_bits_matches_decoder():
    """The stage's declared K = N - (parity checks) matches the code's rate."""
    path, n, k, _ = _CODES[0]
    step = LdpcStep(block_size=n, check_nodes=parse_check_nodes(path, n))
    assert step.info_bits == k


_OK = {"block_size": 4, "check_nodes": [[0, 1, 2], [1, 2, 3]]}


def test_ldpc_params_accept_valid():
    step = Ldpc().step_model.model_validate(_OK)
    assert step.block_size == 4 and step.info_bits == 2 and step.max_iterations == 50


@pytest.mark.parametrize(
    "override",
    [
        {"check_nodes": []},  # no checks
        {"check_nodes": [[0, 1, 2, 3]] * 4},  # m >= n
        {"check_nodes": [[0, 1], []]},  # empty check
        {"check_nodes": [[0, 1, 1], [1, 2, 3]]},  # non-distinct within a check
        {"check_nodes": [[0, 1, 4], [1, 2, 3]]},  # index out of range
        {"max_iterations": 0},  # < 1
    ],
)
def test_ldpc_params_reject_malformed(override):
    with pytest.raises(ValidationError):
        Ldpc().step_model.model_validate({**_OK, **override})
