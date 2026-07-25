from pathlib import Path

from marconi.engine.backends.gnuradio.blocks import GR_BLOCKS, _factories


def test_io_kinds_present() -> None:
    for kind in (
        "iq_file_source",
        "iq_file_sink",
        "bits_file_source",
        "bits_file_sink",
        "soft_bits_file_sink",
        "symbols_file_sink",
    ):
        assert kind in GR_BLOCKS


def test_factories_construct_real_blocks(tmp_path: Path) -> None:
    (tmp_path / "x.iq").write_bytes(b"\x00" * 16)
    fac = _factories(1.0)
    src = fac["iq_file_source"]({"path": str(tmp_path / "x.iq")})
    snk = fac["soft_bits_file_sink"]({"path": str(tmp_path / "s.f32")})
    assert hasattr(src, "to_basic_block") and hasattr(snk, "to_basic_block")


def test_unknown_kind_absent() -> None:
    assert "no_such_block" not in _factories(1.0)


def test_fsk_dsp_blocks_construct() -> None:
    fac = _factories(4.0)
    assert hasattr(fac["quadrature_demod"]({"gain": 0.6}), "to_basic_block")
    assert hasattr(fac["symbol_sync_ff"]({"sps": 4.0}), "to_basic_block")
    assert hasattr(fac["binary_slicer"]({}), "to_basic_block")
    assert hasattr(fac["chunks_to_symbols"]({"symbols": [-1.0, 1.0]}), "to_basic_block")
    assert hasattr(fac["repeat_f"]({"interp": 4}), "to_basic_block")
    assert hasattr(fac["frequency_modulator"]({"sensitivity": 1.57}), "to_basic_block")
