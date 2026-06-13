from marconi import sigmf
from marconi.mcp import tools as T


def _write_capture(server_state, make_iq):
    x = make_iq([(100e3, 1.0)], sample_rate=1e6, duration=0.05, noise_amplitude=0.01)
    ref = sigmf.write_capture(
        x,
        server_state.workspace.new_capture_path("t"),
        center_freq=100e6,
        sample_rate=1e6,
    )
    return str(ref.path)


def test_psd_returns_summary_not_arrays(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    out = T.psd(path)
    assert "noise_floor_db" in out and "peaks" in out
    assert "freqs" not in out and "psd_db" not in out  # arrays omitted
    assert out["num_bins"] > 0


def test_find_signals_returns_list(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    sigs = T.find_signals(path)
    assert isinstance(sigs, list) and sigs
    assert any(abs(s["center_freq"] - 100.1e6) < 5e3 for s in sigs)


def test_measure_returns_measurement(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    m = T.measure(path, center_freq=100.1e6)
    assert m["snr_db"] > 8
    assert m["occupied_bw_99"] >= 0


def test_detect_bursts_returns_list(server_state, make_iq):
    path = _write_capture(server_state, make_iq)
    bursts = T.detect_bursts(path)
    assert isinstance(bursts, list)  # a steady tone yields no bursts
