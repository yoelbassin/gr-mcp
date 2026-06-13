from pathlib import Path

from marconi.mcp import tools as T


def test_simulate_scene_registers_and_persists(server_state):
    info = T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.01}])
    assert info["id"] == "sim0"
    assert info["kind"] == "simulated"
    assert (server_state.workspace.root / "artifacts" / "scenes" / "sim0.yaml").exists()
    assert any(d["id"] == "sim0" for d in T.list_devices())


def test_devices_persist_across_restart(server_state):
    from marconi.mcp.state import ServerState, set_state
    from marconi.workspace import Workspace

    T.simulate_scene("sim0", [{"kind": "noise", "amplitude": 0.01}])
    # Simulate a new server process pointed at the same workspace.
    set_state(ServerState(Workspace(server_state.workspace.root)))
    assert any(d["id"] == "sim0" for d in T.list_devices())


def test_capture_returns_reference(server_state):
    T.simulate_scene(
        "sim0",
        [
            {"kind": "tone", "freq": 100.05e6, "amplitude": 0.5},
            {"kind": "noise", "amplitude": 0.005},
        ],
    )
    ref = T.capture("sim0", center_freq=100e6, sample_rate=1e6, duration=0.05)
    assert ref["sample_rate"] == 1e6
    assert ref["num_samples"] > 0
    assert Path(ref["path"]).exists()


def test_load_capture_cf32(server_state, tmp_path):
    import numpy as np

    raw = tmp_path / "x.cf32"
    np.arange(8, dtype=np.complex64).tofile(raw)
    ref = T.load_capture(str(raw), sample_rate=1e6)
    assert ref["num_samples"] == 8
    assert ref["sample_rate"] == 1e6
