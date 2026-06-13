from pathlib import Path

import pytest

import marconi
from marconi.mcp.state import ServerState, get_state, reset_state, set_state
from marconi.models import RunResult, SceneElement, SceneSpec
from marconi.specs import save_scene
from marconi.workspace import Workspace


@pytest.fixture(autouse=True)
def _clean():
    reset_state()
    marconi.clear_devices()
    yield
    reset_state()
    marconi.clear_devices()


def test_get_state_before_init_raises():
    with pytest.raises(RuntimeError):
        get_state()


def test_set_and_get_state(tmp_path: Path):
    state = ServerState(Workspace(tmp_path))
    set_state(state)
    assert get_state() is state


def test_init_registers_persisted_scenes(tmp_path: Path):
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    save_scene(
        SceneSpec(name="s", elements=[SceneElement(kind="noise", amplitude=0.01)]),
        scenes / "sim0.yaml",
    )
    ServerState(Workspace(tmp_path))  # scans on init
    assert [d.id for d in marconi.list_devices()] == ["sim0"]


def test_init_is_idempotent_and_authoritative(tmp_path: Path):
    # stale device is cleared; workspace is the source of truth
    marconi.add_simulated_device("ghost", SceneSpec(name="g"))
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    save_scene(SceneSpec(name="s"), scenes / "real.yaml")
    ServerState(Workspace(tmp_path))
    ids = sorted(d.id for d in marconi.list_devices())
    assert ids == ["real"]


def test_init_skips_unreadable_scene_file(tmp_path: Path):
    # one corrupt/partial scene must not abort startup before any tool is
    # reachable: it is skipped and the loadable devices still register.
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "broken.yaml").write_text("name: bad\nelements:\n- kind: qpsk\n")
    save_scene(SceneSpec(name="s"), scenes / "good.yaml")
    ServerState(Workspace(tmp_path))  # must not raise
    assert [d.id for d in marconi.list_devices()] == ["good"]


def test_run_history(tmp_path: Path):
    state = ServerState(Workspace(tmp_path))
    rec = state.record_run(
        "fm_rx",
        RunResult(
            status="ok", elapsed_seconds=1.2, artifacts=[Path("a.wav")], error=None
        ),
    )
    assert rec.run_id == "run-1"
    assert rec.status == "ok"
    assert rec.artifacts == ["a.wav"]
    assert state.runs[-1].run_id == "run-1"
