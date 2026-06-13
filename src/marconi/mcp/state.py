"""Per-process server state.

The workspace is the source of truth: on startup the simulated-device registry
is rebuilt from the scene YAML files under workspace/scenes/, so devices created
in a previous session reappear. Runs are synchronous in v1.0; each is recorded
in an in-process history for observability and artifact recall."""

from __future__ import annotations

from marconi.devices import add_simulated_device, clear_devices
from marconi.models import RunResult
from marconi.specs import load_scene
from marconi.workspace import Workspace


class ServerState:
    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.runs: list[dict] = []
        self._run_counter = 0
        self._load_persisted_devices()

    def _load_persisted_devices(self) -> None:
        # The workspace is authoritative: clear any prior registry, then rebuild
        # from scene files (device id == scene file stem).
        clear_devices()
        scenes_dir = self.workspace.root / "scenes"
        if not scenes_dir.is_dir():
            return
        for path in sorted(scenes_dir.glob("*.yaml")):
            add_simulated_device(path.stem, load_scene(path))

    def next_run_id(self) -> str:
        self._run_counter += 1
        return f"run-{self._run_counter}"

    def record_run(self, run_id: str, pipeline_name: str, result: RunResult) -> dict:
        rec = {
            "run_id": run_id,
            "pipeline": pipeline_name,
            "status": result.status,
            "elapsed_seconds": result.elapsed_seconds,
            "artifacts": [str(p) for p in result.artifacts],
            "error": result.error,
        }
        self.runs.append(rec)
        return rec


_STATE: ServerState | None = None


def set_state(state: ServerState) -> None:
    global _STATE
    _STATE = state


def get_state() -> ServerState:
    if _STATE is None:
        raise RuntimeError("server state not initialized; call set_state() first")
    return _STATE


def reset_state() -> None:
    """Test helper: drop the state and empty the device registry."""
    global _STATE
    _STATE = None
    clear_devices()
