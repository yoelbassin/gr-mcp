from pathlib import Path

from marconi.workspace import Workspace


def test_subdirs_created_on_demand(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    cap = ws.new_capture_path("scan")
    assert cap == tmp_path / "artifacts" / "captures" / "scan"
    assert cap.parent.is_dir()
    png = ws.new_render_path("spec")
    assert png == tmp_path / "artifacts" / "renders" / "spec.png"
    assert png.parent.is_dir()


def test_paths_deduplicate(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    first = ws.new_capture_path("scan")
    (first.parent / "scan.sigmf-data").touch()
    second = ws.new_capture_path("scan")
    assert second == tmp_path / "artifacts" / "captures" / "scan-1"

    p1 = ws.new_render_path("spec")
    p1.touch()
    p2 = ws.new_render_path("spec")
    assert p2 == tmp_path / "artifacts" / "renders" / "spec-1.png"


def test_dedupe_glob_metacharacters(tmp_path: Path) -> None:
    """Names with glob metacharacters (e.g. brackets) must not be mis-expanded."""
    ws = Workspace(tmp_path)
    # First call — no collision yet.
    first = ws.new_capture_path("ISM[2.4GHz]")
    assert first == tmp_path / "artifacts" / "captures" / "ISM[2.4GHz]"
    # Simulate the sidecar file that new_capture_path would eventually create.
    (first.parent / "ISM[2.4GHz].sigmf-data").touch()
    # Second call — must detect the collision and bump to -1.
    second = ws.new_capture_path("ISM[2.4GHz]")
    assert second == tmp_path / "artifacts" / "captures" / "ISM[2.4GHz]-1"


def test_new_scene_path(tmp_path: Path) -> None:
    """new_scene_path creates scenes/ and returns a .yaml path."""
    ws = Workspace(tmp_path)
    p = ws.new_scene_path("downtown")
    assert p == tmp_path / "artifacts" / "scenes" / "downtown.yaml"
    assert p.parent.is_dir()


def test_new_pipeline_path(tmp_path: Path) -> None:
    """new_pipeline_path creates pipelines/ and returns a .yaml path."""
    ws = Workspace(tmp_path)
    p = ws.new_pipeline_path("demod")
    assert p == tmp_path / "artifacts" / "pipelines" / "demod.yaml"
    assert p.parent.is_dir()
