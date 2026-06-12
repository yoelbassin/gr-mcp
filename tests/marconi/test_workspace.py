from pathlib import Path

from marconi.workspace import Workspace


def test_subdirs_created_on_demand(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    cap = ws.new_capture_path("scan")
    assert cap == tmp_path / "captures" / "scan"
    assert cap.parent.is_dir()
    png = ws.new_render_path("spec")
    assert png == tmp_path / "renders" / "spec.png"
    assert png.parent.is_dir()


def test_paths_deduplicate(tmp_path: Path) -> None:
    ws = Workspace(tmp_path)
    first = ws.new_capture_path("scan")
    (first.parent / "scan.sigmf-data").touch()
    second = ws.new_capture_path("scan")
    assert second == tmp_path / "captures" / "scan-1"

    p1 = ws.new_render_path("spec")
    p1.touch()
    p2 = ws.new_render_path("spec")
    assert p2 == tmp_path / "renders" / "spec-1.png"
