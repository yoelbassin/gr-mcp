from pathlib import Path


class Workspace:
    """The user's RF project directory.

    Layout: captures/ renders/ pipelines/ scenes/ — created on demand.
    Artifacts are exchanged as paths into this directory, never as blobs.
    """

    def __init__(self, root: Path | str = ".") -> None:
        self.root = Path(root)

    def _subdir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _dedupe(directory: Path, stem: str, probe_suffix: str, suffix: str) -> Path:
        candidate = stem
        i = 0
        while (
            list(directory.glob(candidate + probe_suffix + "*"))
            or (directory / (candidate + suffix)).exists()
        ):
            i += 1
            candidate = f"{stem}-{i}"
        return directory / (candidate + suffix)

    def new_capture_path(self, name: str) -> Path:
        """Extension-less base path for a SigMF pair under captures/."""
        return self._dedupe(self._subdir("captures"), name, ".sigmf", "")

    def new_render_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("renders"), name, ".png", ".png")

    def new_scene_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("scenes"), name, ".yaml", ".yaml")

    def new_pipeline_path(self, name: str) -> Path:
        return self._dedupe(self._subdir("pipelines"), name, ".yaml", ".yaml")
