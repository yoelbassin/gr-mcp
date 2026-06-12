from glob import escape as glob_escape
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
        """Return a collision-free path under *directory*.

        Two checks per candidate:
        - *probe* — ``glob(escaped_candidate + probe_suffix + "*")`` catches any
          sidecar files that share the same base name (e.g. ``.sigmf-meta`` /
          ``.sigmf-data`` for SigMF pairs, or a bare ``.yaml`` alongside others).
        - *exact* — ``(directory / (candidate + suffix)).exists()`` catches the
          primary output file itself when *probe_suffix == suffix*.

        ``glob_escape`` is applied to the candidate before globbing so that names
        containing glob metacharacters (e.g. ``ISM[2.4GHz]``) are treated as
        literals rather than patterns.
        """
        candidate = stem
        i = 0
        while (
            next(directory.glob(glob_escape(candidate) + probe_suffix + "*"), None)
            is not None
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
